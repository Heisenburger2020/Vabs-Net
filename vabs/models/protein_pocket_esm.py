# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
from unicore import utils
from unicore.models import (
    BaseUnicoreModel,
    register_model,
    register_model_architecture,
)
from typing import Any, Dict, Optional, Union, Tuple, List
import math
from unicore.modules import softmax_dropout, LayerNorm, SelfMultiheadAttention

import torch_scatter
from .layers import (
    NonLinear,
    SE3InvariantKernel,
    Embedding,
    ProteinEdgeFeature,
    CLSHead,
    Linear,
    MovementPredictionHead,
    DropPath,
    Attention,
    LayerNorm,
    Dropout,
    Transition,
    OuterProduct,
    TriangleMultiplication, 
    MLPs,
    VecEdgeFeature,
    NodeEmbedHead,
    SinusoidalPositionalEmbedding,
    SidechainAngleResnet,
    SidechainAngleResnetv2,
)
from .vabs_encoder import VabsEncoder
from unicore.modules import init_bert_params
from unicore.utils import checkpoint_sequential
from functools import partial
from .esm_utils import load_model_and_alphabet_local

logger = logging.getLogger(__name__)

def print_all_shape(*tensors):
    print("shape>>", [_.shape for _ in tensors])

@register_model("ESM_pocket")
class ESMProteinModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument(
            "--num-3d-bias-kernel",
            type=int,
            default=128,
            metavar="D",
            help="number of kernel in 3D attention bias",
        )
        parser.add_argument(
            "--debug",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--droppath-prob",
            type=float,
            metavar="D",
            help="stochastic path probability",
            default=0.0,
        )

        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" " attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability after" " activation in FFN",
        )

        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--pair-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--pair-hidden-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--pair-dropout",
            type=float,
            metavar="D",
            help="dropout probability for pair",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )

        # Arguments related to input and output embeddings
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )

        # misc params
        # parser.add_argument(
        #     "--activation-fn",
        #     choices=utils.get_available_activation_fns(),
        #     help="activation function to use",
        # )
        parser.add_argument(
            "--num-block",
            type=int,
            metavar="N",
            help="number of recycle",
        )
        parser.add_argument(
            "--noise-scale",
            default=0.3,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.35,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--label-prob",
            default=0.4,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--mid-prob",
            default=0.2,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--mid-upper",
            default=0.6,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--mid-lower",
            default=0.4,
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--pos-step-size",
            type=float,
            help="step size for pos update",
        )
        parser.add_argument(
            "--gaussian-std-width",
            type=float,
        )
        parser.add_argument(
            "--gaussian-mean-start",
            type=float,
        )
        parser.add_argument(
            "--gaussian-mean-stop",
            type=float,
        )
        parser.add_argument(
            "--notri", type=int, default=1
        )
        parser.add_argument("--preln", type=int, default=0)
        parser.add_argument("--lti", type=int, default=0)
        parser.add_argument(
            "--data-seed",
            type=int,
        )
        parser.add_argument(
            "--gamma-pos",
            type=float,
        )
        parser.add_argument(
            "--gamma-neg",
            type=int,
        )
        parser.add_argument(
            "--sample-atom",
            type=int,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--res-type-mask-prob",
            default=0.35,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--res-pos-mask-prob",
            default=0.35,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--res-noise-scale",
            default=0.3,
            type=float,
            help="coordinate noise for masked atoms",
        )

        parser.add_argument(
            "--num-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--node-dim",
            type=int,
            metavar="H",
            help="node embedding dimension",
        )
        parser.add_argument(
            "--edge-dim",
            type=int,
            metavar="H",
            help="edge embedding dimension",
        )
        parser.add_argument(
            "--kernel-size",
            type=int,
            help="kernel size for distance map",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the pooler layers",
        )
        parser.add_argument(
            "--init-method", type=str, default="bert", choices=["bert", "paper", "default"],
        )
        parser.add_argument(
            "--pred-init-method", type=str, choices=["bert", "paper", "default"], # None means the same as --init-method
        )
        parser.add_argument(
            "--activation-fn", type=str,
        )
        parser.add_argument(
            "--node-mlp-layers", type=int,
        )
        parser.add_argument(
            "--edge-mlp-layers", type=int,
        )
        parser.add_argument(
            "--outer-product-dim", type=int,
        )
        parser.add_argument(
            "--inter-edge-mlp-layers", type=int,
        )
        parser.add_argument(
            "--norm-layer", type=str, choices=["none", "layernorm"],
        )
        parser.add_argument(
            "--scatter-op", type=str, choices=["sum", "mean"],
        )
        parser.add_argument(
            "--crop",
            type=int,
        )
        parser.add_argument(
            "--keep",
            type=int,
        )
        parser.add_argument(
            "--pair-update",
            type=int,
            default=1,
        )

        parser.add_argument(
            "--use-outer",
            type=int,
        )
        parser.add_argument(
            "--use-trimul",
            type=int,
        )
        parser.add_argument(
            "--use-pairupdate",
            type=int,
        )
        parser.add_argument(
            "--use-triangleattn",
            type=int,
        )
        parser.add_argument(
            "--use-ipa",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-ipa-norm",
            type=int,
            default=1
        )

        parser.add_argument(
            "--virtual-num",
            type=int,
            default=4
        )
        parser.add_argument(
            "--cutoff",
            type=int,
            default=600
        )
        parser.add_argument(
            "--use-vec",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-attn",
            type=int,
            default=1
        )
        parser.add_argument(
            "--use-esm-feat",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-virtual-node",
            type=int,
            default=0
        )
        parser.add_argument(
            "--knn-atom2atom",
            type=int,
        )
        parser.add_argument(
            "--knn-res",
            type=int,
        )
        parser.add_argument(
            "--finetune",
            type=int,
            default=0
        )
        parser.add_argument(
            "--pos-pred",
            type=int,
            default=0
        )
        parser.add_argument(
            "--res-type-pred",
            type=int,
            default=0
        )
        parser.add_argument(
            "--pocket-pred",
            type=int,
            default=0
        )
        parser.add_argument(
            "--dihedral-pred",
            type=int,
            default=0
        )
        parser.add_argument(
            "--pocket-type",
            type=str,
            default=""
        )
        parser.add_argument(
            "--gb-feat",
            type=int,
            default=0
        )


    def __init__(self, args):
        super().__init__()
        self.args = args   
        self._num_updates = 0
        self.dtype = torch.float32

        esm_ckpt_path = r"/mnt/vepfs/fs_projects/uni-prot/protein_pocket/codes/esm/ckpt/esm2_t30_150M_UR50D.pt"
        self.esm_model, self.esm_alphabet = load_model_and_alphabet_local(esm_ckpt_path)


        no_loss = [
        "contact_head.regression.weight",
        "contact_head.regression.bias",
        "lm_head.bias",
        "lm_head.dense.weight",
        "lm_head.dense.bias",
        "lm_head.layer_norm.weight",
        "lm_head.layer_norm.bias",
        ]
        for param in self.esm_model.named_parameters():
            if param[0] in no_loss:
                param[1].requires_grad = False
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.residue_types = 21 + 4
        self.atom_types = 37 + 3

        self.pocket_cls_all = CLSHead(input_dim=640, output_cls=2, preln=args.preln)

    def half(self):
        super().half()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        super().bfloat16()
        self.dtype = torch.bfloat16
        return self

    def float(self):
        super().float()
        self.dtype = torch.float32
        return self

    def forward(
            self, 
            list_str: List[str],
            batch_index: torch.Tensor,
            **kwargs
        ):

        input_batch = [(str(_), list_str[_]) for _ in range(len(list_str))]
        _, _, batch_tokens = self.batch_converter(input_batch)
        batch_tokens = batch_tokens.to(batch_index.device)
        esm_embed = self.esm_model(batch_tokens, repr_layers=[30], return_contacts=True)
        esm_embed_list = [esm_embed["representations"][30][id, 1 : len(item) + 1] for id, item in enumerate(list_str)]
        all_rep = torch.cat(esm_embed_list)

        if self.args.pocket_pred:
            pocket_pred = self.pocket_cls_all(all_rep)
        else:
            pocket_pred = None

        return pocket_pred, self._num_updates

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

@register_model_architecture("ESM_pocket", "ESM_pocket")
def base_architecture(args):
    pass
