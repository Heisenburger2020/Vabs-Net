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
from typing import Any, Dict, Optional, Union, Tuple
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
    ECGOCLSHead,
)
from .vabs_encoder import VabsEncoder
from unicore.modules import init_bert_params
from unicore.utils import checkpoint_sequential
from functools import partial
from .esm_utils import load_model_and_alphabet_local

logger = logging.getLogger(__name__)

def print_all_shape(*tensors):
    print("shape>>", [_.shape for _ in tensors])

restype_order_with_x = {'*': 23,'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20}
res_num2str = {v: k for k, v in restype_order_with_x.items()}

@register_model("protein_pocket")
class MDProteinModel(BaseUnicoreModel):
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
            "--angle-dim",
            type=int,
        )
        parser.add_argument(
            "--node-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for node fitting",
        )
        parser.add_argument(
            "--min-node-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for node fitting",
        )
        parser.add_argument(
            "--dist-loss",
            default=0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--eng-loss-weight",
            type=float,
            metavar="D",
            help="loss weight for energy fitting",
        )
        parser.add_argument(
            "--use-ion",
            type=float,
            metavar="D",
            help="dropout probability for pair",
        )
        parser.add_argument(
            "--data-seed",
            type=int,
        )

        parser.add_argument(
            "--usefl",
            type=int,
        )
        parser.add_argument(
            "--usemae",
            type=int,
        )
        parser.add_argument(
            "--fl-alpha",
            type=float,
        )
        parser.add_argument(
            "--fl-gamma",
            type=int,
        )
        parser.add_argument(
            "--gamma-pos",
            type=float,
        )
        parser.add_argument(
            "--atom-base",
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--res2res-mask",
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--res2atom-mask",
            type=float,
            help="the probability of using label conformer as input",
        )
        parser.add_argument(
            "--atom2res-mask",
            type=float,
            help="the probability of using label conformer as input",
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

        ## new
        parser.add_argument(
            "--knn", type=int, help="num encoder layers"
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
            "--cross-layer",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-clean",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-pointgnn",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-esm-feat",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-pos",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-virtual-node",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-sas",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-maskinter",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-res2atom",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-ps-data",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-rel-pos",
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
            "--use-af",
            type=int,
        )
        parser.add_argument(
            "--finetune",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-pdb",
            type=int,
            default=1
        )
        parser.add_argument(
            "--use-graphv2",
            type=int,
            default=0
        )
        parser.add_argument(
            "--aa-neighbor",
            type=int,
            default=4
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
            "--if-ratio",
            type=float,
            default=0.5
        )
        parser.add_argument(
            "--use-torsion",
            type=float,
            default=0
        )
        parser.add_argument(
            "--pred-r-pos",
            type=float,
            default=0
        )
        parser.add_argument(
            "--refine",
            type=float,
            default=0
        )
        parser.add_argument(
            "--use-largest",
            type=float,
            default=1
        )
        parser.add_argument(
            "--fold-3d",
            type=float,
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
        parser.add_argument(
            "--use-ss",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-HMM",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-DSSP",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-PSSM",
            type=int,
            default=0
        )
        parser.add_argument(
            "--sas-pred",
            type=int,
            default=0
        )
        parser.add_argument(
            "--mask-same",
            type=int,
            default=0
        )
        parser.add_argument(
            "--mask-side-chain",
            type=int,
            default=0
        )
        parser.add_argument(
            "--local-all",
            type=int,
            default=0
        )
        parser.add_argument(
            "--concat-style",
            type=int,
            default=0
        )
        parser.add_argument(
            "--learnable-pos",
            type=int,
            default=0
        )
        parser.add_argument(
            "--add-style",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-relative",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-absolute",
            type=int,
            default=1
        )
        parser.add_argument(
            "--use-nerf-encoding",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-all-layers",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-valid",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-asl",
            type=int,
            default=0
        )
        parser.add_argument(
            "--EC",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-context",
            type=int,
            default=0
        )
        parser.add_argument(
            "--gamma-neg",
            type=int,
            default=4
        )
        parser.add_argument(
            "--cls-type",
            type=str,
            default=""
        )
        parser.add_argument(
            "--tmpstore",
            type=str,
            default=""
        )
        parser.add_argument(
            "--esm-dim",
            type=int,
            default=1280
        )
        parser.add_argument(
            "--esm-stack",
            type=int,
            default=0
        )
        parser.add_argument(
            "--atom-feat",
            type=int,
            default=0
        )
        parser.add_argument(
            "--edge-mask-prob",
            type=float,
            default=0
        )
        parser.add_argument(
            "--extra-loss",
            type=float,
            default=0
        )
        parser.add_argument(
            "--random-fourier",
            type=float,
            default=0
        )
        parser.add_argument(
            "--inference",
            type=float,
            default=0
        )
        parser.add_argument(
            "--reverse",
            type=float,
            default=0
        )
        parser.add_argument(
            "--rdfft",
            type=int,
            default=0
        )
        parser.add_argument(
            "--residue-only-ratio",
            type=float,
            default=0
        )
        parser.add_argument(
            "--span-mask",
            type=int,
            default=0
        )
        parser.add_argument(
            "--residue-only",
            type=int,
            default=0
        )
        parser.add_argument(
            "--v-attn",
            type=int,
            default=0
        )
        parser.add_argument(
            "--sequence-only",
            type=int,
            default=0
        )
        parser.add_argument(
            "--structure-bias-only",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-rope",
            type=int,
            default=0
        )
        parser.add_argument(
            "--use-edge-attn",
            type=int,
            default=0
        )
        parser.add_argument(
            "--query2label",
            type=int,
            default=0
        )
        parser.add_argument(
            "--num-virtual-point",
            type=int,
            default=3
        )
        parser.add_argument(
            "--ft-esm",
            type=int,
            default=0
        )
        parser.add_argument(
            "--more-cls",
            type=int,
            default=0
        )



    def __init__(self, args):
        super().__init__()
        base_architecture(args)
        self.args = args   
        self.num_rbf = 16
        self.num_embeding = 16
            
        self._num_updates = 0
        self.dtype = torch.float32

        if self.args.ft_esm:
            esm_ckpt_path = r"./ckpt/esm2_t33_650M_UR50D.pt"
            self.esm_model, self.esm_alphabet = load_model_and_alphabet_local(esm_ckpt_path)


            self.no_loss = [
            "contact_head.regression.weight",
            "contact_head.regression.bias",
            "lm_head.bias",
            "lm_head.dense.weight",
            "lm_head.dense.bias",
            "lm_head.layer_norm.weight",
            "lm_head.layer_norm.bias",
            ]
            for param in self.esm_model.named_parameters():
                # if param[0] in self.no_loss:
                #     param[1].requires_grad = False
                if param[0] in self.no_loss:
                    param[1].requires_grad = False
            self.batch_converter = self.esm_alphabet.get_batch_converter()

        self.molecule_encoder =  VabsEncoder(
            num_encoder_layers=args.num_layers,
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            pair_hidden_dim=args.pair_hidden_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            activation_fn=args.activation_fn,
            droppath_prob=args.droppath_prob,
            pair_dropout=args.pair_dropout,
            use_outer=args.use_outer,
            use_trimul=args.use_trimul,
            use_pairupdate=args.use_pairupdate,
            use_triangleattn=args.use_triangleattn,
            cross_layer = args.cross_layer,
            preln=args.preln,
            concat_style=args.concat_style,
            use_context=args.use_context,
            residue_only = args.residue_only,
            v_attn=args.v_attn,
            sequence_only=self.args.sequence_only,
            use_rope=args.use_rope,
            use_edge_attn=args.use_edge_attn
        )

        self.residue_types = 21 + 4
        self.atom_types = 37 + 3

        self.residue_feature = nn.Embedding(
            self.residue_types, self.args.node_dim, padding_idx=21
        )

        if self.args.use_ss:
            # secondary structure
            self.ss = nn.Embedding(
                9, self.args.node_dim
            )
        if not self.args.residue_only:
            self.atom_feature = nn.Embedding(
                self.atom_types, self.args.node_dim,
            )
        
        if not self.args.sequence_only:
            self.rr_se3_invariant_kernel = SE3InvariantKernel(
                pair_dim=args.edge_dim,
                num_pair=512,
                num_kernel=args.num_3d_bias_kernel
            )

            if not self.args.residue_only:
                self.aa_se3_invariant_kernel = SE3InvariantKernel(
                    pair_dim=args.edge_dim,
                    num_pair=512,
                    num_kernel=args.num_3d_bias_kernel
                )

            self.rr_edge_encoder = ProteinEdgeFeature(args.edge_dim, None, self.atom_types)
            if not self.args.residue_only:
                self.aa_edge_encoder = ProteinEdgeFeature(args.edge_dim, self.atom_types, None)

        if self.args.use_esm_feat:
            self.esm_feat_encoder = NodeEmbedHead(args.esm_dim, args.node_dim)

        if self.args.gb_feat:
            # hmm pssm dssp
            self.node_gb_dim = 0
            if self.args.use_HMM:
                self.node_gb_dim += 30
            if self.args.use_PSSM:
                self.node_gb_dim += 20
            if self.args.use_DSSP:
                self.node_gb_dim += 14
            self.embed_gb_node = MLPs(self.node_gb_dim + args.node_dim, args.node_dim)

        self.pos_encoding_dim = self.num_embeding
        if not self.args.sequence_only:
            if not self.args.learnable_pos:
                assert 0, "better learnable"
                self.rr_positional_edge = Linear(self.pos_encoding_dim, args.edge_dim)
                if not self.args.residue_only:
                    self.aa_positional_edge = Linear(self.pos_encoding_dim, args.edge_dim)
            else:
                self.pos_cut_off = 256
                self.rr_positional_edge = nn.Embedding(
                    2 * self.pos_cut_off + 1, args.edge_dim,
                )

                self.v_pos_cut_off = 1200
                self.v_positional_edge = nn.Embedding(
                    2 * self.v_pos_cut_off + 1, args.edge_dim,
                )

                if not self.args.residue_only:
                    self.aa_positional_edge = nn.Embedding(
                        2 * self.pos_cut_off + 1, args.edge_dim,
                    )
                    self.aa_v_pos_cut_off = 1200
                    self.aa_v_positional_edge = nn.Embedding(
                        2 * self.v_pos_cut_off + 1, args.edge_dim,
                    )

            if self.args.use_vec:
                vec_dim = 0
                if not self.args.use_nerf_encoding:
                    vec_feat_dim = 6
                else:
                    vec_feat_dim= 90
                if self.args.use_absolute:
                    vec_dim += vec_feat_dim
                if self.args.use_relative:
                    vec_dim += vec_feat_dim
                assert vec_dim > 0
                self.vec_rr_embed = VecEdgeFeature(vec_dim + args.edge_dim, args.edge_dim)
                if not self.args.residue_only:
                    self.vec_aa_embed = VecEdgeFeature(vec_dim + args.edge_dim, args.edge_dim)

        if self.args.pocket_pred:
            if not self.args.use_all_layers:
                self.pocket_cls_all = CLSHead(input_dim=args.node_dim, output_cls=2, preln=args.preln)
            elif self.args.esm_stack:
                self.pocket_cls_all = CLSHead(input_dim=args.node_dim * self.args.num_layers + self.args.esm_dim, output_cls=2, preln=args.preln)
            else:
                self.pocket_cls_all = CLSHead(input_dim=args.node_dim * self.args.num_layers, output_cls=2, preln=args.preln)

        if self.args.res_type_pred:
            self.res_type_cls = CLSHead(input_dim=args.node_dim, output_cls=self.residue_types, preln=args.preln)
            if self.args.refine:
                self.res_type_cls_halfway = CLSHead(input_dim=args.node_dim, output_cls=self.residue_types, preln=args.preln)

        if self.args.pos_pred:
            self.atom_pos_head = MovementPredictionHead(
                args.node_dim, args.edge_dim, args.encoder_attention_heads, args.preln
            )
            self.atom_pos_head.zero_init()

        if self.args.fold_3d:
            self.fold_class_num = 1195
            self.fold3d_cls = CLSHead(input_dim=args.node_dim * args.num_layers, output_cls=self.fold_class_num, preln=args.preln)

        if self.args.EC:
            if self.args.cls_type == "EC":
                self.EC_class_num = 538 + self.args.more_cls
            elif self.args.cls_type == "MF":
                self.EC_class_num = 489 + self.args.more_cls
            elif self.args.cls_type == "BP":
                self.EC_class_num = 1943 + self.args.more_cls
            elif self.args.cls_type == "CC":
                self.EC_class_num = 320 + self.args.more_cls
            elif self.args.cls_type == "PSR":
                self.EC_class_num = 1
            if not self.args.query2label:
                self.EC_cls = CLSHead(input_dim=args.node_dim * args.num_layers, output_cls=self.EC_class_num, preln=args.preln)
            else:
                self.EC_cls = ECGOCLSHead(input_dim=args.node_dim, hidden_dim=args.node_dim, output_cls=self.EC_class_num, preln=args.preln)

        if self.args.sas_pred:
            self.sas_cls = CLSHead(input_dim=args.node_dim, output_cls=1, preln=args.preln)

        if self.args.dihedral_pred:
            self.dihedral_cls = SidechainAngleResnet(d_in=args.node_dim, d_hid=128, num_blocks=2, num_angles=7, preln=self.args.preln)
    
    def half(self):
        super().half()

        self.molecule_encoder = self.molecule_encoder.half()
        self.residue_feature = self.residue_feature.half()
        if not self.args.residue_only:
            self.atom_feature = self.atom_feature.half()
        if self.args.use_esm_feat:
            self.esm_feat_encoder = self.esm_feat_encoder.half()
        if not self.args.sequence_only:
            if not self.args.residue_only:
                self.aa_edge_encoder = self.aa_edge_encoder.half()
            self.rr_edge_encoder = self.rr_edge_encoder.half()
            if not self.args.residue_only:
                self.aa_se3_invariant_kernel = self.aa_se3_invariant_kernel.half()

            self.rr_se3_invariant_kernel = self.rr_se3_invariant_kernel.half()

        if self.args.pocket_pred:
            self.pocket_cls_all = self.pocket_cls_all.half()
        if self.args.res_type_pred:
            self.res_type_cls = self.res_type_cls.half()
        if self.args.pos_pred:
            self.atom_pos_head = self.atom_pos_head.half()
        if self.args.dihedral_pred:
            self.dihedral_cls = self.dihedral_cls.half()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        super().bfloat16()

        self.molecule_encoder = self.molecule_encoder.bfloat16()
        self.residue_feature = self.residue_feature.bfloat16()
        if not self.args.residue_only:
            self.atom_feature = self.atom_feature.bfloat16()
        if self.args.use_esm_feat:
            self.esm_feat_encoder = self.esm_feat_encoder.bfloat16()
        if not self.args.sequence_only:
            if not self.args.residue_only:
                self.aa_edge_encoder = self.aa_edge_encoder.bfloat16()
            self.rr_edge_encoder = self.rr_edge_encoder.bfloat16()
            if not self.args.residue_only:
                self.aa_se3_invariant_kernel = self.aa_se3_invariant_kernel.bfloat16()

            self.rr_se3_invariant_kernel = self.rr_se3_invariant_kernel.bfloat16()


        if self.args.pocket_pred:
            self.pocket_cls_all = self.pocket_cls_all.bfloat16()
        if self.args.res_type_pred:
            self.res_type_cls = self.res_type_cls.bfloat16()
        if self.args.pos_pred:
            self.atom_pos_head = self.atom_pos_head.bfloat16()
        if self.args.dihedral_pred:
            self.dihedral_cls = self.dihedral_cls.bfloat16()
        self.dtype = torch.bfloat16
        return self

    def float(self):
        super().float()

        self.molecule_encoder = self.molecule_encoder.float()
        self.residue_feature = self.residue_feature.float()
        if not self.args.residue_only:
            self.atom_feature = self.atom_feature.float()
        if self.args.use_esm_feat:
            self.esm_feat_encoder = self.esm_feat_encoder.float()
        if not self.args.sequence_only:
            if not self.args.residue_only:
                self.aa_edge_encoder = self.aa_edge_encoder.float()
            self.rr_edge_encoder = self.rr_edge_encoder.float()
            if not self.args.residue_only:
                self.aa_se3_invariant_kernel = self.aa_se3_invariant_kernel.float()

            self.rr_se3_invariant_kernel = self.rr_se3_invariant_kernel.float()


        if self.args.pocket_pred:
            self.pocket_cls_all = self.pocket_cls_all.float()
        if self.args.res_type_pred:
            self.res_type_cls = self.res_type_cls.float()
        if self.args.pos_pred:
            self.atom_pos_head = self.atom_pos_head.float()
        if self.args.dihedral_pred:
            self.dihedral_cls = self.dihedral_cls.float()
        self.dtype = torch.float32
        return self

    def forward(
            self, 
            # residue_pos: torch.Tensor,
            # residue_type: torch.Tensor, 
            # residue_pair_type: torch.Tensor,
            atom_pos: torch.Tensor,
            residue_type: torch.Tensor,
            residue_idx_all: torch.Tensor,
            atom_type: torch.Tensor,
            edge_index: torch.Tensor,
            aa_edge_index: torch.Tensor,     
            edge_vec: torch.Tensor,
            edge_aa_vec: torch.Tensor,
            res_mask: torch.Tensor,
            batch_index_res: torch.Tensor,
            batch_index: torch.Tensor,
            edge_index_left: torch.Tensor,
            edge_index_right: torch.Tensor,
            gb_feat: torch.Tensor=None,
            ss: torch.Tensor=None,
            torsion: torch.Tensor=None,
            torsion_mask: torch.Tensor=None,
            atom_sas: torch.Tensor = None,
            esm_feat: torch.Tensor = None,
            atom_pred_pos_index: torch.Tensor = None,
            atom_feat: torch.Tensor = None,
            idx: torch.Tensor = None,
            is_train: torch.Tensor = None,
            edge_bias_index: torch.Tensor = None,
            **kwargs
        ):
        residue_type = residue_type.long()
        if self.args.ft_esm:
            residue_type_str = ""
            
            for idx in range(residue_type.shape[0]):
                num = residue_type[idx]
                residue_type_str += res_num2str[num.item()]

            list_str = residue_type_str.split("*")[:-1]
            input_batch = [(str(_), list_str[_]) for _ in range(len(list_str))]
            _, _, batch_tokens = self.batch_converter(input_batch)
            batch_tokens = batch_tokens.to(batch_index.device)
            esm_embed = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
            esm_embed_list = [esm_embed["representations"][33][id, 1 : len(item) + 1] for id, item in enumerate(list_str)]
            esm_embed_cls_list = [esm_embed["representations"][33][id, 0].unsqueeze(0) for id, item in enumerate(list_str)]
            esm_embed_tail_list = [esm_embed["representations"][33][id, len(item) + 1].unsqueeze(0) for id, item in enumerate(list_str)]
            head = torch.cat(esm_embed_cls_list, dim=0)
            tail = torch.cat(esm_embed_tail_list, dim=0)
            all_rep = []
            for i in range(len(esm_embed_cls_list)):
                all_rep.append(esm_embed_list[i])
                all_rep.append(esm_embed_cls_list[i])
            esm_feat = torch.cat(all_rep)

        res_mask = res_mask.bool()
        res_residue_type = residue_type[res_mask]

        if edge_bias_index is not None:
            assert 0
            edge_index_all = edge_index.clone()
            edge_index = edge_index_all[edge_bias_index]
        else:
            edge_index_all = None

        if not self.args.sequence_only:
            atom_pos = atom_pos.type(self.dtype)
            edge_vec = edge_vec.type(self.dtype)
            if not self.args.residue_only:
                edge_aa_vec = edge_aa_vec.type(self.dtype)
            v_mask = ((res_residue_type[edge_index[:,0]] == 23) + (res_residue_type[edge_index[:,1]] == 23)) > 0
            v_mask_2 = res_residue_type[edge_index[:,0]] == 23

            if not self.args.residue_only:
                v_mask_aa = ((residue_type[aa_edge_index[:,0]] == 23) + (residue_type[aa_edge_index[:,1]] == 23)) > 0
        
        if not self.args.residue_only:
            atom_nodes = self.atom_feature(atom_type) + self.residue_feature(residue_type)
        else:
            atom_nodes = self.residue_feature(res_residue_type)

        if self.args.use_esm_feat:
            esm_feat_ = self.esm_feat_encoder(esm_feat)
            if not self.args.residue_only:
                atom_nodes = esm_feat_[residue_idx_all] + atom_nodes
            else:
                atom_nodes = esm_feat_ + atom_nodes
        
        if not self.args.sequence_only:
            residue_pos = atom_pos[res_mask]
            rr_edges = self.rr_edge_encoder(res_residue_type, edge_index)

            if not self.args.residue_only:
                aa_edges = self.aa_edge_encoder(None, aa_edge_index, atom_type)
            else:
                aa_edges = None

            # res to res
            edge_rr_attr = torch.cat([res_residue_type[edge_index[:,0]].unsqueeze(1), res_residue_type[edge_index[:,1]].unsqueeze(1) ], dim=-1)
            edge_rr_attr[:,0] += 1
            edge_rr_attr[:,1] += 1+128
            edge_diff = residue_pos[edge_index[:,0]] - residue_pos[edge_index[:,1]]
            edge_weight = edge_diff.norm(dim=-1)
            attn_bias_3d = self.rr_se3_invariant_kernel(edge_weight.detach(), edge_rr_attr)
            rr_edges = rr_edges + attn_bias_3d

            if not self.args.residue_only:
                # atom to atom
                edge_aa_attr = torch.cat([atom_type[aa_edge_index[:,0]].unsqueeze(1), atom_type[aa_edge_index[:,1]].unsqueeze(1)], dim=-1)
                edge_aa_attr[:,0] += 1
                edge_aa_attr[:,1] += 1+128

                aa_edge_diff = atom_pos[aa_edge_index[:,0]] - atom_pos[aa_edge_index[:,1]]
                edge_weight = aa_edge_diff.norm(dim=-1)
                attn_bias_3d = self.aa_se3_invariant_kernel(edge_weight.detach(), edge_aa_attr)
                aa_edges = aa_edges + attn_bias_3d

            if self.args.use_vec:
                rr_edges = self.vec_rr_embed(rr_edges, edge_vec)       
                if not self.args.residue_only:
                    aa_edges = self.vec_aa_embed(aa_edges, edge_aa_vec)

            if not self.args.learnable_pos:
                assert 0
                edges_pos_embed = self.rr_positional_edge(self._positional_embeddings(edge_index)).type(self.dtype)
                aa_edges_pos_embed = self.aa_positional_edge(self._positional_embeddings(torch.cat([residue_idx_all[aa_edge_index[:, 0]].unsqueeze(-1), residue_idx_all[aa_edge_index[:, 1]].unsqueeze(-1)], dim=-1)))
                rr_edges += edges_pos_embed
                aa_edges += aa_edges_pos_embed
            else:
                rr_edges[v_mask] += self.v_positional_edge(torch.clamp(edge_index[:, 0][v_mask] - edge_index[:, 1][v_mask], min=-self.v_pos_cut_off, max=self.v_pos_cut_off) + self.v_pos_cut_off)
                rr_edges[~v_mask] += self.rr_positional_edge(torch.clamp(edge_index[:, 0][~v_mask] - edge_index[:, 1][~v_mask], min=-self.pos_cut_off, max=self.pos_cut_off) + self.pos_cut_off)
                if not self.args.residue_only:
                    aa_edges[v_mask_aa] += self.aa_v_positional_edge(torch.clamp(residue_idx_all[aa_edge_index[:, 0]][v_mask_aa] - residue_idx_all[aa_edge_index[:, 1]][v_mask_aa], min=-self.v_pos_cut_off, max=self.v_pos_cut_off) + self.v_pos_cut_off)
                    aa_edges[~v_mask_aa] += self.aa_positional_edge(torch.clamp(residue_idx_all[aa_edge_index[:, 0]][~v_mask_aa] - residue_idx_all[aa_edge_index[:, 1]][~v_mask_aa], min=-self.pos_cut_off, max=self.pos_cut_off) + self.pos_cut_off)

            rr_edges = rr_edges.type(self.dtype)     

            if self.args.gb_feat:
                if not self.args.add_style:
                    atom_nodes = self.embed_gb_node(atom_nodes, gb_feat[residue_idx_all])
                else:
                    atom_nodes += self.embed_gb_node(atom_nodes, gb_feat[residue_idx_all])
            if self.args.use_ss:
                assert torch.all(ss < 8), ss
                atom_nodes = atom_nodes + self.ss(ss[residue_idx_all])
        else:
            rr_edges = aa_edges = None


        atom_nodes = atom_nodes.type(self.dtype)
        if not self.args.residue_only:
            aa_edges = aa_edges.type(self.dtype) 
            atom_pos = atom_pos.type(self.dtype)

        if edge_bias_index is not None:
            atom_nodes, rr_edges, aa_edges, atom_nodes_all_layer = self.molecule_encoder(
                nodes=atom_nodes, 
                edges=rr_edges, 
                aa_edges=aa_edges, 
                edge_index=edge_index, 
                aa_edge_index=aa_edge_index,
                res_mask=res_mask,
                batch_index_res=batch_index_res,
                edge_index_left=edge_index_left,
                edge_index_right=edge_index_right,
                v_mask=v_mask_2
            )
        else:
            atom_nodes, rr_edges, aa_edges, atom_nodes_all_layer = self.molecule_encoder(
                nodes=atom_nodes, 
                edges=rr_edges, 
                aa_edges=aa_edges, 
                edge_index=edge_index, 
                aa_edge_index=aa_edge_index,
                res_mask=res_mask,
                batch_index_res=batch_index_res,
                edge_index_left=edge_index_left,
                edge_index_right=edge_index_right,
                v_mask=v_mask_2,
                edge_index_all=edge_index_all,
                edge_bias_index=edge_bias_index,
            )

        if self.args.inference:
            assert 0, "used to inference feature"
            not_cls = residue_type.ne(23)
            not_cls = not_cls[res_mask]
            def output_res(atom_types, residue_ids, atom_features, num_residues):
                residue_features = torch.zeros(num_residues, 38, 768).type(self.dtype)

                for i in range(atom_types.size(0)):
                    if atom_types[i] < 37:
                        residue_features[residue_ids[i], atom_types[i]] = atom_features[i]
                return residue_features
            residue_features = output_res(atom_type, residue_idx_all, atom_nodes, res_mask.sum())
            residue_features = residue_features[not_cls.cpu()]
            residue_features = residue_features[:, :37, :]
            assert residue_features.shape[0] == atom_nodes[res_mask][not_cls].shape[0]
            assert residue_features.shape[1] == 37
            # print(residue_features.shape, atom_nodes.shape, atom_nodes[res_mask][not_cls].shape)
            return atom_nodes[res_mask][not_cls], residue_type[res_mask][not_cls], batch_index_res[not_cls], residue_features
        if self.args.res_type_pred:
            res_pred_type_index = residue_type.eq(22)
            if not self.args.residue_only:
                res_type_pred = self.res_type_cls(atom_nodes[res_pred_type_index])
            else:
                res_pred_type_index = res_residue_type.eq(22)
                res_type_pred = self.res_type_cls(atom_nodes[res_pred_type_index])

        else:
            res_pred_type_index = residue_type.eq(22)
            res_type_pred = None
          
        if self.args.dihedral_pred:
            res_pred_type_index_t = residue_type < 23
            if self.args.refine:
                unnormalized_dihedral_perd, dihedral_perd = self.dihedral_cls(atom_nodes[res_pred_type_index_t][res_mask[res_pred_type_index_t]], atom_nodes_halfway[res_pred_type_index_t][res_mask[res_pred_type_index_t]])
            else:
                if not self.args.residue_only:
                    unnormalized_dihedral_perd, dihedral_perd = self.dihedral_cls(atom_nodes[res_pred_type_index_t][res_mask[res_pred_type_index_t]])
                else:
                    res_pred_type_index_t = res_residue_type < 23
                    unnormalized_dihedral_perd, dihedral_perd = self.dihedral_cls(atom_nodes[res_pred_type_index_t][res_mask[res_pred_type_index_t]])

        else:
            dihedral_perd = None
            res_pred_type_index_t = residue_type.ne(23)
            unnormalized_dihedral_perd = None

        if self.args.pos_pred:
            if not self.args.residue_only:
                atom_pos_pred = atom_pos + self.atom_pos_head(
                    atom_nodes,
                    aa_edge_index,
                    aa_edge_diff.detach(),
                    aa_edges,
                )
            else:
                atom_pos_pred = atom_pos + self.atom_pos_head(
                    atom_nodes,
                    edge_index,
                    edge_diff.detach(),
                    rr_edges,
                )
        else:
            atom_pos_pred = None

        if self.args.pocket_pred:
            if not self.args.use_all_layers:
                pocket_pred = self.pocket_cls_all(atom_nodes)
            else:
                pocket_pred = self.pocket_cls_all(torch.cat(atom_nodes_all_layer, dim=-1))
        else:
            pocket_pred = None

        if self.args.sas_pred:
            sas_pred = self.sas_cls(atom_nodes)
        else:
            sas_pred = None

        if self.args.fold_3d:
            mask_pred_label = residue_type.eq(23)
            pred_label = self.fold3d_cls(torch.cat(atom_nodes_all_layer, dim=-1)[mask_pred_label])
        elif self.args.EC:
            if not self.args.use_virtual_node:
                assert 0
                bsz = (batch_index_res[-1].data + 1).cpu().item()
                bsz_count = int(bsz)
                pred_label = torch.zeros(int(bsz), self.EC_class_num).to(atom_nodes.device)
                for i in range(bsz_count):
                    pred_label[i] = self.EC_cls(torch.cat(atom_nodes_all_layer, dim=-1)[res_mask][batch_index_res == i].sum(0))
            else:
                # only use virtual node

                if not self.args.query2label:
                    mask_pred_label = residue_type.eq(23)
                    pred_label = self.EC_cls(torch.cat(atom_nodes_all_layer, dim=-1)[mask_pred_label])
                else:
                    pred_label = self.EC_cls(atom_nodes[res_mask], batch_index_res)
        else:
            pred_label = None
        
        return res_type_pred, None, res_pred_type_index, pocket_pred, atom_pos_pred, atom_pred_pos_index, (unnormalized_dihedral_perd, dihedral_perd, res_pred_type_index_t), pred_label, sas_pred, self._num_updates


    def _positional_embeddings(self, edge_index, 
                               num_embeddings=16):
        num_embeddings = num_embeddings
        d = edge_index[:, 0] - edge_index[:, 1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32,
                device=edge_index.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E.type(self.dtype)


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates



@register_model_architecture("protein_pocket", "protein_pocket")
def base_architecture(args):

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.pair_embed_dim = getattr(args, "pair_embed_dim", 256)
    args.pair_hidden_dim = getattr(args, "pair_hidden_dim", 32)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 48)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.droppath_prob = getattr(args, "droppath_prob", 0.1)
    args.pair_dropout = getattr(args, "pair_dropout", 0.25)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.1)
    args.dropout = getattr(args, "dropout", 0.0)
    args.num_3d_bias_kernel = getattr(args, "num_3d_bias_kernel", 128)
    args.num_block = getattr(args, "num_block", 4)
    args.pretrain = getattr(args, "pretrain", False)
    args.notri = getattr(args, "notri", False)
    args.pos_step_size = getattr(args, "pos_step_size", 1.0)
    args.gaussian_std_width = getattr(args, "gaussian_std_width", 1.0)
    args.gaussian_mean_start = getattr(args, "gaussian_mean_start", 0.0)
    args.gaussian_mean_stop = getattr(args, "gaussian_mean_stop", 9.0)


    args.num_layers = getattr(args, "num_layers", 5)
    args.node_dim = getattr(args, "node_dim", 128)
    args.edge_dim = getattr(args, "edge_dim", 128)
    args.kernel_size = getattr(args, "kernel_size", 32)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 2048)
    args.init_method = getattr(args, "init_method", "bert")
    args.pred_init_method = getattr(args, "pred_init_method", None)
    args.outer_product_dim = getattr(args, "outer_product_dim", 32)
    args.inter_edge_mlp_layers = getattr(args, "inter_edge_mlp_layers", 2)
    args.norm_layer = getattr(args, "norm_layer", "none")


    args.use_outer = getattr(args, "use_outer", 0)
    args.use_trimul = getattr(args, "use_trimul", 0)
    args.use_pairupdate = getattr(args, "use_pairupdate", 0)
    args.use_triangleattn = getattr(args, "use_triangleattn", 0)
    

@register_model_architecture("protein_pocket", "protein_pocket_base")
def unimol_base_architecture(args):
    base_architecture(args)

@register_model_architecture("protein_pocket", "protein_pocket_large")
def unimol_large_architecture(args):
    args.node_dim = getattr(args, "node_dim", 512)
    args.edge_dim = getattr(args, "edge_dim", 512)
    args.kernel_size = getattr(args, "kernel_size", 64)
    # args.edge_mlp_layers = getattr(args, "edge_mlp_layers", 10)
    # args.outer_product_dim = getattr(args, "outer_product_dim", 64)
    base_architecture(args)
    