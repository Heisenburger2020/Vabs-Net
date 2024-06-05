# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging
from collections.abc import Sequence

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

try:
    import torch_scatter
except:
    pass

try:
# if True:
    import torchdrug
    from torchdrug.layers.geometry import GraphConstruction, AlphaCarbonNode, SequentialEdge, SpatialEdge, KNNEdge
except:
    pass
from .layer_gear import GeometricRelationalGraphConv, IEConvLayer, GearNetIEConv, GeometryAwareRelationalGraphNeuralNetwork

logger = logging.getLogger(__name__)

def print_all_shape(*tensors):
    print("shape>>", [_.shape for _ in tensors])

@register_model("Gear_pocket")
class Gear_pocket_Encoder(BaseUnicoreModel):
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
            "--gamma-neg",
            type=int,
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
            "--inference",
            type=float,
            default=0
        )
        parser.add_argument(
            "--siam-type",
            type=int,
            default=0
        )

    def __init__(self, args):
        super().__init__()
        self.args = args   
        self._num_updates = 0
        self.dtype = torch.float32
        node_layers = nn.ModuleList([AlphaCarbonNode()])
        edge_layers = nn.ModuleList([SequentialEdge(max_distance=2),
                                     SpatialEdge(radius=10.0, min_distance=5),
                                     KNNEdge(k=10, max_distance=5)])

        self.graph_construct = GraphConstruction(node_layers=node_layers,
                                                 edge_layers=edge_layers,
                                                 edge_feature="gearnet")
        self.gearNet = GeometryAwareRelationalGraphNeuralNetwork(
            input_dim = 21,
            hidden_dims = [512, 512, 512, 512, 512, 512],
            batch_norm = True,
            concat_hidden = True,
            short_cut = True,
            readout = 'sum',
            num_relation = 7,
            edge_input_dim = 59,
            num_angle_bin = 8,
        ).float()

        loaded_params = torch.load('./store/mc_gearnet_edge.pth')
        # 将参数加载到模型中
        self.gearNet.load_state_dict(loaded_params)

        self.pocket_cls_all = CLSHead(input_dim=3072, output_cls=2, preln=args.preln)

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
            graph, 
            input,
            residue_type,
            **kwargs
        ):
        graph = self.graph_construct(graph).to(input.device)
        node_feat, graph_feat = self.gearNet(graph, input)
        pocket_pred = self.pocket_cls_all(node_feat)
        if self.args.inference:
            return node_feat[:, -512:], residue_type
        return pocket_pred, self._num_updates


    def get_ieconv_edge_feature(self, graph):
        u = torch.ones_like(graph.node_position)
        u[1:] = graph.node_position[1:] - graph.node_position[:-1]
        u = F.normalize(u, dim=-1)
        b = torch.ones_like(graph.node_position)
        b[:-1] = u[:-1] - u[1:]
        b = F.normalize(b, dim=-1)
        n = torch.ones_like(graph.node_position)
        n[:-1] = torch.cross(u[:-1], u[1:])
        n = F.normalize(n, dim=-1)

        local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)

        node_in, node_out = graph.edge_list.t()[:2]
        t = graph.node_position[node_out] - graph.node_position[node_in]
        t = torch.einsum('ijk, ij->ik', local_frame[node_in], t)
        r = torch.sum(local_frame[node_in] * local_frame[node_out], dim=1)
        delta = torch.abs(graph.atom2residue[node_in] - graph.atom2residue[node_out]).float() / 6
        delta = delta.unsqueeze(-1)

        return torch.cat([
            t, r, delta, 
            1 - 2 * t.abs(), 1 - 2 * r.abs(), 1 - 2 * delta.abs()
        ], dim=-1)


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

@register_model_architecture("Gear_pocket", "Gear_pocket")
def base_architecture(args):
    pass
