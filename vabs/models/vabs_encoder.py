import imp
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from unicore.modules import LayerNorm

from .layers import (
    TransformerEncoderLayer,
    Dropout,
)

class VabsEncoder(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 6,
        node_dim: int = 768,
        edge_dim: int = 64,
        pair_hidden_dim: int = 32,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "gelu",
        droppath_prob: float = 0.0,
        pair_dropout: float = 0.25,
        use_outer: float = 0.0,
        use_trimul: float = 0.0,
        use_pairupdate: float = 0.0,
        use_triangleattn: int = 0.0,
        cross_layer: int=0,
        preln:int=0,
        concat_style=0,
        use_context=0,
        residue_only=0,
        v_attn=0,
        sequence_only=0,
        use_rope=0,
        use_edge_attn=0
    ) -> None:

        super().__init__()
        self.embedding_dim = node_dim
        self.num_head = num_attention_heads
        self.residue_only = residue_only
        self.sequence_only = sequence_only
        if not preln:
            self.layer_norm = LayerNorm(node_dim)
            if not sequence_only:
                self.pair_layer_norm = LayerNorm(edge_dim)

        self.layers = nn.ModuleList([])
        self.preln = preln
        
        if droppath_prob > 0:
            droppath_probs = [
                x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)
            ]
        else:
            droppath_probs = None

        if cross_layer == 0:
            cross_layers = [True for _ in range(num_encoder_layers)]
        else:
            cross_layers = [ _ % cross_layer == 0 for _ in range(num_encoder_layers)]
        
        self.layers.extend(
            [
               TransformerEncoderLayer(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    pair_hidden_dim=pair_hidden_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    droppath_prob=droppath_probs[i],
                    pair_dropout=pair_dropout,
                    use_outer=use_outer,
                    use_trimul=use_trimul,
                    use_pairupdate=use_pairupdate,
                    use_triangleattn=use_triangleattn,
                    cross_layer=cross_layers[i],
                    is_end=i==(num_encoder_layers - 1),
                    preln=preln,
                    concat_style=concat_style,
                    use_context=use_context,
                    residue_only=residue_only,
                    v_attn=v_attn,
                    sequence_only=sequence_only,
                    use_rope=use_rope,
                    use_edge_attn=use_edge_attn
                )
                for i in range(num_encoder_layers)
            ]
        )

    def forward(
        self,
        nodes, 
        edges, 
        aa_edges, 
        edge_index, 
        aa_edge_index,
        res_mask,
        batch_index_res,
        edge_index_left,
        edge_index_right,
        v_mask=None,
        edge_index_all=None,
        edge_bias_index=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.preln:
            nodes = self.layer_norm(nodes)
            if not self.sequence_only:
                edges = self.pair_layer_norm(edges)
        atom_nodes_all_layer = []
        for i, layer in  enumerate(self.layers):
            nodes, edges, aa_edges = layer(
                                nodes, 
                                edges, 
                                aa_edges, 
                                edge_index, 
                                aa_edge_index,
                                res_mask,
                                batch_index_res,
                                edge_index_left,
                                edge_index_right,
                                v_mask,
                                edge_index_all,
                                edge_bias_index,)
            atom_nodes_all_layer.append(nodes)

        return nodes, edges, aa_edges, atom_nodes_all_layer
