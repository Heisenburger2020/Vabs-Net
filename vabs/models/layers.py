import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicore import utils
from unicore.modules import softmax_dropout, SelfMultiheadAttention, LayerNorm
from typing import Tuple, Optional

from unicore.utils import (
    permute_final_dims,
)
import numpy as np
import torch_scatter
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean

from torch import Tensor
from typing import Callable, Optional
from torch.utils.checkpoint import checkpoint
from unicore.utils import checkpoint_sequential
from functools import partial
from torch_sparse import SparseTensor
from typing import List, Optional, Union
from .frame import Frame, Rotation, Quaternion
from torch.nn import MultiheadAttention

class SimpleModuleList(nn.ModuleList):
    def __repr__(self):
        return str(len(self)) + " X ...\n" + self[0].__repr__()

class ResnetBlock(nn.Module):
    def __init__(self, d):
        """
        Args:
            d:
                Hidden channel dimension
        """
        super(ResnetBlock, self).__init__()

        self.linear_1 = Linear(d, d, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(d, d, init="final")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_0 = x

        x = self.act(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)

        return residual(x, x_0, self.training)

class SideChainAngleResnetIteration(nn.Module):
    def __init__(self, d_hid):
        super(SideChainAngleResnetIteration, self).__init__()

        self.d_hid = d_hid

        self.linear_1 = Linear(self.d_hid, self.d_hid, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(self.d_hid, self.d_hid, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:

        x = self.act(s)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)

        return residual(s, x, self.training)

class SidechainAngleResnetv2(nn.Module):
    def __init__(self, d_in, d_hid, num_blocks, num_angles, preln=0):
        super(SidechainAngleResnetv2, self).__init__()

        self.preln = preln
        self.linear_in = Linear(d_in, d_hid)
        self.act = nn.GELU()
        self.linear_initial = Linear(d_in, d_hid)

        self.layers = SimpleModuleList()
        for _ in range(num_blocks):
            self.layers.append(SideChainAngleResnetIteration(d_hid=d_hid))

        self.linear_out = Linear(d_hid, num_angles * 2)
        if self.preln:
            self.layer_norm_init = LayerNorm(d_in)
    def forward(
        self, s: torch.Tensor, initial_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.preln:
            s = self.layer_norm_init(s)

        initial_s = self.linear_initial(self.act(initial_s))
        s = self.linear_in(self.act(s))

        s = s + initial_s

        for layer in self.layers:
            s = layer(s)

        s = self.linear_out(self.act(s))

        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s.float() ** 2, dim=-1, keepdim=True),
                min=1e-12,
            )
        )
        s = s.float() / norm_denom

        return unnormalized_s, s.type(unnormalized_s.dtype)

class SidechainAngleResnet(nn.Module):
    def __init__(self, d_in, d_hid, num_blocks, num_angles, preln=0):
        super().__init__()
        self.preln = preln
        self.linear_in = Linear(d_in, d_hid)
        self.act = nn.GELU()
        self.layers = SimpleModuleList([
            ResnetBlock(d=d_hid) for _ in range(num_blocks)
        ])

        self.linear_out = Linear(d_hid, num_angles * 2)

        if self.preln:
            self.layer_norm_init = LayerNorm(d_in)

    def forward(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.preln:
            s = self.layer_norm_init(s)

        s = self.linear_in(self.act(s))
        for layer in self.layers:
            s = layer(s)
        s = self.linear_out(self.act(s))

        s = s.view(s.shape[:-1] + (-1, 2))
        unnormalized_s = s

        norm_denom = s.float().norm(dim=-1, keepdim=True)
        s = s.float() / norm_denom

        return unnormalized_s, s.type(unnormalized_s.dtype)

class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and self.training:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")


class Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
    ):
        super(Embedding, self).__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self._normal_init()

        if padding_idx is not None:
            self.weight.data[self.padding_idx].zero_()

    def _normal_init(self, std=0.02):
        nn.init.normal_(self.weight, mean=0.0, std=std)


class Transition(nn.Module):
    def __init__(self, d_in, n, dropout=0.0):

        super(Transition, self).__init__()

        self.d_in = d_in
        self.n = n

        self.linear_1 = Linear(self.d_in, self.n * self.d_in, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(self.n * self.d_in, d_in, init="final")
        self.dropout = dropout

    def _transition(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self._transition(x=x)
        return x
class MLPs(nn.Module):
    def __init__(
        self,
        input_dim=768,
        output_dim=768,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.linear_in = Linear(input_dim, input_dim, init='glorot')
        # self.linear_out = Linear(input_dim, output_dim, bias=True, init="glorot")

        self.embed = nn.Sequential(
            Linear(input_dim, output_dim, bias=True, init='glorot'),
            nn.LeakyReLU(),
            LayerNorm(output_dim),
            Linear(output_dim, output_dim, bias=True, init='glorot'),
            nn.LeakyReLU(),
            LayerNorm(output_dim),
            Linear(output_dim, output_dim, bias=True, init='glorot')
        )

    def forward(self, base, input):
        x = torch.cat([base, input], dim=-1)
        x = x.type(self.embed[0].weight.dtype)
        # x = F.gelu(self.linear_in(x))
        # x = self.linear_out(x)
        return self.embed(x)

class Context(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, node_context = False, edge_context = False):
        super(Context, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_context = node_context
        self.edge_context = edge_context

        # self.V_MLP = nn.Sequential(
        #                         nn.Linear(num_hidden, num_hidden),
        #                         nn.ReLU(),
        #                         nn.Linear(num_hidden,num_hidden),
        #                         nn.ReLU(),
        #                         nn.Linear(num_hidden,num_hidden),
        #                         )
        
        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

        # self.E_MLP = nn.Sequential(
        #                         nn.Linear(num_hidden, num_hidden),
        #                         nn.ReLU(),
        #                         nn.Linear(num_hidden,num_hidden),
        #                         nn.ReLU(),
        #                         nn.Linear(num_hidden,num_hidden)
        #                         )
        
        self.E_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_in),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        if self.node_context:
            c_V = scatter_mean(h_V, batch_id, dim=0)
            h_V = h_V * self.V_MLP_g(c_V[batch_id])
            # h_V = h_V + h_V * self.V_MLP_g(c_V[batch_id])
            # h_V = self.V_MLP(h_V) * self.V_MLP_g(c_V[batch_id])
            # h_V = h_V + self.V_MLP(h_V) * self.V_MLP_g(c_V[batch_id])
        
        if self.edge_context:
            c_V = scatter_mean(h_V, batch_id, dim=0)
            h_E = h_E * self.E_MLP_g(c_V[batch_id[edge_idx[:, 0]]])

        return h_V, h_E


class PointTransformerLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, share_planes=8):
        super().__init__()
        out_planes = node_dim
        in_planes = node_dim
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(edge_dim, 1))
        self.linear_w = nn.Sequential(nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                edges: torch.Tensor,
                edge_index: torch.Tensor,) -> torch.Tensor:
        x_q, x_k, x_v = self.linear_q(q), self.linear_k(k), self.linear_v(v)  # (n, c)

        x_q = x_q[edge_index[:, 0]]
        x_k = x_k[edge_index[:, 1]]
        x_v = x_v[edge_index[:, 1]]
        p_r = edges

        for i, layer in enumerate(self.linear_p): 
            p_r = layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): 
            w = layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, c = x_v.shape; s = self.share_planes
        x = torch_scatter.scatter(((x_v + p_r).view(n, s, c // s) * w.unsqueeze(1)).view(n, c), edge_index[:, 0], dim=1, dim_size=q.shape[0], reduce="sum").view(n, c)
        return x

class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0.0, output_mlp=True):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        self.output_mlp = output_mlp
        
        self.W_V = nn.Sequential(nn.Linear(num_in, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, num_hidden)
        )
        self.Bias = nn.Sequential(
                                nn.Linear(num_hidden + num_in, num_hidden),
                                nn.ReLU(),
                                # nn.Linear(num_hidden,num_hidden),
                                # nn.ReLU(),
                                nn.Linear(num_hidden,num_heads)
                                )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, h_E, center_id, batch_id, dst_idx=None):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        
        w = self.Bias(torch.cat([h_V[center_id], h_E],dim=-1)).view(E, n_heads, 1) 
        attend_logits = w/np.sqrt(d) 

        V = self.W_V(h_E).view(-1, n_heads, d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([-1, self.num_hidden])

        if self.output_mlp:
            h_V_update = self.W_O(h_V)
        else:
            h_V_update = h_V
        return h_V_update


# RoPe

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    batch_index: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    bsz = int((batch_index[-1].data + 1).cpu().data)
    xq_out = []
    xk_out = []
    for i in range(bsz):

        # xq.shape = [batch_size, seq_len, dim]
        # xq_.shape = [batch_size, seq_len, dim // 2, 2]
        xq_ = xq[batch_index == i]
        xk_ = xk[batch_index == i]

        xq_ = xq_.float().reshape(*xq_.shape[:-1], -1, 2)
        xk_ = xk_.float().reshape(*xk_.shape[:-1], -1, 2)

        # 转为复数域
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)

        # 应用旋转操作，然后将结果转回实数域
        # xq_out.shape = [batch_size, seq_len, dim]
        xq_out.append(torch.view_as_real(xq_ * freqs_cis[:xq_.shape[0]].unsqueeze(1)).flatten(2))
        xk_out.append(torch.view_as_real(xk_ * freqs_cis[:xq_.shape[0]].unsqueeze(1)).flatten(2))
    xq_out = torch.cat(xq_out, dim=0)
    xk_out = torch.cat(xk_out, dim=0)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        pair_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = False,
        dropout: float = 0.0,
        preln: bool = False,
        use_rope: bool=False,
        max_seq_len: int = 1200,
        sequence_only: bool = False,
    ):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.sequence_only = sequence_only
        self.gating = gating
        self.linear_q = Linear(q_dim, total_dim, bias=False, init="glorot")
        self.linear_k = Linear(k_dim, total_dim, bias=False, init="glorot")
        self.linear_v = Linear(v_dim, total_dim, bias=False, init="glorot")
        self.linear_o = Linear(total_dim, q_dim, init="final")
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(q_dim, total_dim, init="gating")
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5
        self.dropout = dropout
        if not self.sequence_only:
            self.linear_bias = Linear(pair_dim, num_heads)
        self.preln = preln
        if self.preln:
            self.layer_norm = LayerNorm(pair_dim)
        self.dropout_module = Dropout(dropout)

        self.use_rope = use_rope
        if use_rope:
            self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2)


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor=None,
    ) -> torch.Tensor:
        g = None
        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)

        q = self.linear_q(q).view(q.shape[:-1] + (self.num_heads, -1))
        k = self.linear_k(k).view(k.shape[:-1] + (self.num_heads, -1))
        v = self.linear_v(v).view(v.shape[:-1] + (self.num_heads, -1))
        q *= self.norm
        if self.use_rope:
            q, k = apply_rotary_emb(q, k, batch_index, freqs_cis=self.freqs_cis.to(q.device))
        n_node = q.size(0)

        q = q[edge_index[:, 0]].transpose(0, 1).contiguous()
        k = k[edge_index[:, 1]].transpose(0, 1).contiguous()
        v = v[edge_index[:, 1]].transpose(0, 1).contiguous()

        attn = (q * k).view(self.num_heads, edge_index.shape[0], -1).sum(dim=-1)
        # attn = torch.matmul(q, k.transpose(-1, -2))
        del q, k
        # bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()
        # attn = softmax_dropout(attn, self.dropout, self.training, mask=mask, bias=bias)
        # o = torch.matmul(attn, v)
        if not self.sequence_only:
            bias = self.linear_bias(edges).permute(1, 0).contiguous()
            attn += bias
        attn = torch_scatter.composite.scatter_softmax(attn, edge_index[:, 0].unsqueeze(0))
        # print('???', attn.shape, edges.shape, v.shape)
        attn = self.dropout_module(attn)
        o = torch_scatter.scatter(attn.unsqueeze(-1) * v, edge_index[:, 0], dim=1, dim_size=n_node, reduce="sum")

        del attn, v

        o = o.transpose(-2, -3).contiguous()
        o = o.view(*o.shape[:-2], -1)

        if g is not None:
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o

class EdgeAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = False,
        dropout: float = 0.0,
        preln: bool = False,
        max_seq_len: int = 1200
    ):
        super(EdgeAttention, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = Linear(q_dim, total_dim, bias=False, init="glorot")
        self.linear_k = Linear(k_dim, total_dim, bias=False, init="glorot")
        self.linear_v = Linear(v_dim, total_dim, bias=False, init="glorot")
        self.linear_o = Linear(total_dim, q_dim, init="final")
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(q_dim, total_dim, init="gating")
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5
        self.dropout = dropout
        self.preln = preln

        self.dropout_module = Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = edge_index.cpu()
        edge_index_np = edge_index.numpy()
        edge_edge_index_np = []
        for i in range(len(edge_index_np)):
            edge_j_np = np.argwhere((edge_index_np[:, 0] == edge_index_np[i, 0]) + (edge_index_np[:, 0] == edge_index_np[i, 1]))
            edge_i_np = np.ones((len(edge_j_np), 1)) * i
            edge_edge_index_np.append(np.concatenate([edge_i_np, edge_j_np], axis=1))

        edge_edge_index_np = np.concatenate(edge_edge_index_np, axis=0)
        edge_edge_index = torch.Tensor(edge_edge_index_np)
        # edge_edge_index = []
        # for i in range(len(edge_index)):
        #     edge_j = torch.nonzero((edge_index[:, 0] == edge_index[i, 0]) + (edge_index[:, 0] == edge_index[i, 1]))
        #     edge_i = torch.ones((len(edge_j), 1)) * i
        #     edge_edge_index.append(torch.cat([edge_i, edge_j], dim=1))
        # edge_edge_index = torch.cat(edge_edge_index, dim=0)
        edge_index = edge_edge_index.long().to(q.device)
        g = None
        if self.linear_g is not None:
            g = self.linear_g(q)

        q = self.linear_q(q).view(q.shape[:-1] + (self.num_heads, -1))
        k = self.linear_k(k).view(k.shape[:-1] + (self.num_heads, -1))
        v = self.linear_v(v).view(v.shape[:-1] + (self.num_heads, -1))
        q *= self.norm
        n_node = q.size(0)

        q = q[edge_index[:, 0]].transpose(0, 1).contiguous()
        k = k[edge_index[:, 1]].transpose(0, 1).contiguous()
        v = v[edge_index[:, 1]].transpose(0, 1)
        attn = (q * k).view(self.num_heads, edge_index.shape[0], -1).sum(dim=-1)
        del q, k
        attn = torch_scatter.composite.scatter_softmax(attn, edge_index[:, 0].unsqueeze(0))
        attn = self.dropout_module(attn)
        o = torch_scatter.scatter(attn.unsqueeze(-1) * v, edge_index[:, 0], dim=1, dim_size=n_node, reduce="sum")

        del attn, v

        o = o.transpose(-2, -3).contiguous()
        o = o.view(*o.shape[:-2], -1)

        if g is not None:
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, learned=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(self, x):
        seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return self.weights.index_select(0, positions.view(-1)).view(seq_len, -1).detach()

    def make_positions(self, x):
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(self, num_embeddings):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb

class OuterProduct(nn.Module):
    def __init__(self, d_atom, d_pair, d_hid=32):
        super(OuterProduct, self).__init__()

        self.d_atom = d_atom
        self.d_pair = d_pair
        self.d_hid = d_hid

        self.linear_in = nn.Linear(d_atom, d_hid*2)
        self.linear_out = nn.Linear(d_hid**2, d_pair)
        self.act = nn.GELU()

    def _opm(self, a, b, edge_index):
        # n, d = a.shape
        # outer = torch.einsum("...bc,...de->...bdce", a, b)
        # a = a.view(bsz, n, 1, d, 1)
        # b = b.view(bsz, 1, n, 1, d)

        a = a[edge_index[:, 0],:].unsqueeze(-1)
        b = b[edge_index[:, 1],:].unsqueeze(-2)

        outer = a * b
        # print('???', outer.shape, a.shape, b.shape)
        outer = outer.view(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        return outer

    def forward(
        self,
        m: torch.Tensor,
        edge_index: torch.Tensor,
        # op_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    
        n_node = m.size(0)
        ab = self.linear_in(m)
        a, b = ab.chunk(2, dim=-1)
        z = self._opm(a, b, edge_index)
        return z


class SE3InvariantKernel(nn.Module):
    """
    Compute 3D attention bias according to the position information for each head.
    """

    def __init__(
        self,
        pair_dim,
        num_pair,
        num_kernel,
        std_width=1.0,
        start=0.0,
        stop=9.0,
    ):
        super(SE3InvariantKernel, self).__init__()
        self.num_kernel = num_kernel
        self.pair_dim = pair_dim
        self.gaussian = GaussianKernel(
            self.num_kernel,
            num_pair,
            std_width=std_width,
            start=start,
            stop=stop,
        )
        self.out_proj = NonLinear(self.num_kernel, pair_dim)

    def forward(self, dist, node_type_edge):
        edge_feature = self.gaussian(
            dist,
            node_type_edge.long(),
        ).to(dist.dtype)
        edge_feature = self.out_proj(edge_feature)
        return edge_feature

@torch.jit.script
def gaussian(x, mean, std:float):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianKernel(nn.Module):
    def __init__(self, K=128, num_pair=512, std_width=1.0, start=0.0, stop=9.0):
        super().__init__()
        self.K = K
        std_width = std_width
        start = start
        stop = stop
        mean = torch.linspace(start, stop, K)
        self.std = (std_width * (mean[1] - mean[0])).item()
        self.register_buffer("mean", mean)
        self.mul = Embedding(num_pair, 1, padding_idx=0)
        self.bias = Embedding(num_pair, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1.0)

    def forward(self, x, atom_pair):
        mul = self.mul(atom_pair).abs().sum(dim=-2)
        bias = self.bias(atom_pair).sum(dim=-2)
        # assert 1==0, (atom_pair.shape, mul.shape, bias.shape)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, self.K)
        mean = self.mean.float().view(-1)
        return gaussian(x.float(), mean, self.std)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = Linear(input, hidden, init="relu")
        self.layer2 = Linear(hidden, output_size, init="final")

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

    def zero_init(self):
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)


class EnergyHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()
        self.layer_norm = LayerNorm(input_dim)
        self.linear_in = Linear(input_dim, input_dim, init="relu")

        self.linear_out = Linear(input_dim, output_dim, bias=True, init="final")

    def forward(self, x):
        x = x.type(self.linear_in.weight.dtype)
        x = F.gelu(self.layer_norm(self.linear_in(x)))
        x = self.linear_out(x)
        return x


class CLSHead(nn.Module):
    def __init__(
        self,
        input_dim=768,
        output_cls=2,
        preln: bool=False,
        gating: bool=False,
    ):
        super().__init__()
        self.gating = gating
        self.layer_norm = LayerNorm(input_dim)
        self.linear_in = Linear(input_dim, input_dim, init='relu')
        self.linear_out = Linear(input_dim, output_cls, bias=True, init="final")
        # self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.preln = preln
        if self.preln:
            self.layer_norm_init = LayerNorm(input_dim)
        if self.gating:
            self.sigmoid_z = nn.Sigmoid()
            self.gating = Linear(input_dim, input_dim, init="gating")


    def forward(self, x):

        x = x.type(self.linear_in.weight.dtype)
        if self.preln:
            x = self.layer_norm_init(x)
        if self.gating:
            g = self.gating(x)
            
        x = F.gelu(self.layer_norm(self.linear_in(x)))
        # x = self.logsoftmax(self.linear_out(x))
        if self.gating:
            x = x * g
        x = self.linear_out(x)
        return x


class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class QueryAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = False,
        dropout: float = 0.0,
        preln: bool = False,
    ):
        super(QueryAttention, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = Linear(q_dim, total_dim, bias=False, init="glorot")
        self.linear_k = Linear(k_dim, total_dim, bias=False, init="glorot")
        self.linear_v = Linear(v_dim, total_dim, bias=False, init="glorot")
        self.linear_o = Linear(total_dim, q_dim, init="final")
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(q_dim, total_dim, init="gating")
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5
        self.dropout = dropout
        # self.linear_bias = Linear(pair_dim, num_heads)
        self.preln = preln

        self.dropout_module = Dropout(dropout)

        # self.use_rope = use_rope
        # if use_rope:
        #     self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2)


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        # edges: torch.Tensor,
        edge_index: torch.Tensor,
        # batch_index: torch.Tensor=None,
    ) -> torch.Tensor:
        g = None
        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)

        q = self.linear_q(q).view(q.shape[:-1] + (self.num_heads, -1))
        k = self.linear_k(k).view(k.shape[:-1] + (self.num_heads, -1))
        v = self.linear_v(v).view(v.shape[:-1] + (self.num_heads, -1))
        q *= self.norm
        # if self.use_rope:
        #     q, k = apply_rotary_emb(q, k, batch_index, freqs_cis=self.freqs_cis.to(q.device))
        n_node = q.size(0)

        q = q[edge_index[:, 0]].transpose(0, 1).contiguous()
        k = k[edge_index[:, 1]].transpose(0, 1).contiguous()
        v = v[edge_index[:, 1]].transpose(0, 1)
        # bias = self.linear_bias(edges).permute(1, 0).contiguous()
        attn = (q * k).view(self.num_heads, edge_index.shape[0], -1).sum(dim=-1)
        # attn = torch.matmul(q, k.transpose(-1, -2))
        del q, k
        # bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()
        # attn = softmax_dropout(attn, self.dropout, self.training, mask=mask, bias=bias)
        # o = torch.matmul(attn, v)
        # attn += bias
        attn = torch_scatter.composite.scatter_softmax(attn, edge_index[:, 0].unsqueeze(0))
        # print('???', attn.shape, edges.shape, v.shape)
        attn = self.dropout_module(attn)
        o = torch_scatter.scatter(attn.unsqueeze(-1) * v, edge_index[:, 0], dim=1, dim_size=n_node, reduce="sum")

        del attn, v

        o = o.transpose(-2, -3).contiguous()
        o = o.view(*o.shape[:-2], -1)

        if g is not None:
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o


import torch.nn.utils.rnn as rnn_utils
def pad_1d(samples, add=0, multiplier=4, pad=0):
    max_len = max(x.size(0) for x in samples)
    max_len = (max_len + multiplier) // multiplier * multiplier - 1
    
    out = []
    mask = []
    lens = []
    
    for sample in samples:
        x_len = sample.size(0)
        padded_sample = torch.cat([sample, torch.zeros((max_len - x_len, sample.shape[1]), dtype=sample.dtype).to(sample.device) + add])
        out.append(padded_sample)
        lens.append(x_len)
        
        # 创建掩码
        sample_mask = torch.tensor([False] * x_len + [True] * (max_len - x_len))
        mask.append(sample_mask)
    
    out = torch.stack(out)
    lens = torch.tensor(lens)
    mask = torch.stack(mask).to(sample.device)
    packed_out = out
    # packed_out = rnn_utils.pack_padded_sequence(out, lens, enforce_sorted=False, batch_first=True)
    
    return packed_out, mask

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=786, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None
        self.omit_selfattn = False

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = tgt

        if not self.omit_selfattn:
            tgt2, sim_mat_1 = self.self_attn(query=q, key=k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2, sim_mat_2 = self.multihead_attn(query=tgt,
                                key=memory,
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class ECGOCLSHead(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=768,
        output_cls=576,
        preln: bool=False,
        gating: bool=False,
        num_decoder_layers: int=1,
    ):
        super().__init__()
        self.gating = gating
        self.hidden_dim = hidden_dim
        self.layer_norm = LayerNorm(input_dim)
        self.linear_in = Linear(input_dim, input_dim, init='relu')
        self.linear_out = Linear(input_dim, hidden_dim, bias=True, init="relu")
        # self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.preln = preln
        if self.preln:
            self.layer_norm_init = LayerNorm(input_dim)
        if self.gating:
            self.sigmoid_z = nn.Sigmoid()
            self.gating = Linear(input_dim, input_dim, init="gating")

        self.query_embed = nn.Embedding(output_cls, hidden_dim)
        self.fc = GroupWiseLinear(output_cls, hidden_dim, bias=True)
        self.num_attention_heads = 48
        head_dim = self.hidden_dim // self.num_attention_heads

        self.num_decoder_layers = num_decoder_layers
        self.self_attn_layers = nn.ModuleList([])
        self.self_attn_layers.extend(
            [
                TransformerDecoderLayer(d_model=hidden_dim, nhead=48, dim_feedforward=hidden_dim)
                for i in range(num_decoder_layers)
            ]
        )



    def forward(self, x, batch_index):
        bsz = int((batch_index[-1].data + 1).cpu().data)

        x = x.type(self.linear_in.weight.dtype)
        if self.preln:
            x = self.layer_norm_init(x)
        if self.gating:
            g = self.gating(x)
            
        x = F.gelu(self.layer_norm(self.linear_in(x)))
        # x = self.logsoftmax(self.linear_out(x))
        if self.gating:
            x = x * g
        x = self.linear_out(x) # [bsz, 1, dim]
        samples = [x[batch_index==i] for i in range(bsz)]
        packed_out, mask = pad_1d(samples)
        query_input = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1) # [bsz, cls, dim]

        for i in range(self.num_decoder_layers):
            residual = query_input
            query_input = self.self_attn_layers[i](tgt=query_input, memory=packed_out, memory_key_padding_mask=mask)
            query_input = residual + query_input
        out = self.fc(query_input)

        return out

class MovementPredictionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        pair_dim: int,
        num_head: int,
        preln: bool = True,
    ):
        super().__init__()
        self.preln = preln
        
        if self.preln:
            self.pair_norm = LayerNorm(pair_dim)
            self.layer_norm = LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.q_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.k_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.v_proj = Linear(embed_dim, embed_dim, bias=False, init="glorot")
        self.num_head = num_head
        self.scaling = (embed_dim // num_head) ** -0.5
        self.force_proj1 = Linear(embed_dim, 1, init="final")
        self.force_proj2 = Linear(embed_dim, 1, init="final")
        self.force_proj3 = Linear(embed_dim, 1, init="final")
        self.linear_bias = Linear(pair_dim, num_head)
        self.dropout = 0.1

    def zero_init(self):
        nn.init.zeros_(self.force_proj1.weight)
        nn.init.zeros_(self.force_proj1.bias)
        nn.init.zeros_(self.force_proj2.weight)
        nn.init.zeros_(self.force_proj2.bias)
        nn.init.zeros_(self.force_proj3.weight)
        nn.init.zeros_(self.force_proj3.bias)

    def forward(
        self,
        query: Tensor,
        edge_index: Tensor,
        edge_diff: Tensor,
        pair: Tensor,

    ) -> Tensor:
        n_node, _ = query.size()
        if self.preln:
            query = self.layer_norm(query)
            pair = self.pair_norm(pair)
        # print(query.shape, edge_index.shape, edge_diff.shape, pair.shape, self.q_proj(query).shape) 
        # torch.Size([1314, 128]) torch.Size([148868, 2]) torch.Size([148868, 3]) torch.Size([148868, 128])
        q = (
            self.q_proj(query).view(n_node, self.num_head, -1)[edge_index[:, 0]].transpose(0, 1)
            * self.scaling
        ) # shape '[1314, 48, -1]' is invalid for input of size 168192
        k = self.k_proj(query).view(n_node, self.num_head, -1)[edge_index[:, 1]].transpose(0, 1)
        v = self.v_proj(query).view(n_node, self.num_head, -1)[edge_index[:, 1]].transpose(0, 1) # [head, n, d]
        bias = self.linear_bias(pair).permute(1, 0).contiguous()
        attn = (q * k).view(self.num_head, edge_index.shape[0], -1).sum(dim=-1)  # [head, n, n]
        
        attn += bias # [head, nedge]
        attn_probs = torch_scatter.composite.scatter_softmax(attn, edge_index[:, 0].unsqueeze(0)) # [head, nedge]
        rot_attn_probs = attn_probs.unsqueeze(-1) * edge_diff.unsqueeze(0).type_as(
            attn_probs
        )  # [head, nedge, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 2, 1) # [head, 3, nedge]
        x = torch_scatter.scatter(rot_attn_probs.unsqueeze(-1) * v.unsqueeze(1), edge_index[:, 0], dim=2, dim_size=n_node, reduce="sum")
        # [head, 3, nnode, d] torch.Size([32, 3, 1314, 4])
        x = x.permute(2, 1, 0, 3).reshape(n_node, 3, -1) 
        f1 = self.force_proj1(x[:, 0, :]).view(n_node, 1)
        f2 = self.force_proj2(x[:, 1, :]).view(n_node, 1)
        f3 = self.force_proj3(x[:, 2, :]).view(n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force

class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f"prob={self.drop_prob}"


class TriangleAttention(nn.Module):
    def __init__(self, d_pair, head_dim: int, num_heads: int, use_qk=False):
        super(TriangleAttention, self).__init__()

        self.use_qk = use_qk
        d_hid = head_dim * num_heads
        self.num_head = num_heads
        
        self.linear_q = Linear(d_pair, d_hid, bias=False, init="glorot")
        if not use_qk:
            self.linear_k = Linear(d_pair, d_hid, bias=False, init="glorot")
        self.linear_v = Linear(d_pair, d_hid, bias=False, init="glorot")
        self.linear_o = Linear(d_hid, d_pair, init="final")
        self.norm = head_dim**-0.5
        # self.dropout = dropout
        self.linear_g = Linear(d_pair, d_hid, init="gating")

    def forward(
        self,
        z: torch.Tensor,
        edge_index_left: torch.Tensor,
        edge_index_right: torch.Tensor,
        # mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # based on triangle attention
        n_edge = z.size(0)
        q = self.linear_q(z)
        q *= self.norm
        if not self.use_qk:
            k = self.linear_k(z)
        else:
            q, k = torch.chunk(q, 2, dim=-1)
        v = self.linear_v(z)

        g = torch.sigmoid(self.linear_g(z))
        
        if edge_index_left is not None:
            idx_s = edge_index_left[...,1]
            idx_t = edge_index_left[...,0]
        else:
            idx_s = edge_index_right[...,0]
            idx_t = edge_index_right[...,1]

        qkv_edge_col = idx_s
        qkv_edge_row = idx_t

        n_qkv_edge = qkv_edge_col.size(0)
        n_kv_edge = qkv_edge_col.size(-1)

        q = q[qkv_edge_row].view(n_qkv_edge, n_kv_edge, self.num_head, -1).permute(2, 0, 1, 3)
        k = k[qkv_edge_col].view(n_qkv_edge, n_kv_edge, self.num_head, -1).permute(2, 0, 1, 3)
       
        attn = (q * k).view(self.num_head, n_qkv_edge, n_kv_edge, -1).sum(dim=-1)
        del q, k

        v = v[qkv_edge_col].view(n_qkv_edge, n_kv_edge, self.num_head, -1).permute(2, 0, 1, 3)
        o = (attn.unsqueeze(-1) * v).sum(dim=-2)

        del attn, v

        o = o.permute(1, 0, 2).contiguous()
        o = o.view(*o.shape[:-2], -1)

        o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o


class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hid):
        super(TriangleMultiplication, self).__init__()

        self.linear_ab_p = Linear(d_pair, d_hid * 2)
        self.linear_ab_g = Linear(d_pair, d_hid * 2, init="gating")

        self.linear_g = Linear(d_pair, d_pair, init="gating")
        self.linear_z = Linear(d_hid, d_pair, init="final")

        self.layer_norm_out = LayerNorm(d_hid)

    def forward(
        self,
        z: torch.Tensor,
        edge_index_left: torch.Tensor,
        edge_index_right: torch.Tensor,
        # mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        
        # based on triangle update
        g = torch.sigmoid(self.linear_g(z))
        if self.training:
            # ab = self.linear_ab_p(z) * mask * torch.sigmoid(self.linear_ab_g(z))
            ab = self.linear_ab_p(z) * torch.sigmoid(self.linear_ab_g(z))
        else:
            ab = self.linear_ab_p(z)
            # ab *= mask
            ab *= torch.sigmoid(self.linear_ab_g(z))
        a, b = torch.chunk(ab, 2, dim=-1)
        del z, ab
        # [n_edge, n_k, 2]
        x = a[edge_index_left[:, :, 0], :] * b[edge_index_left[:, :, 1], :] + a[edge_index_right[:, :, 0], :] * b[edge_index_right[:, :, 1], :]
        x = torch.mean(x, dim=-2)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        return g * x


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1):
        super(EdgeMLP, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx):
        src_idx = edge_idx[:, 0]
        dst_idx = edge_idx[:, 1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = h_E + self.dropout(h_message)
        return h_E
       
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        node_dim=128,
        edge_dim=128,
        pair_hidden_dim: int = 32,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        droppath_prob: float = 0.0,
        pair_dropout: float = 0.25,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        use_outer=False,
        use_trimul=False,
        use_pairupdate=False,
        use_triangleattn=False,
        cross_layer=True,
        is_end=False,
        preln=False,
        concat_style=False,
        use_context=False,
        residue_only=False,
        v_attn=False,
        use_edge_attn=False,
        sequence_only=False,
        use_rope=False
    ):
        super().__init__()
        self.embedding_dim = node_dim
        self.pair_dim = edge_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout

        self.use_outer = use_outer
        self.use_trimul = use_trimul
        self.use_pairupdate = use_pairupdate
        self.use_triangleattn = use_triangleattn
        self.is_end = is_end
        self.preln = preln
        self.qkv = True
        self.use_context = use_context
        self.concat_style = concat_style
        self.residue_only=residue_only
        self.virtual_attn = v_attn
        self.use_edge_attn = use_edge_attn
        self.sequence_only = sequence_only

        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = Dropout(dropout)

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        head_dim = self.embedding_dim // self.num_attention_heads

        self.self_attn = Attention(
            self.embedding_dim,
            self.embedding_dim,
            self.embedding_dim,
            pair_dim=edge_dim,
            head_dim=head_dim,
            num_heads=self.num_attention_heads,
            gating=False,
            dropout=attention_dropout,
            use_rope=use_rope,
            sequence_only=self.sequence_only
        )

        if self.use_edge_attn and not is_end:
            edge_head_dim = edge_dim // 16
            self.edge_attn = EdgeAttention(
                edge_dim,
                edge_dim,
                edge_dim,
                head_dim=edge_head_dim,
                num_heads=self.num_attention_heads,
                gating=False,
                dropout=attention_dropout,
            )
            self.edge_attn_norm = LayerNorm(self.pair_dim)

        if self.virtual_attn:
            self.v_attn = Attention(
                self.embedding_dim,
                self.embedding_dim,
                self.embedding_dim,
                pair_dim=edge_dim,
                head_dim=head_dim,
                num_heads=self.num_attention_heads,
                gating=False,
                dropout=attention_dropout,
                use_rope=False,
                sequence_only=self.sequence_only
            )
            self.v_attn_layer_norm = LayerNorm(self.embedding_dim)

            self.v_ffn = Transition(
                self.embedding_dim,
                ffn_embedding_dim // self.embedding_dim,
                dropout=activation_dropout,
            )

        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.ffn = Transition(
            self.embedding_dim,
            ffn_embedding_dim // self.embedding_dim,
            dropout=activation_dropout,
        )
        self.final_layer_norm = LayerNorm(self.embedding_dim)

        if not self.residue_only:
            self.aa_ffn = Transition(
                self.embedding_dim,
                ffn_embedding_dim // self.embedding_dim,
                dropout=activation_dropout,
            )

            self.aa_final_layer_norm = LayerNorm(self.embedding_dim)

        if self.use_trimul and not self.is_end:
            self.pair_layer_norm_trimul = LayerNorm(self.pair_dim)
            self.pair_tri_mul = TriangleMultiplication(self.pair_dim, pair_hidden_dim)
            # self.left_tri_attn = TriangleAttention(self.pair_dim, pair_hidden_dim, self.num_attention_heads)
            # self.pair_layer_norm_tri_left_attn = LayerNorm(self.pair_dim)
            # self.right_tri_attn = TriangleAttention(self.pair_dim, pair_hidden_dim, self.num_attention_heads)
            # self.pair_layer_norm_tri_right_attn = LayerNorm(self.pair_dim)

        if use_outer and not is_end:
            self.opm = OuterProduct(self.embedding_dim, self.pair_dim, d_hid=pair_hidden_dim)
            self.pair_layer_norm_opm = LayerNorm(self.pair_dim)
            # self.opm_aa = OuterProduct(self.embedding_dim, self.pair_dim, d_hid=pair_hidden_dim)
            # if not self.preln:
            #     self.pair_layer_norm_opm_aa = LayerNorm(self.pair_dim)
            # else:
            #     self.pair_layer_norm_opm_aa = LayerNorm(self.embedding_dim)


        if use_pairupdate and not is_end:
            self.pair_ffn = Transition(
                self.pair_dim,
                1,
                dropout=activation_dropout,
            )
            self.pair_layer_norm_ffn = LayerNorm(self.pair_dim)
            # self.pair_ffn_aa = Transition(
            #     self.pair_dim,
            #     1,
            #     dropout=activation_dropout,
            # )
            # self.pair_layer_norm_ffn_aa = LayerNorm(self.pair_dim)
            self.pair_dropout = pair_dropout

        if not residue_only:
            if not concat_style:
                self.aa_attn = Attention(
                    self.embedding_dim,
                    self.embedding_dim,
                    self.embedding_dim,
                    pair_dim=edge_dim,
                    head_dim=head_dim,
                    num_heads=self.num_attention_heads,
                    gating=False,
                    dropout=attention_dropout,

                )
            else:
                # self.aa_attn = GVPConvLayer(
                #     (self.embedding_dim, self.embedding_dim // 4),
                #     (edge_dim, 1),
                #     vector_gate=True,
                #     attention_heads=0,
                #     n_message=3,
                #     conv_activations=(F.relu, torch.sigmoid),
                #     n_edge_gvps=0,
                #     eps=1e-4,
                #     layernorm=True,
                # )
                self.aa_attn = NeighborAttention(self.embedding_dim, self.embedding_dim + edge_dim, num_heads=self.num_attention_heads) 
            self.aa_attn_layer_norm = LayerNorm(self.embedding_dim)


        if self.use_context and not self.is_end:
            assert 0
            self.context = Context(self.embedding_dim, edge_dim, num_heads=None, node_context=True, edge_context=True)



    def shared_dropout(self, x, n_node, edge_index, dropout, reverse=False):
        # shape = list(x.shape)
        # shape[shared_dim] = 1
        # with torch.no_grad():
        #     mask = x.new_ones(shape)

        with torch.no_grad():
            mask = torch.ones(n_node, x.size(-1), device=x.device, dtype=x.dtype)
        mask = F.dropout(mask, p=dropout, training=self.training)
        if reverse:
            mask = mask[edge_index[:,0], :]
        else:
            mask = mask[edge_index[:,1], :]
        return  mask * x
    
    def update_pair(self, x, edges, edge_index):
        edges = edges + self.shared_dropout(
            self.pair_tri_mul(x.size(0), edges, edge_index), x.size(0), edge_index, self.pair_dropout
        )
        edges = edges + self.shared_dropout(
            self.pair_tri_mul_reverse(x.size(0), edges, edge_index), x.size(0), edge_index, self.pair_dropout, reverse=True
        )
        edges = self.pair_layer_norm_trimul(edges)
        return edges

    def forward(self, 
                atom_nodes, 
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
                ):
        
        if not self.residue_only:
            # a->a
            residual = atom_nodes
            if self.preln:
                atom_nodes = self.aa_attn_layer_norm(atom_nodes)

            if not self.concat_style:
                atom_nodes = self.aa_attn(
                    atom_nodes,
                    atom_nodes,
                    atom_nodes,
                    edges=aa_edges,
                    edge_index=aa_edge_index,
                ) 
            else:
                # edge_index_ = aa_edge_index.transpose(0, 1)
                # atom_nodes, aa_edges= self.self_attn(atom_nodes, edge_index_, aa_edges)
                dst_idx = aa_edge_index[:, 0]
                src_idx = aa_edge_index[:, 1]
                atom_nodes = self.aa_attn(atom_nodes, torch.cat([aa_edges, atom_nodes[dst_idx]], dim=-1), src_idx, None, dst_idx)



            atom_nodes = self.dropout_module(atom_nodes)
            atom_nodes = residual + atom_nodes
            if not self.preln:
                atom_nodes = self.aa_attn_layer_norm(atom_nodes)

            residual = atom_nodes
            if self.preln:
                atom_nodes = self.aa_final_layer_norm(atom_nodes)

            atom_nodes = self.aa_ffn(atom_nodes)
            atom_nodes = self.dropout_module(atom_nodes)

            atom_nodes = residual + atom_nodes
            if not self.preln:
                atom_nodes = self.aa_final_layer_norm(atom_nodes)

        
        # r->r
        x = atom_nodes[res_mask]
        residual = x
        if self.preln:
            x = self.self_attn_layer_norm(x)
        
        x = self.self_attn(
            x,
            x,
            x,
            edges=edges,    
            edge_index=edge_index,
            batch_index=batch_index_res
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.preln:
            x = self.self_attn_layer_norm(x)
    
        residual = x
        if self.preln:
            x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.preln:
            x = self.final_layer_norm(x)

        if self.virtual_attn:
            residual = x
            if self.preln:
                x = self.v_attn_layer_norm(x)
            # 单向边
            x = self.v_attn(
                x,
                x,
                x,
                edges=edges[v_mask],
                edge_index=edge_index[v_mask],
                batch_index=batch_index_res,
            )
            x = self.dropout_module(x)
            x = residual + x
            residual = x
            x = self.v_ffn(x)
            x = self.dropout_module(x)
            x = residual + x
            if not self.preln:
                x = self.v_attn_layer_norm(x)


        if self.use_context and not self.is_end:
            atom_nodes[res_mask], edges = self.context(x, edges, edge_index, batch_index_res.long())
        else:
            atom_nodes[res_mask] = x

        if self.use_trimul and not self.is_end:
            residual = edges
            if self.preln:
                edges = self.pair_layer_norm_trimul(edges)
            edges = residual + self.pair_tri_mul(edges, edge_index_left, edge_index_right)
            if not self.preln:
                edges = self.pair_layer_norm_trimul(edges)

            # residual = aa_edges
            # if self.preln:
            #     aa_edges = self.pair_layer_norm_trimul(aa_edges)
            # aa_edges = residual + self.pair_tri_mul(aa_edges, edge_index_left, edge_index_right)
            # if not self.preln:
            #     aa_edges = self.pair_layer_norm_trimul(aa_edges)

            # block_mul = [
            #     self.pair_tri_mul
            # ]
            # residual = aa_edges
            # aa_edges = checkpoint_sequential(
            #     block_mul,
            #     input=(aa_edges, edge_index_left, edge_index_right),
            # )
            # aa_edges = residual + aa_edges[0]
            # aa_edges = self.pair_layer_norm_trimul(aa_edges)

            # aa_edges = aa_edges + self.left_tri_attn(aa_edges, edge_index_left, None)
            # aa_edges = self.pair_layer_norm_tri_left_attn(aa_edges)
            # aa_edges = aa_edges + self.right_tri_attn(aa_edges, None, edge_index_right)
            # aa_edges = self.pair_layer_norm_tri_right_attn(aa_edges)

        if self.use_outer and not self.is_end:
            residual = edges
            if self.preln:
                edges = self.pair_layer_norm_opm(edges)
            edges = residual + self.dropout_module(self.opm(x, edge_index))
            if not self.preln:
                edges = self.pair_layer_norm_opm(edges)
            # residual = aa_edges
            # if self.preln:
            #     aa_edges = residual + self.dropout_module(self.opm_aa(self.pair_layer_norm_opm_aa(atom_nodes), aa_edge_index))
            # if not self.preln:
            #     aa_edges = residual + self.dropout_module(self.opm_aa(atom_nodes, aa_edge_index))
            #     aa_edges = self.pair_layer_norm_opm_aa(aa_edges)

        if self.use_pairupdate and not self.is_end:
            residual = edges
            if self.use_edge_attn:
                edges = self.edge_attn(
                    edges,
                    edges,
                    edges,
                    edge_index=edge_index,
                )
                edges = edges + residual
                edges = self.edge_attn_norm(edges)


            residual = edges
            if self.preln:
                edges = self.pair_layer_norm_ffn(edges)
            edges = residual + self.dropout_module(self.pair_ffn(edges))
            if not self.preln:
                edges = self.pair_layer_norm_ffn(edges)
            # residual = aa_edges
            # if self.preln:
            #     aa_edges = self.pair_layer_norm_ffn_aa(aa_edges)
            # aa_edges = residual + self.dropout_module(self.pair_ffn_aa(aa_edges))
            # if not self.preln:
            #     aa_edges = self.pair_layer_norm_ffn_aa(aa_edges)

    
        return atom_nodes, edges, aa_edges


class VecEdgeFeature(nn.Module):
    def __init__(
        self,
        input_dim=768,
        output_dim=768,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.linear_in = Linear(input_dim, input_dim, init='glorot')
        # self.linear_out = Linear(input_dim, output_dim, bias=True, init="glorot")

        self.embed = nn.Sequential(
            Linear(input_dim, output_dim, bias=True, init='glorot'),
            nn.LeakyReLU(),
            LayerNorm(output_dim),
            Linear(output_dim, output_dim, bias=True, init='glorot'),
            nn.LeakyReLU(),
            LayerNorm(output_dim),
            Linear(output_dim, output_dim, bias=True, init='glorot'),
        )

    def forward(self, base, input):
        x = torch.cat([base, input], dim=-1)
        x = x.type(self.embed[0].weight.dtype)
        # x = F.gelu(self.linear_in(x))
        # x = self.linear_out(x)
        return self.embed(x)


class RobustModule(nn.Module):
    def __init__(
        self,
        enable_nan_check: bool = False,
        tolerance: float = 1e3,
        strict: bool = False
    ):
        super().__init__()
        self.enable_nan_check = enable_nan_check
        self.tolerance = tolerance
        self.strict = strict
    
    @staticmethod
    def _display_warning(tensors: List[torch.Tensor], key: str):
        t: torch.Tensor
        if tensors is None:
            return
        for i, t in enumerate(tensors):   # todo use logging instead
            print(f"{key}_{i}: min={t.min():.2f} max={t.max():.2f}.")
    
    @staticmethod
    def _convert_to_lists(*args) -> tuple:
        return (
            v if type(v) in (tuple, list) else [v] for v in args
        )
    
    def check_nan(
        self,
        outputs: List[torch.Tensor],
        inputs: Optional[List[torch.Tensor]] = None,
        key: str = '',
        enable: Optional[bool] = None,
        tolerance: Optional[float] = None,
        strict: Optional[bool] = None,
    ):
        enable = enable if enable is not None else self.enable_nan_check
        if not enable:
            return
        tolerance = tolerance or self.tolerance
        strict = strict if strict is not None else self.strict
        outputs, inputs = self._convert_to_lists(outputs, inputs)
        for outv in outputs:
            if torch.any(outv.isnan()) or torch.any(outv.abs() > tolerance):
                print(f"nan detected in {key} (tol.={tolerance})")
                self._display_warning(outputs, "output")
                if inputs is not None:
                    self._display_warning(inputs, "input")
                if strict:
                    raise ValueError(f"nan detected in {key} (tol.={tolerance})")

def residual(residual, x, training):
    if training:
        return x + residual
    else:
        residual += x
        return residual

class DeltaFeature(nn.Module):
    def __init__(
        self,
        input_dim=768,
        output_dim=768,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.linear_in = Linear(input_dim, input_dim, init='glorot')
        # self.linear_out = Linear(input_dim, output_dim, bias=True, init="glorot")

        self.embed = nn.Sequential(
            Linear(input_dim, input_dim, bias=True, init='glorot'),
            nn.LeakyReLU(),
            LayerNorm(input_dim),
            Linear(input_dim, output_dim, bias=True, init='glorot'),
        )

    def forward(self, x):
        x = x.type(self.embed[0].weight.dtype)
        # x = F.gelu(self.linear_in(x))
        # x = self.linear_out(x)
        return self.embed(x)

class NodeEmbedHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()
        
        self.layer_norm = LayerNorm(input_dim)
        self.linear_in = Linear(input_dim, input_dim, init="relu")

        self.linear_out = Linear(input_dim, output_dim, bias=True, init="glorot")

    def forward(self, x):
        x = x.type(self.linear_in.weight.dtype)
        x = F.gelu(self.layer_norm(self.linear_in(x)))
        x = self.linear_out(x)
        return x


class IPA(RobustModule):
    """
    IPA takes care of linear attentions only.
    do layer norms before ipa; do transition after ipa.
    """
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_ipa: int=16,
        num_heads: int=12,
        num_qk_points: int=4,
        num_v_points: int=8,
        use_rel_pts_update: bool=False,
        use_point_bias: bool=True,
        eps: float = 1e-8,      # for pts norm calculation.
        use_ipa_edge_update=False,
        **base_args,
    ):
        super().__init__(**base_args)

        self.d_hid = d_ipa
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.eps = eps

        # self attention
        hc = self.d_hid * self.num_heads
        self.linear_q = Linear(d_single, hc, init="glorot")
        self.linear_k = Linear(d_single, hc, init="glorot")
        self.linear_v = Linear(d_single, hc, init="glorot")
        self.norm_attn = (1.0 / (3 * self.d_hid)) ** (-0.5)

        # biased attention
        self.linear_b = Linear(d_pair, self.num_heads, init="normal")
        self.norm_bias_attn = (1.0 / 3) ** (-0.5)

        # point attention
        hpqk = self.num_heads * self.num_qk_points * 3
        hpv = self.num_heads * self.num_v_points * 3
        self.linear_q_points = Linear(d_single, hpqk, bias=use_point_bias, init="glorot")
        self.linear_k_points = Linear(d_single, hpqk, bias=use_point_bias, init="glorot")
        self.linear_v_points = Linear(d_single, hpv, bias=use_point_bias, init="glorot")
        self.softplus = nn.Softplus()
        self.head_weights = nn.Parameter(torch.zeros((num_heads)))
        self.ipa_point_weights_init_(self.head_weights)
        self.norm_pt_att = -0.5 * (1.0 / (3 * (self.num_qk_points * 9.0 / 2))) **(-0.5)

        # triangle attention for z
        # self.linear_z_v_left = Linear(d_pair + 2 * d_single, d_pair, init="glorot")
        # self.linear_z_v_right = Linear(d_pair, hc, init="glorot")

        # gate & output linears
        self.linear_pair = Linear(d_pair, hc)
        s_out_dim = self.num_heads * (self.d_hid * 2 + self.num_v_points * 4)
        # self.linear_g_s = Linear(d_single, s_out_dim, init="gating")
        # self.sigmoid_s = nn.Sigmoid()
        self.linear_o_s = Linear(s_out_dim, d_single, init="final")

        self.use_rel_pts_update = use_rel_pts_update

        # self.linear_g_z = Linear(d_pair, d_pair, init="gating")
        # self.sigmoid_z = nn.Sigmoid()
        # self.linear_o_z = Linear(d_pair, d_pair, init="final")

    def ipa_point_weights_init_(self, weights):
        with torch.no_grad():
            softplus_inverse_1 = 0.541324854612918
            weights.fill_(softplus_inverse_1)

    def forward(
        self,
        s: torch.Tensor,            # *is
        z: torch.Tensor,            # *ijz
        f: Frame,                   # *if, in float.
        edge_index: torch.Tensor, 
    ) -> torch.Tensor:
        f = Frame.from_tensor_4x4(f)
        def h_linear(layer, x):     # *ia -> *ihb
            return layer(x).view(*x.shape[:-1], self.num_heads, -1)
        
        def pts_linear(layer, x, num_pts):  # *is -> *ihp3
            return layer(x).view(*x.shape[:-1], self.num_heads, num_pts, 3).float()

        def local_to_global(pts):   # *ihp3 -> *ihp3
            # s.dtype -> f.dtype
            return f[..., None, None].apply(pts.to(f.dtype))

        # node level: i -> E
        # edge level: ij -> E
        n_nodes = s.shape[0]
        n_edge = z.shape[0]
        # 1. calculate q, k & bias
        # (i, h, d) [E, H, D]
        n_node = s.shape[0]
        q = h_linear(self.linear_q, s)[edge_index[:, 0]]
        k = h_linear(self.linear_k, s)[edge_index[:, 1]]

        # (i, h, p, 3) [E, H, P, 3]
        q_pts_global = local_to_global(pts_linear(self.linear_q_points, s, self.num_qk_points))[edge_index[:, 0]]
        k_pts_global = local_to_global(pts_linear(self.linear_k_points, s, self.num_qk_points))[edge_index[:, 1]]

        # bias = permute_final_dims(self.linear_b(z), (2, 0, 1))  # *hij
        # (h, i, j) [H, E]
        bias = self.linear_b(z).permute(1, 0).contiguous()

        # 2. calculate attention matrix
        # attn = torch.einsum("...ihd,...jhd->...hij", q, k) * self.norm_attn
        # [H, E, D]
        attn = (q * k).permute(1, 0, 2).view(self.num_heads, edge_index.shape[0], -1).sum(dim=-1) * self.norm_attn
        del q, k
        
        # attn += attn_mask[..., None, :, :]
        # 
        # attn = residual(attn, bias * self.norm_bias_attn, self.training)
        attn = attn + bias * self.norm_bias_attn
        pt_att = q_pts_global - k_pts_global    # *ijhp3
        self.check_nan(pt_att, (q_pts_global, k_pts_global), key="pt_att")
        del q_pts_global, k_pts_global

        # [E, H]
        pt_att = (pt_att ** 2).sum(dim=(-1, -2)).to(z.dtype)   # *ijhp3 -> *ijh

        head_weights = self.softplus(self.head_weights)
        pt_att *= (head_weights * self.norm_pt_att)
        # [H, E]
        pt_att = pt_att.permute(1, 0).contiguous().to(z.dtype)

        attn = residual(attn, pt_att, self.training)
        # attn = torch.softmax(attn, dim=-1)
        attn = torch_scatter.composite.scatter_softmax(attn, edge_index[:, 0].unsqueeze(0))

        # 3. apply attention matrix to values

        # 3.1 self attention
        # v = h_linear(self.linear_v, s)  # *ihv
        v = h_linear(self.linear_v, s)[edge_index[:, 1]].permute(1, 0, 2).view(self.num_heads, edge_index.shape[0], -1)
        o = torch_scatter.scatter(attn.unsqueeze(-1) * v, edge_index[:, 0], dim=1, dim_size=n_node, reduce="sum")

        # o = torch.einsum("...hij,...jhd->...ihd", attn, v)
        # o = o.contiguous().view(*o.shape[:-2], -1)  # *io
        o = o.transpose(-2, -3).contiguous()
        o = o.view(*o.shape[:-2], -1)
        # self.check_nan(o, (v, attn), key="self_attn")
        del v

        # 3.2 point attention
        v_pts_global = local_to_global(pts_linear(self.linear_v_points, s, self.num_v_points))[edge_index[:, 1]].permute(1, 0, 2, 3).view(self.num_heads, edge_index.shape[0], -1, 3)
        o_pts_global = torch_scatter.scatter(attn.unsqueeze(-1).unsqueeze(-1).to(f.dtype) * v_pts_global, edge_index[:, 0], dim=1, dim_size=n_node, reduce="sum")
        # o_pts_global = torch.einsum("...hij,...jhpr->...ihpr", attn.to(f.dtype), v_pts_global)
        # o_pts_global = o_pts_global.contiguous().view(*o_pts_global.shape[:-3], -1, 3)  # *i(hp)3

        o_pts_global = o_pts_global.transpose(1, 0).contiguous()
        o_pts_global = o_pts_global.view(*o_pts_global.shape[:-3], -1, 3).contiguous()
        # o_pts_global = o_pts_global.view(*o_pts_global.shape[:-2], 8, 3)
        # o_pts = global_to_local(o_pts_global)
        o_pts = f[..., None].invert_apply(o_pts_global)
        self.check_nan((o_pts, o_pts_global), (v_pts_global, attn), key="o_pts")
        del v_pts_global

        if self.training:
            o_pts_norm = torch.sqrt(torch.sum(o_pts.float() ** 2, dim=-1) + self.eps)
        else:
            o_pts_norm = torch.sqrt(torch.sum(o_pts ** 2, dim=-1) + self.eps)
        self.check_nan(o_pts_norm, o_pts, key="o_pts_norm")
        # o_pts_norm = o_pts_norm.view(*o_pts_norm.shape[:-2], -1)
        o_pts = o_pts.view(*o_pts.shape[:-2], -1)   # *i(hp3)

        # 3.3 aggregate pair repr into s
        v_pair = h_linear(self.linear_pair, z).transpose(1, 0).view(self.num_heads, edge_index.shape[0], -1)
        o_pair = torch_scatter.scatter(attn.unsqueeze(-1) * v_pair, edge_index[:, 0], dim=1, dim_size=n_node, reduce="sum")
        # o_pair = torch.einsum("...hij,...ijhd->...ihd", attn, v_pair)
        # o_pair = o_pair.contiguous().view(*o_pair.shape[:-2], -1)
        o_pair = o_pair.transpose(1, 0).contiguous()
        o_pair = o_pair.view(*o_pair.shape[:-2], -1)
        del v_pair

        # 3.4 gated output
        o_s = torch.cat((o, o_pts.to(s.dtype), o_pts_norm.to(s.dtype), o_pair), dim=-1) # torch.Size([1006, 192]) torch.Size([1006, 96, 3]) torch.Size([1006, 96]) torch.Size([12, 12, 16096])
        del o, o_pts, o_pts_norm, o_pair
        # g_s = self.sigmoid_s(self.linear_g_s(s))
        s = self.linear_o_s(o_s)
        # s = self.linear_o_s(g_s * o_s)

        # 4. triangle attention for pair repr.
        # z_t = torch.cat([z, s[edge_index[:, 0]], s[edge_index[:, 1]]], dim=-1)
        # v_z = self.linear_z_v_left(z_t)

        # g_z = self.sigmoid_z(self.linear_g_z(z))
        # z = self.linear_o_z(g_z * v_z)

        return s, z


class EdgeFeature(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        pair_dim,
        num_edge,
        num_spatial,
    ):
        super(EdgeFeature, self).__init__()
        self.pair_dim = pair_dim

        self.edge_encoder = Embedding(num_edge, pair_dim, padding_idx=0)
        self.shorest_path_encoder = Embedding(num_spatial, pair_dim, padding_idx=0)
        self.vnode_virtual_distance = Embedding(1, pair_dim)

    def forward(self, batched_data, graph_attn_bias):
        shortest_path = batched_data["shortest_path"]
        edge_input = batched_data["edge_feat"]

        n_graph = graph_attn_bias.shape[0]
        n_node = graph_attn_bias.shape[-1] - 1

        graph_attn_bias[:, 1:, 1:, :] = self.shorest_path_encoder(shortest_path)

        # reset spatial pos here
        t = self.vnode_virtual_distance.weight.view(1, 1, self.pair_dim)
        graph_attn_bias[:, 1:, 0, :] = t
        graph_attn_bias[:, 0, :, :] = t

        edge_input = self.edge_encoder(edge_input).mean(-2)
        graph_attn_bias[:, 1:, 1:, :] = graph_attn_bias[:, 1:, 1:, :] + edge_input
        return graph_attn_bias

class ProteinEdgeFeature(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        pair_dim, # embed_dim
        num_atom_type, # 64
        num_residue_type,
    ):
        super(ProteinEdgeFeature, self).__init__()
        self.pair_dim = pair_dim
        self.num_atom_type = num_atom_type
        self.num_residue_type = num_residue_type 
        if num_atom_type is not None:
            if self.num_residue_type is not None:
                self.residue_atom_pair_encoder = Embedding(self.num_residue_type * self.num_atom_type, pair_dim)
            else:
                self.residue_aa_pair_encoder = Embedding(self.num_atom_type * self.num_atom_type, pair_dim)
        else:
            self.residue_pair_encoder = Embedding(self.num_residue_type * self.num_residue_type, pair_dim)       


    def forward(self, residue, edge_index, atom=None):
        # residue_pair = residue.unsqueeze(-1) * self.num_residue_type + residue.unsqueeze(1)

        if atom is None:
            residue_pair = residue[edge_index[:,0]] * self.num_residue_type + residue[edge_index[:,1]] 
            graph_attn_bias = self.residue_pair_encoder(residue_pair)
        else:
            if residue is not None:
                residue_atom_pair = residue[edge_index[:,0]] * self.num_residue_type + atom[edge_index[:,1]] 
                graph_attn_bias = self.residue_atom_pair_encoder(residue_atom_pair)
            else:
                aa_pair = atom[edge_index[:,0]] * self.num_atom_type + atom[edge_index[:,1]] 
                graph_attn_bias = self.residue_aa_pair_encoder(aa_pair)
        
        return graph_attn_bias


class AngleEmbedding(nn.Module):
    def __init__(self, N_freqs):
        """
        Current solution: nerf embeding strategy
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(AngleEmbedding, self).__init__()
        self.N_freqs = N_freqs

        self.funcs = [torch.sin, torch.cos]
        self.out_channels = len(self.funcs)*N_freqs+1
        self.freq_bands = 2**torch.linspace(0, N_freqs-1, int(N_freqs))
 
    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
 
        Inputs:
            x: (B, self.in_channels)
 
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x[:, :, :, 0], x[:, :, :, 1]]
        xsin = torch.asin(x[:, :, :, 0])
        xcos = torch.acos(x[:, :, :, 1])
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*xsin)]
                out += [func(freq*xcos)]
 
        return torch.cat(out, -1)


class TorsionFeature(nn.Module):
    def __init__(self, input, middle, output_size, hidden=None):
        super(TorsionFeature, self).__init__()
        self.middle = middle
        # self.angle_embed = AngleEmbedding((middle - 2) / 4)
        # self.layers =  nn.ModuleList([])
        # self.layers.extend(
        #     [
        #         Linear(middle, output_size, init="relu") for i in range(7)
        #     ]
        # )
        # for i in range(7):
        #     setattr(self, f"layer1_{i}", Linear(middle, output_size, init="relu"))
        self.nofeature = Embedding(1, 2)
        self.layers = NonLinear(2*7, output_size)

        
    def forward(self, x, mask):
        batch, length = x.shape[0], x.shape[1]
        x = x.reshape(batch, length, 7, 2)
        # x = self.angle_embed(x)
        # print('???', mask)
        x[mask.eq(0),:] = self.nofeature.weight.view(1, 2)
        x = x.reshape(batch, length, -1)

        # x = x.reshape(batch, length, 7, -1)
        # assert x.shape[3] == self.middle
        # feature = torch.zeros((x.shape[0],x.shape[1], x.shape[2], 768), dtype=x.dtype, device=x.device)
        # for i in range(7):
        #     feature[:, :, i, :] = self.layers[i](x[:, :, i, :])
        # feature[mask.eq(0)] = 0
        # x [2, 259, 7, 34]
        # mask [2, 259, 7]
        # feature [2, 259, 7, 768]
        
        # return feature.sum(dim=-2)

        return self.layers(x)

    def zero_init(self):
        nn.init.zeros_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

class AtomFeature(nn.Module):
    """
    Compute atom features for each atom in the molecule.
    """

    def __init__(
        self,
        num_atom,
        num_degree,
        hidden_dim,
    ):
        super(AtomFeature, self).__init__()
        self.atom_encoder = Embedding(num_atom, hidden_dim, padding_idx=0)
        self.degree_encoder = Embedding(num_degree, hidden_dim, padding_idx=0)
        self.vnode_encoder = Embedding(1, hidden_dim)

    def forward(self, batched_data):
        x, degree = (
            batched_data["atom_feat"],
            batched_data["degree"],
        )
        n_graph, n_node = x.size()[:2]

        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        dtype = node_feature.dtype
        degree_feature = self.degree_encoder(degree)
        node_feature = node_feature + degree_feature

        graph_token_feature = self.vnode_encoder.weight.unsqueeze(0).repeat(
            n_graph, 1, 1
        )

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature.type(dtype)