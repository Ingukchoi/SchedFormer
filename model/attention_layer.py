import math
import typing
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import softmax

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload

from model.sub_layers import RMSNorm


class MultiheadSelfAttention_for_OP_withRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 use_rope: bool = True, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        # RoPE is defined on the per-head dimension (d_k)
        if self.use_rope:
            assert self.d_k % 2 == 0, "For RoPE, d_k must be even"
            inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
            # [d_k/2]
            self.register_buffer("inv_freq", inv_freq)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _apply_rope(self, Q, K):
        BJ, h, O, d_k = Q.shape
        device = Q.device

        # [O]
        pos = torch.arange(O, device=device, dtype=torch.float32)

        # freqs: [O, d_k/2]
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)

        # cos, sin: broadcast to [1,1,O,d_k/2]
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]

        # split Q, K into even / odd dimensions
        q1 = Q[..., 0::2]   # [BJ,h,O,d_k/2]
        q2 = Q[..., 1::2]
        k1 = K[..., 0::2]
        k2 = K[..., 1::2]

        # rotation: (x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin)
        q_rot_1 = q1 * cos - q2 * sin
        q_rot_2 = q2 * cos + q1 * sin
        k_rot_1 = k1 * cos - k2 * sin
        k_rot_2 = k2 * cos + k1 * sin

        Q_rot = torch.stack([q_rot_1, q_rot_2], dim=-1).reshape(BJ, h, O, d_k)
        K_rot = torch.stack([k_rot_1, k_rot_2], dim=-1).reshape(BJ, h, O, d_k)
        return Q_rot, K_rot

    def forward(self, op_feature: torch.Tensor, deleted_op_nodes: torch.Tensor) -> torch.Tensor:
        B, J, O, d = op_feature.shape
        assert d == self.d_model
        device = op_feature.device

        BJ = B * J
        if deleted_op_nodes.dtype is not torch.bool:
            deleted_op_nodes = deleted_op_nodes.bool()
        del_mask = deleted_op_nodes.view(B, J, O).reshape(BJ, O)
        x = op_feature.reshape(BJ, O, d)

        # Projections
        Q = self.W_q(x)  # [BJ,O,d]
        K = self.W_k(x)
        V = self.W_v(x)

        # To multi-head: [BJ,h,O,d_k]
        Q = Q.view(BJ, O, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(BJ, O, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(BJ, O, self.num_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q, K = self._apply_rope(Q, K)

        # Scores: [BJ,h,O,O]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        base = (~del_mask).unsqueeze(1).expand(BJ, O, O)
        self_mask = torch.eye(O, dtype=torch.bool, device=device).unsqueeze(0).expand(BJ, O, O)
        mask = base | self_mask
        mask = mask.unsqueeze(1)        # [BJ,1,O,O]
        scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)  # [BJ,h,O,d_k]
        out = out.transpose(1, 2).contiguous().view(BJ, O, self.d_model)  # [BJ,O,d]

        out = self.W_o(out)
        return out.view(B, J, O, d)


class MultiheadSelfAttention_for_MC(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, use_residual: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_residual = use_residual

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def _shape(x, n_heads):
        B, L, D = x.shape
        d_head = D // n_heads
        return x.view(B, L, n_heads, d_head).permute(0, 2, 1, 3).contiguous()

    def forward(self, fea_m: torch.Tensor, mc_mask: torch.Tensor):
        B, M, d = fea_m.shape
        assert d == self.n_heads * self.d_head
        assert mc_mask.shape == (B, M) and mc_mask.dtype == torch.bool

        device = fea_m.device
        eye = torch.eye(M, dtype=torch.bool, device=device).unsqueeze(0)   # [1,M,M]
        alive_ij = (mc_mask.unsqueeze(-1) & mc_mask.unsqueeze(-2))         # [B,M,M]
        allowed = alive_ij | eye                                           # [B,M,M]
        mask = allowed.unsqueeze(1)                                        # [B,1,M,M] (True=allowed)

        # Q/K/V
        Q = self._shape(self.W_q(fea_m), self.n_heads)  # [B,h,M,d_h]
        K = self._shape(self.W_k(fea_m), self.n_heads)  # [B,h,M,d_h]
        V = self._shape(self.W_v(fea_m), self.n_heads)  # [B,h,M,d_h]
        scale = self.d_head ** 0.5

        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale          # [B,h,M,M]
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        updated_embed = torch.matmul(attn, V)                          # [B,h,M,d_h]

        updated_embed = updated_embed.permute(0, 2, 1, 3).contiguous().view(B, M, d)  # [B,M,d]
        out = self.W_o(updated_embed)
        return out


class MixedCrossAttention(nn.Module):
    def __init__(self, d_op, d_mch, d_edge, d_hidden, heads=4, dropout=0.1, eps=1e-8):
        super().__init__()
        assert d_hidden % heads == 0
        self.d_hidden = d_hidden
        self.per_head = d_hidden // heads
        self.eps = eps

        self.proj = nn.ModuleDict({
            'op': nn.Linear(d_op, d_hidden),
            'mch': nn.Linear(d_mch, d_hidden),
        })

        self.CrossAttn = HeteroConv({
            ('op', 'assign', 'mch'): TransformerConv2(
                in_channels=(d_hidden, d_hidden),
                out_channels=self.per_head,
                heads=heads,
                concat=True,
                edge_dim=d_edge,
                dropout=dropout,
                beta=False
            ),
            ('mch', 'recruit', 'op'): TransformerConv2(
                in_channels=(d_hidden, d_hidden),
                out_channels=self.per_head,
                heads=heads,
                concat=True,
                edge_dim=d_edge,
                dropout=dropout,
                beta=False
            ),
        }, aggr='sum')

        self.mix = nn.ModuleDict({
            'op': nn.Linear(d_hidden, d_hidden),
            'mch': nn.Linear(d_hidden, d_hidden),
        })

    def forward(self, hetero_batch: Batch):
        x_dict = {'op': hetero_batch['op'].x,
                  'mch': hetero_batch['mch'].x}

        edge_attr_dict = {}
        for et in [('op', 'assign', 'mch'), ('mch', 'recruit', 'op')]:
            edge_attr_dict[et] = getattr(hetero_batch[et], 'edge_attr', None)

        out = self.CrossAttn(
            x_dict,
            hetero_batch.edge_index_dict,
            edge_attr_dict=edge_attr_dict
        )

        out_op = out.get('op', torch.zeros_like(x_dict['op']))
        out_mch = out.get('mch', torch.zeros_like(x_dict['mch']))

        out_op = self.mix['op'](out_op)    # [N_op, d_hidden]
        out_mch = self.mix['mch'](out_mch)  # [N_mch, d_hidden]

        return out_op, out_mch


class EdgeUpdater(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.W3 = nn.Linear(d, d, bias=False)
        self.W4 = nn.Linear(d, d, bias=False)
        self.W5 = nn.Linear(d, d, bias=False)

        self.norm = RMSNorm(d)

    def forward(self, e_attr, x_src, y_dst):
        """
        e_attr: [E, d]
        x_src : [E, d]  (source node embedding of the edge)
        y_dst : [E, d]  (target node embedding of the edge)
        """
        fused = self.W3(self.norm(e_attr)) + self.W4(x_src) + self.W5(y_dst)
        fused = F.silu(fused)

        return e_attr + fused


class TransformerConv2(MessagePassing):
    """
    A variant of PyG's TransformerConv in which the edge attributes are projected
    separately into the key, the value and the query streams.
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        # separate edge projections for the key / value / query streams
        if edge_dim is not None:
            self.lin_edge_k = Linear(edge_dim, heads * out_channels, bias=False)
            self.lin_edge_v = Linear(edge_dim, heads * out_channels, bias=False)
            self.lin_edge_q = Linear(edge_dim, heads * out_channels, bias=False)
            self.norm = RMSNorm(edge_dim)
        else:
            self.lin_edge_k = self.register_parameter('lin_edge_k', None)
            self.lin_edge_v = self.register_parameter('lin_edge_v', None)
            self.lin_edge_q = self.register_parameter('lin_edge_q', None)
            self.norm = self.register_parameter('norm', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge_k.reset_parameters()
            self.lin_edge_v.reset_parameters()
            self.lin_edge_q.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    @overload
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module."""
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor,
        #                  edge_attr: OptTensor)
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        H, C = self.heads, self.out_channels

        if edge_attr is not None:
            edge_attr = self.norm(edge_attr)

        if self.lin_edge_k is not None:
            key_j = key_j + self.lin_edge_k(edge_attr).view(-1, H, C)

        if self.lin_edge_v is not None:
            value_j = value_j + self.lin_edge_v(edge_attr).view(-1, H, C)

        if self.lin_edge_q is not None:
            query_i = query_i + self.lin_edge_q(edge_attr).view(-1, H, C)

        # attention score
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(C)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        out = out * alpha.view(-1, H, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
