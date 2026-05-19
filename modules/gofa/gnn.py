from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, add_self_loops
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, repeat_kv, \
    MistralMLP, MistralRMSNorm, apply_rotary_pos_emb, rotate_half

import pdb
from gp.nn.models.util_model import MLP


def apply_rotary_pos_emb_single(q, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


class GOFAAttention(MessagePassing):
    """
    GOFA based on Grouped query attention from LlaMA-2. This is the FULL attention version, not the INDEX attention
    version mentioned in the paper.
    TODO: upload full attention version.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(node_dim=0, aggr="add")
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.gq_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.gk_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.gv_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.go_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_hidden_states: torch.Tensor,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.gq_proj(hidden_states)
        key_states = self.gk_proj(hidden_states)
        value_states = self.gv_proj(hidden_states)

        edge_key_states = self.gk_proj(edge_hidden_states)
        edge_value_states = self.gv_proj(edge_hidden_states)

        out = self.propagate(edge_index, query=query_states, key=key_states, value=value_states, edge_key=edge_key_states, edge_value=edge_value_states)

        out = self.go_proj(out)

        return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j, edge_key: Tensor, edge_value: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        bsz, q_len, dim = query_i.size()
        query_states = query_i
        key_states = torch.stack([key_j, edge_key], dim=2).view(bsz, q_len*2, -1)
        value_states = torch.stack([value_j, edge_value], dim=2).view(bsz, q_len*2, -1)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, 2*q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, 2*q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos_q, sin_q = self.rotary_emb(value_states, torch.arange(q_len, device=value_j.device).unsqueeze(0))
        cos_k, sin_k = self.rotary_emb(value_states, torch.arange(q_len * 2, device=value_j.device).unsqueeze(0))
        query_states = apply_rotary_pos_emb_single(query_states, cos_q, sin_q)
        key_states = apply_rotary_pos_emb_single(key_states, cos_k, sin_k)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        alpha = (query_states @ key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        softmax_ind = index.repeat_interleave(q_len*2)

        alpha = alpha.permute(1, 0, 3, 2).reshape(self.num_heads, -1, q_len)

        if alpha.size() != (self.num_heads, q_len * 2 * bsz, q_len):
            raise ValueError(f"`alpha` should be of size {(self.num_heads, q_len * 2 * bsz, q_len)}, but is"
                             f" {alpha.size()}")

        # Attention: Training with ditributed or deepspeed bf16 on PyG softmax is extremely slow.

        alpha = softmax(alpha, softmax_ind, num_nodes=size_i, dim=1)
        alpha = F.dropout(alpha, p=self.attention_dropout, training=self.training)
        alpha = alpha.view(self.num_heads, bsz, q_len*2, q_len).permute(1, 0, 3, 2)

        out = alpha @ value_states
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(bsz, q_len, self.hidden_size)

        return out

class GOFAAttentionIndex(GOFAAttention):
    """
    GOFA based on Grouped query attention from LlaMA-2. This is the INDEX attention version.
    """
    def message(self, query_i: Tensor, key_j: Tensor, value_j, edge_key: Tensor, edge_value: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        bsz, q_len, dim = query_i.size()
        query_states = query_i
        key_states = edge_key + key_j
        value_states = value_j + edge_value
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(value_states, torch.arange(q_len, device=value_j.device).unsqueeze(0))
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        alpha = (query_states * key_states).sum(dim=-1) / math.sqrt(self.head_dim)

        alpha = softmax(alpha, index, ptr, size_i, dim=0)

        alpha = F.dropout(alpha, p=self.attention_dropout, training=self.training)

        out = value_states * alpha.unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()
        out = out.view(-1, q_len, self.hidden_size)

        return out


class GOFADecoderLayer(nn.Module):
    """
    GOFA decoder layer, this is the version without tanh gate.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.gnn_type == "full":
            self.self_attn = GOFAAttention(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = GOFAAttentionIndex(config=config, layer_idx=layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_hidden_states: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        edge_hidden_states = self.input_layernorm(edge_hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            edge_index=edge_index,
            edge_hidden_states=edge_hidden_states,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GOFAGatedDecoderLayer(GOFADecoderLayer):
    """
    GOFA decoder layer, this is the version with tanh gate.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_hidden_states: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        edge_hidden_states = self.input_layernorm(edge_hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            edge_index=edge_index,
            edge_hidden_states=edge_hidden_states,
        )
        # apply tanh gate to ensure identical initial output.
        hidden_states = residual + hidden_states * self.attn_gate.tanh()

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # apply tanh gate to ensure identical initial output.
        hidden_states = residual + hidden_states * self.ff_gate.tanh()

        return hidden_states


###########################################################
#                  Legacy implementation                  #
###########################################################

class GOFAGNNConv(MessagePassing):
    _alpha: OptTensor

    # Basic GOFA implementation using PyG. Use Multi-Head attention.

    def __init__(self, config):
        super().__init__(node_dim=0, aggr="add")

        self.in_dim = config.dim
        self.in_layer = config.mem_token
        self.dropout = config.dropout
        self.head = config.dim

        assert self.in_dim % self.head == 0

        self.d_model = int(self.in_dim / self.head)

        self.add_self_loops = False

        self.lin_qkv = Linear(self.in_dim, self.in_dim * 3, bias=False)

        self.e_proj = Linear(self.in_dim, self.in_dim * 2, bias=False)
        self.layer_norm_ek = MistralRMSNorm(self.in_dim)
        self.layer_norm_ev = MistralRMSNorm(self.in_dim)

        self.o_proj = Linear(self.in_dim, self.in_dim, bias=False)

        if config.gnn_mlp_type == "gp":

            self.ff = MLP([self.in_dim, 2 * self.in_dim, self.in_dim], dropout=self.dropout, act=config.gnn_hidden_act)
        elif config.gnn_mlp_type == "llama":
            self.ff = MistralMLP(config)
        else:
            raise NotImplementedError("Unknown mlp type")

        self.x_norm = MistralRMSNorm(self.in_dim)
        self.xe_norm = MistralRMSNorm(self.in_dim)

        self.post_gnn_norm = MistralRMSNorm(self.in_dim)

        if config.gating:
            self.attn_gate = nn.Parameter(torch.tensor([0.]))
            self.ff_gate = nn.Parameter(torch.tensor([0.]))
        else:
            self.attn_gate = None
            self.ff_gate = None

        if config.position_encoding == "rotary":
            self.rotary_emb = MistralRotaryEmbedding(config)
        elif config.position_encoding == "none":
            self.rotary_emb = None
        else:
            raise ValueError("Unknown position encoding.")

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, xe: Tensor):
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
        """
        x = x.view(x.size()[0], self.in_layer, self.in_dim)
        # Q = x.clone().detach()
        residual = x
        x = self.x_norm(x)
        xe = xe.view(xe.size()[0], self.in_layer, self.in_dim)
        xe = self.xe_norm(xe)

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, xe = add_self_loops(edge_index, edge_attr=xe, fill_value=0.0, num_nodes=num_nodes)

        qkv = self.lin_qkv(x)
        query, key, value = torch.chunk(qkv, 3, -1)

        xe = self.e_proj(xe)

        xe_key, xe_value = torch.chunk(xe, 2, -1)

        out = self.propagate(edge_index, query=query, key=key, value=value, xe_key=xe_key, xe_value=xe_value)
        out = self.o_proj(out)

        # Initital gating
        if self.attn_gate is not None:
            out = residual + out * self.attn_gate.tanh()
        else:
            out = residual + out

        residual = out

        out = self.post_gnn_norm(out)
        out = self.ff(out)
        if self.ff_gate is not None:
            out = residual + out * self.ff_gate.tanh()
        else:
            out = residual + out
        #
        # H = out.clone().detach().view(Q.size())
        # diff = (((H - Q).to(torch.float32))**2).sum(dim=(-2))/((Q.to(torch.float32))**2).sum(dim=(-2))
        # print(diff.sum(), len(diff.view(-1)), diff.mean())

        return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor, xe_key: Tensor, xe_value: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        key_j = xe_key + key_j
        value_j = value_j + xe_value
        query_i = query_i.view(-1, self.in_layer, self.head, self.d_model)
        key_j = key_j.view(-1, self.in_layer, self.head, self.d_model)
        value_j = value_j.view(-1, self.in_layer, self.head, self.d_model)
        if self.rotary_emb is not None:
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
            cos, sin = self.rotary_emb(value_j, torch.arange(self.in_layer, device=value_j.device).unsqueeze(0))
            query_i, key_j = apply_rotary_pos_emb(query_i, key_j, cos, sin)
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.d_model)

        alpha = softmax(alpha, index, ptr, size_i, dim=0)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j.view(-1, self.d_model)
        out = out * alpha.view(-1, 1)
        out = out.view(-1, self.in_layer, self.in_dim)

        return out

class GOFAGNNConvFullAtt(GOFAGNNConv):
    def __init__(self, config):
        super().__init__(config)

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor, xe_key: Tensor, xe_value: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        key_j = key_j + xe_key
        value_j = value_j + xe_value
        query_i = query_i.view(-1, self.in_layer, self.head, self.d_model)
        key_j = key_j.view(-1, self.in_layer, self.head, self.d_model)
        value_j = value_j.view(-1, self.in_layer, self.head, self.d_model)
        if self.rotary_emb is not None:
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
            cos, sin = self.rotary_emb(value_j, self.in_layer)
            query_i, key_j = apply_rotary_pos_emb(query_i, key_j, cos, sin,
                                                  torch.arange(self.in_layer, device=value_j.device).unsqueeze(0))
            key_j = key_j.transpose(1, 2)
            query_i = query_i.transpose(1, 2)
        key_j = key_j.permute(2, 0, 1, 3)
        query_i = query_i.permute(2, 0, 3, 1)
        value_j = value_j.permute(2, 0, 1, 3)
        alpha = (key_j @ query_i) / math.sqrt(self.d_model)
        softmax_ind = index.repeat_interleave(self.in_layer)

        alpha = alpha.view(self.head, -1, self.in_layer)

        alpha = softmax(alpha, softmax_ind, num_nodes=size_i, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha = alpha.view(self.head, -1, self.in_layer, self.in_layer).transpose(-1, -2)

        out = alpha @ value_j
        out = out.permute(1, 2, 0, 3).reshape(-1, self.in_layer, self.in_dim)

        return out
