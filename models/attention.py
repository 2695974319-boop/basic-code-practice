import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_keys = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, query, keys, src_mask=None):
        scores = self.V(torch.tanh(self.W_query(query) + self.W_keys(keys))).squeeze(-1)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(1)
        context = torch.bmm(weights, keys)
        return context, weights


class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query, keys, src_mask=None):
        scores = torch.bmm(self.W_query(query), keys.transpose(1, 2)).squeeze(1)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(1)
        context = torch.bmm(weights, keys)
        return context, weights


class DotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = math.sqrt(hidden_size)

    def forward(self, query, keys, src_mask=None):
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / self.scale
        if src_mask is not None:
            scores = scores.masked_fill(src_mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(1)
        context = torch.bmm(weights, keys)
        return context, weights


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        if hidden_size % num_heads != 0:
            num_heads = 1
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, query, keys, src_mask=None):
        context, weights = self.attention(
            query=query,
            key=keys,
            value=keys,
            key_padding_mask=src_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        return context, weights


def build_attention(attention_type, hidden_size):
    attention_type = str(attention_type or "bahdanau").lower()
    if attention_type == "bahdanau":
        return BahdanauAttention(hidden_size)
    if attention_type == "luong":
        return LuongAttention(hidden_size)
    if attention_type == "dot":
        return DotProductAttention(hidden_size)
    if attention_type == "multihead":
        return MultiHeadCrossAttention(hidden_size)
    raise ValueError(
        "Unsupported attention_type. "
        "Choose one of: bahdanau, luong, dot, multihead."
    )
