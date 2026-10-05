import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(
            torch.softmax(scale * scores, dim=-1)
        )  # Scaled Dot-Product Attention
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # multi-head
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class SharedQueryAttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        num_queries=None,
        conditioned=True,
        query_dropout=0.0,
    ):
        super(SharedQueryAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        num_queries = int(num_queries) if num_queries is not None else 1
        if num_queries <= 0:
            raise ValueError("num_queries must be a positive integer.")

        self.inner_attention = attention
        self.n_heads = n_heads
        self.d_model = d_model
        self.num_queries = num_queries
        self.conditioned = conditioned

        self.shared_queries = nn.Parameter(torch.empty(1, num_queries, d_model))
        nn.init.trunc_normal_(self.shared_queries, std=0.02)

        if conditioned:
            self.query_conditioner = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.query_conditioner = None

        self.query_dropout = nn.Dropout(query_dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def _resize_shared_queries(self, target_length: int) -> torch.Tensor:
        """Return shared query templates with length equal to target_length."""
        if self.shared_queries.shape[1] == target_length:
            return self.shared_queries
        query = self.shared_queries.transpose(1, 2)  # [1, D, M]
        query = F.interpolate(
            query,
            size=target_length,
            mode="linear",
            align_corners=False,
        )
        return query.transpose(1, 2)  # [1, L, D]

    def _build_queries(self, source_tokens: torch.Tensor, target_length: int) -> torch.Tensor:
        B = source_tokens.shape[0]
        queries = self._resize_shared_queries(target_length).to(
            device=source_tokens.device,
            dtype=source_tokens.dtype,
        )
        queries = queries.expand(B, -1, -1)

        if self.query_conditioner is not None:
            context = source_tokens.mean(dim=1)  # [B, D]
            delta = self.query_conditioner(context).unsqueeze(1)
            queries = queries + delta

        return self.query_dropout(queries)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        shared_queries = self._build_queries(keys, L)

        queries = self.query_projection(shared_queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
