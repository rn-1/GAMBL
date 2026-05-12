"""
Encoder-only transformer for grokking experiments (bidirectional attention).
NOT the Power et al. setup — included for comparison only.
Note: encoder does NOT reliably reproduce grokking; use transformer_decoder.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class GrokTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, output_dim=97, max_seq_len=4, dropout=0.1,
                 use_positional_encoding=True, pool='last'):
        super().__init__()
        self.pool = pool
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model) if use_positional_encoding else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, padding_mask=None):
        B, T = x.shape
        h = self.token_embedding(x)
        if self.pos_embedding is not None:
            h = self.dropout(h + self.pos_embedding(torch.arange(T, device=x.device)))
        else:
            h = self.dropout(h)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        pooled = h[:, -1, :] if self.pool == 'last' else h.mean(dim=1)
        return self.head(pooled)
