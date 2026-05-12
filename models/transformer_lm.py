"""Decoder-only transformer language model for sequence generation tasks."""

import torch
import torch.nn as nn
from torch import Tensor


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, causal_mask: Tensor, padding_mask: Tensor = None) -> Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class GrokTransformerLM(nn.Module):
    """Causal decoder that returns per-token vocabulary logits."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model) if use_positional_encoding else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Tensor:
        _, t = x.shape
        h = self.token_embedding(x)
        if self.pos_embedding is not None:
            positions = torch.arange(t, device=x.device)
            h = self.dropout(h + self.pos_embedding(positions))
        else:
            h = self.dropout(h)

        causal_mask = torch.triu(
            torch.full((t, t), float('-inf'), device=x.device),
            diagonal=1,
        )

        for block in self.blocks:
            h = block(h, causal_mask=causal_mask, padding_mask=padding_mask)

        h = self.norm(h)
        return self.lm_head(h)
