"""
Decoder-only transformer for grokking experiments.

Matches the Power et al. (2022) architecture exactly:
  - Decoder-only with causal attention masking
  - 2 transformer layers (configurable)
  - d_model = 128 (configurable)
  - 4 attention heads (configurable)
  - FF dim = 512 = 4 * d_model (configurable)
  - Classification head on the last-position representation

Input: tokenized sequence [a, op_token, b, eq_token] of length 4.
Output: logits over p classes (the modular arithmetic result).

The causal mask means each token can only attend to itself and earlier tokens,
which is the critical difference from encoder-only (bidirectional) attention.
This forces the model to make its prediction based on [a, op, b, =] in order,
matching the original paper's setup.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block (LayerNorm before attention and FF).
    Pre-norm is more training-stable than post-norm for small models.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

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

    def forward(self, x: Tensor, causal_mask: Tensor = None) -> Tensor:
        """
        Args:
            x:           (batch, seq_len, d_model)
            causal_mask: (seq_len, seq_len) additive attention mask.
                         -inf at positions that should be masked, 0 elsewhere.

        Returns:
            x: (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm and residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=causal_mask,
                                need_weights=False)
        x = x + attn_out

        # Feed-forward with pre-norm and residual
        x = x + self.ff(self.norm2(x))
        return x


class GrokTransformerDecoder(nn.Module):
    """
    Decoder-only transformer for algorithmic classification tasks.

    Matches Power et al. (2022) exactly:
      1. Token embedding: vocab_size → d_model
      2. Learned positional embedding: max_seq_len → d_model
      3. n_layers TransformerBlock layers with causal attention mask
      4. Final LayerNorm
      5. Read off last position (the '=' token)
      6. Linear head: d_model → output_dim

    The causal mask is the key difference from encoder-only: each token
    can only attend to itself and earlier tokens, forcing autoregressive
    prediction and matching the original paper's decoder-only setup.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        output_dim: int = 97,
        max_seq_len: int = 4,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        pool: str = 'last',
    ):
        super().__init__()
        if pool not in ('last', 'mean'):
            raise ValueError(f"pool must be 'last' or 'mean', got '{pool}'")

        self.pool = pool
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_embedding = None
        if use_positional_encoding:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following common transformer practice."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len) LongTensor of token indices.

        Returns:
            logits: (batch, output_dim)
        """
        B, T = x.shape

        tok_emb = self.token_embedding(x)  # (B, T, d_model)

        if self.pos_embedding is not None:
            positions = torch.arange(T, device=x.device)
            pos_emb = self.pos_embedding(positions)  # (T, d_model)
            h = self.dropout(tok_emb + pos_emb)
        else:
            h = self.dropout(tok_emb)

        # Causal mask: upper triangle is -inf, diagonal and below is 0
        # Shape (T, T) — position i can attend to positions 0..i only
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )

        for block in self.blocks:
            h = block(h, causal_mask=causal_mask)

        h = self.norm(h)  # (B, T, d_model)

        # Always use last position (the '=' token) — no pooling option needed
        # for decoder-only since causal masking makes last position the natural
        # aggregation point
        logits = self.head(h[:, -1, :])   # (B, output_dim)
        return logits
