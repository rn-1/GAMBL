"""
Encoder-only transformer for grokking classification experiments.

Closely follows the Power et al. (2022) architecture:
  - 2 transformer layers (configurable)
  - d_model = 128 (configurable)
  - 4 attention heads (configurable)
  - FF dim = 512 = 4 * d_model (configurable)
  - Classification head on the last-position representation

Input: tokenized sequence [a, op_token, b, eq_token] of length 4.
Output: logits over p classes (the modular arithmetic result).

No padding mask is needed since all sequences are exactly 4 tokens.
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x: (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm and residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out

        # Feed-forward with pre-norm and residual
        x = x + self.ff(self.norm2(x))
        return x


class GrokTransformer(nn.Module):
    """
    Encoder-only transformer for algorithmic classification tasks.

    Architecture:
      1. Token embedding: vocab_size → d_model
      2. Optional learned positional embedding: max_seq_len → d_model
      3. n_layers TransformerBlock layers
      4. Final LayerNorm
      5. Pool over sequence (last position or mean)
      6. Linear head: d_model → output_dim

    Args:
        vocab_size:               Total number of tokens (p + 5).
        d_model:                  Model dimension.
        n_heads:                  Number of attention heads.
        n_layers:                 Number of transformer blocks.
        d_ff:                     Feed-forward hidden dimension.
        output_dim:               Number of output classes (= p).
        max_seq_len:              Maximum sequence length (4 for modular arith.).
        dropout:                  Dropout probability.
        use_positional_encoding:  If True, add learned positional embeddings.
        pool:                     'last' (use last token) or 'mean' (average).
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

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)  # (B, T, d_model)

        if self.pool == 'last':
            pooled = h[:, -1, :]   # last position = '=' token representation
        else:
            pooled = h.mean(dim=1)

        return self.head(pooled)   # (B, output_dim)
