"""
MLP model for grokking experiments.

Takes the same tokenized input as the transformer ([a, op, b, eq] indices),
embeds each token, flattens the embeddings, and passes through MLP layers.
This makes the MLP/transformer comparison clean: they differ only in their
sequence processing mechanism (flatten vs attention), not in input format.
"""

import torch
import torch.nn as nn
from torch import Tensor


ACTIVATIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
}


class MLP(nn.Module):
    """
    Embedding → flatten → MLP → logits.

    Architecture:
      1. Shared token embedding: vocab_size → embed_dim
      2. Flatten: embed_dim * seq_len → first hidden layer
      3. (num_layers - 1) hidden layers: hidden_dim → hidden_dim
      4. Output layer: hidden_dim → output_dim

    Args:
        vocab_size:   Total number of tokens (from get_vocab_size(p)).
        embed_dim:    Embedding dimension per token.
        hidden_dim:   Width of each hidden layer.
        num_layers:   Total number of linear layers (including output).
                      num_layers=1 → linear classifier with no hidden layers.
        output_dim:   Number of output classes (= p for modular arithmetic).
        activation:   'relu', 'gelu', or 'tanh'.
        dropout:      Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        output_dim: int = 97,
        activation: str = 'relu',
        dropout: float = 0.0,
        seq_len: int = 4,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)}, got '{activation}'")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        act_cls = ACTIVATIONS[activation]
        input_dim = embed_dim * seq_len

        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Tensor:
        """
        Args:
            x:            (batch, seq_len) LongTensor of token indices.
            padding_mask: ignored — MLP embeds each token independently.

        Returns:
            logits: (batch, output_dim)
        """
        embedded = self.embedding(x)           # (B, seq_len, embed_dim)
        flat = embedded.flatten(start_dim=1)   # (B, seq_len * embed_dim)
        return self.net(flat)                  # (B, output_dim)
