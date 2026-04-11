"""
Modular arithmetic dataset for grokking experiments.

Generates all p² pairs (a, b) for a op b mod p, randomly split into
train and test. Tokenizes as [a, op_token, b, eq_token] → label.

Tokenization scheme:
  - Tokens 0..p-1:   the numbers
  - Token p:          '+' operator
  - Token p+1:        '-' operator
  - Token p+2:        '*' operator
  - Token p+3:        '/' operator
  - Token p+4:        '=' (separator / end-of-input marker)
  Input sequence length: 4
  Target: scalar int in [0, p-1]
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

# Map operation string → token offset from p and computation function
OP_TOKEN_OFFSET = {'+': 0, '-': 1, '*': 2, '/': 3}

OPERATIONS = {
    '+': lambda a, b, p: (a + b) % p,
    '-': lambda a, b, p: (a - b) % p,
    '*': lambda a, b, p: (a * b) % p,
    '/': lambda a, b, p: (a * pow(b, p - 2, p)) % p,  # Fermat's little theorem
}


def get_vocab_size(p: int) -> int:
    """Total vocabulary size: p numbers + 4 ops + 1 eq symbol."""
    return p + 5


def get_op_token(p: int, operation: str) -> int:
    """Return the token ID for a given operation."""
    return p + OP_TOKEN_OFFSET[operation]


def get_eq_token(p: int) -> int:
    """Return the token ID for the '=' separator."""
    return p + 4


class ModularArithmeticDataset(Dataset):
    """
    Dataset of modular arithmetic problems.

    Each item is:
      inputs:  LongTensor of shape (4,)  = [a, op_token, b, eq_token]
      label:   LongTensor scalar         = (a op b) mod p
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs  # (N, 4)
        self.labels = labels  # (N,)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


def get_modular_arithmetic_datasets(
    p: int = 97,
    operation: str = '+',
    train_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[ModularArithmeticDataset, ModularArithmeticDataset]:
    """
    Generate all valid (a, b) pairs for `a op b mod p` and split into
    train / test.

    Args:
        p:               Prime modulus (97 and 113 are standard choices).
        operation:       One of '+', '-', '*', '/'.
        train_fraction:  Fraction of all pairs to use for training (0 < f < 1).
        seed:            Random seed for reproducible split.

    Returns:
        (train_dataset, test_dataset)

    Notes:
        - For '/', pairs with b=0 are excluded (division by zero).
        - The split is uniformly random over all valid pairs; never sequential.
        - Replicates the Power et al. (2022) experimental setup.
    """
    if operation not in OPERATIONS:
        raise ValueError(f"operation must be one of {list(OPERATIONS)}, got '{operation}'")

    op_fn = OPERATIONS[operation]
    op_token = get_op_token(p, operation)
    eq_token = get_eq_token(p)

    # Generate all valid pairs
    pairs = []
    labels = []
    for a in range(p):
        for b in range(p):
            if operation == '/' and b == 0:
                continue  # skip division by zero
            result = op_fn(a, b, p)
            pairs.append((a, b))
            labels.append(result)

    n = len(pairs)
    n_train = int(n * train_fraction)

    # Reproducible random shuffle
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    def build_tensors(idx_array):
        inputs_list = []
        labels_list = []
        for i in idx_array:
            a, b = pairs[i]
            inputs_list.append([a, op_token, b, eq_token])
            labels_list.append(labels[i])
        return (
            torch.tensor(inputs_list, dtype=torch.long),
            torch.tensor(labels_list, dtype=torch.long),
        )

    train_inputs, train_labels = build_tensors(train_idx)
    test_inputs, test_labels = build_tensors(test_idx)

    return (
        ModularArithmeticDataset(train_inputs, train_labels),
        ModularArithmeticDataset(test_inputs, test_labels),
    )


if __name__ == '__main__':
    # Quick sanity check
    p = 97
    train_ds, test_ds = get_modular_arithmetic_datasets(p=p, operation='+', train_fraction=0.5)
    total = train_ds.inputs.shape[0] + test_ds.inputs.shape[0]
    assert total == p * p, f"Expected {p**2} total pairs, got {total}"

    # Verify no overlap between train and test
    train_set = set(map(tuple, train_ds.inputs.tolist()))
    test_set = set(map(tuple, test_ds.inputs.tolist()))
    overlap = train_set & test_set
    assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} pairs"

    # Verify a label
    x, y = train_ds[0]
    a, op_tok, b, eq_tok = x.tolist()
    expected = (a + b) % p
    assert y.item() == expected, f"Label mismatch: expected {expected}, got {y.item()}"

    print(f"Dataset OK: {len(train_ds)} train, {len(test_ds)} test, vocab_size={get_vocab_size(p)}")
    print(f"Sample: {x.tolist()} → {y.item()}")
