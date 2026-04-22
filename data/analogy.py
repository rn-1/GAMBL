"""
Synthetic analogy dataset for grokking experiments.

Entities live on a P×Q grid whose group structure is Z_P × Z_Q.
Entity at position (r, c) has token id r*Q + c.

Analogy format:  a : b :: c : ?
  relation = (Δr, Δc) = (row(b)-row(a) mod P,  col(b)-col(a) mod Q), ≠ (0,0)
  answer   d = ((row(c) + Δr) % P, (col(c) + Δc) % Q)

This is 2-D modular arithmetic expressed as an analogy completion task.
Total valid quadruples: N*(N-1)*(N-1) where N = P*Q  (a≠b, a≠c enforced).
Default P=Q=5 → N=25 → 14 400 examples, comparable to mod-97 (9 409).

Tokenisation:
  0 .. P*Q-1   entity tokens
  P*Q          ':' separator  ("is to")
  P*Q + 1      '::' marker    ("as … is to")
  P*Q + 2      '=' query end
  Input length: 6   [a, ':', b, '::', c, '=']
  Label:        d   (entity token in [0, P*Q-1])
"""

import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_LEN = 6


def get_analogy_vocab_size(p: int, q: int) -> int:
    return p * q + 3


def get_colon_token(p: int, q: int) -> int:
    return p * q


def get_analogy_marker_token(p: int, q: int) -> int:
    return p * q + 1


def get_eq_token(p: int, q: int) -> int:
    return p * q + 2


class AnalogyDataset(Dataset):
    """
    Synthetic analogy completion dataset.

    Each item:
      inputs: LongTensor (6,) = [a, colon_tok, b, analogy_tok, c, eq_tok]
      label:  LongTensor scalar = d
    """

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


def get_analogy_datasets(
    p: int = 5,
    q: int = 5,
    train_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[AnalogyDataset, AnalogyDataset]:
    """
    Generate all valid analogy quadruples for a P×Q entity grid and split.

    Exclusions:
      - identity relation (Δr=0, Δc=0) — would make a==b
      - trivial c==a — answer would just be b

    Args:
        p:               Rows in the entity grid (Z_P component).
        q:               Columns in the entity grid (Z_Q component).
        train_fraction:  Fraction of examples for training.
        seed:            RNG seed for the split.

    Returns:
        (train_dataset, test_dataset)
    """
    n = p * q

    colon_tok   = get_colon_token(p, q)
    analogy_tok = get_analogy_marker_token(p, q)
    eq_tok      = get_eq_token(p, q)

    inputs_list: list[list[int]] = []
    labels_list: list[int] = []

    for a in range(n):
        row_a, col_a = divmod(a, q)
        for dr in range(p):
            for dc in range(q):
                if dr == 0 and dc == 0:
                    continue
                row_b = (row_a + dr) % p
                col_b = (col_a + dc) % q
                b = row_b * q + col_b
                for c in range(n):
                    if c == a:
                        continue
                    row_c, col_c = divmod(c, q)
                    row_d = (row_c + dr) % p
                    col_d = (col_c + dc) % q
                    d = row_d * q + col_d
                    inputs_list.append([a, colon_tok, b, analogy_tok, c, eq_tok])
                    labels_list.append(d)

    total = len(inputs_list)
    n_train = int(total * train_fraction)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    all_inputs = torch.tensor(inputs_list, dtype=torch.long)
    all_labels = torch.tensor(labels_list, dtype=torch.long)

    return (
        AnalogyDataset(all_inputs[train_idx], all_labels[train_idx]),
        AnalogyDataset(all_inputs[test_idx],  all_labels[test_idx]),
    )


if __name__ == '__main__':
    p, q = 5, 5
    train_ds, test_ds = get_analogy_datasets(p=p, q=q, train_fraction=0.5, seed=42)
    n = p * q
    total = len(train_ds) + len(test_ds)
    expected = n * (n - 1) * (n - 1)
    assert total == expected, f"Expected {expected}, got {total}"

    # Verify no train/test overlap
    train_set = set(map(tuple, train_ds.inputs.tolist()))
    test_set  = set(map(tuple, test_ds.inputs.tolist()))
    assert len(train_set & test_set) == 0, "Train/test overlap detected"

    # Verify a sample label
    x, y = train_ds[0]
    a, _ct, b, _at, c, _eq = x.tolist()
    row_a, col_a = divmod(a, q)
    row_b, col_b = divmod(b, q)
    row_c, col_c = divmod(c, q)
    dr = (row_b - row_a) % p
    dc = (col_b - col_a) % q
    d_expected = ((row_c + dr) % p) * q + (col_c + dc) % q
    assert y.item() == d_expected, f"Label mismatch: expected {d_expected}, got {y.item()}"

    print(f"AnalogyDataset OK: P={p} Q={q} N={n}")
    print(f"  total={total} (expected {expected}), train={len(train_ds)}, test={len(test_ds)}")
    print(f"  vocab_size={get_analogy_vocab_size(p, q)}, seq_len={SEQ_LEN}")
    print(f"  sample: {x.tolist()} → {y.item()}")
    print(f"    ({row_a},{col_a}):({row_b},{col_b}) :: ({row_c},{col_c}):(?, relation=({dr},{dc}))")
