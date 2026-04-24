"""Repeated-subsequence autoregressive dataset.

Each example is a token sequence where one contiguous subsequence is repeated
later in the same sequence. Training target is next-token prediction:

  input_ids = seq[:-1]
  labels    = seq[1:]

Optionally, this dataset can append an END token target. When enabled:

  input_ids = seq
  labels    = seq[1:] + [END]
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


class SubsequenceDataset(Dataset):
    """Tensor-backed dataset wrapper matching the project API."""

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


def _generate_example(
    vocab_size: int,
    seq_len: int,
    subseq_len: int,
    rng: random.Random,
    append_end_token: bool,
    end_token_id: int,
) -> tuple[list[int], list[int]]:
    seq = [rng.randint(1, vocab_size - 1) for _ in range(seq_len)]

    start = rng.randint(0, seq_len - 2 * subseq_len - 1)
    subseq = seq[start:start + subseq_len]
    repeat_start = rng.randint(start + subseq_len, seq_len - subseq_len)
    seq[repeat_start:repeat_start + subseq_len] = subseq

    if append_end_token:
        return seq, seq[1:] + [end_token_id]
    return seq[:-1], seq[1:]


def get_subsequence_datasets(
    vocab_size: int = 50,
    seq_len: int = 64,
    subseq_len: int = 8,
    num_samples: int = 4096,
    train_fraction: float = 0.8,
    seed: int = 42,
    append_end_token: bool = True,
    end_token_id: int = 0,
) -> tuple[SubsequenceDataset, SubsequenceDataset, int, int, int]:
    """Create train/test splits for repeated-subsequence next-token prediction."""
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    if seq_len < 4:
        raise ValueError(f"seq_len must be >= 4, got {seq_len}")
    if subseq_len < 1:
        raise ValueError(f"subseq_len must be >= 1, got {subseq_len}")
    if 2 * subseq_len >= seq_len:
        raise ValueError(
            "subseq_len is too large for seq_len; must satisfy 2 * subseq_len < seq_len"
        )
    if num_samples < 2:
        raise ValueError(f"num_samples must be >= 2, got {num_samples}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0 <= end_token_id < vocab_size:
        raise ValueError(
            f"end_token_id must be in [0, vocab_size), got {end_token_id}"
        )

    rng = random.Random(seed)
    input_rows = []
    label_rows = []
    for _ in range(num_samples):
        x, y = _generate_example(
            vocab_size=vocab_size,
            seq_len=seq_len,
            subseq_len=subseq_len,
            rng=rng,
            append_end_token=append_end_token,
            end_token_id=end_token_id,
        )
        input_rows.append(x)
        label_rows.append(y)

    inputs = torch.tensor(input_rows, dtype=torch.long)
    labels = torch.tensor(label_rows, dtype=torch.long)

    n_train = int(num_samples * train_fraction)
    n_train = max(1, min(num_samples - 1, n_train))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_inputs = inputs[train_idx]
    test_inputs = inputs[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    train_ds = SubsequenceDataset(train_inputs, train_labels)
    test_ds = SubsequenceDataset(test_inputs, test_labels)

    output_dim = vocab_size
    lm_seq_len = seq_len if append_end_token else seq_len - 1
    return train_ds, test_ds, vocab_size, output_dim, lm_seq_len


if __name__ == '__main__':
    train_ds, test_ds, vocab_size, output_dim, seq_len = get_subsequence_datasets()
    x, y = train_ds[0]
    print(f"train={len(train_ds)} test={len(test_ds)}")
    print(f"vocab_size={vocab_size} output_dim={output_dim} seq_len={seq_len}")
    print("append_end_token=True end_token_id=0")
    print("input shape:", x.shape)
    print("label shape:", y.shape)
