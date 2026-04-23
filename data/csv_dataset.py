"""CSV-backed analogy datasets for grokking experiments.

The expected CSV layout is four columns named:
    word_one, word_two, word_three, word_four

Each row contributes one analogy sample:
    (word_one, word_two, word_three) -> word_four

This represents: "word_one relates to word_two as word_three relates to ?".
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ('word_one', 'word_two', 'word_three', 'word_four')


class CsvDataset(Dataset):
    """Simple dataset wrapper matching the modular-arithmetic API."""

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


def _read_csv(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"CSV file must contain columns {list(REQUIRED_COLUMNS)}; missing {missing}"
        )
    return frame


def _build_input_vocab(values: list[object]) -> tuple[dict[str, int], int]:
    vocab: OrderedDict[str, int] = OrderedDict()
    for value in values:
        key = str(value)
        if key not in vocab:
            vocab[key] = len(vocab)
    return dict(vocab), len(vocab)


def _build_label_map(values: list[object]) -> tuple[dict[str, int], int]:
    label_map: OrderedDict[str, int] = OrderedDict()
    for value in values:
        key = str(value)
        if key not in label_map:
            label_map[key] = len(label_map)
    return dict(label_map), len(label_map)


def _encode_inputs_with_vocab(values: list[object], vocab: dict[str, int]) -> torch.Tensor:
    encoded = [vocab[str(value)] for value in values]
    return torch.tensor(encoded, dtype=torch.long)


def _encode_labels_with_map(values: list[object], label_map: dict[str, int]) -> torch.Tensor:
    encoded = [label_map[str(value)] for value in values]
    return torch.tensor(encoded, dtype=torch.long)


def get_csv_datasets(
    csv_path: str | Path,
    train_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[CsvDataset, CsvDataset, int, int, int]:
    """Load CSV analogies and split rows into train/test by train_fraction."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    frame = _read_csv(csv_path)

    word_one = frame['word_one'].tolist()
    word_two = frame['word_two'].tolist()
    word_three = frame['word_three'].tolist()
    word_four = frame['word_four'].tolist()

    # Build a shared token vocab for the 3-token input sequence.
    all_input_values = word_one + word_two + word_three
    input_vocab, vocab_size = _build_input_vocab(all_input_values)

    # Output classes are the target tokens to predict.
    label_map, output_dim = _build_label_map(word_four)

    x1 = _encode_inputs_with_vocab(word_one, input_vocab)
    x2 = _encode_inputs_with_vocab(word_two, input_vocab)
    x3 = _encode_inputs_with_vocab(word_three, input_vocab)
    inputs = torch.stack([x1, x2, x3], dim=1)  # (N, 3)
    labels = _encode_labels_with_map(word_four, label_map)

    n = inputs.shape[0]
    n_train = int(n * train_fraction)
    n_train = max(1, min(n - 1, n_train))

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_inputs = inputs[train_idx]
    test_inputs = inputs[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    train_ds = CsvDataset(train_inputs, train_labels)
    test_ds = CsvDataset(test_inputs, test_labels)
    seq_len = 3
    return train_ds, test_ds, vocab_size, output_dim, seq_len
