"""CSV-backed datasets for grokking experiments.

The expected CSV layout is four columns named:
  word_one, word_two, word_three, word_four

Each row contributes two samples:
  - (word_one, word_two)    -> train input / label
  - (word_three, word_four) -> test input / label

The loader keeps the rest of the training pipeline unchanged by returning
the same dataset shape as the modular-arithmetic loader.
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
    numeric_values: list[int] = []
    all_numeric = True
    for value in values:
        try:
            numeric_values.append(int(value))
        except (TypeError, ValueError):
            all_numeric = False
            break

    if all_numeric:
        vocab_size = max(numeric_values) + 1 if numeric_values else 0
        return {}, vocab_size

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
    if vocab:
        encoded = [vocab[str(value)] for value in values]
    else:
        encoded = [int(value) for value in values]
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(1)


def _encode_labels_with_map(values: list[object], label_map: dict[str, int]) -> torch.Tensor:
    encoded = [label_map[str(value)] for value in values]
    return torch.tensor(encoded, dtype=torch.long)


def get_csv_datasets(csv_path: str | Path) -> tuple[CsvDataset, CsvDataset, int, int, int]:
    """Load a CSV where each row contributes one train sample and one test sample."""
    frame = _read_csv(csv_path)

    train_inputs_raw = frame['word_one'].tolist()
    train_labels_raw = frame['word_two'].tolist()
    test_inputs_raw = frame['word_three'].tolist()
    test_labels_raw = frame['word_four'].tolist()

    all_input_values = train_inputs_raw + test_inputs_raw
    all_label_values = train_labels_raw + test_labels_raw

    input_vocab, vocab_size = _build_input_vocab(all_input_values)
    label_map, output_dim = _build_label_map(all_label_values)

    train_inputs = _encode_inputs_with_vocab(train_inputs_raw, input_vocab)
    test_inputs = _encode_inputs_with_vocab(test_inputs_raw, input_vocab)

    train_labels = _encode_labels_with_map(train_labels_raw, label_map)
    test_labels = _encode_labels_with_map(test_labels_raw, label_map)

    train_ds = CsvDataset(train_inputs, train_labels)
    test_ds = CsvDataset(test_inputs, test_labels)
    seq_len = 1
    return train_ds, test_ds, vocab_size, output_dim, seq_len
