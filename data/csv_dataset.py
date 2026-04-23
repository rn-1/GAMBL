"""CSV-backed analogy datasets for grokking experiments.

The expected CSV layout is four columns named:
    word_one, word_two, word_three, word_four

Each row contributes one generative analogy sample:
    prompt:  "word_one:word_two::word_three->"
    target:  "word_four"

This represents: "word_one relates to word_two as word_three relates to ?".
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ('word_one', 'word_two', 'word_three', 'word_four')
PAD_ID = 0
EOS_ID = 1
IGNORE_INDEX = -100


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


def _build_char_vocab(texts: list[str]) -> tuple[dict[str, int], int]:
    vocab: OrderedDict[str, int] = OrderedDict()
    next_id = 2  # 0=<PAD>, 1=<EOS>
    for text in texts:
        for ch in text:
            if ch not in vocab:
                vocab[ch] = next_id
                next_id += 1
    return dict(vocab), next_id


def _encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab[ch] for ch in text]


def _pad_2d(seqs: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def get_csv_datasets(
    csv_path: str | Path,
    train_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[CsvDataset, CsvDataset, int, int, int]:
    """Load CSV analogies for sequence generation and split by train_fraction."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    frame = _read_csv(csv_path)

    prompts = [
        f"{str(a)}:{str(b)}::{str(c)}->"
        for a, b, c in zip(frame['word_one'], frame['word_two'], frame['word_three'])
    ]
    targets = [str(v) for v in frame['word_four'].tolist()]

    vocab, vocab_size = _build_char_vocab(prompts + targets)

    # Build autoregressive training sequences.
    # sequence = prompt + target + <EOS>
    # input_ids = sequence[:-1], labels = sequence[1:]
    # Ignore loss on prompt-prediction positions so we train relation transfer.
    input_seqs: list[list[int]] = []
    label_seqs: list[list[int]] = []
    for prompt, target in zip(prompts, targets):
        prompt_ids = _encode_text(prompt, vocab)
        target_ids = _encode_text(target, vocab)
        full_ids = prompt_ids + target_ids + [EOS_ID]
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        prompt_cutoff = max(0, len(prompt_ids) - 1)
        for i in range(prompt_cutoff):
            labels[i] = IGNORE_INDEX

        input_seqs.append(input_ids)
        label_seqs.append(labels)

    inputs = _pad_2d(input_seqs, PAD_ID)
    labels = _pad_2d(label_seqs, IGNORE_INDEX)

    n = len(input_seqs)
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
    train_ds.padding_mask = train_inputs.eq(PAD_ID)
    test_ds.padding_mask = test_inputs.eq(PAD_ID)

    output_dim = vocab_size
    seq_len = inputs.shape[1]
    return train_ds, test_ds, vocab_size, output_dim, seq_len
