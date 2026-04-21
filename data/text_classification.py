"""
Text classification dataset for grokking experiments.

Uses the TREC question classification dataset (HuggingFace: CogComp/trec).
6 coarse-grained question categories:
  0=ABBR, 1=ENTY, 2=DESC, 3=HUM, 4=LOC, 5=NUM

Tokenization: character-level with padding/truncation to max_seq_len.
This keeps dependencies minimal (no sentencepiece/BPE tokenizer needed)
and makes the vocab/output_dim setup parallel to modular_arithmetic.py.

Special tokens:
  0:          <PAD>
  1:          <UNK>
  2..N+1:     printable ASCII characters (space, !"#...~)
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

# Printable ASCII chars: space (0x20) through tilde (0x7E) = 95 chars
_PRINTABLE = [chr(i) for i in range(0x20, 0x7F)]

PAD_ID = 0
UNK_ID = 1
CHAR_OFFSET = 2  # char tokens start at index 2

CHAR_TO_ID = {c: i + CHAR_OFFSET for i, c in enumerate(_PRINTABLE)}

NUM_CLASSES = 6   # TREC coarse labels
VOCAB_SIZE = CHAR_OFFSET + len(_PRINTABLE)  # 2 + 95 = 97


def get_vocab_size() -> int:
    return VOCAB_SIZE


def get_num_classes() -> int:
    return NUM_CLASSES


def tokenize(text: str, max_seq_len: int) -> list[int]:
    """
    Character-level tokenize, truncate/pad to max_seq_len.
    Unknown chars map to UNK_ID.
    """
    ids = [CHAR_TO_ID.get(c, UNK_ID) for c in text[:max_seq_len]]
    # Pad to max_seq_len
    ids += [PAD_ID] * (max_seq_len - len(ids))
    return ids


class TextClassificationDataset(Dataset):
    """
    Generic text classification dataset.

    Each item:
      inputs:  LongTensor (max_seq_len,)
      label:   LongTensor scalar
    """

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


def get_trec_datasets(
    max_seq_len: int = 64,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple['TextClassificationDataset', 'TextClassificationDataset']:
    """
    Load TREC question classification from HuggingFace and return
    (train_dataset, test_dataset).

    Args:
        max_seq_len:      Character sequence length after truncation/padding.
        train_fraction:   Fraction of the TREC train split to use as train.
                          The rest becomes our test set (the official TREC test
                          split is tiny at 500 examples, so we split the 5452
                          train examples ourselves for a bigger test set).
        seed:             Random seed for reproducible split.

    Returns:
        (train_dataset, test_dataset)

    Notes:
        - TREC coarse labels: 0=ABBR, 1=ENTY, 2=DESC, 3=HUM, 4=LOC, 5=NUM
        - Vocab is fixed printable ASCII, vocab_size=97 (same as mod-97 by
          coincidence, which keeps model configs comparable).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for TREC. "
            "Install it with: pip install datasets"
        )

    # Load only the train split — TREC's official test split is 500 examples
    # which is too small; we re-split the 5452-example train set instead.
    raw = load_dataset("CogComp/trec", split="train", trust_remote_code=True)

    texts = [ex['text'] for ex in raw]
    labels = [ex['coarse_label'] for ex in raw]

    n = len(texts)
    n_train = int(n * train_fraction)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    def build_tensors(idx_array):
        inputs_list = [tokenize(texts[i], max_seq_len) for i in idx_array]
        labels_list = [labels[i] for i in idx_array]
        return (
            torch.tensor(inputs_list, dtype=torch.long),
            torch.tensor(labels_list, dtype=torch.long),
        )

    train_inputs, train_labels = build_tensors(train_idx)
    test_inputs, test_labels = build_tensors(test_idx)

    return (
        TextClassificationDataset(train_inputs, train_labels),
        TextClassificationDataset(test_inputs, test_labels),
    )


if __name__ == '__main__':
    train_ds, test_ds = get_trec_datasets(max_seq_len=64, train_fraction=0.8)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"Vocab size: {VOCAB_SIZE}, Num classes: {NUM_CLASSES}")

    x, y = train_ds[0]
    print(f"Sample input shape: {x.shape}, label: {y.item()}")
    # Decode first sample back to text
    id_to_char = {v: k for k, v in CHAR_TO_ID.items()}
    decoded = ''.join(id_to_char.get(t.item(), '?') for t in x if t.item() != PAD_ID)
    print(f"Decoded: '{decoded}'")
