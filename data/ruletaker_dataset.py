"""
RuleTaker dataset loader for grokking experiments.

RuleTaker (Clark et al., 2020) tests logical reasoning:
  context: "Anne is quiet. Kind young things are not smart."
  question: "Bob is kind."
  label: "entailment" or "not-entailment"

The model must apply rules to determine if the question follows
from the context — rule-based logical inference with no surface shortcuts.

We cap at max_examples to keep dataset size comparable to modular arithmetic.
depth-1 examples (simple single-step reasoning) are easiest to grokk.

Requirements: pip install datasets transformers
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


class RuleTakerDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, padding_mask: torch.Tensor, labels: torch.Tensor):
        self.inputs = inputs
        self.padding_mask = padding_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.padding_mask[idx], self.labels[idx]


def get_ruletaker_datasets(
    max_examples:   int   = 5000,
    train_fraction: float = 0.5,
    seed:           int   = 42,
    max_seq_len:    int   = 128,
    tokenizer_name: str   = 'bert-base-uncased',
    depth:          str   = 'depth-1',   # filter to single-step reasoning only
) -> tuple[RuleTakerDataset, RuleTakerDataset, int, int]:
    """
    Load RuleTaker and return (train_ds, test_ds, vocab_size, num_classes).

    Args:
        max_examples:   Cap total examples (keep dataset small for grokking).
        train_fraction: Fraction for training.
        seed:           RNG seed.
        max_seq_len:    Pad/truncate to this length.
        tokenizer_name: HuggingFace tokenizer.
        depth:          Filter to this reasoning depth. 'depth-1' = single-step,
                        easiest to grokk. Set to None for all depths.
    """
    # Load a capped slice to avoid downloading all 480k rows
    raw = load_dataset("tasksource/ruletaker", split=f"train[:{max_examples * 3}]")

    # Filter to specified depth if requested
    if depth is not None:
        raw = raw.filter(lambda ex: ex['config'] == depth)

    # Cap after filtering
    raw = raw.select(range(min(max_examples, len(raw))))

    print(f"Loaded {len(raw)} RuleTaker examples (depth={depth})")
    print(f"Sample: {raw[0]}")

    # Combine context + question as input
    texts  = [ex['context'] + ' [SEP] ' + ex['question'] for ex in raw]

    # Map string labels to int
    unique_labels = sorted(set(ex['label'] for ex in raw))
    label_map     = {l: i for i, l in enumerate(unique_labels)}
    labels        = [label_map[ex['label']] for ex in raw]
    print(f"Label map: {label_map}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    enc = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_seq_len,
        return_tensors='pt',
    )

    input_ids    = enc['input_ids']
    padding_mask = enc['attention_mask'].eq(0)   # True = PAD
    labels_t     = torch.tensor(labels, dtype=torch.long)

    # Shuffle and split
    rng     = np.random.default_rng(seed)
    n       = len(texts)
    perm    = rng.permutation(n)
    n_train = int(n * train_fraction)

    train_ds = RuleTakerDataset(
        input_ids[perm[:n_train]],
        padding_mask[perm[:n_train]],
        labels_t[perm[:n_train]],
    )
    test_ds = RuleTakerDataset(
        input_ids[perm[n_train:]],
        padding_mask[perm[n_train:]],
        labels_t[perm[n_train:]],
    )

    vocab_size  = tokenizer.vocab_size
    num_classes = len(unique_labels)
    print(f"Train={len(train_ds)}, Test={len(test_ds)}, vocab={vocab_size}, classes={num_classes}")

    return train_ds, test_ds, vocab_size, num_classes
