"""
RuleTaker dataset loader for grokking experiments.

RuleTaker (Clark et al., 2020) tests logical reasoning:
  context: "Anne is quiet. Kind young things are not smart."
  question: "Bob is kind."
  label: "entailment" or "not-entailment"

depth-1 examples (single-step reasoning) are easiest to grokk.
Use max_context_words to filter to short contexts for easier memorization.

Requirements: pip install datasets transformers
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


class RuleTakerDataset(Dataset):
    def __init__(self, inputs, padding_mask, labels):
        self.inputs = inputs
        self.padding_mask = padding_mask
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.inputs[idx], self.padding_mask[idx], self.labels[idx]


def get_ruletaker_datasets(
    max_examples:      int   = 5000,
    train_fraction:    float = 0.5,
    seed:              int   = 42,
    max_seq_len:       int   = 128,
    tokenizer_name:    str   = 'bert-base-uncased',
    depth:             str   = 'depth-1',
    max_context_words: int   = None,
) -> tuple[RuleTakerDataset, RuleTakerDataset, int, int]:
    """
    Load RuleTaker and return (train_ds, test_ds, vocab_size, num_classes).

    Args:
        max_examples:      Cap total examples after filtering.
        train_fraction:    Fraction for training.
        seed:              RNG seed.
        max_seq_len:       Pad/truncate to this length.
        tokenizer_name:    HuggingFace tokenizer.
        depth:             'depth-0', 'depth-1', 'depth-2', 'depth-3', 'depth-5'
                           or None for all depths.
        max_context_words: Filter to contexts with at most this many words.
                           e.g. 30 for shorter, more memorizable examples.
    """
    fetch = max_examples * 5 if (depth or max_context_words) else max_examples
    raw = load_dataset("tasksource/ruletaker", split=f"train[:{fetch}]")

    if depth is not None:
        raw = raw.filter(lambda ex: ex['config'] == depth)
        print(f"After depth={depth} filter: {len(raw)} examples")

    if max_context_words is not None:
        raw = raw.filter(lambda ex: len(ex['context'].split()) <= max_context_words)
        print(f"After max_context_words={max_context_words} filter: {len(raw)} examples")

    raw = raw.select(range(min(max_examples, len(raw))))
    print(f"Final: {len(raw)} examples | Sample: {raw[0]}")

    texts         = [ex['context'] + ' [SEP] ' + ex['question'] for ex in raw]
    unique_labels = sorted(set(ex['label'] for ex in raw))
    label_map     = {l: i for i, l in enumerate(unique_labels)}
    labels        = [label_map[ex['label']] for ex in raw]
    print(f"Label map: {label_map}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    enc = tokenizer(texts, padding='max_length', truncation=True,
                    max_length=max_seq_len, return_tensors='pt')

    input_ids    = enc['input_ids']
    padding_mask = enc['attention_mask'].eq(0)
    labels_t     = torch.tensor(labels, dtype=torch.long)

    rng     = np.random.default_rng(seed)
    n       = len(texts)
    perm    = rng.permutation(n)
    n_train = int(n * train_fraction)

    train_ds = RuleTakerDataset(input_ids[perm[:n_train]], padding_mask[perm[:n_train]], labels_t[perm[:n_train]])
    test_ds  = RuleTakerDataset(input_ids[perm[n_train:]], padding_mask[perm[n_train:]], labels_t[perm[n_train:]])

    vocab_size  = tokenizer.vocab_size
    num_classes = len(unique_labels)
    print(f"Train={len(train_ds)}, Test={len(test_ds)}, vocab={vocab_size}, classes={num_classes}")

    return train_ds, test_ds, vocab_size, num_classes
