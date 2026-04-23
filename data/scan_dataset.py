"""
SCAN dataset loader for grokking experiments.

SCAN (Simplified version of the CommAI Navigation tasks) maps short
compositional commands to action sequences.

Example:
    input:  "jump twice and walk"
    output: "JUMP JUMP WALK"

Why SCAN is ideal for grokking:
  - Compositional rule-based structure (like modular arithmetic)
  - No surface shortcuts — model must discover composition rules
  - Finite discrete output space
  - Small vocabulary (~20 input words, 6 output tokens)
  - Sharp generalization transitions known to occur

We treat this as SEQUENCE CLASSIFICATION by hashing each unique
action sequence to an integer label. This keeps the setup identical
to modular arithmetic (CrossEntropyLoss, same train.py, same metrics).

Splits available:
  simple        ~16k pairs, random split
  addprim_jump  compositional split: "jump" held out from combinations
  addprim_turn_left  compositional split: "turn left" held out

For grokking we use 'simple' with train_fraction=0.5 to mirror
the modular arithmetic setup.

Tokenization: word-level on the input command.
  - vocab built from training data
  - unknown words map to <UNK>
  - sequences padded/truncated to max_seq_len

Requirements: pip install datasets
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


PAD_ID  = 0
UNK_ID  = 1
VOCAB_OFFSET = 2  # real tokens start at 2


class SCANDataset(Dataset):
    """
    SCAN dataset returning (input_ids, label) pairs.
    Mirrors ModularArithmeticDataset API — no padding mask needed
    since we fix seq_len and pad deterministically.
    """

    def __init__(self, inputs: Tensor, labels: Tensor):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


def get_scan_datasets(
    split:          str   = 'simple',
    train_fraction: float = 0.5,
    seed:           int   = 42,
    max_seq_len:    int   = 16,
) -> tuple[SCANDataset, SCANDataset, int, int]:
    """
    Load SCAN, tokenize word-level, and return (train_ds, test_ds, vocab_size, num_classes).

    Args:
        split:          SCAN split — 'simple', 'addprim_jump', 'addprim_turn_left'
        train_fraction: Fraction of examples for training.
        seed:           RNG seed.
        max_seq_len:    Max input command length in words (padded/truncated).
                        SCAN commands are short — 16 is sufficient for 'simple'.

    Returns:
        (train_ds, test_ds, vocab_size, num_classes)
    """
    from datasets import load_dataset

    # Load SCAN — use the HuggingFace mirror
    raw = load_dataset("scan", split, split="train")

    commands = [ex['commands'] for ex in raw]
    actions  = [ex['actions']  for ex in raw]

    n = len(commands)

    # ------------------------------------------------------------------
    # 1. Build action label mapping (action sequence → integer)
    # ------------------------------------------------------------------
    unique_actions = sorted(set(actions))
    action_to_id   = {a: i for i, a in enumerate(unique_actions)}
    labels         = [action_to_id[a] for a in actions]
    num_classes    = len(unique_actions)

    # ------------------------------------------------------------------
    # 2. Build word vocabulary from commands
    # ------------------------------------------------------------------
    all_words = set()
    for cmd in commands:
        all_words.update(cmd.split())
    word_to_id = {w: i + VOCAB_OFFSET for i, w in enumerate(sorted(all_words))}
    vocab_size = VOCAB_OFFSET + len(word_to_id)

    def tokenize(cmd: str) -> list[int]:
        tokens = [word_to_id.get(w, UNK_ID) for w in cmd.split()[:max_seq_len]]
        tokens += [PAD_ID] * (max_seq_len - len(tokens))
        return tokens

    # ------------------------------------------------------------------
    # 3. Tokenize all commands
    # ------------------------------------------------------------------
    input_ids = torch.tensor([tokenize(c) for c in commands], dtype=torch.long)
    labels_t  = torch.tensor(labels, dtype=torch.long)

    # ------------------------------------------------------------------
    # 4. Shuffle and split
    # ------------------------------------------------------------------
    rng     = np.random.default_rng(seed)
    perm    = rng.permutation(n)
    n_train = int(n * train_fraction)

    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]

    train_ds = SCANDataset(input_ids[train_idx], labels_t[train_idx])
    test_ds  = SCANDataset(input_ids[test_idx],  labels_t[test_idx])

    return train_ds, test_ds, vocab_size, num_classes


if __name__ == '__main__':
    train_ds, test_ds, vocab_size, num_classes = get_scan_datasets()
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"Vocab size: {vocab_size}, Num classes: {num_classes}")
    x, y = train_ds[0]
    print(f"Sample input: {x}, label: {y.item()}")
