"""
SCAN dataset generator for grokking experiments.

Generates SCAN directly from the grammar — no HuggingFace dependency.

SCAN maps short compositional commands to action sequences:
  "jump twice and walk" → "JUMP JUMP WALK"

Why SCAN is ideal for grokking:
  - Compositional rule-based structure (like modular arithmetic)
  - No surface shortcuts — model must discover composition rules
  - Finite discrete output space (~200-300 unique action sequences)
  - Sharp generalization transitions known to occur
"""

from __future__ import annotations
from itertools import product

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


PAD_ID       = 0
UNK_ID       = 1
VOCAB_OFFSET = 2


# ---------------------------------------------------------------------------
# SCAN grammar interpreter
# ---------------------------------------------------------------------------

def _interpret_primitive(prim: str) -> list[str]:
    mapping = {
        'walk':              ['WALK'],
        'run':               ['RUN'],
        'jump':              ['JUMP'],
        'look':              ['LOOK'],
        'turn left':         ['LTURN'],
        'turn right':        ['RTURN'],
        'turn around left':  ['LTURN', 'LTURN', 'LTURN', 'LTURN'],
        'turn around right': ['RTURN', 'RTURN', 'RTURN', 'RTURN'],
    }
    return mapping[prim]


def _interpret_single(cmd: str) -> list[str]:
    for prim in [
        'turn around left', 'turn around right',
        'turn left', 'turn right',
        'walk', 'run', 'jump', 'look',
    ]:
        if cmd.startswith(prim):
            rest = cmd[len(prim):].strip()
            actions = _interpret_primitive(prim)
            if rest == 'twice':
                return actions * 2
            elif rest == 'thrice':
                return actions * 3
            elif rest == '':
                return actions
    raise ValueError(f"Cannot interpret: {cmd!r}")


def _interpret(cmd: str) -> list[str]:
    if ' after ' in cmd:
        parts = cmd.split(' after ', 1)
        return _interpret(parts[1].strip()) + _interpret(parts[0].strip())
    if ' and ' in cmd:
        parts = cmd.split(' and ', 1)
        return _interpret(parts[0].strip()) + _interpret(parts[1].strip())
    return _interpret_single(cmd)


def generate_scan_pairs() -> list[tuple[str, str]]:
    primitives = [
        'walk', 'run', 'jump', 'look',
        'turn left', 'turn right',
        'turn around left', 'turn around right',
    ]
    adverbs   = ['', 'twice', 'thrice']
    conjuncts = ['and', 'after']

    singles = []
    for prim in primitives:
        for adv in adverbs:
            cmd = f"{prim} {adv}".strip()
            singles.append(cmd)

    pairs = []
    seen  = set()

    for cmd in singles:
        key = (cmd, ' '.join(_interpret(cmd)))
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    for s1, s2, conj in product(singles, singles, conjuncts):
        cmd = f"{s1} {conj} {s2}"
        try:
            actions = _interpret(cmd)
            key = (cmd, ' '.join(actions))
            if key not in seen:
                seen.add(key)
                pairs.append(key)
        except ValueError:
            pass

    return pairs


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SCANDataset(Dataset):
    def __init__(self, inputs: Tensor, labels: Tensor):
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def get_scan_datasets(
    split:          str   = 'simple',
    train_fraction: float = 0.5,
    seed:           int   = 42,
    max_seq_len:    int   = 16,
) -> tuple[SCANDataset, SCANDataset, int, int]:
    """
    Generate SCAN and return (train_ds, test_ds, vocab_size, num_classes).
    No HuggingFace required — generated from grammar directly.
    """
    pairs    = generate_scan_pairs()
    commands = [p[0] for p in pairs]
    actions  = [p[1] for p in pairs]

    unique_actions = sorted(set(actions))
    action_to_id   = {a: i for i, a in enumerate(unique_actions)}
    labels         = [action_to_id[a] for a in actions]
    num_classes    = len(unique_actions)

    all_words  = set()
    for cmd in commands:
        all_words.update(cmd.split())
    word_to_id = {w: i + VOCAB_OFFSET for i, w in enumerate(sorted(all_words))}
    vocab_size = VOCAB_OFFSET + len(word_to_id)

    def tokenize(cmd: str) -> list[int]:
        tokens = [word_to_id.get(w, UNK_ID) for w in cmd.split()[:max_seq_len]]
        tokens += [PAD_ID] * (max_seq_len - len(tokens))
        return tokens

    input_ids = torch.tensor([tokenize(c) for c in commands], dtype=torch.long)
    labels_t  = torch.tensor(labels, dtype=torch.long)

    rng     = np.random.default_rng(seed)
    n       = len(commands)
    perm    = rng.permutation(n)
    n_train = int(n * train_fraction)

    train_ds = SCANDataset(input_ids[perm[:n_train]], labels_t[perm[:n_train]])
    test_ds  = SCANDataset(input_ids[perm[n_train:]], labels_t[perm[n_train:]])

    print(f"SCAN: {n} pairs, {num_classes} classes, vocab={vocab_size}")
    return train_ds, test_ds, vocab_size, num_classes


if __name__ == '__main__':
    tr, te, v, c = get_scan_datasets()
    print(f"Train={len(tr)}, Test={len(te)}, Vocab={v}, Classes={c}")
    x, y = tr[0]
    print(f"Input: {x}, Label: {y.item()}")
