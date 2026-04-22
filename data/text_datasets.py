"""
Text classification datasets from HuggingFace for grokking experiments.

Each dataset is loaded from the HuggingFace `train` split, shuffled, then
split by `train_fraction` — mirroring the modular-arithmetic setup so the
same sweep infrastructure works unchanged.

Supported datasets
------------------
  rte      GLUE RTE         ~2.5 k train   2-class NLI
  mrpc     GLUE MRPC        ~3.7 k train   2-class paraphrase detection
  cola     GLUE CoLA        ~8.5 k train   2-class grammatical acceptability
  sst2     GLUE SST-2       ~67  k train   2-class sentiment  (subsample w/ max_dataset_size)
  boolq    BoolQ            ~9.3 k train   2-class yes/no QA
  ag_news  AG News          ~120 k train   4-class topic  (needs max_dataset_size ≤ 10000)

Tokenization
------------
Uses a HuggingFace AutoTokenizer (default: bert-base-uncased, vocab_size=30522).
Sequences are padded/truncated to max_seq_len.  The returned padding_mask is a
boolean tensor (True = PAD token) matching PyTorch MultiheadAttention's
key_padding_mask convention.

Input format
------------
  Single text :  [CLS] text [SEP] <PAD>...
  Text pair   :  [CLS] text_a [SEP] text_b [SEP] <PAD>...

Requirements: pip install datasets transformers
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict] = {
    'rte': {
        'hf_path':   'glue',
        'hf_config': 'rte',
        'text_a':    'sentence1',
        'text_b':    'sentence2',
        'label':     'label',
        'num_classes': 2,
        'label_names': ['entailment', 'not_entailment'],
    },
    'mrpc': {
        'hf_path':   'glue',
        'hf_config': 'mrpc',
        'text_a':    'sentence1',
        'text_b':    'sentence2',
        'label':     'label',
        'num_classes': 2,
        'label_names': ['not_paraphrase', 'paraphrase'],
    },
    'cola': {
        'hf_path':   'glue',
        'hf_config': 'cola',
        'text_a':    'sentence',
        'text_b':    None,
        'label':     'label',
        'num_classes': 2,
        'label_names': ['unacceptable', 'acceptable'],
    },
    'sst2': {
        'hf_path':   'glue',
        'hf_config': 'sst2',
        'text_a':    'sentence',
        'text_b':    None,
        'label':     'label',
        'num_classes': 2,
        'label_names': ['negative', 'positive'],
    },
    'boolq': {
        'hf_path':   'google/boolq',
        'hf_config': None,
        'text_a':    'question',
        'text_b':    'passage',
        'label':     'answer',
        'num_classes': 2,
        'label_names': ['no', 'yes'],
        'label_transform': int,  # bool → 0/1
    },
    'ag_news': {
        'hf_path':   'ag_news',
        'hf_config': None,
        'text_a':    'text',
        'text_b':    None,
        'label':     'label',
        'num_classes': 4,
        'label_names': ['world', 'sports', 'business', 'sci_tech'],
    },
}


def list_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Text classification dataset that mirrors the ModularArithmeticDataset API.

    Each item returns (input_ids, padding_mask, label) so the training loop
    can tell it apart from the 2-tuple modular-arithmetic batches.

    Attributes:
        inputs:       (N, seq_len)  LongTensor of token IDs
        padding_mask: (N, seq_len)  BoolTensor — True = PAD token (ignored in attention)
        labels:       (N,)          LongTensor of class indices
    """

    def __init__(
        self,
        inputs:       Tensor,
        padding_mask: Tensor,
        labels:       Tensor,
    ):
        assert inputs.shape[0] == padding_mask.shape[0] == labels.shape[0]
        self.inputs       = inputs
        self.padding_mask = padding_mask
        self.labels       = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.padding_mask[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def get_text_datasets(
    dataset_name:    str,
    train_fraction:  float = 0.5,
    seed:            int   = 42,
    tokenizer_name:  str   = 'bert-base-uncased',
    max_seq_len:     int   = 128,
    max_dataset_size: int  = -1,
) -> tuple[TextDataset, TextDataset, int, int]:
    """
    Load, tokenize, and split a text classification dataset.

    Args:
        dataset_name:     Key in DATASET_REGISTRY (e.g. 'rte', 'cola').
        train_fraction:   Fraction of shuffled examples used for training.
        seed:             RNG seed for reproducible shuffle and split.
        tokenizer_name:   HuggingFace tokenizer name or local path.
        max_seq_len:      Pad/truncate all sequences to this length.
        max_dataset_size: If > 0, cap total examples before splitting.
                          Useful for large datasets like ag_news (120k).

    Returns:
        (train_ds, test_ds, vocab_size, num_classes)
    """
    # Lazy imports so the rest of the repo works without these packages
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list_datasets()}"
        )

    cfg = DATASET_REGISTRY[dataset_name]

    # ------------------------------------------------------------------
    # 1. Load HuggingFace train split
    # ------------------------------------------------------------------
    hf_ds = load_dataset(
        cfg['hf_path'],
        cfg['hf_config'],
        split='train',
        trust_remote_code=False,
    )

    # Cap and shuffle
    n_total = len(hf_ds)
    if max_dataset_size > 0 and n_total > max_dataset_size:
        n_total = max_dataset_size

    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(len(hf_ds))[:n_total]
    hf_ds = hf_ds.select(perm.tolist())

    # ------------------------------------------------------------------
    # 2. Tokenize
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    field_a = cfg['text_a']
    field_b = cfg['text_b']
    texts_a = [str(ex[field_a]) for ex in hf_ds]
    texts_b = [str(ex[field_b]) for ex in hf_ds] if field_b else None

    label_transform = cfg.get('label_transform', lambda x: x)
    labels = [label_transform(ex[cfg['label']]) for ex in hf_ds]

    enc_kwargs = dict(
        padding='max_length',
        truncation=True,
        max_length=max_seq_len,
        return_tensors='pt',
    )

    if texts_b is not None:
        encoding = tokenizer(texts_a, texts_b, **enc_kwargs)
    else:
        encoding = tokenizer(texts_a, **enc_kwargs)

    input_ids    = encoding['input_ids']                # (N, max_seq_len)
    attn_mask    = encoding['attention_mask']           # (N, max_seq_len) 1=real 0=pad
    padding_mask = attn_mask.eq(0)                      # True = PAD (PyTorch convention)
    labels_t     = torch.tensor(labels, dtype=torch.long)

    # ------------------------------------------------------------------
    # 3. Train / test split (mirrors modular arithmetic)
    # ------------------------------------------------------------------
    n        = n_total
    n_train  = int(n * train_fraction)

    train_ds = TextDataset(
        input_ids[:n_train],
        padding_mask[:n_train],
        labels_t[:n_train],
    )
    test_ds  = TextDataset(
        input_ids[n_train:],
        padding_mask[n_train:],
        labels_t[n_train:],
    )

    vocab_size  = tokenizer.vocab_size
    num_classes = cfg['num_classes']

    return train_ds, test_ds, vocab_size, num_classes
