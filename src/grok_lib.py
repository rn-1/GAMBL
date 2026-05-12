"""Shared library for GAMBL grokking experiments.

All models, dataset loading, training loop, and analysis helpers live here so
the parallel sweep driver and the results notebook can import from one place.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Transformer (Power et al. 2022 style: 2-layer pre-norm encoder)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class GrokTransformer(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
        n_layers: int = 2, d_ff: int = 512, output_dim: int = 2,
        max_seq_len: int = 128, dropout: float = 0.1, pool: str = 'mean',
    ):
        super().__init__()
        self.pool = pool
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.dropout(self.token_embedding(x) + self.pos_embedding(positions))
        for block in self.blocks:
            h = block(h, padding_mask=padding_mask)
        h = self.norm(h)
        if self.pool == 'cls':
            pooled = h[:, 0, :]
        elif self.pool == 'last':
            pooled = h[:, -1, :]
        else:
            if padding_mask is not None:
                valid = (~padding_mask).float().unsqueeze(-1)
                pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            else:
                pooled = h.mean(dim=1)
        return self.head(pooled)

    @torch.no_grad()
    def collect_activations(self, x: Tensor, padding_mask: Tensor = None) -> list[Tensor]:
        """Return pooled hidden states from embedding layer + each transformer
        block (one vector per sample per layer).

        Used for dense-analogue pathway metrics (Li et al. §4). We include the
        embedding as layer 0 so models with only a few blocks still produce
        enough layer transitions for the consistency metric to be meaningful.
        Each element is a (B, d_model) tensor of mean-pooled hidden states.
        """
        was_training = self.training
        self.eval()
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.token_embedding(x) + self.pos_embedding(positions)

        def _pool(hidden):
            if padding_mask is not None:
                valid = (~padding_mask).float().unsqueeze(-1)
                return (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            return hidden.mean(dim=1)

        acts = [_pool(h)]  # embedding layer
        for block in self.blocks:
            h = block(h, padding_mask=padding_mask)
            acts.append(_pool(h))
        if was_training:
            self.train()
        return acts


# ---------------------------------------------------------------------------
# MLP baseline (no attention)
# ---------------------------------------------------------------------------

_ACTIVATIONS = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}


class MLP(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 512,
        num_layers: int = 3, output_dim: int = 2, activation: str = 'relu',
        dropout: float = 0.0, seq_len: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        act_cls = _ACTIVATIONS[activation]
        # Keep hidden-layer blocks separately so collect_activations() can
        # record the post-activation state of each hidden layer.
        self.hidden_blocks = nn.ModuleList()
        in_dim = embed_dim * seq_len
        for _ in range(num_layers - 1):
            block = [nn.Linear(in_dim, hidden_dim), act_cls()]
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            self.hidden_blocks.append(nn.Sequential(*block))
            in_dim = hidden_dim
        self.out_head = nn.Linear(in_dim, output_dim)

    def forward(self, x: Tensor, padding_mask: Tensor = None) -> Tensor:
        h = self.embedding(x).flatten(start_dim=1)
        for block in self.hidden_blocks:
            h = block(h)
        return self.out_head(h)

    @torch.no_grad()
    def collect_activations(self, x: Tensor, padding_mask: Tensor = None) -> list[Tensor]:
        """Return per-layer activations, starting with the flattened embedding.

        For dense-analogue pathway metrics (Li et al. §4). The flattened
        embedding is included as layer 0 (dim=embed_dim*seq_len); subsequent
        layers have dim=hidden_dim. Downstream pathway metrics handle
        dim mismatches by truncating to the smallest common dim.
        """
        was_training = self.training
        self.eval()
        h = self.embedding(x).flatten(start_dim=1)
        acts = [h]
        for block in self.hidden_blocks:
            h = block(h)
            acts.append(h)
        if was_training:
            self.train()
        return acts


def _apply_init_scale(model: nn.Module, scale: float) -> None:
    """Omnigrok-style weight rescaling (Liu et al. 2022).

    Multiplies every weight matrix (tensors with >=2 dims — Linear, Embedding,
    MultiheadAttention in_proj_weight) by ``scale``. Biases and LayerNorm
    gammas (1D) are left alone so activation magnitudes stay sane. Liu et al.
    showed that large-norm initialization can dramatically accelerate — or
    outright induce — grokking on tasks that don't otherwise grok.
    """
    if scale == 1.0:
        return
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.mul_(scale)


def build_model(
    arch: str, vocab_size: int, num_classes: int, max_seq_len: int,
    *,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 512,
    dropout: float = 0.1,
    init_scale: float = 1.0,
) -> nn.Module:
    """Build a model with configurable width/depth/regularization.

    For MLP: ``d_model`` -> embed_dim, ``d_ff`` -> hidden_dim, ``n_layers`` ->
    num_layers. ``n_heads`` is ignored. This keeps the sweep config uniform
    across architectures.
    """
    if arch == 'transformer':
        model = GrokTransformer(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, output_dim=num_classes,
            max_seq_len=max_seq_len, dropout=dropout, pool='mean',
        )
    elif arch == 'mlp':
        model = MLP(
            vocab_size=vocab_size, embed_dim=d_model, hidden_dim=d_ff,
            num_layers=n_layers, output_dim=num_classes, activation='relu',
            dropout=dropout, seq_len=max_seq_len,
        )
    else:
        raise ValueError(f"unknown arch: {arch}")
    _apply_init_scale(model, init_scale)
    return model


# ---------------------------------------------------------------------------
# Text datasets
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    'rte':   {'hf_path': 'glue', 'hf_config': 'rte',   'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'num_classes': 2},
    'mrpc':  {'hf_path': 'glue', 'hf_config': 'mrpc',  'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'num_classes': 2},
    'cola':  {'hf_path': 'glue', 'hf_config': 'cola',  'text_a': 'sentence',  'text_b': None,        'label': 'label', 'num_classes': 2},
    'sst2':  {'hf_path': 'glue', 'hf_config': 'sst2',  'text_a': 'sentence',  'text_b': None,        'label': 'label', 'num_classes': 2},
    'boolq': {'hf_path': 'google/boolq', 'hf_config': None, 'text_a': 'question', 'text_b': 'passage', 'label': 'answer', 'num_classes': 2, 'label_transform': int},
    'ag_news': {'hf_path': 'ag_news', 'hf_config': None, 'text_a': 'text', 'text_b': None, 'label': 'label', 'num_classes': 4},
}


class TextDataset(Dataset):
    def __init__(self, inputs: Tensor, padding_mask: Tensor, labels: Tensor):
        self.inputs = inputs
        self.padding_mask = padding_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.padding_mask[idx], self.labels[idx]


def get_text_datasets(
    dataset_name: str, train_fraction: float = 0.5, seed: int = 42,
    tokenizer_name: str = 'bert-base-uncased', max_seq_len: int = 128,
    max_dataset_size: int = -1,
):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    cfg = DATASET_REGISTRY[dataset_name]
    hf_ds = load_dataset(cfg['hf_path'], cfg['hf_config'], split='train')

    n_total = len(hf_ds)
    if max_dataset_size > 0 and n_total > max_dataset_size:
        n_total = max_dataset_size

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(hf_ds))[:n_total]
    hf_ds = hf_ds.select(perm.tolist())

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    texts_a = [str(ex[cfg['text_a']]) for ex in hf_ds]
    texts_b = [str(ex[cfg['text_b']]) for ex in hf_ds] if cfg['text_b'] else None
    label_transform = cfg.get('label_transform', lambda x: x)
    labels = [label_transform(ex[cfg['label']]) for ex in hf_ds]

    enc_kwargs = dict(padding='max_length', truncation=True, max_length=max_seq_len, return_tensors='pt')
    encoding = tokenizer(texts_a, texts_b, **enc_kwargs) if texts_b else tokenizer(texts_a, **enc_kwargs)
    input_ids = encoding['input_ids']
    padding_mask = encoding['attention_mask'].eq(0)
    labels_t = torch.tensor(labels, dtype=torch.long)

    n_train = int(n_total * train_fraction)
    train_ds = TextDataset(input_ids[:n_train], padding_mask[:n_train], labels_t[:n_train])
    test_ds = TextDataset(input_ids[n_train:], padding_mask[n_train:], labels_t[n_train:])
    return train_ds, test_ds, tokenizer.vocab_size, cfg['num_classes']


# ---------------------------------------------------------------------------
# Multi-task text dataset (Li et al. §3.2 asynchronous-grokking setup)
#
# Concatenates several GLUE-style binary classification tasks into one
# training corpus. Every example keeps a `task_id` so the training loop can
# eval per-task separately and look for different grokking onsets across
# tasks — the dense-model analogue of their cross-domain pretraining study.
#
# A reserved task-marker token is prepended to each sequence so the model
# can distinguish tasks. Vocab is extended from tokenizer.vocab_size to
# tokenizer.vocab_size + len(tasks).
# ---------------------------------------------------------------------------

class MultiTaskDataset(TextDataset):
    def __init__(self, inputs, padding_mask, labels, task_ids, task_names):
        super().__init__(inputs, padding_mask, labels)
        self.task_ids = task_ids
        self.task_names = list(task_names)


def get_multitask_datasets(
    task_names=('rte', 'mrpc', 'cola', 'boolq'),
    train_fraction: float = 0.5, seed: int = 42,
    tokenizer_name: str = 'bert-base-uncased', max_seq_len: int = 128,
    max_per_task: int = -1,
):
    train_inputs, train_mask, train_labels, train_tids = [], [], [], []
    test_inputs, test_mask, test_labels, test_tids = [], [], [], []
    base_vocab = None

    for tid, name in enumerate(task_names):
        tr, te, vs, nc = get_text_datasets(
            name, train_fraction=train_fraction, seed=seed,
            tokenizer_name=tokenizer_name, max_seq_len=max_seq_len,
            max_dataset_size=max_per_task,
        )
        assert nc == 2, f"multitask currently assumes binary tasks, got {nc} for {name}"
        base_vocab = vs if base_vocab is None else base_vocab
        assert vs == base_vocab, "tokenizer vocab mismatch across tasks"

        # Prepend task-marker token (id = base_vocab + tid), shifting seq by 1 and truncating the last position.
        def _prepend(inputs, mask, tid):
            marker = torch.full((inputs.size(0), 1), base_vocab + tid, dtype=inputs.dtype)
            new_inputs = torch.cat([marker, inputs[:, :-1]], dim=1)
            mark_mask = torch.zeros((mask.size(0), 1), dtype=mask.dtype)
            new_mask = torch.cat([mark_mask, mask[:, :-1]], dim=1)
            return new_inputs, new_mask

        tr_inp, tr_msk = _prepend(tr.inputs, tr.padding_mask, tid)
        te_inp, te_msk = _prepend(te.inputs, te.padding_mask, tid)

        train_inputs.append(tr_inp); train_mask.append(tr_msk); train_labels.append(tr.labels)
        train_tids.append(torch.full((len(tr),), tid, dtype=torch.long))
        test_inputs.append(te_inp); test_mask.append(te_msk); test_labels.append(te.labels)
        test_tids.append(torch.full((len(te),), tid, dtype=torch.long))

    train_inputs = torch.cat(train_inputs, dim=0)
    train_mask = torch.cat(train_mask, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    train_tids = torch.cat(train_tids, dim=0)
    test_inputs = torch.cat(test_inputs, dim=0)
    test_mask = torch.cat(test_mask, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    test_tids = torch.cat(test_tids, dim=0)

    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(train_inputs.size(0))
    train_inputs = train_inputs[perm]; train_mask = train_mask[perm]
    train_labels = train_labels[perm]; train_tids = train_tids[perm]

    train_ds = MultiTaskDataset(train_inputs, train_mask, train_labels, train_tids, task_names)
    test_ds = MultiTaskDataset(test_inputs, test_mask, test_labels, test_tids, task_names)
    return train_ds, test_ds, base_vocab + len(task_names), 2


# ---------------------------------------------------------------------------
# Modular arithmetic (Power et al. 2022 sanity-check task)
#
# Task: given (a, op, b), predict (a op b) mod p. Classification with p classes.
# Input encoding: 3 tokens [a, op_token, b]. vocab_size = p + 1 (digits + op).
# Power et al.'s canonical demo: addition mod 97, train_fraction=0.5, wd=1.0.
# Known to grok in ~10^4 - 10^5 steps with this recipe.
# ---------------------------------------------------------------------------

MOD_ARITH_OPS = ('add', 'mul', 'sub')


def get_modular_arithmetic_datasets(
    p: int = 97, op: str = 'add', train_fraction: float = 0.5, seed: int = 42,
):
    if op not in MOD_ARITH_OPS:
        raise ValueError(f"op must be one of {MOD_ARITH_OPS}, got {op!r}")

    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    if op == 'add':
        c = (a + b) % p
    elif op == 'mul':
        c = (a * b) % p
    elif op == 'sub':
        c = (a - b) % p

    op_token = torch.full_like(a, p)  # single reserved token id = p
    inputs = torch.stack([a, op_token, b], dim=1)  # (p^2, 3)
    padding_mask = torch.zeros_like(inputs, dtype=torch.bool)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(inputs.size(0))
    inputs = inputs[perm]
    padding_mask = padding_mask[perm]
    labels = c[perm]

    n_train = int(inputs.size(0) * train_fraction)
    train_ds = TextDataset(inputs[:n_train], padding_mask[:n_train], labels[:n_train])
    test_ds = TextDataset(inputs[n_train:], padding_mask[n_train:], labels[n_train:])
    return train_ds, test_ds, p + 1, p


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _infinite_loader(dataset: Dataset, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    while True:
        yield from loader


def _accuracy(logits: Tensor, labels: Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def _relative_range(values: list[float]) -> float:
    """``(max - min) / |mean|`` — robust flatness measure. 0 means perfectly
    flat; returns +inf for zero-mean degenerate cases to avoid false kills."""
    if not values:
        return float('inf')
    lo, hi = min(values), max(values)
    mean = sum(values) / len(values)
    if abs(mean) < 1e-8:
        return float('inf')
    return (hi - lo) / abs(mean)


def train_model(
    model: nn.Module,
    train_ds: TextDataset,
    test_ds: TextDataset,
    *,
    n_steps: int,
    batch_size: int = -1,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    betas: tuple = (0.9, 0.98),
    label_smoothing: float = 0.0,
    log_every: int = 100,
    device: torch.device | None = None,
    verbose: bool = True,
    eval_batch_size: int = 2048,
    csv_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 0,
    pathway_every: int = 0,
    pathway_probe_size: int = 32,
    pathway_top_k: int = 8,
    flat_kill_start_after: int = 0,
    flat_kill_window: int = 10,
    flat_kill_min_delta: float = 0.01,
    run_name: str = '',
) -> pd.DataFrame:
    """Train and return per-log-step metrics.

    If ``csv_path`` is given, each log row is flushed to disk immediately so a
    KeyboardInterrupt or crash still leaves a usable partial CSV behind.

    If the dataset is a ``MultiTaskDataset`` (has ``task_ids``/``task_names``),
    per-task test accuracy is logged as additional columns ``test_acc_<task>``
    — this is what enables the Li et al. §3.2 asynchronous-grokking analysis.

    If ``checkpoint_dir`` and ``checkpoint_every > 0`` are given, model
    state_dicts are saved every N steps (plus step 1 and the last step) for
    post-hoc pathway-metric analysis (Li et al. §4).

    If ``pathway_every > 0``, inline pathway metrics (edit distance +
    consistency, plus per-task versions on multitask) are computed on a fixed
    test-set probe every N steps and added to the log rows. This lets the
    run self-monitor the memorization->generalization transition without
    needing a post-hoc analysis pass.

    If ``flat_kill_start_after > 0``, training exits early once both
    pathway_edit_dist and pathway_consistency have shown relative range
    < ``flat_kill_min_delta`` over the last ``flat_kill_window`` pathway
    measurements. The rationale: if the model has stopped reorganizing its
    internal routing long after the kill threshold, it will not grok.
    Attached to the returned DataFrame as ``df.attrs``:
        ``early_killed`` (bool), ``kill_reason`` (str), ``kill_step`` (int).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    effective_bs = len(train_ds) if batch_size == -1 else batch_size
    train_loader = _infinite_loader(train_ds, batch_size=effective_bs)

    def _eval_subset(ds: TextDataset, indices=None):
        """Chunked eval. If `indices` is given, evaluate only those rows."""
        model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0
        crit = nn.CrossEntropyLoss(reduction='sum')
        n = len(ds) if indices is None else len(indices)
        idx_t = torch.arange(len(ds)) if indices is None else torch.as_tensor(indices)
        with torch.no_grad():
            for i in range(0, n, eval_batch_size):
                chunk = idx_t[i:i + eval_batch_size]
                inp = ds.inputs[chunk].to(device)
                mask = ds.padding_mask[chunk].to(device)
                lab = ds.labels[chunk].to(device)
                logits = model(inp, padding_mask=mask)
                total_loss += crit(logits, lab).item()
                total_correct += (logits.argmax(dim=-1) == lab).sum().item()
                total_n += lab.size(0)
        model.train()
        return total_loss / total_n, total_correct / total_n

    task_names = getattr(test_ds, 'task_names', None)
    task_ids_train = getattr(train_ds, 'task_ids', None)
    task_ids_test = getattr(test_ds, 'task_ids', None)
    is_multitask = task_names is not None and task_ids_test is not None

    # Precompute per-task index lists once (constant across training)
    train_task_idx = None
    test_task_idx = None
    if is_multitask:
        train_task_idx = [
            (task_ids_train == tid).nonzero(as_tuple=True)[0] for tid in range(len(task_names))
        ]
        test_task_idx = [
            (task_ids_test == tid).nonzero(as_tuple=True)[0] for tid in range(len(task_names))
        ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    metrics = []
    model.train()
    start = time.time()

    csv_f = None
    csv_columns = None  # written on first log row so schema matches multi-task additions

    ckpt_dir = None
    if checkpoint_dir is not None and checkpoint_every > 0:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(step):
        if ckpt_dir is None:
            return
        torch.save({'step': step, 'state_dict': model.state_dict()}, ckpt_dir / f'step_{step:07d}.pt')

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_f = open(csv_path, 'w', buffering=1)  # line-buffered

    # Fixed probe from test set for inline pathway metrics (stratified per task).
    probe_inputs = probe_mask = None
    probe_task_indices = None  # list of index-tensors (one per task) into the probe
    if pathway_every > 0:
        n_test = len(test_ds)
        if is_multitask:
            per_task = max(pathway_probe_size // len(task_names), 2)
            idx_parts, tid_parts = [], []
            for tid in range(len(task_names)):
                tid_idx = test_task_idx[tid][:per_task]
                idx_parts.append(tid_idx)
                tid_parts.append(torch.full((len(tid_idx),), tid, dtype=torch.long))
            probe_indices = torch.cat(idx_parts)
            probe_tids = torch.cat(tid_parts)
            probe_task_indices = [
                (probe_tids == tid).nonzero(as_tuple=True)[0] for tid in range(len(task_names))
            ]
        else:
            probe_indices = torch.arange(min(pathway_probe_size, n_test))
        probe_inputs = test_ds.inputs[probe_indices].to(device)
        probe_mask = test_ds.padding_mask[probe_indices].to(device)

    # Flat-metric early-kill state.
    edit_hist: list[float] = []
    cons_hist: list[float] = []
    early_killed = False
    kill_reason = ''
    kill_step = None

    try:
        for step in range(1, n_steps + 1):
            inputs, mask, labels = (t.to(device) for t in next(train_loader))
            optimizer.zero_grad()
            loss = criterion(model(inputs, padding_mask=mask), labels)
            loss.backward()
            optimizer.step()

            should_log = (step == 1) or (step % log_every == 0)
            should_pathway = pathway_every > 0 and ((step == 1) or (step % pathway_every == 0))

            if should_log or should_pathway:
                tr_loss, tr_acc = _eval_subset(train_ds)
                te_loss, te_acc = _eval_subset(test_ds)
                row = {
                    'step': step, 'train_loss': tr_loss, 'train_acc': tr_acc,
                    'test_loss': te_loss, 'test_acc': te_acc,
                }
                if is_multitask:
                    for tid, tname in enumerate(task_names):
                        _, ta = _eval_subset(test_ds, test_task_idx[tid])
                        row[f'test_acc_{tname}'] = ta
                        _, tr_a = _eval_subset(train_ds, train_task_idx[tid])
                        row[f'train_acc_{tname}'] = tr_a

                if should_pathway:
                    ed = pathway_edit_distance(model, probe_inputs, probe_mask, top_k=pathway_top_k, device=device)
                    cs = pathway_consistency(model, probe_inputs, probe_mask, device=device)
                    row['pathway_edit_dist'] = ed
                    row['pathway_consistency'] = cs
                    edit_hist.append(ed)
                    cons_hist.append(cs)
                    if is_multitask and probe_task_indices is not None:
                        for tid, tname in enumerate(task_names):
                            sub_idx = probe_task_indices[tid]
                            if sub_idx.numel() < 2:
                                continue
                            sub_inp = probe_inputs[sub_idx]
                            sub_msk = probe_mask[sub_idx]
                            row[f'edit_dist_{tname}'] = pathway_edit_distance(
                                model, sub_inp, sub_msk, top_k=pathway_top_k, device=device,
                            )
                            row[f'consistency_{tname}'] = pathway_consistency(
                                model, sub_inp, sub_msk, device=device,
                            )
                elif pathway_every > 0:
                    # keep CSV schema stable
                    row['pathway_edit_dist'] = None
                    row['pathway_consistency'] = None
                    if is_multitask:
                        for tname in task_names:
                            row[f'edit_dist_{tname}'] = None
                            row[f'consistency_{tname}'] = None

                metrics.append(row)

                if csv_f is not None:
                    if csv_columns is None:
                        csv_columns = list(row.keys())
                        csv_f.write(','.join(csv_columns) + '\n')
                    csv_f.write(','.join(
                        '' if row.get(k) is None else str(row[k]) for k in csv_columns
                    ) + '\n')

                if verbose and (step == 1 or step % (log_every * 10) == 0):
                    elapsed = time.time() - start
                    tail = ''
                    if is_multitask:
                        tail = ' | per-task test: ' + ' '.join(
                            f"{tn}={row[f'test_acc_{tn}']:.3f}" for tn in task_names
                        )
                    pwy_tail = ''
                    if should_pathway:
                        pwy_tail = f" | edit={row['pathway_edit_dist']:.2f} cons={row['pathway_consistency']:.4f}"
                    tag = f"[{run_name}] " if run_name else '  '
                    print(f"{tag}step={step:6d} | train {tr_loss:.4f}/{tr_acc:.4f} | "
                          f"test {te_loss:.4f}/{te_acc:.4f} | {elapsed:.0f}s{tail}{pwy_tail}", flush=True)

                # Flat-metric early-kill (applied only at pathway-update steps so
                # we actually have fresh measurements to judge on).
                if (
                    should_pathway
                    and flat_kill_start_after > 0
                    and step >= flat_kill_start_after
                    and len(edit_hist) >= flat_kill_window
                    and len(cons_hist) >= flat_kill_window
                ):
                    edit_range = _relative_range(edit_hist[-flat_kill_window:])
                    cons_range = _relative_range(cons_hist[-flat_kill_window:])
                    if edit_range < flat_kill_min_delta and cons_range < flat_kill_min_delta:
                        early_killed = True
                        kill_step = step
                        kill_reason = (
                            f"pathway metrics flat at step {step}: "
                            f"edit rel_range={edit_range:.4f} cons rel_range={cons_range:.4f} "
                            f"< min_delta={flat_kill_min_delta} over last {flat_kill_window} measurements"
                        )
                        tag = f"[{run_name}] " if run_name else ''
                        print(f"{tag}EARLY_KILL: {kill_reason}", flush=True)
                        break

            if ckpt_dir is not None and (step == 1 or step % checkpoint_every == 0 or step == n_steps):
                _save_checkpoint(step)
    finally:
        if csv_f is not None:
            csv_f.close()

    df = pd.DataFrame(metrics)
    df.attrs['early_killed'] = early_killed
    df.attrs['kill_reason'] = kill_reason
    df.attrs['kill_step'] = kill_step
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def find_grokking_step(df: pd.DataFrame, threshold: float = 0.75) -> int | None:
    hits = df[df['test_acc'] >= threshold]
    return int(hits.iloc[0]['step']) if not hits.empty else None


def find_memorization_step(df: pd.DataFrame, threshold: float = 0.99) -> int | None:
    hits = df[df['train_acc'] >= threshold]
    return int(hits.iloc[0]['step']) if not hits.empty else None


def plot_grokking_curve(
    df: pd.DataFrame, title: str = '', log_x: bool = True,
    grok_threshold: float = 0.75, mem_threshold: float = 0.99,
    save_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    ax1.plot(df['step'], df['train_acc'], label='Train', color='steelblue', linewidth=1.6)
    ax1.plot(df['step'], df['test_acc'], label='Test', color='darkorange', linewidth=1.6)
    mem = find_memorization_step(df, mem_threshold)
    grok = find_grokking_step(df, grok_threshold)
    if mem is not None:
        ax1.axvline(mem, color='blue', linestyle=':', alpha=0.6, label=f'Mem @ {mem:,}')
    if grok is not None:
        ax1.axvline(grok, color='green', linestyle='--', alpha=0.7, label=f'Grok @ {grok:,}')
    ax1.set_xlabel('Training steps' + (' (log)' if log_x else ''))
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(-0.05, 1.05)
    if log_x:
        ax1.set_xscale('log')
    ax1.legend(fontsize=9)
    ax1.set_title(f'{title} — Accuracy')
    ax1.grid(True, alpha=0.3)

    ax2.plot(df['step'], df['train_loss'], label='Train', color='steelblue', linewidth=1.6)
    ax2.plot(df['step'], df['test_loss'], label='Test', color='darkorange', linewidth=1.6)
    ax2.set_xlabel('Training steps' + (' (log)' if log_x else ''))
    ax2.set_ylabel('Cross-entropy loss')
    if log_x:
        ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.set_title(f'{title} — Loss')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def plot_comparison(
    results: dict[str, pd.DataFrame], metric: str = 'test_acc',
    title: str = '', log_x: bool = True, save_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10
    for i, (name, df) in enumerate(results.items()):
        ax.plot(df['step'], df[metric], label=name, color=cmap(i % 10), linewidth=1.5)
    ax.set_xlabel('Training steps' + (' (log)' if log_x else ''))
    ax.set_ylabel(metric.replace('_', ' ').title())
    if 'acc' in metric:
        ax.set_ylim(-0.05, 1.05)
    if log_x:
        ax.set_xscale('log')
    ax.legend(fontsize=8, loc='best')
    ax.set_title(title or metric)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Dense-analogue pathway metrics (Li et al. §4 applied to dense models)
#
# Li et al. define pathways in MoE as the sequence of expert ids chosen at
# each layer. They call for extending this to dense models via "virtual
# pathways" (conclusion, p.10). Our construction:
#
#   - pathway(x)  := at each layer, the sorted indices of the top-k neurons
#                    (or hidden-state components) with the largest absolute
#                    activation, concatenated across layers.
#                 Matches Li et al. §4.1 — a discrete per-layer selection
#                 sequence whose edit distance measures structural similarity.
#
#   - consistency(x) := 1 - mean over consecutive layer pairs of
#                       cos(h_l, h_l+1) / (max_l cos(h_l, h_l+1) + eps).
#                 Matches Li et al. §4.2 — lower value = smoother cross-layer
#                 transitions. We need per-layer hidden states of matching
#                 dim; we project all layers to a common dim via PCA on
#                 activations for MLP (where layer widths differ) or use raw
#                 vectors when dims already match (transformer).
#
#   - effective_rank(W) := exp(-sum(p_i log p_i)) where p_i are normalized
#                 singular values. Dense analogue of Li et al. §4.4's
#                 effective dimension of the routing kernel (Liu et al. 2022
#                 "Omnigrok" observed that weight-matrix effective rank drops
#                 as grokking fires — our theoretical hook).
# ---------------------------------------------------------------------------

def _pathway_string(acts: list[Tensor], sample_idx: int, top_k: int = 8) -> str:
    """Build Li-et-al-style pathway string for one sample: per-layer top-k
    neuron indices, joined with hyphens."""
    layers = []
    for h in acts:
        row = h[sample_idx].abs()
        top = torch.topk(row, k=min(top_k, row.numel())).indices.tolist()
        layers.append(','.join(str(i) for i in sorted(top)))
    return '-'.join(layers)


def _levenshtein(a: str, b: str) -> int:
    """Edit distance over character sequences (Li et al. use this over their
    pathway strings)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr[j] = min(ins, dele, sub)
        prev = curr
    return prev[-1]


@torch.no_grad()
def pathway_edit_distance(
    model: nn.Module, inputs: Tensor, padding_mask: Tensor = None,
    top_k: int = 8, device: torch.device | None = None,
) -> float:
    """Average pairwise Levenshtein edit distance between pathway strings over
    the given probe inputs. Dense analogue of Li et al. §4.1 pathway distance.
    """
    if device is None:
        device = next(model.parameters()).device
    inputs = inputs.to(device)
    mask = padding_mask.to(device) if padding_mask is not None else None
    acts = model.collect_activations(inputs, padding_mask=mask)
    N = inputs.size(0)
    strs = [_pathway_string(acts, i, top_k=top_k) for i in range(N)]
    total, pairs = 0, 0
    for i in range(N):
        for j in range(i + 1, N):
            total += _levenshtein(strs[i], strs[j])
            pairs += 1
    return total / max(pairs, 1)


@torch.no_grad()
def pathway_consistency(
    model: nn.Module, inputs: Tensor, padding_mask: Tensor = None,
    device: torch.device | None = None, eps: float = 1e-8,
) -> float:
    """Mean single-sample pathway consistency (Li et al. §4.2) across probe
    inputs. Lower values = smoother cross-layer transitions = "more coherent
    routing"."""
    if device is None:
        device = next(model.parameters()).device
    inputs = inputs.to(device)
    mask = padding_mask.to(device) if padding_mask is not None else None
    acts = model.collect_activations(inputs, padding_mask=mask)
    # Project all layers to a common dim if they differ (MLP case).
    target_dim = min(h.shape[1] for h in acts)
    projected = [h[:, :target_dim] for h in acts]  # simple truncation — common dim via top components
    # Stack: (N, L, D)
    H = torch.stack(projected, dim=1)
    N, L, D = H.shape
    if L < 2:
        return 0.0
    # cosine similarity between consecutive layer pairs per sample
    a = H[:, :-1, :]
    b = H[:, 1:, :]
    cos = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)
    # Li et al.: C_i = 1 - mean_l cos(h_l, h_l+1) / (max_l cos + eps)
    max_per_sample = cos.max(dim=-1).values.clamp(min=eps)
    norm_cos = cos / max_per_sample.unsqueeze(-1)
    C = 1.0 - norm_cos.mean(dim=-1)
    return float(C.mean().item())


@torch.no_grad()
def effective_rank(matrix: Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank of a matrix. Dense analogue of Li et al.'s
    effective dimension (§4.4). Drops as the model finds lower-complexity
    representations — a classical grokking signal (Liu et al. 2022).
    """
    sv = torch.linalg.svdvals(matrix.float().detach().cpu())
    total = sv.sum()
    if total <= eps:
        return 0.0
    p = sv / total
    entropy = -(p * (p + eps).log()).sum()
    return float(entropy.exp().item())
