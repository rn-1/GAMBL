"""
Main training script for grokking experiments.

Usage examples:

  # Reproduce original Power et al. grokking (transformer, mod97, addition):
  python train.py --model transformer --n_steps 100000

  # MLP comparison:
  python train.py --model mlp --n_steps 100000

  # No weight decay (grokking should NOT occur):
  python train.py --model transformer --weight_decay 0.0 --n_steps 50000

  # Smaller training set (grokking more likely):
  python train.py --model transformer --train_fraction 0.3 --n_steps 100000

  # Smoke test:
  python train.py --n_steps 500 --log_every 10 --checkpoint_every 100
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.modular_arithmetic import get_modular_arithmetic_datasets, get_vocab_size
from data.text_datasets import get_text_datasets, list_datasets as list_text_datasets
from data.analogy import get_analogy_datasets, get_analogy_vocab_size, SEQ_LEN as ANALOGY_SEQ_LEN
from models.mlp import MLP
from models.transformer import GrokTransformer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train a model to study the grokking phenomenon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    p.add_argument('--dataset', type=str, default='modular_arithmetic',
                   choices=['modular_arithmetic', 'text', 'analogy'],
                   help='Dataset family.')
    p.add_argument('--prime', type=int, default=97,
                   help='Prime modulus p for modular arithmetic.')
    # Analogy-dataset-specific
    p.add_argument('--analogy_rows', type=int, default=5,
                   help='P: number of rows in the entity grid (Z_P component) for analogy dataset.')
    p.add_argument('--analogy_cols', type=int, default=5,
                   help='Q: number of columns in the entity grid (Z_Q component) for analogy dataset.')
    p.add_argument('--operation', type=str, default='+',
                   choices=['+', '-', '*', '/'],
                   help='Arithmetic operation.')
    p.add_argument('--train_fraction', type=float, default=0.5,
                   help='Fraction of all pairs / examples used for training (0 < f < 1).')
    # Text-dataset-specific
    p.add_argument('--hf_dataset', type=str, default=None,
                   help=f'HuggingFace text dataset key. One of: {list_text_datasets()}')
    p.add_argument('--tokenizer_name', type=str, default='bert-base-uncased',
                   help='HuggingFace tokenizer name or local path.')
    p.add_argument('--max_seq_len', type=int, default=4,
                   help='Max token sequence length. 4 for modular arithmetic, '
                        '128 for text datasets.')
    p.add_argument('--max_dataset_size', type=int, default=-1,
                   help='Cap total examples before train/test split. '
                        'Useful for large datasets like ag_news (default: no cap).')

    # Model
    p.add_argument('--model', type=str, default='transformer',
                   choices=['transformer', 'mlp'],
                   help='Model architecture.')
    # Transformer-specific
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--d_ff', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--pool', type=str, default='last', choices=['last', 'mean', 'cls'])
    p.add_argument('--no_pos_encoding', action='store_true',
                   help='Disable positional encoding in the transformer.')
    # MLP-specific
    p.add_argument('--embed_dim', type=int, default=128,
                   help='Embedding dim per token (MLP).')
    p.add_argument('--hidden_dim', type=int, default=512,
                   help='Hidden layer width (MLP).')
    p.add_argument('--num_mlp_layers', type=int, default=3,
                   help='Number of linear layers including output (MLP).')
    p.add_argument('--activation', type=str, default='relu',
                   choices=['relu', 'gelu', 'tanh'],
                   help='Activation function (MLP).')

    # Optimiser
    p.add_argument('--lr', type=float, default=1e-3,
                   help='AdamW learning rate.')
    p.add_argument('--weight_decay', type=float, default=1.0,
                   help='AdamW weight decay. Power et al. use 1.0 — critical for grokking.')
    p.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.98),
                   metavar=('BETA1', 'BETA2'),
                   help='AdamW beta coefficients.')

    # Training schedule
    p.add_argument('--n_steps', type=int, default=100_000,
                   help='Total number of gradient steps.')
    p.add_argument('--batch_size', type=int, default=-1,
                   help='Mini-batch size. Use -1 for full-batch training.')

    # Logging & checkpointing
    p.add_argument('--log_every', type=int, default=100,
                   help='Log metrics every N steps.')
    p.add_argument('--checkpoint_every', type=int, default=500,
                   help='Save a checkpoint every N steps.')

    # Experiment management
    p.add_argument('--exp_name', type=str, default='auto',
                   help='Experiment name. "auto" generates from hyperparams.')
    p.add_argument('--results_dir', type=str, default='results',
                   help='Root directory for all experiment outputs.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto',
                   help='"auto" uses CUDA if available, else CPU.')

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def make_exp_name(args) -> str:
    size_sfx = f"_d{args.d_model}_l{args.n_layers}" if (args.d_model != 128 or args.n_layers != 2) else ""
    drop_sfx = f"_do{args.dropout}"                 if  args.dropout != 0.1                        else ""

    if args.dataset == 'modular_arithmetic':
        op_name = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'div'}[args.operation]
        return (
            f"{args.model}_mod{args.prime}_{op_name}"
            f"_wd{args.weight_decay}"
            f"_frac{args.train_fraction}"
            f"_seed{args.seed}"
            f"{size_sfx}{drop_sfx}"
        )
    elif args.dataset == 'analogy':
        return (
            f"{args.model}_analogy_p{args.analogy_rows}q{args.analogy_cols}"
            f"_wd{args.weight_decay}"
            f"_frac{args.train_fraction}"
            f"_seed{args.seed}"
            f"{size_sfx}{drop_sfx}"
        )
    else:
        return (
            f"{args.model}_text_{args.hf_dataset}"
            f"_wd{args.weight_decay}"
            f"_frac{args.train_fraction}"
            f"_seed{args.seed}"
            f"{size_sfx}{drop_sfx}"
        )


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def build_datasets(args):
    """Return (train_ds, test_ds, vocab_size, output_dim, seq_len)."""
    if args.dataset == 'modular_arithmetic':
        train_ds, test_ds = get_modular_arithmetic_datasets(
            p=args.prime,
            operation=args.operation,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
        vocab_size = get_vocab_size(args.prime)
        output_dim = args.prime
        return train_ds, test_ds, vocab_size, output_dim, args.max_seq_len
    elif args.dataset == 'analogy':
        train_ds, test_ds = get_analogy_datasets(
            p=args.analogy_rows,
            q=args.analogy_cols,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
        vocab_size = get_analogy_vocab_size(args.analogy_rows, args.analogy_cols)
        output_dim = args.analogy_rows * args.analogy_cols
        return train_ds, test_ds, vocab_size, output_dim, ANALOGY_SEQ_LEN
    elif args.dataset == 'text':
        if args.hf_dataset is None:
            raise ValueError("--hf_dataset is required when --dataset text")
        train_ds, test_ds, vocab_size, num_classes = get_text_datasets(
            dataset_name=args.hf_dataset,
            train_fraction=args.train_fraction,
            seed=args.seed,
            tokenizer_name=args.tokenizer_name,
            max_seq_len=args.max_seq_len,
            max_dataset_size=args.max_dataset_size,
        )
        return train_ds, test_ds, vocab_size, num_classes, args.max_seq_len
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def build_model(args, vocab_size: int, output_dim: int) -> nn.Module:
    if args.model == 'transformer':
        return GrokTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            output_dim=output_dim,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            use_positional_encoding=not args.no_pos_encoding,
            pool=args.pool,
        )
    elif args.model == 'mlp':
        return MLP(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_mlp_layers,
            output_dim=output_dim,
            activation=args.activation,
            dropout=args.dropout,
            seq_len=args.max_seq_len,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


def make_infinite_loader(dataset, batch_size: int):
    """Yield batches indefinitely, reshuffling each epoch."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- reproducibility ------------------------------------------------
    set_seed(args.seed)
    device = get_device(args.device)

    # ---- experiment setup -----------------------------------------------
    if args.exp_name == 'auto':
        exp_name = make_exp_name(args)
    else:
        exp_name = args.exp_name

    exp_dir = os.path.join(args.results_dir, exp_name)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    metrics_path = os.path.join(exp_dir, 'metrics.csv')
    config_path = os.path.join(exp_dir, 'config.json')

    # Save config immediately
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Experiment: {exp_name}")
    print(f"Results dir: {exp_dir}")
    print(f"Device: {device}")

    # ---- data -----------------------------------------------------------
    train_ds, test_ds, vocab_size, output_dim, seq_len = build_datasets(args)
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}, "
          f"Vocab size: {vocab_size}, Output dim: {output_dim}, Seq len: {seq_len}")

    batch_size = len(train_ds) if args.batch_size == -1 else args.batch_size
    train_loader = make_infinite_loader(train_ds, batch_size=batch_size)

    # For evaluation, load everything at once.
    # TextDataset exposes a padding_mask attribute; the others do not.
    test_inputs  = test_ds.inputs.to(device)
    test_labels  = test_ds.labels.to(device)
    train_inputs = train_ds.inputs.to(device)
    train_labels = train_ds.labels.to(device)
    test_mask  = getattr(test_ds,  'padding_mask', None)
    train_mask = getattr(train_ds, 'padding_mask', None)
    if test_mask  is not None: test_mask  = test_mask.to(device)
    if train_mask is not None: train_mask = train_mask.to(device)

    # ---- model ----------------------------------------------------------
    args.max_seq_len = seq_len  # ensure model positional embeddings match the dataset
    model = build_model(args, vocab_size=vocab_size, output_dim=output_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, Parameters: {n_params:,}")

    # ---- optimizer ------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # ---- logging --------------------------------------------------------
    csv_file = open(metrics_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
    csv_file.flush()

    # ---- training loop --------------------------------------------------
    model.train()
    start_time = time.time()

    for step in range(1, args.n_steps + 1):
        # --- gradient step ---
        batch = next(train_loader)
        if len(batch) == 3:
            inputs, mask, labels = (t.to(device) for t in batch)
        else:
            inputs, labels = (t.to(device) for t in batch)
            mask = None

        optimizer.zero_grad()
        logits = model(inputs, padding_mask=mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # --- logging ---
        if step % args.log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                # Training metrics (full dataset for accurate reporting)
                train_logits = model(train_inputs, padding_mask=train_mask)
                train_loss = criterion(train_logits, train_labels).item()
                train_acc = compute_accuracy(train_logits, train_labels)

                # Test metrics
                test_logits = model(test_inputs, padding_mask=test_mask)
                test_loss = criterion(test_logits, test_labels).item()
                test_acc = compute_accuracy(test_logits, test_labels)

            writer.writerow([step, train_loss, train_acc, test_loss, test_acc])
            csv_file.flush()

            elapsed = time.time() - start_time
            print(
                f"step={step:7d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
                f"elapsed={elapsed:.0f}s"
            )
            model.train()

        # --- checkpointing ---
        if step % args.checkpoint_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, ckpt_path)

    csv_file.close()
    print(f"\nTraining complete. Results saved to {exp_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    train(args)
