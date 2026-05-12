"""Run a single grokking experiment config and save CSV + PNG artifacts.

Invoked via subprocess by run_sweep.py (with CUDA_VISIBLE_DEVICES set per-GPU),
but can also be run standalone:

    python run_single.py --name rte_wd1.0 --dataset rte --arch transformer \\
        --weight-decay 1.0 --n-steps 50000 --out-dir results/weight_decay
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_ROOT = Path(__file__).resolve().parent.parent

import torch

from grok_lib import (
    build_model,
    get_modular_arithmetic_datasets,
    get_multitask_datasets,
    get_text_datasets,
    plot_grokking_curve,
    set_seed,
    train_model,
)

TEXT_DATASETS = ['rte', 'mrpc', 'cola', 'sst2', 'boolq', 'ag_news']
MODULAR_DATASETS = ['mod_add', 'mod_mul', 'mod_sub']
MULTITASK_DATASETS = ['multitask']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help='run name (used for output filenames)')
    p.add_argument('--dataset', required=True, choices=TEXT_DATASETS + MODULAR_DATASETS + MULTITASK_DATASETS)
    p.add_argument('--arch', default='transformer', choices=['transformer', 'mlp'])
    p.add_argument('--weight-decay', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--train-fraction', type=float, default=0.5)
    p.add_argument('--n-steps', type=int, default=50000)
    p.add_argument('--batch-size', type=int, default=-1, help='-1 for full batch')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--max-seq-len', type=int, default=128, help='ignored for modular arithmetic (uses 3)')
    p.add_argument('--max-dataset-size', type=int, default=-1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', type=Path, default=_REPO_ROOT / 'results')
    p.add_argument('--grok-threshold', type=float, default=0.75)
    p.add_argument('--mod-p', type=int, default=97, help='prime modulus for modular arithmetic')
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='save model state_dict every N steps (0 = off). Enables post-hoc pathway metrics.')

    # --- Architecture knobs (all default to the Power-et-al. Small model) ---
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--d-ff', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.1)

    # --- Regularization / Omnigrok interventions ---
    p.add_argument('--init-scale', type=float, default=1.0,
                   help='Omnigrok-style weight rescaling (Liu et al. 2022). Multiply all >=2D weights by this at init.')
    p.add_argument('--label-smoothing', type=float, default=0.0)

    # --- Inline pathway metrics (Li et al. §4, computed during training) ---
    p.add_argument('--pathway-every', type=int, default=0,
                   help='Compute pathway metrics every N steps (0 = off). Adds columns to the main metrics CSV.')
    p.add_argument('--pathway-probe-size', type=int, default=32)
    p.add_argument('--pathway-top-k', type=int, default=8)

    # --- Flat-metric early-kill ---
    p.add_argument('--flat-kill-start-after', type=int, default=0,
                   help='Start checking flatness after this step (0 = disabled).')
    p.add_argument('--flat-kill-window', type=int, default=10,
                   help='Number of recent pathway measurements to check for flatness.')
    p.add_argument('--flat-kill-min-delta', type=float, default=0.01,
                   help='Relative range below which a metric is considered flat.')

    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    print(f"[{args.name}] device={device} ({gpu_name})", flush=True)

    set_seed(args.seed)
    is_modular = args.dataset in MODULAR_DATASETS
    is_multitask = args.dataset in MULTITASK_DATASETS
    if is_modular:
        op = args.dataset.removeprefix('mod_')
        train_ds, test_ds, vocab_size, num_classes = get_modular_arithmetic_datasets(
            p=args.mod_p, op=op, train_fraction=args.train_fraction, seed=args.seed,
        )
        effective_seq_len = 3
    elif is_multitask:
        train_ds, test_ds, vocab_size, num_classes = get_multitask_datasets(
            train_fraction=args.train_fraction, seed=args.seed,
            max_seq_len=args.max_seq_len, max_per_task=args.max_dataset_size,
        )
        effective_seq_len = args.max_seq_len
        print(f"[{args.name}] multitask tasks={train_ds.task_names}", flush=True)
    else:
        train_ds, test_ds, vocab_size, num_classes = get_text_datasets(
            args.dataset, train_fraction=args.train_fraction, seed=args.seed,
            max_seq_len=args.max_seq_len, max_dataset_size=args.max_dataset_size,
        )
        effective_seq_len = args.max_seq_len
    print(f"[{args.name}] train={len(train_ds)} test={len(test_ds)} vocab={vocab_size} classes={num_classes}", flush=True)

    # Auto-switch to mini-batch for large training sets (avoid full-batch OOM).
    # Modular arithmetic (seq_len=3) is tiny even at ~9k examples, keep full-batch.
    batch_size = args.batch_size
    if batch_size == -1 and not is_modular and len(train_ds) > 4000:
        batch_size = 512
        print(f"[{args.name}] auto mini-batch: train set={len(train_ds)} -> batch_size={batch_size}", flush=True)

    set_seed(args.seed)
    model = build_model(
        args.arch, vocab_size, num_classes, effective_seq_len,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout, init_scale=args.init_scale,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.name}] arch={args.arch} params={n_params:,} "
          f"(d_model={args.d_model} layers={args.n_layers} d_ff={args.d_ff} "
          f"dropout={args.dropout} init_scale={args.init_scale})", flush=True)

    csv_path = args.out_dir / f'{args.name}.csv'
    png_path = args.out_dir / f'{args.name}.png'
    json_path = args.out_dir / f'{args.name}.json'
    checkpoint_dir = args.out_dir / f'{args.name}_checkpoints' if args.checkpoint_every > 0 else None

    interrupted = False
    try:
        df = train_model(
            model, train_ds, test_ds,
            n_steps=args.n_steps, batch_size=batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            log_every=args.log_every, device=device, verbose=True,
            csv_path=csv_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            pathway_every=args.pathway_every,
            pathway_probe_size=args.pathway_probe_size,
            pathway_top_k=args.pathway_top_k,
            flat_kill_start_after=args.flat_kill_start_after,
            flat_kill_window=args.flat_kill_window,
            flat_kill_min_delta=args.flat_kill_min_delta,
            run_name=args.name,
        )
    except KeyboardInterrupt:
        interrupted = True
        print(f"[{args.name}] INTERRUPTED — reloading partial CSV from {csv_path}", flush=True)
        import pandas as pd
        df = pd.read_csv(csv_path) if csv_path.exists() and csv_path.stat().st_size > 0 else None

    if df is None or len(df) == 0:
        print(f"[{args.name}] no metrics logged before interrupt — exiting", flush=True)
        sys.exit(130 if interrupted else 1)

    # Early-kill info (set by train_model via df.attrs when running inline).
    early_killed = bool(df.attrs.get('early_killed', False)) if hasattr(df, 'attrs') else False
    kill_reason = df.attrs.get('kill_reason', '') if hasattr(df, 'attrs') else ''
    kill_step = df.attrs.get('kill_step', None) if hasattr(df, 'attrs') else None

    meta = vars(args).copy()
    meta['out_dir'] = str(meta['out_dir'])
    meta['n_params'] = n_params
    meta['train_size'] = len(train_ds)
    meta['test_size'] = len(test_ds)
    meta['final_train_acc'] = float(df['train_acc'].iloc[-1])
    meta['final_test_acc'] = float(df['test_acc'].iloc[-1])
    meta['logged_rows'] = len(df)
    meta['last_logged_step'] = int(df['step'].iloc[-1])
    # An early-killed run IS complete (don't rerun it on sweep retry). An
    # interrupted (SIGINT) run is not.
    meta['completed'] = not interrupted
    meta['early_killed'] = early_killed
    meta['kill_reason'] = kill_reason
    meta['kill_step'] = kill_step
    json_path.write_text(json.dumps(meta, indent=2))

    plot_grokking_curve(df, title=args.name, grok_threshold=args.grok_threshold, save_path=png_path)

    if early_killed:
        status = 'EARLY_KILL'
    elif interrupted:
        status = 'INTERRUPTED'
    else:
        status = 'DONE'
    print(f"[{args.name}] {status} -> {csv_path} ({len(df)} rows, last step {meta['last_logged_step']}) | "
          f"final test_acc={meta['final_test_acc']:.4f}", flush=True)
    if interrupted:
        sys.exit(130)


if __name__ == '__main__':
    sys.exit(main())
