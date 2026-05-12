"""Post-hoc dense-analogue pathway analysis (Li et al. §4 applied to dense nets).

For a run that was trained with `--checkpoint-every N > 0`, this script:

1. Reads the run's JSON config + CSV of per-step metrics
2. Reconstructs the model and probe dataset
3. For every saved checkpoint, computes:
   - Pathway edit distance (pairwise, over probe) — Li et al. §4.1
   - Pathway consistency (mean over probe) — Li et al. §4.2
   - Effective rank of head weight matrix — dense analogue of §4.4
   - Per-task versions of edit distance / consistency (if multi-task)
4. Writes `<run>_pathway.csv` indexed by step and saves a summary plot

Usage:
    python compute_pathway_metrics.py --run-dir results/multitask
    python compute_pathway_metrics.py --run-name multitask_transformer_wd1.0 \\
        --run-dir results/multitask --probe-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import torch.nn as nn

from grok_lib import (
    build_model,
    effective_rank,
    get_modular_arithmetic_datasets,
    get_multitask_datasets,
    get_text_datasets,
    pathway_consistency,
    pathway_edit_distance,
    set_seed,
)


def _interesting_weights(model) -> dict[str, 'torch.Tensor']:
    """Collect weight matrices worth computing effective rank on.

    The classification head on binary tasks collapses to rank ~1 because only
    logit differences matter under softmax. We also look at interior weights
    (attention output projection and FFN projections) which have much higher
    achievable rank and therefore track meaningful representational changes.

    Dense analogue of Li et al. §4.4's effective-dimension-collapse claim.
    """
    weights = {}
    head = getattr(model, 'head', None) or getattr(model, 'out_head', None)
    if head is not None:
        weights['head'] = head.weight

    # Transformer: first block's FFN and attention output projection
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        blk = model.blocks[0]
        # FFN is nn.Sequential([Linear, GELU, Dropout, Linear, Dropout])
        linears = [m for m in blk.ff if isinstance(m, nn.Linear)]
        if len(linears) >= 1:
            weights['ffn_in'] = linears[0].weight   # shape (d_ff, d_model)
        if len(linears) >= 2:
            weights['ffn_out'] = linears[1].weight  # shape (d_model, d_ff)
        if hasattr(blk, 'attn') and hasattr(blk.attn, 'out_proj'):
            weights['attn_out'] = blk.attn.out_proj.weight  # (d_model, d_model)

    # MLP: first and second hidden layer weight matrices
    if hasattr(model, 'hidden_blocks') and len(model.hidden_blocks) > 0:
        linears = [m for m in model.hidden_blocks[0] if isinstance(m, nn.Linear)]
        if linears:
            weights['hidden1'] = linears[0].weight
        if len(model.hidden_blocks) >= 2:
            linears2 = [m for m in model.hidden_blocks[1] if isinstance(m, nn.Linear)]
            if linears2:
                weights['hidden2'] = linears2[0].weight

    return weights

MULTITASK_DATASETS = ['multitask']
MODULAR_DATASETS = ['mod_add', 'mod_mul', 'mod_sub']


def _reconstruct_dataset(meta):
    dataset = meta['dataset']
    seed = meta['seed']
    max_seq_len = meta['max_seq_len']
    set_seed(seed)
    if dataset in MODULAR_DATASETS:
        op = dataset.removeprefix('mod_')
        train_ds, test_ds, vocab_size, num_classes = get_modular_arithmetic_datasets(
            p=meta.get('mod_p', 97), op=op,
            train_fraction=meta['train_fraction'], seed=seed,
        )
        return train_ds, test_ds, vocab_size, num_classes, 3
    if dataset in MULTITASK_DATASETS:
        train_ds, test_ds, vocab_size, num_classes = get_multitask_datasets(
            train_fraction=meta['train_fraction'], seed=seed,
            max_seq_len=max_seq_len, max_per_task=meta.get('max_dataset_size', -1),
        )
        return train_ds, test_ds, vocab_size, num_classes, max_seq_len
    train_ds, test_ds, vocab_size, num_classes = get_text_datasets(
        dataset, train_fraction=meta['train_fraction'], seed=seed,
        max_seq_len=max_seq_len, max_dataset_size=meta.get('max_dataset_size', -1),
    )
    return train_ds, test_ds, vocab_size, num_classes, max_seq_len


def _probe_from_test(test_ds, probe_size: int, seed: int = 0):
    """Fixed probe: first `probe_size` test samples (stratified across tasks if multitask)."""
    n = len(test_ds)
    if hasattr(test_ds, 'task_ids') and hasattr(test_ds, 'task_names'):
        per_task = probe_size // len(test_ds.task_names)
        indices = []
        for tid in range(len(test_ds.task_names)):
            tid_idx = (test_ds.task_ids == tid).nonzero(as_tuple=True)[0]
            indices.extend(tid_idx[:per_task].tolist())
        indices = torch.tensor(indices)
    else:
        rng = np.random.default_rng(seed)
        indices = torch.tensor(rng.permutation(n)[:min(probe_size, n)])
    return indices, test_ds.inputs[indices], test_ds.padding_mask[indices], (
        test_ds.task_ids[indices] if hasattr(test_ds, 'task_ids') else None
    )


def analyze_run(run_dir: Path, run_name: str, probe_size: int = 64) -> Path:
    meta_path = run_dir / f'{run_name}.json'
    ckpt_dir = run_dir / f'{run_name}_checkpoints'
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"no checkpoints at {ckpt_dir}")

    meta = json.loads(meta_path.read_text())
    print(f"[{run_name}] dataset={meta['dataset']} arch={meta['arch']} seed={meta['seed']}", flush=True)

    train_ds, test_ds, vocab_size, num_classes, seq_len = _reconstruct_dataset(meta)
    probe_indices, probe_inputs, probe_mask, probe_task_ids = _probe_from_test(test_ds, probe_size)
    task_names = getattr(test_ds, 'task_names', None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(meta['arch'], vocab_size, num_classes, seq_len).to(device)
    probe_inputs = probe_inputs.to(device)
    probe_mask = probe_mask.to(device)

    ckpts = sorted(ckpt_dir.glob('step_*.pt'))
    print(f"[{run_name}] analyzing {len(ckpts)} checkpoints over probe_size={probe_size}", flush=True)

    rows = []
    for i, ckpt_path in enumerate(ckpts, start=1):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state['state_dict'])
        model.eval()

        row = {'step': int(state['step'])}
        row['pathway_edit_dist'] = pathway_edit_distance(model, probe_inputs, probe_mask)
        row['pathway_consistency'] = pathway_consistency(model, probe_inputs, probe_mask)
        for w_name, w in _interesting_weights(model).items():
            row[f'eff_rank_{w_name}'] = effective_rank(w)

        if task_names is not None and probe_task_ids is not None:
            for tid, tname in enumerate(task_names):
                mask = (probe_task_ids == tid)
                if mask.sum().item() < 2:
                    continue
                sub_inputs = probe_inputs[mask]
                sub_pm = probe_mask[mask]
                row[f'edit_dist_{tname}'] = pathway_edit_distance(model, sub_inputs, sub_pm)
                row[f'consistency_{tname}'] = pathway_consistency(model, sub_inputs, sub_pm)

        rows.append(row)
        if i % 5 == 0 or i == len(ckpts):
            rank_tail = '  '.join(
                f"{k.removeprefix('eff_rank_')}={row[k]:.2f}"
                for k in row if k.startswith('eff_rank_')
            )
            print(f"  [{i}/{len(ckpts)}] step={row['step']:6d} "
                  f"edit={row['pathway_edit_dist']:.2f} "
                  f"cons={row['pathway_consistency']:.4f} | {rank_tail}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = run_dir / f'{run_name}_pathway.csv'
    df.to_csv(out_csv, index=False)
    print(f"[{run_name}] wrote {out_csv} ({len(df)} rows)", flush=True)
    return out_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True,
                   help='sweep directory, e.g. results/multitask')
    p.add_argument('--run-name', default=None,
                   help='specific run name; if omitted, analyze all runs in --run-dir')
    p.add_argument('--probe-size', type=int, default=64)
    args = p.parse_args()

    if args.run_name:
        analyze_run(args.run_dir, args.run_name, probe_size=args.probe_size)
        return 0

    run_names = sorted({
        p.name.removesuffix('_checkpoints')
        for p in args.run_dir.iterdir()
        if p.is_dir() and p.name.endswith('_checkpoints')
    })
    if not run_names:
        print(f"No runs with checkpoints found in {args.run_dir}")
        return 1
    for name in run_names:
        analyze_run(args.run_dir, name, probe_size=args.probe_size)
    return 0


if __name__ == '__main__':
    sys.exit(main())
