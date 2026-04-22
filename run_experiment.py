"""
Sweep runner for grokking experiments.

Reads a YAML sweep config, generates all hyperparameter combinations via
Cartesian product, and runs each as a separate training job.

Usage:
  # Dry run — just print what would be run:
  python run_experiment.py configs/sweep_weight_decay.yaml --dry-run

  # Sequential execution:
  python run_experiment.py configs/sweep_weight_decay.yaml

  # Parallel execution (N jobs at a time):
  python run_experiment.py configs/sweep_weight_decay.yaml --parallel 4

Sweep config format (YAML):
  base_config: configs/base.yaml   # optional base to merge from
  sweep:
    weight_decay: [0.0, 0.1, 1.0]
    seed: [0, 1, 2]
"""

import argparse
import csv
import itertools
import json
import multiprocessing
import os
import subprocess
import sys

import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_sweep_config(sweep_path: str) -> tuple[dict, dict, list[dict]]:
    """
    Load a sweep config YAML.

    Returns:
        base_params: flat dict of fixed hyperparameter overrides.
        sweep_grid:  dict mapping param name → list of values (Cartesian product).
        sweep_zip:   list of dicts that vary together (no Cartesian expansion).

    sweep_zip example — vary d_model/n_heads/d_ff as a unit:
        sweep_zip:
          - {d_model: 64,  n_heads: 2, d_ff: 256}
          - {d_model: 256, n_heads: 8, d_ff: 1024}
    """
    cfg = load_yaml(sweep_path)

    # Load and merge base config if specified
    base_params = {}
    if 'base_config' in cfg:
        base_cfg = load_yaml(cfg['base_config'])
        base_params.update(base_cfg)
        # Remove meta-keys that aren't train.py args
        for meta in ('base_config', 'sweep', 'sweep_zip'):
            base_params.pop(meta, None)

    # Override base with any fixed params in the sweep config (non-meta keys)
    sweep_grid = cfg.get('sweep', {})
    sweep_zip  = cfg.get('sweep_zip', [])
    for k, v in cfg.items():
        if k not in ('base_config', 'sweep', 'sweep_zip'):
            base_params[k] = v

    return base_params, sweep_grid, sweep_zip


def expand_sweep(sweep_grid: dict, sweep_zip: list[dict] | None = None) -> list[dict]:
    """
    Expand a sweep into a list of override dicts.

    sweep_grid is expanded via Cartesian product.  sweep_zip entries are then
    crossed with every grid combo (but not with each other), so they vary as a
    unit rather than independently.

    Examples:
      grid={'wd': [0.1, 1.0], 'seed': [0, 1]}, zip=None
        → 4 combos (standard Cartesian)

      grid={'n_layers': [1, 2]}, zip=[{d_model:64, d_ff:256}, {d_model:256, d_ff:1024}]
        → 4 combos: each n_layers value paired with each zip entry
    """
    if sweep_grid:
        keys   = list(sweep_grid.keys())
        values = [sweep_grid[k] for k in keys]
        grid_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    else:
        grid_combos = [{}]

    if not sweep_zip:
        return grid_combos

    return [
        {**grid_combo, **zip_entry}
        for grid_combo in grid_combos
        for zip_entry  in sweep_zip
    ]


# ---------------------------------------------------------------------------
# Experiment naming (mirrors train.py logic)
# ---------------------------------------------------------------------------

def make_exp_name(params: dict) -> str:
    model    = params.get('model', 'transformer')
    wd       = params.get('weight_decay', 1.0)
    frac     = params.get('train_fraction', 0.5)
    seed     = params.get('seed', 42)
    d_model  = params.get('d_model', 128)
    n_layers = params.get('n_layers', 2)
    dropout  = params.get('dropout', 0.1)

    # Append suffixes only when params deviate from base.yaml defaults so that
    # standard runs keep their original names and model-size / regularization
    # sweep runs get unambiguous, non-colliding names.
    size_sfx = f"_d{d_model}_l{n_layers}" if (d_model != 128 or n_layers != 2) else ""
    drop_sfx = f"_do{dropout}"            if  dropout != 0.1                    else ""

    dataset = params.get('dataset', 'modular_arithmetic')

    if dataset == 'text':
        hf_dataset = params.get('hf_dataset', 'unknown')
        return f"{model}_text_{hf_dataset}_wd{wd}_frac{frac}_seed{seed}{size_sfx}{drop_sfx}"

    if dataset == 'analogy':
        p = params.get('analogy_rows', 5)
        q = params.get('analogy_cols', 5)
        return f"{model}_analogy_p{p}q{q}_wd{wd}_frac{frac}_seed{seed}{size_sfx}{drop_sfx}"

    op_name = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'div'}.get(
        params.get('operation', '+'), params.get('operation', '+')
    )
    return (
        f"{model}"
        f"_mod{params.get('prime', 97)}"
        f"_{op_name}"
        f"_wd{wd}"
        f"_frac{frac}"
        f"_seed{seed}"
        f"{size_sfx}"
        f"{drop_sfx}"
    )


# ---------------------------------------------------------------------------
# Run a single experiment
# ---------------------------------------------------------------------------

def is_complete(params: dict, results_dir: str) -> bool:
    """
    Return True if this experiment already has a complete metrics.csv
    (i.e., the last logged step matches n_steps within one log_every interval).
    """
    exp_name = params.get('exp_name', make_exp_name(params))
    metrics_path = os.path.join(results_dir, exp_name, 'metrics.csv')
    if not os.path.exists(metrics_path):
        return False
    try:
        with open(metrics_path) as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return False
        last_step = int(rows[-1][0])
        n_steps = int(params.get('n_steps', 100_000))
        log_every = int(params.get('log_every', 100))
        return last_step >= n_steps - log_every
    except Exception:
        return False


def params_to_argv(params: dict) -> list[str]:
    """Convert a params dict to a list of CLI arguments for train.py."""
    argv = []
    for k, v in params.items():
        if k in ('base_config', 'sweep'):
            continue
        # Handle list values (e.g., betas)
        if isinstance(v, (list, tuple)):
            argv.append(f'--{k}')
            argv.extend(str(x) for x in v)
        elif isinstance(v, bool):
            if v:
                argv.append(f'--{k}')
        else:
            argv.append(f'--{k}')
            argv.append(str(v))
    return argv


def run_single(params: dict, results_dir: str, dry_run: bool = False) -> int:
    """
    Run a single training job. Returns the subprocess return code (0 = success).
    """
    # Idempotency: skip if already complete
    if not dry_run and is_complete(params, results_dir):
        exp_name = params.get('exp_name', make_exp_name(params))
        print(f"[SKIP] Already complete: {exp_name}")
        return 0

    argv = params_to_argv(params)
    cmd = [sys.executable, 'train.py'] + argv

    exp_name = params.get('exp_name', make_exp_name(params))
    print(f"[RUN]  {exp_name}")
    if dry_run:
        print('  ' + ' '.join(cmd))
        return 0

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"[ERROR] Job failed with exit code {result.returncode}: {exp_name}")
    return result.returncode


def _run_single_worker(args_tuple):
    """Worker function for multiprocessing.Pool."""
    params, results_dir, dry_run = args_tuple
    return run_single(params, results_dir, dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run a hyperparameter sweep for grokking experiments.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('config', help='Path to sweep YAML config file.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing them.')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel jobs. 1 = sequential.')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Override results_dir from base config.')
    args = parser.parse_args()

    base_params, sweep_grid, sweep_zip = load_sweep_config(args.config)
    overrides = expand_sweep(sweep_grid, sweep_zip)

    # Build full param dicts for each combination
    all_params = []
    for override in overrides:
        params = {**base_params, **override}
        # Ensure results_dir is set (CLI override takes priority)
        if args.results_dir != 'results' or 'results_dir' not in params:
            params['results_dir'] = args.results_dir
        all_params.append(params)

    results_dir = all_params[0].get('results_dir', 'results') if all_params else 'results'

    print(f"Sweep: {len(all_params)} jobs from {args.config}")
    if args.dry_run:
        print("(dry run — no jobs will be executed)\n")

    if args.parallel > 1 and not args.dry_run:
        worker_args = [(p, results_dir, args.dry_run) for p in all_params]
        with multiprocessing.Pool(processes=args.parallel) as pool:
            return_codes = pool.map(_run_single_worker, worker_args)
        failed = sum(1 for rc in return_codes if rc != 0)
    else:
        failed = 0
        for params in all_params:
            rc = run_single(params, results_dir=results_dir, dry_run=args.dry_run)
            if rc != 0:
                failed += 1

    print(f"\nSweep complete. {len(all_params) - failed}/{len(all_params)} jobs succeeded.")
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
