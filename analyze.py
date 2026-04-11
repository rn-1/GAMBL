"""
Analysis and visualization for grokking experiments.

Usage:
  # Plot a single experiment's grokking curve:
  python analyze.py --plot grokking_curve \
      --exp transformer_mod97_plus_wd1.0_frac0.5_seed42

  # Compare multiple experiments on one plot:
  python analyze.py --plot comparison \
      --pattern "transformer_mod97_plus_wd*_frac0.5_seed42" \
      --metric test_acc

  # Print a summary table of grokking steps across a sweep:
  python analyze.py --plot sweep_summary \
      --pattern "transformer_mod97_plus_wd*" \
      --groupby weight_decay

  # List all available experiments:
  python analyze.py --list
"""

import argparse
import fnmatch
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(results_dir: str, exp_name: str) -> pd.DataFrame:
    """Load metrics.csv for one experiment."""
    path = os.path.join(results_dir, exp_name, 'metrics.csv')
    return pd.read_csv(path)


def load_config(results_dir: str, exp_name: str) -> dict:
    """Load config.json for one experiment."""
    path = os.path.join(results_dir, exp_name, 'config.json')
    with open(path) as f:
        return json.load(f)


def load_all_metrics(
    results_dir: str,
    pattern: str = '*',
) -> dict[str, pd.DataFrame]:
    """
    Load metrics for all experiments whose names match `pattern` (fnmatch glob).

    Returns:
        dict mapping exp_name → DataFrame
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_exp = {}
    for name in sorted(os.listdir(results_dir)):
        exp_dir = os.path.join(results_dir, name)
        metrics_path = os.path.join(exp_dir, 'metrics.csv')
        if not os.path.isdir(exp_dir) or not os.path.exists(metrics_path):
            continue
        if fnmatch.fnmatch(name, pattern):
            try:
                all_exp[name] = load_metrics(results_dir, name)
            except Exception as e:
                print(f"Warning: could not load {name}: {e}")
    return all_exp


# ---------------------------------------------------------------------------
# Grokking detection
# ---------------------------------------------------------------------------

def find_grokking_step(df: pd.DataFrame, threshold: float = 0.95) -> int | None:
    """
    Return the first step where test_acc >= threshold.
    Returns None if the model never reaches the threshold.
    """
    hits = df[df['test_acc'] >= threshold]
    if hits.empty:
        return None
    return int(hits.iloc[0]['step'])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })


def plot_grokking_curve(
    df: pd.DataFrame,
    title: str = '',
    log_x: bool = True,
    threshold: float = 0.95,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot train and test accuracy over training steps for a single experiment.

    Args:
        df:         metrics DataFrame with columns [step, train_acc, test_acc].
        title:      Plot title.
        log_x:      Use logarithmic x-axis (standard in grokking papers).
        threshold:  Threshold for annotating the grokking point.
        save_path:  If provided, save figure to this path.
        show:       If True, call plt.show().
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df['step'], df['train_acc'], label='Train accuracy', color='steelblue')
    ax.plot(df['step'], df['test_acc'], label='Test accuracy', color='darkorange')

    grok_step = find_grokking_step(df, threshold=threshold)
    if grok_step is not None:
        ax.axvline(grok_step, color='green', linestyle='--', alpha=0.7,
                   label=f'Grokking @ step {grok_step:,}')

    ax.set_xlabel('Training steps')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(-0.05, 1.05)
    if log_x:
        ax.set_xscale('log')
        ax.set_xlabel('Training steps (log scale)')
    ax.legend()
    ax.set_title(title or 'Grokking curve')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison(
    metrics_dict: dict[str, pd.DataFrame],
    metric: str = 'test_acc',
    log_x: bool = True,
    title: str = '',
    save_path: str = None,
    show: bool = True,
):
    """
    Plot one metric from multiple experiments on the same axes.

    Args:
        metrics_dict:  dict exp_name → DataFrame.
        metric:        Column name to plot ('test_acc', 'train_acc', etc.).
        log_x:         Use logarithmic x-axis.
        title:         Plot title.
        save_path:     If provided, save figure to this path.
        show:          If True, call plt.show().
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.cm.tab10
    for i, (name, df) in enumerate(metrics_dict.items()):
        if metric not in df.columns:
            print(f"Warning: column '{metric}' not found in {name}, skipping.")
            continue
        label = name
        ax.plot(df['step'], df[metric], label=label, color=cmap(i % 10))

    ax.set_xlabel('Training steps' + (' (log scale)' if log_x else ''))
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_ylim(-0.05, 1.05)
    if log_x:
        ax.set_xscale('log')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title(title or f'Comparison: {metric}')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_loss_curve(
    df: pd.DataFrame,
    title: str = '',
    log_x: bool = True,
    save_path: str = None,
    show: bool = True,
):
    """Plot train and test loss for a single experiment."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df['step'], df['train_loss'], label='Train loss', color='steelblue')
    ax.plot(df['step'], df['test_loss'], label='Test loss', color='darkorange')

    ax.set_xlabel('Training steps' + (' (log scale)' if log_x else ''))
    ax.set_ylabel('Cross-entropy loss')
    if log_x:
        ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title(title or 'Loss curve')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep summary
# ---------------------------------------------------------------------------

def summarize_sweep(
    metrics_dict: dict[str, pd.DataFrame],
    results_dir: str,
    groupby: str = 'weight_decay',
    threshold: float = 0.95,
) -> pd.DataFrame:
    """
    For each experiment, find the grokking step and read the groupby param
    from config.json. Returns a summary DataFrame.

    Columns: [<groupby>, seed, grokking_step, final_test_acc, exp_name]
    """
    rows = []
    for name, df in metrics_dict.items():
        grok_step = find_grokking_step(df, threshold=threshold)
        final_test_acc = float(df['test_acc'].iloc[-1]) if len(df) > 0 else float('nan')
        try:
            cfg = load_config(results_dir, name)
            groupby_val = cfg.get(groupby, None)
            seed = cfg.get('seed', None)
        except Exception:
            groupby_val = None
            seed = None
        rows.append({
            groupby: groupby_val,
            'seed': seed,
            'grokking_step': grok_step,
            'final_test_acc': final_test_acc,
            'exp_name': name,
        })

    summary = pd.DataFrame(rows)
    return summary.sort_values([groupby, 'seed']).reset_index(drop=True)


def print_sweep_summary(summary: pd.DataFrame, groupby: str = 'weight_decay'):
    """Print a human-readable summary grouped by a hyperparameter."""
    print(f"\n{'='*60}")
    print(f"Sweep summary (grouped by {groupby})")
    print(f"{'='*60}")
    grouped = summary.groupby(groupby)
    for val, grp in grouped:
        grok_steps = grp['grokking_step'].dropna()
        n_grokked = len(grok_steps)
        n_total = len(grp)
        mean_step = grok_steps.mean() if n_grokked > 0 else float('nan')
        std_step = grok_steps.std() if n_grokked > 1 else 0.0
        final_acc = grp['final_test_acc'].mean()
        print(
            f"  {groupby}={val:<8} | "
            f"grokked={n_grokked}/{n_total} | "
            f"grok_step={mean_step:.0f}±{std_step:.0f} | "
            f"final_test_acc={final_acc:.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize grokking experiment results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--figures_dir', type=str, default='figures')
    parser.add_argument('--plot', type=str,
                        choices=['grokking_curve', 'loss_curve', 'comparison', 'sweep_summary'],
                        default='grokking_curve',
                        help='Type of plot to generate.')
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment name for single-experiment plots.')
    parser.add_argument('--pattern', type=str, default='*',
                        help='Glob pattern to match experiment names for multi-experiment plots.')
    parser.add_argument('--metric', type=str, default='test_acc',
                        help='Metric column for comparison plots.')
    parser.add_argument('--groupby', type=str, default='weight_decay',
                        help='Hyperparameter to group by in sweep_summary.')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Test accuracy threshold to define grokking.')
    parser.add_argument('--no_log_x', action='store_true',
                        help='Use linear x-axis instead of log scale.')
    parser.add_argument('--save', action='store_true',
                        help='Save figures to figures_dir instead of displaying.')
    parser.add_argument('--list', action='store_true',
                        help='List all available experiments and exit.')
    args = parser.parse_args()

    log_x = not args.no_log_x
    show = not args.save

    # --list
    if args.list:
        if not os.path.isdir(args.results_dir):
            print(f"Results directory not found: {args.results_dir}")
            return
        exps = sorted(
            name for name in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, name))
        )
        if not exps:
            print("No experiments found.")
        else:
            print(f"Experiments in {args.results_dir}:")
            for e in exps:
                grok_note = ''
                try:
                    df = load_metrics(args.results_dir, e)
                    grok_step = find_grokking_step(df, threshold=args.threshold)
                    if grok_step:
                        grok_note = f'  → grokked @ step {grok_step:,}'
                    else:
                        grok_note = f'  → no grokking (final test_acc={df["test_acc"].iloc[-1]:.3f})'
                except Exception:
                    grok_note = '  (could not load metrics)'
                print(f"  {e}{grok_note}")
        return

    # Single-experiment plots
    if args.plot in ('grokking_curve', 'loss_curve'):
        if args.exp is None:
            parser.error(f"--exp is required for --plot {args.plot}")
        df = load_metrics(args.results_dir, args.exp)
        save_path = (
            os.path.join(args.figures_dir, args.exp, f'{args.plot}.png')
            if args.save else None
        )
        if args.plot == 'grokking_curve':
            plot_grokking_curve(df, title=args.exp, log_x=log_x,
                                threshold=args.threshold, save_path=save_path, show=show)
        else:
            plot_loss_curve(df, title=args.exp, log_x=log_x,
                            save_path=save_path, show=show)

    # Multi-experiment plots
    elif args.plot == 'comparison':
        metrics_dict = load_all_metrics(args.results_dir, pattern=args.pattern)
        if not metrics_dict:
            print(f"No experiments matched pattern '{args.pattern}'")
            return
        save_path = (
            os.path.join(args.figures_dir, f'comparison_{args.pattern}_{args.metric}.png')
            if args.save else None
        )
        plot_comparison(metrics_dict, metric=args.metric, log_x=log_x,
                        save_path=save_path, show=show)

    elif args.plot == 'sweep_summary':
        metrics_dict = load_all_metrics(args.results_dir, pattern=args.pattern)
        if not metrics_dict:
            print(f"No experiments matched pattern '{args.pattern}'")
            return
        summary = summarize_sweep(
            metrics_dict, args.results_dir,
            groupby=args.groupby, threshold=args.threshold,
        )
        print_sweep_summary(summary, groupby=args.groupby)
        print(summary.to_string(index=False))

        if args.save:
            save_path = os.path.join(args.figures_dir, f'sweep_{args.groupby}.csv')
            os.makedirs(args.figures_dir, exist_ok=True)
            summary.to_csv(save_path, index=False)
            print(f"\nSaved summary CSV: {save_path}")


if __name__ == '__main__':
    main()
