# GAMBL local sweeps — runbook

Parallel sweep driver for the 2× RTX 4090 setup. Each sweep config is its own
subprocess with `CUDA_VISIBLE_DEVICES` pinned per worker — at most N_GPUs
configs run concurrently.

## Files

- [grok_lib.py](../src/grok_lib.py) — shared: models, datasets, `train_model`, analysis helpers
- [run_single.py](../src/run_single.py) — runs one config; writes `<name>.csv`, `<name>.json`, `<name>.png`
- [run_sweep.py](../src/run_sweep.py) — parallel driver; distributes configs across GPUs
- [sweeps.py](../src/sweeps.py) — sweep definitions (weight_decay, architecture, cross_dataset, train_fraction, smoke)
- [compute_pathway_metrics.py](../src/compute_pathway_metrics.py) — post-hoc Li et al. §4 pathway metrics
- [grokking_text_experiments.ipynb](../notebooks/grokking_text_experiments.ipynb) — results-only notebook
- [grokking_metrics_explained.ipynb](../notebooks/grokking_metrics_explained.ipynb) — metrics walkthrough

## tmux workflow

Run from repo root. Sweep outputs land in `./results/<sweep>/` regardless of cwd
(default `--out-dir` is resolved relative to the repo, not the shell).

```bash
tmux new -s grok

# Smoke test first (~2 min total across both GPUs)
python src/run_sweep.py smoke --gpus 0,1

# Real sweeps
python src/run_sweep.py weight_decay   --gpus 0,1
python src/run_sweep.py architecture   --gpus 0,1
python src/run_sweep.py cross_dataset  --gpus 0,1
python src/run_sweep.py train_fraction --gpus 0,1

# detach: Ctrl-b d    reattach: tmux attach -t grok
```

Watch a specific run's progress from another shell:

```bash
tail -f results/weight_decay/logs/rte_wd1.0.log
nvidia-smi -l 2
```

Sweep is idempotent — if `results/<sweep>/<name>.csv` exists it's skipped, so
you can re-run after a crash.

## Post-hoc pathway analysis

For runs trained with `--checkpoint-every N > 0`:

```bash
python src/compute_pathway_metrics.py --run-dir results/multitask
```

Writes `<run>_pathway.csv` per run plus a summary plot.

## Notebook

```bash
jupyter lab notebooks/grokking_text_experiments.ipynb
```

All cells just read CSVs/PNGs from `../results/`. No training happens in the
notebook. Cells gracefully no-op if a sweep hasn't been run yet.
