# GAMBL local sweeps — runbook

Parallel sweep driver for the 2× RTX 4090 setup. Each sweep config is its own
subprocess with `CUDA_VISIBLE_DEVICES` pinned per worker — at most N_GPUs
configs run concurrently.

## Files

- [grok_lib.py](grok_lib.py) — shared: models, datasets, `train_model`, analysis helpers
- [run_single.py](run_single.py) — runs one config; writes `<name>.csv`, `<name>.json`, `<name>.png`
- [run_sweep.py](run_sweep.py) — parallel driver; distributes configs across GPUs
- [sweeps.py](sweeps.py) — sweep definitions (weight_decay, architecture, cross_dataset, train_fraction, smoke)
- [grokking_text_experiments.ipynb](grokking_text_experiments.ipynb) — results-only notebook

## tmux workflow

```bash
tmux new -s grok

# Smoke test first (~2 min total across both GPUs)
python run_sweep.py smoke --gpus 0,1

# Real sweeps
python run_sweep.py weight_decay   --gpus 0,1
python run_sweep.py architecture   --gpus 0,1
python run_sweep.py cross_dataset  --gpus 0,1
python run_sweep.py train_fraction --gpus 0,1

# detach: Ctrl-b d    reattach: tmux attach -t grok
```

Watch a specific run's progress from another shell:

```bash
tail -f results/weight_decay/logs/rte_wd1.0.log
nvidia-smi -l 2
```

Sweep is idempotent — if `results/<sweep>/<name>.csv` exists it's skipped, so
you can re-run after a crash.

## Notebook

```bash
jupyter lab grokking_text_experiments.ipynb
```

All cells just read CSVs/PNGs from `results/`. No training happens in the
notebook. Cells gracefully no-op if a sweep hasn't been run yet.
