# GAMBL: Grokking across Models, Benchmarks, and LLMs

Code, configs, and per-run output for the project report
*GaMBL: Grokking across Models, Benchmarks, and LLMs* (Tsai, Liang, Wang, Nene).
We study when delayed generalization ("grokking") arises across MLPs and small
transformers, on modular arithmetic, a structured analogy task, and natural-language
classification benchmarks, and test whether the pathway-based metrics of Li et al.
(2024) transfer from Mixture-of-Experts models to dense transformers.

## Setup

```bash
pip install -r requirements.txt   # torch, numpy, pandas, matplotlib, pyyaml
```

For the natural-language tasks you also need `datasets` and `transformers` from
HuggingFace:

```bash
pip install datasets transformers
```

Smoke test (under two minutes on a single GPU):

```bash
python src/run_sweep.py smoke --gpus 0
```

This runs four short (500-step) configs --- two RTE, one modular addition, one
multitask transformer --- and writes outputs to `results/smoke/`. If it completes
with `SWEEP SUMMARY` and no `FAIL`s, the install is good. Pass
`--gpus 0,1` to parallelize across two GPUs.

## Reproducing the paper

Each main-body result maps to one command below. Outputs land in
`results/<sweep>/` (CSV per run, PNG training curve, JSON config, optional
pathway-metric CSV).

### Section 5 — modular-arithmetic reproduction

```bash
# Transformer reproduction (mod 97 addition, train fraction 0.3):
python run_experiment.py configs/transformer_modular.yaml

# MLP reproduction:
python run_experiment.py configs/mlp_modular.yaml
```

### Section 6.1 — training-split sweep

```bash
python run_experiment.py configs/sweep_train_fraction.yaml
```

### Section 6.2 — weight-decay sweep (Figure 1)

```bash
python run_experiment.py configs/sweep_weight_decay.yaml
```

The Figure 1 panels were rendered from these CSVs by
`figures/ModArith_and_NLP_plots.ipynb`.

### Section 7.1 — structured analogy task

```bash
# Weight-decay sweep at small train fractions (0.04–0.06):
python run_experiment.py configs/sweep_analogy_weight_decay.yaml
python run_experiment.py configs/sweep_analogy_train_fraction.yaml
```

### Section 7.2 — natural-language classification sweeps

The appendix figures (CoLA / MRPC / RTE) come from these sweeps:

```bash
python run_experiment.py configs/sweep_text_weight_decay.yaml
python run_experiment.py configs/sweep_text_train_fraction.yaml
python run_experiment.py configs/sweep_text_datasets.yaml
```

### Section 8 — pathway metrics (Table 2 + the negative-transfer result)

```bash
# Multitask-GLUE transformer + pathway-metric logging:
python src/run_sweep.py multitask

# grok_hard recipe (Omnigrok init-scale sweep on the repeating-subsequence task):
python src/run_sweep.py grok_hard

# Post-hoc pathway analysis on the multitask run's checkpoints:
python src/compute_pathway_metrics.py --run-dir results/multitask
```

The correlations in Table 2 are computed in
`notebooks/grokking_text_experiments.ipynb` from
`results/multitask/multitask_transformer_wd1.0_pathway.csv`.

### Sweep driver options

`src/run_sweep.py` accepts one of:
`{weight_decay, architecture, cross_dataset, train_fraction, modular_arithmetic, multitask, grok_hard, smoke}`,
plus `--gpus 0,1` to parallelize one subprocess per GPU. Sweeps are idempotent
(re-running skips completed configs).

## Notebooks

* `notebooks/grokking_text_experiments.ipynb` — loads CSVs from `results/`,
  produces the main-body figures, and computes the Table 2 correlations.
* `notebooks/grokking_metrics_explained.ipynb` — walkthrough of the metrics
  we track (train/test accuracy, train/test loss, dense pathway metrics) with
  concrete examples from our runs.

Both notebooks are read-only with respect to `results/` — no training happens
inside them. Cells gracefully no-op if a sweep hasn't been run yet.

## Datasets

| Dataset | Source | Used in |
|---|---|---|
| Modular arithmetic (mod 97 +, −, ×, ÷) | generated in-repo (`data/modular_arithmetic.py`) | §5, §6 |
| Analogy task | `data/analogy.py` + `data/questions-words.csv` | §7.1 |
| Repeating subsequence | generated in-repo (`data/subsequence.py`) | §7.2, §8 (`grok_hard`) |
| GLUE (RTE / MRPC / CoLA / SST-2) | HuggingFace `datasets` | §7.2, §8 (multitask) |
| SuperGLUE BoolQ | HuggingFace `datasets` | §7.2, §8 (multitask) |
| TREC | HuggingFace `datasets` | §7.2 |
| Ruletaker | `data/ruletaker_dataset.py` (HuggingFace) | §7.2 |

HuggingFace datasets are downloaded on first use and cached under the default
`~/.cache/huggingface/` location.

## Repository layout

```
src/                Freddie's parallel sweep driver + shared library
  grok_lib.py       models, datasets, train_model, analysis helpers
  run_sweep.py      parallel sweep driver
  run_single.py     single-config runner used by run_sweep
  sweeps.py         8 sweep definitions
  compute_pathway_metrics.py   post-hoc pathway analysis

train.py            original Power-et-al-style training loop
run_experiment.py   config-driven runner (loads configs/*.yaml)
configs/            sweep + dataset configs (modular, analogy, text, sweep_*)
data/               dataset adapters (modular, text, analogy, subsequence,
                    ruletaker, scan, csv, polynomial)
models/             MLP and transformer variants
                    (transformer, transformer_decoder, transformer_encoder,
                    transformer_lm)
notebooks/          results aggregation + metrics walkthrough
figures/            modular-arithmetic plotting notebook (Figure 1)
docs/               final-report slides, slide_images/, references/
results/            per-run outputs (gitignored)
```

## Sanity checks

The merged tree has been verified end-to-end at the import and data-pathway
level: all four contributors' dataset loaders construct valid splits, every
Python file parses, and all three `src/` entry points (`run_single`,
`run_sweep`, `compute_pathway_metrics`) expose a working `--help`. The smoke
sweep above is the fastest way to verify the train loop end-to-end.
