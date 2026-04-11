# GaMBL: Grokking across Models, Benchmarks, and LLMs

A research codebase for studying the **grokking** phenomenon — delayed generalization where a model memorizes training data first, then later transitions sharply to high test accuracy — across MLP and transformer architectures.

---

## Setup

Activate the conda environment before running any script:

```bash
conda activate nanopore
```

All scripts are run from the project root (`/scratch1/rnene/csci567/`).

---

## Project Structure

```
csci567/
├── data/
│   └── modular_arithmetic.py   # Dataset generation
├── models/
│   ├── mlp.py                  # MLP architecture
│   └── transformer.py          # Encoder transformer architecture
├── train.py                    # Main training script
├── run_experiment.py           # Hyperparameter sweep runner
├── analyze.py                  # Plotting and analysis
├── configs/
│   ├── base.yaml               # Default hyperparameters (Power et al. 2022)
│   ├── transformer_modular.yaml
│   ├── mlp_modular.yaml
│   ├── sweep_weight_decay.yaml
│   ├── sweep_architecture.yaml
│   └── sweep_train_fraction.yaml
└── results/                    # Auto-created; one subdirectory per experiment
    └── <exp_name>/
        ├── config.json
        ├── metrics.csv
        └── checkpoints/
            └── step_XXXXXXX.pt
```

---

## Running a Single Experiment

```bash
python train.py [OPTIONS]
```

### Common examples

**Canonical grokking reproduction** (transformer, mod 97, addition, Power et al. defaults):
```bash
python train.py --model transformer --n_steps 100000
```

**MLP on the same task** (for architecture comparison):
```bash
python train.py --model mlp --n_steps 100000
```

**Control: no weight decay** (grokking should NOT occur):
```bash
python train.py --model transformer --weight_decay 0.0 --n_steps 100000
```

**Smaller training set** (grokking onset is earlier):
```bash
python train.py --model transformer --train_fraction 0.3 --n_steps 100000
```

**Smoke test** (completes in ~2 minutes on CPU):
```bash
python train.py --n_steps 500 --log_every 10 --checkpoint_every 100
```

### Full list of options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `transformer` | Architecture: `transformer` or `mlp` |
| `--dataset` | `modular_arithmetic` | Dataset to use |
| `--prime` | `97` | Prime modulus *p* |
| `--operation` | `+` | Arithmetic op: `+`, `-`, `*`, `/` |
| `--train_fraction` | `0.5` | Fraction of all *p²* pairs used for training |
| `--d_model` | `128` | Transformer model dimension |
| `--n_layers` | `2` | Number of transformer blocks |
| `--n_heads` | `4` | Number of attention heads |
| `--d_ff` | `512` | Transformer feed-forward dimension |
| `--dropout` | `0.1` | Dropout probability |
| `--pool` | `last` | Transformer pooling: `last` or `mean` |
| `--no_pos_encoding` | off | Disable positional encoding |
| `--embed_dim` | `128` | MLP embedding dimension per token |
| `--hidden_dim` | `512` | MLP hidden layer width |
| `--num_mlp_layers` | `3` | MLP total linear layers (including output) |
| `--activation` | `relu` | MLP activation: `relu`, `gelu`, `tanh` |
| `--lr` | `1e-3` | AdamW learning rate |
| `--weight_decay` | `1.0` | AdamW weight decay — **critical for grokking** |
| `--n_steps` | `100000` | Total gradient steps |
| `--batch_size` | `-1` | Mini-batch size; `-1` = full-batch training |
| `--log_every` | `100` | Log metrics every N steps |
| `--checkpoint_every` | `500` | Save checkpoint every N steps |
| `--seed` | `42` | Random seed (controls both data split and weight init) |
| `--exp_name` | `auto` | Experiment name; `auto` generates from hyperparams |
| `--results_dir` | `results` | Root output directory |

### Output files

Each run creates `results/<exp_name>/`:

- **`config.json`** — all hyperparameters for reproducibility
- **`metrics.csv`** — columns: `step, train_loss, train_acc, test_loss, test_acc`; written every `--log_every` steps
- **`checkpoints/step_XXXXXXX.pt`** — model + optimizer state; saved every `--checkpoint_every` steps

---

## Running Hyperparameter Sweeps

```bash
python run_experiment.py <sweep_config.yaml> [OPTIONS]
```

### Provided sweep configs

| Config | What it sweeps |
|--------|---------------|
| `configs/sweep_weight_decay.yaml` | weight_decay ∈ {0, 0.1, 0.5, 1.0, 5.0} × 3 seeds = 15 jobs |
| `configs/sweep_architecture.yaml` | model ∈ {transformer, mlp} × 3 seeds = 6 jobs |
| `configs/sweep_train_fraction.yaml` | train_fraction ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.8} × 3 seeds = 18 jobs |

### Examples

```bash
# Dry run — print all commands without executing:
python run_experiment.py configs/sweep_weight_decay.yaml --dry-run

# Sequential execution:
python run_experiment.py configs/sweep_weight_decay.yaml

# Parallel execution (4 jobs at once, useful on a compute node):
python run_experiment.py configs/sweep_weight_decay.yaml --parallel 4
```

Runs are **idempotent**: if a job's `metrics.csv` already exists and is complete, it is skipped automatically.

### Writing a custom sweep config

```yaml
base_config: configs/base.yaml    # baseline hyperparameters to inherit
sweep:
  weight_decay: [0.0, 1.0]        # any train.py flag can be swept
  seed: [0, 1, 2]                 # Cartesian product is taken automatically
```

Any key not listed under `sweep:` is treated as a fixed override on top of the base config.

---

## Analyzing Results

```bash
python analyze.py [OPTIONS]
```

### List all experiments

```bash
python analyze.py --list
```

Prints each experiment name, whether grokking was detected, and the grokking step.

### Plot a single grokking curve

```bash
python analyze.py --plot grokking_curve \
    --exp transformer_mod97_plus_wd1.0_frac0.5_seed42
```

Shows train and test accuracy vs. training steps (log-scale x-axis by default), with a vertical line marking the grokking point.

### Compare multiple experiments

```bash
# All weight-decay variants for one seed:
python analyze.py --plot comparison \
    --pattern "transformer_mod97_plus_wd*_frac0.5_seed42" \
    --metric test_acc
```

### Sweep summary table

```bash
python analyze.py --plot sweep_summary \
    --pattern "transformer_mod97_plus_wd*" \
    --groupby weight_decay
```

Prints mean ± std of grokking step grouped by the specified hyperparameter.

### Save figures to disk instead of displaying

```bash
python analyze.py --plot grokking_curve \
    --exp transformer_mod97_plus_wd1.0_frac0.5_seed42 \
    --save
```

Figures are saved under `figures/<exp_name>/`.

### Full list of analyze.py options

| Flag | Default | Description |
|------|---------|-------------|
| `--plot` | `grokking_curve` | Plot type: `grokking_curve`, `loss_curve`, `comparison`, `sweep_summary` |
| `--exp` | — | Experiment name (required for single-experiment plots) |
| `--pattern` | `*` | Glob pattern for multi-experiment plots |
| `--metric` | `test_acc` | Metric column for `comparison` plots |
| `--groupby` | `weight_decay` | Hyperparameter to group by in `sweep_summary` |
| `--threshold` | `0.95` | Test accuracy threshold defining grokking |
| `--no_log_x` | off | Use linear x-axis |
| `--save` | off | Save figures to `figures/` instead of displaying |
| `--list` | off | List all experiments and exit |

---

## Key Concepts

### Why weight decay matters

Weight decay is the single most important factor for grokking. Power et al. (2022) use `weight_decay=1.0` with AdamW — roughly 100–1000× higher than typical deep learning. Without high weight decay, the model memorizes and stays memorized. The codebase uses `AdamW` (not `Adam`) throughout; the decoupled weight decay in AdamW is not equivalent to L2 regularization in Adam.

### Full-batch vs. mini-batch

The canonical setup uses full-batch training (`--batch_size -1`), meaning every gradient step processes the entire training set. This produces the sharpest grokking phase transition. Mini-batch training (`--batch_size 512`, etc.) introduces noise that can blur the transition timing, which is itself an interesting thing to study.

### Training duration

Grokking happens **long after** 100% training accuracy is reached — often thousands to tens of thousands of steps later. Always set `--n_steps` to at least 100,000 for reliable results. Do not stop training early when train accuracy reaches 1.0.

### Reading metrics during a run

The `metrics.csv` file is flushed after every logged step, so you can monitor a running experiment:

```bash
tail -f results/<exp_name>/metrics.csv
```

### Resuming from a checkpoint

Checkpoints save full optimizer state. To resume (not currently automated), load the checkpoint dict:

```python
ckpt = torch.load('results/<exp>/checkpoints/step_0050000.pt')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
# then continue training from ckpt['step']
```

---

## References

- Power et al. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* https://arxiv.org/abs/2201.02177
- Nanda et al. (2023). *Progress measures for grokking via mechanistic interpretability.* https://arxiv.org/abs/2301.05217 (useful for the interpretability extension)
- Liu et al. (2023). https://arxiv.org/abs/2306.13253
- https://arxiv.org/html/2502.01774v1
