"""Sweep definitions for GAMBL grokking experiments.

Each sweep is a list of config dicts. Keys match run_single.py arguments
(underscores here, converted to --dashes by run_sweep.py).
"""

from __future__ import annotations


_COMMON = {
    'arch': 'transformer',
    'lr': 1e-3,
    'train_fraction': 0.5,
    'n_steps': 50000,
    'batch_size': -1,
    'log_every': 100,
    'max_seq_len': 128,
    'seed': 42,
    'grok_threshold': 0.75,
}


def _make(name: str, **overrides) -> dict:
    cfg = dict(_COMMON)
    cfg.update(overrides)
    cfg['name'] = name
    return cfg


# ---------------------------------------------------------------------------
# Experiment 1: Weight decay sweep on RTE
#   Tests Power et al.'s core claim: weight decay is the critical knob for
#   inducing grokking. 0.0 = no regularization; 1.0 = their sweet spot.
# ---------------------------------------------------------------------------
WEIGHT_DECAY = [
    _make(f'rte_wd{wd}', dataset='rte', weight_decay=wd)
    for wd in [0.0, 0.1, 0.5, 1.0, 5.0]
]


# ---------------------------------------------------------------------------
# Experiment 2: Architecture comparison on RTE at wd=1.0
#   Does attention matter, or is grokking a general property of
#   overparameterized nets + strong regularization?
# ---------------------------------------------------------------------------
ARCHITECTURE = [
    _make('rte_transformer', dataset='rte', arch='transformer', weight_decay=1.0),
    _make('rte_mlp', dataset='rte', arch='mlp', weight_decay=1.0),
]


# ---------------------------------------------------------------------------
# Experiment 3: Cross-dataset comparison at wd=1.0
#   Li et al. observation: different data domains grok at different rates.
#   Do different text tasks show different memorization->generalization lags?
# ---------------------------------------------------------------------------
CROSS_DATASET = [
    _make(f'{ds}_wd1.0', dataset=ds, weight_decay=1.0)
    for ds in ['rte', 'mrpc', 'cola', 'boolq']
]


# ---------------------------------------------------------------------------
# Experiment 4: Train fraction sweep on RTE at wd=1.0
#   Power et al. observation: smaller training fractions -> faster grokking.
# ---------------------------------------------------------------------------
TRAIN_FRACTION = [
    _make(f'rte_frac{frac}', dataset='rte', weight_decay=1.0, train_fraction=frac)
    for frac in [0.2, 0.3, 0.5, 0.7]
]


# ---------------------------------------------------------------------------
# Multi-task (Li et al. §3.2 asynchronous local grokking)
#
# Train ONE model on a concatenation of RTE + MRPC + CoLA + BoolQ, eval
# per-task, look for different grokking onsets across tasks. This is the
# direct test of Li et al.'s core empirical claim: different data domains
# transition from memorization to generalization at different steps within
# the same training trajectory.
#
# --checkpoint-every 2500 saves model state_dicts so we can later compute
# dense-analogue pathway metrics (pathway edit distance + consistency +
# effective rank) and correlate them with test accuracy, applying Li et al.
# §4 to dense architectures as their conclusion explicitly invites.
# ---------------------------------------------------------------------------
MULTITASK = [
    _make('multitask_transformer_wd1.0', dataset='multitask', arch='transformer',
          weight_decay=1.0, batch_size=512, checkpoint_every=2500),
    _make('multitask_mlp_wd1.0', dataset='multitask', arch='mlp',
          weight_decay=1.0, batch_size=512, checkpoint_every=2500),
]


# ---------------------------------------------------------------------------
# Modular arithmetic CONTROL — sanity-check that our pipeline reproduces
# Power et al.'s canonical grokking. If `mod_add_wd1.0` groks (test acc rising
# from ~0 to ~1 *after* train saturates), the pipeline is correct and the
# text negative result is meaningful. If it doesn't, there's a pipeline bug.
#
# mod_add with p=97, train_fraction=0.5, wd=1.0 is Power et al.'s Figure 1.
# They report grokking at ~10^4 steps, so 30k is a comfortable budget.
# mod_add_wd0.0 is the matching control — should NOT grok.
# ---------------------------------------------------------------------------
MODULAR_ARITHMETIC = [
    # Checkpoints enable the "positive control" pathway analysis:
    # when grokking DOES happen (mod-add), do our dense pathway metrics
    # correlate strongly with test accuracy? This is the direct analogue of
    # Li et al. Table 1's strong positive correlation on LLM pretraining.
    _make('mod_add_wd1.0', dataset='mod_add', weight_decay=1.0, n_steps=30000,
          log_every=50, checkpoint_every=1500),
    _make('mod_add_wd0.0', dataset='mod_add', weight_decay=0.0, n_steps=30000,
          log_every=50, checkpoint_every=1500),
]


# ---------------------------------------------------------------------------
# GROK_HARD: Omnigrok init-scale sweep on RTE+MRPC+CoLA+BoolQ multitask.
#
# Our earlier single-task sweeps all failed to grok on GLUE text at the small
# Power-et-al. model scale, with max improvement_after_mem of +0.023. This
# sweep combines the remaining untried interventions:
#
#   1. Larger model (~25-30M params) — moves into overparameterized regime
#      relative to the natural-language task, which is what Li et al.
#      observed grokking in (they used 7B; we scale up modestly).
#   2. Omnigrok init-scale (Liu et al. 2022) — scale all weight matrices
#      by alpha at init. alpha=1 is the baseline, alpha={2,4,8} are the
#      interventions. Liu et al. showed this induces grokking on tasks
#      that don't otherwise grok.
#   3. Regularization stack — high weight decay (1.0), dropout (0.3),
#      and label smoothing (0.1) all pushing toward simpler solutions.
#   4. Multitask — trains on RTE+MRPC+CoLA+BoolQ jointly so each run is
#      also a test of Li et al. §3.2 asynchronous local grokking.
#   5. Inline pathway metrics every 2000 steps — Li et al. §4 sensor that
#      should move before test accuracy does.
#   6. Flat-metric early kill: after step 40000, if both pathway_edit_dist
#      and pathway_consistency have moved less than 1% relative range over
#      the last 10 pathway measurements (~20k training steps of signal),
#      the run exits. Saves GPU time on dead configs.
#
# If ANY of the four init-scale variants shows pathway metrics that keep
# moving past step 40000, extend that run to 500k+ steps — grokking is
# possible but slow.
# ---------------------------------------------------------------------------
_GROK_HARD_COMMON = dict(
    dataset='multitask', arch='transformer',
    weight_decay=1.0, dropout=0.3, label_smoothing=0.1,
    d_model=512, n_heads=8, n_layers=4, d_ff=2048,
    n_steps=200000, batch_size=512, lr=3e-4,
    log_every=500,
    pathway_every=2000, pathway_probe_size=32,
    checkpoint_every=10000,  # also save checkpoints for post-hoc analysis
    flat_kill_start_after=40000, flat_kill_window=10, flat_kill_min_delta=0.01,
)

GROK_HARD = [
    _make(f'grok_hard_init{a}', init_scale=float(a), **_GROK_HARD_COMMON)
    for a in [1, 2, 4, 8]
]


# ---------------------------------------------------------------------------
# Smoke test: 500 steps per config, for validating the pipeline end-to-end.
# ---------------------------------------------------------------------------
SMOKE = [
    _make('smoke_rte_wd1.0', dataset='rte', weight_decay=1.0, n_steps=500, log_every=50),
    _make('smoke_rte_wd0.0', dataset='rte', weight_decay=0.0, n_steps=500, log_every=50),
    _make('smoke_mod_add', dataset='mod_add', weight_decay=1.0, n_steps=500, log_every=25),
    # Exercises every new knob (init_scale, dropout, label_smoothing, inline
    # pathway metrics, flat-kill) on a small budget so sweep-integration bugs
    # surface before we commit to 200k-step runs.
    _make('smoke_grok_hard', dataset='multitask', arch='transformer',
          weight_decay=1.0, dropout=0.3, label_smoothing=0.1,
          d_model=128, n_heads=4, n_layers=2, d_ff=512,
          init_scale=2.0, batch_size=128, lr=3e-4,
          n_steps=400, log_every=50,
          pathway_every=100, pathway_probe_size=16,
          flat_kill_start_after=200, flat_kill_window=3, flat_kill_min_delta=0.01,
          max_dataset_size=200),
]


SWEEPS: dict[str, list[dict]] = {
    'weight_decay': WEIGHT_DECAY,
    'architecture': ARCHITECTURE,
    'cross_dataset': CROSS_DATASET,
    'train_fraction': TRAIN_FRACTION,
    'modular_arithmetic': MODULAR_ARITHMETIC,
    'multitask': MULTITASK,
    'grok_hard': GROK_HARD,
    'smoke': SMOKE,
}
