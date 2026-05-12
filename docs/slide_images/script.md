# Speaking script — Li et al. on dense models + grok_hard

Spoken notes for the one slide titled *"Li et al. on dense models — and why GLUE didn't grok."*

---

## Background (15–20s)

The paper we built on is Li et al. (2026), "Grokking in LLM Pretraining." Their big result is that grokking isn't just a toy-task phenomenon — they observed it during real 7B MoE pretraining on OLMoE. Two specific claims matter to us:

1. Grokking is **asynchronous across data domains** — math, code, and commonsense all transition from memorization to generalization at different training steps.
2. Their **pathway metrics** — edit distance between expert-routing paths, and cross-layer consistency — correlate with test accuracy at |ρ| greater than 0.9, while training loss correlates at about zero.

In §5 they explicitly say this should be extended to dense models. That's what we tried.

---

## What we did (15–20s)

We built dense analogues of their MoE pathway metrics: top-k activated neurons per layer act as virtual pathways, cross-layer hidden-state consistency stands in for routing consistency, and we track effective rank of weight matrices.

Then we ran two sweeps:
- **Multitask GLUE** — one transformer trained jointly on RTE, MRPC, CoLA, BoolQ, with per-task accuracy tracked throughout.
- **grok_hard** — a hardened recipe: 28M-parameter transformer, weight decay 1.0, dropout 0.3, label smoothing 0.1, four init scales (1, 2, 4, 8 — testing Liu et al.'s Omnigrok claim), and 200,000 steps.

*(Point to multitask image.)* Per-task curves here track each other instead of diverging — no asynchronous grokking at our scale.

---

## Key result — correlations collapse (30s)

This is the main finding.

On the multitask transformer, the dense pathway edit-distance correlation with test accuracy **collapses to about −0.06** — essentially no signal, where Li et al. see −0.93 on MoE. Pathway consistency is weak at +0.43. And — this is the surprising part — **training loss correlates with test accuracy more strongly than either pathway metric**, at +0.70.

That's an **inversion of Li et al.'s headline ordering**. On dense transformers at our scale, the pathway metrics are not a better generalization signal than loss — they're worse.

Sanity check: on our mod-add positive control (where grokking actually happens), the dense metrics track test accuracy in the right direction. So they work when the phenomenon is present; they're weak, not broken.

---

## grok_hard (15s)

*(Point to the two grok_hard images — init1 on the left, init8 on the right.)*

Across all four init scales, final test accuracy lands in the **0.55 to 0.61 band** — indistinguishable from each other and from our shorter runs. Train accuracy hit 0.99 by step 5,000 in every case. No late grok. Large-norm init, which should accelerate grokking per Omnigrok, didn't rescue it at any scale.

---

## Takeaways (10s)

So: no grokking on real text at our scale. Dense pathway metrics transfer weakly — the ordering inverts. Asynchronous local grokking is absent.

**Next steps:** longer runs up to 1M steps (OLMoE's delays were much longer than ours), scaling to ~100M parameters (closer to where Li et al. observed the phenomenon), structured-text tasks that bridge mod-add's clean-rule advantage, and potentially discrete routing to approximate MoE's signal.

---

**Total spoken time: ~90 seconds.** Can expand the correlation-table section if Q&A pushes on it.
