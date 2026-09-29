# MLP Overfit (tabular) — Séance 1

Pedagogical lab for **Master 2 Finance, Data et IA** — Deep Learning course.

**Repo:** https://github.com/Amzil-AI/mlp-overfit-tabular

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Amzil-AI/mlp-overfit-tabular/blob/main/mlp_overfit_colab.ipynb)

Atelier slide: *Faire surapprendre un petit jeu*.

Train a wide MLP on a **non-financial** tabular dataset and show that it can stick to the train set.

1. **Regression** — California Housing, MSE  
2. **Binary classification** — Breast Cancer, BCE with logits  
3. Plot **train and validation on the same chart**

No dropout, no weight decay — the point is to see the gap grow.

## Fastest path (Colab)

1. Open the badge above (or [this Colab link](https://colab.research.google.com/github/Amzil-AI/mlp-overfit-tabular/blob/main/mlp_overfit_colab.ipynb))
2. Runtime → Run all
3. Read the two charts: where does train / validation diverge?

CPU runtime is enough (~1–2 minutes).

## Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_mlp_overfit.py
```

## What you should see

| Task | Typical pattern |
|---|---|
| Housing (MSE) | Train keeps falling; val flattens / rises; gap clear early |
| Cancer (BCE) | Train → ~0; val stays higher — classic memorize-train |

You must be able to say **whether** the gap grows and **from which epoch**.

## Outputs

| Path | Content |
|---|---|
| `figures/regression_curves.png` | Housing MSE: train vs validation |
| `figures/classification_curves.png` | Cancer BCE: train vs validation |
| `outputs/metrics.json` | Final losses + approximate diverge epoch |

## Course context

Warm-up before finance / time series. Isolates feed-forward mechanics and learning-curve reading. Séance 2 reuses the same MLP and adds learning rate + dropout.
