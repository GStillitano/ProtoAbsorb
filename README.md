# GOSTTA — Geometry of Open-Set Test-Time Adaptation

**Thesis.** TENT's entropy minimisation inflates feature norms uniformly for both csID and csOOD samples, collapsing the norm gap that energy-based OOD scores rely on. OOD detection degrades monotonically over the stream regardless of how well the ID/OOD split is performed.

**Goal.** Reproduce TENT, characterise the failure mode geometrically, then either fix it (Path A) or fully characterise the vector-field dynamics (Path B).

---

## Setup

```bash
uv sync
```

Data is downloaded automatically on first run (CIFAR-10-C via RobustBench, SVHN via torchvision). Rome32 is a stub until the dataset is available.

---

## Model

WideResNet-40-2, pretrained with AugMix+JSD on CIFAR-10 (`Hendrycks2020AugMix` via RobustBench).  
Clean acc ≈ 95.8%, mean corruption error ≈ 11.2% on CIFAR-10-C severity 5.  
BN always runs in train mode (batch statistics, no running stats).

---

## Quickstart

```bash
# Validate setup: reproduce published TENT numbers (closed-set, all 15 corruptions)
uv run python scripts/reproduce_tent.py

# Run Phase 1 adaptation for one stream (defaults: gaussian_noise, α=0.5, svhn_c, seed=0)
uv run python scripts/phase1_adapt.py --method tent
uv run python scripts/phase1_adapt.py --method bn_adapt

# Compute metrics (output → results/)
uv run python scripts/exp1_auroc.py \
    --streams tent/gaussian_noise_svhn_c_0.50_seed0 \
              bn_adapt/gaussian_noise_svhn_c_0.50_seed0

uv run python scripts/exp2_geometry.py --stream tent/gaussian_noise_svhn_c_0.50_seed0
uv run python scripts/exp3_layerwise.py --stream tent/gaussian_noise_svhn_c_0.50_seed0

# Render figures (output → figures/)
uv run python scripts/plot.py --exp 1 \
    --streams tent/gaussian_noise_svhn_c_0.50_seed0 \
              bn_adapt/gaussian_noise_svhn_c_0.50_seed0

uv run python scripts/plot.py --exp 2 --streams tent/gaussian_noise_svhn_c_0.50_seed0
uv run python scripts/plot.py --exp 3 --streams tent/gaussian_noise_svhn_c_0.50_seed0

# Full sweep over all corruptions × alphas × seeds
uv run python scripts/sweep.py --method tent
```

---

## Two-phase design

**Phase 1** runs a TTA method on a stream of T batches and saves BN affine state (γ, β) after each step → `checkpoints/`.  
**Phase 2** loads any checkpoint, evaluates on a fixed held-out diagnostic set D, and computes metrics. No re-adaptation.

A **stream** is identified by `(method, corruption, csood_source, α, seed)`. Alpha controls the OOD fraction per batch (0 = closed-set, 0.5 = balanced, etc.).

---

## Repository

```
configs/
  model.yaml          # RobustBench model spec
  stream.yaml         # single-stream defaults (corruption, α, N, T, seed, …)
  tent.yaml           # Adam lr
  diagnostic.yaml     # D size: 2000 csID + 2000 csOOD
  sweep.yaml          # full grid: corruptions × alphas × csood_sources × seeds

src/
  model.py            # load_model(), get_embeddings(), classifier_weights()
  bn_affine.py        # extract / inject / save / load / evaluate  ← Phase 2 primitive
  prototypes.py       # source class centroids μ_c^(0) from θ_0 on D_csID
  data/
    cifar10c.py       # csID loader
    svhnc.py          # csOOD loader (SVHN + corruptions)
    rome32.py         # csOOD loader stub (data not yet available)
    pools.py          # DataPools — seeded adapt / diagnostic split
    stream.py         # build_stream() — frozen T-batch adaptation stream
  tta/
    tent.py           # official TENT, unchanged
  metrics/
    ood_scores.py     # energy_score, max_logit_score, max_softmax_score
    ood_metrics.py    # auroc, fpr_at_tpr, oscr, h_score
    geometry.py       # feature_norms, cosine_to_weights, centroid_distances, …
  viz/
    common.py         # shared style and color palettes
    exp1.py / exp2.py / exp3.py

scripts/
  reproduce_tent.py   # closed-set TENT reproduction, matches published protocol
  phase1_adapt.py     # Phase 1: adapt on stream, save checkpoints
  exp1_auroc.py       # AUROC + csID accuracy trajectory → results.json
  exp2_geometry.py    # norm / cosine / distance / confidence → results.json
  exp3_layerwise.py   # BN affine drift (checkpoint-only, no forward pass) → results.json
  plot.py             # render figures from results JSON → figures/
  sweep.py            # run phase1_adapt over full sweep grid

# gitignored at runtime
checkpoints/          # BN affine checkpoints (theta_*.pt, meta.json, base_model.pt)
results/              # JSON metrics, one subdir per stream × experiment
figures/              # PNG plots produced by scripts/plot.py
data/                 # downloaded datasets
```

## Docs

| File | Contents |
|---|---|
| `theory.md` | Problem setup, notation, norm-inflation hypothesis, experiment definitions |
| `implementation.md` | Two-phase design, evaluation protocol, repo structure, run sequence |
