# GOSTTA — Geometry of Open-Set Test-Time Adaptation

**Thesis.** TENT's entropy minimisation inflates feature norms uniformly for both csID and csOOD samples, collapsing the norm gap that energy-based OOD scores rely on. OOD detection degrades monotonically over the stream regardless of how well the ID/OOD split is performed.

**Goal.** Reproduce TENT, characterise the failure mode geometrically (Experiments 1–2), then fix it with **NOVA-TTA** — a soft-labeled, norm-suppressing open-set TTA method.

---

## Repository

```
configs/          experiment parameters: model spec, stream defaults, sweep grid

src/
  model.py        model loading, feature extraction, classifier head access
  bn_affine.py    BN affine state (γ,β): extract / inject / save / load / evaluate_diagnostic_stream
                  evaluate_diagnostic_stream() is the single Phase 2 primitive
  centroids.py    source class centroids from original model weights on clean CIFAR-10
  data/cifar10.py clean CIFAR-10 test set loader (consistent preprocessing with CIFAR-10-C)
  data/           dataset loaders (csID: CIFAR-10-C; csOOD: SVHN-C, Rome32 stub),
                  seeded adapt/diagnostic pool split, frozen stream builder,
                  shared held-out diagnostic loader (diagnostic.py)
  tta/            TTA methods: TENT (official, unchanged) + NOVA-TTA (the fix)
  metrics/        OOD scores (energy, max-logit, max-softmax),
                  OOD metrics (AUROC, FPR95, OSCR, H-score),
                  geometry (feature norms, cosines, centroid distances)
  viz/            matplotlib plotting modules (exp1, exp2, maxcos_dist) + shared style

scripts/          runnable entry points: adaptation, metric computation, plotting, sweep

# gitignored at runtime
checkpoints/      BN affine checkpoints per stream (theta_*.pt, meta.json, base_model.pt)
results/          JSON metrics, one subdir per stream × experiment
figures/          PNG plots produced by scripts/plot.py
data/             downloaded datasets
```

---

## Setup

```bash
uv sync
```

Data is downloaded automatically on first run (CIFAR-10-C via RobustBench, SVHN via torchvision). Rome32 uses the same sample budget as SVHN's test split and loads at most 26,032 images when the raw folder is populated.

---

## Model

WideResNet-40-2, pretrained with AugMix+JSD on CIFAR-10 (`Hendrycks2020AugMix_WRN` via RobustBench).  
Clean acc ≈ 95.8%, mean corruption error ≈ 11.2% on CIFAR-10-C severity 5.  
BN always runs in train mode (batch statistics, no running stats).

---

## Two-phase design

**Phase 1** runs a TTA method on a stream of T batches and saves BN affine state (γ, β) after each step → `checkpoints/`.  
**Phase 2** loads any checkpoint, evaluates on a fixed held-out diagnostic set D, and computes metrics. No re-adaptation.

A **stream** is identified by `(method, corruption, csood_source, open_set, seed)`. `open_set=True` = balanced ID/OOD (N//2 each); `open_set=False` = closed-set (ID only).

---

## Quickstart

```bash
# Validate setup: reproduce published TENT numbers (closed-set, all 15 corruptions)
uv run python scripts/reproduce_tent.py

# Run Phase 1 adaptation for one stream (defaults: gaussian_noise, open_set=true, svhn_c, seed=0)
uv run python scripts/phase1_adapt.py --method tent
uv run python scripts/phase1_adapt.py --method bn_adapt
uv run python scripts/phase1_adapt.py --method nova-tta

# Compute metrics (output → results/)
uv run python scripts/exp1_auroc.py \
    --streams tent/gaussian_noise_svhn_c_open_seed0 \
              bn_adapt/gaussian_noise_svhn_c_open_seed0 \
              nova-tta/gaussian_noise_svhn_c_open_seed0

uv run python scripts/exp2_geometry.py --stream tent/gaussian_noise_svhn_c_open_seed0
uv run python scripts/maxcos_dist.py   --stream nova-tta/gaussian_noise_svhn_c_open_seed0

# Render figures (output → figures/)
uv run python scripts/plot.py --exp 1 \
    --streams tent/gaussian_noise_svhn_c_open_seed0 \
              bn_adapt/gaussian_noise_svhn_c_open_seed0 \
              nova-tta/gaussian_noise_svhn_c_open_seed0

uv run python scripts/plot.py --exp 2 --streams tent/gaussian_noise_svhn_c_open_seed0

# Full sweep over all corruptions × open_set × seeds
uv run python scripts/sweep.py --method tent
```


