# The Geometry of Open-Set Test-Time Adaptation

Code for the project **"The Geometry of Open-Set Test-Time Adaptation"**
(Leonardo Sani, Giuseppe Stillitano, Can Lin, Xavier Del Giudice — Sapienza University of Rome).

---

## TL;DR

Test-Time Adaptation (TTA) keeps a deployed classifier accurate under covariate shift, but a
realistic stream is **open-set**: alongside corrupted known-class inputs (**csID**) it carries
semantically novel ones (**csOOD**) the model must *reject*. The standard method **TENT**
(entropy minimisation on batch-norm affine parameters) recovers csID accuracy — but its
energy-score OOD detection decays **monotonically** over the stream.

We trace *why*, and fix it:

- **Diagnosis (geometric, not the obvious one).** With the linear head frozen, TENT can raise the
  winning logit only by reshaping the embedding through two levers: **norm inflation** (‖g(x)‖↑)
  and **re-alignment** (angle to the class-weight vector ↓). The feature-**norm** gap between csID
  and csOOD actually *stays intact and widens* — so the norm is **not** what breaks detection. The
  damage is **angular**: csOOD embeddings rotate onto the nearest class weight and get **absorbed**
  into the source classes, collapsing the alignment gap that separated the two populations at `t=0`.
  Because the energy score is dominated by that angular logit term, AUROC collapses while accuracy rises.
- **Fix — NOVA-TTA (Norm-Oriented Vector Alignment).** Score each input with a **frozen,
  norm-invariant** alignment statistic (max cosine to the *original* class weights) — a signal
  adaptation cannot corrupt. Turn pooled scores into a soft OOD posterior with a 2-component GMM, then
  split the loss: entropy minimisation weighted by `P(ID)` (recover accuracy) **plus** an `L1`
  feature-norm penalty weighted by `P(OOD)` (suppress csOOD norms, re-opening an energy margin).
  Detection now **rises** instead of falling, at matched accuracy.

| Setting (gaussian_noise) | csID Acc `0→T` | Energy AUROC `0→T` |
|---|---|---|
| **TENT** (SVHN-C, T=80) | 0.77 → 0.81 | 0.79 → **0.65** ↓ |
| **NOVA-TTA** (SVHN-C, T=80) | 0.77 → 0.81 | 0.79 → **0.93** ↑ |
| **TENT** — 15-corruption mean | 0.82 → 0.84 | 0.80 → **0.66** ↓ |
| **NOVA-TTA** — 15-corruption mean | 0.82 → 0.85 | 0.80 → **0.90** ↑ |

The same finding replicates on a second, semantically distinct csOOD source (Rome32, `T=20`):
TENT mean AUROC 0.74 → 0.73, NOVA-TTA 0.75 → **0.84**, at matched accuracy. See the paper for the
per-corruption tables.

---

## The open-set TTA pipeline

Every open-set TTA method in this repo factors into four stages — NOVA-TTA is one concrete instance:

```
score s(x)  ──►  posterior π(OOD|x)  ──►  labelling ℓ  ──►  loss
   │                    │                      │                │
frozen max-cosine   2-Gaussian GMM         soft weight     P(ID)·entropy(ID)
to class weights    over pooled scores     π(OOD|x)      + P(OOD)·λ·L1(feat)
```

NOVA-TTA's per-step loss on the **adapted** model (one gradient step on the BN affine params):

```
L_t = (1/N) Σ_i [ π(ID|x_i)·H(p_i)  +  π(OOD|x_i)·λ·‖g_θ(x_i)‖_1 ]      λ = l1_weight ≈ 0.03
```

- **Score** comes from a **frozen** copy of the source model (`θ_0`, no gradient, BN uses batch
  stats): `s_i = max_k cos(g_{θ_0}(x_i), w_k)`. Norm-invariant, so it is immune to the drift that
  corrupts the energy score.
- **Posterior** fits a 2-component Gaussian mixture (EM, scikit-learn) to all scores pooled up to
  step `t`; the higher-mean component is ID. Bayes' rule gives the detached soft weight `π(OOD|x)`.
- **Warm-up.** The score pool is small early on, so the learning rate is linearly ramped over the
  first `K=10` steps.

**Baselines** (same two-phase harness, same stream): **TENT** (mean-batch entropy minimisation) and
**BN-Adapt** (batch-norm statistic recalibration only — *no* gradient step; this is the `t=0`
reference point on every plot).

---

## Two-phase design

A clean split keeps every metric comparable across adaptation time.

- **Phase 1 — adapt** (`scripts/phase1_adapt.py`): run a method on a stream of `T` batches; after
  each step save the BN affine state `θ_t = (γ, β)` → `checkpoints/`. Nothing else changes (conv
  weights and the linear head stay frozen; BN runs on batch statistics throughout).
- **Phase 2 — evaluate** (`scripts/exp1_auroc.py`, `exp2_geometry.py`, `maxcos_dist.py`): load any
  `θ_t`, inject it into a frozen-eval model, and score it on a **fixed held-out diagnostic set**
  `D` (2000 csID + 2000 csOOD, disjoint from the adaptation pool, keyed by seed). Evaluation forwards
  `D` in mixed `N=200` batches (100 csID + 100 csOOD) so BN statistics match the adaptation regime.
  **No re-adaptation happens in Phase 2.**

A **stream** is identified by `(method, corruption, csood_source, open_set, seed)` and lives at
`checkpoints/<method>/<corruption>_<csood_source>_<open|closed>_seed<seed>/`. `open_set=true` =
balanced csID/csOOD batches (csOOD per batch = `N//2`, or `n_ood` if overridden); `open_set=false` =
closed-set (csID only).

---

## Repository layout

```
configs/             experiment parameters (see "Configuration" below)

src/
  model.py           RobustBench model loading, feature/logit extraction, classifier-weight access
  bn_affine.py       BN affine state (γ,β): extract / inject / save / load
                     + evaluate_diagnostic_stream() — the single Phase-2 evaluation primitive
  centroids.py       source-class centroids μ_c from original weights on clean CIFAR-10
  seed.py, device.py reproducibility + device selection helpers
  data/
    cifar10c.py      csID stream: CIFAR-10-C (via RobustBench), 15 corruptions
    svhnc.py         csOOD stream: SVHN test set + the same corruption pipeline (SVHN-C)
    rome32.py        csOOD stream: Rome32 32×32 scene patches + corruption pipeline (optional)
    cifar10.py       clean CIFAR-10 test set (for centroids)
    pools.py         seeded, disjoint adapt/diagnostic split
    stream.py        frozen adaptation stream builder (zero within-epoch repeats)
    diagnostic.py    shared held-out diagnostic loader (single source of truth for Phase 2)
  tta/
    tent.py          TENT (official implementation, unchanged) + BN-Adapt path
    nova_tta.py      NOVA-TTA: GmmScorer, soft-labeled loss, LR warmup, model config
  metrics/
    ood_scores.py    energy, max-logit, max-softmax
    ood_metrics.py   AUROC, FPR@95, OSCR, H-score
    geometry.py      feature norms (L1/L2), cosine & max-cosine to weights, centroid distance
  viz/               matplotlib/seaborn plotting (exp1, exp2, maxcos_dist) + shared style

scripts/             runnable entry points (adaptation, metrics, plots, sweeps, TENT reproduction)
report/              LaTeX source (main.tex) and figures for the write-up

# produced at runtime, gitignored
checkpoints/         BN affine checkpoints per stream (theta_NNN.pt, base_model.pt, meta.json)
results/             JSON metrics, one subtree per stream × experiment
figures/             PNG plots
data/                downloaded datasets
```

---

## Setup

```bash
uv sync
```

Requires Python ≥ 3.13. Key dependencies: PyTorch + torchvision, RobustBench (pretrained model +
CIFAR-10-C), scikit-learn (the GMM), `imagecorruptions` (the corruption pipeline applied to SVHN and
Rome32), matplotlib/seaborn, PyYAML.

### Data

- **CIFAR-10-C** (csID) and **SVHN** (csOOD) download automatically on first run, via RobustBench and
  torchvision. SVHN-C is produced on the fly by applying each CIFAR-10-C corruption to SVHN.
- **Rome32** (alternative csOOD) is **not bundled** (gitignored). It is a custom source of 32×32 Roman
  cultural-heritage patches in five scene categories plus a distractor bucket (`affreschi`,
  `fontanelle`, `monete`, `pasta`, `statue`, `_ood`); see Appendix B of the paper for how it was
  built. To use it, place the `export32/` tree at
  `data/rome32/export32/{affreschi,fontanelle,monete,pasta,statue,_ood}/`. The loader skips zero-byte
  placeholder files and yields ~2,597 usable images. Rome32 runs use an adapted stream config
  (`configs/stream_rome32.yaml`: `T=20`, `n_ood=25`, `α=0.125`), selected automatically when
  `--csood_source rome32`.

### Model

WideResNet-40-2 pretrained with AugMix+JSD on CIFAR-10 (`Hendrycks2020AugMix_WRN`, from the
RobustBench model zoo). Clean accuracy ≈ 95.8%; mean CIFAR-10-C severity-5 error ≈ 11.2%.
Batch norm always runs in train mode (batch statistics, no running stats) — TENT adapts only the
BN affine pair `(γ, β)`, which is where the last-layer geometry lives.

---

## Quickstart

```bash
# 0. Sanity check: reproduce published TENT numbers (closed-set, all 15 corruptions, single pass).
uv run python scripts/reproduce_tent.py            # target: ~11.2% mean error

# 1. Phase 1 — adapt one stream (defaults: gaussian_noise, open_set=true, svhn_c, seed=0).
uv run python scripts/phase1_adapt.py --method tent
uv run python scripts/phase1_adapt.py --method bn_adapt
uv run python scripts/phase1_adapt.py --method nova-tta

# 2. Phase 2 — metrics (written to results/).
#    Exp 1: energy-score AUROC + csID accuracy trajectory.
uv run python scripts/exp1_auroc.py \
    --streams tent/gaussian_noise_svhn_c_open_seed0 \
              bn_adapt/gaussian_noise_svhn_c_open_seed0 \
              nova-tta/gaussian_noise_svhn_c_open_seed0

#    Exp 2: geometry over the stream (norms, alignment, centroid distance, OOD confidence).
uv run python scripts/exp2_geometry.py --stream tent/gaussian_noise_svhn_c_open_seed0

#    max-cosine score distribution at θ_0 (the frozen signal NOVA-TTA scores on).
uv run python scripts/maxcos_dist.py   --stream nova-tta/gaussian_noise_svhn_c_open_seed0

# 3. Figures (written to figures/).
uv run python scripts/plot.py --exp 1 \
    --streams tent/gaussian_noise_svhn_c_open_seed0 \
              bn_adapt/gaussian_noise_svhn_c_open_seed0 \
              nova-tta/gaussian_noise_svhn_c_open_seed0
uv run python scripts/plot.py --exp 2 --streams tent/gaussian_noise_svhn_c_open_seed0
```

### Sweeps

```bash
# Full grid for one method: 15 corruptions × {open, closed} × {svhn_c, rome32} × seeds.
# (rome32 closed-set runs are skipped — closed-set never touches the csOOD pool.)
uv run python scripts/sweep.py --method tent

# Rome32 convenience wrappers (bash): adapt all corruptions, then score all three methods.
bash scripts/run_rome32_sweep.sh nova-tta
bash scripts/run_rome32_exp1.sh

# Aggregate exp1 results across the 15 corruptions into a table + ready-to-paste LaTeX.
uv run python scripts/aggregate_corruptions.py --csood_source svhn_c
```

`phase1_adapt.py` accepts overrides for every stream knob: `--corruption --severity --open_set
--csood_source --N --T --n_ood --seed --lr --device`.

---

## Configuration

| File | Purpose |
|---|---|
| `configs/model.yaml` | RobustBench model + csID dataset selection |
| `configs/stream.yaml` | default stream: `gaussian_noise`, severity 5, `N=200`, `T=80`, open-set, SVHN-C, seed 0 |
| `configs/stream_rome32.yaml` | Rome32-adapted stream (`T=20`, `n_ood=25`, `α=0.125`) for the smaller csOOD pool |
| `configs/diagnostic.yaml` | held-out diagnostic split sizes (`2000` csID + `2000` csOOD) |
| `configs/tent.yaml` | TENT optimizer (Adam, `lr=1e-3`) |
| `configs/nova-tta.yaml` | NOVA-TTA: scorer, GMM (pooled, 2-component), warm-up, loss terms, `l1_weight` |
| `configs/sweep.yaml` | the grid `sweep.py` expands |

---

## Reproducibility

Seeds drive the adapt/diagnostic pool split, the adaptation stream order, and the GMM, so a stream is
fully reproducible from its `meta.json`. The Phase-1/Phase-2 split guarantees evaluation never touches
data seen during adaptation.
