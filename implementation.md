# Open-Set TTA — Implementation Specification

Companion to `theory.md`. Code, data, and run details here; math and research structure in theory.

---

## 1. Two-phase design

**Phase 1 — Adaptation (run once).**
Run a TTA method on a stream of $T$ batches. Save BN affine state $(\gamma, \beta)$ after each batch.
Output: $T+1$ checkpoints $\theta_0 \ldots \theta_T$.

**Phase 2 — Analysis (run many times).**
Load any checkpoint, evaluate on fixed diagnostic set $\mathcal{D}$, compute metrics. No re-adaptation.

```
Stream  B_1 … B_T                    Diagnostic set D (fixed, held-out)
        │                                        │
  ┌─────▼──────────────┐                         │
  │  one step/batch    │                         │
  │  save θ_t          │                         │
  └─────┬──────────────┘                         │
        │ checkpoints/{method}/{stream_id}/      │
  ┌─────▼──────────────────────────────────────┐ │
  │  for t in 0..T:                            │ │
  │    bn_affine.evaluate(model, θ_t, D)       │─┘
  │    → metric(t)                             │
  └────────────────────────────────────────────┘
```

---

## 2. Model and datasets

**Model:** WideResNet-40-2, pretrained with AugMix+JSD on CIFAR-10. Loaded via RobustBench (`Hendrycks2020AugMix`). Clean acc ~95.83%, mean corruption error ~11.2% on CIFAR-10-C.

```python
from src.model import load_model
model = load_model()  # BN train mode, batch stats, no running stats
```

**csID:** CIFAR-10-C — 15 corruption types × 5 severity levels, 10 000 images each. Default severity: 5.

**csOOD:** two sources, run separately:
- `svhn_c` — SVHN test set + same 15 corruptions. Run first.
- `rome32` — research group images (classic Rome) resized to 32×32 + same corruptions. Stub until data available (`data/rome32/raw/`).

---

## 3. Stream and data split

A **stream** is fully defined by: `(method, corruption, severity, csood_source, open_set, N, T, seed)`.
One stream → one checkpoint directory.

```
checkpoints/{method}/{corruption}_{csood_source}_{open|closed}_seed{seed}/
```

**Data split per stream** (one corruption, one severity):
```
Full dataset
  ├── csID adapt pool    → sampled into B_1 … B_T
  ├── csID diag pool     → D_csID  (fixed, never adapted on)
  ├── csOOD adapt pool   → sampled into B_1 … B_T  (unused when open_set=False)
  └── csOOD diag pool    → D_csOOD (fixed, never adapted on)
```

Split is done by `DataPools` using **stream seed** — same seed in Phase 1 and Phase 2 guarantees identical diagnostic indices.

**open_set** controls OOD presence per batch:

| open_set | n_ood | n_id | Role |
|---|---|---|---|
| False | 0 | N | Closed-set sanity check |
| True | N//2 | N//2 | Balanced open-set (primary) |

**No repeated examples across batches.** `build_stream` pre-shuffles the adapt pool once and slices sequentially. With T=80, N=200, open_set=True: exactly 8000 csID and 8000 csOOD draws match the adapt pool size — zero repeats. If T·n > pool size (e.g. closed-set), pool reshuffles after exhaustion.

**Diagnostic set** (2000 csID + 2000 csOOD) is independent of open_set — always the same held-out indices.

---

## 4. Repository structure

```
├── configs/
│   ├── model.yaml       # RobustBench model spec (fixed)
│   ├── stream.yaml      # defaults for a single stream run
│   ├── tent.yaml        # optimizer, lr
│   ├── diagnostic.yaml  # diagnostic set sizes
│   └── sweep.yaml       # grid: corruptions × open_set × csood_sources × seeds
│
├── src/
│   ├── model.py         # load_model(), get_embeddings(), classifier_weights()
│   ├── bn_affine.py     # extract, inject, save, load, evaluate_diagnostic_stream  ← Phase 2 primitive
│   ├── centroids.py     # compute() — source centroids μ_c^(0) from original weights on clean CIFAR-10
│   ├── data/
│   │   ├── cifar10.py   # load_cifar10_data() — clean test set, [0,1] CHW, no normalization
│   │   ├── cifar10c.py  # load_cifar10c_data()
│   │   ├── svhnc.py     # load_svhn_c()
│   │   ├── rome32.py    # load_rome32_c()  [stub]
│   │   ├── pools.py     # DataPools — disjoint adapt/diagnostic split
│   │   └── stream.py    # build_stream() — frozen adaptation stream, sequential non-repeating
│   ├── tta/
│   │   └── tent.py      # official TENT (github.com/DequanWang/tent), unchanged
│   ├── metrics/
│   │   ├── ood_scores.py   # energy_score, max_logit_score, max_softmax_score
│   │   ├── ood_metrics.py  # auroc, fpr_at_tpr, oscr, h_score
│   │   └── geometry.py     # feature_norms, cosine_to_weights, centroid_distances, …
│   └── viz/
│       ├── common.py    # shared style: colors, rcParams
│       ├── exp1.py      # plot AUROC + Acc trajectory (overlay multiple streams)
│       ├── exp2.py      # plot norm/cosine/distance/confidence panels
│       └── exp3.py      # plot BN drift heatmaps and layer profile
│
├── scripts/
│   ├── phase1_adapt.py     # Phase 1: run method on stream, save checkpoints
│   ├── reproduce_tent.py   # closed-set TENT reproduction (matches published protocol)
│   ├── exp1_auroc.py       # compute AUROC + Acc → results.json
│   ├── exp2_geometry.py    # compute geometry metrics → results.json
│   ├── exp3_layerwise.py   # compute BN drift → results.json  (no forward pass)
│   ├── plot.py             # render figures from results JSON → figures/
│   └── sweep.py            # run phase1_adapt over full sweep.yaml grid
│
├── checkpoints/            # gitignored
│   └── {method}/{corruption}_{csood_source}_{open|closed}_seed{seed}/
│       ├── meta.json        # full stream spec + per-batch indices
│       ├── base_model.pt    # full θ_0 weights (saved once)
│       ├── theta_000.pt     # BN affine state at t=0
│       └── theta_001.pt …
├── results/                # gitignored — JSON metrics, one subdir per experiment
│   └── {method}/{stream_id}/{experiment}/results.json
└── figures/                # gitignored — PNG plots produced by scripts/plot.py
    └── {stream_id}/
```

---

## 5. Evaluation protocol

### 5.1 TENT reproduction (`scripts/reproduce_tent.py`)

Matches the original TENT protocol exactly:

- Single pass: 10 000 images per corruption → $T = 50$ batches of $N = 200$, each image used once.
- Weights at evaluation: **θ_{t−1}** (pre-update). `forward_and_adapt` runs forward with θ_{t−1}, computes entropy loss, updates to θ_t, returns the logits from the pre-update forward. Accuracy is measured from those logits — the batch is unseen at evaluation time.
- Eval data: **same batch** used for adaptation.
- Model reset between corruption types.

Run: `uv run python scripts/reproduce_tent.py`. Expected mean corruption error ≈ 11.2%.

### 5.2 Our experiments (Phase 2)

We depart from the reproduction protocol for two reasons: (i) we need a stable OOD detection signal alongside accuracy, and (ii) per-batch evaluation on the adaptation batch conflates BN stat variance with affine drift.

- Weights at evaluation: **θ_t** (post-update, saved after each step).
- Eval data: **fixed held-out diagnostic set $\mathcal{D}$** (2000 csID + 2000 csOOD), never touched during Phase 1.

---

#### Evaluation protocol — overview

```
Phase 1 produces T+1 checkpoints:

  stream:  B_1  B_2  ...  B_T          (adaptation, N=200 mixed, seen once)
            │    │         │
           θ_1  θ_2  ...  θ_T          (BN affine state saved after each step)
           θ_0                         (original weights, saved before step 1)


Phase 2 evaluates each θ_t on the SAME fixed diagnostic set D:

  D = [ D_csID  (2000 held-out csID) ]    ← never touched in Phase 1
      [ D_csOOD (2000 held-out csOOD)]    ← never touched in Phase 1

  For EVERY t in {0, 1, ..., T}:
    load θ_t  (freeze weights — NO adaptation)
    ↓
    evaluate_diagnostic_stream(θ_t, D_csID, D_csOOD)
    ↓
    metric(t)     ← AUROC, Acc, norms, cosines, ...

  Result: metric trajectory over t = [metric(0), metric(1), ..., metric(T)]
  Any change in metric(t) reflects ONLY affine drift θ_{t-1} → θ_t.
```

---

#### Inside `evaluate_diagnostic_stream` — one call for θ_t

D is partitioned into **K fixed batches of size N=200** (same as adaptation).  
Partition fixed by seed — identical for every θ_t.

```
  D_csID  = [ id_0 | id_1 | ... | id_1999 ]   (2000 examples, fixed order)
  D_csOOD = [ood_0 |ood_1 | ... |ood_1999]    (2000 examples, fixed order)

  Split into K=20 batches of 100 ID + 100 OOD each:

  batch_1:  [ id_0 .. id_99   | ood_0 .. ood_99  ]   ← 200 examples, MIXED
  batch_2:  [ id_100..id_199  | ood_100..ood_199 ]
  ...
  batch_20: [ id_1900..id_1999| ood_1900..ood_1999]

  For each batch k:
  ┌─────────────────────────────────────────────────────────────┐
  │  x_batch = cat([100 csID, 100 csOOD])                       │
  │                    │                                        │
  │             model(x_batch)   ← BN stats from mixed 200      │
  │                    │           (same regime as adaptation)  │
  │             feat[0:100]   → feat_id   (csID features)       │
  │             feat[100:200] → feat_ood  (csOOD features)      │
  └─────────────────────────────────────────────────────────────┘

  After K=20 batches:
    feat_id   = cat(feat_id_k   for k=1..20)  → shape [2000, d]
    logits_id = cat(logits_id_k for k=1..20)  → shape [2000, 10]
    feat_ood  = cat(feat_ood_k  for k=1..20)  → shape [2000, d]
    logits_ood= cat(logits_ood_k for k=1..20) → shape [2000, 10]

  Output ordering preserved: logits_id[i] ↔ D_csID[i],  logits_ood[i] ↔ D_csOOD[i]
```

**Metric aggregation:**
- AUROC: pool all 2000 csID scores + 2000 csOOD scores → one AUROC value
- Accuracy: `(logits_id.argmax(dim=-1) == y_id).mean()`
- Geometric metrics (norms, cosines, distances): mean over all 2000 per-sample values

---

#### Why this design

| ❌ Alternative | Problem |
|---|---|
| One large batch (4000 examples) | BN stats from 4000 >> N=200 used during adaptation. (γ,β) learned under small-batch normalization; evaluating under large-batch normalization changes the effective feature space. |
| Separate ID forward + OOD forward | Each population gets its own BN stats (OOD normalised by OOD-only mean/var). Never happens in deployment. Artificially stabilises OOD features, masking norm-gap collapse — exactly what we are trying to measure. |
| Evaluate on adaptation batch | Conflates BN stat variance with affine drift. Batch varies across t; metric changes could come from sampling noise, not θ drift. |

### 5.3 Path A method comparison

If a fix is identified, TENT, BN Adapt, UniEnt, ROSETTA, and the proposed fix are compared using the **same pre-update convention** as the TENT reproduction: evaluate each method on the batch used for adaptation, with the weights in effect before that batch's update. This aligns with published baselines and avoids favoring methods that happen to show post-update improvement on $\mathcal{D}$.

---

## 6. Key design decisions

**`load_model()` owns BN mode.**
On load: `model.train()`, `track_running_stats=False`, `running_mean/var=None` for all BN layers. Grad policy left to callers.

**`bn_affine.evaluate_diagnostic_stream` is the single Phase 2 primitive.**
`evaluate_diagnostic_stream(model, ckpt_path, x_id, x_ood, device)` = inject (γ,β) + K forward passes of N=200 mixed examples each. csID and csOOD concatenated per batch; outputs split back after. Returns `feat_id, logits_id, feat_ood, logits_ood`.

**`src/centroids.py` uses clean CIFAR-10.**
`compute(model, x_clean, y_clean, device)` — no checkpoint injection. Caller passes freshly loaded model (original weights). Clean CIFAR-10 loaded via `src/data/cifar10.load_cifar10_data()`. Cache stored as `centroids_clean.pt`.

**BN Adapt has no module.**
BN Adapt in Phase 1 = `with torch.no_grad(): model(x)`. One line in `phase1_adapt.py`.

**TENT `configure_model` does one thing `load_model` does not.**
Re-enables `requires_grad_(True)` on BN (γ,β) so Adam can update them. Everything else is already set by `load_model`.

**Pool seed = stream seed.**
`DataPools(seed=meta["seed"])` in both Phase 1 and Phase 2. Never use a separate diagnostic seed.

---

## 7. Checkpoint format

```python
# theta_{t:03d}.pt
{"t": int, "gamma": {layer_name: Tensor}, "beta": {layer_name: Tensor}}
```

`base_model.pt` — full state dict of $\theta_0$. `meta.json` — full stream spec + per-batch csID/csOOD indices (exact reproducibility).

---

## 8. Configs

```yaml
# stream.yaml — defaults for phase1_adapt.py; override via CLI
corruption:   gaussian_noise
severity:     5
open_set:     true         # true = balanced OOD (n_ood=N//2), false = closed-set
N:            200          # batch size
T:            80           # number of batches = steps (80 × 100 = 8000 = adapt pool size, zero repeats)
csood_source: svhn_c       # svhn_c | rome32
seed:         0
```

```yaml
# tent.yaml
optimizer: adam
lr:        1.0e-3
```

```yaml
# diagnostic.yaml
n_csid:  2000
n_csood: 2000
```

```yaml
# sweep.yaml — full grid for sweep.py
corruption:   [gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,
               motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
               elastic_transform, pixelate, jpeg_compression]
open_set:     [true, false]
csood_source: [svhn_c, rome32]
seed:         [0, 1, 2]
```

---

## 9. Run sequence

```bash
# ── TENT reproduction (closed-set, matches published protocol) ───────────────
uv run python scripts/reproduce_tent.py

# ── Single stream (use stream.yaml defaults or override via CLI) ─────────────
uv run python scripts/phase1_adapt.py --method tent
uv run python scripts/phase1_adapt.py --method bn_adapt

# ── Compute metrics (output: results/{stream}/expN/results.json) ─────────────
uv run python scripts/exp1_auroc.py \
  --streams tent/gaussian_noise_svhn_c_open_seed0 \
            bn_adapt/gaussian_noise_svhn_c_open_seed0

uv run python scripts/exp2_geometry.py --stream tent/gaussian_noise_svhn_c_open_seed0
uv run python scripts/exp3_layerwise.py --stream tent/gaussian_noise_svhn_c_open_seed0

# ── Render figures (output: figures/{stream}/expN_*.png) ─────────────────────
uv run python scripts/plot.py --exp 1 \
  --streams tent/gaussian_noise_svhn_c_open_seed0 \
            bn_adapt/gaussian_noise_svhn_c_open_seed0

uv run python scripts/plot.py --exp 2 --streams tent/gaussian_noise_svhn_c_open_seed0
uv run python scripts/plot.py --exp 3 --streams tent/gaussian_noise_svhn_c_open_seed0

# ── Inspect results → decide Path A or B ────────────────────────────────────

# ── Full sweep (after decision) ──────────────────────────────────────────────
uv run python scripts/sweep.py --method tent
```

---

## 10. Decision point

```
Experiments 1, 2, 3 done
        │
        ├── Path A (fix found)
        │     add src/tta/fix.py, src/tta/unient.py, src/tta/rosetta.py
        │     run phase1_adapt for each
        │     compare: exp1_auroc --streams tent bn_adapt unient rosetta fix
        │     metrics: Acc, AUROC, FPR95, OSCR, H-score
        │
        └── Path B (no fix)
              add scripts/exp4_vectorfield.py
              run on tent/gaussian_noise_svhn_c_0.50_seed0
```
