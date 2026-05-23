# Open-Set TTA — Implementation Specification

Companion to `ostta_theory.md`. Implementation details here; mathematical definitions and research structure in the theory document.

---

## 1. Core architecture: two phases

The central design decouples **adaptation** (expensive, done once) from **analysis** (cheap, run many times).

**Phase 1 — Adaptation.**
Run a TTA method on a fixed stream of $T$ batches. After each batch, save the BN affine state. Output: $T+1$ lightweight checkpoints $\theta_0, \theta_1, \ldots, \theta_T$.

**Phase 2 — Analysis.**
Every experiment is a pure function of saved checkpoints and a fixed held-out diagnostic set $\mathcal{D}$. Load $\theta_t$, evaluate on $\mathcal{D}$, record. No re-adaptation ever needed.

```
Adaptation stream  B_1, B_2, ..., B_T       Diagnostic set D (fixed, held-out)
                          │                             │
              ┌───────────▼────────────┐                │
  Phase 1     │  one gradient step     │                │
  (once)      │  per batch →           │                │
              │  save θ_t after each   │                │
              └───────────┬────────────┘                │
                          │  checkpoints/{stream_id}/   │
              ┌───────────▼──────────────────────────┐  │
  Phase 2     │  for t in 0..T:                      │  │
  (many)      │    load θ_t → forward on D ──────────┼──┘
              │    → metric(t)                       │
              └──────────────────────────────────────┘
```

**One batch = one gradient step = one checkpoint.**
TENT processes $\mathcal{B}_t$ with a single forward pass, single backward pass, and single optimiser step on $(\gamma, \beta)$. Checkpoint $\theta_t$ is the BN affine state after that step. The words **batch** and **step** are interchangeable; both refer to index $t \in \{1,\ldots,T\}$.

**The two phases are orthogonal to the two paths** (Path A / Path B from theory §5). Phase 1 and Phase 2 describe the computation pipeline; Path A and Path B describe the research direction chosen after the core experiments.

---

## 2. Model and datasets

### Pretrained model

| Field | Value |
|---|---|
| Architecture | WideResNet-40-2 |
| Pretraining | AugMix + JSD loss on CIFAR-10 |
| Source | RobustBench |
| Model string | `Hendrycks2020AugMix` |
| Clean CIFAR-10 accuracy | ~95.83% |
| Mean corruption error (CIFAR-10-C) | ~11.2% |

Used in both UniEnt and ROSETTA. RobustBench downloads and caches weights automatically:

```python
from robustbench.utils import load_model
model = load_model(model_name='Hendrycks2020AugMix',
                   dataset='cifar10', threat_model='corruptions')
```

CIFAR-10-C data loads the same way:

```python
from robustbench.data import load_cifar10c
x, y = load_cifar10c(n_examples=10000, corruptions=['gaussian_noise'],
                     severity=5, data_dir='./data')
```

Both calls download on first use and cache locally. `backbone.py` and `cifar10c.py` are thin wrappers — no manual weight files, no architecture definition.

### csID dataset: CIFAR-10-C

15 corruption types × 5 severity levels applied to the CIFAR-10 test set (10,000 images). Loaded via RobustBench. **Default severity: 5.**

### csOOD datasets

Two csOOD sources. Each runs as a separate set of streams; the two are never mixed within a stream.

| Name | Source | Order |
|---|---|---|
| `svhn_c` | SVHN test set + same 15 corruptions | Run first |
| `homemade` | Research group images + same 15 corruptions | Run after SVHN-C results |

**`svhn_c`**: `torchvision.datasets.SVHN` + UniEnt corruption pipeline (`github.com/gaozhengqing/UniEnt`). SVHN digits are semantically unrelated to CIFAR-10.

**`homemade`**: Flat folder of images from the research group. Same 15 corruptions applied after resizing to 32×32. No labels needed.

```
data/homemade/raw/
    img_0001.png
    img_0002.png   ...
```

`src/data/homemade.py` interface (mirrors `svhnc.py`):
```python
def load_homemade_c(folder, corruption, severity, indices=None):
    """Load images, resize to 32×32, apply corruption.
    Returns (x: Tensor[N,3,32,32], y: Tensor[N]) — y is zeros, labels unused."""
```

### Code sources

| Component | Source |
|---|---|
| TENT | `github.com/DequanWang/tent` — adapt `tent.py` directly |
| UniEnt / UniEnt+ | `github.com/gaozhengqing/UniEnt` — Path A baseline; also provides SVHN-C corruption pipeline |
| ROSETTA | Paper only — repo unreleased; implement from Zhao et al. 2026 |
| Corruption pipeline | `github.com/gaozhengqing/UniEnt` — use for both SVHN-C and homemade |
| WideResNet-40-2 | RobustBench `Hendrycks2020AugMix` |
| CIFAR-10-C | RobustBench `load_cifar10c` |

### model.yaml

```yaml
# model.yaml  — fixed for this project, not swept
robustbench_name:    Hendrycks2020AugMix
robustbench_dataset: cifar10
robustbench_threat:  corruptions
csid_dataset:        cifar10c
```

---

## 3. Data design

All data for a stream comes from **one corruption type, one severity**. Before running, split into four disjoint pools:

> **Train/test separation guarantee.** The pretrained model was trained on the CIFAR-10 *training* set (50,000 images). All adaptation and diagnostic data come from the CIFAR-10 *test* set (10,000 different images) with corruptions applied. The model has never seen these images during training. SVHN was not part of training at all. This guarantee is enforced by the standard CIFAR-10 split and by using a verified RobustBench checkpoint.

```
Corrupted test set  (CIFAR-10 test split only — disjoint from training data)
    ├── csID adaptation pool    →  sampled into batches B_1 … B_T
    ├── csID diagnostic pool    →  D_csID  (fixed, never adapted on)
    ├── csOOD adaptation pool   →  sampled into batches B_1 … B_T
    └── csOOD diagnostic pool   →  D_csOOD (fixed, never adapted on)
```

Assert disjointness at construction time.

### Adaptation stream
Each batch $\mathcal{B}_t$ has size $N$ with OOD proportion $\alpha$:
- $(1-\alpha)N$ csID samples from the csID adaptation pool
- $\alpha N$ csOOD samples from the csOOD adaptation pool

Generated once with a fixed seed; frozen. Exact indices saved in `meta.json`.

### Diagnostic set $\mathcal{D}$
Fixed, class-balanced (csID side), sized for stable AUROC estimation. Never touched during adaptation.

### Corruption policy
**One corruption type per stream.** All $T$ batches use the same corruption and severity. Different corruptions → separate streams.

### OOD proportion sweep
| $\alpha$ | Role |
|---|---|
| 0.0 | Closed-set sanity check |
| 0.1 | OOD minority |
| 0.25 | Moderate OOD |
| 0.5 | Balanced — **primary** |
| 0.75 | OOD majority / stress test |

Run the full suite at $\alpha = 0.5$ first.

---

## 4. Repository structure

```
ostta/
├── configs/
│   ├── model.yaml           # architecture + weights (fixed)
│   ├── stream.yaml          # corruption, alpha, csood_source, N, T, seed
│   ├── tent.yaml            # optimiser, lr
│   ├── diagnostic.yaml      # diagnostic set size and seed
│   └── sweep.yaml           # alpha × corruption × csood_source × seed grid
├── src/
│   ├── models/
│   │   └── backbone.py      # wrapper around robustbench.utils.load_model
│   ├── data/
│   │   ├── pools.py         # disjoint adaptation / diagnostic split
│   │   ├── stream.py        # build and freeze the adaptation stream
│   │   ├── cifar10c.py      # wrapper around robustbench.data.load_cifar10c
│   │   ├── svhnc.py         # torchvision SVHN + UniEnt corruption pipeline
│   │   └── homemade.py      # flat folder loader → resize 32×32 → apply corruptions
│   ├── tta/
│   │   ├── tent.py          # adapted from github.com/DequanWang/tent
│   │   ├── bn_adapt.py      # BN statistics update only, no gradient step
│   │   ├── unient.py        # adapted from github.com/gaozhengqing/UniEnt  [Path A]
│   │   ├── rosetta.py       # implemented from paper (repo unreleased)     [Path A]
│   │   └── fix.py           # proposed fix — blank placeholder              [Path A]
│   ├── checkpoint/
│   │   ├── bn_state.py      # extract / inject BN affine state (γ, β)
│   │   └── runner.py        # Phase 1 driver
│   ├── experiments/
│   │   ├── exp1_auroc.py          # core — always run
│   │   ├── exp2_norm_align.py     # core — always run
│   │   ├── exp21_relative.py      # core — always run
│   │   ├── exp3_layerwise.py      # core — always run
│   │   └── exp4_vectorfield.py    # Path B only
│   ├── metrics/
│   │   ├── ood_scores.py    # energy, max-logit
│   │   ├── ood_metrics.py   # AUROC, FPR95, OSCR, H-score
│   │   └── geometry.py      # norms, cosines, centroid distances, radial/tangential
│   └── prototypes.py        # compute and cache μ_c^(0)
├── scripts/
│   ├── run_adaptation.py    # Phase 1: run one TTA method on one stream
│   ├── run_experiment.py    # Phase 2: run one experiment on one stream's checkpoints
│   └── run_sweep.py         # grid over sweep.yaml
├── checkpoints/
│   └── {method}/{corruption}_{csood_source}_{alpha:.2f}_seed{seed}/
│       ├── meta.json
│       ├── base_model.pt    # full θ_0 (saved once)
│       ├── theta_000.pt     # BN affine state at t=0
│       ├── theta_001.pt     # after batch 1
│       └── ...
└── results/
    └── {method}/{stream_id}/{experiment}/
```

The `{method}` directory level (e.g. `tent/`, `bn_adapt/`, `unient/`, `rosetta/`, `fix/`) allows Phase 2 experiments to compare methods by loading checkpoints from different directories.

---

## 5. Checkpoint format

TENT only modifies $(\gamma, \beta)$. The checkpoint is minimal:

```python
# theta_{t:03d}.pt
{
    "t":     int,
    "gamma": {layer_name: Tensor},   # γ_l^(t)
    "beta":  {layer_name: Tensor},   # β_l^(t)
}
```

`base_model.pt` holds the full $\theta_0$ (all weights). Saved once per stream.

### meta.json

```json
{
  "method":       "tent",
  "corruption":   "gaussian_noise",
  "severity":     5,
  "alpha":        0.50,
  "N":            200,
  "T":            100,
  "csood_source": "svhn_c",
  "seed":         0,
  "batches": [
    {"t": 1, "csid_indices": [...], "csood_indices": [...]},
    ...
  ]
}
```

---

## 6. Phase 1 — adaptation runner

`scripts/run_adaptation.py` (logic in `src/checkpoint/runner.py`). Takes `--method` as an argument: `tent`, `bn_adapt`, `unient`, `rosetta`, `fix`.

1. Load backbone; save `base_model.pt` and `theta_000.pt`.
2. Build the frozen adaptation stream from `stream.yaml`.
3. For each batch $\mathcal{B}_t$, $t = 1 \ldots T$:
   - Forward: BN in train mode, uses $\mathcal{B}_t$'s own statistics.
   - Compute method-specific loss.
   - Single backward; single optimiser step on the method's parameters.
   - Save `theta_{t:03d}.pt`.
4. Write `meta.json`.

**TENT** (`tent.py`): optimise $(\gamma, \beta)$ only; loss = mean entropy over batch.

**BN Adapt** (`bn_adapt.py`): no gradient step, no parameter update. Forward pass only — BN statistics are updated from $\mathcal{B}_t$ but $(\gamma, \beta)$ stay at $\theta_0$. Checkpoint still saved at every $t$ (always equal to $\theta_0$ in affine params, but the run records the stream metadata). Used as the no-gradient reference baseline.

**UniEnt / ROSETTA / fix** [Path A]: same loop, different loss. See §9.

---

## 7. Phase 2 — core experiments (always run)

Experiments 1, 2, 2.1, and 3 run regardless of which path is taken. They use the TENT checkpoints (and optionally the BN Adapt checkpoints for comparison).

Common evaluation primitive:

```python
def embed(theta_t_path, D):
    """Load (γ, β) from checkpoint into backbone.
    Run D in BN train mode — statistics recomputed from D at every t.
    Return (features, logits) for all x in D."""
```

### Experiment 1 — AUROC (`exp1_auroc.py`)
For each $t$: `embed(θ_t, D)` → energy score → `AUROC(t)`; argmax → `Acc_csID(t)`. Run for both TENT and BN Adapt checkpoints; plot both trajectories.

### Experiment 2 — Norm vs alignment (`exp2_norm_align.py`)
At $t=0$: `F_csID`, `F_csOOD`. For each $t$: `‖z‖_csID`, `‖z‖_csOOD`, `Δnorm`; `cos`/`maxcos` (both versions); `d_csID`, `d_csOOD`; `C_csOOD`; `Change_csOOD` (iterate consecutively, keep $t-1$ predictions in memory).

### Experiment 2.1 — Relative representation (`exp21_relative.py`)
Design open (theory §4). Reuses `embed` and cosine utilities from `geometry.py`.

### Experiment 3 — Layer-wise affine drift (`exp3_layerwise.py`)
Pure checkpoint analysis — no diagnostic forward pass needed. Load `(gamma, beta)` from each `theta_{t}.pt`. Compute $\|\Gamma_l^{(t)}\|_2$, $\|B_l^{(t)}\|_2$, $\mathrm{CV}_l^{(t)}$ per layer per $t$. Output heatmaps and layer profiles.

---

## 8. Configs

```yaml
# stream.yaml
corruption:   gaussian_noise   # one of the 15 CIFAR-C corruptions
severity:     5
alpha:        0.5              # OOD proportion per batch
N:            200              # batch size (= number of samples per gradient step)
T:            100              # number of batches = number of gradient steps
csood_source: svhn_c           # svhn_c | homemade
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
seed:    0
```

```yaml
# sweep.yaml
corruption:   [gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,
               motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
               elastic, pixelate, jpeg]
alpha:        [0.0, 0.1, 0.25, 0.5, 0.75]
csood_source: [svhn_c, homemade]
seed:         [0, 1, 2]
primary:      {corruption: gaussian_noise, alpha: 0.5, csood_source: svhn_c, seed: 0}
```

---

## 9. Decision point and two paths

After the core experiments (§7), inspect the results and take **one** path. This mirrors theory §5 exactly.

```
Core experiments done
        │
        ├── fix identified? YES → Path A
        │                              ├── implement fix (fix.py)
        │                              ├── run Phase 1 for: unient, rosetta, fix
        │                              ├── run Experiment 1 for all methods
        │                              └── compare on open-set metrics
        │
        └── fix identified? NO  → Path B
                                       └── run Experiment 4 (vector field)
```

### Path A — fix identified

**9.A.1 Proposed fix** (`src/tta/fix.py`)

> *[Blank — implement once experimental findings point to a direction. Theory §5.A.1 discusses likely sources.]*

Once `fix.py` is implemented, run Phase 1 with `--method fix` to produce checkpoints in the same format as TENT.

**9.A.2 Baseline implementations**

*UniEnt / UniEnt+* (`src/tta/unient.py`): adapt from `github.com/gaozhengqing/UniEnt`. Key components: energy-score GMM split; entropy minimisation on presumed csID; entropy maximisation on presumed csOOD.

*ROSETTA* (`src/tta/rosetta.py`): implement from Zhao et al. 2026 (repo unreleased). Three losses — $\mathcal{L}_\text{csID}$ (entropy min + diversity), $\mathcal{L}_\text{ang}$ (angular loss for csID), $\mathcal{L}_\text{norm}$ ($\ell_1$ norm suppression for csOOD). Prototype momentum $\alpha = 0.005$, linear classifier weights as source prototypes.

Run Phase 1 for each: `--method unient`, `--method rosetta`.

**9.A.3 Comparison**

All methods share the same checkpoint format and Phase 2 runners. Comparison = run Experiment 1 (and full open-set metrics) across the five checkpoint sets: Source, BN Adapt, TENT, UniEnt (+ UniEnt+), ROSETTA, Fix.

```python
# comparison script (scripts/run_comparison.py)
methods = ['tent', 'bn_adapt', 'unient', 'rosetta', 'fix']
for method in methods:
    run_experiment('exp1_auroc', stream_id=f'{method}/gaussian_noise_svhn_c_0.50_seed0')
```

Open-set metrics to report: Acc, AUROC, FPR95, OSCR, H-score. Primary question: does the fix maintain AUROC over the stream where baselines degrade?

---

### Path B — no fix identified

**9.B.1 Experiment 4 — Vector field (`exp4_vectorfield.py`)**

Requires consecutive TENT checkpoints $\theta_{t-1}$, $\theta_t$.

- **Empirical field**: $v^{(t)}(x) = \mathrm{embed}(\theta_t, x) - \mathrm{embed}(\theta_{t-1}, x)$ on $\mathcal{D}$.
- **JVP approximation**: $v_\text{approx}^{(t)}(x) = \mathrm{jvp}(g_{\theta_{t-1}},\ x,\ \Delta\gamma^{(t)},\ \Delta\beta^{(t)})$ where $\Delta\gamma = \gamma^{(t)} - \gamma^{(t-1)}$ from consecutive checkpoints. Both evaluations use $\mathcal{D}$'s own BN statistics — the transient $(\Delta\mu, \Delta\sigma)$ components cancel. Use `torch.func.jvp` (forward-mode AD; no full Jacobian materialised).
- **Relative error**: $\bar\varepsilon^{(t)} = \mathbb{E}_x[\|v^{(t)} - v_\text{approx}^{(t)}\| / \|v^{(t)}\|]$ per population.
- **Radial fraction**: $\rho^{(t)}(x) = \|v_r^{(t)}\| / \|v^{(t)}\|$ per population. $\rho \approx 1$ for OOD confirms norm inflation dominates.
- **Cumulative field**: $V^{(t)}(x) = \mathrm{embed}(\theta_t, x) - \mathrm{embed}(\theta_0, x)$ — primary visualisation.

---

## 10. Implementation notes

- **BN mode.** Train mode throughout — both during adaptation (uses $\mathcal{B}_t$ statistics) and during diagnostic evaluation (uses $\mathcal{D}$ statistics). Never call `eval()`.
- **Diagnostic consistency.** Pass all of $\mathcal{D}$ in one batch (or fixed sub-batches with the same seed) so BN statistics are identical across all $t$. Varying batch composition contaminates the metric trajectory.
- **Consecutive checkpoints** (`Change_csOOD`, Experiment 4): iterate $t$ in order; keep $t-1$ state in memory.
- **Frozen prototypes** $\mu_c^{(0)}$: compute once from $\theta_0$ on $\mathcal{D}_\text{csID}$; cache to disk.
- **Disjointness**: assert adaptation and diagnostic pools share no indices before Phase 1.
- **Checkpoint size**: BN affine tensors are a few KB; saving every step for $T=100$ is negligible.

---

## 11. Run sequence

```bash
# ── Phase 1: primary stream ──────────────────────────────────────────────────
python scripts/run_adaptation.py \
  --method tent --corruption gaussian_noise --alpha 0.5 --csood_source svhn_c --seed 0

python scripts/run_adaptation.py \
  --method bn_adapt --corruption gaussian_noise --alpha 0.5 --csood_source svhn_c --seed 0

# ── Phase 2: core experiments (always run) ───────────────────────────────────
for exp in exp1_auroc exp2_norm_align exp21_relative exp3_layerwise; do
  python scripts/run_experiment.py \
    --stream tent/gaussian_noise_svhn_c_0.50_seed0 --exp $exp
done

# ── Decision: inspect results, choose path ───────────────────────────────────

# PATH A — if a fix is found
python scripts/run_adaptation.py --method fix     ... 
python scripts/run_adaptation.py --method unient  ...
python scripts/run_adaptation.py --method rosetta ...
python scripts/run_comparison.py --stream_base gaussian_noise_svhn_c_0.50_seed0

# PATH B — if no fix
python scripts/run_experiment.py \
  --stream tent/gaussian_noise_svhn_c_0.50_seed0 --exp exp4_vectorfield

# ── Full sweep ───────────────────────────────────────────────────────────────
python scripts/run_sweep.py
```
