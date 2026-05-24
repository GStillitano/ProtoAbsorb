# Open-Set TTA — Theory and Method Specification

Companion to `implementation.md`. Math and research structure here; code, data, and run details in implementation.

---

## 1. Problem

TENT adapts a model online by minimising entropy per batch, updating only BN affine params $(\gamma, \beta)$. It assumes a **closed-set** stream (csID only). In the **open-set** setting (OSTTA), the stream also contains semantically novel samples (**csOOD**) that should be rejected.

The standard fix (UniEnt, ROSETTA): split each batch into presumed csID/csOOD via an OOD score, then apply opposing objectives — entropy min on csID, entropy max or norm suppression on csOOD.

**This project's thesis:** the split-and-oppose recipe has a deeper failure mode. Even without split errors, entropy minimisation inflates feature norms uniformly for both csID and csOOD, collapsing the ID/OOD norm gap that energy-based scores rely on. OOD detection degrades monotonically over the stream regardless of split quality.

**Goal:** (i) reproduce TENT, (ii) characterise the failure mode experimentally, (iii) either fix it (Path A) or fully characterise the geometry (Path B).

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $f_\theta : \mathcal{X} \to \mathbb{R}^K$ | classifier, $K$ known classes |
| $g_\theta : \mathcal{X} \to \mathbb{R}^d$ | penultimate-layer embedding |
| $\mathbf{W}^{(L)} \in \mathbb{R}^{K \times d},\ \mathbf{b} \in \mathbb{R}^K$ | linear head — **frozen during TTA** |
| $\psi(x) = \mathbf{W}^{(L)} g_\theta(x) + \mathbf{b}$ | logit vector |
| $p_\theta(x) = \mathrm{softmax}(\psi(x))$ | predicted distribution |
| $H(p) = -\sum_k p_k \log p_k$ | Shannon entropy |
| $t \in \{1,\ldots,T\}$ | batch / step index |
| $\mathcal{B}_t$ | adaptation batch at step $t$, size $N$, OOD fraction $\alpha$ |
| $\theta_t = (\bm{\gamma}^{(t)}, \bm{\beta}^{(t)})$ | BN affine state after step $t$ |
| $\mu_c^{(0)} = \mathbb{E}[g_{\theta_0}(x) \mid y=c]$ | source class centroid, frozen at $t=0$ |
| $\mathcal{D} = \mathcal{D}_\text{csID} \cup \mathcal{D}_\text{csOOD}$ | held-out diagnostic set, never adapted on |

### BN state decomposition

BN layer $l$: $\mathrm{BN}(x) = \gamma_l \hat{x}_l + \beta_l$, $\hat{x}_l = (x-\mu_l)/\sigma_l$.

- $(\bm{\gamma}, \bm{\beta})$ — **persistent**: accumulate across steps, stored in checkpoints
- $(\bm{\mu}, \bm{\sigma})$ — **transient**: recomputed from each batch, never persist

TENT runs BN in train mode throughout: batch statistics, never running statistics.

### Logit decomposition

$$\psi_{c^*}(x) = \underbrace{\|\mathbf{w}_{c^*}\| \cdot \|g_\theta(x)\| \cdot \cos\theta_{c^*}}_{\text{norm} \times \text{alignment}} + b_{c^*}, \qquad c^* = \arg\max_k \psi_k(x)$$

With $\mathbf{W}^{(L)}, \mathbf{b}$ frozen, TENT can only raise $\psi_{c^*}$ by increasing **norm** $\|g_\theta(x)\|$ or improving **alignment** $\cos\theta_{c^*}$. The thesis is that norm dominates.

### Key observations

- **Norm inflation.** BN affine params are channel-wise globals — they scale all features, not rotate individual vectors. Entropy min raises $\|g_\theta(x)\|$ for csID and csOOD uniformly.
- **Norm gap collapse.** At $t=0$ a well-trained model has larger norms for csID than csOOD. Energy score $E(x) = -\log\sum_k e^{\psi_k}$ depends on this gap. Uniform norm inflation collapses it → AUROC degrades.
- **Bias regime.** If $b_{c^*} > 0$, a sample can have $\psi_{c^*} > 0$ with $\cos\theta_{c^*} < 0$ — classified by bias not geometry. For these samples norm inflation *hurts* the geometric term. Fraction $F_\text{csOOD}$ of OOD samples in this regime is measured at $t=0$ (Experiment 2).

---

## 3. TENT reproduction

One gradient step per batch on $(\gamma, \beta)$ only; all other params frozen. BN train mode throughout. No replay.

Validation target: closed-set corruption error matches published TENT numbers (WideResNet-40-2, CIFAR-10-C severity 5, Adam lr=1e-3). See implementation §2, §7.

**BN Adapt baseline**: same setup but no gradient step — only batch statistics $(\mu, \sigma)$ update transiently. Isolates the transient-statistics component from the persistent-affine component. Expected not to show monotone AUROC collapse.

---

## 4. Core experiments

**Evaluation protocol.** All experiments are pure functions of checkpoints and $\mathcal{D}$. Evaluating $\theta_t$ on $\mathcal{D}$: inject $(\bm{\gamma}^{(t)}, \bm{\beta}^{(t)})$ into model, run $\mathcal{D}$ as one batch in BN train mode (statistics recomputed from $\mathcal{D}$ identically at every $t$). Metric trajectory reflects only affine drift. Code: `bn_affine.evaluate(model, ckpt_t, D)`.

---

### Experiment 1 — AUROC degrades monotonically (`exp1_auroc.py`)

**Goal.** Show OOD detection degrades while ID accuracy holds — TENT optimises orthogonal to OOD separability.

- $\mathrm{AUROC}(t)$ — energy score on $\mathcal{D}_\text{csID}$ vs $\mathcal{D}_\text{csOOD}$
- $\mathrm{Acc}_\text{csID}(t)$ — top-1 accuracy on $\mathcal{D}_\text{csID}$

Run for both TENT and BN Adapt. Expected: TENT AUROC falls monotonically; BN Adapt does not.

---

### Experiment 2 — Norm vs alignment (`exp2_geometry.py`)

**Goal.** Confirm norm inflation dominates over angular realignment; OOD samples not absorbed into ID clusters.

**t=0 diagnostic.**
$F_\text{csID}, F_\text{csOOD}$ = fraction of samples with $\cos\theta_{c^*}^{(0)} > 0$. Conditions interpretation of angular metrics.

**Metrics over $t$.**

| Metric | Definition |
|---|---|
| $\|z\|_\text{csID}^{(t)},\ \|z\|_\text{csOOD}^{(t)}$ | mean feature norm per population |
| $\Delta\mathrm{norm}(t)$ | $\|z\|_\text{csID} - \|z\|_\text{csOOD}$ |
| $\cos_\text{csID/OOD}^{(t)}$ | cosine to source-predicted class $c^{(0)}$ (fixed at $t=0$) |
| $\mathrm{maxcos}_\text{csID/OOD}^{(t)}$ | max cosine over all class weight vectors — flat under pure norm inflation |
| $d_\text{csID/OOD}^{(t)}$ | mean distance to nearest frozen centroid $\mu_c^{(0)}$ |
| $C_\text{csOOD}^{(t)}$ | mean max-softmax confidence of OOD samples |
| $\mathrm{Change}_\text{csOOD}^{(t)}$ | fraction of OOD samples whose predicted class changed vs step $t-1$ |

Expected signature of norm-dominant adaptation: norms rise, maxcos flat, distances stable, OOD confidence rises with stable predicted class.

---

### Experiment 2.1 — Relative representations (`exp21_relative.py`)

**Design open** — to be specified after Experiments 1 and 2.

Using the RRZS framework (ICLR 2023): represent each sample as its cosine similarities to a fixed anchor set. This representation is invariant to rescalings (norm changes). If TENT acts as a near-rescaling, relative representations should be stable — measure drift in relative space for csID vs csOOD separately.

---

### Experiment 3 — Layer-wise BN drift (`exp3_layerwise.py`)

**Goal.** Which layers drift most; is drift uniform across channels (pure scaling) or structured.

Pure checkpoint analysis — no forward pass needed.

- $\|\Gamma_l^{(t)}\|_2,\ \|B_l^{(t)}\|_2$ where $\Gamma_l^{(t)} = \gamma_l^{(t)} - \gamma_l^{(0)}$
- $\mathrm{CV}_l^{(t)} = \mathrm{std}_k(\Gamma_{l,k}^{(t)}) / |\mathrm{mean}_k(\Gamma_{l,k}^{(t)})|$ — low CV = pure scaling

Output: heatmaps over $(l, t)$ and layer profile at $t=T$.

---

## 5. Two paths

```
Experiments 1, 2, 2.1, 3
        │
        ├── fix identified? → PATH A
        │     implement fix + UniEnt + ROSETTA
        │     compare on Acc, AUROC, FPR95, OSCR, H-score over stream
        │
        └── no fix → PATH B
              Experiment 4: vector field analysis
```

### Path A — fix identified

**5.A.1 Proposed fix.** *[To be filled after experiments. Exp 2.1 and Exp 3 layer profiles are the likely sources.]*

**5.A.2 Baselines.**
- **UniEnt / UniEnt+** (`github.com/gaozhengqing/UniEnt`): energy GMM split; entropy min csID, entropy max csOOD.
- **ROSETTA** (Zhao et al. 2026, repo unreleased): angular loss csID; $\ell_1$ norm suppression csOOD. Prototype momentum $\alpha=0.005$.

### Path B — no fix

**Experiment 4 — Vector field** (`exp4_vectorfield.py`, not yet implemented).

Per-step displacement: $v^{(t)}(x) = g_{\theta_t}(x) - g_{\theta_{t-1}}(x)$.
First-order approximation via JVP (torch.func.jvp): $v_\text{approx}^{(t)} = J_{g_{\theta_{t-1}}} \cdot (\Delta\bm{\gamma}^{(t)}, \Delta\bm{\beta}^{(t)})$.
Transient $(\Delta\mu, \Delta\sigma)$ cancel when both evaluations use $\mathcal{D}$'s own batch stats.

Metrics: relative error $\bar\varepsilon^{(t)}$, radial fraction $\rho^{(t)} = \|v_r\|/\|v\|$, cumulative field $V^{(t)} = g_{\theta_t} - g_{\theta_0}$.
