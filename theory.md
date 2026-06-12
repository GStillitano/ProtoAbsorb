# Open-Set TTA — Theory and Method Specification

Companion to `implementation.md`. Math and research structure here; code, data, and run details in implementation.

---

## 1. Problem

TENT adapts a model online by minimising entropy per batch, updating only BN affine params $(\gamma, \beta)$. It assumes a **closed-set** stream (csID only). In the **open-set** setting (OSTTA), the stream also contains semantically novel samples (**csOOD**) that should be rejected.

The standard fix (UniEnt, ROSETTA): split each batch into presumed csID/csOOD via an OOD score, then apply opposing objectives — entropy min on csID, entropy max or norm suppression on csOOD.

**This project's thesis:** the split-and-oppose recipe has a deeper failure mode. Even without split errors, entropy minimisation inflates feature norms uniformly for both csID and csOOD, collapsing the ID/OOD norm gap that energy-based scores rely on. OOD detection degrades monotonically over the stream regardless of split quality.

**Goal:** (i) reproduce TENT, (ii) characterise the failure mode experimentally (Experiments 1–2), (iii) fix it with **Cassano** — a soft-labeled, norm-suppressing open-set TTA method.

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

## 5. The fix — Cassano

Experiments 1–2 establish the failure: entropy minimisation inflates feature norms for csID and csOOD alike, collapsing the energy-score gap. **Cassano** breaks the uniform inflation by giving csOOD samples the opposite objective — norm suppression — weighted by a soft ID/OOD posterior, so no hard split is required.

### 5.1 Two models, one learner

A **frozen scorer** ($\theta_0$, BN batch stats, no grad) scores each sample; the **adapted model** (BN affine $(\gamma,\beta)$ trainable, TENT-style) is the deployed network. Because the scorer never changes, its score distribution is ~stationary across $t$, so scores can be pooled across steps without bias.

### 5.2 Score and soft label

Per sample, maxcos score against the frozen head $\mathbf{W}^{(L)}$:

$$s(x) = \max_k \frac{g_\theta(x)\cdot \mathbf{w}_k}{\|g_\theta(x)\|\,\|\mathbf{w}_k\|}$$

Pool all scores seen so far, fit a 2-Gaussian GMM (ID = higher-mean component). The Bayes posterior gives a soft OOD weight, detached (a weight, not a target):

$$P_\text{OOD}(x) = \frac{\pi_\text{ood}\,\mathcal{N}(s\mid\mu_\text{ood},\sigma_\text{ood})}{\sum_c \pi_c\,\mathcal{N}(s\mid\mu_c,\sigma_c)}, \qquad P_\text{ID} = 1 - P_\text{OOD}$$

### 5.3 Loss

$$L = \frac{1}{N}\sum_i \Big[\, P_\text{ID}(x_i)\,H(p_i) \;+\; P_\text{OOD}(x_i)\,\lambda\,\|g_\theta(x_i)\|_1 \,\Big]$$

- $H(p)$ — softmax entropy: csID-soft samples pushed confident (TENT-style).
- $\|g_\theta(x)\|_1$ — feature $\ell_1$ norm: csOOD-soft samples pushed to *suppress* norm inflation, directly countering the Experiment-1 mechanism.
- $\lambda$ balances the terms ($H \le \ln 10 \approx 2.30$, $\|g\|_1 \approx 28$–$30$, so $\lambda \approx 0.03$).

An LR warmup $\mathrm{lr}_t = \mathrm{lr}\cdot r(t/K)$ shrinks early steps while the GMM pool is small and noisy.

**Evaluation.** Cassano produces the same checkpoint format as TENT and is evaluated under the identical Phase 2 protocol — Experiments 1 and 2 run on Cassano streams unchanged. Expected signature: AUROC held (or improved) over the stream while csID accuracy tracks TENT.

Full method and config: `cassano.md`, `configs/cassano.yaml`, `src/tta/cassano.py`.
