# Open-Set TTA — Theory and Method Specification

Companion to `ostta_implementation.md`. Mathematical definitions and research structure here; model, data, repository, and configuration details in the implementation document.

---

## 1. Introduction

### Open-Set Test-Time Adaptation

Test-time adaptation (TTA) adapts a pre-trained model to a shifted test stream online, without source data or labels. **TENT** is the canonical method: it minimises the entropy of the model's predictions by performing one gradient step per batch, updating only the batch-normalisation (BN) affine parameters $(\gamma, \beta)$. The stream is a sequence of $T$ batches $\mathcal{B}_1, \ldots, \mathcal{B}_T$, producing a chain of model states:

$$\theta_0 \;\xrightarrow{\mathcal{B}_1}\; \theta_1 \;\xrightarrow{\mathcal{B}_2}\; \theta_2 \;\;\cdots\;\; \xrightarrow{\mathcal{B}_T}\; \theta_T.$$

Each $\theta_t$ is the BN affine state after exactly one gradient step on batch $\mathcal{B}_t$. The words **batch** and **step** are used interchangeably; both refer to index $t$.

TENT assumes a **closed-set** stream: every test sample belongs to a known class, only shifted by covariate noise (covariate-shifted in-distribution, **csID**). The **open-set** setting (OSTTA) is more realistic: the stream also contains semantically novel samples (covariate-shifted out-of-distribution, **csOOD**) that the model should reject.

### The OOD detection problem in TTA

OSTTA requires two things at once: classify csID samples correctly, and assign low confidence to csOOD samples. The standard recipe (UniEnt, ROSETTA) is a two-step process: first split each batch into presumed csID and csOOD using an OOD score, then apply opposing objectives — entropy minimisation to presumed csID, entropy maximisation (or norm suppression) to presumed csOOD.

This recipe has a structural weakness:

1. **The OOD detector is imperfect.** Even at the optimal threshold, a non-negligible fraction of samples are misclassified, producing conflicting gradients.
2. **Entropy minimisation inflates feature norms uniformly.** Because BN affine parameters are channel-wise globals, entropy minimisation increases the feature norm of all samples rather than improving their angular alignment with class prototypes. Since OOD detection scores depend on the norm gap between csID and csOOD, this gap collapses over the stream and OOD detection degrades monotonically.
3. **Both effects accumulate.** The online, sequential nature of TENT compounds gradient errors across batches.

The goal of this project is to (i) reproduce TENT, (ii) characterise the failure mode through experiments, and (iii) either propose a principled fix or fully characterise the underlying geometry.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $f_\theta : \mathcal{X} \to \mathbb{R}^K$ | classifier with $K$ known classes |
| $g_\theta : \mathcal{X} \to \mathbb{R}^d$ | feature extractor (penultimate-layer embedding) |
| $\mathbf{W}^{(L)} \in \mathbb{R}^{d \times K},\ \mathbf{b} \in \mathbb{R}^K$ | linear classifier weights and bias (frozen during TTA) |
| $\psi(x) = \mathbf{W}^{(L)\top} g_\theta(x) + \mathbf{b}$ | logit vector |
| $p_\theta(x) = \mathrm{softmax}(\psi(x))$ | predicted distribution |
| $H(p) = -\sum_k p_k \log p_k$ | Shannon entropy |
| $\hat{y}(x) = \arg\max_k \psi_k(x)$ | predicted class |
| $t \in \{1,\ldots,T\}$ | batch / step index |
| $\mathcal{B}_t$ | test batch at step $t$, size $N$ |
| $\alpha$ | OOD proportion of each batch (fraction csOOD), unknown to the model |
| $\theta_t$ | BN affine state $(\gamma^{(t)}, \beta^{(t)})$ after the gradient step on $\mathcal{B}_t$ |
| $\mu_c^{(0)} = \mathbb{E}[g_{\theta_0}(x) \mid y=c]$ | source class centroid, frozen at $t=0$ |
| $\mathcal{D} = \mathcal{D}_\text{csID} \cup \mathcal{D}_\text{csOOD}$ | held-out diagnostic set, never adapted on |

### BN as a formal parameter vector

Each BN layer $l \in \{1,\ldots,L\}$ computes $\mathrm{BN}(x) = \gamma_l \hat{x}_l + \beta_l$ where $\hat{x}_l = (x - \mu_l)/\sigma_l$. We collect the full BN state into

$$\bm{\Phi} = (\bm{\gamma}, \bm{\beta}, \bm{\mu}, \bm{\sigma}).$$

TENT keeps BN in train mode: $(\mu_l, \sigma_l)$ are recomputed from each batch and never persist. Only $(\bm{\gamma}, \bm{\beta})$ accumulate across steps (**persistent**); $(\bm{\mu}, \bm{\sigma})$ reset each batch (**transient**). The checkpoint $\theta_t$ therefore stores only $(\bm{\gamma}^{(t)}, \bm{\beta}^{(t)})$.

### The logit decomposition

$$\psi_{c^*}(x) = \underbrace{\|\mathbf{w}_{c^*}\| \cdot \|g_\theta(x)\| \cdot \cos\theta_{c^*}}_{\text{geometric term}} + \underbrace{b_{c^*}}_{\text{frozen}}, \qquad c^* = \arg\max_k \psi_k(x).$$

Since $\mathbf{W}^{(L)}$ and $\mathbf{b}$ are frozen, TENT can only increase $\psi_{c^*}$ through the geometric term — by raising the **norm** $\|g_\theta(x)\|$ or improving **angular alignment** $\cos\theta_{c^*}$.

### Key observations

- **Norm inflation.** BN affine parameters cannot selectively rotate individual feature vectors. Entropy minimisation raises $\|g_\theta(x)\|$ uniformly for csID and csOOD alike, rather than improving $\cos\theta_{c^*}$.
- **Bias and the $\cos < 0$ case.** Because $b_{c^*}$ can be positive, a sample can have $\psi_{c^*} > 0$ even when $\cos\theta_{c^*} < 0$ — classified by the bias, not by geometric proximity. For these samples, norm inflation *decreases* the geometric term. The fraction of samples in this regime must be checked empirically.
- **Norm gap.** A well-trained model produces larger feature norms for csID than csOOD at $t=0$. By Hölder duality, $|\psi_k| \le \|g_\theta(x)\|_1 \cdot \|\mathbf{w}_k\|_\infty$, so the $\ell_1$ feature norm directly bounds the maximum logit. Norm inflation collapses the ID/OOD norm gap, degrading energy-based OOD scores.

---

## 3. Reproducing TENT

The first deliverable is a faithful re-implementation of TENT in the closed-set setting, validated against published corruption-robustness numbers.

Requirements:
- One gradient step per batch on $(\gamma, \beta)$ only; all other parameters frozen.
- BN in train mode throughout — uses each batch's own statistics for normalisation.
- No replay; strictly online.

Successful replication: closed-set corruption error matches published TENT numbers within a small tolerance. Concrete model, dataset, optimiser, and hyperparameter choices are in the implementation document (§2, §8).

---

## 4. Core Experiments

**Checkpoint-based evaluation.** All experiments are pure functions of saved checkpoints and the held-out diagnostic set $\mathcal{D}$. Checkpoint $\theta_t$ stores only $(\bm{\gamma}^{(t)}, \bm{\beta}^{(t)})$. Evaluating $\theta_t$ on $\mathcal{D}$ runs BN in train mode with $\mathcal{D}$ as its own batch — statistics are recomputed from $\mathcal{D}$ identically at every $t$, so the metric trajectory $\mathrm{metric}(t) = f(\theta_t, \mathcal{D})$ reflects changes in $(\bm{\gamma}^{(t)}, \bm{\beta}^{(t)})$ only. See implementation §6–7 for the checkpoint format and Phase 1/2 pipeline.

**Reference baselines.** Two baselines run alongside TENT and share the same checkpoint format and Phase 2 experiments:
- **Source** ($\theta_0$, $t=0$): the pre-trained model without any adaptation.
- **BN Adapt**: updates batch statistics $(\bm{\mu}, \bm{\sigma})$ from the test batch but performs no gradient step — no persistent changes. Isolates the transient statistics component. Expected not to show monotone AUROC collapse.

---

### Experiment 1 — AUROC degrades monotonically

**Goal.** Show OOD detection degrades over the stream while ID accuracy holds, demonstrating TENT optimises a metric orthogonal to OOD separability.

**Metrics.**
- $\mathrm{AUROC}(t)$ — AUROC of the energy score on $\mathcal{D}_\text{csID}$ vs $\mathcal{D}_\text{csOOD}$.
- $\mathrm{Acc}_\text{csID}(t)$ — top-1 accuracy on $\mathcal{D}_\text{csID}$.

**Expected.** $\mathrm{AUROC}(t)$ falls monotonically under TENT; $\mathrm{Acc}_\text{csID}(t)$ stays stable or improves. BN Adapt should not degrade monotonically, confirming the persistent gradient component is the cause.

---

### Experiment 2 — Norm vs alignment

**Goal.** Establish that norm inflation is the dominant optimisation channel, and that OOD samples are not geometrically absorbed into ID clusters.

**Preliminary diagnostic (at $t=0$ only).**
$$F_\text{csID},\ F_\text{csOOD} = \text{fraction of samples with } \cos\theta_{c^*}^{(0)}(x) > 0.$$
If $F_\text{csOOD} \ll 1$, many OOD samples are classified by the bias rather than geometry; the angular metrics below must be interpreted accordingly.

**Metrics over time.**

*Norm.* Mean feature norm per population — $\|z\|_\text{csID}^{(t)}$, $\|z\|_\text{csOOD}^{(t)}$ — and the gap $\Delta\mathrm{norm}(t) = \|z\|_\text{csID}^{(t)} - \|z\|_\text{csOOD}^{(t)}$.

*Alignment — two versions read together.*
- $\cos_\text{csID}^{(t)}$, $\cos_\text{csOOD}^{(t)}$: cosine to the **source-predicted class** $c^{(0)} = \arg\max_k \psi_k^{(0)}(x)$, fixed at $t=0$.
- $\mathrm{maxcos}_\text{csID}^{(t)}$, $\mathrm{maxcos}_\text{csOOD}^{(t)}$: **maximum** cosine over all class weight vectors. This is the angular component of the dominant logit; under pure norm inflation it stays flat.

*Distance.* $d_\text{csID}^{(t)}$, $d_\text{csOOD}^{(t)}$ — mean distance to the nearest frozen source centroid $\mu_c^{(0)}$.

*Prediction.* $C_\text{csOOD}^{(t)}$ — mean max confidence of OOD samples. $\mathrm{Change}_\text{csOOD}^{(t)}$ — fraction of OOD samples whose predicted class changed since the previous step.

**Expected.** Norms rise; $\mathrm{maxcos}$ stays approximately flat; distances do not decrease; OOD confidence rises while the predicted class is stable. This joint reading is the signature of norm-dominant adaptation.

---

### Experiment 2.1 — Angle preservation under TENT via relative representations

**Goal.** Investigate, using the relative-representation framework (RRZS, ICLR 2023), whether the transformation TENT induces on the embedding space is approximately angle-preserving, and whether this differs between csID and csOOD.

A representation that records the cosine similarity of each sample to a fixed anchor set is invariant to angle-preserving transformations (rotations, reflections, rescalings). Since norm inflation is a rescaling, this relative representation should be stable under TENT if and only if TENT acts as an angle-preserving map.

This experiment is left **open in design**: anchor choice, similarity measure, invariance metric, and how to contrast csID vs csOOD trajectories are to be determined after seeing Experiments 1 and 2.

---

### Experiment 3 — Layer-wise BN parameter drift

**Goal.** Determine which BN layers drift most and whether the update is uniform across channels (pure scaling) or structured.

**Notation.** Cumulative drift at layer $l$: $\Gamma_l^{(t)} = \gamma_l^{(t)} - \gamma_l^{(0)}$, $B_l^{(t)} = \beta_l^{(t)} - \beta_l^{(0)}$.

**Metrics.**
- $\|\Gamma_l^{(t)}\|_2$, $\|B_l^{(t)}\|_2$ — drift magnitude per layer and step.
- $\mathrm{CV}_l^{(t)} = \mathrm{std}_k(\Gamma_{l,k}^{(t)}) / |\mathrm{mean}_k(\Gamma_{l,k}^{(t)})|$ — channel-wise coefficient of variation. Low CV indicates pure scaling — the layer-level confirmation of norm inflation as the dominant mechanism.

Output: heatmaps over $(l, t)$ and layer profiles at $t = T$.

---

## 5. Two Paths

After Experiments 1, 2, 2.1, and 3, the project takes **one** of two paths. The decision and code structure for both are in implementation §9.

### Path A — A fix is identified

If the experiments point to a concrete intervention, we pursue it.

**5.A.1 Proposed fix.**

> *[Blank — to be filled in based on experimental findings. Experiment 2.1 and Experiment 3's layer-wise contamination profile are the most likely sources of a fix direction.]*

**5.A.2 Baselines.** Implement and evaluate the two state-of-the-art OSTTA methods:
- **UniEnt / UniEnt+** (`github.com/gaozhengqing/UniEnt`): energy-based GMM split; entropy minimisation on csID, entropy maximisation on csOOD.
- **ROSETTA** (from paper, repo unreleased): angular loss for csID; $\ell_1$ feature-norm suppression for csOOD.

Both baselines use the same Phase 1 checkpoint format and Phase 2 experiment runners as TENT. The comparison is therefore a direct function of the saved checkpoints across all methods.

**5.A.3 Comparison.** Evaluate the proposed fix against Source, BN Adapt, TENT, UniEnt/UniEnt+, and ROSETTA on (Acc, AUROC, FPR95, OSCR, H-score), with attention to whether AUROC is maintained over the stream.

### Path B — No fix is identified

If the experiments do not suggest a clear fix, fully characterise the geometry of the failure through the vector-field analysis.

**5.B.1 Experiment 4 — Vector field: approximation vs experimental observation.**

**Goal.** Characterise the displacement field TENT induces on the embedding space and verify whether it is captured by the first-order Jacobian approximation.

The displacement of sample $x$ after one batch is exactly
$$v^{(t)}(x) = g_{\theta_t}(x) - g_{\theta_{t-1}}(x),$$
and to first order
$$v^{(t)}_\text{approx}(x) = J_{g_{\theta_{t-1}}}(x) \cdot (\Delta\bm{\gamma}^{(t)}, \Delta\bm{\beta}^{(t)}).$$
Because both evaluations use $\mathcal{D}$'s own BN statistics, the transient $(\Delta\bm{\mu}, \Delta\bm{\sigma})$ components cancel; only $(\Delta\bm{\gamma}, \Delta\bm{\beta})$ — readable directly from consecutive checkpoints — enter the JVP.

**Metrics.**
- $\bar{\varepsilon}^{(t)} = \mathbb{E}_x[\|v^{(t)} - v^{(t)}_\text{approx}\| / \|v^{(t)}\|]$ — relative approximation error per population. Small error means the linear regime holds.
- $\rho^{(t)}(x) = \|v_r^{(t)}\| / \|v^{(t)}\|$ — radial fraction. $\rho \approx 1$ for OOD confirms norm inflation dominates over angular change.
- $V^{(t)}(x) = g_{\theta_t}(x) - g_{\theta_0}(x) = \sum_{s=1}^t v^{(s)}(x)$ — cumulative field, the primary visualisation object.

If $\bar{\varepsilon}^{(t)}$ grows over $t$, the model has drifted into a nonlinear regime — itself a measure of failure severity.

---

## Summary of the decision flow

```
Reproduce TENT  →  Experiments 1, 2, 2.1, 3
                                │
               ┌────────────────┴────────────────┐
       fix identified?                    no fix
               │                                 │
           PATH A                            PATH B
  implement fix + UniEnt + ROSETTA      Experiment 4
  compare on open-set metrics           vector field analysis
```
