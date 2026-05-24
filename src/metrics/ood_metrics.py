"""OOD detection metrics: AUROC, FPR95, OSCR, H-score."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(scores_id: torch.Tensor, scores_ood: torch.Tensor) -> float:
    """AUROC where higher score = more ID-like (label 1 for ID, 0 for OOD)."""
    labels = torch.cat([torch.ones(len(scores_id)), torch.zeros(len(scores_ood))])
    scores = torch.cat([scores_id, scores_ood])
    return float(roc_auc_score(labels.numpy(), scores.numpy()))


def fpr_at_tpr(scores_id: torch.Tensor, scores_ood: torch.Tensor, tpr: float = 0.95) -> float:
    """FPR when TPR = tpr (default 95%)."""
    labels = torch.cat([torch.ones(len(scores_id)), torch.zeros(len(scores_ood))])
    scores = torch.cat([scores_id, scores_ood])
    fpr_arr, tpr_arr, _ = roc_curve(labels.numpy(), scores.numpy())
    idx = np.searchsorted(tpr_arr, tpr)
    idx = min(idx, len(fpr_arr) - 1)
    return float(fpr_arr[idx])


def oscr(
    logits_id: torch.Tensor,
    labels_id: torch.Tensor,
    scores_id: torch.Tensor,
    scores_ood: torch.Tensor,
) -> float:
    """Open-Set Classification Rate (OSCR).

    Sweep threshold on OOD score. For each threshold: CCR = fraction of ID
    samples correctly classified AND score above threshold; FPR = fraction of
    OOD samples with score above threshold. OSCR = area under CCR–FPR curve.
    """
    preds_id = logits_id.argmax(dim=-1)
    correct = (preds_id == labels_id)

    all_scores = torch.cat([scores_id, scores_ood]).numpy()
    thresholds = np.unique(all_scores)

    ccr_list, fpr_list = [], []
    for thr in thresholds:
        ccr = float(((scores_id >= thr) & correct).float().mean())
        fpr = float((scores_ood >= thr).float().mean())
        ccr_list.append(ccr)
        fpr_list.append(fpr)

    # sort by fpr for trapezoidal integration
    order = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[order]
    ccr_sorted = np.array(ccr_list)[order]
    return float(np.trapz(ccr_sorted, fpr_sorted))


def h_score(acc_id: float, auroc_val: float) -> float:
    """Harmonic mean of closed-set accuracy and AUROC."""
    if acc_id + auroc_val == 0:
        return 0.0
    return 2 * acc_id * auroc_val / (acc_id + auroc_val)
