"""Geometric metrics: norms, cosines, centroid distances, radial/tangential split."""
import torch
import torch.nn.functional as F


def feature_norms(features: torch.Tensor) -> torch.Tensor:
    """L2 norm per sample. Shape: [N]."""
    return features.norm(dim=-1)


def feature_norms_l1(features: torch.Tensor) -> torch.Tensor:
    """L1 norm per sample. Shape: [N]."""
    return features.abs().sum(dim=-1)


def cosine_to_weights(features: torch.Tensor, weight_matrix: torch.Tensor, class_indices: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of each feature to its designated class weight vector.

    features:      [N, d]
    weight_matrix: [K, d]  (classifier weights, one row per class)
    class_indices: [N]     (which class weight to use per sample)
    Returns: [N] cosine similarities.
    """
    w = weight_matrix[class_indices]          # [N, d]
    return F.cosine_similarity(features, w, dim=-1)


def max_cosine_to_weights(features: torch.Tensor, weight_matrix: torch.Tensor) -> torch.Tensor:
    """Maximum cosine similarity over all class weight vectors.

    features:      [N, d]
    weight_matrix: [K, d]
    Returns: [N] max cosine similarities.
    """
    feat_norm = F.normalize(features, dim=-1)       # [N, d]
    w_norm = F.normalize(weight_matrix, dim=-1)     # [K, d]
    sims = feat_norm @ w_norm.T                      # [N, K]
    return sims.max(dim=-1).values


def centroid_distances(features: torch.Tensor, centroids: dict[int, torch.Tensor]) -> torch.Tensor:
    """Mean distance from each feature to the nearest frozen source centroid.

    features:  [N, d]
    centroids: {class_id: Tensor[d]}
    Returns:   [N] distances to nearest centroid.
    """
    centroid_matrix = torch.stack(list(centroids.values()))   # [C, d]
    dists = torch.cdist(features, centroid_matrix)            # [N, C]
    return dists.min(dim=-1).values


def radial_tangential_split(
    v: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split displacement v into radial (along z) and tangential components.

    v, z: [N, d]
    Returns (v_radial, v_tangential), each [N, d].
    """
    z_norm = F.normalize(z, dim=-1)                          # [N, d]
    radial_mag = (v * z_norm).sum(dim=-1, keepdim=True)      # [N, 1]
    v_radial = radial_mag * z_norm
    v_tangential = v - v_radial
    return v_radial, v_tangential


def radial_fraction(v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """||v_radial|| / ||v|| per sample.

    v, z: [N, d]
    Returns: [N].
    """
    v_r, _ = radial_tangential_split(v, z)
    return v_r.norm(dim=-1) / (v.norm(dim=-1) + 1e-8)
