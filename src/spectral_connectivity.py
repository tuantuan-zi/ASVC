# Density-aware BSV assignment and spectral utilities
import numpy as np
from scipy.sparse import csr_matrix

def _assign_bsvs_density_aware(X, labels, bsv_mask, R_sq_all):
    bsv_indices = np.where(bsv_mask)[0]
    if len(bsv_indices) == 0:
        return labels
    labeled_mask = labels >= 0
    labeled_indices = np.where(labeled_mask)[0]
    unique_clusters = np.unique(labels[labeled_mask])
    for bsv_idx in bsv_indices:
        x_b = X[bsv_idx]
        R_b = R_sq_all[bsv_idx]
        dists = np.sum((x_b - X[labeled_indices]) ** 2, axis=1)
        cluster_scores = {}
        for c in unique_clusters:
            cluster_mask = labels[labeled_indices] == c
            if not cluster_mask.any():
                continue
            cluster_dists = dists[cluster_mask]
            cluster_R = R_sq_all[labeled_indices][cluster_mask]
            k = min(5, len(cluster_dists))
            nearest_idx = np.argpartition(cluster_dists, min(k, len(cluster_dists) - 1))[:k]
            dist_score = 1.0 / (np.sqrt(cluster_dists[nearest_idx]).mean() + 1e-10)
            R_sim = np.exp(-np.abs(R_b - cluster_R[nearest_idx]).mean())
            cluster_scores[c] = dist_score * R_sim
        if cluster_scores:
            labels[bsv_idx] = max(cluster_scores, key=cluster_scores.get)
        else:
            nearest = np.argmin(dists)
            labels[bsv_idx] = labels[labeled_indices[nearest]]
    return labels

def _estimate_n_clusters(A, max_k=12):
    from scipy.sparse.linalg import eigsh
    N = A.shape[0]
    max_k = min(max_k, N - 2)
    if max_k <= 2:
        return 2
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0)
    rows, cols = A.nonzero()
    L_data = -A.data * d_inv_sqrt[rows] * d_inv_sqrt[cols]
    L_sp = csr_matrix((L_data, (rows, cols)), shape=A.shape)
    L_sp = L_sp.tolil(); L_sp.setdiag(1.0 + L_sp.diagonal()); L_sp = L_sp.tocsr()
    try:
        eigenvalues = eigsh(L_sp, k=max_k + 1, which='SM', return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
        gaps = eigenvalues[1:] - eigenvalues[:-1]
        k_opt = np.argmax(gaps[:max_k]) + 1
        return max(2, k_opt)
    except Exception:
        return 3
