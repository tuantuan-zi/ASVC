"""Adaptive parameter selection for SVC and ASVC."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_q_range(X, k=8):
    """Estimate a reasonable q search range from k-NN distances.

    Uses k=8 (fewer neighbors) to reduce cross-cluster contamination,
    especially for datasets like ConcentricCircles where points from
    different clusters are spatially interleaved.

    q_base = 1 / (2 * d_p25^2) where d_p25 is the 25th percentile of
    k-NN distances (more robust than median for different geometries).
    """
    N = X.shape[0]
    k_use = min(k + 1, N)
    nn = NearestNeighbors(n_neighbors=k_use)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    # distances[:, 0] is self-distance (0), use columns 1..k
    knn_distances = distances[:, 1:].ravel()
    d_p25 = np.percentile(knn_distances, 25)
    d_med = np.median(knn_distances)
    if d_p25 < 1e-10:
        d_p25 = 1e-6
    if d_med < 1e-10:
        d_med = 1e-6
    q_center = 1.0 / (2.0 * d_med ** 2)
    q_min = q_center * 0.02  # much wider lower bound
    q_max = q_center * 80.0
    # Safety: ensure minimum search breadth
    if q_max / max(q_min, 1e-10) < 10.0:
        q_min = q_max / 100.0
    return q_min, q_max


def compute_C_from_nu(nu, N):
    """C = 1 / (nu * N), following nu-SVM parameterization."""
    nu = np.clip(nu, 1e-5, 1.0)
    return 1.0 / (nu * N)


def _score_clustering(res, N):
    """Score a clustering result on a continuous scale."""
    n_clus = res["n_clusters"]
    if n_clus <= 0:
        return -100.0
    sv_ratio = res["n_sv"] / N
    bsv_ratio = res["n_bsv"] / N
    # cluster bonus
    cluster_score = np.clip(np.log2(max(n_clus, 1)) * 0.8, 0, 2.0)
    # SV score: Gaussian peak at sv_ratio~0.25
    sv_score = np.exp(-((sv_ratio - 0.25) ** 2) / (2 * 0.25 ** 2))
    # BSV penalty
    if bsv_ratio < 0.05:
        bsv_penalty = 0.0
    elif bsv_ratio < 0.3:
        bsv_penalty = bsv_ratio * 1.5
    else:
        bsv_penalty = bsv_ratio * 5.0 + 2.0
    return cluster_score + sv_score * 1.5 - bsv_penalty


def adaptive_q_search(X, nu, n_trials=20, n_sample_points=30,
                      k_neighbors=15, random_state=42):
    """Search for the optimal q using two-phase search.

    Phase 1: Coarse search over wide range.
    Phase 2: Fine search around best q from phase 1.
    """
    from .svc import SVC
    N = X.shape[0]
    q_min, q_max = estimate_q_range(X)
    C = compute_C_from_nu(nu, N)
    # Phase 1: coarse search
    q_safe_min = max(q_min, 1e-8)
    q_values = np.logspace(np.log10(q_safe_min), np.log10(q_max), n_trials)
    all_results = []
    for q in q_values:
        try:
            svc = SVC(q=q, C=C, n_sample_points=n_sample_points,
                       k_neighbors=k_neighbors)
            svc.fit(X)
            n_clusters = len(set(l for l in svc.labels_ if l >= 0))
            n_sv = int(svc.sv_mask_.sum())
            n_bsv = int(svc.bsv_mask_.sum())
            all_results.append({"q": q, "C": C,
                "n_clusters": n_clusters, "n_sv": n_sv,
                "n_bsv": n_bsv, "labels": svc.labels_})
        except Exception:
            all_results.append({"q": q, "C": C,
                "n_clusters": 0, "n_sv": 0, "n_bsv": 0, "labels": None})
    # Score and select best
    best_score = -np.inf
    best_q = q_values[len(q_values) // 2]
    for res in all_results:
        score = _score_clustering(res, N)
        if score > best_score:
            best_score = score
            best_q = res["q"]
    # Phase 2: fine search around best_q
    if len(all_results) >= 10:
        fine_ratio = 0.3
        q_fine_min = max(best_q * (1 - fine_ratio), q_safe_min)
        q_fine_max = best_q * (1 + fine_ratio)
        if q_fine_max > q_fine_min * 1.1:
            q_fine = np.linspace(q_fine_min, q_fine_max, 8)
            for q in q_fine:
                try:
                    svc = SVC(q=q, C=C, n_sample_points=n_sample_points,
                               k_neighbors=k_neighbors)
                    svc.fit(X)
                    n_clusters = len(set(l for l in svc.labels_ if l >= 0))
                    n_sv = int(svc.sv_mask_.sum())
                    n_bsv = int(svc.bsv_mask_.sum())
                    res = {"q": q, "C": C, "n_clusters": n_clusters,
                        "n_sv": n_sv, "n_bsv": n_bsv, "labels": svc.labels_}
                    all_results.append(res)
                    score = _score_clustering(res, N)
                    if score > best_score:
                        best_score = score
                        best_q = res["q"]
                except Exception:
                    pass
    return best_q, all_results
