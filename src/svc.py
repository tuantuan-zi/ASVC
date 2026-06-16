"""
Original Support Vector Clustering (SVC) algorithm.
Based on Ben-Hur, Horn, Siegelmann, Vapnik (2000, 2001).
"""
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from .utils import pairwise_distances_squared, gaussian_kernel

SV_TOL = 1e-6

def _solve_dual_qp(K, C):
    """Solve the SVDD dual QP: min alpha^T K alpha s.t. 0 <= alpha_i <= C, sum alpha = 1.
    Tries trust-constr first (better for QP), falls back to SLSQP.
    """
    N = K.shape[0]
    reg = 1e-8
    K_reg = K + reg * np.eye(N)

    def objective(alpha):
        return alpha @ K_reg @ alpha

    def gradient(alpha):
        return 2 * K_reg @ alpha

    x0 = np.ones(N) / N
    bounds = Bounds([0.0] * N, [C] * N)
    constraints = LinearConstraint(np.ones(N), 1.0, 1.0)

    for method in ['SLSQP', 'trust-constr']:
        try:
            opt = {'maxiter': 2000, 'gtol': 1e-12, 'xtol': 1e-12} if method == 'trust-constr' \
                   else {'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
            result = minimize(objective, x0, method=method, jac=gradient,
                            bounds=bounds, constraints=constraints, options=opt)
            alpha = np.clip(result.x, 0, C)
            alpha /= alpha.sum()
            return alpha, result.success
        except Exception:
            continue

    alpha = np.ones(N) / N
    return alpha, False

def _compute_R_squared(x, X, alpha, q, K_diag_terms=None):
    """Compute R^2(y): squared feature-space distance from sphere center."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    D_sq = np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(x ** 2, axis=1)[np.newaxis, :] \
           - 2 * X @ x.T
    D_sq = np.maximum(D_sq, 0)
    K_vals = np.exp(-q * D_sq)
    linear_terms = 2 * alpha @ K_vals
    if K_diag_terms is None:
        K_mat = gaussian_kernel(X, q)
        K_diag_terms = alpha @ K_mat @ alpha
    R_sq = 1.0 - linear_terms + K_diag_terms
    return R_sq.squeeze()

def _classify_points(alpha, C, tol=SV_TOL):
    """Classify points: SV (0 < alpha < C), BSV (alpha ~= C), interior (alpha ~= 0)."""
    interior_mask = alpha <= tol
    bsv_mask = np.abs(alpha - C) <= tol
    sv_mask = ~(interior_mask | bsv_mask)
    return sv_mask, bsv_mask, interior_mask

def _connectivity_labeling(X, alpha, q, C, R_sv, sv_mask, bsv_mask,
                           n_sample_points=30, k_neighbors=15):
    """SVC cluster labeling via k-NN filtered line-segment sampling.
    
    Only checks spatial k-nearest-neighbor pairs instead of all O(N^2) pairs.
    This prevents false bridges between clusters and is much faster.
    """
    N = X.shape[0]
    K_mat = gaussian_kernel(X, q)
    K_diag_terms = alpha @ K_mat @ alpha
    R_sq_all = _compute_R_squared(X, X, alpha, q, K_diag_terms)
    
    inside_mask = R_sq_all <= R_sv + SV_TOL
    inside_mask[bsv_mask] = False
    inside_indices = np.where(inside_mask)[0]
    n_inside = len(inside_indices)

    if n_inside == 0:
        return np.zeros(N, dtype=int)

    X_inside = X[inside_indices]

    # Only check spatial neighbors (k-NN) -- prevents false bridges
    k = min(max(k_neighbors, n_sample_points // 2), n_inside - 1)
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_inside)
    knn_graph = nn.kneighbors_graph(X_inside, mode='connectivity')

    row_indices = []
    col_indices = []

    knn_rows, knn_cols = knn_graph.nonzero()
    for li, lj in zip(knn_rows, knn_cols):
        if li >= lj:
            continue
        xi, xj = X_inside[li], X_inside[lj]

        # Adaptive sampling: fewer samples for short segments
        seg_len = np.sqrt(max(np.sum((xi - xj) ** 2), 0))
        n_samp = max(5, min(n_sample_points, int(n_sample_points * seg_len)))
        n_samp = max(n_samp, 3)

        connected = True
        for t in np.linspace(0, 1, n_samp + 2)[1:-1]:
            y = xi + t * (xj - xi)
            R_sq_y = _compute_R_squared(y, X, alpha, q, K_diag_terms)
            if R_sq_y > R_sv + SV_TOL:
                connected = False
                break
        
        if connected:
            row_indices.extend([li, lj])
            col_indices.extend([lj, li])

    if len(row_indices) == 0:
        labels = -np.ones(N, dtype=int)
        for l_idx in range(n_inside):
            labels[inside_indices[l_idx]] = l_idx
        return labels

    data = np.ones(len(row_indices))
    adj = csr_matrix((data, (row_indices, col_indices)), shape=(n_inside, n_inside))
    n_components, component_labels = connected_components(adj, directed=False)

    labels = -np.ones(N, dtype=int)
    for l_idx in range(n_inside):
        labels[inside_indices[l_idx]] = component_labels[l_idx]
    return labels

def _assign_bsvs(X, labels, bsv_mask, R_sq_all):
    """Assign BSVs to nearest labeled cluster."""
    bsv_indices = np.where(bsv_mask)[0]
    if len(bsv_indices) == 0:
        return labels
    labeled_mask = labels >= 0
    labeled_indices = np.where(labeled_mask)[0]
    labeled_labels = labels[labeled_mask]
    for bsv_idx in bsv_indices:
        distances = np.sum((X[bsv_idx] - X[labeled_indices]) ** 2, axis=1)
        nearest = np.argmin(distances)
        labels[bsv_idx] = labeled_labels[nearest]
    return labels

class SVC:
    """Support Vector Clustering (Ben-Hur et al., 2001).
    
    Parameters
    ----------
    q : float
        Gaussian kernel width.
    C : float
        Soft-margin penalty parameter.
    n_sample_points : int
        Samples per line segment for connectivity check.
    k_neighbors : int
        Number of spatial neighbors for connectivity graph.
    """
    def __init__(self, q=1.0, C=1.0, n_sample_points=30, k_neighbors=15):
        self.q = q
        self.C = C
        self.n_sample_points = n_sample_points
        self.k_neighbors = k_neighbors

    def fit(self, X):
        N = X.shape[0]
        self.X_ = X
        self.C = max(self.C, 1.0 / N)

        K = gaussian_kernel(X, self.q)
        self.alpha_, success = _solve_dual_qp(K, self.C)
        if not success:
            import warnings
            warnings.warn("QP solver may be suboptimal.")

        self.sv_mask_, self.bsv_mask_, self.interior_mask_ = \
            _classify_points(self.alpha_, self.C)

        K_diag_terms = self.alpha_ @ K @ self.alpha_
        R_sq_all = _compute_R_squared(X, X, self.alpha_, self.q, K_diag_terms)
        if self.sv_mask_.any():
            self.R_sv_ = np.median(R_sq_all[self.sv_mask_])
        else:
            non_bsv = ~self.bsv_mask_
            self.R_sv_ = R_sq_all[non_bsv].max() if non_bsv.any() else R_sq_all.mean()

        self.labels_ = _connectivity_labeling(
            X, self.alpha_, self.q, self.C, self.R_sv_,
            self.sv_mask_, self.bsv_mask_,
            n_sample_points=self.n_sample_points,
            k_neighbors=self.k_neighbors
        )
        self.labels_ = _assign_bsvs(X, self.labels_, self.bsv_mask_, R_sq_all)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
