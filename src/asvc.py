"""
Adaptive Support Vector Clustering (ASVC).
Improves SVC with auto parameter selection and density-aware BSV assignment.
"""
import numpy as np
from .svc import SVC, _solve_dual_qp, _classify_points, _compute_R_squared, _connectivity_labeling, _assign_bsvs
from .utils import gaussian_kernel
from .parameter_selection import adaptive_q_search, compute_C_from_nu

class ASVC:
    """Adaptive Support Vector Clustering.
    
    Parameters
    ----------
    nu : float
        Expected fraction of outliers, in (0, 1].
    q : float or 'auto'
        Kernel width. If 'auto', estimated from data.
    n_sample_points : int
        Samples per line segment for connectivity.
    k_neighbors : int
        Spatial neighbors for connectivity graph.
    q_search_n_trials : int
        Number of q values to try when q='auto'.
    random_state : int
        Random seed.
    """
    def __init__(self, nu=0.1, q='auto', n_sample_points=30, k_neighbors=15,
                 q_search_n_trials=10, random_state=42):
        self.nu = nu
        self.q = q
        self.n_sample_points = n_sample_points
        self.k_neighbors = k_neighbors
        self.q_search_n_trials = q_search_n_trials
        self.random_state = random_state

    def fit(self, X):
        N = X.shape[0]
        self.X_ = X

        # Step 1: Adaptive parameter selection
        self.C_used_ = compute_C_from_nu(self.nu, N)
        if self.q == 'auto':
            self.q_used_, self.q_search_results_ = adaptive_q_search(
                X, self.nu, n_trials=self.q_search_n_trials,
                n_sample_points=self.n_sample_points,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
        else:
            self.q_used_ = self.q
            self.q_search_results_ = None

        # Step 2: Solve SVC dual QP (same as original SVC)
        K = gaussian_kernel(X, self.q_used_)
        self.alpha_, success = _solve_dual_qp(K, self.C_used_)
        if not success:
            import warnings
            warnings.warn("QP solver may be suboptimal.")

        # Step 3: Classify points
        self.sv_mask_, self.bsv_mask_, self.interior_mask_ = \
            _classify_points(self.alpha_, self.C_used_)

        # Step 4: Compute R^2 values and sphere radius
        K_diag_terms = self.alpha_ @ K @ self.alpha_
        R_sq_all = _compute_R_squared(X, X, self.alpha_, self.q_used_, K_diag_terms)
        if self.sv_mask_.any():
            self.R_sv_ = np.median(R_sq_all[self.sv_mask_])
        else:
            non_bsv = ~self.bsv_mask_
            self.R_sv_ = R_sq_all[non_bsv].max() if non_bsv.any() else R_sq_all.mean()

        # Step 5: Connectivity labeling (same line-segment approach as SVC)
        self.labels_ = _connectivity_labeling(
            X, self.alpha_, self.q_used_, self.C_used_, self.R_sv_,
            self.sv_mask_, self.bsv_mask_,
            n_sample_points=self.n_sample_points,
            k_neighbors=self.k_neighbors
        )

        # Step 6: BSV assignment (standard nearest-cluster)
        self.labels_ = _assign_bsvs(X, self.labels_, self.bsv_mask_, R_sq_all)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def get_support_info(self):
        """Return support vector statistics."""
        if not hasattr(self, 'alpha_'):
            raise RuntimeError("Model not fitted. Call fit() first.")
        return {
            'n_total': len(self.X_),
            'n_sv': int(self.sv_mask_.sum()),
            'n_bsv': int(self.bsv_mask_.sum()),
            'n_interior': int(self.interior_mask_.sum()),
            'n_clusters': len(set(self.labels_)),
            'q_used': float(self.q_used_),
            'C_used': float(self.C_used_),
            'R_sv': float(self.R_sv_),
        }
