"""
Utility functions for SVC and ASVC implementations.
Includes data loading, evaluation metrics, and data generation helpers.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

def standardize(X):
    """Standardize data to zero mean and unit variance."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def pairwise_distances_squared(X):
    """Compute squared Euclidean distance matrix.
    Returns (N, N) matrix D where D[i,j] = ||x_i - x_j||^2.
    """
    sum_sq = np.sum(X ** 2, axis=1)
    D = sum_sq[:, np.newaxis] + sum_sq[np.newaxis, :] - 2 * X @ X.T
    D = np.maximum(D, 0)  # clip small negatives from numerical error
    return D

def gaussian_kernel(X, q):
    """Compute Gaussian kernel matrix K where K[i,j] = exp(-q * ||x_i - x_j||^2).
    Args:
        X: (N, d) data matrix.
        q: kernel width parameter.
    Returns:
        K: (N, N) kernel matrix.
    """
    D_sq = pairwise_distances_squared(X)
    return np.exp(-q * D_sq)

def clustering_metrics(y_true, y_pred):
    """Compute comprehensive clustering evaluation metrics.
    Returns dict with ARI, NMI, homogeneity, completeness, V-measure.
    """
    return {
        'ARI': adjusted_rand_score(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred),
        'Homogeneity': homogeneity_score(y_true, y_pred),
        'Completeness': completeness_score(y_true, y_pred),
        'V_measure': v_measure_score(y_true, y_pred),
    }

def generate_synthetic_datasets(random_state=42):
    """Generate a collection of synthetic clustering datasets for testing.
    Returns dict of (X, y) tuples keyed by dataset name.
    """
    datasets = {}
    # Two moons
    X, y = make_moons(n_samples=300, noise=0.08, random_state=random_state)
    datasets['TwoMoons'] = (standardize(X), y)
    # Concentric circles
    X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=random_state)
    datasets['ConcentricCircles'] = (standardize(X), y)
    # Three blobs
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0,
                      random_state=random_state)
    datasets['ThreeBlobs'] = (standardize(X), y)
    # Overlapping Gaussians (3 clusters, high overlap)
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.0,
                      random_state=random_state)
    datasets['OverlappingGaussians'] = (standardize(X), y)
    # Varying-sized blobs (different variances, same density)
    # DBSCAN struggles with this because a single eps cannot handle
    # all three cluster scales simultaneously.
    centers = [[-4, -4], [4, 4], [4, -4]]
    stds = [0.5, 2.0, 3.0]
    X_parts = []
    y_parts = []
    for i, (c, s) in enumerate(zip(centers, stds)):
        np.random.seed(random_state + i * 100)
        Xp = np.random.randn(100, 2) * s + np.array(c)
        X_parts.append(Xp)
        y_parts.append(np.full(100, i))
    X = np.vstack(X_parts)
    y = np.hstack(y_parts)
    datasets['VaryingBlobs'] = (standardize(X), y)

    # Two moons + noise (outliers)
    X, y = make_moons(n_samples=300, noise=0.12, random_state=random_state)
    datasets['TwoMoonsNoisy'] = (standardize(X), y)
    return datasets

def load_uci_iris():
    """Load the Iris dataset from scikit-learn.
    Returns X, y, feature_names.
    """
    from sklearn.datasets import load_iris
    data = load_iris()
    return standardize(data.data), data.target, data.feature_names

def load_uci_wine():
    """Load the Wine dataset from scikit-learn."""
    from sklearn.datasets import load_wine
    data = load_wine()
    return standardize(data.data), data.target, data.feature_names

def load_uci_breast_cancer():
    """Load the Breast Cancer Wisconsin dataset from scikit-learn."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    return standardize(data.data), data.target, data.feature_names