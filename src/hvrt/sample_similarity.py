"""
Sample-wise Similarity Metrics for H-VRT

Provides different similarity metrics for comparing samples based on their feature distributions.
The choice of metric affects how the tree partitions samples.

**Key Distinction:**
- Pearson/Cosine: Linear similarity (fast, works for normal data)
- Mutual Information: Non-linear similarity (robust, handles complex distributions)
"""

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from typing import Literal


def compute_sample_similarity_matrix(
    X: np.ndarray,
    method: Literal['pearson', 'mi_binned', 'mi_knn'] = 'pearson',
    n_bins: int = 10,
    k_neighbors: int = 3
) -> np.ndarray:
    """
    Compute pairwise similarity matrix between samples.

    **Sample-wise** = Treat each sample as a distribution over features.
    This is STRUCTURAL: captures multivariate distributional patterns.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    method : {'pearson', 'mi_binned', 'mi_knn'}, default='pearson'
        Similarity metric:
        - 'pearson': Z-score + cosine similarity (linear, fast)
        - 'mi_binned': Mutual information with histogram binning (non-linear)
        - 'mi_knn': MI with k-NN estimator (non-linear, no binning)
    n_bins : int, default=10
        Number of bins for histogram-based MI (if method='mi_binned')
    k_neighbors : int, default=3
        Number of neighbors for k-NN MI estimator (if method='mi_knn')

    Returns
    -------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Symmetric similarity matrix where [i, j] = similarity between sample i and j

    Notes
    -----
    **Why Sample-wise MI is Structural:**

    Unlike feature-wise MI (which measures dependency between two features across
    samples), sample-wise MI measures how similar the distributional patterns of
    two samples are across all features.

    This captures:
    1. Non-linear relationships (heavy tails, skew, modality)
    2. Multivariate structure (entire feature vector)
    3. Distributional similarity (not just correlation)

    **Performance Characteristics:**

    Pearson (Z-score + cosine):
    - Complexity: O(n² * d)
    - Captures: Linear relationships
    - Best for: Normal distributions, well-behaved data
    - Fails when: Heavy tails, rare events, non-linear structure

    MI (Binned):
    - Complexity: O(n² * d * log(n_bins))
    - Captures: Non-linear dependencies
    - Best for: Heavy-tailed data, rare events
    - Tuning: n_bins controls granularity

    MI (k-NN):
    - Complexity: O(n² * d * log(k))
    - Captures: Non-linear dependencies (no binning artifacts)
    - Best for: Continuous data, no discretization needed
    - Tuning: k controls local neighborhood size

    Examples
    --------
    >>> X = np.random.randn(100, 10)  # 100 samples, 10 features
    >>>
    >>> # Pearson (fast, linear)
    >>> sim_pearson = compute_sample_similarity_matrix(X, method='pearson')
    >>>
    >>> # MI (robust to non-linear structure)
    >>> sim_mi = compute_sample_similarity_matrix(X, method='mi_binned', n_bins=10)
    """
    n_samples, n_features = X.shape

    if method == 'pearson':
        return _pearson_sample_similarity(X)
    elif method == 'mi_binned':
        return _mi_binned_sample_similarity(X, n_bins=n_bins)
    elif method == 'mi_knn':
        return _mi_knn_sample_similarity(X, k=k_neighbors)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: 'pearson', 'mi_binned', 'mi_knn'")


def _pearson_sample_similarity(X: np.ndarray) -> np.ndarray:
    """
    Pearson correlation via Z-score + cosine similarity.

    This is the current default approach:
    1. Z-score normalize features
    2. Cosine similarity = Pearson correlation

    **Limitation**: Only captures linear relationships.
    """
    n_samples = X.shape[0]

    # Z-score normalization
    X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    # Cosine similarity = Pearson correlation
    similarity = np.dot(X_normalized, X_normalized.T)

    # Normalize by sample norms
    norms = np.sqrt(np.diag(similarity))
    similarity = similarity / (norms[:, None] * norms[None, :] + 1e-10)

    # Ensure diagonal is 1.0
    np.fill_diagonal(similarity, 1.0)

    # Convert to [0, 1] range: (corr + 1) / 2
    similarity = (similarity + 1.0) / 2.0

    return similarity


def _mi_binned_sample_similarity(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Sample-wise mutual information using histogram binning.

    For each pair of samples (i, j):
    1. Treat sample i as a discrete distribution over feature bins
    2. Treat sample j as a discrete distribution over feature bins
    3. Compute MI(sample_i, sample_j)

    **Captures non-linear structure**: MI detects any statistical dependency,
    not just linear correlation.

    **Tuning n_bins:**
    - Too few bins: May miss fine-grained structure
    - Too many bins: Sparse histograms, noisy MI estimates
    - Rule of thumb: sqrt(n_features) to 2*sqrt(n_features)
    """
    n_samples, n_features = X.shape
    similarity = np.zeros((n_samples, n_samples))

    # Compute global bin edges for consistent discretization
    bin_edges = np.percentile(X, np.linspace(0, 100, n_bins + 1), axis=0)

    # Digitize all samples
    X_binned = np.zeros_like(X, dtype=int)
    for j in range(n_features):
        X_binned[:, j] = np.digitize(X[:, j], bin_edges[1:-1, j])

    # Compute pairwise MI
    for i in range(n_samples):
        for j in range(i, n_samples):
            if i == j:
                similarity[i, j] = 1.0  # Self-similarity is 1
            else:
                # Flatten binned samples into 1D distributions
                dist_i = X_binned[i, :]
                dist_j = X_binned[j, :]

                # Compute mutual information
                mi = mutual_info_score(dist_i, dist_j)

                # Normalize to [0, 1]: MI / log(n_bins)
                # Theoretical max MI = log(n_bins) when distributions are identical
                mi_normalized = mi / (np.log(n_bins) + 1e-10)

                similarity[i, j] = mi_normalized
                similarity[j, i] = mi_normalized

    return similarity


def _mi_knn_sample_similarity(X: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Sample-wise mutual information using k-NN estimator.

    Uses the Kraskov-Stoegbauer-Grassberger (KSG) estimator for continuous MI.
    This avoids binning artifacts and works better for continuous features.

    **Advantages:**
    - No binning required (works on continuous data directly)
    - More accurate for continuous distributions
    - Adapts to local density

    **Tuning k:**
    - Smaller k: More local, captures fine structure
    - Larger k: More global, smoother estimates
    - Rule of thumb: 3-10 neighbors

    Reference:
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information. Physical review E, 69(6), 066138.
    """
    from sklearn.feature_selection import mutual_info_regression

    n_samples, n_features = X.shape
    similarity = np.zeros((n_samples, n_samples))

    # For each pair of samples
    for i in range(n_samples):
        for j in range(i, n_samples):
            if i == j:
                similarity[i, j] = 1.0
            else:
                # Treat each sample as a distribution over features
                # Compute MI between the two feature distributions

                # Reshape for MI computation
                sample_i = X[i, :].reshape(-1, 1)  # (n_features, 1)
                sample_j = X[j, :]                # (n_features,)

                # Use k-NN MI estimator
                # Note: This treats features as "samples" for MI estimation
                mi = mutual_info_regression(sample_i, sample_j, n_neighbors=k, random_state=42)[0]

                # Normalize: MI is unbounded, but typically 0-3 for moderate dependence
                # Use sigmoid-like normalization: 1 - exp(-mi)
                mi_normalized = 1.0 - np.exp(-mi)

                similarity[i, j] = mi_normalized
                similarity[j, i] = mi_normalized

    return similarity


def recommend_similarity_method(X: np.ndarray) -> str:
    """
    Recommend similarity method based on data characteristics.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix

    Returns
    -------
    method : str
        Recommended method: 'pearson', 'mi_binned', or 'mi_knn'

    Notes
    -----
    Decision logic:
    1. Check for heavy tails (high kurtosis)
    2. Check for skewness
    3. Check sample size (MI needs sufficient samples)

    If data looks normal → Pearson (fast)
    If heavy-tailed or skewed → MI (robust)
    If small sample size (< 100) → Pearson (MI unreliable)
    """
    n_samples, n_features = X.shape

    # Small sample size: Pearson more reliable
    if n_samples < 100:
        return 'pearson'

    # Check for heavy tails (excess kurtosis > 1)
    from scipy.stats import kurtosis
    kurt = kurtosis(X, axis=0, fisher=True)  # Fisher=True gives excess kurtosis
    mean_kurtosis = np.abs(kurt).mean()

    # Check for skewness
    from scipy.stats import skew
    skewness = skew(X, axis=0)
    mean_skewness = np.abs(skewness).mean()

    # Decision rules
    if mean_kurtosis > 1.0:  # Heavy tails
        return 'mi_binned'
    elif mean_skewness > 0.75:  # Strong skew
        return 'mi_binned'
    else:  # Well-behaved data
        return 'pearson'
