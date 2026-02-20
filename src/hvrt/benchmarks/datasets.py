"""
Reproducible synthetic benchmark datasets for HVRT evaluation.

All datasets are generated programmatically (no external data files required)
so benchmarks are fully reproducible from a fixed random seed.

Datasets
--------
make_adult_like          n=20000, d=8,  classification, skewed capital-gain
make_fraud_like          n=20000, d=15, classification, 3% imbalance, clusters
make_housing_like        n=20000, d=6,  regression, log-normal features
make_multimodal          n=20000, d=10, regression, 3 distinct clusters
make_emergence_divergence n=10000, d=5, synthetic emergence pattern
make_emergence_bifurcation n=10000, d=5, synthetic emergence pattern
"""

import numpy as np


# ---------------------------------------------------------------------------
# Adult-like dataset
# ---------------------------------------------------------------------------

def make_adult_like(n=20000, random_state=42):
    """
    Simulate an Adult-census-like dataset.

    Features (continuous)
    ---------------------
    age              Normal(38, 13), clipped [18, 90]
    education_num    Integer 1..16, biased toward middle values
    hours_per_week   Normal(40, 12), clipped [1, 99]
    capital_gain     90% zero, 10% log-normal (extreme skew)
    capital_loss     95% zero, 5%  log-normal (extreme skew)
    fnlwgt           Log-normal (survey weight)
    relationship     Ordinal 0..5
    marital_status   Ordinal 0..6

    Target
    ------
    income : binary (0 = <=50K, 1 = >50K)

    Returns
    -------
    X : ndarray (n, 8)
    y : ndarray (n,) binary
    feature_names : list of str
    """
    rng = np.random.RandomState(random_state)

    age = np.clip(rng.normal(38, 13, n), 18, 90)
    edu = np.clip(rng.normal(10, 3, n), 1, 16).astype(float)
    hours = np.clip(rng.normal(40, 12, n), 1, 99)

    # Skewed capital gain: mixture
    capital_gain = np.where(
        rng.uniform(size=n) < 0.90,
        0.0,
        np.exp(rng.normal(9, 3, n)),
    )
    capital_loss = np.where(
        rng.uniform(size=n) < 0.95,
        0.0,
        np.exp(rng.normal(7, 2, n)),
    )
    fnlwgt = np.exp(rng.normal(11, 1, n))
    relationship = rng.randint(0, 6, n).astype(float)
    marital_status = rng.randint(0, 7, n).astype(float)

    # Binary target: logistic function of key features
    logit = (
        -5.0
        + 0.03 * age
        + 0.2 * edu
        + 0.02 * hours
        + 0.00001 * capital_gain
        + 0.5 * (relationship < 2).astype(float)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(int)

    X = np.column_stack([
        age, edu, hours, capital_gain, capital_loss, fnlwgt,
        relationship, marital_status,
    ])
    names = [
        'age', 'education_num', 'hours_per_week',
        'capital_gain', 'capital_loss', 'fnlwgt',
        'relationship', 'marital_status',
    ]
    return X, y, names


# ---------------------------------------------------------------------------
# Fraud-like dataset
# ---------------------------------------------------------------------------

def make_fraud_like(n=20000, random_state=42):
    """
    Simulate an imbalanced fraud-detection dataset.

    97% normal transactions (multivariate normal with moderate correlations).
    3% fraud transactions across 3 tight clusters in a corner of feature space.

    Returns
    -------
    X : ndarray (n, 15)
    y : ndarray (n,) binary  (1 = fraud)
    feature_names : list of str
    """
    rng = np.random.RandomState(random_state)
    n_fraud = int(n * 0.03)
    n_normal = n - n_fraud

    # Normal transactions
    cov_base = np.eye(15) * 0.5 + np.full((15, 15), 0.1)
    X_normal = rng.multivariate_normal(np.zeros(15), cov_base, n_normal)

    # Fraud: 3 tight clusters
    fraud_parts = []
    per_cluster = n_fraud // 3
    cluster_centres = [
        np.array([3, 3, -3, -3, 2, -2, 3, -3, 2, -2, 3, -3, 1, -1, 2]),
        np.array([-3, -3, 3, 3, -2, 2, -3, 3, -2, 2, -3, 3, -1, 1, -2]),
        np.array([4, -4, 4, -4, 0, 0, 4, -4, 0, 0, 4, -4, 2, -2, 0]),
    ]
    for c_idx, centre in enumerate(cluster_centres):
        cnt = per_cluster if c_idx < 2 else n_fraud - 2 * per_cluster
        fraud_parts.append(rng.multivariate_normal(centre, np.eye(15) * 0.3, cnt))
    X_fraud = np.vstack(fraud_parts)

    X = np.vstack([X_normal, X_fraud])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(int)

    # Shuffle
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    names = [f'v{i}' for i in range(1, 16)]
    return X, y, names


# ---------------------------------------------------------------------------
# Housing-like dataset
# ---------------------------------------------------------------------------

def make_housing_like(n=20000, random_state=42):
    """
    Simulate a housing-price regression dataset with log-normal features.

    Returns
    -------
    X : ndarray (n, 6)
    y : ndarray (n,) continuous (log-normal price)
    feature_names : list of str
    """
    rng = np.random.RandomState(random_state)

    sqft = np.exp(rng.normal(7.2, 0.5, n))         # house size
    lot_size = np.exp(rng.normal(8.0, 0.7, n))     # lot
    rooms = np.exp(rng.normal(1.8, 0.3, n))        # number of rooms
    age = np.exp(rng.normal(3.0, 0.8, n))          # house age
    distance = np.exp(rng.normal(3.5, 0.6, n))     # distance to CBD
    income = np.exp(rng.normal(10.5, 0.4, n))      # neighbourhood income

    log_price = (
        8.0
        + 0.6 * np.log(sqft)
        + 0.2 * np.log(lot_size)
        + 0.3 * np.log(rooms)
        - 0.15 * np.log(age)
        - 0.4 * np.log(distance)
        + 0.3 * np.log(income)
        + rng.normal(0, 0.3, n)
    )
    y = np.exp(log_price)

    X = np.column_stack([sqft, lot_size, rooms, age, distance, income])
    names = ['sqft', 'lot_size', 'rooms', 'age', 'distance_cbd', 'neighbourhood_income']
    return X, y, names


# ---------------------------------------------------------------------------
# Multimodal dataset
# ---------------------------------------------------------------------------

def make_multimodal(n=20000, random_state=42):
    """
    Simulate a dataset with 3 distinct Gaussian clusters in feature space.

    Returns
    -------
    X : ndarray (n, 10)
    y : ndarray (n,) continuous
    feature_names : list of str
    """
    rng = np.random.RandomState(random_state)

    cluster_props = [0.5, 0.3, 0.2]
    cluster_means = [
        np.zeros(10),
        np.array([5, 5, 5, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        np.array([-4, 0, 4, -4, 0, 4, 0, 0, 0, 0], dtype=float),
    ]
    cluster_stds = [1.0, 0.8, 1.2]
    cluster_slopes = [
        np.array([1.0, 0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.4, 0.7, 0.9]),
        np.array([0.5, 1.0, 0.8, 0.3, 0.6, 0.2, 0.9, 0.4, 0.1, 0.7]),
        np.array([0.8, 0.2, 0.6, 1.0, 0.4, 0.9, 0.5, 0.3, 0.7, 0.1]),
    ]

    X_parts, y_parts = [], []
    for prop, mean, std, slope in zip(cluster_props, cluster_means, cluster_stds, cluster_slopes):
        cnt = int(n * prop)
        Xc = rng.normal(0, std, (cnt, 10)) + mean
        yc = Xc @ slope + rng.normal(0, 0.5, cnt)
        X_parts.append(Xc)
        y_parts.append(yc)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    names = [f'f{i}' for i in range(1, 11)]
    return X, y, names


# ---------------------------------------------------------------------------
# Emergence datasets
# ---------------------------------------------------------------------------

def make_emergence_divergence(n=10000, random_state=42):
    """
    Emergence-Divergence dataset.

    A non-linear, non-additive response: when f1 * f2 exceeds a threshold
    the outcome diverges sharply from its baseline, creating an interaction-
    driven emergence pattern that simple additive models miss.

    Returns
    -------
    X : ndarray (n, 5)
    y : ndarray (n,) continuous
    feature_names : list of str
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, (n, 5))

    # Base signal
    y = 2 * X[:, 0] + X[:, 1] + 0.5 * X[:, 2]

    # Emergence: interaction term causes divergence above threshold
    interaction = X[:, 0] * X[:, 1]
    divergence_mask = interaction > 1.5
    y[divergence_mask] += 5 * interaction[divergence_mask]

    y += rng.normal(0, 0.5, n)

    names = ['f1', 'f2', 'f3', 'f4', 'f5']
    return X, y, names


def make_emergence_bifurcation(n=10000, random_state=42):
    """
    Emergence-Bifurcation dataset.

    Samples with |f1| > threshold bifurcate into two branches based on the
    sign of f2, creating a non-linear splitting emergence pattern.

    Returns
    -------
    X : ndarray (n, 5)
    y : ndarray (n,) continuous
    feature_names : list of str
    """
    rng = np.random.RandomState(random_state)

    X = rng.normal(0, 1, (n, 5))
    y = X[:, 0] + 0.5 * X[:, 2] + rng.normal(0, 0.5, n)

    # Bifurcation when |f1| exceeds threshold
    extreme_mask = np.abs(X[:, 0]) > 1.5
    y[extreme_mask & (X[:, 1] > 0)] += 4 * np.abs(X[extreme_mask & (X[:, 1] > 0), 0])
    y[extreme_mask & (X[:, 1] <= 0)] -= 4 * np.abs(X[extreme_mask & (X[:, 1] <= 0), 0])

    names = ['f1', 'f2', 'f3', 'f4', 'f5']
    return X, y, names


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

BENCHMARK_DATASETS = {
    'adult':                 make_adult_like,
    'fraud':                 make_fraud_like,
    'housing':               make_housing_like,
    'multimodal':            make_multimodal,
    'emergence_divergence':  make_emergence_divergence,
    'emergence_bifurcation': make_emergence_bifurcation,
}
