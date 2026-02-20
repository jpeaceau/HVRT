"""
Benchmark runners for HVRT v2.

Implements the full benchmark matrix:

Reduction (96 configurations):
  4 datasets × 4 ratios × 6 competitor methods
  + all 4 HVRT-family reduce combinations
  = 4 × 4 × (4 + 6) = 160 configurations total

Expansion (runnable methods, small-n by default):
  4 datasets × 3 expansion ratios × (4 HVRT + 5 classical + 2 deep-learning)
  Training set capped at n=500 by default to test generation from small data.
  Published reference rows (TabDDPM†, MOSTLY AI†) appended when methods='all'.

The 8 HVRT/FastHVRT combinations (per the specification):
  1. HVRT      + size-weighted  + reduce
  2. HVRT      + variance-weighted + reduce
  3. HVRT      + size-weighted  + expand
  4. HVRT      + variance-weighted + expand
  5. FastHVRT  + size-weighted  + reduce
  6. FastHVRT  + variance-weighted + reduce
  7. FastHVRT  + size-weighted  + expand
  8. FastHVRT  + variance-weighted + expand
"""

import json
import time
import warnings
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split

from .. import HVRT, FastHVRT
from .datasets import BENCHMARK_DATASETS, make_emergence_divergence, make_emergence_bifurcation
from .metrics import evaluate_reduction, evaluate_expansion, ml_utility_tstr


# ---------------------------------------------------------------------------
# Competitor: Kennard-Stone
# ---------------------------------------------------------------------------

def _kennard_stone(X, n_select):
    """
    Memory-efficient Kennard-Stone uniform distribution sampling.
    O(n × n_select) time, O(n) memory.
    """
    n = len(X)
    if n_select >= n:
        return np.arange(n)

    # Find initial pair (most extreme samples)
    chunk = 500
    max_dist = -1.0
    max_i, max_j = 0, 1
    for i in range(0, n, chunk):
        Xi = X[i:i + chunk]
        diff = Xi[:, None, :] - X[None, :, :]           # (chunk, n, d)
        D_chunk = np.sqrt((diff ** 2).sum(axis=2))       # (chunk, n)
        local = np.unravel_index(np.argmax(D_chunk), D_chunk.shape)
        if D_chunk[local] > max_dist:
            max_dist = D_chunk[local]
            max_i = i + local[0]
            max_j = local[1]

    selected = [int(max_i), int(max_j)]
    remaining = np.ones(n, dtype=bool)
    remaining[max_i] = remaining[max_j] = False

    # Initialise min-dist vector
    min_dists = np.minimum(
        np.sqrt(((X - X[max_i]) ** 2).sum(axis=1)),
        np.sqrt(((X - X[max_j]) ** 2).sum(axis=1)),
    )

    for _ in range(n_select - 2):
        remaining_idx = np.where(remaining)[0]
        next_local = np.argmax(min_dists[remaining])
        next_global = int(remaining_idx[next_local])
        selected.append(next_global)
        remaining[next_global] = False
        # Update running min-dist
        d_to_new = np.sqrt(((X - X[next_global]) ** 2).sum(axis=1))
        np.minimum(min_dists, d_to_new, out=min_dists)

    return np.array(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# Competitor: QR Pivot
# ---------------------------------------------------------------------------

def _qr_pivot(X, n_select):
    """
    QR decomposition with column pivoting for representative sample selection.
    """
    from scipy.linalg import qr
    _, _, piv = qr(X.T, pivoting=True)
    return piv[:n_select].astype(np.int64)


# ---------------------------------------------------------------------------
# Competitor: Stratified (for reduction)
# ---------------------------------------------------------------------------

def _stratified_reduce(X, y, n_select, random_state=42):
    """
    Stratified sampling: preserve class proportions (classification) or
    quantile proportions (regression).
    """
    n = len(X)
    n_bins = min(20, n_select)

    if len(np.unique(y)) <= 20:
        # Classification: sample proportionally per class
        classes, counts = np.unique(y, return_counts=True)
        idx = []
        rng = np.random.RandomState(random_state)
        for cls, cnt in zip(classes, counts):
            cls_idx = np.where(y == cls)[0]
            k = max(1, int(round(n_select * cnt / n)))
            k = min(k, len(cls_idx))
            idx.extend(rng.choice(cls_idx, k, replace=False).tolist())
        return np.array(idx[:n_select], dtype=np.int64)
    else:
        # Regression: quantile bins
        bins = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        bin_ids = np.digitize(y, bins[1:-1])
        idx = []
        rng = np.random.RandomState(random_state)
        unique_bins = np.unique(bin_ids)
        for bid in unique_bins:
            bin_idx = np.where(bin_ids == bid)[0]
            cnt = len(bin_idx)
            k = max(1, int(round(n_select * cnt / n)))
            k = min(k, len(bin_idx))
            idx.extend(rng.choice(bin_idx, k, replace=False).tolist())
        return np.array(idx[:n_select], dtype=np.int64)


# ---------------------------------------------------------------------------
# Expansion competitors
# ---------------------------------------------------------------------------

def _gmm_expand(X, n_synthetic, random_state=42):
    from sklearn.mixture import GaussianMixture
    n_components = min(20, max(2, len(X) // 200))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gmm = GaussianMixture(n_components=n_components, random_state=random_state,
                              covariance_type='full', max_iter=200)
        gmm.fit(X)
        X_synth, _ = gmm.sample(n_synthetic)
    return X_synth


def _gaussian_copula_expand(X, n_synthetic, random_state=42):
    """Empirical Gaussian copula via rank-based normalisation."""
    from scipy.stats import norm
    from scipy.stats import rankdata as _rankdata

    n, d = X.shape
    # Empirical CDF via ranks
    U = np.column_stack([_rankdata(X[:, j]) / (n + 1) for j in range(d)])
    # Map to normal
    Z = norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
    mu = Z.mean(axis=0)
    cov = np.cov(Z.T) if d > 1 else np.array([[Z.var()]])

    rng = np.random.RandomState(random_state)
    Z_synth = rng.multivariate_normal(mu, cov, n_synthetic)
    U_synth = norm.cdf(Z_synth)

    # Map back to original marginals via empirical quantile
    X_synth = np.column_stack([
        np.quantile(X[:, j], np.clip(U_synth[:, j], 0, 1))
        for j in range(d)
    ])
    return X_synth


def _bootstrap_noise_expand(X, n_synthetic, noise_level=0.1, random_state=42):
    """Resample with replacement, add Gaussian noise scaled to feature std."""
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), n_synthetic, replace=True)
    X_boot = X[idx].copy()
    stds = X.std(axis=0)
    stds = np.where(stds > 1e-10, stds, 1.0)
    X_boot += rng.normal(0, noise_level, X_boot.shape) * stds
    return X_boot


def _smote_expand(X, n_synthetic, random_state=42):
    """
    SMOTE-style k-NN interpolation expansion.

    Treats all samples as a single class and synthesises n_synthetic
    additional points by interpolating between k nearest neighbours.
    No label variable is required; the dummy-class trick is used to
    satisfy imbalanced-learn's API.

    Requires: pip install imbalanced-learn
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "Install with: pip install imbalanced-learn"
        )

    n = len(X)
    # SMOTE requires ≥2 classes.  Create one fake minority sample so the API
    # is satisfied, then oversample the majority (class 0) to n + n_synthetic.
    X_aug = np.vstack([X, X[0:1]])          # n+1 samples
    y_dummy = np.zeros(n + 1, dtype=int)
    y_dummy[-1] = 1                          # one minority sample
    n_target = n + n_synthetic              # desired size of class 0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sm = SMOTE(
            sampling_strategy={0: n_target},
            k_neighbors=min(5, n - 1),
            random_state=random_state,
        )
        X_res, y_res = sm.fit_resample(X_aug, y_dummy)

    # Return only the synthetic class-0 samples beyond the original n
    return X_res[y_res == 0][n:]


def _ctgan_expand(X, n_synthetic, epochs=300, random_state=42):
    """
    CTGAN expansion (requires ctgan library: pip install ctgan).

    Conditional Tabular GAN.  Slower than classical methods but captures
    complex conditional distributions.
    """
    try:
        from ctgan import CTGAN
        import pandas as pd
    except ImportError:
        raise ImportError(
            "ctgan is required for CTGAN. Install with: pip install ctgan"
        )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = CTGAN(epochs=epochs, verbose=False)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        model.fit(df)
        X_synth_df = model.sample(n_synthetic)
    return X_synth_df.values


def _tvae_expand(X, n_synthetic, epochs=300, random_state=42):
    """
    TVAE expansion (requires ctgan library: pip install ctgan).

    Tabular Variational Autoencoder.
    """
    try:
        from ctgan import TVAE
        import pandas as pd
    except ImportError:
        raise ImportError(
            "ctgan is required for TVAE. Install with: pip install ctgan"
        )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = TVAE(epochs=epochs)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        model.fit(df)
        X_synth_df = model.sample(n_synthetic)
    return X_synth_df.values


# Published benchmark numbers for methods that cannot be run locally.
# Source: HVRT_V2_SPECIFICATION_REVISED.md §4 and published papers.
# Appended as reference rows (published_only=True) when methods='all'.
PUBLISHED_EXPANSION_REFERENCES = [
    {
        'method': 'TabDDPM\u2020',
        'metrics': {
            'marginal_fidelity':      0.960,
            'discriminator_accuracy': 0.520,
            'tail_preservation':      0.700,
            'correlation_fidelity':   0.970,
        },
        'note': 'Kotelnikov et al. 2023 — not run locally',
        'published_only': True,
    },
    {
        'method': 'MOSTLY AI\u2020',
        'metrics': {
            'marginal_fidelity':      0.975,
            'discriminator_accuracy': 0.510,
            'tail_preservation':      0.850,
            'correlation_fidelity':   0.980,
        },
        'note': 'MOSTLY AI evaluation 2024 — commercial service, not run locally',
        'published_only': True,
    },
]


# ---------------------------------------------------------------------------
# Single-run helpers
# ---------------------------------------------------------------------------

def _run_one_reduction(dataset_name, X_train, y_train, X_test, y_test,
                        method_name, ratio, random_state, is_emergence,
                        trtr_baseline=None):
    """Execute one reduction configuration and return a result dict."""
    n_target = max(2, int(len(X_train) * ratio))
    t0 = time.perf_counter()

    if method_name in (
        'HVRT-size', 'HVRT-var', 'FastHVRT-size', 'FastHVRT-var'
    ):
        ModelCls = HVRT if method_name.startswith('HVRT-') else FastHVRT
        var_weighted = method_name.endswith('-var')
        model = ModelCls(random_state=random_state)
        model.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0
        t1 = time.perf_counter()
        X_red, idx = model.reduce(n=n_target, variance_weighted=var_weighted, return_indices=True)
        y_red = y_train[idx]
        t_op = time.perf_counter() - t1

    elif method_name == 'Kennard-Stone':
        # Cap at 5 000 rows — KS is O(n × n_select) and becomes prohibitive on
        # large datasets.  When the training set is larger we subsample first so
        # the competitor is still meaningful (uniform-coverage intent preserved).
        _KS_MAX = 5_000
        t_fit = 0.0
        t1 = time.perf_counter()
        if len(X_train) > _KS_MAX:
            rng_ks = np.random.RandomState(random_state)
            sub_idx = rng_ks.choice(len(X_train), _KS_MAX, replace=False)
            X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
            n_sub_target = max(2, int(_KS_MAX * ratio))
            ks_idx = _kennard_stone(X_sub, n_sub_target)
            idx = sub_idx[ks_idx]
        else:
            idx = _kennard_stone(X_train, n_target)
        X_red, y_red = X_train[idx], y_train[idx]
        t_op = time.perf_counter() - t1

    elif method_name == 'QR-Pivot':
        t_fit = 0.0
        t1 = time.perf_counter()
        idx = _qr_pivot(X_train, n_target)
        X_red, y_red = X_train[idx], y_train[idx]
        t_op = time.perf_counter() - t1

    elif method_name == 'Stratified':
        t_fit = 0.0
        t1 = time.perf_counter()
        idx = _stratified_reduce(X_train, y_train, n_target, random_state)
        X_red, y_red = X_train[idx], y_train[idx]
        t_op = time.perf_counter() - t1

    elif method_name == 'Random':
        t_fit = 0.0
        t1 = time.perf_counter()
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_train), n_target, replace=False)
        X_red, y_red = X_train[idx], y_train[idx]
        t_op = time.perf_counter() - t1

    else:
        raise ValueError(f"Unknown reduction method: {method_name!r}")

    metrics = evaluate_reduction(
        X_train, y_train, X_red, y_red, X_test, y_test,
        is_emergence=is_emergence,
    )
    metrics['train_time_seconds'] = round(t_fit, 4)
    metrics['operation_time_seconds'] = round(t_op, 4)

    if trtr_baseline is not None:
        metrics['ml_utility_trtr'] = trtr_baseline
        metrics['ml_delta'] = round(
            metrics.get('ml_utility_retention', 0.0) - trtr_baseline, 4
        )

    return {
        'task': 'reduce',
        'dataset': dataset_name,
        'method': method_name,
        'params': {'ratio': ratio},
        'metrics': metrics,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def _run_one_expansion(dataset_name, X_train, y_train, X_test, y_test,
                        method_name, expansion_ratio, random_state,
                        trtr_baseline=None):
    """Execute one expansion configuration and return a result dict."""
    n_synthetic = int(len(X_train) * expansion_ratio)
    t0 = time.perf_counter()
    y_synth = None   # will be set per-method or via proxy below

    if method_name in (
        'HVRT-size', 'HVRT-var', 'FastHVRT-size', 'FastHVRT-var'
    ):
        ModelCls = HVRT if method_name.startswith('HVRT-') else FastHVRT
        var_weighted = method_name.endswith('-var')
        model = ModelCls(random_state=random_state)
        # Stack y as an extra column so the joint (X, y) distribution is
        # modelled.  Synthetic y then emerges from the same partition rather
        # than being predicted by a separate proxy model.
        y_col = y_train.reshape(-1, 1).astype(float)
        XY_train = np.column_stack([X_train, y_col])
        model.fit(XY_train)
        t_fit = time.perf_counter() - t0
        t1 = time.perf_counter()
        XY_synth = model.expand(n=n_synthetic, variance_weighted=var_weighted)
        t_op = time.perf_counter() - t1
        X_synth = XY_synth[:, :-1]
        y_synth_raw = XY_synth[:, -1]
        # For classification snap continuous KDE output to nearest observed class
        is_cls = len(np.unique(y_train)) <= 20
        if is_cls:
            classes = np.unique(y_train)
            y_synth = classes[
                np.argmin(np.abs(y_synth_raw[:, None] - classes[None, :]), axis=1)
            ]
        else:
            y_synth = y_synth_raw

    elif method_name == 'GMM':
        t_fit = 0.0
        t1 = time.perf_counter()
        X_synth = _gmm_expand(X_train, n_synthetic, random_state)
        t_op = time.perf_counter() - t1

    elif method_name == 'Gaussian-Copula':
        t_fit = 0.0
        t1 = time.perf_counter()
        X_synth = _gaussian_copula_expand(X_train, n_synthetic, random_state)
        t_op = time.perf_counter() - t1

    elif method_name == 'Bootstrap-Noise':
        t_fit = 0.0
        t1 = time.perf_counter()
        X_synth = _bootstrap_noise_expand(X_train, n_synthetic, random_state=random_state)
        t_op = time.perf_counter() - t1

    elif method_name == 'SMOTE':
        t_fit = 0.0
        t1 = time.perf_counter()
        X_synth = _smote_expand(X_train, n_synthetic, random_state=random_state)
        t_op = time.perf_counter() - t1

    elif method_name == 'CTGAN':
        t_fit = 0.0
        t1 = time.perf_counter()
        X_synth = _ctgan_expand(X_train, n_synthetic, random_state=random_state)
        t_op = time.perf_counter() - t1

    elif method_name == 'TVAE':
        t_fit = 0.0
        t1 = time.perf_counter()
        X_synth = _tvae_expand(X_train, n_synthetic, random_state=random_state)
        t_op = time.perf_counter() - t1

    else:
        raise ValueError(f"Unknown expansion method: {method_name!r}")

    # Synthetic y: HVRT/FastHVRT generate y jointly (set above).
    # All other methods generate X only — use a proxy model trained on real
    # (X_train, y_train) to assign synthetic labels.
    if y_synth is None:
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        is_cls = len(np.unique(y_train)) <= 20
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if is_cls:
                proxy = GradientBoostingClassifier(n_estimators=50, random_state=42)
            else:
                proxy = GradientBoostingRegressor(n_estimators=50, random_state=42)
            proxy.fit(X_train, y_train)
        y_synth = proxy.predict(X_synth)

    metrics = evaluate_expansion(
        X_train, y_train, X_synth, y_synth, X_test, y_test,
    )
    metrics['train_time_seconds'] = round(t_fit, 4)
    metrics['operation_time_seconds'] = round(t_op, 4)

    if trtr_baseline is not None:
        metrics['ml_utility_trtr'] = trtr_baseline
        metrics['ml_delta'] = round(
            metrics.get('ml_utility_tstr', 0.0) - trtr_baseline, 4
        )

    return {
        'task': 'expand',
        'dataset': dataset_name,
        'method': method_name,
        'params': {'expansion_ratio': expansion_ratio},
        'metrics': metrics,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

REDUCTION_METHODS = [
    'HVRT-size', 'HVRT-var', 'FastHVRT-size', 'FastHVRT-var',
    'Kennard-Stone', 'QR-Pivot', 'Stratified', 'Random',
]

EXPANSION_METHODS = [
    'HVRT-size', 'HVRT-var', 'FastHVRT-size', 'FastHVRT-var',
    'GMM', 'Gaussian-Copula', 'Bootstrap-Noise', 'SMOTE',
]

# Deep-learning methods require optional dependencies (pip install ctgan).
# Included when deep_learning=True is passed to run_expansion_benchmark().
DEEP_LEARNING_EXPANSION_METHODS = ['CTGAN', 'TVAE']

REDUCTION_DATASETS = ['adult', 'fraud', 'housing', 'multimodal']
EXPANSION_DATASETS = ['adult', 'fraud', 'housing', 'multimodal']
EMERGENCE_DATASETS = ['emergence_divergence', 'emergence_bifurcation']

DEFAULT_REDUCTION_RATIOS = [0.5, 0.3, 0.2, 0.1]
DEFAULT_EXPANSION_RATIOS = [1.0, 2.0, 5.0]


def run_reduction_benchmark(
    datasets='all',
    methods='all',
    ratios=None,
    random_state=42,
    save_path=None,
    verbose=True,
    max_n=None,
):
    """
    Run the full reduction benchmark.

    Parameters
    ----------
    datasets : 'all' or list of str
    methods  : 'all' or list of str
    ratios   : list of float or None (defaults to [0.5, 0.3, 0.2, 0.1])
    random_state : int
    save_path : str or None
    verbose : bool

    Returns
    -------
    list of result dicts
    """
    if datasets == 'all':
        ds_list = REDUCTION_DATASETS + EMERGENCE_DATASETS
    else:
        ds_list = list(datasets)

    if methods == 'all':
        method_list = REDUCTION_METHODS
    else:
        method_list = list(methods)

    if ratios is None:
        ratios = DEFAULT_REDUCTION_RATIOS

    results = []

    for ds_name in ds_list:
        is_emergence = ds_name in EMERGENCE_DATASETS
        gen_fn = BENCHMARK_DATASETS[ds_name]
        X, y, _ = gen_fn(random_state=random_state)
        if max_n is not None:
            X, y = X[:max_n], y[:max_n]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        trtr_baseline = round(ml_utility_tstr(X_train, y_train, X_test, y_test), 4)

        for ratio in ratios:
            for method in method_list:
                if verbose:
                    print(f"  [{ds_name}] ratio={ratio:.0%}  method={method} ...", end=' ', flush=True)
                try:
                    r = _run_one_reduction(
                        ds_name, X_train, y_train, X_test, y_test,
                        method, ratio, random_state, is_emergence,
                        trtr_baseline=trtr_baseline,
                    )
                    results.append(r)
                    if verbose:
                        mf = r['metrics'].get('marginal_fidelity', float('nan'))
                        ml = r['metrics'].get('ml_utility_retention', float('nan'))
                        print(f"mf={mf:.3f}  ml={ml:.3f}")
                except Exception as exc:
                    if verbose:
                        print(f"ERROR: {exc}")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to {save_path}")

    return results


def run_expansion_benchmark(
    datasets='all',
    methods='all',
    expansion_ratios=None,
    random_state=42,
    save_path=None,
    verbose=True,
    max_n=500,
    deep_learning=False,
    include_references=True,
):
    """
    Run the full expansion benchmark.

    Parameters
    ----------
    datasets          : 'all' or list of str
    methods           : 'all' or list of str
    expansion_ratios  : list of float or None  (defaults to [1.0, 2.0, 5.0])
    random_state      : int
    save_path         : str or None
    verbose           : bool
    max_n             : int or None
        Cap on training-set size.  Defaults to 500 so that generation quality
        is evaluated in the data-scarce regime (the most interesting test case).
        Pass None to use the full generated dataset (~20 k samples).
    deep_learning     : bool
        Include CTGAN and TVAE.  Requires: pip install ctgan.
    include_references: bool
        Append published-only reference rows (TabDDPM†, MOSTLY AI†) when
        methods='all'.  These rows carry fixed metric values from published
        papers and are marked with published_only=True.

    Returns
    -------
    list of result dicts
    """
    if datasets == 'all':
        ds_list = EXPANSION_DATASETS
    else:
        ds_list = list(datasets)

    if methods == 'all':
        method_list = list(EXPANSION_METHODS)
        if deep_learning:
            method_list = method_list + DEEP_LEARNING_EXPANSION_METHODS
    else:
        method_list = list(methods)

    if expansion_ratios is None:
        expansion_ratios = DEFAULT_EXPANSION_RATIOS

    results = []

    for ds_name in ds_list:
        gen_fn = BENCHMARK_DATASETS[ds_name]
        X, y, _ = gen_fn(random_state=random_state)
        if max_n is not None:
            X, y = X[:max_n], y[:max_n]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        trtr_baseline = round(ml_utility_tstr(X_train, y_train, X_test, y_test), 4)

        for exp_ratio in expansion_ratios:
            for method in method_list:
                if verbose:
                    print(f"  [{ds_name}] exp={exp_ratio:.0f}x  method={method} ...", end=' ', flush=True)
                try:
                    r = _run_one_expansion(
                        ds_name, X_train, y_train, X_test, y_test,
                        method, exp_ratio, random_state,
                        trtr_baseline=trtr_baseline,
                    )
                    results.append(r)
                    if verbose:
                        da = r['metrics'].get('discriminator_accuracy', float('nan'))
                        mf = r['metrics'].get('marginal_fidelity', float('nan'))
                        print(f"discriminator={da:.3f}  mf={mf:.3f}")
                except Exception as exc:
                    if verbose:
                        print(f"ERROR: {exc}")

    # Append published-only reference rows (once, not per-dataset/ratio)
    if include_references and methods == 'all':
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        for ref in PUBLISHED_EXPANSION_REFERENCES:
            results.append({
                'task':           'expand',
                'dataset':        'published_benchmark',
                'method':         ref['method'],
                'params':         {'expansion_ratio': None},
                'metrics':        ref['metrics'],
                'note':           ref.get('note', ''),
                'published_only': True,
                'timestamp':      ts,
            })
        if verbose:
            print(f"\n  Appended {len(PUBLISHED_EXPANSION_REFERENCES)} published reference rows "
                  f"({', '.join(r['method'] for r in PUBLISHED_EXPANSION_REFERENCES)})")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to {save_path}")

    return results


def run_full_benchmark(
    datasets='all',
    tasks=None,
    random_state=42,
    save_path=None,
    verbose=True,
    max_n=None,
    max_n_expand=500,
    deep_learning=False,
    include_references=True,
):
    """
    Run both reduction and expansion benchmarks and combine results.

    Parameters
    ----------
    datasets          : 'all' or list of str
    tasks             : list of str, default ['reduce', 'expand']
    random_state      : int
    save_path         : str or None  path for combined JSON output
    verbose           : bool
    max_n             : int or None
        Override for BOTH tasks (useful for smoke tests).  When set,
        takes precedence over max_n_expand.
    max_n_expand      : int or None
        Cap on training-set size for expansion benchmarks only.
        Defaults to 500 (data-scarce regime).  Pass None for full dataset.
    deep_learning     : bool
        Include CTGAN and TVAE in expansion benchmarks.
    include_references: bool
        Append published-only reference rows (TabDDPM†, MOSTLY AI†).

    Returns
    -------
    list of result dicts (combined)
    """
    if tasks is None:
        tasks = ['reduce', 'expand']

    # max_n overrides both when set (backward-compat / smoke-test shortcut)
    effective_max_n_expand = max_n if max_n is not None else max_n_expand

    all_results = []

    if 'reduce' in tasks:
        if verbose:
            print("=" * 60)
            print("REDUCTION BENCHMARKS")
            print("=" * 60)
        all_results.extend(
            run_reduction_benchmark(
                datasets=datasets, random_state=random_state,
                verbose=verbose, max_n=max_n,
            )
        )

    if 'expand' in tasks:
        if verbose:
            print("=" * 60)
            print(f"EXPANSION BENCHMARKS  (max_n={effective_max_n_expand})")
            print("=" * 60)
        all_results.extend(
            run_expansion_benchmark(
                datasets=datasets, random_state=random_state,
                verbose=verbose, max_n=effective_max_n_expand,
                deep_learning=deep_learning,
                include_references=include_references,
            )
        )

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        if verbose:
            print(f"\nFull results saved to {save_path}")

    return all_results


def load_results(path):
    """Load benchmark results from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
