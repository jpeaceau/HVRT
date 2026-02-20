"""
Evaluation metrics for HVRT benchmark suite.

All metrics follow a consistent convention:
  - higher is better (fidelity scores) or well-defined target (discriminator)
  - operate on numpy arrays in original feature scale unless stated otherwise

Metric glossary
---------------
marginal_fidelity      1 - mean normalised 1-D Wasserstein distance per feature
correlation_fidelity   Frobenius similarity of correlation matrices
tail_preservation      Geometric mean of percentile-range ratios (5th / 95th)
emergence_score        Conditional structure preservation via tree agreement
ml_utility_tstr        Train-on-Synthetic, Test-on-Real  (F1 or R²)
discriminator_accuracy Logistic regression accuracy (target ≈ 50%)
privacy_dcr            Distance-to-Closest-Record ratio (target > 1.0)
novelty_min            Min distance from any synthetic sample to any original
"""

import warnings
import time
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_std(arr):
    s = arr.std()
    return s if s > 1e-10 else 1.0


# ---------------------------------------------------------------------------
# Distribution fidelity
# ---------------------------------------------------------------------------

def marginal_fidelity(X_original, X_transformed):
    """
    1 - mean normalised 1-D Wasserstein distance across features.

    Each feature's Wasserstein distance is normalised by the original
    feature's standard deviation so that all features contribute equally
    regardless of scale.

    Returns
    -------
    float in [0, 1]  (1.0 = perfect marginal match)
    """
    X_orig = np.asarray(X_original, dtype=float)
    X_trans = np.asarray(X_transformed, dtype=float)
    n_feat = X_orig.shape[1]

    distances = []
    for j in range(n_feat):
        std = _safe_std(X_orig[:, j])
        d = wasserstein_distance(X_orig[:, j], X_trans[:, j]) / std
        distances.append(d)

    return float(1.0 - np.mean(distances))


def correlation_fidelity(X_original, X_transformed):
    """
    Similarity of correlation matrices: 1 - (Frobenius norm of difference /
    Frobenius norm of original).

    Returns
    -------
    float  (1.0 = identical correlation structure)
    """
    C_orig = np.corrcoef(np.asarray(X_original, dtype=float).T)
    C_trans = np.corrcoef(np.asarray(X_transformed, dtype=float).T)

    diff_norm = np.linalg.norm(C_orig - C_trans, 'fro')
    orig_norm = np.linalg.norm(C_orig, 'fro')
    if orig_norm < 1e-10:
        return 1.0

    return float(1.0 - diff_norm / orig_norm)


def tail_preservation(X_original, X_transformed, pct=5):
    """
    Geometric mean of per-feature percentile-range ratios.

    ratio_j = IQR_transformed_j / IQR_original_j
    where IQR is computed at (pct, 100-pct) percentiles.

    A value near 1.0 indicates tails are well preserved.  Values > 1 imply
    heavier synthesised tails; < 1 implies tail shrinkage.

    Returns
    -------
    float  (1.0 = perfect tail match)
    """
    X_orig = np.asarray(X_original, dtype=float)
    X_trans = np.asarray(X_transformed, dtype=float)
    n_feat = X_orig.shape[1]

    log_ratios = []
    for j in range(n_feat):
        lo, hi = np.percentile(X_orig[:, j], [pct, 100 - pct])
        iqr_orig = hi - lo
        if iqr_orig < 1e-10:
            continue
        lo2, hi2 = np.percentile(X_trans[:, j], [pct, 100 - pct])
        iqr_trans = hi2 - lo2
        if iqr_trans < 1e-10:
            log_ratios.append(-10.0)
        else:
            log_ratios.append(np.log(iqr_trans / iqr_orig))

    if not log_ratios:
        return 1.0

    return float(np.exp(np.mean(log_ratios)))


# ---------------------------------------------------------------------------
# Emergence score (CES)
# ---------------------------------------------------------------------------

def emergence_score(X_original, y_original, X_transformed, y_transformed):
    """
    Conditional Emergence Score: how well the transformed set reproduces
    the conditional structure (tree-based) of the original data.

    Implementation
    --------------
    1. Fit a shallow DecisionTreeRegressor/Classifier on (X_original, y_original).
    2. Apply it to X_transformed to get predicted partition IDs.
    3. Fit the same structure on (X_transformed, y_transformed).
    4. Score = proportion of X_original samples whose leaf assignment
       agrees between the two trees.  Higher = better emergence preservation.

    Returns
    -------
    float in [0, 1]
    """
    X_orig = np.asarray(X_original, dtype=float)
    y_orig = np.asarray(y_original).ravel()
    X_trans = np.asarray(X_transformed, dtype=float)
    y_trans = np.asarray(y_transformed).ravel()

    is_classification = len(np.unique(y_orig)) <= 20

    if is_classification:
        tree_orig = DecisionTreeClassifier(max_depth=5, random_state=42)
        tree_trans = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        tree_orig = DecisionTreeRegressor(max_depth=5, random_state=42)
        tree_trans = DecisionTreeRegressor(max_depth=5, random_state=42)

    tree_orig.fit(X_orig, y_orig)
    tree_trans.fit(X_trans, y_trans)

    leaves_orig = tree_orig.apply(X_orig)
    leaves_trans = tree_trans.apply(X_orig)

    agreement = np.mean(leaves_orig == leaves_trans)
    return float(agreement)


# ---------------------------------------------------------------------------
# ML utility (TSTR)
# ---------------------------------------------------------------------------

def ml_utility_tstr(X_train, y_train, X_test, y_test, task='auto'):
    """
    Train-on-Synthetic (or reduced), Test-on-Real.

    Fits a GradientBoosting model on (X_train, y_train) and evaluates on
    (X_test, y_test).  Returns F1 (weighted) for classification or R² for
    regression.

    Parameters
    ----------
    X_train, y_train : transformed / synthetic training data
    X_test, y_test   : held-out real test data
    task : 'auto', 'classification', or 'regression'

    Returns
    -------
    float  score (higher is better)
    """
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import f1_score, r2_score

    y_tr = np.asarray(y_train).ravel()
    y_te = np.asarray(y_test).ravel()

    if task == 'auto':
        task = 'classification' if len(np.unique(y_tr)) <= 20 else 'regression'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if task == 'classification':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_tr)
            preds = model.predict(X_test)
            return float(f1_score(y_te, preds, average='weighted', zero_division=0))
        else:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_tr)
            preds = model.predict(X_test)
            return float(r2_score(y_te, preds))


# ---------------------------------------------------------------------------
# AUC-based ML utility
# ---------------------------------------------------------------------------

def ml_utility_auc(X_train, y_train, X_test, y_test, task='auto'):
    """
    Train GradientBoosting on X_train; return ROC-AUC on X_test for
    classification, or R² for regression.

    Parameters
    ----------
    X_train, y_train : training data (real, reduced, or synthetic)
    X_test, y_test   : held-out real test data
    task : 'auto', 'classification', or 'regression'

    Returns
    -------
    float  (higher is better)
    """
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import roc_auc_score, r2_score

    y_tr = np.asarray(y_train).ravel()
    y_te = np.asarray(y_test).ravel()

    if task == 'auto':
        task = 'classification' if len(np.unique(y_tr)) <= 20 else 'regression'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if task == 'classification':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_tr)
            proba = model.predict_proba(X_test)
            classes = model.classes_
            if len(classes) == 2:
                return float(roc_auc_score(y_te, proba[:, 1]))
            return float(roc_auc_score(y_te, proba,
                                       multi_class='ovr', average='weighted'))
        else:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_tr)
            return float(r2_score(y_te, model.predict(X_test)))


def error_explanation_rate(X_train_real, y_train_real,
                            X_train_synth, y_train_synth,
                            X_test, y_test):
    """
    Measure how well a model trained on synthetic data covers the hard cases
    that a model trained on real data fails on.

    Procedure
    ---------
    1. Train TRTR model on real training data.
    2. Identify the error set E — test samples the TRTR model mis-classifies.
    3. Train TSTR model on synthetic training data.
    4. On E, compute:
       - coverage_rate : fraction of E that TSTR predicts correctly
       - error_set_auc : ROC-AUC of TSTR's probability score on E
         (how confidently TSTR ranks the true class on the hardest cases)

    A high coverage_rate / error_set_auc means the synthetic data captures
    enough structure to handle the cases the real model struggles with.

    Parameters
    ----------
    X_train_real, y_train_real   : real training data
    X_train_synth, y_train_synth : synthetic training data
    X_test, y_test               : held-out real test data

    Returns
    -------
    dict with keys:
        n_errors        – size of error set E
        coverage_rate   – accuracy of TSTR on E  (float, higher = better)
        error_set_auc   – ROC-AUC of TSTR proba on E  (float, higher = better)
        full_trtr_auc   – ROC-AUC of TRTR on full test set (reference)
        full_tstr_auc   – ROC-AUC of TSTR on full test set
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    y_te = np.asarray(y_test).ravel()
    if len(np.unique(y_te)) > 20:
        # Regression: not applicable
        return {k: float('nan') for k in
                ('n_errors', 'coverage_rate', 'error_set_auc',
                 'full_trtr_auc', 'full_tstr_auc')}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        real_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        real_model.fit(X_train_real, np.asarray(y_train_real).ravel())
        y_pred_real  = real_model.predict(X_test)
        proba_real   = real_model.predict_proba(X_test)

        synth_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        synth_model.fit(X_train_synth, np.asarray(y_train_synth).ravel())
        y_pred_synth = synth_model.predict(X_test)
        proba_synth  = synth_model.predict_proba(X_test)

    # Error set: indices where TRTR is wrong
    error_mask = (y_pred_real != y_te)
    n_errors   = int(error_mask.sum())

    def _auc(y_true, proba, model):
        classes = model.classes_
        if len(classes) == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba,
                                   multi_class='ovr', average='weighted'))

    full_trtr_auc = _auc(y_te, proba_real,  real_model)
    full_tstr_auc = _auc(y_te, proba_synth, synth_model)

    if n_errors == 0:
        return {
            'n_errors':       0,
            'coverage_rate':  1.0,
            'error_set_auc':  1.0,
            'full_trtr_auc':  full_trtr_auc,
            'full_tstr_auc':  full_tstr_auc,
        }

    coverage_rate = float((y_pred_synth[error_mask] == y_te[error_mask]).mean())

    y_te_err    = y_te[error_mask]
    proba_err   = proba_synth[error_mask]

    if len(np.unique(y_te_err)) < 2:
        error_set_auc = float('nan')
    else:
        error_set_auc = _auc(y_te_err, proba_err, synth_model)

    return {
        'n_errors':       n_errors,
        'coverage_rate':  coverage_rate,
        'error_set_auc':  error_set_auc,
        'full_trtr_auc':  full_trtr_auc,
        'full_tstr_auc':  full_tstr_auc,
    }


# ---------------------------------------------------------------------------
# Expansion-specific metrics
# ---------------------------------------------------------------------------

def discriminator_accuracy(X_real, X_synthetic):
    """
    Logistic-regression discriminator accuracy (real vs synthetic).

    A balanced binary classification task: 50% real, 50% synthetic.
    Score near 50% → synthetic is indistinguishable from real.
    Score near 100% → synthetic is easily detected.

    Returns
    -------
    float  accuracy in [0, 1]  (target ≈ 0.50)
    """
    X_r = np.asarray(X_real, dtype=float)
    X_s = np.asarray(X_synthetic, dtype=float)

    # Balance classes
    n = min(len(X_r), len(X_s))
    rng = np.random.RandomState(0)
    idx_r = rng.choice(len(X_r), n, replace=False)
    idx_s = rng.choice(len(X_s), n, replace=False)

    X_all = np.vstack([X_r[idx_r], X_s[idx_s]])
    y_all = np.concatenate([np.ones(n), np.zeros(n)])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = LogisticRegression(max_iter=500, random_state=42)
        scores = cross_val_score(clf, X_scaled, y_all, cv=3, scoring='accuracy')

    return float(scores.mean())


def privacy_dcr(X_real, X_synthetic, chunk_size=500):
    """
    Distance-to-Closest-Record ratio.

    DCR = median(min_dist(synthetic_i to real)) /
          median(min_dist(real_i to real minus {i}))

    A ratio > 1.0 means synthetic samples are further from the real data
    than real samples are from each other — indicating privacy protection.

    Returns
    -------
    float  (target > 1.0)
    """
    X_r = np.asarray(X_real, dtype=float)
    X_s = np.asarray(X_synthetic, dtype=float)

    # Synthetic → real distances
    synth_min_dists = []
    for start in range(0, len(X_s), chunk_size):
        chunk = X_s[start:start + chunk_size]
        d = cdist(chunk, X_r, 'euclidean').min(axis=1)
        synth_min_dists.extend(d.tolist())
    synth_median = np.median(synth_min_dists)

    # Real → real (leave-one-out) distances
    real_min_dists = []
    for start in range(0, len(X_r), chunk_size):
        chunk = X_r[start:start + chunk_size]
        D = cdist(chunk, X_r, 'euclidean')
        # Exclude self (diagonal = 0)
        np.fill_diagonal(
            D[:, start:start + len(chunk)],
            np.inf,
        )
        real_min_dists.extend(D.min(axis=1).tolist())
    real_median = np.median(real_min_dists)

    if real_median < 1e-10:
        return float('inf')

    return float(synth_median / real_median)


def novelty_min(X_real, X_synthetic, chunk_size=1000):
    """
    Minimum Euclidean distance from any synthetic sample to any real sample.

    Returns
    -------
    float  (higher = more novel; 0 = at least one copy of a real sample)
    """
    X_r = np.asarray(X_real, dtype=float)
    X_s = np.asarray(X_synthetic, dtype=float)

    global_min = np.inf
    for start in range(0, len(X_s), chunk_size):
        chunk = X_s[start:start + chunk_size]
        d = cdist(chunk, X_r, 'euclidean').min(axis=1).min()
        global_min = min(global_min, d)

    return float(global_min)


# ---------------------------------------------------------------------------
# Composite evaluators
# ---------------------------------------------------------------------------

def evaluate_reduction(X_original, y_original, X_reduced, y_reduced, X_test, y_test,
                        is_emergence=False):
    """
    Compute all reduction metrics for a single method/dataset/ratio combination.

    Parameters
    ----------
    X_original : ndarray  full training set
    y_original : ndarray
    X_reduced  : ndarray  reduced training set
    y_reduced  : ndarray
    X_test     : ndarray  held-out test set
    y_test     : ndarray
    is_emergence : bool   compute emergence_score if True

    Returns
    -------
    dict of metric name → float
    """
    m = {}
    m['marginal_fidelity'] = marginal_fidelity(X_original, X_reduced)
    m['correlation_fidelity'] = correlation_fidelity(X_original, X_reduced)
    m['tail_preservation'] = tail_preservation(X_original, X_reduced)
    m['ml_utility_retention'] = ml_utility_tstr(X_reduced, y_reduced, X_test, y_test)
    if is_emergence:
        m['emergence_score'] = emergence_score(X_original, y_original, X_reduced, y_reduced)
    return m


def evaluate_expansion(X_real, y_real, X_synthetic, y_synthetic, X_test, y_test):
    """
    Compute all expansion metrics for a single method/dataset/ratio combination.

    Returns
    -------
    dict of metric name → float
    """
    m = {}
    m['marginal_fidelity'] = marginal_fidelity(X_real, X_synthetic)
    m['correlation_fidelity'] = correlation_fidelity(X_real, X_synthetic)
    m['tail_preservation'] = tail_preservation(X_real, X_synthetic)
    m['discriminator_accuracy'] = discriminator_accuracy(X_real, X_synthetic)
    m['privacy_dcr'] = privacy_dcr(X_real, X_synthetic)
    m['novelty_min'] = novelty_min(X_real, X_synthetic)
    m['ml_utility_tstr'] = ml_utility_tstr(X_synthetic, y_synthetic, X_test, y_test)
    return m
