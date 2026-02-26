"""
HVRTOptimizer: Optuna-backed hyperparameter optimisation for HVRT.

Searches over ``n_partitions``, ``min_samples_leaf``, ``y_weight``,
bandwidth / generation_strategy (unified as ``kernel``), and
``variance_weighted`` using TPE sampling.  The objective is mean TSTR Δ
(train-on-synthetic minus train-on-real) across CV folds.

Install the required extra::

    pip install hvrt[optimizer]

Usage::

    from hvrt import HVRTOptimizer

    opt = HVRTOptimizer(n_trials=50, n_jobs=4, cv=3, random_state=42).fit(X, y)
    print(f'Best TSTR Δ: {opt.best_score_:+.4f}')
    print(f'Best params: {opt.best_params_}')

    X_synth = opt.expand(n=50000)        # uses tuned kernel + params
    X_aug   = opt.augment(n=len(X) * 5)  # originals + synthetic
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Optuna import guard — called at the top of fit(), not at module import time
# ---------------------------------------------------------------------------

def _require_optuna():
    """Raise a clear ImportError if optuna is not installed."""
    try:
        import optuna  # noqa: F401
    except ImportError:
        raise ImportError(
            "HVRTOptimizer requires optuna. "
            "Install with: pip install hvrt[optimizer]"
        ) from None


# ---------------------------------------------------------------------------
# Parameter decoding
# ---------------------------------------------------------------------------

def _decode_params(params: dict):
    """
    Convert raw trial params (with string sentinels) to typed kwargs.

    Parameters
    ----------
    params : dict
        Must contain keys: 'n_partitions', 'min_samples_leaf', 'y_weight',
        'kernel', 'variance_weighted'.

    Returns
    -------
    constructor_kw : dict
        Kwargs for HVRT(**constructor_kw).
    expand_kw : dict
        Kwargs for model.expand(**expand_kw).
    """
    # n_partitions: 'auto' → None, else int
    n_parts_raw = params['n_partitions']
    n_partitions = None if n_parts_raw == 'auto' else int(n_parts_raw)

    # min_samples_leaf: 'auto' → None, else int
    min_leaf_raw = params['min_samples_leaf']
    min_samples_leaf = None if min_leaf_raw == 'auto' else int(min_leaf_raw)

    # y_weight: string → float
    y_weight = float(params['y_weight'])

    # kernel: unified param → bandwidth + generation_strategy
    # 'auto'         → bandwidth='auto', no generation_strategy
    # '0.10'/'0.30'  → bandwidth=float, no generation_strategy
    # 'epanechnikov' → bandwidth='auto', generation_strategy='epanechnikov'
    kernel = params['kernel']
    if kernel == 'epanechnikov':
        bandwidth = 'auto'
        generation_strategy = 'epanechnikov'
    elif kernel == 'auto':
        bandwidth = 'auto'
        generation_strategy = None
    else:
        bandwidth = float(kernel)
        generation_strategy = None

    # variance_weighted: bool (Optuna stores as-is) or fallback string handling
    vw_raw = params['variance_weighted']
    if isinstance(vw_raw, str):
        variance_weighted = vw_raw.lower() in ('true', '1', 'yes')
    else:
        variance_weighted = bool(vw_raw)

    constructor_kw = {
        'n_partitions': n_partitions,
        'min_samples_leaf': min_samples_leaf,
        'y_weight': y_weight,
        'bandwidth': bandwidth,
    }
    expand_kw: dict = {'variance_weighted': variance_weighted}
    if generation_strategy is not None:
        expand_kw['generation_strategy'] = generation_strategy

    return constructor_kw, expand_kw


# ---------------------------------------------------------------------------
# Downstream scoring
# ---------------------------------------------------------------------------

def _downstream_score(
    task: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
) -> float:
    """
    Fit a GBM on (X_tr, y_tr) and evaluate on (X_te, y_te).

    Returns R² for regression, ROC-AUC for classification.
    """
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import r2_score, roc_auc_score

    if task == 'regression':
        model = GradientBoostingRegressor(random_state=seed, n_estimators=100)
        model.fit(X_tr, y_tr)
        return float(r2_score(y_te, model.predict(X_te)))

    # Classification
    model = GradientBoostingClassifier(random_state=seed, n_estimators=100)
    model.fit(X_tr, y_tr)
    n_classes = len(np.unique(y_te))
    if n_classes == 2:
        proba = model.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te, proba))
    else:
        proba = model.predict_proba(X_te)
        return float(
            roc_auc_score(y_te, proba, multi_class='ovr', average='weighted')
        )


# ---------------------------------------------------------------------------
# HVRTOptimizer
# ---------------------------------------------------------------------------

class HVRTOptimizer:
    """
    Optuna-backed hyperparameter optimiser for HVRT.

    Searches over ``n_partitions``, ``min_samples_leaf``, ``y_weight``,
    kernel / bandwidth, and ``variance_weighted`` to maximise a per-fold
    score across CV folds.  By default the score is TSTR Δ (train-on-synthetic
    minus train-on-real).  Pass a custom ``objective`` callable to use any
    combination of metrics — e.g. a weighted mix of privacy and ML utility.

    After fitting, exposes ``expand()`` and ``augment()`` that delegate to
    the best fitted model with the best expansion parameters.

    Parameters
    ----------
    n_trials : int, default=30
        Number of Optuna trials.  Trial 0 is always the HVRT defaults (warm
        start), so HPO can only match or beat defaults when given enough budget.
        Use ≥ 50 trials in production; 20 may be insufficient to distinguish
        signal from noise on heterogeneous datasets.
    n_jobs : int, default=1
        Parallel trials (-1 = all available cores).
    cv : int, default=3
        Number of cross-validation folds.
    expansion_ratio : float, default=5.0
        Synthetic-to-real ratio used during the objective evaluation.
    task : str, default='auto'
        One of ``'auto'``, ``'regression'``, ``'classification'``.
        ``'auto'`` infers from the number of unique y values
        (≤ 20 unique → classification, else regression).
    objective : callable or None, default=None
        Custom per-fold scoring function.  When ``None``, the default TSTR Δ
        objective is used.

        The callable receives a single ``dict`` with the following keys and
        must return a **float to maximise**:

        .. code-block:: python

            {
                'tstr':       float | None,   # downstream score trained on synthetic
                'trtr':       float | None,   # downstream score trained on real
                'tstr_delta': float | None,   # tstr - trtr
                'X_synth':    ndarray,        # synthetic features (n_synth, n_features)
                'X_real':     ndarray,        # real fold train features
                'y_synth':    ndarray | None, # synthetic targets (None if y not provided)
                'y_real':     ndarray | None, # real fold train targets
                'fold':       int,            # fold index (0..cv-1)
                'n_synth':    int,            # number of synthetic samples generated
            }

        ``tstr``, ``trtr``, ``tstr_delta``, ``y_synth``, and ``y_real`` are
        ``None`` when ``y`` is not passed to ``fit()``.  The callable is
        called once per fold; its return values are averaged across folds to
        produce the trial score.

        Return higher values for better configurations.  To penalise something
        (e.g. privacy risk), subtract it from the return value::

            def privacy_utility(m):
                dcr = compute_dcr(m['X_synth'], m['X_real'])
                # DCR in [0, 2]: higher = more private; cap at 1 to avoid
                # rewarding samples that are more spread than real data
                privacy = min(dcr, 1.0)
                return 0.6 * m['tstr_delta'] + 0.4 * privacy

            opt = HVRTOptimizer(objective=privacy_utility, n_trials=50)
            opt.fit(X, y)

    timeout : float or None, default=None
        Wall-clock timeout for the Optuna study (seconds).
    random_state : int or None, default=None
    verbose : int, default=0
        0 = silent, 1 = Optuna trial progress.

    Attributes
    ----------
    best_score_ : float
        Best mean per-fold score across CV folds (TSTR Δ by default, or the
        value returned by ``objective`` when a custom callable is provided).
    best_params_ : dict
        Best constructor kwargs (n_partitions, min_samples_leaf,
        y_weight, bandwidth).
    best_expand_params_ : dict
        Best expand kwargs (variance_weighted, and optionally
        generation_strategy).
    best_model_ : HVRT
        Fitted on full X (plus y as an appended column when y is provided)
        using best_params_.
    study_ : optuna.Study
        Full Optuna study object (for visualisation and diagnostics).

    Examples
    --------
    >>> from hvrt import HVRTOptimizer
    >>> opt = HVRTOptimizer(n_trials=50, n_jobs=4, cv=3, random_state=42)
    >>> opt = opt.fit(X, y)
    >>> X_synth = opt.expand(n=50000)
    >>> X_aug   = opt.augment(n=len(X) * 5)
    """

    def __init__(
        self,
        n_trials: int = 30,
        n_jobs: int = 1,
        cv: int = 3,
        expansion_ratio: float = 5.0,
        task: str = 'auto',
        objective=None,
        timeout: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.cv = cv
        self.expansion_ratio = expansion_ratio
        self.task = task
        self.objective = objective
        self.timeout = timeout
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        Run the Optuna study and refit the best model on the full dataset.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,) or None
            Providing y enables supervised task detection and TSTR evaluation.
            Without y, all trials score 0.0 and optimisation is a no-op.

        Returns
        -------
        self
        """
        _require_optuna()
        import optuna
        from sklearn.model_selection import KFold
        from .model import HVRT

        X = np.asarray(X, dtype=np.float64)
        y_provided = y is not None
        if y_provided:
            y = np.asarray(y, dtype=np.float64).ravel()

        self._n_original_features_ = X.shape[1]
        self._y_included_ = y_provided

        # Task detection
        if y_provided:
            n_unique = len(np.unique(y))
            if self.task == 'auto':
                task_ = 'classification' if n_unique <= 20 else 'regression'
            else:
                task_ = self.task
        else:
            task_ = 'regression'  # unused when y is absent

        seed_base = self.random_state if self.random_state is not None else 0
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=seed_base)
        splits = list(kf.split(X))

        # ------------------------------------------------------------------
        # TRTR pre-computation — constant across all trials
        # ------------------------------------------------------------------
        trtr_per_fold = []
        for fold_i, (tr_idx, te_idx) in enumerate(splits):
            if not y_provided:
                trtr_per_fold.append(0.0)
                continue
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            try:
                trtr = _downstream_score(
                    task_, X_tr, y_tr, X_te, y_te, seed_base + fold_i
                )
            except Exception:
                trtr = 0.0
            trtr_per_fold.append(trtr)

        # ------------------------------------------------------------------
        # Objective closure
        # ------------------------------------------------------------------
        def objective(trial) -> float:
            n_parts_str = trial.suggest_categorical(
                'n_partitions', ['auto', '20', '30', '50', '75', '100']
            )
            min_leaf_str = trial.suggest_categorical(
                'min_samples_leaf', ['auto', '5', '10', '15']
            )
            y_weight_str = trial.suggest_categorical(
                'y_weight', ['0.0', '0.1', '0.3', '0.5']
            )
            kernel = trial.suggest_categorical(
                'kernel', ['auto', '0.10', '0.30', 'epanechnikov']
            )
            variance_weighted = trial.suggest_categorical(
                'variance_weighted', [False, True]
            )

            raw_params = {
                'n_partitions': n_parts_str,
                'min_samples_leaf': min_leaf_str,
                'y_weight': y_weight_str,
                'kernel': kernel,
                'variance_weighted': variance_weighted,
            }
            constructor_kw, expand_kw = _decode_params(raw_params)
            seed = seed_base + trial.number

            fold_scores = []
            for fold_i, (tr_idx, te_idx) in enumerate(splits):
                X_tr, X_te = X[tr_idx], X[te_idx]
                try:
                    if y_provided:
                        y_tr, y_te = y[tr_idx], y[te_idx]
                        XY_tr = np.column_stack([X_tr, y_tr.reshape(-1, 1)])
                        model = HVRT(random_state=seed, **constructor_kw).fit(XY_tr)
                        n_synth = max(4, int(len(X_tr) * self.expansion_ratio))
                        XY_s = model.expand(n=n_synth, **expand_kw)
                        X_s = XY_s[:, :-1]
                        y_s_raw = XY_s[:, -1]
                        if task_ == 'classification':
                            # Snap continuous KDE-generated y back to the
                            # nearest observed class label so the downstream
                            # classifier receives valid discrete targets.
                            classes = np.unique(y_tr)
                            y_s = classes[
                                np.argmin(
                                    np.abs(y_s_raw[:, None] - classes[None, :]),
                                    axis=1,
                                )
                            ]
                        else:
                            y_s = y_s_raw
                        tstr = _downstream_score(
                            task_, X_s, y_s, X_te, y[te_idx], seed + fold_i
                        )
                        if self.objective is not None:
                            metrics = {
                                'tstr':       tstr,
                                'trtr':       trtr_per_fold[fold_i],
                                'tstr_delta': tstr - trtr_per_fold[fold_i],
                                'X_synth':    X_s,
                                'X_real':     X_tr,
                                'y_synth':    y_s,
                                'y_real':     y_tr,
                                'fold':       fold_i,
                                'n_synth':    n_synth,
                            }
                            fold_scores.append(float(self.objective(metrics)))
                        else:
                            fold_scores.append(tstr - trtr_per_fold[fold_i])
                    else:
                        model = HVRT(random_state=seed, **constructor_kw).fit(X_tr)
                        n_synth = max(4, int(len(X_tr) * self.expansion_ratio))
                        X_s = model.expand(n=n_synth, **expand_kw)
                        if self.objective is not None:
                            metrics = {
                                'tstr':       None,
                                'trtr':       None,
                                'tstr_delta': None,
                                'X_synth':    X_s,
                                'X_real':     X_tr,
                                'y_synth':    None,
                                'y_real':     None,
                                'fold':       fold_i,
                                'n_synth':    n_synth,
                            }
                            fold_scores.append(float(self.objective(metrics)))
                        else:
                            fold_scores.append(0.0)
                except Exception:
                    return float('-inf')

            return float(np.mean(fold_scores))

        # ------------------------------------------------------------------
        # Optuna study
        # ------------------------------------------------------------------
        if self.verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        # Always evaluate the HVRT defaults as trial 0 (warm start).
        # This guarantees HPO can only improve on — or tie with — the baseline.
        # Costs one slot from n_trials; the sampler then explores from there.
        study.enqueue_trial({
            'n_partitions':     'auto',
            'min_samples_leaf': 'auto',
            'y_weight':         '0.0',
            'kernel':           'auto',
            'variance_weighted': False,
        })

        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
        )

        # ------------------------------------------------------------------
        # Refit best model on full dataset
        # ------------------------------------------------------------------
        ck, ek = _decode_params(study.best_params)
        if y_provided:
            XY_full = np.column_stack([X, y.reshape(-1, 1)])
        else:
            XY_full = X
        self.best_model_ = HVRT(random_state=self.random_state, **ck).fit(XY_full)
        self.best_params_ = ck
        self.best_expand_params_ = ek
        self.best_score_ = study.best_value
        self.study_ = study

        return self

    # ------------------------------------------------------------------
    # expand / augment — delegate to best_model_ with best expand params
    # ------------------------------------------------------------------

    def expand(self, n: int, **kwargs):
        """
        Generate synthetic samples using the tuned model and best expand params.

        Parameters
        ----------
        n : int
            Number of synthetic samples to generate.
        **kwargs
            Override any best expand parameters (e.g., ``variance_weighted``).

        Returns
        -------
        X_synthetic : ndarray (n, n_features)
            Synthetic samples in the original feature space.
            When y was provided at fit(), the appended y column is stripped
            so the output always has the same number of columns as the
            training X.
        """
        self._check_fitted('expand')
        merged = {**self.best_expand_params_, **kwargs}
        XY_synth = self.best_model_.expand(n=n, **merged)
        return XY_synth[:, :self._n_original_features_]

    def augment(self, n: int, **kwargs):
        """
        Return original X concatenated with synthetic samples.

        Parameters
        ----------
        n : int
            Total output size (original samples + synthetic).
            Must be strictly greater than the number of training samples.
        **kwargs
            Override any best expand parameters.

        Returns
        -------
        X_augmented : ndarray (n, n_features)
            First ``len(X_train)`` rows are the original training samples;
            the remainder are synthetic.
        """
        self._check_fitted('augment')
        X_orig = self.best_model_.X_[:, :self._n_original_features_]
        n_orig = len(X_orig)
        if n <= n_orig:
            raise ValueError(
                f"augment() requires n ({n}) > original sample count ({n_orig})."
            )
        n_synthetic = n - n_orig
        X_synth = self.expand(n=n_synthetic, **kwargs)
        return np.vstack([X_orig, X_synth])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self, method: str):
        if not hasattr(self, 'best_model_'):
            raise ValueError(
                f"HVRTOptimizer must be fitted before calling {method}(). "
                "Call fit(X, y) first."
            )
