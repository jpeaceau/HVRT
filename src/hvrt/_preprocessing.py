"""
Preprocessing utilities for HVRT.

All functions are pure (no class state). _HVRTBase delegates to them so that
the preprocessing logic is independently testable and _base.py stays focused
on the public API.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def fit_preprocess_data(X_raw, feature_types=None):
    """
    Coerce, encode, and z-score normalise input data.

    Parameters
    ----------
    X_raw : array-like or DataFrame (n_samples, n_features)
    feature_types : list of 'continuous' or 'categorical' or None
        All continuous if None.

    Returns
    -------
    X : ndarray (n_samples, n_features)
        Original-scale numpy array (columns in original order).
    X_z : ndarray (n_samples, n_z_features)
        Z-scored; layout: [continuous columns | categorical columns].
    continuous_mask : ndarray (n_features,) bool
    categorical_mask : ndarray (n_features,) bool
    scaler : StandardScaler or None
    cat_scaler : StandardScaler or None
    label_encoders : dict {col_idx: LabelEncoder}
    feature_names_in : list of str or None
    """
    if hasattr(X_raw, 'columns'):
        feature_names_in = list(X_raw.columns)
        X = np.asarray(X_raw, dtype=np.float64)
    else:
        feature_names_in = None
        X = np.asarray(X_raw, dtype=np.float64)

    n_features = X.shape[1]

    if feature_types is None:
        feature_types = ['continuous'] * n_features

    continuous_mask = np.array([ft == 'continuous' for ft in feature_types])
    categorical_mask = ~continuous_mask

    parts = []
    scaler = None
    cat_scaler = None
    label_encoders = {}

    if continuous_mask.any():
        scaler = StandardScaler()
        parts.append(scaler.fit_transform(X[:, continuous_mask]))

    if categorical_mask.any():
        X_enc, label_encoders = encode_categorical(
            X[:, categorical_mask], fit=True
        )
        cat_scaler = StandardScaler()
        parts.append(cat_scaler.fit_transform(X_enc))

    X_z = np.hstack(parts) if len(parts) > 1 else parts[0]

    return (
        X, X_z,
        continuous_mask, categorical_mask,
        scaler, cat_scaler, label_encoders,
        feature_names_in,
    )


def encode_categorical(X_cat, label_encoders=None, fit=True):
    """
    Integer-encode categorical columns.

    Parameters
    ----------
    X_cat : ndarray (n_samples, n_cat_cols)
    label_encoders : dict {col_idx: LabelEncoder} or None
        Required when fit=False.
    fit : bool, default True

    Returns
    -------
    X_encoded : ndarray (n_samples, n_cat_cols) float64
    label_encoders : dict {col_idx: LabelEncoder}
    """
    if label_encoders is None:
        label_encoders = {}

    X_encoded = np.zeros(X_cat.shape, dtype=np.float64)
    for col in range(X_cat.shape[1]):
        if fit:
            le = LabelEncoder()
            X_encoded[:, col] = le.fit_transform(X_cat[:, col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            X_encoded[:, col] = le.transform(X_cat[:, col].astype(str))

    return X_encoded, label_encoders


def to_z(X, continuous_mask, categorical_mask, scaler, cat_scaler, label_encoders):
    """
    Transform X to z-score space using fitted scalers.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    continuous_mask, categorical_mask : ndarray bool
    scaler : fitted StandardScaler or None
    cat_scaler : fitted StandardScaler or None
    label_encoders : dict

    Returns
    -------
    X_z : ndarray (n_samples, n_z_features)
    """
    parts = []
    if continuous_mask.any():
        parts.append(scaler.transform(X[:, continuous_mask]))
    if categorical_mask.any():
        X_enc, _ = encode_categorical(
            X[:, categorical_mask], label_encoders=label_encoders, fit=False
        )
        parts.append(cat_scaler.transform(X_enc))
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def from_z(X_z, continuous_mask, categorical_mask, scaler, cat_scaler, label_encoders):
    """
    Inverse-transform from z-score space to original feature scale.

    Z-space column layout: [continuous... | categorical...]
    Output column layout:  original order (per continuous_mask positions).

    Returns
    -------
    X_out : ndarray
        dtype is object when categorical features are present.
    """
    n_samples = len(X_z)
    n_orig = len(continuous_mask)
    has_cat = categorical_mask.any()

    X_out = np.empty((n_samples, n_orig), dtype=object if has_cat else float)

    z_offset = 0

    if continuous_mask.any():
        n_cont = int(continuous_mask.sum())
        X_cont = scaler.inverse_transform(X_z[:, z_offset:z_offset + n_cont])
        X_out[:, continuous_mask] = X_cont
        z_offset += n_cont

    if has_cat:
        n_cat = int(categorical_mask.sum())
        X_cat_raw = cat_scaler.inverse_transform(
            X_z[:, z_offset:z_offset + n_cat]
        )
        X_cat_int = np.round(X_cat_raw).astype(int)
        cat_col_positions = np.where(categorical_mask)[0]
        for local_idx in range(n_cat):
            le = label_encoders.get(local_idx)
            if le is not None:
                codes = np.clip(
                    X_cat_int[:, local_idx], 0, len(le.classes_) - 1
                )
                X_out[:, cat_col_positions[local_idx]] = le.inverse_transform(codes)
            else:
                X_out[:, cat_col_positions[local_idx]] = X_cat_int[:, local_idx]

    return X_out


def build_cat_partition_freqs(X_cat, partition_ids, unique_partitions):
    """
    Build per-partition empirical frequency distributions for categorical columns.

    Parameters
    ----------
    X_cat : ndarray (n_samples, n_cat_cols)
        Original categorical column values.
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray

    Returns
    -------
    dict {int pid -> list of (unique_values, probs) per categorical column}
    """
    freqs = {}
    for pid in unique_partitions:
        mask = partition_ids == pid
        X_part = X_cat[mask]
        col_freqs = []
        for j in range(X_part.shape[1]):
            unique_vals, counts = np.unique(X_part[:, j], return_counts=True)
            probs = counts.astype(float) / counts.sum()
            col_freqs.append((unique_vals, probs))
        freqs[int(pid)] = col_freqs
    return freqs
