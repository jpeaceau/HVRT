"""
_HARTBase: shared logic for HART and FastHART.

MAD-based analog to _HVRTBase.  Three differences from the variance-based
variants:
  1. Input normalisation: median + MAD scaling (_MADScaler) instead of
     mean + std (StandardScaler).
  2. Synthetic y-component: MAD-normalised instead of std-normalised.
  3. Tree criterion: 'absolute_error' (minimises MAE over leaves) instead of
     'squared_error'.

The x-component (_compute_x_component) is identical to HVRT/FastHVRT but
operates on MAD-normalised X_z, so the structural signal shifts from joint
variance to joint absolute deviation.
"""

from __future__ import annotations

import numpy as np

from ._base import _HVRTBase
from ._preprocessing import fit_preprocess_data, _MADScaler

_MAD_CONSISTENCY = 1.4826


class _HARTBase(_HVRTBase):
    """
    Base class for HART and FastHART (MAD-based analogs to HVRT/FastHVRT).

    Inherits the full public API from _HVRTBase.  Overrides three hooks to
    replace variance-based assumptions with absolute-deviation equivalents.
    """

    _TREE_CRITERION = 'absolute_error'

    def _preprocess_data(self, X, feature_types):
        """Use _MADScaler (median/MAD) instead of StandardScaler."""
        return fit_preprocess_data(X, feature_types, scaler_factory=_MADScaler)

    def _normalize_y(self, y):
        """
        MAD-based y-extremeness signal.

        1. Median-centre and MAD-scale y → y_norm (robust z-scores).
        2. Compute absolute deviation from the median of y_norm → y_extremeness.
        3. Median-centre and MAD-scale y_extremeness → final signal.

        The double application of median/MAD mirrors the double application of
        mean/std in the base class, keeping the pipeline structurally symmetric.
        """
        y_med = np.median(y)
        y_mad = np.median(np.abs(y - y_med))
        y_norm = (y - y_med) / (_MAD_CONSISTENCY * y_mad + 1e-10)

        y_extremeness = np.abs(y_norm - np.median(y_norm))
        ext_med = np.median(y_extremeness)
        ext_mad = np.median(np.abs(y_extremeness - ext_med))
        return (y_extremeness - ext_med) / (_MAD_CONSISTENCY * ext_mad + 1e-10)
