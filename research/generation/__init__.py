"""
research.generation — in-partition generation method implementations.

Provides drop-in replacements for the per-partition sampling step used
inside HVRT's expand() operation.  Each method receives a partition's
normalised feature matrix (X_part) and returns n_samples synthetic rows
in the same space.

Available methods
-----------------
MultivariateKDE            Full-matrix Gaussian KDE (current HVRT default)
UnivariateKDERankCoupled   Per-feature KDE + Gaussian copula rank coupling
UnivariateKDEIndependent   Per-feature KDE, features sampled independently
PartitionGMM               Gaussian Mixture Model per partition
KNNInterpolation           SMOTE-style k-NN convex interpolation
PartitionBootstrap         Resample + Gaussian noise baseline
"""

from .methods import (
    MultivariateKDE,
    UnivariateKDERankCoupled,
    UnivariateKDEIndependent,
    PartitionGMM,
    KNNInterpolation,
    PartitionBootstrap,
    ALL_METHODS,
)

__all__ = [
    'MultivariateKDE',
    'UnivariateKDERankCoupled',
    'UnivariateKDEIndependent',
    'PartitionGMM',
    'KNNInterpolation',
    'PartitionBootstrap',
    'ALL_METHODS',
]
