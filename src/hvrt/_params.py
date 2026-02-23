"""
Operation parameter containers for HVRT.

ReduceParams, ExpandParams, and AugmentParams hold the per-call parameters for
reduce(), expand(), and augment(). They serve two purposes:

1. Construction-time configuration for sklearn pipelines::

       pipe = Pipeline([('hvrt', HVRT(reduce_params=ReduceParams(ratio=0.3)))])
       X_red = pipe.fit_transform(X, y)

2. Reusable parameter sets for direct use::

       params = ReduceParams(ratio=0.3, method='fps')
       X_red  = model.reduce(**vars(params))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union


@dataclass
class ReduceParams:
    """
    Parameters for reduce().

    Parameters
    ----------
    n : int, optional
        Absolute target count.
    ratio : float, optional
        Proportion to keep (e.g. 0.3 = keep 30 %).
    method : str or callable, default 'fps'
        Within-partition selection strategy.
    variance_weighted : bool, default True
        Allocate proportionally to mean |z-score| per partition.
    return_indices : bool, default False
        Also return global indices into the source X.
    n_partitions : int, optional
        Override tree leaf count for this operation only.
    """

    n: Optional[int] = None
    ratio: Optional[float] = None
    method: Union[
        Literal['fps', 'centroid_fps', 'medoid_fps', 'variance_ordered', 'stratified'],
        Callable,
    ] = 'fps'
    variance_weighted: bool = True
    return_indices: bool = False
    n_partitions: Optional[int] = None

    def __post_init__(self):
        if self.n is not None and self.ratio is not None:
            raise ValueError("ReduceParams: provide either n or ratio, not both.")


@dataclass
class ExpandParams:
    """
    Parameters for expand().

    Parameters
    ----------
    n : int
        Number of synthetic samples to generate.
    variance_weighted : bool, default False
        True = oversample high-variance (tail) partitions.
    bandwidth : float or str, optional
        KDE bandwidth scalar or selector.  ``None`` uses the instance default.
        Accepts ``'auto'``, ``'scott'``, ``'silverman'``, or a float scalar.
    adaptive_bandwidth : bool, default False
        Scale each partition's KDE bandwidth with the local expansion ratio.
    generation_strategy : str or callable, optional
        Per-partition sampling strategy.
    return_novelty_stats : bool, default False
        Also return distance statistics relative to source data.
    n_partitions : int, optional
        Override tree leaf count for this operation only.
    """

    n: int = field(default=None)
    variance_weighted: bool = False
    bandwidth: Union[float, str, None] = None
    adaptive_bandwidth: bool = False
    generation_strategy: Union[
        Literal['multivariate_kde', 'univariate_kde_copula', 'bootstrap_noise', 'epanechnikov'],
        Callable,
        None,
    ] = None
    return_novelty_stats: bool = False
    n_partitions: Optional[int] = None

    def __post_init__(self):
        if self.n is None:
            raise ValueError("ExpandParams requires n.")


@dataclass
class AugmentParams:
    """
    Parameters for augment().

    Parameters
    ----------
    n : int
        Total output size. Must be strictly greater than len(X).
    variance_weighted : bool, default False
    n_partitions : int, optional
        Override tree granularity for the expansion step.
    """

    n: int = field(default=None)
    variance_weighted: bool = False
    n_partitions: Optional[int] = None

    def __post_init__(self):
        if self.n is None:
            raise ValueError("AugmentParams requires n.")
