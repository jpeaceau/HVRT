"""
HVRT: Hierarchical Variance-Retaining Transformer
==================================================

Variance-aware sample transformation for tabular data — reduce, expand,
or augment while preserving distributional structure.

Primary API (v2)
----------------
    from hvrt import HVRT, FastHVRT

    model = HVRT().fit(X)                        # pairwise interactions
    X_reduced   = model.reduce(ratio=0.3)
    X_synthetic = model.expand(n=50000)
    X_augmented = model.augment(n=30000)

    model = FastHVRT().fit(X)                    # z-score sum (faster)
    X_synthetic = model.expand(n=50000)

C++ backend (v2.8+)
-------------------
When the compiled extension (_hvrt_cpp) is available, performance-critical
operations (pairwise target, centroid FPS, medoid FPS) automatically use the
C++ implementation. Install via scikit-build-core or build from source. The
dispatch order is: C++ → Numba (hvrt[fast]) → NumPy.

Deprecated (removed in v3.0)
-----------------------------
- Constructor params ``reduce_params``, ``expand_params``, ``augment_params``
- ``fit_transform()``
- ``get_params()`` / ``set_params()``
"""

from ._warnings import HVRTWarning, HVRTFeatureWarning, HVRTDeprecationWarning
from ._params import ReduceParams, ExpandParams, AugmentParams
from .model import HVRT, FastHVRT
from .optimizer import HVRTOptimizer

from .reduction_strategies import (
    # New stateful protocol
    StatefulSelectionStrategy,
    # Context dataclass
    SelectionContext,
    # Strategy classes
    StratifiedStrategy,
    VarianceOrderedStrategy,
    CentroidFPSStrategy,
    MedoidFPSStrategy,
    # Module-level singletons (same names as old functions — drop-in compat)
    centroid_fps,
    medoid_fps,
    variance_ordered,
    stratified,
    # Registry
    BUILTIN_STRATEGIES,
    get_strategy,
)
from .generation_strategies import (
    # New stateful protocol
    StatefulGenerationStrategy,
    # Context dataclasses
    PartitionContext,
    EpanechnikovContext,
    BootstrapNoiseContext,
    MultivariateKDEContext,
    UnivariateCopulaContext,
    # Strategy classes (accessible as singleton instances via module-level names)
    EpanechnikovStrategy,
    BootstrapNoiseStrategy,
    MultivariateKDEStrategy,
    UnivariateCopulaStrategy,
    # Module-level singletons (same names as old functions — drop-in compat)
    multivariate_kde,
    univariate_kde_copula,
    bootstrap_noise,
    epanechnikov,
    # Registry
    BUILTIN_GENERATION_STRATEGIES,
    get_generation_strategy,
)

__version__ = '2.8.0'

__all__ = [
    # v2 primary API
    'HVRT',
    'FastHVRT',
    'HVRTOptimizer',
    # Operation params
    'ReduceParams',
    'ExpandParams',
    'AugmentParams',
    # Warning classes
    'HVRTWarning',
    'HVRTFeatureWarning',
    'HVRTDeprecationWarning',
    # Selection strategies — new stateful protocol
    'StatefulSelectionStrategy',
    'SelectionContext',
    'StratifiedStrategy',
    'VarianceOrderedStrategy',
    'CentroidFPSStrategy',
    'MedoidFPSStrategy',
    # Module-level singletons
    'centroid_fps',
    'medoid_fps',
    'variance_ordered',
    'stratified',
    'BUILTIN_STRATEGIES',
    'get_strategy',
    # Generation strategies — new stateful protocol
    'StatefulGenerationStrategy',
    'PartitionContext',
    'EpanechnikovContext',
    'BootstrapNoiseContext',
    'MultivariateKDEContext',
    'UnivariateCopulaContext',
    'EpanechnikovStrategy',
    'BootstrapNoiseStrategy',
    'MultivariateKDEStrategy',
    'UnivariateCopulaStrategy',
    # Module-level singletons
    'multivariate_kde',
    'univariate_kde_copula',
    'bootstrap_noise',
    'epanechnikov',
    'BUILTIN_GENERATION_STRATEGIES',
    'get_generation_strategy',
]
