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

Pipeline API (v2)
-----------------
    from hvrt import HVRT, ReduceParams, ExpandParams
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([('hvrt', HVRT(reduce_params=ReduceParams(ratio=0.3)))])
    X_red = pipe.fit_transform(X, y)

    pipe = Pipeline([('hvrt', FastHVRT(expand_params=ExpandParams(n=50000)))])
    X_synth = pipe.fit_transform(X)

Legacy API (v1, kept for backward compatibility)
------------------------------------------------
    from hvrt import HVRTSampleReducer, AdaptiveHVRTReducer
"""

from ._warnings import HVRTWarning, HVRTFeatureWarning, HVRTDeprecationWarning
from ._params import ReduceParams, ExpandParams, AugmentParams
from .model import HVRT, FastHVRT

# Legacy v1 classes — kept for backward compatibility
from .legacy.sample_reduction import HVRTSampleReducer
from .legacy.adaptive_reducer import AdaptiveHVRTReducer
from .reduction_strategies import (
    SelectionStrategy,
    centroid_fps,
    medoid_fps,
    variance_ordered,
    stratified,
    BUILTIN_STRATEGIES,
    get_strategy,
)
from .generation_strategies import (
    GenerationStrategy,
    multivariate_kde,
    univariate_kde_copula,
    bootstrap_noise,
    BUILTIN_GENERATION_STRATEGIES,
    get_generation_strategy,
)

__version__ = '2.1.0'

__all__ = [
    # v2 primary API
    'HVRT',
    'FastHVRT',
    # Operation params
    'ReduceParams',
    'ExpandParams',
    'AugmentParams',
    # Warning classes
    'HVRTWarning',
    'HVRTFeatureWarning',
    'HVRTDeprecationWarning',
    # Selection strategies
    'SelectionStrategy',
    'centroid_fps',
    'medoid_fps',
    'variance_ordered',
    'stratified',
    'BUILTIN_STRATEGIES',
    'get_strategy',
    # Generation strategies
    'GenerationStrategy',
    'multivariate_kde',
    'univariate_kde_copula',
    'bootstrap_noise',
    'BUILTIN_GENERATION_STRATEGIES',
    'get_generation_strategy',
    # v1 legacy API
    'HVRTSampleReducer',
    'AdaptiveHVRTReducer',
]
