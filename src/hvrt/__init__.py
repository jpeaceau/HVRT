"""
HVRT: Hierarchical Variance-Retaining Transformer
==================================================

Variance-aware sample transformation for tabular data — reduce, expand,
or augment while preserving distributional structure.

Primary API (v2)
----------------
    from hvrt import HVRT, FastHVRT

    model = HVRT().fit(X)                        # pairwise interactions
    X_reduced  = model.reduce(ratio=0.3)
    X_synthetic = model.expand(n=50000, min_novelty=0.2)
    X_augmented = model.augment(n=30000)

    model = FastHVRT().fit(X)                    # z-score sum (faster)
    X_synthetic = model.expand(n=50000)

Legacy API (v1, kept for backward compatibility)
------------------------------------------------
    from hvrt import HVRTSampleReducer, AdaptiveHVRTReducer
"""

from .model import HVRT, FastHVRT

# Legacy v1 classes — kept for backward compatibility
from .sample_reduction import HVRTSampleReducer
from .adaptive_reducer import AdaptiveHVRTReducer
from .selection_strategies import (
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

__version__ = '2.0.0'

__all__ = [
    # v2 primary API
    'HVRT',
    'FastHVRT',
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
