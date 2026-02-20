"""
hvrt.pipeline — convenience imports for sklearn pipeline use.

    from hvrt.pipeline import HVRT, FastHVRT, ReduceParams, ExpandParams, AugmentParams

    pipe = Pipeline([('hvrt', HVRT(reduce_params=ReduceParams(ratio=0.3)))])
    pipe = Pipeline([('hvrt', FastHVRT(expand_params=ExpandParams(n=50000)))])

This package re-exports the same classes available at the top level.
The separate import path signals to readers that these objects are being
used in a pipeline context.
"""

from ..model import HVRT, FastHVRT
from .._params import ReduceParams, ExpandParams, AugmentParams

__all__ = [
    'HVRT',
    'FastHVRT',
    'ReduceParams',
    'ExpandParams',
    'AugmentParams',
]
