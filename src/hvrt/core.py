"""
Backward-compatibility shim.

HVRT and FastHVRT are defined in hvrt.model.hvrt and hvrt.model.fast_hvrt.
This module re-exports them so any code that previously imported from
hvrt.core continues to work unchanged.
"""

from .model.hvrt import HVRT
from .model.fast_hvrt import FastHVRT

__all__ = ['HVRT', 'FastHVRT']
