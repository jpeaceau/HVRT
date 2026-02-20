"""
HVRT warning class hierarchy.

All HVRT-specific warnings inherit from ``HVRTWarning`` so callers can
suppress the entire family with a single filter::

    import warnings
    from hvrt import HVRTWarning
    warnings.filterwarnings('ignore', category=HVRTWarning)

Individual sub-classes can also be targeted::

    from hvrt import HVRTFeatureWarning
    warnings.filterwarnings('ignore', category=HVRTFeatureWarning)
"""


class HVRTWarning(UserWarning):
    """Base class for all HVRT warnings."""


class HVRTFeatureWarning(HVRTWarning):
    """
    Warning emitted when external X has a different feature layout from the
    training data (extra columns are ignored; column order is adjusted).
    """


class HVRTDeprecationWarning(HVRTWarning, DeprecationWarning):
    """
    Warning emitted for deprecated HVRT API parameters.

    Inherits from both ``HVRTWarning`` (filterable as a group) and
    ``DeprecationWarning`` (suppressed by default in non-test code, shown
    by pytest's default warning filters).
    """
