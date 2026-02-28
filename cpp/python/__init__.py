"""
hvrt_cpp — C++ backend for the HVRT library.

Tries to import the compiled _hvrt_cpp extension; falls back to the
pure-Python ``hvrt`` package (must be installed separately) on ImportError.
"""

from __future__ import annotations

try:
    # When installed as part of the hvrt package, the extension is hvrt._hvrt_cpp
    from hvrt._hvrt_cpp import (  # noqa: F401
        HVRT,
        HVRTConfig,
        PartitionInfo,
        ParamRecommendation,
        PartitionTree,
    )
    _BACKEND = "cpp"
except ImportError:
    try:
        # Development / standalone build: _hvrt_cpp on sys.path directly
        from _hvrt_cpp import (  # noqa: F401
            HVRT,
            HVRTConfig,
            PartitionInfo,
            ParamRecommendation,
            PartitionTree,
        )
        _BACKEND = "cpp"
    except ImportError:
        try:
            from hvrt._hvrt_py import HVRT  # noqa: F401  # pure-Python fallback
            _BACKEND = "python"
        except ImportError as exc:
            raise ImportError(
                "Neither the _hvrt_cpp C++ extension nor the pure-Python "
                "fallback could be imported. Build with CMake or reinstall: "
                "pip install hvrt"
            ) from exc

__all__ = ["HVRT", "HVRTConfig", "PartitionInfo", "ParamRecommendation", "_BACKEND"]
