"""C++ backend import — safe, never raises on ImportError."""
try:
    from _hvrt_cpp import (
        compute_pairwise_target as _cpp_pairwise_target,
        centroid_fps            as _cpp_centroid_fps,
        medoid_fps              as _cpp_medoid_fps,
    )
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    _cpp_pairwise_target = None
    _cpp_centroid_fps    = None
    _cpp_medoid_fps      = None
