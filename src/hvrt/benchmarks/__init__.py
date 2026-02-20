"""
HVRT Benchmark Suite

Provides reproducible dataset generators, evaluation metrics, benchmark
runners, and visualization utilities.

Quick start::

    from hvrt.benchmarks import run_full_benchmark, load_results, plot_unified_summary

    results = run_full_benchmark(
        datasets='all',
        tasks=['reduce', 'expand'],
        save_path='./benchmark_results.json',
    )
    fig = plot_unified_summary(results)
"""

from .datasets import (
    make_adult_like,
    make_fraud_like,
    make_housing_like,
    make_multimodal,
    make_emergence_divergence,
    make_emergence_bifurcation,
    BENCHMARK_DATASETS,
)
from .metrics import (
    marginal_fidelity,
    correlation_fidelity,
    tail_preservation,
    discriminator_accuracy,
    privacy_dcr,
    novelty_min,
    ml_utility_tstr,
    emergence_score,
    evaluate_reduction,
    evaluate_expansion,
)
from .runners import run_reduction_benchmark, run_expansion_benchmark, run_full_benchmark, load_results
from .visualization import plot_comparison, plot_unified_summary, print_results_table

__all__ = [
    # datasets
    'make_adult_like', 'make_fraud_like', 'make_housing_like',
    'make_multimodal', 'make_emergence_divergence', 'make_emergence_bifurcation',
    'BENCHMARK_DATASETS',
    # metrics
    'marginal_fidelity', 'correlation_fidelity', 'tail_preservation',
    'discriminator_accuracy', 'privacy_dcr', 'novelty_min',
    'ml_utility_tstr', 'emergence_score',
    'evaluate_reduction', 'evaluate_expansion',
    # runners
    'run_reduction_benchmark', 'run_expansion_benchmark',
    'run_full_benchmark', 'load_results',
    # visualization
    'plot_comparison', 'plot_unified_summary', 'print_results_table',
]
