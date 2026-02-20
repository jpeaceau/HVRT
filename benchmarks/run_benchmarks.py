"""
HVRT Benchmark CLI

Run the full benchmark suite from the command line.

Usage
-----
    python benchmarks/run_benchmarks.py                              # full suite
    python benchmarks/run_benchmarks.py --tasks reduce               # reduction only
    python benchmarks/run_benchmarks.py --tasks expand               # expansion only
    python benchmarks/run_benchmarks.py --datasets adult fraud       # specific datasets
    python benchmarks/run_benchmarks.py --output results/out.json
    python benchmarks/run_benchmarks.py --print-table reduce         # print stored results
    python benchmarks/run_benchmarks.py --deep-learning              # include CTGAN + TVAE
    python benchmarks/run_benchmarks.py --max-n-expand 200           # tiny training sets
    python benchmarks/run_benchmarks.py --no-references              # skip published rows

Dataset sizes
-------------
    Reduction : full dataset (~20 k samples) — tests large-scale compression
    Expansion : 500 samples by default (--max-n-expand) — tests generation
                from small data, the regime where synthetic data matters most
"""

import argparse
import os
import sys

# Ensure the package is importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hvrt.benchmarks import (
    run_full_benchmark,
    run_reduction_benchmark,
    run_expansion_benchmark,
    load_results,
    print_results_table,
    plot_unified_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description='HVRT v2 Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--tasks', nargs='+', choices=['reduce', 'expand'], default=['reduce', 'expand'],
        help='Which benchmark tasks to run (default: both)',
    )
    parser.add_argument(
        '--datasets', nargs='+', default=['all'],
        help='Datasets to benchmark (default: all)',
    )
    parser.add_argument(
        '--ratios', nargs='+', type=float, default=None,
        help='Reduction ratios to test (default: 0.5 0.3 0.2 0.1)',
    )
    parser.add_argument(
        '--expansion-ratios', nargs='+', type=float, default=None,
        help='Expansion ratios to test (default: 1.0 2.0 5.0)',
    )
    parser.add_argument(
        '--max-n-expand', type=int, default=500, metavar='N',
        help=(
            'Cap training-set size for expansion benchmarks (default: 500). '
            'Set to 0 to use the full dataset.'
        ),
    )
    parser.add_argument(
        '--deep-learning', action='store_true',
        help='Include CTGAN and TVAE (requires: pip install ctgan)',
    )
    parser.add_argument(
        '--no-references', action='store_true',
        help='Exclude published-only reference rows (TabDDPM\u2020, MOSTLY AI\u2020)',
    )
    parser.add_argument(
        '--output', '-o', type=str, default='benchmarks/results/benchmark_results.json',
        help='Path to save JSON results (default: benchmarks/results/benchmark_results.json)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress per-run progress output',
    )
    parser.add_argument(
        '--print-table', metavar='TASK', type=str, default=None,
        help='Print ASCII results table for TASK from existing --output file',
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate and save a unified summary plot after running',
    )

    args = parser.parse_args()

    datasets = 'all' if args.datasets == ['all'] else args.datasets
    max_n_expand = args.max_n_expand if args.max_n_expand > 0 else None

    # Print table from existing results
    if args.print_table:
        if not os.path.exists(args.output):
            print(f"Results file not found: {args.output}", file=sys.stderr)
            sys.exit(1)
        results = load_results(args.output)
        print_results_table(results, task=args.print_table)
        return

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    verbose = not args.quiet

    if verbose:
        print(f"Expansion training-set cap : {max_n_expand or 'full dataset'}")
        print(f"Deep-learning methods      : {'CTGAN, TVAE' if args.deep_learning else 'disabled (--deep-learning to enable)'}")
        print(f"Published references       : {'excluded (--no-references)' if args.no_references else 'included (TabDDPM\u2020, MOSTLY AI\u2020)'}")
        print()

    # Run benchmarks
    results = run_full_benchmark(
        datasets=datasets,
        tasks=args.tasks,
        random_state=args.seed,
        save_path=args.output,
        verbose=verbose,
        max_n_expand=max_n_expand,
        deep_learning=args.deep_learning,
        include_references=not args.no_references,
    )

    # Print summary tables
    if 'reduce' in args.tasks:
        print('\n')
        print_results_table(results, task='reduce')
    if 'expand' in args.tasks:
        print('\n')
        print_results_table(results, task='expand')

    # Optional plot
    if args.plot:
        try:
            fig = plot_unified_summary(results)
            plot_path = args.output.replace('.json', '_summary.png')
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f'\nSummary plot saved to {plot_path}')
        except ImportError:
            print('\nmatplotlib not available — skipping plot. pip install matplotlib')


if __name__ == '__main__':
    main()
