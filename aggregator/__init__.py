"""
Aggregator package for Army of Safeguards.

This package provides a unified interface for running multiple safeguards
and aggregating their results using KNN-based classification.
"""

from .aggregator import (
    evaluate_text,
    load_knn_reference_data,
    run_all_safeguards,
    aggregate_results,
)

__all__ = [
    "evaluate_text",
    "load_knn_reference_data",
    "run_all_safeguards",
    "aggregate_results",
]

