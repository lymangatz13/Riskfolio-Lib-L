"""
Trade Bot Module
================
Enhanced data ingestion and portfolio optimization using Riskfolio-Lib.
"""

from .block1_data_ingestion import (
    run_data_ingestion,
    quick_freshness_check,
    verify_data_freshness,
    validate_with_riskfolio,
    ASSET_CLASSES,
    BENCHMARKS,
    FRESHNESS_THRESHOLDS,
)

__all__ = [
    'run_data_ingestion',
    'quick_freshness_check',
    'verify_data_freshness',
    'validate_with_riskfolio',
    'ASSET_CLASSES',
    'BENCHMARKS',
    'FRESHNESS_THRESHOLDS',
]
