"""
Example usage of the enhanced Block 1 Data Ingestion module.

This demonstrates:
1. Quick freshness check without full data load
2. Full data ingestion with Riskfolio-Lib validation
3. Using the DATA_PACKAGE for portfolio optimization
"""

# Quick freshness check (minimal data fetch)
print("="*60)
print("QUICK FRESHNESS CHECK")
print("="*60)

from block1_data_ingestion import quick_freshness_check

freshness = quick_freshness_check()
print(f"\nStatus: {freshness['status']}")
print(f"Dataset Date: {freshness['dataset_date']}")
print(f"Expected Date: {freshness['expected_date']}")
print(f"Days Stale: {freshness['days_stale']}")
print(f"Is Fresh: {freshness['is_fresh']}")

# Full data ingestion (uncomment to run)
# print("\n" + "="*60)
# print("FULL DATA INGESTION")
# print("="*60)
#
# from block1_data_ingestion import run_data_ingestion
# DATA_PACKAGE = run_data_ingestion()
#
# # Now use with Riskfolio-Lib for optimization
# if DATA_PACKAGE['data_is_fresh']:
#     import riskfolio as rp
#
#     # Create portfolio with validated data
#     port = rp.Portfolio(returns=DATA_PACKAGE['returns'][DATA_PACKAGE['full_confidence_assets']])
#
#     # Use pre-computed validated covariance if available
#     if DATA_PACKAGE['cov'] is not None:
#         port.cov = DATA_PACKAGE['cov']
#         port.mu = DATA_PACKAGE['mu']
#     else:
#         port.assets_stats(method_mu='hist', method_cov='ledoit')
#
#     # Optimize
#     w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0, hist=True)
#     print("\nOptimal Portfolio Weights:")
#     print(w.T)
# else:
#     print(f"\n⚠️ Data is stale ({DATA_PACKAGE['freshness_report']['overall_status']})")
#     print("Consider refreshing data before making trading decisions.")
