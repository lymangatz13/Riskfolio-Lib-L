"""
==========================================
BLOCK 1: ENVIRONMENT, DATA INGESTION & QC
==========================================
Enhanced version integrating Riskfolio-Lib validation
with comprehensive data freshness verification.

Author: Portfolio Manager
"""

# --- 1.1 IMPORTS ---
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
import time
from scipy import stats
from scipy.optimize import minimize
from datetime import datetime, timedelta
from sklearn.covariance import LedoitWolf

# Riskfolio-Lib imports for validation
import riskfolio as rp
import riskfolio.src.AuxFunctions as af
import riskfolio.src.ParamsEstimation as pe

# warnings.filterwarnings("ignore")

# --- 1.2 ENVIRONMENT LOGGING ---
print(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"NumPy: {np.__version__} | Pandas: {pd.__version__}")
print(f"Riskfolio-Lib: {rp.__version__}")

# --- 1.3 ASSET UNIVERSE DEFINITION ---
ASSET_CLASSES = {
    'crypto': ['BTC-USD', 'SOL-USD'],
    'single_stock': ['OKLO', 'QBTS', 'CRML', 'TMC', 'MU', 'INTC', 'SLB', 'UUUU', 'USAR', 'TSLA', 'NVDA', 'GOOG', 'HOOD', 'CME'],
    'etf': ['SLV', 'IAUM', 'SHLD', 'QQQM', 'XLF', 'MAGS', 'REET', 'USRT', 'CPER', 'URA']
}
assets = ASSET_CLASSES['crypto'] + ASSET_CLASSES['single_stock'] + ASSET_CLASSES['etf']

BENCHMARKS = {
    'market': '^GSPC',       # S&P 500 (US large cap)
    'small_cap': '^SP600',   # S&P 600 (US small cap)
    'intl': 'VXUS'           # Vanguard Ex-US (International)
}
benchmarks = list(BENCHMARKS.values())
all_tickers = assets + benchmarks

# --- 1.4 CONFIGURATION ---
FIXED_START = "2023-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Minimum days for full inclusion vs. flagged inclusion
MIN_HISTORY_FULL = 252      # Full confidence in estimates
MIN_HISTORY_PARTIAL = 180    # Include with uncertainty flag

# Walk-forward holdout
TRAIN_END = (datetime.today() - timedelta(days=63)).strftime('%Y-%m-%d')

# Data freshness thresholds
FRESHNESS_THRESHOLDS = {
    'critical': 1,    # Data older than 1 business day is critical
    'warning': 3,     # Data older than 3 business days is warning
    'acceptable': 5   # Data older than 5 business days is stale
}

print(f"\n[CONFIG] Universe: {len(assets)} assets | Benchmarks: {list(BENCHMARKS.keys())}")
print(f"[CONFIG] Date Range: {FIXED_START} → {end_date}")


# --- 1.5 DATA FRESHNESS VERIFICATION FUNCTIONS ---
def get_market_status():
    """
    Check if US markets are currently open and get last trading day.
    Returns dict with market status info.
    """
    now = datetime.now()
    today = now.date()
    weekday = today.weekday()  # 0=Monday, 6=Sunday

    # Calculate last expected trading day
    if weekday == 0:  # Monday
        last_trading_day = today - timedelta(days=3)  # Friday
    elif weekday == 6:  # Sunday
        last_trading_day = today - timedelta(days=2)  # Friday
    elif weekday == 5:  # Saturday
        last_trading_day = today - timedelta(days=1)  # Friday
    else:
        # Weekday - check if markets closed yet (4 PM ET)
        market_close = now.replace(hour=16, minute=0, second=0)
        if now.hour >= 16:
            last_trading_day = today
        else:
            last_trading_day = today - timedelta(days=1)
            if last_trading_day.weekday() == 6:  # Sunday
                last_trading_day -= timedelta(days=2)
            elif last_trading_day.weekday() == 5:  # Saturday
                last_trading_day -= timedelta(days=1)

    return {
        'current_time': now,
        'today': today,
        'weekday': weekday,
        'last_expected_trading_day': last_trading_day,
        'is_weekend': weekday >= 5,
        'day_name': today.strftime('%A')
    }


def verify_data_freshness(prices_df, asset_classes_dict, benchmarks_dict, thresholds):
    """
    Comprehensive data freshness verification.

    Checks:
    1. Latest date in dataset vs expected
    2. Per-asset staleness (some assets may have gaps)
    3. Weekend handling for crypto vs equities
    4. Returns detailed freshness report
    """
    market_status = get_market_status()
    last_expected = market_status['last_expected_trading_day']

    report = {
        'overall_status': 'OK',
        'market_status': market_status,
        'dataset_last_date': None,
        'expected_last_date': last_expected,
        'days_stale': 0,
        'stale_assets': [],
        'fresh_assets': [],
        'missing_recent_data': [],
        'warnings': [],
        'errors': []
    }

    # Get dataset's most recent date
    dataset_last = prices_df.index.max().date() if hasattr(prices_df.index.max(), 'date') else prices_df.index.max()
    report['dataset_last_date'] = dataset_last

    # Calculate staleness (in business days)
    if isinstance(dataset_last, str):
        dataset_last = datetime.strptime(dataset_last, '%Y-%m-%d').date()

    days_diff = (last_expected - dataset_last).days

    # Adjust for weekends (crypto trades, equities don't)
    business_days_stale = 0
    check_date = dataset_last
    while check_date < last_expected:
        check_date += timedelta(days=1)
        if check_date.weekday() < 5:  # Monday-Friday
            business_days_stale += 1

    report['days_stale'] = business_days_stale

    # Determine overall status
    if business_days_stale >= thresholds['acceptable']:
        report['overall_status'] = 'STALE'
        report['errors'].append(f"Data is {business_days_stale} business days old (threshold: {thresholds['acceptable']})")
    elif business_days_stale >= thresholds['warning']:
        report['overall_status'] = 'WARNING'
        report['warnings'].append(f"Data is {business_days_stale} business days old")
    elif business_days_stale >= thresholds['critical']:
        report['overall_status'] = 'DELAYED'
        report['warnings'].append(f"Data may be delayed by {business_days_stale} business day(s)")

    # Per-asset freshness check
    crypto_tickers = asset_classes_dict.get('crypto', [])

    for col in prices_df.columns:
        asset_last_valid = prices_df[col].last_valid_index()
        if asset_last_valid is None:
            report['missing_recent_data'].append(col)
            continue

        asset_last_date = asset_last_valid.date() if hasattr(asset_last_valid, 'date') else asset_last_valid

        # For crypto, compare to today (trades 24/7)
        if col in crypto_tickers:
            expected_for_asset = datetime.today().date() - timedelta(days=1)
        else:
            expected_for_asset = last_expected

        asset_days_stale = (expected_for_asset - asset_last_date).days

        if asset_days_stale > thresholds['warning']:
            report['stale_assets'].append({
                'ticker': col,
                'last_date': asset_last_date,
                'days_stale': asset_days_stale,
                'is_crypto': col in crypto_tickers
            })
        else:
            report['fresh_assets'].append(col)

    return report


def print_freshness_report(report):
    """Pretty print the freshness verification report."""
    print("\n" + "="*70)
    print("DATA FRESHNESS VERIFICATION")
    print("="*70)

    status_emoji = {
        'OK': '✓',
        'DELAYED': '⚠️',
        'WARNING': '⚠️',
        'STALE': '⛔'
    }

    print(f"\nOverall Status: {status_emoji.get(report['overall_status'], '?')} {report['overall_status']}")
    print(f"Current Time: {report['market_status']['current_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Today: {report['market_status']['day_name']}")
    print(f"\nDataset Latest Date: {report['dataset_last_date']}")
    print(f"Expected Latest Date: {report['expected_last_date']}")
    print(f"Business Days Stale: {report['days_stale']}")

    if report['errors']:
        print(f"\n⛔ ERRORS:")
        for err in report['errors']:
            print(f"   - {err}")

    if report['warnings']:
        print(f"\n⚠️ WARNINGS:")
        for warn in report['warnings']:
            print(f"   - {warn}")

    if report['stale_assets']:
        print(f"\n⚠️ STALE ASSETS ({len(report['stale_assets'])}):")
        for asset in report['stale_assets']:
            crypto_tag = " [CRYPTO]" if asset['is_crypto'] else ""
            print(f"   - {asset['ticker']}: last data {asset['last_date']} ({asset['days_stale']} days old){crypto_tag}")

    if report['missing_recent_data']:
        print(f"\n⛔ MISSING DATA:")
        print(f"   {report['missing_recent_data']}")

    print(f"\n✓ Fresh Assets: {len(report['fresh_assets'])}/{len(report['fresh_assets']) + len(report['stale_assets'])}")

    return report['overall_status']


# --- 1.6 DATA FETCH (with rate limit handling) ---
def fetch_with_rate_limit_handling(tickers, start, end, max_retries=3):
    """
    Fetch data with 15-min interval fallback to EOD on rate limit.
    Uses yfinance with adaptive interval handling.
    """
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=True,
                interval='1d'  # EOD data - most reliable
            )
            if raw.empty:
                raise ValueError("Empty data returned")
            print(f"  ✓ Data fetched successfully (EOD interval)")
            return raw, 'EOD'

        except Exception as e:
            error_msg = str(e).lower()

            if any(x in error_msg for x in ['rate limit', '429', 'too many requests', 'exceeded']):
                wait_time = 15 * 60  # 15 minutes
                print(f"  ⚠️ API rate limit hit (attempt {attempt+1}/{max_retries})")
                print(f"  Waiting {wait_time//60} minutes before retry...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️ Fetch error: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in 30 seconds...")
                    time.sleep(30)

    # Final fallback
    print("  Attempting final EOD fallback...")
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
            interval='1d'
        )
        return raw, 'EOD_FALLBACK'
    except Exception as e:
        raise RuntimeError(f"All fetch attempts failed: {e}")


# --- 1.7 EXTRACT CLOSE PRICES ---
def extract_close_prices(data):
    """Handle yfinance's inconsistent return formats."""
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            return data.xs('Close', level=0, axis=1)
        elif 'Close' in data.columns.get_level_values(1):
            return data.xs('Close', level=1, axis=1)
    if 'Close' in data.columns:
        return data[['Close']]
    return data


# --- 1.8 BUILD UNIFIED CALENDAR (Crypto + Equity) ---
def build_unified_calendar(prices_df, benchmarks_dict, asset_classes):
    """
    Create a calendar that includes:
    - All market trading days (from S&P 500)
    - Weekends (for crypto price changes)
    """
    crypto_tickers = asset_classes['crypto']
    non_crypto = [c for c in prices_df.columns if c not in crypto_tickers]

    market_ticker = benchmarks_dict['market']
    if market_ticker not in prices_df.columns:
        raise ValueError(f"Benchmark {market_ticker} not in data")

    market_days = prices_df[market_ticker].dropna().index
    full_calendar = prices_df.index
    df_unified = pd.DataFrame(index=full_calendar)

    # CRYPTO: Use actual daily prices
    for ticker in crypto_tickers:
        if ticker in prices_df.columns:
            df_unified[ticker] = prices_df[ticker]

    # NON-CRYPTO: Market days only, then forward-fill weekends
    for ticker in non_crypto:
        if ticker in prices_df.columns:
            df_unified[ticker] = np.nan
            market_day_mask = df_unified.index.isin(market_days)
            df_unified.loc[market_day_mask, ticker] = prices_df.loc[
                prices_df.index.isin(market_days), ticker
            ]
            df_unified[ticker] = df_unified[ticker].ffill()

    return df_unified, market_days


# --- 1.9 RETURNS CALCULATION ---
def calculate_returns(prices_df, asset_classes, market_days):
    """
    Calculate returns with proper handling:
    - Crypto: Daily returns (including weekends)
    - Equities: Trading-day returns only
    """
    crypto_tickers = asset_classes['crypto']

    df_returns_full = prices_df.pct_change()
    df_prices_market = prices_df.loc[prices_df.index.isin(market_days)].copy()
    df_returns_market = df_prices_market.pct_change()

    df_returns_full = df_returns_full.iloc[1:]
    df_returns_market = df_returns_market.iloc[1:]

    return df_returns_full, df_returns_market


# --- 1.10 HISTORY ANALYSIS & ASSET METADATA ---
def analyze_history(returns_df, min_full, min_partial, asset_classes):
    """Categorize assets by data availability for downstream weighting."""
    meta = {}
    total_days = len(returns_df)
    window_start = returns_df.index[0]
    early_start_threshold = total_days * 0.20

    for col in returns_df.columns:
        valid_count = returns_df[col].notna().sum()
        first_valid = returns_df[col].first_valid_index()
        last_valid = returns_df[col].last_valid_index()

        asset_class = 'benchmark'
        for cls, tickers in asset_classes.items():
            if col in tickers:
                asset_class = cls
                break

        if first_valid is not None:
            days_from_start = (first_valid - window_start).days
        else:
            days_from_start = total_days

        coverage_pct = valid_count / total_days * 100

        if valid_count >= min_full and days_from_start <= early_start_threshold:
            confidence = 'full'
        elif valid_count >= min_partial:
            confidence = 'partial'
        else:
            confidence = 'insufficient'

        meta[col] = {
            'asset_class': asset_class,
            'valid_days': valid_count,
            'total_days': total_days,
            'first_date': first_valid,
            'last_date': last_valid,
            'days_from_start': days_from_start,
            'confidence': confidence,
            'coverage_pct': coverage_pct
        }

    return meta


# --- 1.11 RISKFOLIO-LIB VALIDATION ---
def validate_with_riskfolio(returns_df, method_cov='ledoit'):
    """
    Use Riskfolio-Lib to validate and optionally fix the covariance matrix.

    Returns:
        dict with validation results and fixed covariance if needed
    """
    validation = {
        'covariance_valid': False,
        'covariance_fixed': False,
        'original_cov': None,
        'fixed_cov': None,
        'mu': None,
        'warnings': []
    }

    # Drop any columns with NaN for covariance calculation
    clean_returns = returns_df.dropna(axis=1, how='any')

    if clean_returns.shape[1] < 2:
        validation['warnings'].append("Insufficient assets for covariance calculation")
        return validation

    try:
        # Estimate mean using Riskfolio's methods
        mu = pe.mean_vector(clean_returns, method='hist')
        validation['mu'] = mu

        # Estimate covariance using Riskfolio's methods
        cov = pe.covar_matrix(clean_returns, method=method_cov)
        validation['original_cov'] = cov

        # Check positive definiteness using Riskfolio's function
        is_valid = af.is_pos_def(cov, threshold=1e-8)
        validation['covariance_valid'] = is_valid

        if not is_valid:
            validation['warnings'].append("Covariance matrix is not positive definite")
            # Fix using Riskfolio's cov_fix function
            fixed_cov = af.cov_fix(cov, method="clipped", threshold=1e-8)
            validation['fixed_cov'] = fixed_cov
            validation['covariance_fixed'] = True

            # Verify fix worked
            if af.is_pos_def(fixed_cov, threshold=1e-8):
                validation['warnings'].append("Covariance matrix successfully fixed")
            else:
                validation['warnings'].append("WARNING: Could not fix covariance matrix")

        return validation

    except Exception as e:
        validation['warnings'].append(f"Validation error: {str(e)}")
        return validation


def print_riskfolio_validation(validation):
    """Print Riskfolio validation results."""
    print("\n" + "-"*50)
    print("RISKFOLIO-LIB COVARIANCE VALIDATION")
    print("-"*50)

    if validation['covariance_valid']:
        print("✓ Covariance matrix is positive definite")
    else:
        print("⚠️ Covariance matrix was NOT positive definite")
        if validation['covariance_fixed']:
            print("✓ Matrix successfully fixed using clipped eigenvalue method")

    if validation['warnings']:
        for warn in validation['warnings']:
            print(f"  - {warn}")

    if validation['mu'] is not None:
        print(f"\n  Expected returns estimated for {validation['mu'].shape[1]} assets")

    if validation['original_cov'] is not None:
        print(f"  Covariance matrix: {validation['original_cov'].shape[0]}x{validation['original_cov'].shape[1]}")


# ==========================================
# MAIN EXECUTION
# ==========================================
def run_data_ingestion():
    """Main function to run data ingestion and QC."""

    print(f"\nFetching data...")
    raw_data, data_interval = fetch_with_rate_limit_handling(all_tickers, FIXED_START, end_date)
    print(f"  Data Interval: {data_interval}")

    # Extract close prices
    df_prices_raw = extract_close_prices(raw_data)
    df_prices_raw = df_prices_raw.dropna(axis=1, how='all')

    # === DATA FRESHNESS VERIFICATION ===
    freshness_report = verify_data_freshness(
        df_prices_raw,
        ASSET_CLASSES,
        BENCHMARKS,
        FRESHNESS_THRESHOLDS
    )
    freshness_status = print_freshness_report(freshness_report)

    # Build unified calendar
    df_prices, market_days = build_unified_calendar(df_prices_raw, BENCHMARKS, ASSET_CLASSES)

    # Calculate returns
    df_returns_full, df_returns = calculate_returns(df_prices, ASSET_CLASSES, market_days)

    # Analyze history
    ASSET_META = analyze_history(df_returns, MIN_HISTORY_FULL, MIN_HISTORY_PARTIAL, ASSET_CLASSES)

    # === RISKFOLIO-LIB VALIDATION ===
    riskfolio_validation = validate_with_riskfolio(df_returns, method_cov='ledoit')
    print_riskfolio_validation(riskfolio_validation)

    # === DATA QUALITY REPORT ===
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)

    print(f"Full Calendar Days: {len(df_returns_full)}")
    print(f"Market Trading Days: {len(df_returns)}")
    print(f"Assets Loaded: {len(df_returns.columns)}/{len(all_tickers)}")

    # Failed tickers
    failed_tickers = set(all_tickers) - set(df_prices.columns)
    if failed_tickers:
        print(f"\n⛔ FAILED TO LOAD: {failed_tickers}")

    # History summary
    print(f"\n{'Ticker':<10} {'Class':<12} {'Days':<6} {'Coverage':<10} {'Confidence'}")
    print("-"*55)

    full_conf = []
    partial_conf = []
    insufficient = []

    for ticker, meta in ASSET_META.items():
        if meta['asset_class'] == 'benchmark':
            continue

        conf_display = {
            'full': '✓ Full',
            'partial': '⚠️ Partial',
            'insufficient': '⛔ Low'
        }[meta['confidence']]

        if meta['confidence'] != 'full':
            print(f"{ticker:<10} {meta['asset_class']:<12} {meta['valid_days']:<6} "
                  f"{meta['coverage_pct']:<10.1f}% {conf_display}")

        if meta['confidence'] == 'full':
            full_conf.append(ticker)
        elif meta['confidence'] == 'partial':
            partial_conf.append(ticker)
        else:
            insufficient.append(ticker)

    print("-"*55)
    print(f"Full Confidence: {len(full_conf)} assets")
    print(f"Partial (will shrink estimates): {len(partial_conf)} assets → {partial_conf}")
    print(f"Insufficient (exclude from optimization): {len(insufficient)} assets → {insufficient}")

    # === PACKAGE DATA ===
    DATA_PACKAGE = {
        # Price data
        'prices': df_prices,
        'prices_market_only': df_prices.loc[df_prices.index.isin(market_days)],

        # Returns data
        'returns_full': df_returns_full,
        'returns': df_returns,

        # Riskfolio-Lib validation results
        'riskfolio_validation': riskfolio_validation,
        'mu': riskfolio_validation.get('mu'),
        'cov': riskfolio_validation.get('fixed_cov') or riskfolio_validation.get('original_cov'),

        # Data freshness
        'freshness_report': freshness_report,
        'data_is_fresh': freshness_status == 'OK',

        # Metadata
        'asset_meta': ASSET_META,
        'asset_classes': ASSET_CLASSES,
        'benchmarks': BENCHMARKS,
        'market_days': market_days,

        # Config
        'min_history_full': MIN_HISTORY_FULL,
        'min_history_partial': MIN_HISTORY_PARTIAL,
        'train_end': TRAIN_END,

        # Quick access lists
        'full_confidence_assets': full_conf,
        'partial_confidence_assets': partial_conf,
        'insufficient_assets': insufficient
    }

    print("\n" + "="*70)
    print("DATA PACKAGE READY")
    print(f"  → Data Freshness: {'✓ FRESH' if DATA_PACKAGE['data_is_fresh'] else '⚠️ ' + freshness_status}")
    print(f"  → Use DATA_PACKAGE['returns'] for regression (market-day aligned)")
    print(f"  → Use DATA_PACKAGE['returns_full'] for portfolio NAV tracking")
    print(f"  → Use DATA_PACKAGE['cov'] for validated covariance matrix")
    print(f"  → Partial-confidence assets: {partial_conf}")
    print("="*70)

    return DATA_PACKAGE


# Quick freshness check function for external use
def quick_freshness_check(prices_df=None):
    """
    Quick standalone check if data is fresh.
    Can be called without running full ingestion.
    """
    if prices_df is None:
        # Fetch just benchmarks for quick check
        try:
            quick_data = yf.download(
                ['^GSPC'],
                period='5d',
                progress=False,
                auto_adjust=False
            )
            prices_df = quick_data['Close'] if 'Close' in quick_data.columns else quick_data
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    report = verify_data_freshness(
        pd.DataFrame(prices_df) if not isinstance(prices_df, pd.DataFrame) else prices_df,
        ASSET_CLASSES,
        BENCHMARKS,
        FRESHNESS_THRESHOLDS
    )

    return {
        'status': report['overall_status'],
        'dataset_date': report['dataset_last_date'],
        'expected_date': report['expected_last_date'],
        'days_stale': report['days_stale'],
        'is_fresh': report['overall_status'] == 'OK'
    }


if __name__ == "__main__":
    DATA_PACKAGE = run_data_ingestion()
