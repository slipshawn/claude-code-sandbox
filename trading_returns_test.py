"""
Multi-period trading returns testing module.

This module provides functionality to test trading strategies across multiple time periods,
generate symbol combinations, calculate returns statistics, and compare period vs future
period metrics.
"""

import itertools
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import pandas as pd
import numpy as np


def filter_symbols_by_dates(
    symbol_dfs: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    date_column: str = 'timestamp'
) -> Dict[str, pd.DataFrame]:
    """
    Filter symbol dataframes to contain only data within the specified date range.

    Parameters
    ----------
    symbol_dfs : Dict[str, pd.DataFrame]
        Dictionary of symbols with their OHLCV dataframes.
    start_date : str
        Start date string (YYYY-MM-DD format).
    end_date : str
        End date string (YYYY-MM-DD format).
    date_column : str, optional
        Name of the date/timestamp column. Default is 'timestamp'.
        If None, assumes the index contains datetime values.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Filtered dataframes containing only data within the date range.
        Symbols with no data in the range are excluded.
    """
    filtered_dfs = {}
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    for symbol, df in symbol_dfs.items():
        if df is None or df.empty:
            continue

        df_copy = df.copy()

        # Handle date filtering based on column or index
        if date_column and date_column in df_copy.columns:
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            mask = (df_copy[date_column] >= start_dt) & (df_copy[date_column] <= end_dt)
            filtered_df = df_copy[mask]
        else:
            # Assume datetime index
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
            filtered_df = df_copy[(df_copy.index >= start_dt) & (df_copy.index <= end_dt)]

        # Only include if data exists after filtering
        if not filtered_df.empty:
            filtered_dfs[symbol] = filtered_df

    return filtered_dfs


def generate_symbol_combinations(
    valid_symbols: List[str],
    n_combinations: int,
    combos_subset: float = 1.0,
    random_seed: Optional[int] = None
) -> List[Tuple[str, ...]]:
    """
    Generate combinations of symbols (asset baskets) for testing.

    Parameters
    ----------
    valid_symbols : List[str]
        List of valid symbol names to create combinations from.
    n_combinations : int
        Number of symbols per combination (basket size).
    combos_subset : float, optional
        Fraction of total combinations to sample (0.0 to 1.0). Default is 1.0.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    List[Tuple[str, ...]]
        List of symbol combinations (tuples).
    """
    if len(valid_symbols) < n_combinations:
        raise ValueError(
            f"Not enough valid symbols ({len(valid_symbols)}) "
            f"for requested combination size ({n_combinations})"
        )

    # Generate all possible combinations
    all_combos = list(itertools.combinations(valid_symbols, n_combinations))

    if combos_subset < 1.0:
        if random_seed is not None:
            random.seed(random_seed)
        sample_size = max(1, int(len(all_combos) * combos_subset))
        combos = random.sample(all_combos, sample_size)
    else:
        combos = all_combos

    return combos


def calculate_returns_statistics(
    symbol_dfs: Dict[str, pd.DataFrame],
    price_column: str = 'close',
    date_column: str = 'timestamp'
) -> Dict[str, any]:
    """
    Calculate returns statistics between valid symbols.

    Parameters
    ----------
    symbol_dfs : Dict[str, pd.DataFrame]
        Filtered dataframes for each symbol.
    price_column : str, optional
        Column name for price data. Default is 'close'.
    date_column : str, optional
        Column name for date/timestamp. Default is 'timestamp'.

    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - correlation_matrix: Correlation matrix of returns
        - mean_returns: Mean return per symbol
        - volatility: Standard deviation of returns per symbol
        - sharpe_ratios: Annualized Sharpe ratio per symbol (assuming risk-free rate = 0)
        - covariance_matrix: Covariance matrix of returns
        - max_drawdown: Maximum drawdown per symbol
    """
    if not symbol_dfs:
        return {}

    # Calculate returns for each symbol
    returns_dict = {}

    for symbol, df in symbol_dfs.items():
        if df.empty:
            continue

        df_copy = df.copy()

        # Set datetime index for alignment
        if date_column and date_column in df_copy.columns:
            df_copy = df_copy.set_index(date_column)

        if price_column not in df_copy.columns:
            continue

        prices = df_copy[price_column]
        returns = prices.pct_change().dropna()
        returns_dict[symbol] = returns

    if not returns_dict:
        return {}

    # Create aligned returns DataFrame
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna(how='all')

    # Calculate statistics
    stats = {}

    # Correlation matrix
    stats['correlation_matrix'] = returns_df.corr()

    # Covariance matrix
    stats['covariance_matrix'] = returns_df.cov()

    # Mean returns (annualized for 1-min bars: 525600 minutes/year)
    minutes_per_year = 525600
    stats['mean_returns'] = returns_df.mean()
    stats['annualized_mean_returns'] = returns_df.mean() * minutes_per_year

    # Volatility (annualized)
    stats['volatility'] = returns_df.std()
    stats['annualized_volatility'] = returns_df.std() * np.sqrt(minutes_per_year)

    # Sharpe ratios (annualized, assuming risk-free rate = 0)
    mean_returns = returns_df.mean()
    std_returns = returns_df.std()
    stats['sharpe_ratios'] = (mean_returns / std_returns) * np.sqrt(minutes_per_year)
    stats['sharpe_ratios'] = stats['sharpe_ratios'].replace([np.inf, -np.inf], np.nan)

    # Maximum drawdown per symbol
    max_drawdowns = {}
    for symbol in returns_df.columns:
        symbol_returns = returns_df[symbol].dropna()
        if len(symbol_returns) > 0:
            cumulative = (1 + symbol_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = cumulative / rolling_max - 1
            max_drawdowns[symbol] = drawdowns.min()
        else:
            max_drawdowns[symbol] = np.nan
    stats['max_drawdown'] = pd.Series(max_drawdowns)

    # Average pairwise correlation (useful metric for diversification)
    corr_matrix = stats['correlation_matrix']
    if corr_matrix.shape[0] > 1:
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        stats['avg_pairwise_correlation'] = upper_triangle.stack().mean()
    else:
        stats['avg_pairwise_correlation'] = np.nan

    return stats


def generate_date_periods(
    all_dates: List[str],
    period_length_days: int = 90,
    step_days: Optional[int] = None,
    future_period_days: Optional[int] = None
) -> List[Dict[str, any]]:
    """
    Generate rolling date periods for testing.

    Parameters
    ----------
    all_dates : List[str]
        List of all available date strings (YYYY-MM-DD format).
    period_length_days : int, optional
        Length of each test period in days. Default is 90.
    step_days : int, optional
        Number of days to step forward between periods.
        Default is period_length_days (non-overlapping).
    future_period_days : int, optional
        Length of future period to compare against.
        Default is same as period_length_days.

    Returns
    -------
    List[Dict[str, any]]
        List of period dictionaries with:
        - period_start: Start date of test period
        - period_end: End date of test period
        - future_start: Start date of future period (for comparison)
        - future_end: End date of future period
    """
    if step_days is None:
        step_days = period_length_days
    if future_period_days is None:
        future_period_days = period_length_days

    # Sort dates
    sorted_dates = sorted([pd.to_datetime(d) for d in all_dates])

    if len(sorted_dates) < 2:
        return []

    min_date = sorted_dates[0]
    max_date = sorted_dates[-1]

    periods = []
    current_start = min_date

    while current_start + timedelta(days=period_length_days) <= max_date:
        period_end = current_start + timedelta(days=period_length_days - 1)
        future_start = period_end + timedelta(days=1)
        future_end = future_start + timedelta(days=future_period_days - 1)

        period_info = {
            'period_start': current_start.strftime('%Y-%m-%d'),
            'period_end': period_end.strftime('%Y-%m-%d'),
            'future_start': future_start.strftime('%Y-%m-%d'),
            'future_end': future_end.strftime('%Y-%m-%d'),
            'has_future_data': future_end <= max_date
        }
        periods.append(period_info)

        current_start += timedelta(days=step_days)

    return periods


def run_multi_period_trading_test(
    symbol_dfs: Dict[str, pd.DataFrame],
    all_dates: List[str],
    trading_test_func: Callable,
    period_length_days: int = 90,
    step_days: Optional[int] = None,
    future_period_days: Optional[int] = None,
    n_combinations: int = 2,
    combos_subset: float = 0.5,
    freq: str = '1min',
    date_column: str = 'timestamp',
    random_seed: Optional[int] = None,
    min_symbols_required: int = 2,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Run multi-period trading tests to compare period vs future period performance.

    This function iterates over date periods, filters symbols, generates combinations,
    runs trading tests, and calculates statistics to identify metrics that predict
    future performance.

    Parameters
    ----------
    symbol_dfs : Dict[str, pd.DataFrame]
        Dictionary of symbols with their OHLCV 1-minute dataframes.
        Example: {'BTC': df_btc, 'ETH': df_eth}
    all_dates : List[str]
        List of all available date strings (YYYY-MM-DD format).
    trading_test_func : Callable
        The trading test function to call. Should match signature:
        iterrative_combinations_test(symbols_data, combos, date_period, freq) -> dict
    period_length_days : int, optional
        Length of each test period in days. Default is 90.
    step_days : int, optional
        Number of days to step forward between periods.
        Default is period_length_days (non-overlapping).
    future_period_days : int, optional
        Length of future period for comparison. Default is same as period_length_days.
    n_combinations : int, optional
        Number of symbols per combination (basket size). Default is 2.
    combos_subset : float, optional
        Fraction of total combinations to sample (0.0 to 1.0). Default is 0.5.
    freq : str, optional
        Frequency string for the bar periods. Default is '1min'.
    date_column : str, optional
        Name of the date/timestamp column in dataframes. Default is 'timestamp'.
    random_seed : int, optional
        Random seed for reproducibility.
    min_symbols_required : int, optional
        Minimum number of symbols required to run tests. Default is 2.
    verbose : bool, optional
        Whether to print progress information. Default is True.

    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - periods_results: List of results for each period with:
            - period_info: Date range information
            - valid_symbols: Symbols with data in period
            - combinations_tested: Number of combinations tested
            - period_stats: Returns statistics for the period
            - period_trading_results: Results from trading test
            - future_stats: Returns statistics for future period (if available)
            - future_trading_results: Results from future trading test (if available)
        - summary: Aggregated metrics across all periods
        - comparison_analysis: Period vs future period comparison metrics
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Generate date periods
    periods = generate_date_periods(
        all_dates=all_dates,
        period_length_days=period_length_days,
        step_days=step_days,
        future_period_days=future_period_days
    )

    if not periods:
        raise ValueError("Not enough dates to generate any periods")

    if verbose:
        print(f"Generated {len(periods)} test periods")

    all_results = []

    for i, period_info in enumerate(periods):
        if verbose:
            print(f"\n--- Period {i+1}/{len(periods)} ---")
            print(f"  Test: {period_info['period_start']} to {period_info['period_end']}")

        period_result = {
            'period_index': i,
            'period_info': period_info,
            'valid_symbols': [],
            'combinations_tested': 0,
            'period_stats': {},
            'period_trading_results': {},
            'future_stats': {},
            'future_trading_results': {},
            'error': None
        }

        try:
            # Filter symbols for current period
            filtered_dfs = filter_symbols_by_dates(
                symbol_dfs=symbol_dfs,
                start_date=period_info['period_start'],
                end_date=period_info['period_end'],
                date_column=date_column
            )

            valid_symbols = list(filtered_dfs.keys())
            period_result['valid_symbols'] = valid_symbols

            if verbose:
                print(f"  Valid symbols: {len(valid_symbols)}")

            if len(valid_symbols) < min_symbols_required:
                period_result['error'] = f"Insufficient symbols: {len(valid_symbols)} < {min_symbols_required}"
                all_results.append(period_result)
                continue

            # Calculate statistics for current period
            period_stats = calculate_returns_statistics(
                symbol_dfs=filtered_dfs,
                date_column=date_column
            )
            period_result['period_stats'] = period_stats

            # Generate combinations
            actual_n_combos = min(n_combinations, len(valid_symbols))
            combos = generate_symbol_combinations(
                valid_symbols=valid_symbols,
                n_combinations=actual_n_combos,
                combos_subset=combos_subset,
                random_seed=random_seed
            )
            period_result['combinations_tested'] = len(combos)

            if verbose:
                print(f"  Testing {len(combos)} combinations")

            # Run trading test for current period
            date_period = [period_info['period_start'], period_info['period_end']]

            trading_results = trading_test_func(
                symbols_data=filtered_dfs,
                combos=combos,
                date_period=date_period,
                freq=freq
            )
            period_result['period_trading_results'] = trading_results

            # Process future period if data is available
            if period_info['has_future_data']:
                if verbose:
                    print(f"  Future: {period_info['future_start']} to {period_info['future_end']}")

                # Filter symbols for future period
                future_filtered_dfs = filter_symbols_by_dates(
                    symbol_dfs=symbol_dfs,
                    start_date=period_info['future_start'],
                    end_date=period_info['future_end'],
                    date_column=date_column
                )

                # Only test symbols that were valid in both periods
                common_symbols = set(valid_symbols) & set(future_filtered_dfs.keys())

                if len(common_symbols) >= min_symbols_required:
                    # Filter to common symbols only
                    future_common_dfs = {s: future_filtered_dfs[s] for s in common_symbols}

                    # Calculate future statistics
                    future_stats = calculate_returns_statistics(
                        symbol_dfs=future_common_dfs,
                        date_column=date_column
                    )
                    period_result['future_stats'] = future_stats

                    # Filter combos to only include those with all symbols in common
                    valid_combos = [c for c in combos if all(s in common_symbols for s in c)]

                    if valid_combos:
                        # Run trading test for future period
                        future_date_period = [period_info['future_start'], period_info['future_end']]

                        future_trading_results = trading_test_func(
                            symbols_data=future_common_dfs,
                            combos=valid_combos,
                            date_period=future_date_period,
                            freq=freq
                        )
                        period_result['future_trading_results'] = future_trading_results

                        if verbose:
                            print(f"  Future combinations tested: {len(valid_combos)}")
                    else:
                        if verbose:
                            print("  No valid combos for future period")
                else:
                    if verbose:
                        print(f"  Insufficient common symbols for future: {len(common_symbols)}")
            else:
                if verbose:
                    print("  No future data available")

        except Exception as e:
            period_result['error'] = str(e)
            if verbose:
                print(f"  Error: {e}")

        all_results.append(period_result)

    # Generate summary and comparison analysis
    summary = _generate_summary(all_results)
    comparison = _generate_comparison_analysis(all_results)

    return {
        'periods_results': all_results,
        'summary': summary,
        'comparison_analysis': comparison
    }


def _generate_summary(all_results: List[Dict]) -> Dict[str, any]:
    """Generate summary statistics across all periods."""
    summary = {
        'total_periods': len(all_results),
        'successful_periods': 0,
        'periods_with_future': 0,
        'avg_valid_symbols': 0,
        'avg_combinations_tested': 0,
        'errors': []
    }

    valid_symbols_counts = []
    combos_counts = []

    for result in all_results:
        if result['error']:
            summary['errors'].append({
                'period_index': result['period_index'],
                'error': result['error']
            })
        else:
            summary['successful_periods'] += 1
            valid_symbols_counts.append(len(result['valid_symbols']))
            combos_counts.append(result['combinations_tested'])

            if result.get('future_trading_results'):
                summary['periods_with_future'] += 1

    if valid_symbols_counts:
        summary['avg_valid_symbols'] = np.mean(valid_symbols_counts)
    if combos_counts:
        summary['avg_combinations_tested'] = np.mean(combos_counts)

    return summary


def _generate_comparison_analysis(all_results: List[Dict]) -> Dict[str, any]:
    """
    Generate comparison analysis between period and future period metrics.

    This identifies which metrics/statistics from the test period correlate
    with better performance in the future period.
    """
    comparison = {
        'period_vs_future_correlations': {},
        'metric_predictiveness': {},
        'top_performing_periods': [],
        'observations': []
    }

    # Collect paired period/future data
    paired_data = []

    for result in all_results:
        if (result.get('period_stats') and
            result.get('future_stats') and
            not result.get('error')):
            paired_data.append({
                'period_index': result['period_index'],
                'period_stats': result['period_stats'],
                'future_stats': result['future_stats'],
                'period_trading': result.get('period_trading_results', {}),
                'future_trading': result.get('future_trading_results', {})
            })

    if len(paired_data) < 2:
        comparison['observations'].append(
            "Insufficient paired data for meaningful comparison analysis"
        )
        return comparison

    # Analyze correlation persistence
    period_avg_corrs = []
    future_avg_corrs = []

    for pair in paired_data:
        if 'avg_pairwise_correlation' in pair['period_stats']:
            period_avg_corrs.append(pair['period_stats']['avg_pairwise_correlation'])
        if 'avg_pairwise_correlation' in pair['future_stats']:
            future_avg_corrs.append(pair['future_stats']['avg_pairwise_correlation'])

    if len(period_avg_corrs) > 1 and len(future_avg_corrs) > 1:
        if len(period_avg_corrs) == len(future_avg_corrs):
            corr_persistence = np.corrcoef(period_avg_corrs, future_avg_corrs)[0, 1]
            comparison['period_vs_future_correlations']['avg_correlation_persistence'] = corr_persistence

            if not np.isnan(corr_persistence):
                if corr_persistence > 0.5:
                    comparison['observations'].append(
                        f"Strong correlation persistence ({corr_persistence:.2f}): "
                        "Historical correlations tend to persist into future periods"
                    )
                elif corr_persistence < -0.5:
                    comparison['observations'].append(
                        f"Correlation reversal ({corr_persistence:.2f}): "
                        "Historical correlations tend to reverse in future periods"
                    )

    # Analyze volatility persistence
    period_vols = []
    future_vols = []

    for pair in paired_data:
        if 'volatility' in pair['period_stats'] and not pair['period_stats']['volatility'].empty:
            period_vols.append(pair['period_stats']['volatility'].mean())
        if 'volatility' in pair['future_stats'] and not pair['future_stats']['volatility'].empty:
            future_vols.append(pair['future_stats']['volatility'].mean())

    if len(period_vols) > 1 and len(future_vols) > 1:
        if len(period_vols) == len(future_vols):
            vol_persistence = np.corrcoef(period_vols, future_vols)[0, 1]
            comparison['period_vs_future_correlations']['volatility_persistence'] = vol_persistence

            if not np.isnan(vol_persistence) and vol_persistence > 0.3:
                comparison['observations'].append(
                    f"Volatility clustering detected ({vol_persistence:.2f}): "
                    "High volatility periods tend to be followed by high volatility"
                )

    comparison['total_paired_periods'] = len(paired_data)

    return comparison


# Convenience function for quick testing
def create_sample_test_runner():
    """
    Create a sample test runner for demonstration purposes.

    Returns a mock trading test function that can be used for testing
    the multi-period framework.
    """
    def mock_trading_test(
        symbols_data: Dict[str, pd.DataFrame],
        combos: List[Tuple],
        date_period: List[str],
        freq: str
    ) -> Dict:
        """Mock trading test function for demonstration."""
        results = {}

        for i, combo in enumerate(combos):
            test_id = f"test_{i}_{'-'.join(combo)}"

            # Generate mock results
            results[test_id] = {
                'symbols': combo,
                'date_period': date_period,
                'total_return': np.random.uniform(-0.2, 0.4),
                'strategy_volatility': np.random.uniform(0.1, 0.5),
                'sharpe_ratio': np.random.uniform(-1, 2),
                'max_drawdown': np.random.uniform(-0.3, 0),
                'win_rate': np.random.uniform(0.4, 0.6)
            }

        return results

    return mock_trading_test


if __name__ == "__main__":
    # Example usage demonstration
    print("Multi-period Trading Returns Test Module")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from trading_returns_test import run_multi_period_trading_test

    # Your symbol dataframes
    symbol_dfs = {
        'BTC': btc_df,  # OHLCV 1-min dataframe
        'ETH': eth_df,
        'SOL': sol_df,
        ...
    }

    # All available dates
    all_dates = ['2024-01-01', '2024-01-02', ..., '2024-12-31']

    # Run multi-period test
    results = run_multi_period_trading_test(
        symbol_dfs=symbol_dfs,
        all_dates=all_dates,
        trading_test_func=iterrative_combinations_test,
        period_length_days=90,
        step_days=30,  # Rolling 30-day step
        future_period_days=30,  # Compare to next 30 days
        n_combinations=3,
        combos_subset=0.3,
        freq='1min',
        verbose=True
    )

    # Access results
    print(results['summary'])
    print(results['comparison_analysis'])

    for period_result in results['periods_results']:
        print(period_result['period_info'])
        print(period_result['period_stats'])
    """)
