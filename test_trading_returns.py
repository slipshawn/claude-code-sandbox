"""
Unit tests for the multi-period trading returns test module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from trading_returns_test import (
    filter_symbols_by_dates,
    generate_symbol_combinations,
    calculate_returns_statistics,
    generate_date_periods,
    run_multi_period_trading_test,
    create_sample_test_runner
)


def create_sample_ohlcv_df(
    symbol: str,
    start_date: str,
    num_days: int = 30,
    freq_minutes: int = 1
) -> pd.DataFrame:
    """Create a sample OHLCV dataframe for testing."""
    start = pd.to_datetime(start_date)
    periods = num_days * 24 * 60 // freq_minutes

    dates = pd.date_range(start=start, periods=periods, freq=f'{freq_minutes}min')

    # Generate realistic-ish price data
    np.random.seed(hash(symbol) % 2**32)
    base_price = 100 + np.random.uniform(0, 1000)
    returns = np.random.normal(0, 0.001, periods)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, periods)),
        'high': prices * (1 + np.random.uniform(0, 0.002, periods)),
        'low': prices * (1 - np.random.uniform(0, 0.002, periods)),
        'close': prices,
        'volume': np.random.uniform(1000, 100000, periods)
    })

    return df


class TestFilterSymbolsByDates:
    """Tests for filter_symbols_by_dates function."""

    def test_basic_filtering(self):
        """Test basic date filtering."""
        df = create_sample_ohlcv_df('BTC', '2024-01-01', num_days=30)
        symbol_dfs = {'BTC': df}

        result = filter_symbols_by_dates(
            symbol_dfs=symbol_dfs,
            start_date='2024-01-05',
            end_date='2024-01-10'
        )

        assert 'BTC' in result
        assert len(result['BTC']) > 0

        # Check all dates are within range
        dates = pd.to_datetime(result['BTC']['timestamp'])
        assert all(dates >= pd.to_datetime('2024-01-05'))
        assert all(dates <= pd.to_datetime('2024-01-10'))

    def test_empty_after_filtering(self):
        """Test that symbols with no data in range are excluded."""
        df = create_sample_ohlcv_df('BTC', '2024-01-01', num_days=10)
        symbol_dfs = {'BTC': df}

        result = filter_symbols_by_dates(
            symbol_dfs=symbol_dfs,
            start_date='2024-03-01',
            end_date='2024-03-10'
        )

        assert 'BTC' not in result
        assert len(result) == 0

    def test_multiple_symbols(self):
        """Test filtering with multiple symbols."""
        symbol_dfs = {
            'BTC': create_sample_ohlcv_df('BTC', '2024-01-01', num_days=30),
            'ETH': create_sample_ohlcv_df('ETH', '2024-01-15', num_days=30),
            'SOL': create_sample_ohlcv_df('SOL', '2024-02-01', num_days=30)
        }

        result = filter_symbols_by_dates(
            symbol_dfs=symbol_dfs,
            start_date='2024-01-20',
            end_date='2024-01-25'
        )

        # Only BTC and ETH should have data in this range
        assert 'BTC' in result
        assert 'ETH' in result
        assert 'SOL' not in result

    def test_none_and_empty_handling(self):
        """Test handling of None and empty dataframes."""
        symbol_dfs = {
            'BTC': create_sample_ohlcv_df('BTC', '2024-01-01', num_days=30),
            'ETH': None,
            'SOL': pd.DataFrame()
        }

        result = filter_symbols_by_dates(
            symbol_dfs=symbol_dfs,
            start_date='2024-01-05',
            end_date='2024-01-10'
        )

        assert 'BTC' in result
        assert 'ETH' not in result
        assert 'SOL' not in result


class TestGenerateSymbolCombinations:
    """Tests for generate_symbol_combinations function."""

    def test_basic_combinations(self):
        """Test basic combination generation."""
        symbols = ['BTC', 'ETH', 'SOL', 'ADA']
        combos = generate_symbol_combinations(symbols, n_combinations=2)

        # Should have C(4,2) = 6 combinations
        assert len(combos) == 6
        assert all(len(c) == 2 for c in combos)
        assert ('BTC', 'ETH') in combos

    def test_subset_sampling(self):
        """Test combination subset sampling."""
        symbols = ['A', 'B', 'C', 'D', 'E']
        combos = generate_symbol_combinations(
            symbols,
            n_combinations=2,
            combos_subset=0.5,
            random_seed=42
        )

        # Should have roughly half of C(5,2) = 10 combinations
        assert len(combos) <= 5
        assert len(combos) >= 1

    def test_reproducibility(self):
        """Test that random seed ensures reproducibility."""
        symbols = ['A', 'B', 'C', 'D', 'E']

        combos1 = generate_symbol_combinations(
            symbols, n_combinations=2, combos_subset=0.5, random_seed=42
        )
        combos2 = generate_symbol_combinations(
            symbols, n_combinations=2, combos_subset=0.5, random_seed=42
        )

        assert combos1 == combos2

    def test_insufficient_symbols_error(self):
        """Test error when not enough symbols."""
        symbols = ['BTC']

        with pytest.raises(ValueError, match="Not enough valid symbols"):
            generate_symbol_combinations(symbols, n_combinations=2)


class TestCalculateReturnsStatistics:
    """Tests for calculate_returns_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        symbol_dfs = {
            'BTC': create_sample_ohlcv_df('BTC', '2024-01-01', num_days=10),
            'ETH': create_sample_ohlcv_df('ETH', '2024-01-01', num_days=10)
        }

        stats = calculate_returns_statistics(symbol_dfs)

        assert 'correlation_matrix' in stats
        assert 'mean_returns' in stats
        assert 'volatility' in stats
        assert 'sharpe_ratios' in stats
        assert 'max_drawdown' in stats
        assert 'covariance_matrix' in stats
        assert 'avg_pairwise_correlation' in stats

    def test_correlation_matrix_shape(self):
        """Test correlation matrix has correct shape."""
        symbol_dfs = {
            'BTC': create_sample_ohlcv_df('BTC', '2024-01-01', num_days=10),
            'ETH': create_sample_ohlcv_df('ETH', '2024-01-01', num_days=10),
            'SOL': create_sample_ohlcv_df('SOL', '2024-01-01', num_days=10)
        }

        stats = calculate_returns_statistics(symbol_dfs)

        assert stats['correlation_matrix'].shape == (3, 3)
        assert all(sym in stats['correlation_matrix'].columns for sym in ['BTC', 'ETH', 'SOL'])

    def test_empty_input(self):
        """Test handling of empty input."""
        stats = calculate_returns_statistics({})
        assert stats == {}

    def test_single_symbol(self):
        """Test statistics with single symbol."""
        symbol_dfs = {
            'BTC': create_sample_ohlcv_df('BTC', '2024-01-01', num_days=10)
        }

        stats = calculate_returns_statistics(symbol_dfs)

        assert 'mean_returns' in stats
        assert 'BTC' in stats['mean_returns'].index


class TestGenerateDatePeriods:
    """Tests for generate_date_periods function."""

    def test_basic_period_generation(self):
        """Test basic period generation."""
        dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range(365)]

        periods = generate_date_periods(
            all_dates=dates,
            period_length_days=90,
            step_days=90
        )

        assert len(periods) > 0
        assert 'period_start' in periods[0]
        assert 'period_end' in periods[0]
        assert 'future_start' in periods[0]
        assert 'future_end' in periods[0]
        assert 'has_future_data' in periods[0]

    def test_rolling_periods(self):
        """Test rolling period generation with smaller step."""
        dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range(180)]

        periods = generate_date_periods(
            all_dates=dates,
            period_length_days=30,
            step_days=10
        )

        # With 180 days, 30-day periods, and 10-day steps
        # Should have overlapping periods
        assert len(periods) > 6

    def test_insufficient_dates(self):
        """Test with insufficient dates."""
        dates = ['2024-01-01']
        periods = generate_date_periods(dates, period_length_days=90)
        assert len(periods) == 0


class TestRunMultiPeriodTradingTest:
    """Tests for the main run_multi_period_trading_test function."""

    def test_basic_run(self):
        """Test basic multi-period test run."""
        # Create sample data spanning 200 days
        symbol_dfs = {
            'BTC': create_sample_ohlcv_df('BTC', '2024-01-01', num_days=200),
            'ETH': create_sample_ohlcv_df('ETH', '2024-01-01', num_days=200),
            'SOL': create_sample_ohlcv_df('SOL', '2024-01-01', num_days=200)
        }

        all_dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
                     for i in range(200)]

        mock_test_func = create_sample_test_runner()

        results = run_multi_period_trading_test(
            symbol_dfs=symbol_dfs,
            all_dates=all_dates,
            trading_test_func=mock_test_func,
            period_length_days=30,
            step_days=30,
            future_period_days=30,
            n_combinations=2,
            combos_subset=0.5,
            verbose=False,
            random_seed=42
        )

        assert 'periods_results' in results
        assert 'summary' in results
        assert 'comparison_analysis' in results
        assert len(results['periods_results']) > 0

    def test_summary_structure(self):
        """Test that summary has correct structure."""
        symbol_dfs = {
            'BTC': create_sample_ohlcv_df('BTC', '2024-01-01', num_days=150),
            'ETH': create_sample_ohlcv_df('ETH', '2024-01-01', num_days=150)
        }

        all_dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
                     for i in range(150)]

        mock_test_func = create_sample_test_runner()

        results = run_multi_period_trading_test(
            symbol_dfs=symbol_dfs,
            all_dates=all_dates,
            trading_test_func=mock_test_func,
            period_length_days=30,
            verbose=False
        )

        summary = results['summary']
        assert 'total_periods' in summary
        assert 'successful_periods' in summary
        assert 'avg_valid_symbols' in summary

    def test_period_result_structure(self):
        """Test that individual period results have correct structure."""
        symbol_dfs = {
            'BTC': create_sample_ohlcv_df('BTC', '2024-01-01', num_days=100),
            'ETH': create_sample_ohlcv_df('ETH', '2024-01-01', num_days=100)
        }

        all_dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
                     for i in range(100)]

        mock_test_func = create_sample_test_runner()

        results = run_multi_period_trading_test(
            symbol_dfs=symbol_dfs,
            all_dates=all_dates,
            trading_test_func=mock_test_func,
            period_length_days=30,
            verbose=False
        )

        period_result = results['periods_results'][0]
        assert 'period_index' in period_result
        assert 'period_info' in period_result
        assert 'valid_symbols' in period_result
        assert 'period_stats' in period_result
        assert 'period_trading_results' in period_result

    def test_insufficient_symbols_handling(self):
        """Test handling when symbols become insufficient."""
        symbol_dfs = {
            'BTC': create_sample_ohlcv_df('BTC', '2024-01-01', num_days=30)
        }

        all_dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d')
                     for i in range(90)]

        mock_test_func = create_sample_test_runner()

        results = run_multi_period_trading_test(
            symbol_dfs=symbol_dfs,
            all_dates=all_dates,
            trading_test_func=mock_test_func,
            period_length_days=30,
            min_symbols_required=2,
            verbose=False
        )

        # Should have errors for periods with insufficient symbols
        assert any(r.get('error') for r in results['periods_results'])


class TestCreateSampleTestRunner:
    """Tests for the sample test runner."""

    def test_mock_returns_results(self):
        """Test that mock function returns proper structure."""
        mock_func = create_sample_test_runner()

        symbol_dfs = {
            'BTC': create_sample_ohlcv_df('BTC', '2024-01-01', num_days=10),
            'ETH': create_sample_ohlcv_df('ETH', '2024-01-01', num_days=10)
        }

        combos = [('BTC', 'ETH')]
        date_period = ['2024-01-01', '2024-01-10']

        results = mock_func(
            symbols_data=symbol_dfs,
            combos=combos,
            date_period=date_period,
            freq='1min'
        )

        assert len(results) == 1
        result = list(results.values())[0]
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
