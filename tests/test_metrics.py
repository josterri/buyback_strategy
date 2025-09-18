import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.metrics import (
    calculate_vwap,
    calculate_benchmark_price,
    calculate_execution_performance_bps,
    calculate_performance_statistics,
    calculate_all_metrics,
    format_performance_summary
)


class TestMetrics:
    def test_vwap_calculation(self):
        """Test VWAP calculation."""
        # Simple case: buy 1 share at $100, 2 shares at $200
        usd_executed = np.array([[100, 400, 0]])
        shares_acquired = np.array([[1, 2, 0]])

        vwap = calculate_vwap(usd_executed, shares_acquired)

        # VWAP = (100 + 400) / (1 + 2) = 500 / 3 = 166.67
        np.testing.assert_almost_equal(vwap[0], 166.67, decimal=2)

    def test_vwap_with_actual_end_days(self):
        """Test VWAP calculation with actual end days."""
        usd_executed = np.array([[100, 200, 300, 400]])
        shares_acquired = np.array([[1, 2, 3, 4]])
        actual_end_days = np.array([2])  # Only use first 2 days

        vwap = calculate_vwap(usd_executed, shares_acquired, actual_end_days)

        # VWAP = (100 + 200) / (1 + 2) = 300 / 3 = 100
        np.testing.assert_almost_equal(vwap[0], 100)

    def test_benchmark_price_calculation(self):
        """Test benchmark price calculation."""
        prices = np.array([[100, 110, 120, 130]])

        benchmark = calculate_benchmark_price(prices)

        # Arithmetic mean = (100 + 110 + 120 + 130) / 4 = 115
        np.testing.assert_almost_equal(benchmark[0], 115)

    def test_benchmark_price_with_discount(self):
        """Test benchmark price with discount."""
        prices = np.array([[100, 100, 100]])

        # 100 bps = 1% discount
        benchmark = calculate_benchmark_price(prices, discount_bps=100)

        # Mean = 100, with 1% discount = 99
        np.testing.assert_almost_equal(benchmark[0], 99)

    def test_execution_performance_bps(self):
        """Test execution performance calculation in basis points."""
        vwap = np.array([102, 98, 100])
        benchmark_prices = np.array([100, 100, 100])

        performance_bps = calculate_execution_performance_bps(vwap, benchmark_prices)

        # Performance = -((vwap - benchmark) / benchmark) * 10000
        # For vwap=102, benchmark=100: -((102-100)/100) * 10000 = -200 bps (worse)
        # For vwap=98, benchmark=100: -((98-100)/100) * 10000 = 200 bps (better)
        # For vwap=100, benchmark=100: 0 bps (equal)
        np.testing.assert_almost_equal(performance_bps[0], -200)
        np.testing.assert_almost_equal(performance_bps[1], 200)
        np.testing.assert_almost_equal(performance_bps[2], 0)

    def test_performance_statistics(self):
        """Test performance statistics calculation."""
        performance_bps = np.array([100, 200, 300, 400, 500])

        stats = calculate_performance_statistics(performance_bps)

        assert stats['mean'] == 300
        assert stats['median'] == 300
        assert stats['min'] == 100
        assert stats['max'] == 500
        assert stats['p25'] == 200
        assert stats['p75'] == 400
        np.testing.assert_almost_equal(
            stats['std_error'],
            np.std(performance_bps) / np.sqrt(5)
        )

    def test_calculate_all_metrics(self):
        """Test comprehensive metrics calculation."""
        # Create simple strategy results
        strategy_results = {
            'strategy_1': {
                'usd_executed': np.array([[100, 100, 0]]),
                'shares_acquired': np.array([[1, 1, 0]]),
                'actual_end_days': np.array([2])
            }
        }

        prices = np.array([[100, 100, 100]])

        metrics = calculate_all_metrics(strategy_results, prices, benchmark_discount_bps=0)

        assert 'strategy_1' in metrics
        assert 'vwap' in metrics['strategy_1']
        assert 'benchmark_prices' in metrics['strategy_1']
        assert 'performance_bps' in metrics['strategy_1']
        assert 'statistics' in metrics['strategy_1']

        # VWAP should be 100 (200 USD / 2 shares)
        np.testing.assert_almost_equal(metrics['strategy_1']['vwap'][0], 100)

        # Benchmark should be 100 (mean of first 2 prices)
        np.testing.assert_almost_equal(metrics['strategy_1']['benchmark_prices'][0], 100)

        # Performance should be 0 bps (VWAP = benchmark)
        np.testing.assert_almost_equal(metrics['strategy_1']['performance_bps'][0], 0)

    def test_format_performance_summary(self):
        """Test performance summary formatting."""
        metrics = {
            'strategy_1': {
                'statistics': {
                    'mean': 100.5,
                    'std_error': 10.2,
                    'median': 99.8,
                    'std': 50.3,
                    'min': -100.5,
                    'max': 200.7
                }
            }
        }

        summary = format_performance_summary(metrics)

        assert "Strategy 1:" in summary
        assert "100.50 bps" in summary  # Mean
        assert "10.20" in summary  # Std error
        assert "99.80 bps" in summary  # Median
        assert "50.30 bps" in summary  # Std dev
        assert "-100.50" in summary  # Min
        assert "200.70" in summary  # Max

    def test_vwap_zero_shares(self):
        """Test VWAP calculation with zero shares."""
        usd_executed = np.array([[0, 0, 0]])
        shares_acquired = np.array([[0, 0, 0]])

        vwap = calculate_vwap(usd_executed, shares_acquired)

        # Should return 0 when no shares acquired
        assert vwap[0] == 0

    def test_performance_with_multiple_simulations(self):
        """Test metrics with multiple simulations."""
        strategy_results = {
            'strategy_1': {
                'usd_executed': np.array([[100, 100], [200, 200], [150, 150]]),
                'shares_acquired': np.array([[1, 2], [2, 1], [1.5, 1.5]]),
                'actual_end_days': np.array([2, 2, 2])
            }
        }

        prices = np.array([[100, 110], [90, 100], [95, 105]])

        metrics = calculate_all_metrics(strategy_results, prices, 0)

        # Check that we have metrics for all 3 simulations
        assert len(metrics['strategy_1']['vwap']) == 3
        assert len(metrics['strategy_1']['performance_bps']) == 3

        # Check that statistics are calculated across all simulations
        stats = metrics['strategy_1']['statistics']
        assert 'mean' in stats
        assert 'std' in stats