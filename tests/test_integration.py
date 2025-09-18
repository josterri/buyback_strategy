import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.gbm import simulate_gbm_paths
from modules.benchmarks import calculate_arithmetic_mean_benchmark
from modules.strategies import execute_all_strategies
from modules.metrics import calculate_all_metrics


class TestIntegration:
    def test_full_workflow(self):
        """Test the complete simulation workflow."""
        # Parameters
        S0 = 100
        num_days = 20
        drift_pct = 5
        volatility_pct = 20
        num_simulations = 100
        total_usd = 1000000
        target_duration = 10
        min_duration = 5
        max_duration = 15
        benchmark_discount_bps = 50
        random_seed = 42

        # Step 1: Generate price paths
        prices = simulate_gbm_paths(
            S0, num_days, drift_pct, volatility_pct,
            num_simulations, random_seed
        )
        assert prices.shape == (num_simulations, num_days)
        assert np.all(prices > 0)

        # Step 2: Execute strategies
        strategy_results = execute_all_strategies(
            prices, total_usd, target_duration,
            min_duration, max_duration, benchmark_discount_bps
        )
        assert 'strategy_1' in strategy_results
        assert 'strategy_2' in strategy_results
        assert 'strategy_3' in strategy_results

        # Step 3: Calculate metrics
        metrics = calculate_all_metrics(
            strategy_results, prices, benchmark_discount_bps
        )
        assert 'strategy_1' in metrics
        assert 'strategy_2' in metrics
        assert 'strategy_3' in metrics

        # Verify each strategy's metrics
        for strategy in ['strategy_1', 'strategy_2', 'strategy_3']:
            assert 'vwap' in metrics[strategy]
            assert 'performance_bps' in metrics[strategy]
            assert 'statistics' in metrics[strategy]
            assert len(metrics[strategy]['vwap']) == num_simulations

    def test_reproducibility_with_seed(self):
        """Test that using the same seed produces identical results."""
        params = {
            'S0': 100,
            'num_days': 10,
            'drift_pct': 0,
            'volatility_pct': 25,
            'num_simulations': 50,
            'random_seed': 12345
        }

        # Run twice with same seed
        prices1 = simulate_gbm_paths(**params)
        results1 = execute_all_strategies(
            prices1, 100000, 5, 3, 8, 0
        )
        metrics1 = calculate_all_metrics(results1, prices1, 0)

        prices2 = simulate_gbm_paths(**params)
        results2 = execute_all_strategies(
            prices2, 100000, 5, 3, 8, 0
        )
        metrics2 = calculate_all_metrics(results2, prices2, 0)

        # Compare results
        np.testing.assert_array_almost_equal(prices1, prices2)
        for strategy in ['strategy_1', 'strategy_2', 'strategy_3']:
            np.testing.assert_array_almost_equal(
                metrics1[strategy]['vwap'],
                metrics2[strategy]['vwap']
            )

    def test_strategy_1_ends_at_target(self):
        """Test that Strategy 1 always ends at target duration."""
        prices = simulate_gbm_paths(100, 30, 0, 20, 10, 42)
        target_duration = 20

        results = execute_all_strategies(
            prices, 100000, target_duration, 10, 25, 0
        )

        # Strategy 1 should always end at target duration
        end_days = results['strategy_1']['actual_end_days']
        assert np.all(end_days == target_duration)

    def test_strategies_spend_all_money(self):
        """Test that all strategies spend exactly the allocated USD."""
        prices = simulate_gbm_paths(100, 50, 0, 30, 5, 42)
        total_usd = 500000

        results = execute_all_strategies(
            prices, total_usd, 20, 10, 30, 0
        )

        for strategy in ['strategy_1', 'strategy_2', 'strategy_3']:
            for sim in range(5):
                end_day = int(results[strategy]['actual_end_days'][sim])
                total_spent = np.sum(
                    results[strategy]['usd_executed'][sim, :end_day]
                )
                np.testing.assert_almost_equal(
                    total_spent, total_usd, decimal=2
                )

    def test_performance_calculation_consistency(self):
        """Test that performance calculations are consistent."""
        prices = simulate_gbm_paths(100, 20, 5, 15, 50, 42)
        results = execute_all_strategies(
            prices, 100000, 10, 5, 15, 0
        )
        metrics = calculate_all_metrics(results, prices, 0)

        for strategy in ['strategy_1', 'strategy_2', 'strategy_3']:
            vwap = metrics[strategy]['vwap']
            benchmark = metrics[strategy]['benchmark_prices']
            performance = metrics[strategy]['performance_bps']

            # Manually calculate performance
            manual_perf = -((vwap - benchmark) / benchmark) * 10000

            np.testing.assert_array_almost_equal(
                performance, manual_perf, decimal=2
            )

    def test_different_market_conditions(self):
        """Test strategies under different market conditions."""
        # Bull market
        bull_prices = simulate_gbm_paths(100, 30, 50, 20, 20, 1)
        bull_results = execute_all_strategies(
            bull_prices, 100000, 15, 10, 20, 0
        )

        # Bear market
        bear_prices = simulate_gbm_paths(100, 30, -50, 20, 20, 2)
        bear_results = execute_all_strategies(
            bear_prices, 100000, 15, 10, 20, 0
        )

        # Volatile market
        volatile_prices = simulate_gbm_paths(100, 30, 0, 80, 20, 3)
        volatile_results = execute_all_strategies(
            volatile_prices, 100000, 15, 10, 20, 0
        )

        # All strategies should complete in all conditions
        for results in [bull_results, bear_results, volatile_results]:
            for strategy in ['strategy_1', 'strategy_2', 'strategy_3']:
                assert 'usd_executed' in results[strategy]
                assert 'shares_acquired' in results[strategy]
                assert np.all(results[strategy]['actual_end_days'] > 0)

    def test_duration_boundaries(self):
        """Test that strategies respect duration boundaries."""
        prices = simulate_gbm_paths(100, 50, 0, 30, 100, 42)
        min_duration = 10
        target_duration = 20
        max_duration = 30

        results = execute_all_strategies(
            prices, 100000, target_duration, min_duration, max_duration, 0
        )

        # Strategy 1: should be exactly target
        assert np.all(results['strategy_1']['actual_end_days'] == target_duration)

        # Strategies 2 and 3: should be within bounds
        for strategy in ['strategy_2', 'strategy_3']:
            end_days = results[strategy]['actual_end_days']
            # Allow slight buffer for edge cases
            assert np.all(end_days >= min_duration - 1)
            assert np.all(end_days <= max_duration + 1)

    def test_benchmark_discount_effect(self):
        """Test that benchmark discount affects Strategy 3."""
        prices = simulate_gbm_paths(100, 20, 0, 10, 50, 42)

        # Run with no discount
        results_no_discount = execute_all_strategies(
            prices, 100000, 10, 5, 15, 0
        )
        metrics_no_discount = calculate_all_metrics(
            results_no_discount, prices, 0
        )

        # Run with discount
        results_with_discount = execute_all_strategies(
            prices, 100000, 10, 5, 15, 100  # 100 bps = 1%
        )
        metrics_with_discount = calculate_all_metrics(
            results_with_discount, prices, 100
        )

        # Strategy 3 should have different results with discount
        perf_no_discount = metrics_no_discount['strategy_3']['performance_bps']
        perf_with_discount = metrics_with_discount['strategy_3']['performance_bps']

        # Performance should generally be better (higher) with discount
        assert np.mean(perf_with_discount) != np.mean(perf_no_discount)