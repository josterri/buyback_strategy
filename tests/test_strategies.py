import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.strategies import (
    execute_strategy_1,
    execute_strategy_2,
    execute_strategy_3,
    execute_all_strategies
)


class TestStrategies:
    def test_strategy_1_equal_daily_buying(self):
        """Test that Strategy 1 buys equal amounts each day."""
        prices = np.array([[100, 100, 100, 100, 100]])
        total_usd = 1000
        target_duration = 5

        usd_executed, shares_acquired, cumulative_shares, actual_end_day = execute_strategy_1(
            prices, total_usd, target_duration
        )

        # Should buy $200 each day
        expected_daily_usd = total_usd / target_duration
        for day in range(target_duration):
            assert usd_executed[0, day] == expected_daily_usd

        # Should end exactly at target duration
        assert actual_end_day == target_duration

        # Total USD should be fully spent
        assert np.sum(usd_executed[0, :actual_end_day]) == total_usd

    def test_strategy_1_shares_calculation(self):
        """Test that Strategy 1 correctly calculates shares."""
        prices = np.array([[100, 200, 50]])
        total_usd = 300
        target_duration = 3

        usd_executed, shares_acquired, cumulative_shares, actual_end_day = execute_strategy_1(
            prices, total_usd, target_duration
        )

        # Day 1: $100 at $100 = 1 share
        assert shares_acquired[0, 0] == 1

        # Day 2: $100 at $200 = 0.5 shares
        assert shares_acquired[0, 1] == 0.5

        # Day 3: $100 at $50 = 2 shares
        assert shares_acquired[0, 2] == 2

        # Cumulative shares
        np.testing.assert_array_almost_equal(
            cumulative_shares[0, :3],
            [1, 1.5, 3.5]
        )

    def test_strategy_2_basic_logic(self):
        """Test Strategy 2 basic trading logic."""
        # Create a simple price path
        prices = np.array([[100, 95, 90, 110, 120, 100, 95, 90, 85, 80]])
        total_usd = 1000
        target_duration = 5
        min_duration = 3
        max_duration = 8

        usd_executed, shares_acquired, cumulative_shares, actual_end_days = execute_strategy_2(
            prices, total_usd, target_duration, min_duration, max_duration
        )

        # Check that total USD is spent
        total_spent = np.sum(usd_executed[0, :int(actual_end_days[0])])
        np.testing.assert_almost_equal(total_spent, total_usd, decimal=2)

        # Check that trading stops at actual end day
        assert usd_executed[0, int(actual_end_days[0])] == 0

    def test_strategy_2_acceleration(self):
        """Test that Strategy 2 accelerates when price < benchmark."""
        # Declining price path - should trigger acceleration
        prices = np.array([[100, 90, 80, 70, 60, 50, 40, 30, 20, 10]])
        total_usd = 1000
        target_duration = 8
        min_duration = 4
        max_duration = 10

        usd_executed, shares_acquired, cumulative_shares, actual_end_days = execute_strategy_2(
            prices, total_usd, target_duration, min_duration, max_duration
        )

        # Should finish before or at min duration due to declining prices
        assert actual_end_days[0] <= min_duration + 1  # Allow small buffer

    def test_strategy_2_deceleration(self):
        """Test that Strategy 2 decelerates when price > benchmark."""
        # Rising price path - should trigger deceleration
        prices = np.array([[100, 110, 120, 130, 140, 150, 160, 170, 180, 190]])
        total_usd = 1000
        target_duration = 5
        min_duration = 3
        max_duration = 10

        usd_executed, shares_acquired, cumulative_shares, actual_end_days = execute_strategy_2(
            prices, total_usd, target_duration, min_duration, max_duration
        )

        # Should take longer than target due to rising prices
        assert actual_end_days[0] >= target_duration

    def test_strategy_3_uses_discounted_benchmark(self):
        """Test that Strategy 3 uses discounted benchmark."""
        prices = np.array([[100, 100, 100, 100, 100, 100, 100]])  # Need more days for max_duration
        total_usd = 1000
        target_duration = 5
        min_duration = 3
        max_duration = 7
        benchmark_discount_bps = 100  # 1% discount

        usd_executed, shares_acquired, cumulative_shares, actual_end_days = execute_strategy_3(
            prices, total_usd, target_duration, min_duration, max_duration,
            benchmark_discount_bps
        )

        # With constant prices at 100 and discounted benchmark at 99,
        # price (100) > benchmark (99), which should trigger deceleration
        # Strategy should slow down and extend toward max_duration

        # Total USD should still be spent (just over more days)
        total_spent = np.sum(usd_executed[0, :int(actual_end_days[0])])
        np.testing.assert_almost_equal(total_spent, total_usd, decimal=2)

        # Should take longer than target duration due to deceleration
        assert actual_end_days[0] >= target_duration

    def test_execute_all_strategies(self):
        """Test executing all strategies together."""
        prices = np.array([[100, 110, 90, 100, 105]])
        total_usd = 1000
        target_duration = 5
        min_duration = 3
        max_duration = 7
        benchmark_discount_bps = 50

        results = execute_all_strategies(
            prices, total_usd, target_duration, min_duration, max_duration,
            benchmark_discount_bps
        )

        # Check that all strategies are present
        assert 'strategy_1' in results
        assert 'strategy_2' in results
        assert 'strategy_3' in results

        # Check that each strategy has required fields
        for strategy in ['strategy_1', 'strategy_2', 'strategy_3']:
            assert 'usd_executed' in results[strategy]
            assert 'shares_acquired' in results[strategy]
            assert 'cumulative_shares' in results[strategy]
            assert 'actual_end_days' in results[strategy]

    def test_strategy_total_usd_conservation(self):
        """Test that all strategies spend exactly the total USD."""
        prices = np.array([[100, 105, 95, 110, 90, 100, 95, 105]])
        total_usd = 10000
        target_duration = 5
        min_duration = 3
        max_duration = 8

        # Test Strategy 1
        usd1, _, _, end1 = execute_strategy_1(prices, total_usd, target_duration)
        assert np.abs(np.sum(usd1[0, :end1]) - total_usd) < 0.01

        # Test Strategy 2
        usd2, _, _, end2 = execute_strategy_2(
            prices, total_usd, target_duration, min_duration, max_duration
        )
        assert np.abs(np.sum(usd2[0, :int(end2[0])]) - total_usd) < 0.01

        # Test Strategy 3
        usd3, _, _, end3 = execute_strategy_3(
            prices, total_usd, target_duration, min_duration, max_duration, 50
        )
        assert np.abs(np.sum(usd3[0, :int(end3[0])]) - total_usd) < 0.01

    def test_strategy_boundaries(self):
        """Test that strategies respect duration boundaries."""
        prices = np.random.uniform(80, 120, (1, 20))
        total_usd = 10000
        target_duration = 10
        min_duration = 5
        max_duration = 15

        results = execute_all_strategies(
            prices, total_usd, target_duration, min_duration, max_duration, 0
        )

        # Strategy 1 should end exactly at target
        assert results['strategy_1']['actual_end_days'][0] == target_duration

        # Strategies 2 and 3 should be within min and max
        for strategy in ['strategy_2', 'strategy_3']:
            end_day = results[strategy]['actual_end_days'][0]
            assert min_duration <= end_day <= max_duration