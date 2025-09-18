import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.benchmarks import (
    calculate_arithmetic_mean_benchmark,
    calculate_benchmark_for_strategy,
    apply_discount_to_benchmark
)


class TestBenchmarks:
    def test_arithmetic_mean_benchmark(self):
        """Test arithmetic mean benchmark calculation."""
        prices = np.array([
            [100, 110, 120],
            [100, 90, 80],
            [100, 100, 100]
        ])

        benchmark = calculate_arithmetic_mean_benchmark(prices, discount_bps=0)

        # Day 1: mean of all day 1 prices = 100
        # Day 2: mean of all day 1-2 prices = (100+110+100+90+100+100)/6 = 100
        # Day 3: mean of all day 1-3 prices = (100+110+120+100+90+80+100+100+100)/9 = 100
        expected_day1 = np.mean(prices[:, :1])
        expected_day2 = np.mean(prices[:, :2])
        expected_day3 = np.mean(prices[:, :3])

        np.testing.assert_almost_equal(benchmark[0], expected_day1)
        np.testing.assert_almost_equal(benchmark[1], expected_day2)
        np.testing.assert_almost_equal(benchmark[2], expected_day3)

    def test_benchmark_with_discount(self):
        """Test benchmark calculation with discount."""
        prices = np.array([
            [100, 100, 100],
            [100, 100, 100]
        ])

        # 100 bps = 1% discount
        benchmark = calculate_arithmetic_mean_benchmark(prices, discount_bps=100)

        # All prices are 100, so benchmark should be 99 (1% discount)
        np.testing.assert_array_almost_equal(benchmark, [99, 99, 99])

    def test_benchmark_for_strategy_1(self):
        """Test that strategy 1 returns None for benchmark."""
        prices = np.array([[100, 110, 120]])

        benchmark = calculate_benchmark_for_strategy(
            prices, strategy=1, discount_bps=0
        )

        assert benchmark is None

    def test_benchmark_for_strategy_2(self):
        """Test that strategy 2 uses no discount."""
        prices = np.array([[100, 110, 120]])

        benchmark = calculate_benchmark_for_strategy(
            prices, strategy=2, discount_bps=100  # This should be ignored
        )

        expected = calculate_arithmetic_mean_benchmark(prices, discount_bps=0)
        np.testing.assert_array_almost_equal(benchmark, expected)

    def test_benchmark_for_strategy_3(self):
        """Test that strategy 3 uses discount."""
        prices = np.array([[100, 100, 100]])

        benchmark = calculate_benchmark_for_strategy(
            prices, strategy=3, discount_bps=50  # 0.5% discount
        )

        # All prices are 100, with 0.5% discount = 99.5
        np.testing.assert_array_almost_equal(benchmark, [99.5, 99.5, 99.5])

    def test_apply_discount_to_benchmark(self):
        """Test applying discount to existing benchmark."""
        benchmark = np.array([100, 110, 120])

        # No discount
        result = apply_discount_to_benchmark(benchmark, 0)
        np.testing.assert_array_almost_equal(result, benchmark)

        # 100 bps = 1% discount
        result = apply_discount_to_benchmark(benchmark, 100)
        np.testing.assert_array_almost_equal(result, [99, 108.9, 118.8])

    def test_benchmark_expanding_window(self):
        """Test that benchmark uses expanding window correctly."""
        prices = np.array([[100, 120, 80, 110, 90]])

        benchmark = calculate_arithmetic_mean_benchmark(prices, discount_bps=0)

        # Day 1: mean of day 1 = 100
        assert benchmark[0] == 100

        # Day 2: mean of days 1-2 = (100 + 120) / 2 = 110
        assert benchmark[1] == 110

        # Day 3: mean of days 1-3 = (100 + 120 + 80) / 3 = 100
        assert benchmark[2] == 100

        # Day 4: mean of days 1-4 = (100 + 120 + 80 + 110) / 4 = 102.5
        assert benchmark[3] == 102.5

        # Day 5: mean of days 1-5 = (100 + 120 + 80 + 110 + 90) / 5 = 100
        assert benchmark[4] == 100

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        prices = np.array([[100, 110, 120]])

        with pytest.raises(ValueError):
            calculate_benchmark_for_strategy(prices, strategy=4, discount_bps=0)