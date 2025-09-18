import numpy as np
from typing import Optional

def calculate_arithmetic_mean_benchmark(
    prices: np.ndarray,
    discount_bps: float = 0.0
) -> np.ndarray:
    # Calculate expanding window arithmetic mean
    num_days = prices.shape[1]
    benchmark = np.zeros(num_days)

    for day in range(num_days):
        # Arithmetic mean from day 1 to current day (inclusive)
        benchmark[day] = np.mean(prices[:, :day+1])

    # Apply discount if specified
    if discount_bps > 0:
        discount_factor = 1 - (discount_bps / 10000)  # Convert bps to decimal
        benchmark = benchmark * discount_factor

    return benchmark


def calculate_benchmark_for_strategy(
    prices: np.ndarray,
    strategy: int,
    discount_bps: float = 0.0
) -> np.ndarray:
    if strategy == 1:
        # Strategy 1 doesn't use benchmark for trading decisions
        return None
    elif strategy == 2:
        # Strategy 2: arithmetic mean without discount
        return calculate_arithmetic_mean_benchmark(prices, discount_bps=0.0)
    elif strategy == 3:
        # Strategy 3: arithmetic mean with discount
        return calculate_arithmetic_mean_benchmark(prices, discount_bps=discount_bps)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def apply_discount_to_benchmark(
    benchmark: np.ndarray,
    discount_bps: float
) -> np.ndarray:
    if discount_bps == 0:
        return benchmark
    discount_factor = 1 - (discount_bps / 10000)
    return benchmark * discount_factor