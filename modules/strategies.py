import numpy as np
from typing import Tuple, Optional, Dict, Any
from .benchmarks import calculate_arithmetic_mean_benchmark

def execute_strategy_1(
    prices: np.ndarray,
    total_usd: float,
    target_duration: int,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    num_simulations, num_days = prices.shape
    daily_usd = total_usd / target_duration

    # Arrays to store results
    usd_executed = np.zeros((num_simulations, num_days))
    shares_acquired = np.zeros((num_simulations, num_days))
    cumulative_shares = np.zeros((num_simulations, num_days))

    # Execute strategy for each simulation
    for sim in range(num_simulations):
        for day in range(min(target_duration, num_days)):
            usd_executed[sim, day] = daily_usd
            shares_acquired[sim, day] = daily_usd / prices[sim, day]
            if day == 0:
                cumulative_shares[sim, day] = shares_acquired[sim, day]
            else:
                cumulative_shares[sim, day] = cumulative_shares[sim, day-1] + shares_acquired[sim, day]

    actual_end_day = min(target_duration, num_days)
    return usd_executed, shares_acquired, cumulative_shares, actual_end_day


def execute_strategy_2(
    prices: np.ndarray,
    total_usd: float,
    target_duration: int,
    min_duration: int,
    max_duration: int,
    benchmark: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_simulations, num_days = prices.shape

    if benchmark is None:
        benchmark = calculate_arithmetic_mean_benchmark(prices, discount_bps=0.0)

    # Arrays to store results
    usd_executed = np.zeros((num_simulations, num_days))
    shares_acquired = np.zeros((num_simulations, num_days))
    cumulative_shares = np.zeros((num_simulations, num_days))
    actual_end_days = np.zeros(num_simulations, dtype=int)

    # Initial daily value (as if trading over target duration)
    initial_daily_usd = total_usd / target_duration

    for sim in range(num_simulations):
        remaining_usd = total_usd
        first_day_value = initial_daily_usd

        for day in range(min(num_days, max_duration)):  # Never exceed max_duration
            if remaining_usd <= 0.01:  # Use small threshold for floating point
                actual_end_days[sim] = day
                break

            # If we're at max_duration - 1, spend all remaining money
            if day == max_duration - 1:
                daily_trade = remaining_usd
            else:
                current_price = prices[sim, day]
                current_benchmark = benchmark[day] if day < len(benchmark) else benchmark[-1]

                # Determine trading amount based on strategy rules
                if day < min(10, min_duration):
                    # First 10 days or min_duration (whichever is smaller): trade at target rate
                    daily_trade = first_day_value
                else:
                    # After day 10 or min_duration
                    days_left = max_duration - day  # Use max_duration as upper bound

                    if current_price < current_benchmark:
                        # Price below benchmark: speed up
                        if day >= min_duration:
                            # Beyond min duration: finish as quickly as possible
                            daily_trade = min(remaining_usd, first_day_value * 5)
                        else:
                            # Before min duration: speed up to finish by min duration
                            days_to_min = max(1, min_duration - day)
                            daily_trade = remaining_usd / days_to_min
                    else:
                        # Price above benchmark: slow down
                        days_to_max = max(1, max_duration - day)
                        base_daily_trade = remaining_usd / days_to_max

                        # Special condition: if remaining value < 5x first day and >5 days left
                        if remaining_usd < 5 * first_day_value and days_left > 5:
                            daily_trade = min(base_daily_trade, 0.1 * first_day_value)
                        else:
                            daily_trade = base_daily_trade

            # Execute trade
            daily_trade = min(daily_trade, remaining_usd)
            usd_executed[sim, day] = daily_trade
            shares_acquired[sim, day] = daily_trade / prices[sim, day]

            if day == 0:
                cumulative_shares[sim, day] = shares_acquired[sim, day]
            else:
                cumulative_shares[sim, day] = cumulative_shares[sim, day-1] + shares_acquired[sim, day]

            remaining_usd -= daily_trade

        # If we haven't set the end day yet, set it
        if actual_end_days[sim] == 0:
            # Should end at max_duration or when money ran out
            actual_end_days[sim] = min(max_duration, num_days)

    return usd_executed, shares_acquired, cumulative_shares, actual_end_days


def execute_strategy_3(
    prices: np.ndarray,
    total_usd: float,
    target_duration: int,
    min_duration: int,
    max_duration: int,
    benchmark_discount_bps: float = 0.0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Strategy 3 is same as Strategy 2 but with discounted benchmark
    benchmark = calculate_arithmetic_mean_benchmark(prices, discount_bps=benchmark_discount_bps)

    return execute_strategy_2(
        prices=prices,
        total_usd=total_usd,
        target_duration=target_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        benchmark=benchmark
    )


def execute_all_strategies(
    prices: np.ndarray,
    total_usd: float,
    target_duration: int,
    min_duration: int,
    max_duration: int,
    benchmark_discount_bps: float = 0.0
) -> Dict[str, Any]:
    results = {}

    # Strategy 1
    usd1, shares1, cum_shares1, end_day1 = execute_strategy_1(
        prices, total_usd, target_duration
    )
    results['strategy_1'] = {
        'usd_executed': usd1,
        'shares_acquired': shares1,
        'cumulative_shares': cum_shares1,
        'actual_end_days': np.full(prices.shape[0], end_day1)
    }

    # Strategy 2
    usd2, shares2, cum_shares2, end_days2 = execute_strategy_2(
        prices, total_usd, target_duration, min_duration, max_duration
    )
    results['strategy_2'] = {
        'usd_executed': usd2,
        'shares_acquired': shares2,
        'cumulative_shares': cum_shares2,
        'actual_end_days': end_days2
    }

    # Strategy 3
    usd3, shares3, cum_shares3, end_days3 = execute_strategy_3(
        prices, total_usd, target_duration, min_duration, max_duration,
        benchmark_discount_bps
    )
    results['strategy_3'] = {
        'usd_executed': usd3,
        'shares_acquired': shares3,
        'cumulative_shares': cum_shares3,
        'actual_end_days': end_days3
    }

    return results