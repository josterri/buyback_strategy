import numpy as np
from typing import Dict, Any, Tuple

def calculate_vwap(
    usd_executed: np.ndarray,
    shares_acquired: np.ndarray,
    actual_end_days: np.ndarray = None
) -> np.ndarray:
    num_simulations = usd_executed.shape[0]
    vwap = np.zeros(num_simulations)

    for sim in range(num_simulations):
        if actual_end_days is not None:
            end_day = int(actual_end_days[sim])
            total_usd = np.sum(usd_executed[sim, :end_day])
            total_shares = np.sum(shares_acquired[sim, :end_day])
        else:
            total_usd = np.sum(usd_executed[sim])
            total_shares = np.sum(shares_acquired[sim])

        if total_shares > 0:
            vwap[sim] = total_usd / total_shares
        else:
            vwap[sim] = 0

    return vwap


def calculate_benchmark_price(
    prices: np.ndarray,
    actual_end_days: np.ndarray = None,
    discount_bps: float = 0.0
) -> np.ndarray:
    num_simulations = prices.shape[0]
    benchmark_prices = np.zeros(num_simulations)

    for sim in range(num_simulations):
        if actual_end_days is not None:
            end_day = int(actual_end_days[sim])
            benchmark_prices[sim] = np.mean(prices[sim, :end_day])
        else:
            benchmark_prices[sim] = np.mean(prices[sim])

    # Apply discount if specified
    if discount_bps > 0:
        discount_factor = 1 - (discount_bps / 10000)
        benchmark_prices = benchmark_prices * discount_factor

    return benchmark_prices


def calculate_execution_performance_bps(
    vwap: np.ndarray,
    benchmark_prices: np.ndarray
) -> np.ndarray:
    # Calculate performance difference in basis points
    # Negative means execution was better than benchmark
    performance_bps = ((vwap - benchmark_prices) / benchmark_prices) * 10000
    return -performance_bps  # Flip sign so positive is better


def calculate_performance_statistics(
    performance_bps: np.ndarray
) -> Dict[str, float]:
    return {
        'mean': np.mean(performance_bps),
        'std': np.std(performance_bps),
        'std_error': np.std(performance_bps) / np.sqrt(len(performance_bps)),
        'median': np.median(performance_bps),
        'p25': np.percentile(performance_bps, 25),
        'p75': np.percentile(performance_bps, 75),
        'min': np.min(performance_bps),
        'max': np.max(performance_bps)
    }


def calculate_all_metrics(
    strategy_results: Dict[str, Any],
    prices: np.ndarray,
    benchmark_discount_bps: float = 0.0
) -> Dict[str, Dict[str, Any]]:
    metrics = {}

    for strategy_name, results in strategy_results.items():
        # Calculate VWAP
        vwap = calculate_vwap(
            results['usd_executed'],
            results['shares_acquired'],
            results['actual_end_days']
        )

        # Calculate benchmark price
        if strategy_name == 'strategy_3':
            # Strategy 3 uses discounted benchmark
            benchmark_prices = calculate_benchmark_price(
                prices,
                results['actual_end_days'],
                discount_bps=benchmark_discount_bps
            )
        else:
            # Strategies 1 and 2 use regular benchmark
            benchmark_prices = calculate_benchmark_price(
                prices,
                results['actual_end_days'],
                discount_bps=0.0
            )

        # Calculate performance in bps
        performance_bps = calculate_execution_performance_bps(vwap, benchmark_prices)

        # Calculate statistics
        stats = calculate_performance_statistics(performance_bps)

        metrics[strategy_name] = {
            'vwap': vwap,
            'benchmark_prices': benchmark_prices,
            'performance_bps': performance_bps,
            'statistics': stats,
            'actual_end_days': results['actual_end_days']
        }

    return metrics


def format_performance_summary(metrics: Dict[str, Dict[str, Any]]) -> str:
    summary_lines = []
    summary_lines.append("Performance Summary (in basis points):")
    summary_lines.append("=" * 50)

    for strategy_name, strategy_metrics in metrics.items():
        stats = strategy_metrics['statistics']
        strategy_num = strategy_name.replace('strategy_', '')
        summary_lines.append(f"Strategy {strategy_num}:")
        summary_lines.append(f"  Mean: {stats['mean']:.2f} bps +/- {stats['std_error']:.2f}")
        summary_lines.append(f"  Median: {stats['median']:.2f} bps")
        summary_lines.append(f"  Std Dev: {stats['std']:.2f} bps")
        summary_lines.append(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}] bps")
        summary_lines.append("")

    return "\n".join(summary_lines)