import numpy as np
from typing import Optional, Tuple

def simulate_gbm_paths(
    S0: float,
    num_days: int,
    drift_pct: float,
    volatility_pct: float,
    num_simulations: int,
    random_seed: Optional[int] = None
) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(random_seed)

    # Convert percentages to decimals
    mu = drift_pct / 100
    sigma = volatility_pct / 100

    # Time step (assuming 252 trading days per year)
    dt = 1 / 252

    # Initialize price array - shape: (num_simulations, num_days)
    prices = np.zeros((num_simulations, num_days))
    prices[:, 0] = S0

    # Generate random shocks for all simulations and time steps
    Z = np.random.standard_normal((num_simulations, num_days - 1))

    # Simulate GBM paths
    for t in range(1, num_days):
        prices[:, t] = prices[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
        )

    return prices


def calculate_price_statistics(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean_prices = np.mean(prices, axis=0)
    std_prices = np.std(prices, axis=0)
    return mean_prices, std_prices


def calculate_envelopes(
    mean_prices: np.ndarray,
    std_prices: np.ndarray,
    n_sigma_levels: list = [1, 2, 3, 4]
) -> dict:
    envelopes = {}
    for n in n_sigma_levels:
        envelopes[f'+{n}σ'] = mean_prices + n * std_prices
        envelopes[f'-{n}σ'] = mean_prices - n * std_prices
    return envelopes