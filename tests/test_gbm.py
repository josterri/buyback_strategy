import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.gbm import simulate_gbm_paths, calculate_price_statistics, calculate_envelopes


class TestGBM:
    def test_simulate_gbm_paths_shape(self):
        """Test that GBM simulation returns correct shape."""
        prices = simulate_gbm_paths(
            S0=100,
            num_days=10,
            drift_pct=0,
            volatility_pct=20,
            num_simulations=5,
            random_seed=42
        )
        assert prices.shape == (5, 10)

    def test_simulate_gbm_paths_initial_price(self):
        """Test that all simulations start at initial price."""
        S0 = 100
        prices = simulate_gbm_paths(
            S0=S0,
            num_days=10,
            drift_pct=0,
            volatility_pct=20,
            num_simulations=5,
            random_seed=42
        )
        assert np.all(prices[:, 0] == S0)

    def test_simulate_gbm_paths_reproducibility(self):
        """Test that same seed produces same results."""
        params = {
            'S0': 100,
            'num_days': 10,
            'drift_pct': 5,
            'volatility_pct': 20,
            'num_simulations': 5,
            'random_seed': 42
        }
        prices1 = simulate_gbm_paths(**params)
        prices2 = simulate_gbm_paths(**params)
        np.testing.assert_array_almost_equal(prices1, prices2)

    def test_simulate_gbm_paths_positive_prices(self):
        """Test that all prices remain positive."""
        prices = simulate_gbm_paths(
            S0=100,
            num_days=100,
            drift_pct=-50,
            volatility_pct=100,
            num_simulations=100,
            random_seed=42
        )
        assert np.all(prices > 0)

    def test_calculate_price_statistics(self):
        """Test price statistics calculation."""
        prices = np.array([
            [100, 110, 120],
            [100, 90, 80],
            [100, 100, 100]
        ])
        mean_prices, std_prices = calculate_price_statistics(prices)

        np.testing.assert_array_almost_equal(mean_prices, [100, 100, 100])
        np.testing.assert_array_almost_equal(
            std_prices,
            [0, np.std([110, 90, 100]), np.std([120, 80, 100])]
        )

    def test_calculate_envelopes(self):
        """Test envelope calculation."""
        mean_prices = np.array([100, 100, 100])
        std_prices = np.array([0, 10, 20])

        envelopes = calculate_envelopes(mean_prices, std_prices, [1, 2])

        assert '+1σ' in envelopes
        assert '-1σ' in envelopes
        assert '+2σ' in envelopes
        assert '-2σ' in envelopes

        np.testing.assert_array_almost_equal(envelopes['+1σ'], [100, 110, 120])
        np.testing.assert_array_almost_equal(envelopes['-1σ'], [100, 90, 80])
        np.testing.assert_array_almost_equal(envelopes['+2σ'], [100, 120, 140])
        np.testing.assert_array_almost_equal(envelopes['-2σ'], [100, 80, 60])

    def test_drift_effect(self):
        """Test that positive drift increases expected price."""
        prices_no_drift = simulate_gbm_paths(
            S0=100,
            num_days=252,  # One year
            drift_pct=0,
            volatility_pct=20,
            num_simulations=10000,
            random_seed=42
        )

        prices_positive_drift = simulate_gbm_paths(
            S0=100,
            num_days=252,
            drift_pct=10,  # 10% annual drift
            volatility_pct=20,
            num_simulations=10000,
            random_seed=42
        )

        # Mean final price should be higher with positive drift
        assert np.mean(prices_positive_drift[:, -1]) > np.mean(prices_no_drift[:, -1])

    def test_volatility_effect(self):
        """Test that higher volatility increases price dispersion."""
        prices_low_vol = simulate_gbm_paths(
            S0=100,
            num_days=100,
            drift_pct=0,
            volatility_pct=10,
            num_simulations=1000,
            random_seed=42
        )

        prices_high_vol = simulate_gbm_paths(
            S0=100,
            num_days=100,
            drift_pct=0,
            volatility_pct=50,
            num_simulations=1000,
            random_seed=42
        )

        # Standard deviation should be higher with higher volatility
        assert np.std(prices_high_vol[:, -1]) > np.std(prices_low_vol[:, -1])