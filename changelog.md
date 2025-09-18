# Changelog

## [1.0.0] - 2025-09-18

### Added
- Initial release of Fixed-Notional Share Buyback Strategy Streamlit application
- **Core Modules:**
  - `gbm.py`: Geometric Brownian Motion simulation engine
  - `benchmarks.py`: Benchmark calculation functions with discount support
  - `strategies.py`: Implementation of three trading strategies
  - `metrics.py`: VWAP and performance metrics calculations
  - `visualizations.py`: Plotly-based visualization functions

- **Main Application Features:**
  - Tab 1: Full simulation with multiple paths and comprehensive results
  - Tab 2: Detailed single example with execution analysis
  - Tab 3: Mathematical explanations and formulas

- **Trading Strategies:**
  - Strategy 1: Equal daily buying over target duration
  - Strategy 2: Adaptive trading based on price vs benchmark
  - Strategy 3: Adaptive trading with discounted benchmark

- **Visualizations:**
  - Stock price paths with standard deviation envelopes (±1σ to ±4σ)
  - Performance distribution histograms (250 bins)
  - Trading duration distributions
  - Benchmark vs execution scatter plots
  - Daily execution bar charts

- **Testing Suite:**
  - 43 comprehensive unit tests
  - Integration tests for full workflow
  - Reproducibility tests with random seeds

### Fixed
- Strategy 2 and 3 now properly respect max_duration boundaries
- Test suite updated to reflect correct strategy behavior with discounted benchmarks

### Technical Details
- Uses 252 trading days per year for discretization (dt = 1/252)
- VWAP calculation: Total USD spent / Total shares acquired
- Performance metrics in basis points with standard error calculation
- Supports up to 100,000 simulations
- Random seed support for reproducible results

### Dependencies
- streamlit
- plotly
- numpy
- pandas
- scipy
- pytest

### Known Issues
- None reported

### Notes
- Application runs on http://localhost:8502
- Modular architecture allows for easy future enhancements
- All monetary values default to USD
- Link to Candor Partners website included