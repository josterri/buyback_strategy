import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.gbm import simulate_gbm_paths, calculate_price_statistics, calculate_envelopes
from modules.benchmarks import calculate_arithmetic_mean_benchmark
from modules.strategies import execute_all_strategies
from modules.metrics import (
    calculate_vwap,
    calculate_benchmark_price,
    calculate_execution_performance_bps,
    calculate_performance_statistics,
    calculate_all_metrics,
    format_performance_summary
)
from modules.visualizations import (
    create_price_paths_chart,
    create_performance_histogram,
    create_duration_histogram,
    create_benchmark_vs_execution_chart,
    create_example_execution_chart,
    create_daily_execution_chart,
    create_summary_table
)

# Page configuration
st.set_page_config(
    page_title="Fixed-Notional Share Buyback Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and subtitle
st.title("A Fixed-Notional Share Buyback Strategy")
st.subheader("The value of flexible end times and an easy benchmark")

# Add link to Candor Partners
st.markdown("[Visit Candor Partners](https://www.candorpartners.net)")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Simulation & Results", "Example", "Explanation"])

# Tab 1: Simulation & Results
with tab1:
    with st.sidebar:
        st.header("Simulation Parameters")

        # Run Simulation button at the top
        run_simulation = st.button("Run Simulation", key="run_sim_button", type="primary")

        # Input parameters
        initial_price = st.number_input(
            "Initial stock price ($)",
            min_value=1,
            max_value=200,
            value=100,
            step=1,
            key="initial_price_sim"
        )

        num_days = st.slider(
            "Number of days (X)",
            min_value=5,
            max_value=300,
            value=125,
            step=5,
            key="num_days_sim"
        )

        drift = st.number_input(
            "Drift μ (%)",
            min_value=-100,
            max_value=100,
            value=0,
            step=1,
            key="drift_sim"
        )

        volatility = st.number_input(
            "Volatility σ (%)",
            min_value=0,
            max_value=100,
            value=25,
            step=1,
            key="volatility_sim"
        )

        num_simulations = st.number_input(
            "Number of simulations (Z)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            key="num_simulations_sim"
        )

        usd_to_execute = st.number_input(
            "USD to execute",
            min_value=1000000,
            max_value=10000000000,
            value=1000000000,
            step=1000000,
            key="usd_to_execute_sim"
        )

        max_duration = st.number_input(
            "Max duration (days)",
            min_value=1,
            max_value=300,
            value=125,
            step=1,
            key="max_duration_sim"
        )

        # Calculate default min duration
        default_min_duration = int(max_duration * 0.6)
        min_duration = st.number_input(
            "Min duration (days)",
            min_value=1,
            max_value=300,
            value=default_min_duration,
            step=1,
            key="min_duration_sim"
        )

        # Calculate default target duration
        default_target_duration = int((min_duration + max_duration) / 2)
        target_duration = st.number_input(
            "Target duration (days)",
            min_value=1,
            max_value=300,
            value=default_target_duration,
            step=1,
            key="target_duration_sim"
        )

        benchmark_discount_bps = st.number_input(
            "Benchmark discount (bps)",
            min_value=0,
            max_value=200,
            value=0,
            step=1,
            key="benchmark_discount_sim"
        )

        random_seed = st.number_input(
            "Random seed (optional)",
            min_value=0,
            max_value=999999,
            value=None,
            step=1,
            key="random_seed_sim",
            help="Leave empty for random results"
        )

    # Main content area
    if run_simulation or 'simulation_results' in st.session_state:
        if run_simulation:
            with st.spinner("Running simulations..."):
                # Simulate price paths
                prices = simulate_gbm_paths(
                    S0=initial_price,
                    num_days=num_days,
                    drift_pct=drift,
                    volatility_pct=volatility,
                    num_simulations=num_simulations,
                    random_seed=random_seed
                )

                # Calculate statistics and envelopes
                mean_prices, std_prices = calculate_price_statistics(prices)
                envelopes = calculate_envelopes(mean_prices, std_prices)

                # Execute all strategies
                strategy_results = execute_all_strategies(
                    prices=prices,
                    total_usd=usd_to_execute,
                    target_duration=target_duration,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    benchmark_discount_bps=benchmark_discount_bps
                )

                # Calculate metrics
                metrics = calculate_all_metrics(
                    strategy_results=strategy_results,
                    prices=prices,
                    benchmark_discount_bps=benchmark_discount_bps
                )

                # Store results in session state
                st.session_state['simulation_results'] = {
                    'prices': prices,
                    'mean_prices': mean_prices,
                    'std_prices': std_prices,
                    'envelopes': envelopes,
                    'strategy_results': strategy_results,
                    'metrics': metrics,
                    'parameters': {
                        'initial_price': initial_price,
                        'num_days': num_days,
                        'drift': drift,
                        'volatility': volatility,
                        'num_simulations': num_simulations,
                        'usd_to_execute': usd_to_execute,
                        'min_duration': min_duration,
                        'target_duration': target_duration,
                        'max_duration': max_duration,
                        'benchmark_discount_bps': benchmark_discount_bps,
                        'random_seed': random_seed
                    }
                }

        # Display results
        if 'simulation_results' in st.session_state:
            results = st.session_state['simulation_results']

            # Price paths chart
            st.subheader("Stock Price Paths")
            fig_paths = create_price_paths_chart(
                results['prices'],
                results['envelopes'],
                num_paths_to_show=100
            )
            st.plotly_chart(fig_paths, use_container_width=True)

            # Summary table
            st.subheader("Performance Summary")
            summary_df = create_summary_table(results['metrics'])
            st.dataframe(summary_df)

            # Performance histograms
            st.subheader("Execution Performance Distribution")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Strategy 1**")
                fig_perf1 = create_performance_histogram(
                    results['metrics']['strategy_1']['performance_bps'],
                    'strategy_1'
                )
                st.plotly_chart(fig_perf1, use_container_width=True)

            with col2:
                st.write("**Strategy 2**")
                fig_perf2 = create_performance_histogram(
                    results['metrics']['strategy_2']['performance_bps'],
                    'strategy_2'
                )
                st.plotly_chart(fig_perf2, use_container_width=True)

            with col3:
                st.write("**Strategy 3**")
                fig_perf3 = create_performance_histogram(
                    results['metrics']['strategy_3']['performance_bps'],
                    'strategy_3'
                )
                st.plotly_chart(fig_perf3, use_container_width=True)

            # Benchmark vs Execution charts
            st.subheader("Benchmark vs Execution Price")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Strategy 1**")
                fig_bve1 = create_benchmark_vs_execution_chart(
                    results['metrics']['strategy_1']['vwap'],
                    results['metrics']['strategy_1']['benchmark_prices'],
                    'strategy_1'
                )
                st.plotly_chart(fig_bve1, use_container_width=True)

            with col2:
                st.write("**Strategy 2**")
                fig_bve2 = create_benchmark_vs_execution_chart(
                    results['metrics']['strategy_2']['vwap'],
                    results['metrics']['strategy_2']['benchmark_prices'],
                    'strategy_2'
                )
                st.plotly_chart(fig_bve2, use_container_width=True)

            with col3:
                st.write("**Strategy 3**")
                fig_bve3 = create_benchmark_vs_execution_chart(
                    results['metrics']['strategy_3']['vwap'],
                    results['metrics']['strategy_3']['benchmark_prices'],
                    'strategy_3'
                )
                st.plotly_chart(fig_bve3, use_container_width=True)

            # Duration histograms
            st.subheader("Trading Duration Distribution")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Strategy 1**")
                fig_dur1 = create_duration_histogram(
                    results['metrics']['strategy_1']['actual_end_days'],
                    'strategy_1',
                    results['parameters']['min_duration'],
                    results['parameters']['target_duration'],
                    results['parameters']['max_duration']
                )
                st.plotly_chart(fig_dur1, use_container_width=True)

            with col2:
                st.write("**Strategy 2**")
                fig_dur2 = create_duration_histogram(
                    results['metrics']['strategy_2']['actual_end_days'],
                    'strategy_2',
                    results['parameters']['min_duration'],
                    results['parameters']['target_duration'],
                    results['parameters']['max_duration']
                )
                st.plotly_chart(fig_dur2, use_container_width=True)

            with col3:
                st.write("**Strategy 3**")
                fig_dur3 = create_duration_histogram(
                    results['metrics']['strategy_3']['actual_end_days'],
                    'strategy_3',
                    results['parameters']['min_duration'],
                    results['parameters']['target_duration'],
                    results['parameters']['max_duration']
                )
                st.plotly_chart(fig_dur3, use_container_width=True)

    else:
        st.info("Click 'Run Simulation' to start the analysis.")

# Tab 2: Example
with tab2:
    with st.sidebar:
        st.header("Example Parameters")

        # Generate Example button at the top
        generate_example = st.button("Generate Example", key="generate_example_button", type="primary")

        # Use same parameters as simulation
        if 'simulation_results' in st.session_state:
            params = st.session_state['simulation_results']['parameters']
            ex_initial_price = params['initial_price']
            ex_num_days = params['num_days']
            ex_drift = params['drift']
            ex_volatility = params['volatility']
            ex_usd_to_execute = params['usd_to_execute']
            ex_min_duration = params['min_duration']
            ex_target_duration = params['target_duration']
            ex_max_duration = params['max_duration']
            ex_benchmark_discount_bps = params['benchmark_discount_bps']
        else:
            ex_initial_price = 100
            ex_num_days = 125
            ex_drift = 0
            ex_volatility = 25
            ex_usd_to_execute = 1000000000
            ex_max_duration = 125
            ex_min_duration = int(ex_max_duration * 0.6)
            ex_target_duration = int((ex_min_duration + ex_max_duration) / 2)
            ex_benchmark_discount_bps = 0

    if generate_example or 'example_results' in st.session_state:
        if generate_example:
            with st.spinner("Generating example..."):
                # Generate single price path
                single_price = simulate_gbm_paths(
                    S0=ex_initial_price,
                    num_days=ex_num_days,
                    drift_pct=ex_drift,
                    volatility_pct=ex_volatility,
                    num_simulations=1,
                    random_seed=None  # Always random for new examples
                )

                # Execute all strategies on single path
                example_strategy_results = execute_all_strategies(
                    prices=single_price,
                    total_usd=ex_usd_to_execute,
                    target_duration=ex_target_duration,
                    min_duration=ex_min_duration,
                    max_duration=ex_max_duration,
                    benchmark_discount_bps=ex_benchmark_discount_bps
                )

                # Calculate benchmarks
                benchmark_no_discount = calculate_arithmetic_mean_benchmark(single_price, 0)
                benchmark_with_discount = calculate_arithmetic_mean_benchmark(single_price, ex_benchmark_discount_bps)

                st.session_state['example_results'] = {
                    'price': single_price[0],
                    'strategy_results': example_strategy_results,
                    'benchmark_no_discount': benchmark_no_discount,
                    'benchmark_with_discount': benchmark_with_discount,
                    'parameters': {
                        'min_duration': ex_min_duration,
                        'target_duration': ex_target_duration,
                        'max_duration': ex_max_duration,
                        'benchmark_discount_bps': ex_benchmark_discount_bps
                    }
                }

        # Display example results
        if 'example_results' in st.session_state:
            ex_results = st.session_state['example_results']

            st.subheader("Example Simulation Details")

            # For each strategy, show detailed execution
            for strategy_name in ['strategy_1', 'strategy_2', 'strategy_3']:
                st.write(f"### Strategy {strategy_name.replace('strategy_', '')}")

                strategy_data = ex_results['strategy_results'][strategy_name]
                actual_end_day = int(strategy_data['actual_end_days'][0])

                # Calculate VWAP evolution and performance evolution
                vwap_evolution = np.zeros(actual_end_day)
                performance_evolution = np.zeros(actual_end_day)

                for day in range(actual_end_day):
                    total_usd = np.sum(strategy_data['usd_executed'][0, :day+1])
                    total_shares = np.sum(strategy_data['shares_acquired'][0, :day+1])
                    if total_shares > 0:
                        vwap_evolution[day] = total_usd / total_shares

                    if strategy_name == 'strategy_3':
                        benchmark = ex_results['benchmark_with_discount']
                    else:
                        benchmark = ex_results['benchmark_no_discount']

                    if day < len(benchmark):
                        performance_evolution[day] = -((vwap_evolution[day] - benchmark[day]) / benchmark[day]) * 10000

                # Create execution chart
                fig_exec = create_example_execution_chart(
                    prices=ex_results['price'][:actual_end_day],
                    usd_executed=strategy_data['usd_executed'][0],
                    shares_acquired=strategy_data['shares_acquired'][0],
                    benchmark=ex_results['benchmark_with_discount'] if strategy_name == 'strategy_3' else ex_results['benchmark_no_discount'],
                    vwap_evolution=vwap_evolution,
                    performance_evolution=performance_evolution,
                    strategy_name=strategy_name,
                    actual_end_day=actual_end_day,
                    min_duration=ex_results['parameters']['min_duration'],
                    target_duration=ex_results['parameters']['target_duration'],
                    max_duration=ex_results['parameters']['max_duration']
                )
                st.plotly_chart(fig_exec, use_container_width=True)

                # Create daily execution bar chart
                fig_daily = create_daily_execution_chart(
                    usd_executed=strategy_data['usd_executed'][0],
                    strategy_name=strategy_name,
                    actual_end_day=actual_end_day
                )
                st.plotly_chart(fig_daily, use_container_width=True)

                # Show summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Total USD Executed",
                        f"${np.sum(strategy_data['usd_executed'][0, :actual_end_day]):,.0f}"
                    )
                with col2:
                    st.metric(
                        "Total Shares Acquired",
                        f"{np.sum(strategy_data['shares_acquired'][0, :actual_end_day]):,.0f}"
                    )
                with col3:
                    st.metric(
                        "Final VWAP",
                        f"${vwap_evolution[-1]:.2f}"
                    )
                with col4:
                    st.metric(
                        "Final Performance (bps)",
                        f"{performance_evolution[-1]:.2f}"
                    )

                st.divider()

# Tab 3: Explanation
with tab3:
    st.header("Mathematical Formulation and Strategy Logic")

    st.subheader("1. Geometric Brownian Motion (GBM) Model")
    st.markdown(r"""
    The stock price follows a Geometric Brownian Motion:

    $$dS_t = \mu S_t dt + \sigma S_t dW_t$$

    Where:
    - $S_t$ = Stock price at time $t$
    - $\mu$ = Drift (expected return)
    - $\sigma$ = Volatility
    - $dW_t$ = Brownian motion increment

    **Discretized version (daily steps):**

    $$S_{t+1} = S_t \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t} Z\right]$$

    Where:
    - $\Delta t = 1/252$ (one trading day)
    - $Z \sim N(0,1)$ (standard normal random variable)
    """)

    st.subheader("2. Benchmark Definitions")
    st.markdown(r"""
    **Strategy 2 Benchmark:** Arithmetic mean of all daily prices from day 1 to current day $t$:

    $$\text{Benchmark}_t = \frac{1}{t}\sum_{i=1}^{t} S_i$$

    **Strategy 3 Benchmark:** Same as Strategy 2 but with a discount:

    $$\text{Benchmark}_t^{\text{discounted}} = \text{Benchmark}_t \times \left(1 - \frac{\text{discount\_bps}}{10000}\right)$$
    """)

    st.subheader("3. Trading Strategies")

    st.markdown("""
    ### Strategy 1: Equal Daily Buying
    - Buy equal USD amount each day over the target duration
    - Daily trade amount = Total USD / Target Duration
    - Example: $1B over 100 days = $10M per day
    """)

    st.markdown("""
    ### Strategy 2: Adaptive Trading
    **First 10 days (or min_duration if smaller):**
    - Trade at target rate = Total USD / Target Duration

    **After day 10 (or min_duration):**
    - If price < benchmark → Speed up trading
      - Before min_duration: Adjust to finish by min_duration
      - After min_duration: Trade up to 5× first day value
    - If price > benchmark → Slow down trading
      - Extend to max_duration
      - If remaining < 5× first day AND >5 days left: Trade at most 0.1× first day value

    **Numerical Example:**
    - Total USD: $1B, Target: 100 days, Min: 60 days, Max: 125 days
    - Day 1-10: $10M/day (target rate)
    - Day 15, price < benchmark: Speed up to finish by day 60
      - Remaining: $850M, Days to min: 45
      - Daily trade: $850M / 45 = $18.9M
    - Day 61, price < benchmark (beyond min): Trade 5× first day
      - Daily trade: min($50M, remaining)
    - Day 30, price > benchmark: Slow down to max duration
      - Remaining: $700M, Days to max: 95
      - Daily trade: $700M / 95 = $7.4M
    """)

    st.markdown("""
    ### Strategy 3: Adaptive Trading with Discounted Benchmark
    - Same logic as Strategy 2
    - Uses benchmark reduced by specified basis points
    - More likely to trigger "speed up" conditions
    - Example: 10 bps discount means benchmark is 99.9% of arithmetic mean
    """)

    st.subheader("4. VWAP Execution Price")
    st.markdown(r"""
    The Volume Weighted Average Price (VWAP) is:

    $$\text{VWAP} = \frac{\sum_{t=1}^{T} \text{USD}_t}{\sum_{t=1}^{T} \text{Shares}_t} = \frac{\text{Total USD Spent}}{\text{Total Shares Acquired}}$$

    Where:
    - $\text{USD}_t$ = USD amount traded on day $t$
    - $\text{Shares}_t = \text{USD}_t / S_t$ = Shares acquired on day $t$
    - $T$ = Actual end day of trading
    """)

    st.subheader("5. Performance Metrics")
    st.markdown(r"""
    **Performance in basis points:**

    $$\text{Performance (bps)} = -\frac{\text{VWAP} - \text{Benchmark}}{\text{Benchmark}} \times 10000$$

    Note: Negative sign means positive performance when VWAP < Benchmark

    **Standard Error for finite simulations:**

    $$\text{SE} = \frac{\sigma_{\text{performance}}}{\sqrt{N}}$$

    Where:
    - $\sigma_{\text{performance}}$ = Standard deviation of performance across simulations
    - $N$ = Number of simulations
    """)

    st.subheader("6. Detailed Strategy Examples")

    # Example calculation
    st.markdown("""
    ### Detailed Numerical Example

    **Setup:**
    - Initial price: $100
    - Total USD: $1,000,000
    - Target duration: 10 days
    - Min duration: 6 days
    - Max duration: 15 days

    **Strategy 1 Execution:**
    ```
    Daily amount = $1,000,000 / 10 = $100,000
    Day 1: Buy $100,000 at $100 = 1,000 shares
    Day 2: Buy $100,000 at $102 = 980.39 shares
    ...
    Day 10: Buy $100,000 at $98 = 1,020.41 shares
    Total: $1,000,000 spent, 10,123 shares acquired
    VWAP = $1,000,000 / 10,123 = $98.78
    ```

    **Strategy 2 Execution (example scenario):**
    ```
    Days 1-6: Trade at target rate ($100,000/day)
    Day 7: Price ($95) < Benchmark ($99)
      → Speed up: Remaining $400,000 / 4 days = $100,000/day (no change yet)
    Day 8: Price ($93) < Benchmark ($98)
      → Beyond min duration: Trade 5× first day = min($500,000, $300,000) = $300,000
    Day 9: Price ($97) < Benchmark ($97.5)
      → Continue accelerated: $100,000 remaining
    Total: Finished in 9 days instead of 10
    ```
    """)

    st.subheader("7. Key Insights")
    st.markdown("""
    - **Strategy 1** provides a simple baseline with predictable execution
    - **Strategy 2** adapts to market conditions, potentially improving execution
    - **Strategy 3** uses a discounted benchmark, more aggressive in accelerating purchases
    - **Flexibility in end times** allows strategies to capitalize on favorable price movements
    - **Standard error** decreases with more simulations (√N relationship)
    """)