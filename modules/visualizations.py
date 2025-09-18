import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

def create_price_paths_chart(
    prices: np.ndarray,
    envelopes: dict,
    num_paths_to_show: int = 100
) -> go.Figure:
    fig = go.Figure()

    num_simulations, num_days = prices.shape
    days = np.arange(1, num_days + 1)

    # Sample paths to show
    if num_simulations > num_paths_to_show:
        sample_indices = np.random.choice(num_simulations, num_paths_to_show, replace=False)
    else:
        sample_indices = np.arange(num_simulations)

    # Add individual price paths
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=days,
            y=prices[idx, :],
            mode='lines',
            line=dict(width=0.5, color='lightblue'),
            opacity=0.3,
            showlegend=False,
            hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
        ))

    # Add mean price
    mean_prices = np.mean(prices, axis=0)
    fig.add_trace(go.Scatter(
        x=days,
        y=mean_prices,
        mode='lines',
        line=dict(width=2, color='black'),
        name='Mean',
        hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
    ))

    # Add envelopes
    colors = {
        '+1σ': 'rgba(255, 0, 0, 0.3)',
        '-1σ': 'rgba(255, 0, 0, 0.3)',
        '+2σ': 'rgba(255, 100, 0, 0.2)',
        '-2σ': 'rgba(255, 100, 0, 0.2)',
        '+3σ': 'rgba(255, 150, 0, 0.15)',
        '-3σ': 'rgba(255, 150, 0, 0.15)',
        '+4σ': 'rgba(255, 200, 0, 0.1)',
        '-4σ': 'rgba(255, 200, 0, 0.1)',
    }

    for label, values in envelopes.items():
        fig.add_trace(go.Scatter(
            x=days,
            y=values,
            mode='lines',
            line=dict(width=1, color=colors.get(label, 'gray'), dash='dash'),
            name=label,
            hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title="Stock Price Paths with Standard Deviation Envelopes",
        xaxis_title="Day",
        yaxis_title="Price ($)",
        hovermode='x unified',
        showlegend=True,
        height=600
    )

    return fig


def create_performance_histogram(
    performance_bps: np.ndarray,
    strategy_name: str,
    num_bins: int = 250
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=performance_bps,
        nbinsx=num_bins,
        histnorm='percent',
        marker_color='blue',
        opacity=0.7,
        name=f'Strategy {strategy_name.replace("strategy_", "")}',
        hovertemplate='Performance: %{x:.2f} bps<br>Frequency: %{y:.2f}%<extra></extra>'
    ))

    # Add vertical line at mean
    mean_perf = np.mean(performance_bps)
    fig.add_vline(
        x=mean_perf,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_perf:.2f} bps"
    )

    # Add vertical line at zero
    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="black",
        annotation_text="Benchmark"
    )

    fig.update_layout(
        title=f"Execution Performance vs Benchmark - Strategy {strategy_name.replace('strategy_', '')}",
        xaxis_title="Performance (basis points)",
        yaxis_title="Frequency (%)",
        showlegend=True,
        height=400
    )

    return fig


def create_duration_histogram(
    actual_end_days: np.ndarray,
    strategy_name: str,
    min_duration: int,
    target_duration: int,
    max_duration: int,
    num_bins: int = 250
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=actual_end_days,
        nbinsx=num_bins,
        histnorm='percent',
        marker_color='green',
        opacity=0.7,
        name=f'Strategy {strategy_name.replace("strategy_", "")}',
        hovertemplate='Duration: %{x:.0f} days<br>Frequency: %{y:.2f}%<extra></extra>'
    ))

    # Add vertical lines for min, target, max durations
    fig.add_vline(x=min_duration, line_dash="dash", line_color="blue",
                  annotation_text=f"Min: {min_duration}")
    fig.add_vline(x=target_duration, line_dash="dash", line_color="red",
                  annotation_text=f"Target: {target_duration}")
    fig.add_vline(x=max_duration, line_dash="dash", line_color="green",
                  annotation_text=f"Max: {max_duration}")

    fig.update_layout(
        title=f"Trading Duration Distribution - Strategy {strategy_name.replace('strategy_', '')}",
        xaxis_title="Duration (days)",
        yaxis_title="Frequency (%)",
        showlegend=True,
        height=400
    )

    return fig


def create_benchmark_vs_execution_chart(
    vwap: np.ndarray,
    benchmark_prices: np.ndarray,
    strategy_name: str
) -> go.Figure:
    fig = go.Figure()

    # Create scatter plot
    fig.add_trace(go.Scatter(
        x=benchmark_prices,
        y=vwap,
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.5),
        name='Simulations',
        hovertemplate='Benchmark: $%{x:.2f}<br>Execution: $%{y:.2f}<extra></extra>'
    ))

    # Add diagonal line (perfect execution)
    min_val = min(np.min(benchmark_prices), np.min(vwap))
    max_val = max(np.max(benchmark_prices), np.max(vwap))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Execution',
        hovertemplate='Perfect: $%{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Benchmark vs Execution Price - Strategy {strategy_name.replace('strategy_', '')}",
        xaxis_title="Benchmark Price ($)",
        yaxis_title="Execution VWAP ($)",
        showlegend=True,
        height=400
    )

    return fig


def create_example_execution_chart(
    prices: np.ndarray,
    usd_executed: np.ndarray,
    shares_acquired: np.ndarray,
    benchmark: np.ndarray,
    vwap_evolution: np.ndarray,
    performance_evolution: np.ndarray,
    strategy_name: str,
    actual_end_day: int,
    min_duration: int,
    target_duration: int,
    max_duration: int
) -> go.Figure:
    days = np.arange(1, len(prices) + 1)

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Stock price
    fig.add_trace(go.Scatter(
        x=days,
        y=prices,
        mode='lines',
        name='Stock Price',
        line=dict(width=2, color='blue'),
        hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
    ))

    # Benchmark
    fig.add_trace(go.Scatter(
        x=days,
        y=benchmark,
        mode='lines',
        name='Benchmark',
        line=dict(width=2, color='green', dash='dash'),
        hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
    ))

    # Execution price (VWAP evolution)
    fig.add_trace(go.Scatter(
        x=days,
        y=vwap_evolution,
        mode='lines',
        name='Execution Price (VWAP)',
        line=dict(width=2, color='red'),
        hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
    ))

    # Add performance on secondary axis
    fig.add_trace(go.Scatter(
        x=days,
        y=performance_evolution,
        mode='lines',
        name='Execution Performance (bps)',
        line=dict(width=2, color='purple'),
        yaxis='y2',
        hovertemplate='Day %{x}: %{y:.2f} bps<extra></extra>'
    ))

    # Add vertical lines for duration markers
    fig.add_vline(x=min_duration, line_dash="dash", line_color="lightblue",
                  annotation_text=f"Min: {min_duration}")
    fig.add_vline(x=target_duration, line_dash="dash", line_color="orange",
                  annotation_text=f"Target: {target_duration}")
    fig.add_vline(x=max_duration, line_dash="dash", line_color="lightgreen",
                  annotation_text=f"Max: {max_duration}")
    fig.add_vline(x=actual_end_day, line_dash="solid", line_color="red",
                  line_width=2, annotation_text=f"Actual End: {actual_end_day}")

    # Create secondary y-axis
    fig.update_layout(
        title=f"Example Execution Detail - Strategy {strategy_name.replace('strategy_', '')}",
        xaxis_title="Day",
        yaxis=dict(title="Price ($)", side="left"),
        yaxis2=dict(title="Performance (bps)", overlaying="y", side="right"),
        hovermode='x unified',
        showlegend=True,
        height=500
    )

    return fig


def create_daily_execution_chart(
    usd_executed: np.ndarray,
    strategy_name: str,
    actual_end_day: int
) -> go.Figure:
    days = np.arange(1, actual_end_day + 1)
    daily_values = usd_executed[:actual_end_day]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=days,
        y=daily_values,
        marker_color='blue',
        name='Daily USD Executed',
        hovertemplate='Day %{x}: $%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Daily USD Execution - Strategy {strategy_name.replace('strategy_', '')}",
        xaxis_title="Day",
        yaxis_title="USD Executed",
        showlegend=True,
        height=400
    )

    return fig


def create_summary_table(metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    summary_data = []

    for strategy_name, strategy_metrics in metrics.items():
        stats = strategy_metrics['statistics']
        strategy_num = strategy_name.replace('strategy_', '')

        summary_data.append({
            'Strategy': f'Strategy {strategy_num}',
            'Mean (bps)': f"{stats['mean']:.2f}",
            'Std Error (bps)': f"{stats['std_error']:.2f}",
            'Median (bps)': f"{stats['median']:.2f}",
            'Std Dev (bps)': f"{stats['std']:.2f}",
            'Min (bps)': f"{stats['min']:.2f}",
            'Max (bps)': f"{stats['max']:.2f}",
            'Avg Duration (days)': f"{np.mean(strategy_metrics['actual_end_days']):.1f}"
        })

    return pd.DataFrame(summary_data)