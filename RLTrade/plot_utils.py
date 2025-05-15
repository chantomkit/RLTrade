import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_candlestick_subplots(data_list, titles, x_index):
    """
    Plots multiple candlestick charts as subplots.

    Parameters:
        data_list (List[Tuple[open, high, low, close]]): Each tuple contains 4 Series-like arrays.
        titles (List[str]): Subplot titles.
        trace_names (List[str]): Names for each trace.
        x_index (array-like): Common x-axis values (e.g., df.index).
    """
    num_charts = len(data_list)
    fig = make_subplots(
        rows=num_charts, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[1/num_charts] * num_charts,
        subplot_titles=titles
    )

    for i, (o, h, l, c) in enumerate(data_list, start=1):
        fig.add_trace(go.Candlestick(
            x=x_index,
            open=o,
            high=h,
            low=l,
            close=c,
        ), row=i, col=1)

        # Hide range slider for each xaxis
        fig.update_layout(**{f"xaxis{i}_rangeslider_visible": False})

    fig.update_layout(
        height=300 * num_charts,
        showlegend=False,
        margin=dict(t=40, b=20)
    )

    fig.show()

def plot_metrics_subplots(df, columns, titles=None, x_label="Episode Number"):
    """
    Plots specified columns of a DataFrame as vertically stacked subplots.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the metrics.
        columns (List[str]): Column names to plot.
        titles (List[str], optional): Titles for each subplot. Defaults to column names.
        x_label (str): Label for the shared x-axis.
    """
    num_plots = len(columns)
    if titles is None:
        titles = columns

    fig, axs = plt.subplots(num_plots, 1, figsize=(9, 2.5 * num_plots), sharex=True)

    # Ensure axs is iterable even if there's only one subplot
    if num_plots == 1:
        axs = [axs]

    for i, (col, title) in enumerate(zip(columns, titles)):
        axs[i].plot(df[col], label=title)
        axs[i].set_title(title)
        axs[i].legend(loc="upper right")

    axs[-1].set_xlabel(x_label)
    plt.tight_layout()
    plt.show()

def build_fig(historical_info_df: pd.DataFrame,
              buy_df: pd.DataFrame,
              sell_df: pd.DataFrame,
              exit_df: pd.DataFrame):
    """
    Build a 2-row subplot:
     - Row 1: candlestick + buy/sell/exit markers
     - Row 2: portfolio valuation line
    Returns the Plotly Figure.
    """
    sample = historical_info_df
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=["Price Candlestick", "Portfolio Value"]
    )

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=sample.index,
        open=sample['data_open'],
        high=sample['data_high'],
        low=sample['data_low'],
        close=sample['data_close'],
        name="Price"
    ), row=1, col=1)

    # Markers
    marker_specs = {
        'Buy':    dict(df=buy_df,   color='green', symbol='triangle-up'),
        'Sell':   dict(df=sell_df,  color='red',   symbol='triangle-down'),
        'Exit':   dict(df=exit_df,  color='blue',  symbol='circle'),
    }
    for name, spec in marker_specs.items():
        fig.add_trace(go.Scatter(
            x=spec['df'].index,
            y=spec['df']['data_close'],
            mode='markers',
            marker=dict(color=spec['color'], size=10, symbol=spec['symbol']),
            name=name
        ), row=1, col=1)

    # Row 2: Portfolio value
    fig.add_trace(go.Scatter(
        x=sample.index,
        y=sample['portfolio_valuation'],
        mode='lines',
        line=dict(color='blue'),
        name="Portfolio Value"
    ), row=2, col=1)

    # Layout
    fig.update_layout(
        height=500,
        margin=dict(t=40, b=20, l=40, r=20),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False  # if you ever disable or have more subplots
    )
    return fig

def plot_signals_distribution(signals_df, threshold):
    """
    Plots a violin + jittered stripplot of data_close by position,
    with a horizontal line at `threshold`.
    """
    plt.figure(figsize=(8, 6))
    # Violin (no inner box)
    sns.violinplot(
        data=signals_df,
        x='position',
        y='data_close',
        inner=None,
        color=".8"
    )
    # Jittered points
    sns.stripplot(
        data=signals_df,
        x='position',
        y='data_close',
        jitter=0.2,
        size=4,
        alpha=0.6
    )
    # Threshold line
    plt.axhline(threshold, color='r', linestyle='--', label=f'y = {threshold}')
    plt.legend()

    plt.title('Distribution + Jittered Points by Position')
    plt.tight_layout()
    plt.show()