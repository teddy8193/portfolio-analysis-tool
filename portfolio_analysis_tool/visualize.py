import streamlit as st

import numpy as np
import pandas as pd

import datetime as dt
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

from portfolio_analysis_tool import data


def _calc_y_range(df: pd.DataFrame, margin_btm: float = 0.05, margin_top: float = 0.05) -> tuple[float, float]:
    y_min, y_max = df.min(axis=None), df.max(axis=None)
    y_range = y_max - y_min
    y_min, y_max = y_min - y_range * margin_btm, y_max + y_range * margin_top
    return y_min, y_max


def chart_plot(stock_handler: data.StockDataHandler, use_dividend=False) -> None:
    df = stock_handler.pct_from_first['with_div'] if use_dividend else stock_handler.pct_from_first['no_div']
    colors = stock_handler.colors

    fig = go.Figure()
    graphs = []
    y_min, y_max = _calc_y_range(df, margin_top=0.08)
    for i, col in enumerate(df.columns):
        graph = go.Scatter(x=df.index, y=df[col], name=col, mode='lines')
        if colors:
            try:
                graph.update(line=dict(color=colors[i]))
            except BaseException:
                pass
        line_color = graph.line.color
        hovertemplate = '<span style="color:' + line_color + '">%{fullData.name}</span><br>' + '%{y:.1%}<br>%{x|%Y-%m-%d}<extra></extra>'
        graph.hovertemplate = hovertemplate
        graphs.append(graph)
    fig.add_traces(graphs)

    fig.update_xaxes(range=[df.index[0], df.index[-1]])
    fig.update_yaxes(range=[y_min, y_max], showline=True)
    fig.update_layout(yaxis=dict(tickformat=".0%"))

    # Stripes
    for year in df.index.year.unique():
        year_start = dt.date(year, 1, 1)
        year_end = year_start + relativedelta(years=1) - relativedelta(days=1)
        fig.add_shape(
            type="rect",
            x0=year_start,
            x1=year_end,
            y0=y_min,
            y1=y_max,
            fillcolor="white",
            opacity=0.03 if year % 2 == 0 else 0,
            layer="below",
            line_width=0,
        )

    st.plotly_chart(fig)


def performance_plot(stock_handler: data.StockDataHandler, delta: str = 'Yearly') -> None:
    df = stock_handler.yearly_return
    df_band = stock_handler.yearly_risk
    colors = stock_handler.colors

    fig = go.Figure()
    graphs = []
    y_min, y_max = _calc_y_range(df)
    y_abs_max = max(abs(y_min), abs(y_max))
    for i, col in enumerate(df.columns):
        graph = go.Scatter(
            x=df.index,
            y=df[col],
            name=col,
            # mode='lines+markers',
            mode='lines',
            customdata=df_band[col],
            legendgroup=col,
        )
        if colors:
            try:
                graph.update(line=dict(color=colors[i]))
            except BaseException:
                pass
        line_color = graph.line.color
        hovertemplate = (
            '<span style="color:'
            + line_color
            + '">%{fullData.name}</span><br>'
            + 'Return: %{y:.1%}<br>'
            + 'Risk: %{customdata:.1%}<br>'
            + '%{x|%Y-%m-%d}<extra></extra>'
        )
        graph.hovertemplate = hovertemplate
        graphs.append(graph)

        graph_band = go.Scatter(
            x=df.index.append(df.index[::-1]),
            y=np.r_[df[col].values - df_band[col].values, df[col].values[::-1] + df_band[col].values[::-1]],
            fill='toself',
            fillcolor=line_color,
            opacity=0.07,
            line=dict(color='rgba(0,0,0,0)', shape='spline'),
            hoverinfo="skip",
            legendgroup=col,
            showlegend=False,
        )
        graphs.append(graph_band)

    fig.add_traces(graphs)
    fig.update_xaxes(range=[df.index[0], df.index[-1]])
    fig.update_yaxes(range=[-y_abs_max, y_abs_max], showline=True)
    fig.update_layout(yaxis=dict(tickformat=".0%"))

    # Stripes
    for year in pd.to_datetime(df.index, format="%Y-%m").year.unique():
        year_start = dt.date(year, 1, 1)
        year_end = year_start + relativedelta(years=1) - relativedelta(days=1)
        fig.add_shape(
            type="rect",
            x0=year_start,
            x1=year_end,
            y0=-y_abs_max,
            y1=y_abs_max,
            fillcolor="white",
            opacity=0.03 if year_start.year % 2 == 0 else 0,
            layer="below",
            line_width=0,
        )

    st.plotly_chart(fig)


def heat_map(stock_handler: data.StockDataHandler) -> None:
    df = stock_handler.corr
    cel_size = 200
    height = df.columns.size * cel_size
    height = min(max(height, 300), 600)
    fig = px.imshow(df.round(2), text_auto=True, height=height, color_continuous_scale='RdBu', zmin=-1, zmax=1)
    st.plotly_chart(fig)


def scatter_chart(df: pd.DataFrame, colors: list[str] | None = None, sigmas: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    graphs = []
    for i, col in enumerate(df.columns):
        graph = go.Scatter(
            x=df.index,
            y=df[col],
            mode='markers',
            name=col,
            marker=dict(
                size=2,
                opacity=1,
            ),
        )
        if colors:
            try:
                graph.update(line=dict(color=colors[i]))
            except BaseException:
                pass
        # if sigmas:
        #     for sigma in sigmas:

        graphs.append(graph)

    fig.add_traces(graphs)
    fig.update_xaxes(range=[df.index[0], df.index[-1]])
    return fig


def efficient_frontier(
    stock_handler: data.StockDataHandler, count: int = 1000, bias: float = 0.0, seed: int = 23, tickers: list[str] | None = None
) -> None:
    if tickers:
        df_cov = stock_handler.cov.loc[tickers, tickers]
        df_ret = stock_handler.mean_return.loc[tickers]
        df_risk = stock_handler.mean_risk.loc[tickers]
    else:
        df_cov = stock_handler.cov
        df_ret = stock_handler.mean_return
        df_risk = stock_handler.mean_risk

    df_weights = pd.DataFrame(
        data.get_rand_weights(n_samples=count, n_weights=len(df_cov.columns), bias=bias, seed=seed),
        columns=df_cov.columns,
    )
    df_portfolios = data.calc_portfolios(
        df_ret.to_numpy(),
        df_cov.to_numpy(),
        df_weights.to_numpy(),
        stock_handler.mean_risk_free,
    )

    layout = go.Layout(
        xaxis={'title': 'Risk (%)'},
        yaxis={
            'title': 'Return (%)',
        },
    )
    fig = go.Figure(layout=layout)

    # Portfolios
    if len(df_portfolios.columns) >= 2:
        hover_text_weights_label = [f'{x}: ' + '%{customdata[' + str(i) + ']:.2f}' for i, x in enumerate(df_weights.columns)]
        hover_text_weights = "<br>".join(hover_text_weights_label)

        marker_color = df_portfolios['sharpe_ratio'] if "sharpe_ratio" in df_portfolios.columns else 'LightSkyBlue'

        portfolio_samples = go.Scatter(
            x=df_portfolios['std'] * 100,
            # y=data.log_to_simple_return(df_portfolios['return']),
            y=df_portfolios['return'] * 100,
            customdata=df_weights,
            mode='markers',
            name='',
            # text=f'Return : {std_list * 100}<br>Risk',
            hoverinfo='text',
            hovertemplate=hover_text_weights + '<br><br>Return: %{y:.1f}<br>Risk: %{x:.1f}<br>Sharpe Ratio: %{marker.color:.2f}<br>',
            marker=dict(
                size=4,
                color=marker_color,
                colorscale='Viridis',
                colorbar=dict(title='Sharpe Ratio', len=0.5, yanchor='bottom', y=0),
                opacity=1,
            ),
        )

        fig.add_trace(portfolio_samples)

    # Individual Stock Performance
    asset_points = []
    for ticker in df_ret.index.tolist():
        graph = go.Scatter(
            x=[df_risk.loc[ticker] * 100],
            # y=[data.log_to_simple_return(row[1]['total'])],
            y=[df_ret.loc[ticker] * 100],
            mode='markers+text',
            text=ticker,
            textposition='top right',
            name=ticker,
            hoverinfo='text',
            hovertemplate=ticker + '<br><br>Return: %{y:.1f}%<br>Risk: %{x:.1f}%',
            marker=dict(
                symbol="diamond",
            ),
            # marker=dict(size=3, color=samples[:, 2], colorscale='Viridis', opacity=0.3),
        )
        asset_points.append(graph)

    fig.add_traces(asset_points)
    fig.update_layout(width=800, height=800)

    st.plotly_chart(fig)
