import numpy as np
import pandas as pd

from efficient_frontier import data

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go


def efficient_frontier(df_portfolios: pd.DataFrame, df_perfomance: pd.DataFrame, df_weights: pd.DataFrame) -> go.Figure:
    # sharp_ratio = (data.log_to_simple_return(ret_list, percent=False) - MEAN_RISK_FREE) / std_list
    # sharpe_ratio = data.calc_sharpe_ratio(ret_list, MEAN_RISK_FREE, std_list, log_return=True)

    layout = go.Layout(
        xaxis={'title': 'Risk (%)'},
        yaxis={
            'title': 'Return (%)',
        },
    )
    fig = go.Figure(layout=layout)

    # Portfolios
    if len(df_perfomance) > 1:
        hover_text_weights_label = [f'{x}: ' + '%{customdata[' + str(i) + ']:.2f}' for i, x in enumerate(df_weights.columns)]
        hover_text_weights = "<br>".join(hover_text_weights_label)

        marker_color = df_portfolios['sharpe_ratio'] if "sharpe_ratio" in df_portfolios.columns else 'LightSkyBlue'

        portfolio_samples = go.Scatter(
            x=df_portfolios['std'] * 100,
            y=data.log_to_simple_return(df_portfolios['return']),
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
    for row in df_perfomance.iterrows():
        graph = go.Scatter(
            x=[row[1]['std'] * 100],
            y=[data.log_to_simple_return(row[1]['total'])],
            mode='markers+text',
            text=row[0],
            textposition='top right',
            name=row[0],
            hoverinfo='text',
            hovertemplate=row[0] + '<br><br>Return: %{y:.1f}<br>Risk: %{x:.1f}',
            marker=dict(
                symbol="diamond",
            ),
            # marker=dict(size=3, color=samples[:, 2], colorscale='Viridis', opacity=0.3),
        )
        asset_points.append(graph)

    fig.add_traces(asset_points)
    fig.update_layout(width=800, height=800)

    return fig


def line_chart(df: pd.DataFrame, colors: list[str] | None = None) -> go.Figure:
    # fig = px.line(df)
    # fig.update_layout(yaxis_title='')
    fig = go.Figure()
    graphs = []
    for i, col in enumerate(df.columns):
        graph = go.Line(
            x=df.index,
            y=df[col],
            name=col,
        )
        if colors:
            try:
                graph.update(line=dict(color=colors[i]))
            except BaseException:
                pass
        graphs.append(graph)
    fig.add_traces(graphs)
    fig.update_xaxes(range=[df.index[0], df.index[-1]])
    return fig


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
