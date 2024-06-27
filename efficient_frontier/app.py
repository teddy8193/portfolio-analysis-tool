import streamlit as st

import datetime as dt
import time
import pandas as pd

from efficient_frontier import data, visualize, utils


def on_ticker_added():
    print('on_ticker_added')
    ticker_len = len(st.session_state.tickers)
    st.session_state.tickers.append("")
    st.session_state.ticker_validity.append(None)
    color = utils.generate_rand_color(3 + ticker_len * 5, hue_range=(0.3, 1.1), sat_range=(0.65, 0.7), val_range=(0.9, 0.95))
    st.session_state.ticker_colors.append(color)


def on_ticker_entered(index: int):
    print(f'on_ticker_entered {index}')
    key = f"ticker_{index}"
    key_country = f"country_{index}"

    try:
        # Get Ticker Info from Widget
        ticker = st.session_state[key].strip()
        if not ticker:
            # Store Session State
            st.session_state.tickers[index] = ""
            st.session_state.ticker_validity[index] = None
            return

        ticker_country: str = st.session_state[key_country]
        ticker_country_enum = data.Country(ticker_country.lower())
        ticker_valid = data.is_valid_ticker(ticker_country_enum, ticker)

        # Store Session State
        st.session_state.tickers[index] = ticker
        st.session_state.ticker_validity[index] = ticker_valid
    except:
        pass


def on_ticker_deleted(index: int):
    print(f'on_ticker_deleted {index}')
    try:
        del st.session_state.tickers[index]
        del st.session_state.ticker_validity[index]
        del st.session_state.ticker_colors[index]
    except:
        pass


def on_ticker_color_changed(index: int):
    print(f'on_ticker_color_changed {index}')
    key = f"color_{index}"
    color = st.session_state[key]
    st.session_state.ticker_colors[index] = color


st.header("Portfolio Perfomance Checker")


# Initialize session state
if "tickers" not in st.session_state:
    st.session_state.tickers = ["", "", ""]

if "ticker_validity" not in st.session_state:
    st.session_state.ticker_validity = [None, None, None]

if "ticker_colors" not in st.session_state:
    colors = [utils.generate_rand_color(3 + i * 5, hue_range=(0.3, 1.05), sat_range=(0.65, 0.7), val_range=(0.9, 0.95)) for i in range(3)]
    st.session_state.ticker_colors = colors

# print(f"tickers {st.session_state.tickers}")


with st.container(border=True):
    date_c0, date_c1, date_c2 = st.columns([4, 4, 2])
    start = date_c0.date_input('Start', dt.date.today() - dt.timedelta(days=366))
    end = date_c1.date_input('End', dt.date.today())
    time_delta = date_c2.selectbox('Delta', ['Daily', 'Weekly', 'Monthly', 'Annualy'])

    return_c0, return_c1 = st.columns([5, 5])
    return_mode = return_c0.radio('Return Calculation', ['Simple', 'Log'], index=1)
    use_dividend = return_c1.checkbox('Include Dividend Return', value=True)

    currency = st.selectbox('Currency', ['USD', 'JPY'])

    anorm_col1, anorm_col2 = st.columns([2.5, 7.5])
    exclude_anomaly = anorm_col1.checkbox("Exclude anomaly", value=True)
    anomaly_sigma = 5
    if exclude_anomaly:
        anomaly_sigma = anorm_col2.slider("Anomaly Sigma Threshold", min_value=2, max_value=10, value=5)

    st.divider()

    for i, ticker in enumerate(st.session_state.tickers):
        c0, c1, c2, c3 = st.columns([1.5, 7, 0.5, 1])
        with c0:
            st.selectbox('Country', options=['US', 'JP'], key=f"country_{i}")

        with c1:
            st.text_input(f"Ticker {i+1}", value=ticker, key=f"ticker_{i}", on_change=on_ticker_entered, args=(i,))

        with c2:
            st.markdown(
                """
                <div style="height: 32px;">
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.color_picker(
                'color',
                value=st.session_state.ticker_colors[i],
                key=f"color_{i}",
                label_visibility='collapsed',
                on_change=on_ticker_color_changed,
                args=(i,),
            )

        with c3:
            st.markdown(
                """
                <div style="height: 28px;">
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.button("❌", key=f"delete_{i}", on_click=on_ticker_deleted, args=(i,))
    st.button("➕ Add field", on_click=on_ticker_added)

invalids = []
for i, ticker_valid in enumerate(st.session_state.ticker_validity):
    if ticker_valid is False:
        invalids.append(st.session_state.tickers[i])
if invalids:
    invalids_str = ",".join(invalids)
    st.warning(f"{invalids_str} are invalid ticker symbols.")

run_vis = st.button('Visualize')

symbols = [ticker for ticker in st.session_state.tickers if ticker]


if run_vis and not symbols:
    st.warning("No ticker is selected")


if run_vis and symbols:
    df_price, df_dividends = data.get_stock_data(data.Country.US, symbols, start, end)
    colors = st.session_state.ticker_colors

    # Line Chart for Price
    chart_fig = visualize.line_chart(df_price, colors=colors)
    st.plotly_chart(chart_fig)

    # Scatter Chart for Price Change
    df_capital_return = data.calc_return_capital(df_price, data.CalcMode.Log)
    change_fig = visualize.scatter_chart(df_capital_return, colors=colors)
    st.plotly_chart(change_fig)

    # if exclude_anomaly:
    #     df_capital_return = data.drop_outliers(df_capital_return, border_sigma=anomaly_sigma)

    # Risk Free Rate
    df_risk_free = data.get_risk_free_data(data.Country.US, start, end)
    mean_risk_free = df_risk_free.mean() * 0.01

    # Dividend Return
    df_dividend_return = data.calc_diviednd_return(df_dividends, df_price, data.CalcMode.Log)

    # Calculate Performance for Return, Variance
    df_performance, df_cov_matrix = data.calc_stock_performance(df_capital_return, df_dividend_return, calc_mode=data.CalcMode.Log)

    # Weights for multiple portfolios
    df_weights = pd.DataFrame(data.get_rand_weights(n_samples=5000, n_weights=len(df_performance)), columns=df_cov_matrix.columns)

    # Portfolios
    df_portfolios = data.calc_portfolios(df_performance, df_cov_matrix, df_weights, log_return=True, risk_free_rate=mean_risk_free)
    portfolio_fig = visualize.efficient_frontier(df_portfolios, df_performance, df_weights)
    st.plotly_chart(portfolio_fig)
