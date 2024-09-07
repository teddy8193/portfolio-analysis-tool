import streamlit as st


import uuid
import datetime as dt
import numpy as np
from functools import partial

from portfolio_analysis_tool import data, event, utils, log, models, visualize

css = '''
<style>
    section.main > div {
                        width: 100%;
                        max-width: 1000px;
                        margin: 0 auto;}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

rand_color = partial(utils.generate_rand_color, hue_range=(0.3, 1.1), sat_range=(0.65, 0.7), val_range=(0.9, 0.95))

logger = log.logger


def init_session_state(data: dict[str, object]):
    for k, v in data.items():
        if k not in st.session_state:
            st.session_state[k] = v


st.header("Portfolio Perfomance Checker")

# Initialize session state
session_data = dict(
    ticker_ids=[uuid.uuid4() for i in range(3)],
    invalid_tickers=[],
    no_symbols=None,
    stock_handler=data.StockDataHandler(),
    date_specific=False,
)
init_session_state(session_data)

# UI
with st.container(border=True):
    min_date = dt.date(1970, 1, 1)
    max_date = dt.date(2050, 12, 31)
    date_c0, date_c1, date_c2 = st.columns([4, 4, 2])

    if st.session_state.date_specific:
        with date_c0:
            start = st.date_input(
                'Start', dt.date.today() - dt.timedelta(days=365), min_value=min_date, max_value=max_date, key='date_start'
            )
        with date_c1:
            end = st.date_input('End', dt.date.today(), min_value=min_date, max_value=max_date, key='date_end')

    else:
        date_options = np.arange(min_date.year, max_date.year, 1)
        current_index = dt.date.today().year - min_date.year
        with date_c0:
            start = st.selectbox('Start', options=date_options, index=current_index, key='date_start')
        with date_c1:
            end = st.selectbox('End', options=date_options, index=current_index, key='date_end')

    with date_c2:
        fiscal_start_options = [term.name for term in models.Term]
        fiscal_start = st.selectbox("Fiscal Year Start", options=fiscal_start_options, key='fiscal_start')

    date_op0, date_op1, date_op2 = st.columns([3, 3, 4])
    with date_op0:
        st.checkbox('Use specific date', key='check_date_specific', on_change=event.on_specific_date_clicked)
    with date_op1:
        st.container()
    with date_op2:
        st.container()

    st.divider()

    ticker: models.TickerItem
    id: uuid.UUID
    for i, id in enumerate(st.session_state.ticker_ids):
        c0, c1, c2, c3 = st.columns([1.5, 7, 0.5, 1])
        with c0:
            country_sl_options = ['US', 'JP']
            st.selectbox(
                'Country',
                options=country_sl_options,
                key=f"country_{id}",
            )

        with c1:
            text = st.session_state[f"ticker_{id}"] if f"ticker_{id}" in st.session_state else ""
            st.text_input(f"Ticker {i+1}", value=text, key=f"ticker_{id}")

        with c2:
            st.markdown(
                """
                <div style="height: 32px;">
                </div>
            """,
                unsafe_allow_html=True,
            )
            color = st.session_state[f"color_{id}"] if f"color_{id}" in st.session_state else rand_color(3 + i * 5)
            st.color_picker('color', value=color, key=f"color_{id}", label_visibility='collapsed')

        with c3:
            st.markdown(
                """
                <div style="height: 28px;">
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.button("❌", key=f"delete_{id}", on_click=event.on_ticker_deleted, args=(id,))
    st.button("➕ Add field", on_click=event.on_ticker_added)

run_vis = st.button('Visualize', on_click=event.on_visualize_pressed)

if st.session_state.invalid_tickers:
    invalids_str = ",".join(st.session_state.invalid_tickers)
    st.warning(f"Invalid ticker symbols: [{invalids_str}]", icon="⚠️")

if st.session_state.no_symbols:
    st.warning("No valid symbol.", icon="⚠️")
    st.stop()  # If not valid

stock_handler: data.StockDataHandler = st.session_state.stock_handler

if stock_handler.tickers:
    st.markdown('#### Return from Start')
    use_div = st.checkbox('Include Dividend')
    visualize.chart_plot(stock_handler, use_dividend=use_div)

    st.markdown('#### Yearly Return / Risk')
    # c0, c1 = st.columns([2, 8])
    # with c0:
    #     time_delta = st.selectbox('Aggregation Range', ['Daily', 'Weekly', 'Monthly', 'Yearly'], index=3)
    # with c1:
    #     st.container()
    visualize.performance_plot(stock_handler)

    st.markdown('#### Correlation Coefficient')
    visualize.heat_map(stock_handler)

    st.markdown('#### Efficient Frontier')
    col_ef1, col_ef2, col_ef3 = st.columns([0.5, 0.3, 0.2])
    with col_ef1:
        ef_count = st.slider("Count", min_value=1, max_value=20000, value=2000)
    with col_ef2:
        ef_bias = st.slider("Bias", min_value=-3.0, max_value=3.0, value=0.5, step=0.01)
    with col_ef3:
        ef_seed = st.number_input("Seed", step=1, value=23)
    ticker_codes = [t.code for t in stock_handler.tickers] if stock_handler.tickers else []
    ef_tickers = st.multiselect('Tickers', options=ticker_codes, default=ticker_codes, key='ef_tickers')
    visualize.efficient_frontier(stock_handler, ef_count, ef_bias, ef_seed, ef_tickers)
