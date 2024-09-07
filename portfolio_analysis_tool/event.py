import streamlit as st

import uuid
import datetime as dt

from portfolio_analysis_tool import data, utils, log, models

logger = log.logger


def on_ticker_added():
    logger.debug('on_ticker_added')
    st.session_state.ticker_ids.append(uuid.uuid4())


def on_ticker_deleted(id: uuid.UUID):
    logger.debug(f'on_ticker_deleted {id}')

    if (not id) or (id not in st.session_state.ticker_ids):
        return

    if id in st.session_state.ticker_ids:
        st.session_state.ticker_ids.remove(id)


def on_specific_date_clicked():
    st.session_state.date_specific = st.session_state.check_date_specific


def on_visualize_pressed():
    logger.debug('on_visualize_pressed')
    start = st.session_state.date_start
    end = st.session_state.date_end
    if not isinstance(start, dt.date):
        start = dt.date(int(start), 1, 1)
    if not isinstance(end, dt.date):
        end = dt.date(int(end), 12, 31)

    # Collect ticker symbols
    id: uuid.UUID
    tickers: list[models.TickerItem] = []
    for id in st.session_state.ticker_ids:
        key_country = f"country_{id}"
        key_ticker = f"ticker_{id}"
        key_color = f"color_{id}"
        if any([key_country not in st.session_state, key_ticker not in st.session_state, key_color not in st.session_state]):
            continue
        country = st.session_state[key_country]
        ticker_name = st.session_state[key_ticker]
        color = st.session_state[key_color]
        if ticker_name:
            ticker_item = models.TickerItem.new(ticker_name, models.Country[country], color=color)
            tickers.append(ticker_item)

    st.session_state.no_symbols = True
    st.session_state.invalid_tickers = []

    if not tickers:
        return

    stock_handler: data.StockDataHandler = st.session_state.stock_handler
    term_start_month = models.Term[st.session_state.fiscal_start].value
    tickers_loaded = stock_handler.load_data(tickers, start, end, term_start_month=term_start_month)

    if not tickers_loaded:
        return

    st.session_state.no_symbols = False

    ticker_codes_orig = set([t.code for t in tickers])
    ticker_codes_new = set([t.code for t in tickers_loaded])
    invalid_tickers = ticker_codes_orig - ticker_codes_new

    if invalid_tickers:
        st.session_state.invalid_tickers = list(invalid_tickers)
        return
