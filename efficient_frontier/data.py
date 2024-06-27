import pandas_datareader.data as pdr
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

from enum import Enum


class Country(Enum):
    US = 'us'
    JP = 'jp'


class CalcMode(Enum):
    Simple = 'simple'
    Log = 'log'


RISK_FREE_SYMBOLS = {Country.US: '^TNX', Country.JP: 'IRLTLT01JPM156N'}


def is_valid_ticker(country: Country, ticker: str) -> bool:
    try:
        stock = yf.Ticker(ticker)
        if not stock.info or 'regularMarketVolume' not in stock.info:
            return False
        return True
    except (ValueError, KeyError):
        return False


def drop_outliers(df: pd.DataFrame, border_sigma: int = 5) -> pd.DataFrame:
    mean = df.mean()
    std = df.std()
    up_bound = mean + std * border_sigma
    btm_bound = mean - std * border_sigma
    mask = (df < btm_bound) | (df > up_bound)
    return df[~mask.any(axis=1)]


def get_rand_weights(n_samples: int, n_weights: int, seed: int = 7) -> np.ndarray:
    alpha = np.ones(n_weights)
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha, n_samples)


def log_to_simple_return(val: np.ndarray, out_format: str | None = "percent") -> np.ndarray:
    mul_val = 100 if out_format == "percent" else 1
    return (np.exp(val) - 1) * mul_val


def get_risk_free_data(country: Country, start: dt.datetime, end: dt.datetime) -> pd.Series:
    if country == Country.US:
        res: pd.Series = yf.download(
            tickers=RISK_FREE_SYMBOLS[country],
            start=start,
            end=end,
        )['Close']
    else:
        res: pd.Series = pdr.DataReader(RISK_FREE_SYMBOLS[country], 'fred', start, end)
    return res


def get_usd_jpy(start: dt.datetime, end: dt.datetime, fill_mode: str | None = 'ffill') -> pd.Series:
    sr_rate = yf.download(
        tickers="JPY=X",
        start=start,
        end=end,
    )['Close']
    if fill_mode == 'ffill':
        sr_rate.ffill(inplace=True)
    elif fill_mode == 'bfill':
        sr_rate.bfill(inplace=True)
    return sr_rate


def get_stock_data(
    country: Country, tickers: list[str], start: dt.datetime, end: dt.datetime, fill_mode: str | None = 'ffill'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns tuple of historical price data and dividend based on the given country code and tickers

    Args:
        country (Country): Contry code. Currently supported [jp, us]
        tickers (list[str]): List of ticker symbols
        start (dt.datetime): Starte date
        end (dt.datetime): End date
        fill_mode (str): Fill mode of NANs. Supported [None, ffill, bfill]

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Historical price data, historical dividend data
    """

    if country == Country.US:
        raw_df = yf.download(tickers=tickers, start=start, end=end, actions=True)
    else:
        tickers_jp = [ticker + '.JP' for ticker in tickers]
        raw_df = pdr.DataReader(tickers_jp, 'stooq', start, end)

    if fill_mode == 'ffill':
        raw_df.ffill(inplace=True)
    elif fill_mode == 'bfill':
        raw_df.bfill(inplace=True)

    df_close = raw_df['Close']
    if isinstance(df_close, pd.Series):
        df_close = df_close.to_frame(tickers[0])
    df_close = df_close[tickers]

    df_dividends = raw_df['Dividends']
    if isinstance(df_dividends, pd.Series):
        df_dividends = df_dividends.to_frame(tickers[0])
    df_dividends = df_dividends[tickers]

    return df_close, df_dividends


def calc_return_capital(df_price: pd.DataFrame, calc_mode: CalcMode):
    _shift = df_price.shift()
    _shift.iloc[0, :] = _shift.iloc[1, :]

    if calc_mode == CalcMode.Log:
        df_return = np.log(df_price / _shift)
    else:
        df_return = df_price / _shift - 1
    # np.log((df_close + df_dividends) / _shift)
    # if exclude_anomaly:
    #     df_return = drop_outliers(df_return, anomaly_sigma)

    return df_return


def calc_diviednd_return(df_dividend: pd.DataFrame, df_price: pd.DataFrame, calc_mode: CalcMode):
    if calc_mode == CalcMode.Log:
        df_return = np.log((df_dividend + df_price) / df_price)
    else:
        df_return = (df_dividend + df_price) / df_price

    return df_return


def calc_stock_performance(
    df_capital_return: pd.DataFrame, df_dividend_return: pd.DataFrame, calc_mode: CalcMode, business_days: int = 251
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # # Calc log return
    # _shift = df_price.shift()
    # _shift.iloc[0, :] = _shift.iloc[1, :]
    # df_log_return = np.log(df_price / _shift)
    # # np.log((df_close + df_dividends) / _shift)
    # if exclude_anomaly:
    #     df_log_return = drop_outliers(df_log_return, anomaly_sigma)

    # # Calc return
    # df_performance = pd.DataFrame(
    #     {
    #         "capital": df_log_return.mean() * business_days,
    #         "income": np.log((df_dividend + df_price) / df_price).mean(axis=0) * business_days,
    #     }
    # )
    if calc_mode == CalcMode.Log:
        df_performance = pd.DataFrame(
            {
                "capital": df_capital_return.mean(axis=0) * business_days,
                "income": df_dividend_return.mean(axis=0) * business_days,
            }
        )
        # Calc Cross Cov
        df_cov_matrix = df_capital_return.cov() * business_days

    else:
        df_performance = pd.DataFrame(
            {
                "capital": np.exp(np.mean(np.log(df_capital_return + 1), axis=0) * business_days),
                "income": np.exp(np.mean(np.log(df_dividend_return), axis=0) * business_days),
            }
        )
        # Calc Cross Cov
        df_cov_matrix = df_capital_return.cov() * business_days

    df_performance["total"] = df_performance["capital"] + df_performance["income"]

    df_performance['std'] = np.sqrt(np.diagonal(df_cov_matrix))

    return df_performance, df_cov_matrix


def calc_sharpe_ratio(list_return: np.ndarray, risk_free_rate: float, list_str: np.ndarray, log_return: bool = True):
    if log_return:
        sharpe_ratio = (log_to_simple_return(list_return, out_format=None) - risk_free_rate) / list_str
    else:
        sharpe_ratio = (list_return - risk_free_rate) / list_str

    return sharpe_ratio


def calc_portfolios(
    df_performance: pd.DataFrame,
    df_cov_matrix: pd.DataFrame,
    df_weights: pd.DataFrame,
    log_return: bool = True,
    risk_free_rate: float | None = None,
) -> pd.DataFrame:
    weights = df_weights.to_numpy()
    ret_list = weights @ df_performance['total'].to_numpy()
    std_list = np.sqrt(((weights @ df_cov_matrix.to_numpy()) * weights).sum(axis=1))
    result = pd.DataFrame({"return": ret_list, "std": std_list})
    # Optional: Sharpe Ratio
    if risk_free_rate:
        result['sharpe_ratio'] = calc_sharpe_ratio(ret_list, risk_free_rate, std_list, log_return=log_return)
    return result
