import pandas_datareader.data as pdr
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_analysis_tool.models import Country, CalcMode, Term, TickerItem, StockHistorical
from portfolio_analysis_tool import log

logger = log.logger

RISK_FREE_SYMBOLS = {Country.US: '^TNX', Country.JP: 'IRLTLT01JPM156N'}


class StockDataHandler:
    def __init__(self) -> None:
        pass

        self.term = Term.JAN
        self.colors = None
        self.pct_from_first = None
        self.yearly_return = None
        self.yearly_risk = None
        self.yearly_dividend = None
        self.mean_return = None
        self.mean_risk = None
        self.cov = None
        self.corr = None
        self.mean_risk_free = 0
        self.tickers = None

    def load_data(self, tickers: list[TickerItem], start: dt.date, end: dt.date, term_start_month: int) -> list[TickerItem | None]:
        """Load and calculate stock data and performance, store them in instance variable.

        Args:
            tickers (list[TickerItem]): Tickers to load data
            start (dt.date): Start date to load
            end (dt.date): End date to load
            term_start_month (int): Fiscal term to calculate annual performance

        Returns:
            list[TickerItem]: List of ticker items which are successfully loaded
        """
        historical_data: StockHistorical | None = get_stock_data(tickers, start, end, term_start_month)
        if not historical_data:
            return []
        elif not historical_data.tickers:
            return []

        self.term = historical_data.fiscal_month
        self.colors = [t.color for t in historical_data.tickers]
        self.pct_from_first = calc_change_from_first(historical_data)
        self.yearly_return, self.yearly_risk = calc_yearly_return_risk(historical_data)
        self.yearly_dividend = calc_yearly_dividend(historical_data, mode=CalcMode.Simple)
        self.mean_return, self.mean_risk = calc_mean_return_risk(historical_data)
        self.cov, self.corr = cacl_cov_corr(historical_data)

        # print("######## pct_from_first ########", self.pct_from_first.head(5))
        # print("######## yearly_return ########", self.yearly_return.head(5))
        # print("######## yearly_risk ########", self.yearly_risk.head(5))
        # print("######## yearly_dividend ########", self.yearly_dividend.head(5))
        # print("######## mean_return ########", self.mean_return.head(5))
        # print("######## mean_risk ########", self.mean_risk.head(5))
        # print("######## cov ########", self.cov)
        # print("######## corr ########", self.corr)

        # Risk Free Rate
        df_risk_free = get_risk_free_data(Country.US, start, end)
        self.mean_risk_free = df_risk_free.mean() * 0.01

        self.tickers = historical_data.tickers
        return self.tickers


def drop_outliers(df: pd.DataFrame, border_sigma: int = 5) -> pd.DataFrame:
    mean = df.mean()
    std = df.std()
    up_bound = mean + std * border_sigma
    btm_bound = mean - std * border_sigma
    mask = (df < btm_bound) | (df > up_bound)
    return df[~mask.any(axis=1)]


def get_rand_weights(n_samples: int, n_weights: int, bias: float = 0.0, seed: int = 7) -> np.ndarray:
    alpha = np.full(shape=n_weights, fill_value=np.exp(-bias))
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha, n_samples)


def log_to_simple_return(val: np.ndarray, out_format: str | None = "percent") -> np.ndarray:
    mul_val = 100 if out_format == "percent" else 1
    return (np.exp(val) - 1) * mul_val


def get_risk_free_data(country: Country, start: dt.date, end: dt.date) -> pd.Series:
    if country == Country.US:
        res: pd.Series = yf.download(
            tickers=RISK_FREE_SYMBOLS[country],
            start=start,
            end=end,
        )['Close']
    else:
        res: pd.Series = pdr.DataReader(RISK_FREE_SYMBOLS[country], 'fred', start, end)
    return res


def get_usd_jpy(start: dt.date, end: dt.date, fill_mode: str | None = 'ffill') -> pd.Series:
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


def get_fiscal_start_end(start: dt.date, end: dt.date, term_start_month: int) -> tuple[dt.date, dt.date]:
    term_start_month = (term_start_month - 1) % 12 + 1
    term_end_month = (term_start_month - 2) % 12 + 1

    # Calc fiscal year start for this year and last year
    past_year_term_start = start.replace(year=start.year - 1, month=term_start_month, day=1)
    this_year_term_start = start.replace(year=start.year, month=term_start_month, day=1)

    # Calc fiscal year end for this year and next year
    this_year_term_end = end.replace(year=end.year, month=term_end_month, day=31)
    next_year_term_end = end.replace(year=end.year + 1, month=term_end_month, day=31)

    fiscal_start = this_year_term_start if start >= this_year_term_start else past_year_term_start
    fiscal_end = this_year_term_end if end <= this_year_term_end else next_year_term_end

    return fiscal_start, fiscal_end


def get_stock_data(tickers: list[TickerItem], start: dt.date, end: dt.date, term_start_month: int) -> StockHistorical | None:
    fiscal_month: Term = Term((term_start_month - 1) % 12 + 1)
    # fiscal_start, fiscal_end = get_fiscal_start_end(start, end, term_start_month)

    stock_data = None

    ticker_map = {t.code: t for t in tickers}
    ticker_codes = list(ticker_map.keys())
    logger.info(f'Getting Stock Data with Tickers {ticker_codes}')
    logger.info(f'Start Date {start}  End Date {end}')

    # df: pd.DataFrame = yf.download(tickers=ticker_codes, start=fiscal_start, end=fiscal_end, actions=True)
    df: pd.DataFrame = yf.download(tickers=ticker_codes, start=start, end=end, actions=True)

    # Skip if not valid dataframe
    if not len(df) > 0:
        return None

    # Converting to multi index if it's not
    if not isinstance(df.columns, pd.MultiIndex):
        multi_index = pd.MultiIndex.from_product(
            [
                df.columns,
                ticker_codes,
            ]
        )
        df.columns = multi_index

    # Column Validation for Dividend column
    if "Dividends" not in df.columns:
        for ticker_code in ticker_codes:
            df[('Dividends', ticker_code)] = 0
    else:
        for ticker_code in ticker_codes:
            if ticker_code not in df['Dividends'].columns:
                df[('Dividends', ticker_code)] = 0

    # Drop column if all cels are NAN
    ticker_to_drop = df['Close'].columns[df['Close'].isna().all()].tolist()

    df_close: pd.DataFrame = df['Close'].drop(ticker_to_drop, axis=1)
    df_dividends: pd.DataFrame = df['Dividends'].drop(ticker_to_drop, axis=1)

    # Skip if not valid dataframe
    if df_close.columns.size == 0:
        return None

    # Fill for price data
    # if fill_mode == 'ffill':
    #     df_close.ffill(inplace=True)
    # elif fill_mode == 'bfill':
    #     df_close.bfill(inplace=True)

    # Fill for dividend data
    df_dividends.fillna(0, inplace=True)
    current_ticker_codes = df_close.columns.to_list()
    df_merged = pd.concat([df_close, df_dividends], axis=1, keys=['price', 'dividend'])

    stock_data = StockHistorical(
        tickers=[ticker_map[t] for t in current_ticker_codes],
        df_stock=df_merged,
        start=start,
        end=end,
        fiscal_month=fiscal_month,
    )

    return stock_data


def calc_change_from_first(stock_data: StockHistorical) -> pd.DataFrame:
    df_price: pd.DataFrame = stock_data.price
    df_dividend: pd.DataFrame = stock_data.dividend

    first_prices = df_price.bfill().iloc[0, :]
    price_with_div = df_price.add(df_dividend.cumsum(), fill_value=0)
    change = df_price / first_prices - 1
    change_with_div = price_with_div / first_prices - 1

    return pd.concat([change, change_with_div], axis=1, keys=['no_div', 'with_div'])


def calc_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    res = df.ffill().bfill().pct_change() + 1  # Fill NaA from forward value, then backward value
    res[df.isna()] = np.nan  # Resetting NaN where the original value was NaN

    return res


def calc_weight_per_year(df: pd.DataFrame, term_start: Term) -> pd.Series:
    return df.resample(f'YS-{term_start.name}').count() / df.count()


def calc_yearly_return_risk(stock_data: StockHistorical, business_days: int = 250) -> tuple[pd.DataFrame, pd.DataFrame]:
    def prod_with_nan_as_one(array):
        array = np.nan_to_num(array, nan=1)
        return np.prod(array)

    term_start: Term = stock_data.fiscal_month
    df_price: pd.DataFrame = stock_data.price
    df_pct = calc_pct_change(df_price)
    is_na = df_pct.isna()
    # df_ret: pd.DataFrame = df_pct.resample(f'YS-{term_start.name}').prod() - 1
    df_ret: pd.DataFrame = df_pct.ffill().bfill().rolling(window='365D', min_periods=1).apply(prod_with_nan_as_one, raw=True) - 1
    # df_risk: pd.DataFrame = np.log(df_pct).resample(f'YS-{term_start.name}').std() * np.sqrt(business_days)
    df_risk: pd.DataFrame = np.log(df_pct).rolling(window='365D', min_periods=1).std() * np.sqrt(business_days)
    # df_ret.index = df_ret.index.strftime('%Y-%m').astype('object')
    # df_risk.index = df_risk.index.strftime('%Y-%m').astype('object')

    return df_ret, df_risk


def calc_yearly_dividend(stock_data: StockHistorical, mode: CalcMode = CalcMode.Simple) -> pd.DataFrame:
    term_start: Term = stock_data.fiscal_month
    df_price: pd.DataFrame = stock_data.price
    df_dividend: pd.DataFrame = stock_data.dividend
    if mode == CalcMode.Simple:
        df_div_ret = df_dividend.resample(f'YS-{term_start.name}').sum() / df_price.resample(f'YS-{term_start.name}').mean()
    elif mode == CalcMode.Compound:
        div_ret_ratio = df_dividend / df_price.resample(f'YS-{term_start.name}').transform(lambda x: x.mean())
        df_div_ret = (div_ret_ratio + 1).resample(f'YS-{term_start.name}').prod() - 1
    df_div_ret.index = df_div_ret.index.strftime('%Y-%m').astype('object')

    return df_div_ret


def calc_mean_return_risk(stock_data: StockHistorical, business_days: int = 250) -> tuple[pd.Series, pd.Series]:
    df_price: pd.DataFrame = stock_data.price
    term_start: Term = stock_data.fiscal_month
    weights = calc_weight_per_year(df_price, term_start)
    df_pct = calc_pct_change(df_price)

    ret = np.exp((np.log(df_pct.resample(f'YS-{term_start.name}').prod()) * weights).sum(axis=0)) - 1
    risk = np.log(df_pct).std() * np.sqrt(business_days)

    return ret, risk


def cacl_cov_corr(stock_data: StockHistorical, business_days: int = 250) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_price: pd.DataFrame = stock_data.price
    df_pct = calc_pct_change(df_price)
    df_cov = np.log(df_pct).cov() * business_days
    df_corr = np.log(df_pct).corr()

    return df_cov, df_corr


def calc_sharpe_ratio(list_return: np.ndarray, risk_free_rate: float, list_std: np.ndarray) -> np.ndarray:
    """Calculate sharpe ratio by given array of returns, stds, and rist free rate

    Args:
        list_return (np.ndarray): Array of returns
        risk_free_rate (float): Rist free rate
        list_std (np.ndarray): Array of standard deviations

    Returns:
        np.ndarray: Array of sharpe ratio
    """
    return (list_return - risk_free_rate) / list_std


def calc_portfolios(
    return_array: np.ndarray, cov_matrix: np.ndarray, weights: np.ndarray, risk_free_rate: float | None = None
) -> pd.DataFrame:
    """Calculate portfolio performances based on given weights

    Args:
        return_array (np.ndarray): 1d array of return
        cov_matrix (np.ndarray): Covariance matrix of tickers
        weights (np.ndarray): Array of weights
        risk_free_rate (float | None, optional): Rist free rate for sharpe ratio. Defaults to None.

    Returns:
        pd.DataFrame: Return, risk, sharpe ratio(optional) per weight
    """

    ret_list = weights @ return_array
    std_list = np.sqrt(((weights @ cov_matrix) * weights).sum(axis=1))
    result = pd.DataFrame({"return": ret_list, "std": std_list})
    # Optional: Sharpe Ratio
    if risk_free_rate:
        result['sharpe_ratio'] = calc_sharpe_ratio(ret_list, risk_free_rate, std_list)
    return result
