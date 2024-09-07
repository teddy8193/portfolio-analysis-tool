from enum import Enum, auto
from dataclasses import dataclass, field
import pandas as pd
import datetime as dt


class Country(Enum):
    US = 'us'
    JP = 'jp'


class CalcMode(Enum):
    Simple = 'simple'
    Compound = 'compound'


class Term(Enum):
    JAN = 1
    FEB = auto()
    MAR = auto()
    APR = auto()
    MAY = auto()
    JUN = auto()
    JUL = auto()
    AUG = auto()
    SEP = auto()
    OCT = auto()
    NOV = auto()
    DEC = auto()


@dataclass
class TickerItem:
    _name: str  # Ticker name user typed
    _country: Country = Country.US
    color: str = "#FFFFFF"
    _code: str = ""  # Ticker code to search with API

    def __post_init__(self):
        self._code = self._name
        if isinstance(self._code, str):
            self._code = self._code.upper()
        self._validate_code_country()

    def _validate_code_country(self):
        if self._country == Country.JP and not self._code.endswith(".T"):
            self._code += ".T"
        elif self._country == Country.US and self._code.endswith(".T"):
            self._code = self._code.removesuffix(".T")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value
        self._code = self._name
        if isinstance(self._code, str):
            self._code = self._code.upper()
        self._validate_code_country()

    @property
    def code(self) -> str:
        return self._code

    @property
    def country(self) -> Country:
        return self._country

    @country.setter
    def country(self, value: Country):
        self._country = value
        self._validate_code_country()

    @classmethod
    def new(cls, name: str = "", country: Country = Country.US, color: str = "#FFFFFF") -> 'TickerItem':
        return TickerItem(name, country, color)


@dataclass
class StockHistorical:
    tickers: list[TickerItem]
    df_stock: pd.DataFrame
    start: dt.date
    end: dt.date
    fiscal_month: Term
    # fiscal_start: dt.date
    # fiscal_end: dt.date

    @property
    def price(self) -> pd.DataFrame:
        return self.df_stock['price']

    @property
    def dividend(self) -> pd.DataFrame:
        return self.df_stock['dividend']
