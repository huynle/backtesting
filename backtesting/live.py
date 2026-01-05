import pandas as pd
import numpy as np
import datetime
from collections import UserDict
from typing import Dict, Optional

from backtesting import Backtest
from backtesting._stats import compute_stats
from backtesting._util import _Indicator, try_, _Array
from backtesting.backtesting import _Broker, Strategy, _OutOfMoneyError


class LiveMarketOhlcv:
    symbol: str = ""
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    total_qty: float = 0.0

    timestamp_second_start: int = 0
    duration_second: int = 60

    def __init__(
        self,
        timestamp_second=0,
        open_price=0.0,
        high_price=0.0,
        low_price=0.0,
        close_price=0.0,
        total_qty=0.0,
        symbol="",
    ) -> None:
        self.timestamp_second_start = timestamp_second
        self.symbol = symbol
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.total_qty = total_qty

    def __repr__(self) -> str:
        dt = datetime.datetime.fromtimestamp(self.timestamp_second_start)
        ps1 = f"o={self.open_price}, h={self.high_price}, "
        ps2 = f"l={self.low_price}, c={self.close_price}, v={self.total_qty}"
        symbol = f"[{self.symbol}]" if type(self.symbol) is str and len(self.symbol) > 0 else ""
        return f"LiveMarketOhlcv {symbol}[{dt}]: {ps1}{ps2}"


class _DataCachePatch:
    """
    A data array accessor. Provides access to OHLCV "columns"
    as a standard `pd.DataFrame` would, except it's not a DataFrame
    and the returned "series" are _not_ `pd.Series` but `np.ndarray`
    for performance reasons.
    """

    def __init__(self, df: pd.DataFrame):
        self.__df = df
        self.__i = len(df)
        self.__pip: Optional[float] = None
        self.__cache: Dict[str, np.ndarray] = {}
        self.__arrays: Dict[str, _Array] = {}
        self._update()

    def __getitem__(self, item):
        return self.__get_array(item)

    def __getattr__(self, item):
        try:
            return self.__get_array(item)
        except KeyError:
            raise AttributeError(f"Column '{item}' not in data") from None

    def _set_length(self, i):
        self.__i = i
        self.__cache.clear()

    def _update(self):
        index = self.__df.index.copy()
        self.__arrays = {col: _Array(arr, index=index) for col, arr in self.__df.items()}
        # Leave index as Series because pd.Timestamp nicer API to work with
        self.__arrays["__index"] = index

    def __repr__(self):
        i = min(self.__i, len(self.__df) - 1)
        index = self.__df.index[i]
        items = ", ".join(f"{k}={v}" for k, v in self.__df.iloc[i].items())
        return f"<Data i={i} ({index}) {items}>"

    def __len__(self):
        return self.__i

    @property
    def df(self) -> pd.DataFrame:
        return self.__df.iloc[: self.__i] if self.__i < len(self.__df) else self.__df

    def __get_array(self, key) -> np.ndarray:
        arr = self.__cache.get(key, None)
        if arr is None or len(arr) != len(self):
            farr: np.ndarray = self.__df[key].values[: self.__i]
            arr = self.__cache[key] = farr
        return arr

    @property
    def Open(self) -> np.ndarray:
        return self.__get_array("Open")

    @property
    def High(self) -> np.ndarray:
        return self.__get_array("High")

    @property
    def Low(self) -> np.ndarray:
        return self.__get_array("Low")

    @property
    def Close(self) -> np.ndarray:
        return self.__get_array("Close")

    @property
    def Volume(self) -> np.ndarray:
        return self.__get_array("Volume")

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.__df.index[: self.__i]

    # Make pickling in Backtest.optimize() work with our catch-all __getattr__
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


class _ObjectBindingDict(UserDict):
    def __init__(self, obj, is_key_func=lambda _: True):
        self.obj = obj
        self.is_key_func = is_key_func

    def __iter__(self):
        return iter({k: v for k, v in self.obj.__dict__.items() if self.is_key_func(v)}.items())

    def __getitem__(self, key):
        return getattr(self.obj, key)

    def __setitem__(self, key, item) -> None:
        return setattr(self.obj, key, item)


def set_indicator_indices(strategy, indicator_attrs, idx):
    for attr, indicator in indicator_attrs:
        # Slice indicator on the last dimension (case of 2d indicator)
        setattr(strategy, attr, indicator[..., : idx + 1])


class LiveTrade:
    def __init__(self, backtest: Backtest) -> None:
        self.core = backtest

    def init(self, **kwargs):
        self._df = self.core._data.copy(deep=False)
        self.data = _DataCachePatch(self._df)
        self.broker: _Broker = self.core._broker(data=self.data)
        self.strategy: Strategy = self.core._strategy(self.broker, self.data, kwargs)

        self._mirror_data = _DataCachePatch(self._df)
        self._mirror_strategy = self.core._strategy(self.broker, self._mirror_data, kwargs)
        self._mirror_strategy.init()

        self.data._update()  # Strategy.init might have changed/added to data.df

        self.indicator_attrs = _ObjectBindingDict(
            self._mirror_strategy, lambda x: isinstance(x, _Indicator)
        )
        _ = {
            attr: indicator
            for attr, indicator in self.strategy.__dict__.items()
            if isinstance(indicator, _Indicator)
        }.items()

        self.start_idx = 1 + max(
            (
                np.isnan(indicator.astype(float)).argmin(axis=-1).max()
                for _, indicator in self.indicator_attrs
            ),
            default=0,
        )

        self._current_idx = self.start_idx

    def on_bar(self, bar: LiveMarketOhlcv):
        ts_idx = pd.Timestamp(bar.timestamp_second_start, unit="s")
        df = self._df
        last_ts = df.index[-1]
        if ts_idx <= last_ts:
            return
        data_to_set = {
            "Open": bar.open_price,
            "High": bar.high_price,
            "Low": bar.low_price,
            "Close": bar.close_price,
            "Volume": bar.total_qty,
        }
        if type(df.iloc[0, 0]) is float or np.issubdtype(type(df.iloc[0, 0]), np.floating):
            df.loc[ts_idx] = {k: float(v) for k, v in data_to_set.items()}
        else:
            df.loc[ts_idx] = data_to_set
        self.broker._equity = np.append(self.broker._equity, [self.broker._equity[-1]])
        # for performance issue, this should be update partially
        # all indicators are re-calculate here
        self._mirror_data._set_length(len(df))
        self._mirror_strategy.init()

    def run_next(self):
        err = self._run_with_ith_bar(self._current_idx)
        if err is None:
            self._current_idx += 1

    def run(self, to_end=True, ntimes=0):
        if not to_end:
            ntimes = ntimes if ntimes > 0 else max(0, len(self._df) - self._current_idx)
            ntimes = min(ntimes, len(self._df))
        with np.errstate(invalid="ignore"):
            if to_end:
                iterator = range(self._current_idx, len(self._df))
            else:
                iterator = range(self._current_idx, self._current_idx + ntimes)
            for idx in iterator:
                err = self._w_run_with_ith_bar(idx)
                if err is not None:  # outof-money
                    break
        self._current_idx = len(self._df)  # not sure in live mode, what to do if no money

    def close_last_positions(self):
        for trade in self.broker.trades:
            trade.close()
        if self.start_idx < len(self._df):
            try_(self.broker.next, exception=_OutOfMoneyError)

    def process_pending_orders(self):
        sz = self.broker.position.size
        self.broker._process_orders()
        sz_a = self.broker.position.size
        return sz_a - sz

    def get_final_state(self):
        equity = pd.Series(self.broker._equity).bfill().fillna(self.broker._cash).values
        return compute_stats(
            trades=self.broker.closed_trades,
            equity=equity,
            ohlc_data=self._df,
            risk_free_rate=0.0,
            strategy_instance=self.strategy,
        )

    def _run_with_ith_bar(self, idx):
        if idx >= len(self._df) or idx < self.start_idx:
            return False
        with np.errstate(invalid="ignore"):
            self._w_run_with_ith_bar(idx)

    def _w_run_with_ith_bar(self, idx):
        self.data._set_length(idx + 1)
        set_indicator_indices(self.strategy, self.indicator_attrs, idx)
        try:
            self.broker.next()
        except _OutOfMoneyError:
            return False
        self.strategy.next()
