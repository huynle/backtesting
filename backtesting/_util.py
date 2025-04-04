from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager
from functools import partial
from itertools import chain
from datetime import datetime
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm
from numbers import Number
from threading import Lock
from typing import Callable, Dict, List, Optional, Sequence, Union, cast
from pandas_ta import AnalysisIndicators

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
    _tqdm = partial(_tqdm, leave=False)
except ImportError:
    def _tqdm(seq, **_):
        return seq


def try_(lazy_func, default=None, exception=Exception):
    try:
        return lazy_func()
    except exception:
        return default


@contextmanager
def patch(obj, attr, newvalue):
    had_attr = hasattr(obj, attr)
    orig_value = getattr(obj, attr, None)
    setattr(obj, attr, newvalue)
    try:
        yield
    finally:
        if had_attr:
            setattr(obj, attr, orig_value)
        else:
            delattr(obj, attr)


def _as_str(value) -> str:
    if isinstance(value, (Number, str)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.attrs.get('name', None) or 'df'
    name = str(getattr(value, 'name', '') or '')
    if name in ('Open', 'High', 'Low', 'Close', 'Volume'):
        return name[:1]
    if callable(value):
        name = getattr(value, '__name__', value.__class__.__name__).replace('<lambda>', 'λ')
    if len(name) > 10:
        name = name[:9] + '…'
    return name


def _as_list(value) -> List:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value]


def _batch(seq):
    # XXX: Replace with itertools.batched
    n = np.clip(int(len(seq) // (os.cpu_count() or 1)), 1, 300)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _data_period(index) -> Union[pd.Timedelta, Number]:
    """Return data index period as pd.Timedelta"""
    values = pd.Series(index[-100:])
    return values.diff().dropna().median()


def _strategy_indicators(strategy):
    # return {attr: indicator
    #         for attr, indicator in strategy.__dict__.items()
    #         if isinstance(indicator, _Indicator)}.items()
    result = {}
    for attr, indicator in strategy.__dict__.items():
        if isinstance(indicator, _Indicator):
            result[attr] = indicator
    return result.items()



    # result = {}
    # for attr, indicator in strategy.__dict__.items():
    #     if any([indicator is item for item in strategy._indicators]):
    #         result[attr] = indicator
    # return result

def _indicator_warmup_nbars(strategy):
    if strategy is None:
        return 0
    nbars = max((np.isnan(indicator.astype(float)).argmin(axis=-1).max()
                 for _, indicator in _strategy_indicators(strategy)
                 if not indicator._opts['scatter']), default=0)
    return nbars


class _Array(np.ndarray):
    """
    ndarray extended to supply .name and other arbitrary properties
    in ._opts dict.
    """
    def __new__(cls, array, *, name=None, **kwargs):
        obj = np.asarray(array).view(cls)
        obj.name = name or array.name
        obj._opts = kwargs
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.name = getattr(obj, 'name', '')
            self._opts = getattr(obj, '_opts', {})

    # Make sure properties name and _opts are carried over
    # when (un-)pickling.
    def __reduce__(self):
        value = super().__reduce__()
        return value[:2] + (value[2] + (self.__dict__,),)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])
        super().__setstate__(state[:-1])

    def __bool__(self):
        try:
            return bool(self[-1])
        except IndexError:
            return super().__bool__()

    def __float__(self):
        try:
            return float(self[-1])
        except IndexError:
            return super().__float__()

    def to_series(self):
        warnings.warn("`.to_series()` is deprecated. For pd.Series conversion, use accessor `.s`")
        return self.s

    @property
    def s(self) -> pd.Series:
        values = np.atleast_2d(self)
        index = self._opts['index'][:values.shape[1]]
        return pd.Series(values[0], index=index, name=self.name)

    @property
    def df(self) -> pd.DataFrame:
        values = np.atleast_2d(np.asarray(self))
        index = self._opts['index'][:values.shape[1]]
        df = pd.DataFrame(values.T, index=index, columns=[self.name] * len(values))
        return df


class _Indicator(_Array):
    pass


class _Data:
    """
    A data array accessor. Provides access to OHLCV "columns"
    as a standard `pd.DataFrame` would, except it's not a DataFrame
    and the returned "series" are _not_ `pd.Series` but `np.ndarray`
    for performance reasons.
    
    This implementation supports both single-asset and multi-asset data through
    a two-level column index structure, where the first level represents tickers
    and the second level represents OHLCV columns.
    """
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._len = len(df)  # Current length
        self._pip: Optional[float] = None
        self._cache: Dict[str, _Array] = {}
        self._arrays: Dict[str, _Array] = {}
        self._tickers = list(self._df.columns.levels[0])
        self._ta = _TA(self._df)
        self._update()

    def __getitem__(self, item):
        return self._get_array(item)

    def __getattr__(self, item):
        try:
            return self._get_array(item)
        except KeyError:
            raise AttributeError(f"Column '{item}' not in data") from None

    def _set_length(self, length):
        self._len = length
        self._cache.clear()

    def _update(self):
        # Cache slices of the data as DataFrame/Series for faster access
        arrays = (
            {ticker_col: arr for ticker_col, arr in self._df.items()}
            | {col: self._df.xs(col, axis=1, level=1) for col in self._df.columns.levels[1]}
            | {ticker: self._df[ticker] for ticker in self._df.columns.levels[0]}
            | {None: self._df[self.the_ticker] if len(self._tickers) == 1 else self._df}
            | {'__index': self._df.index.copy()}
        )
        arrays = {key: df.iloc[:, 0] if isinstance(df, pd.DataFrame) and len(
            df.columns) == 1 else df for key, df in arrays.items()}
        # Keep another copy as Numpy array
        self._arrays = {key: (df.to_numpy(), df) for key, df in arrays.items()}

    def __repr__(self):
        i = min(self._len, len(self._df)) - 1
        index = self._arrays['__index'][0][i]
        items = ', '.join(f'{k}={v}' for k, v in self._df.iloc[i].items())
        return f'<Data i={i} ({index}) {items}>'

    def __len__(self):
        return self._len

    @property
    def df(self) -> pd.DataFrame:
        df_ = self._df[self.the_ticker] if len(self.tickers) == 1 else self._df
        return df_.iloc[:self._len] if self._len < len(df_) else df_

    @property
    def pip(self) -> float:
        """
        Returns the smallest price unit of change as determined by the decimal precision
        of the Close prices.
        """
        if self._pip is None:
            self._pip = float(10**-np.median([len(s.partition('.')[-1])
                                               for s in self._arrays['Close'][0].ravel().astype(str)]))
        return self._pip

    def _get_array(self, key) -> _Array:
        """
        Retrieves array data for the specified key, using cached values when available.
        """
        arr = self._cache.get(key)
        if arr is None:
            array, df = self._arrays[key]
            arr = self._cache[key] = _Array(df.values[:self._len], name=key, index=self.index)
        return arr

    @property
    def Open(self) -> _Array:
        """Returns Open price data as an _Array."""
        return self._get_array('Open')

    @property
    def High(self) -> _Array:
        """Returns High price data as an _Array."""
        return self._get_array('High')

    @property
    def Low(self) -> _Array:
        """Returns Low price data as an _Array."""
        return self._get_array('Low')

    @property
    def Close(self) -> _Array:
        """Returns Close price data as an _Array."""
        return self._get_array('Close')

    @property
    def Volume(self) -> _Array:
        """Returns Volume data as an _Array."""
        return self._get_array('Volume')

    @property
    def index(self) -> pd.DatetimeIndex:
        # return self.__get_array('__index').df   # return pd.DatetimeIndex
        return self._df.index[:self._len]

    # Make pickling in Backtest.optimize() work with our catch-all __getattr__
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    @property
    def now(self) -> datetime:
        """Returns the timestamp of the current (last) bar."""
        return self.index[-1]

    @property
    def tickers(self) -> List[str]:
        """Returns the list of tickers available in the data."""
        return self._tickers

    @property
    def the_ticker(self) -> str:
        """
        Returns the single ticker when only one is available.
        Raises ValueError if multiple tickers exist, requiring explicit specification.
        """
        if len(self._tickers) == 1:
            return self._tickers[0]
        else:
            raise ValueError('Ticker must explicitly specified for multi-asset backtesting')

    @property
    def ta(self) -> '_TA':
        """Returns the technical analysis accessor for the data."""
        return self._ta



try:
    # delete the accessor created by pandas_ta to avoid warning
    del pd.DataFrame.ta
except AttributeError:
    pass

if sys.version_info >= (3, 13):
    SharedMemory = _mpshm.SharedMemory
else:
    class SharedMemory(_mpshm.SharedMemory):
        # From https://github.com/python/cpython/issues/82300#issuecomment-2169035092
        __lock = Lock()

        def __init__(self, *args, track: bool = True, **kwargs):
            self._track = track
            if track:
                return super().__init__(*args, **kwargs)
            with self.__lock:
                with patch(_mprt, 'register', lambda *a, **kw: None):
                    super().__init__(*args, **kwargs)

        def unlink(self):
            if _mpshm._USE_POSIX and self._name:
                _mpshm._posixshmem.shm_unlink(self._name)
                if self._track:
                    _mprt.unregister(self._name, "shared_memory")


class SharedMemoryManager:
    """
    A simple shared memory contextmanager based on
    https://docs.python.org/3/library/multiprocessing.shared_memory.html#multiprocessing.shared_memory.SharedMemory
    """
    def __init__(self, create=False) -> None:
        self._shms: list[SharedMemory] = []
        self.__create = create

    def SharedMemory(self, *, name=None, create=False, size=0, track=True):
        shm = SharedMemory(name=name, create=create, size=size, track=track)
        shm._create = create
        # Essential to keep refs on Windows
        # https://stackoverflow.com/questions/74193377/filenotfounderror-when-passing-a-shared-memory-to-a-new-process#comment130999060_74194875  # noqa: E501
        self._shms.append(shm)
        return shm

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        for shm in self._shms:
            try:
                shm.close()
                if shm._create:
                    shm.unlink()
            except Exception:
                warnings.warn(f'Failed to unlink shared memory {shm.name!r}',
                              category=ResourceWarning, stacklevel=2)
                raise

    def arr2shm(self, vals):
        """Array to shared memory. Returns (shm_name, shape, dtype) used for restore."""
        assert vals.ndim == 1, (vals.ndim, vals.shape, vals)
        shm = self.SharedMemory(size=vals.nbytes, create=True)
        # np.array can't handle pandas' tz-aware datetimes
        # https://github.com/numpy/numpy/issues/18279
        buf = np.ndarray(vals.shape, dtype=vals.dtype.base, buffer=shm.buf)
        has_tz = getattr(vals.dtype, 'tz', None)
        buf[:] = vals.tz_localize(None) if has_tz else vals  # Copy into shared memory
        return shm.name, vals.shape, vals.dtype

    def df2shm(self, df):
        return tuple((
            (column, *self.arr2shm(values))
            for column, values in chain([(self._DF_INDEX_COL, df.index)], df.items())
        ))

    @staticmethod
    def shm2s(shm, shape, dtype) -> pd.Series:
        arr = np.ndarray(shape, dtype=dtype.base, buffer=shm.buf)
        arr.setflags(write=False)
        return pd.Series(arr, dtype=dtype)

    _DF_INDEX_COL = '__bt_index'

    @staticmethod
    def shm2df(data_shm):
        index_data = None
        data_dict = {}
        shm_map = {}
        shms_to_return = []

        for item in data_shm:
            col, name, shape, dtype = item
            # Create SharedMemory instance without tracking for read-only access in worker
            shm = SharedMemory(name=name, create=False, track=False)
            shm_map[name] = shm  # Keep reference to prevent premature release on some OS
            shms_to_return.append(shm)
            arr = SharedMemoryManager.shm2s(shm, shape, dtype)
            if col == SharedMemoryManager._DF_INDEX_COL:
                index_data = arr
            else:
                data_dict[col] = arr

        if index_data is None:
            raise ValueError("Index data not found in shared memory bundle.")

        df = pd.DataFrame(data_dict)

        # Check if original columns were MultiIndex tuples based on the keys stored
        if data_dict and all(isinstance(c, tuple) for c in data_dict.keys()):
             # Ensure columns are sorted correctly if necessary, though dict order is preserved >= 3.7
             df.columns = pd.MultiIndex.from_tuples(df.columns)

        # Reconstruct index
        df.index = index_data
        df.index.name = None  # Restore original state (index name is not stored)

        # Return df and the list of shm objects to keep refs
        return df, shms_to_return




class PicklableAnalysisIndicators(AnalysisIndicators):
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


@pd.api.extensions.register_dataframe_accessor("ta")
class _TA:
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(self, df: pd.DataFrame):
        if df.empty:
            return
        self.__df = df
        if self.__df.columns.nlevels == 2:
            self.__tickers = list(self.__df.columns.levels[0])
            self.__indicators = {ticker: PicklableAnalysisIndicators(df[ticker]) for ticker in self.__tickers}
        elif self.__df.columns.nlevels == 1:
            self.__tickers = []
            self.__indicator = PicklableAnalysisIndicators(df)
        else:
            raise AttributeError(
                f'df.columns can have at most 2 levels, got {self.__df.columns.nlevels}')

    def __ta(self, method, *args, columns=None, **kwargs):
        if self.__tickers:
            dir_ = {ticker: getattr(indicator, method)(*args, **kwargs)
                    for ticker, indicator in self.__indicators.items()}
            if columns:
                for _, df in dir_.items():
                    df.columns = columns
            return pd.concat(dir_, axis=1) if len(dir_) > 1 else dir_[self.__tickers[0]]
        else:
            return getattr(self.__indicator, method)(*args, **kwargs)

    def __getattr__(self, method: str):
        return partial(self.__ta, method)

    def apply(self, func, *args, **kwargs):
        if self.__tickers:
            dir_ = {ticker: func(self.__df[ticker], *args, **kwargs) for ticker in self.__tickers}
            return pd.concat(dir_, axis=1)
        else:
            return func(self.__df, *args, **kwargs)

    def join(self, df, lsuffix='', rsuffix=''):
        if self.__tickers:
            dir_ = {ticker: self.__df[ticker].join(df[ticker], lsuffix=lsuffix, rsuffix=rsuffix)
                    for ticker in self.__tickers}
            return pd.concat(dir_, axis=1)
        else:
            return self.__df.join(df, lsuffix=lsuffix, rsuffix=rsuffix)
