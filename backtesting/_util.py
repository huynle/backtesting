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
from typing import Dict, List, Optional, Sequence, Union, cast, Tuple, Any
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

    raw_name = getattr(value, 'name', '') # Get the name attribute directly

    # Handle specific column names first (single string or tuple)
    if isinstance(raw_name, str) and raw_name in ('Open', 'High', 'Low', 'Close', 'Volume'):
        return raw_name[:1]
    elif isinstance(raw_name, tuple) and len(raw_name) == 2:
        # Check if the second element is a standard OHLCV column name
        col_name = raw_name[1]
        if isinstance(col_name, str) and col_name in ('Open', 'High', 'Low', 'Close', 'Volume'):
            return col_name[:1] # Return 'O', 'H', 'L', 'C', 'V'

    # Handle callable names
    if callable(value):
        name_str = getattr(value, '__name__', value.__class__.__name__).replace('<lambda>', 'λ')
    else:
        # Convert the raw name (could be tuple, string, etc.) to string
        # If it wasn't handled above as a specific OHLCV tuple, convert the whole thing
        name_str = str(raw_name or '') # Ensure it's a string

    # Removed truncation logic from previous step
    return name_str


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
    return {attr: indicator
            for attr, indicator in strategy.__dict__.items()
            if isinstance(indicator, _Indicator)}.items()


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
        # Capture name from original object if it exists, before potential conversion
        original_name = getattr(array, 'name', None)
        obj = np.asarray(array).view(cls)
        # Prioritize explicit name, then original name, fallback to empty string
        obj.name = name or original_name or ''
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
    
    This implementation supports both single-asset (input as a DataFrame which is
    converted to a single-entry dictionary) and multi-asset data (input as a
    dictionary of DataFrames, e.g., `{'TICKER1': df1, 'TICKER2': df2}`).
    Each DataFrame in the dictionary should share the same index.
    """
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.__df_dict = data_dict
        self._tickers = list(data_dict.keys())
        self._the_ticker = None
        if not self._tickers:
            raise ValueError("Data dictionary cannot be empty.")
        
        # Assume all DataFrames share the same index
        self._shared_index = data_dict[self._tickers[0]].index
        self.__len = len(self._shared_index) # Current length
        self.__pip: Optional[float] = None
        self.__cache: Dict[Tuple[str, str], _Array] = {} # Cache for (ticker, column)
        self.__arrays: Dict[Tuple[str, str], Tuple[np.ndarray, pd.Series]] = {} # Store for (ticker, column)
        self._ta = _TA(self.__df_dict) # Pass the dictionary
        self._update()

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            ticker, column = key
            if ticker not in self._tickers:
                raise KeyError(f"Ticker '{ticker}' not found in data.")
            if column not in self.__df_dict[ticker].columns:
                # Check if it's a dynamically added column via self.data.df[ticker]['new_col']
                # The _update method should handle this if calls it.
                # For now, assume direct column access.
                raise KeyError(f"Column '{column}' not found for ticker '{ticker}'.")
            return self._get_array(ticker, column)
        elif isinstance(key, str):
            if key in self._tickers:
                # Return a DataFrame slice for the specific ticker
                return self.__df_dict[key].iloc[:self.__len]
            elif len(self._tickers) == 1:
                # Single-asset mode, key is a column name
                ticker = self.the_ticker
                if key not in self.__df_dict[ticker].columns:
                     raise KeyError(f"Column '{key}' not found for ticker '{ticker}'.")
                return self._get_array(ticker, key).s # Return pd.Series via .s accessor
            else:
                # Multi-asset mode, key is a column name - ambiguous
                # Or key is a ticker name (already handled)
                raise KeyError(f"Accessing column '{key}' directly is ambiguous in multi-asset mode. Use data[ticker, column_name] or data[ticker]['{key}'].")
        elif isinstance(key, slice):
            if len(self._tickers) == 1:
                # Single-asset mode, return a slice of the DataFrame
                return self.__df_dict[self.the_ticker].iloc[:self.__len][key]
            else:
                # Slicing directly on multi-asset _Data is ambiguous
                raise TypeError(f"Slicing directly on multi-asset _Data is ambiguous. Slice a specific ticker's data instead, e.g., data['{self._tickers[0]}', 'Close'][-5:] or data['{self._tickers[0]}'][-5:].")
        raise TypeError(f"Invalid key type for _Data: {key}")

    def __setitem__(self, key, value):
        # This allows self.data['new_key'] = ... in single-asset mode for tests
        if len(self._tickers) == 1:
            ticker = self.the_ticker
            # Delegate to the DataFrame of the single asset
            # Ensure the value is aligned with the full original index if it's array-like
            if isinstance(value, (np.ndarray, pd.Series, list)):
                if len(value) == self.__len: # currently visible length
                    full_value = pd.Series(np.nan, index=self._shared_index)
                    full_value.iloc[:self.__len] = value
                    self.__df_dict[ticker][key] = full_value
                elif len(value) == len(self._shared_index): # full length
                    self.__df_dict[ticker][key] = value
                else:
                    raise ValueError(f"Length of value ({len(value)}) does not match current data length ({self.__len}) or full data length ({len(self._shared_index)})")
            else: # scalar
                self.__df_dict[ticker][key] = value
            self._update() # Re-cache arrays
        else:
            raise TypeError("Direct item assignment on _Data is only supported in single-asset mode. Use data[ticker]['column'] = ... or data.df[ticker]['column'] = ...")


    def __getattr__(self, item):
        # Allows self.data.CustomColumn in single-asset mode
        # Standard OHLCV access (e.g. self.data.Close) is handled by dedicated properties.
        if len(self._tickers) == 1:
            ticker = self.the_ticker
            if item in self.__df_dict[ticker].columns:
                return self._get_array(ticker, item)
        
        # Fallback for other attributes like 'pip', 'index', 'now', 'tickers', 'the_ticker', 'ta', 'df'
        # These should be defined as properties or methods. If not found, raise AttributeError.
        # This prevents infinite recursion if a property tries to access a non-existent attribute.
        if f'_{self.__class__.__name__}__{item}' in self.__dict__ or item in self.__dict__:
            return self.__dict__[item] # Should not happen if properties are used

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")


    def _set_length(self, length):
        self.__len = length
        self.__cache.clear()

    def _update(self):
        self.__arrays.clear()
        for ticker, df_ticker in self.__df_dict.items():
            for col_name, series in df_ticker.items():
                self.__arrays[(ticker, col_name)] = (series.to_numpy(), series)
        # Cache for __index is not strictly needed in __arrays if self._shared_index is used directly
        # but _get_array expects a tuple from self.__arrays if we were to unify.
        # For simplicity, self.index property will use self._shared_index.

    def __repr__(self):
        i = min(self.__len, len(self._shared_index)) - 1
        idx_val = self._shared_index[i]
        
        if len(self._tickers) == 1:
            ticker = self.the_ticker
            items_str = ', '.join(f'{k}={v}' for k, v in self.__df_dict[ticker].iloc[i].items())
            return f'<Data i={i} ({idx_val}) Ticker: {ticker} {items_str}>'
        else:
            return f'<Data i={i} ({idx_val}) Tickers: {self._tickers} (use data[ticker] or data[ticker, column])>'

    def __len__(self):
        return self.__len

    @property
    def df(self) -> pd.DataFrame:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.df` is ambiguous in a multi-asset context. Use `self.data[ticker].df`.")

        original_df = self.__df_dict[self.the_ticker]
        # In init(), self.__len is the full length of the data.
        # If current view length (__len) is the same as the original DataFrame's length,
        # return the original DataFrame. This allows modifications in init() (e.g., adding a new column)
        # to be made on the actual underlying DataFrame, which will then be picked up by _update().
        # Otherwise (e.g., in next()), return a slice.
        if self.__len == len(original_df.index):
            return original_df
        else:
            return original_df.iloc[:self.__len]

    @property
    def pip(self) -> float:
        if self.__pip is None:
            # Calculate pip based on the 'Close' prices of the first ticker
            # This might need adjustment if pips vary significantly across assets in a multi-asset scenario
            first_ticker = self._tickers[0]
            
            # self.__arrays stores (numpy_array_full_length, pandas_series_full_length)
            # Access the pandas_series_full_length (index 1) for string operations
            cache_key = (first_ticker, 'Close')
            if cache_key in self.__arrays:
                close_prices_series = self.__arrays[cache_key][1]
                self.__pip = float(10**-np.median([len(s.partition('.')[-1])
                                                   for s in close_prices_series.astype(str).values]))
            else:
                # 'Close' column not found for the first ticker, or __arrays not populated.
                # Default pip. This case should ideally be rare if data is validated.
                warnings.warn(f"Could not determine pip size from 'Close' column of ticker '{first_ticker}'. Defaulting to 0.01. Ensure 'Close' column exists.", UserWarning)
                self.__pip = 0.01 # Default pip
        return self.__pip

    def _get_array(self, ticker: str, column: str) -> _Array:
        cache_key = (ticker, column)
        arr = self.__cache.get(cache_key)
        if arr is None:
            try:
                # self.__arrays stores (numpy_array_full_length, pandas_series_full_length)
                np_array_full, pd_series_full = self.__arrays[cache_key]
                # Return a view of the numpy array up to the current length __len
                arr = self.__cache[cache_key] = _Array(np_array_full[:self.__len], name=column, index=self.index)
            except KeyError:
                # This might happen if a column was added dynamically and _update wasn't called,
                # or if the column truly doesn't exist.
                # The __getitem__ should ideally prevent calls for non-existent static columns.
                # If it's a new column added via df accessor, _update should have handled it.
                raise KeyError(f"Array for ticker '{ticker}', column '{column}' not found in internal arrays. Ensure it exists in the input data or was added correctly.")
        return arr

    @property
    def Open(self) -> _Array:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.Open` is ambiguous in a multi-asset context. Use `self.data[ticker, 'Open']`.")
        return self._get_array(self.the_ticker, 'Open')

    @property
    def High(self) -> _Array:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.High` is ambiguous in a multi-asset context. Use `self.data[ticker, 'High']`.")
        return self._get_array(self.the_ticker, 'High')

    @property
    def Low(self) -> _Array:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.Low` is ambiguous in a multi-asset context. Use `self.data[ticker, 'Low']`.")
        return self._get_array(self.the_ticker, 'Low')

    @property
    def Close(self) -> _Array:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.Close` is ambiguous in a multi-asset context. Use `self.data[ticker, 'Close']`.")
        return self._get_array(self.the_ticker, 'Close')

    @property
    def Volume(self) -> _Array:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.Volume` is ambiguous in a multi-asset context. Use `self.data[ticker, 'Volume']`.")
        # Volume might be optional, handle if not present for the ticker
        if 'Volume' not in self.__df_dict[self.the_ticker].columns:
             # Return an array of NaNs or zeros if Volume is missing
             # This matches behavior if Volume column was all NaNs
             vol_data = np.full(self.__len, np.nan)
             return _Array(vol_data, name='Volume', index=self.index)
        return self._get_array(self.the_ticker, 'Volume')

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._shared_index[:self.__len]

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
        elif self._the_ticker is not None:
            return self._the_ticker
        else:
            raise ValueError('Ticker must explicitly specified for multi-asset backtesting')

    @property
    def ta(self) -> '_TA':
        """Returns the technical analysis accessor for the data."""
        return self._ta

    def get_ticker_data(self, ticker=None) -> pd.DataFrame:
        """
        Returns a DataFrame for a specific ticker or the default ticker if none specified.
        
        Parameters:
        -----------
        ticker : str, optional
            The ticker symbol to retrieve data for. If not provided, uses the default ticker.
            
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing data for the specified ticker with all columns from the second level.
        
        Raises:
        -------
        ValueError
            If no ticker is provided and multiple tickers exist in the data.
        KeyError
            If the specified ticker does not exist in the data.
        """
        if ticker is None:
            # Use the default ticker if none provided
            ticker = self.the_ticker
        
        # Check if the ticker exists
        if ticker not in self._tickers:
            raise KeyError(f"Ticker '{ticker}' not found in data. Available tickers: {self._tickers}")
        
        # Return the DataFrame for the specified ticker
        return self.__df_dict[ticker]






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

    def df2shm(self, data_input: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        if isinstance(data_input, pd.DataFrame):
            # Handle single DataFrame
            return tuple((
                (column, *self.arr2shm(values))
                for column, values in chain([(self._DF_INDEX_COL, data_input.index)], data_input.items())
            ))
        elif isinstance(data_input, dict):
            # Handle Dict[str, pd.DataFrame]
            return {
                ticker: tuple((
                    (column, *self.arr2shm(values))
                    for column, values in chain([(self._DF_INDEX_COL, df.index)], df.items())
                ))
                for ticker, df in data_input.items()
            }
        else:
            raise TypeError("df2shm expects a pandas.DataFrame or a Dict[str, pd.DataFrame]")

    @staticmethod
    def shm2s(shm, shape, dtype) -> pd.Series:
        arr = np.ndarray(shape, dtype=dtype.base, buffer=shm.buf)
        arr.setflags(write=False)
        return pd.Series(arr, dtype=dtype)

    _DF_INDEX_COL = '__bt_index'

    @staticmethod
    def shm2df(data_shm_info: Union[tuple, Dict[str, tuple]]):
        if isinstance(data_shm_info, tuple): # Info for a single DataFrame
            shms = [SharedMemory(name=name, create=False, track=False) for _, name, _, _ in data_shm_info]
            df = pd.DataFrame({
                col: SharedMemoryManager.shm2s(shm, shape, dtype)
                for shm, (col, _, shape, dtype) in zip(shms, data_shm_info)})
            df.set_index(SharedMemoryManager._DF_INDEX_COL, drop=True, inplace=True)
            df.index.name = None
            return df, shms
        elif isinstance(data_shm_info, dict): # Info for Dict[str, DataFrame]
            reconstructed_dict = {}
            all_shms = []
            for ticker, single_df_shm_info in data_shm_info.items():
                shms_for_df = [SharedMemory(name=name, create=False, track=False) for _, name, _, _ in single_df_shm_info]
                df = pd.DataFrame({
                    col: SharedMemoryManager.shm2s(shm, shape, dtype)
                    for shm, (col, _, shape, dtype) in zip(shms_for_df, single_df_shm_info)})
                df.set_index(SharedMemoryManager._DF_INDEX_COL, drop=True, inplace=True)
                df.index.name = None
                reconstructed_dict[ticker] = df
                all_shms.extend(shms_for_df)
            return reconstructed_dict, all_shms
        else:
            raise TypeError("shm2df expects shm info for a DataFrame (tuple) or Dict[str, DataFrame] (dict)")
    



class PicklableAnalysisIndicators(AnalysisIndicators):
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


@pd.api.extensions.register_dataframe_accessor("ta")
@pd.api.extensions.register_series_accessor("ta")
class _TA:
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(self, obj: Union[Dict[str, pd.DataFrame], pd.DataFrame, pd.Series]):
        if isinstance(obj, dict): # Dict[str, pd.DataFrame] for multi-asset
            if not obj: # Empty dict
                self._obj = obj
                self._is_series = False
                self._is_dataframe = False # Or True, representing a collection of DataFrames
                self.__tickers = []
                self.__indicators = {}
                self.__indicator = None
                return

            self._obj = obj
            self._is_series = False
            self._is_dataframe = True # Represents a multi-asset collection of DataFrames
            self.__tickers = list(obj.keys())
            self.__indicators = {ticker: PicklableAnalysisIndicators(df) for ticker, df in obj.items()}
            self.__indicator = None
        elif isinstance(obj, pd.DataFrame): # Single DataFrame
            if obj.empty:
                # Handle empty DataFrame case if necessary, or let pandas_ta handle it
                self._obj = obj
                self._is_series = False
                self._is_dataframe = True
                self.__tickers = []
                self.__indicators = {}
                self.__indicator = PicklableAnalysisIndicators(obj) if not obj.empty else None
                return

            self._obj = obj
            self._is_series = False
            self._is_dataframe = True
            self.__tickers = [] # No tickers for single DataFrame context here
            self.__indicators = {}
            self.__indicator = PicklableAnalysisIndicators(obj)
        elif isinstance(obj, pd.Series):
            if obj.empty:
                self._obj = obj
                self._is_series = True
                self._is_dataframe = False
                self.__tickers = []
                self.__indicators = {}
                self.__indicator = None # pandas_ta might not work well with empty series for init
                return

            self._obj = obj
            self._is_series = True
            self._is_dataframe = False
            self.__tickers = []
            self.__indicators = {}
            
            series_name = getattr(obj, 'name', None)
            if isinstance(series_name, tuple) and len(series_name) == 2:
                col_name = series_name[1]
            elif isinstance(series_name, str):
                col_name = series_name
            else:
                col_name = 'Close'
            df_from_series = obj.to_frame(name=col_name)
            self.__indicator = PicklableAnalysisIndicators(df_from_series)
        else:
            raise TypeError("Input must be a dictionary of DataFrames, a pandas DataFrame, or a pandas Series")


    def __ta(self, method_name, *args, columns=None, **kwargs):
        if self.__indicators: # Multi-asset dictionary of DataFrames
            results_dict = {
                ticker: getattr(indicator, method_name)(*args, **kwargs)
                for ticker, indicator in self.__indicators.items()
            }
            if columns: # Rename columns if requested (applies to each DataFrame in dict)
                for ticker_df in results_dict.values():
                    if isinstance(ticker_df, pd.DataFrame):
                        ticker_df.columns = columns
                    # If result is Series, renaming is trickier / less direct
            return results_dict # Return dictionary of results
        elif self.__indicator: # Single DataFrame or Series
            return getattr(self.__indicator, method_name)(*args, **kwargs)
        else: # Empty input object
             if hasattr(self._obj, 'empty') and self._obj.empty:
                 # Attempt to call on an empty df/series via pandas_ta, might return empty result or error
                 # This path is tricky, pandas_ta behavior with empty inputs varies.
                 # For now, let it try, or return an empty structure consistent with expected output.
                 # This depends on what pandas_ta does.
                 # A simple approach: if input was empty, result is likely empty.
                 if isinstance(self._obj, dict): return {}
                 if isinstance(self._obj, pd.DataFrame): return pd.DataFrame()
                 if isinstance(self._obj, pd.Series): return pd.Series(dtype=float)


             raise RuntimeError("TA object not properly initialized or input was empty in a way that prevents TA calculation.")

    def __getattr__(self, method: str):
        if self.__indicator or self.__indicators or (hasattr(self._obj, 'empty') and self._obj.empty):
            return partial(self.__ta, method)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{method}' or is uninitialized")

    def apply(self, func, *args, **kwargs):
        """Apply a custom function to the underlying data."""
        if self.__indicators: # Multi-asset dictionary of DataFrames
            # self._obj is Dict[str, pd.DataFrame]
            return {ticker: func(df, *args, **kwargs) for ticker, df in self._obj.items()}
        elif self._is_dataframe or self._is_series: # Single DataFrame or Series
            return func(self._obj, *args, **kwargs)
        else: # Empty input
             if hasattr(self._obj, 'empty') and self._obj.empty:
                 if isinstance(self._obj, dict): return {}
                 if isinstance(self._obj, pd.DataFrame): return pd.DataFrame() # Or func(self._obj)
                 if isinstance(self._obj, pd.Series): return pd.Series(dtype=float) # Or func(self._obj)
             raise RuntimeError("TA object not properly initialized or input was empty.")

    def join(self, other, lsuffix='', rsuffix=''):
        """Join with another DataFrame or Series."""
        if self._is_series:
            return self._obj.join(other, lsuffix=lsuffix, rsuffix=rsuffix)
        elif self.__indicators: # Multi-asset dictionary of DataFrames
            if not isinstance(other, dict) or set(self.__tickers) != set(other.keys()):
                raise ValueError("For multi-asset join, 'other' must be a dictionary of DataFrames with matching tickers.")
            return {
                ticker: self._obj[ticker].join(other[ticker], lsuffix=lsuffix, rsuffix=rsuffix)
                for ticker in self.__tickers
            }
        elif self._is_dataframe: # Single DataFrame
            return self._obj.join(other, lsuffix=lsuffix, rsuffix=rsuffix)
        else: # Empty input
             if hasattr(self._obj, 'empty') and self._obj.empty:
                 # Behavior for join on empty df/series
                 if isinstance(self._obj, pd.DataFrame): return self._obj.join(other, lsuffix=lsuffix, rsuffix=rsuffix)
                 if isinstance(self._obj, pd.Series): return self._obj.join(other, lsuffix=lsuffix, rsuffix=rsuffix)
                 if isinstance(self._obj, dict): return {} # Or raise error
             raise RuntimeError("TA object not properly initialized or input was empty.")
