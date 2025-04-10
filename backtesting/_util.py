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
from typing import Dict, List, Optional, Sequence, Union, cast
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
    
    This implementation supports both single-asset and multi-asset data through
    a two-level column index structure, where the first level represents tickers
    and the second level represents OHLCV columns.
    """
    def __init__(self, df: pd.DataFrame):
        self.__df = df
        self.__len = len(df)  # Current length
        self.__pip: Optional[float] = None
        self.__cache: Dict[str, _Array] = {}
        self.__arrays: Dict[str, _Array] = {}
        self._tickers = list(self.__df.columns.levels[0])
        self._ta = _TA(self.__df)
        self._update()

    def __getitem__(self, item):
        return self._get_array(item)

    def __getattr__(self, item):
        try:
            return self._get_array(item)
        except KeyError:
            raise AttributeError(f"Column '{item}' not in data") from None

    def _set_length(self, length):
        self.__len = length
        self.__cache.clear()

    def _update(self):
        # Cache slices of the data as DataFrame/Series for faster access
        arrays = (
            {ticker_col: arr for ticker_col, arr in self.__df.items()}
            | {col: self.__df.xs(col, axis=1, level=1) for col in self.__df.columns.levels[1]}
            | {ticker: self.__df[ticker] for ticker in self.__df.columns.levels[0]}
            | {None: self.__df[self.the_ticker] if len(self._tickers) == 1 else self.__df}
            | {'__index': self.__df.index.copy()}
        )
        arrays = {key: df.iloc[:, 0] if isinstance(df, pd.DataFrame) and len(
            df.columns) == 1 else df for key, df in arrays.items()}
        # Keep another copy as Numpy array
        self.__arrays = {key: (df.to_numpy(), df) for key, df in arrays.items()}

    def __repr__(self):
        i = min(self.__len, len(self.__df)) - 1
        index = self.__arrays['__index'][0][i]
        items = ', '.join(f'{k}={v}' for k, v in self.__df.iloc[i].items())
        return f'<Data i={i} ({index}) {items}>'

    def __len__(self):
        return self.__len

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns the underlying DataFrame, either for a single ticker or the full multi-asset DataFrame.
        
        This property also supports adding new columns to the DataFrame which can then be accessed
        as attributes of the _Data object.
        """
        # Pass the full original DataFrame reference
        return _DataFrameAccessor(self, self.__df)

    @property
    def pip(self) -> float:
        if self.__pip is None:
            self.__pip = float(10**-np.median([len(s.partition('.')[-1])
                                               for s in self.__arrays['Close'][0].ravel().astype(str)]))
        return self.__pip

    def _get_array(self, key) -> _Array:
        arr = self.__cache.get(key)
        if arr is None:
            try:
                array, df = self.__arrays[key]
                arr = self.__cache[key] = _Array(df.values[:self.__len], name=key, index=self.index)
            except KeyError:
                # Handle the case where key is not in __arrays
                # This could be a dynamically added column
                if key in self.__df.columns.levels[1]:
                    # For multi-level DataFrame, get all values for this column across tickers
                    df = self.__df.xs(key, axis=1, level=1)
                    array = df.to_numpy()
                    self.__arrays[key] = (array, df)
                    arr = self.__cache[key] = _Array(df.values[:self.__len], name=key, index=self.index)
                else:
                    raise KeyError(f"Column '{key}' not in data")
        return arr
    
    @property
    def Open(self) -> _Array:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.Open` is ambiguous in a multi-asset context. Use `self.data[ticker, 'Open']`.")
        return self._get_array((self.the_ticker, 'Open'))

    @property
    def High(self) -> _Array:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.High` is ambiguous in a multi-asset context. Use `self.data[ticker, 'High']`.")
        return self._get_array((self.the_ticker, 'High'))

    @property
    def Low(self) -> _Array:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.Low` is ambiguous in a multi-asset context. Use `self.data[ticker, 'Low']`.")
        return self._get_array((self.the_ticker, 'Low'))

    @property
    def Close(self) -> _Array:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.Close` is ambiguous in a multi-asset context. Use `self.data[ticker, 'Close']`.")
        return self._get_array((self.the_ticker, 'Close'))

    @property
    def Volume(self) -> _Array:
        if len(self._tickers) > 1:
            raise ValueError("Accessing `self.data.Volume` is ambiguous in a multi-asset context. Use `self.data[ticker, 'Volume']`.")
        return self._get_array((self.the_ticker, 'Volume'))

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.__df.index[:self.__len]

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


class _IndexerWrapper:
    """Helper class to wrap pandas indexers (iloc, loc) and simplify columns of the result if needed."""
    def __init__(self, accessor_obj, original_indexer):
        self._accessor = accessor_obj
        self._original_indexer = original_indexer

    def __getitem__(self, key):
        # Perform the original indexing operation
        result_slice = self._original_indexer[key]
        # Simplify columns if necessary before returning
        return self._accessor._simplify_df_columns(result_slice)


class _DataFrameAccessor:
    """
    A wrapper around DataFrame that updates the _Data object's cache when new columns are added.
    Provides sliced view for __getitem__ and delegates other methods.
    """
    def __init__(self, data_obj, original_df_ref):
        self.__data_obj = data_obj
        # Store a reference to the original DataFrame, not a slice
        self.__original_df = original_df_ref

    def _simplify_df_columns(self, df_slice):
        """Drops the top level of columns if it's a DataFrame with MultiIndex and only one ticker exists."""
        if isinstance(df_slice, pd.DataFrame) and \
           isinstance(df_slice.columns, pd.MultiIndex) and \
           len(self.__data_obj.tickers) == 1:
            # Drop the ticker level (level 0)
            try:
                # Only drop if the resulting columns are unique, otherwise keep MultiIndex
                simplified_cols = df_slice.columns.droplevel(0)
                if simplified_cols.is_unique:
                    return df_slice.droplevel(0, axis=1)
                else:
                    # If dropping level leads to duplicate columns (e.g., custom columns with same name as OHLCV),
                    # keep the MultiIndex to avoid ambiguity.
                    return df_slice
            except ValueError: # Cannot drop level on empty DataFrame or Series
                return df_slice
        return df_slice

    def __getitem__(self, key):
        # Return the appropriate slice from the full DataFrame
        current_len = self.__data_obj._Data__len
        full_index = self.__data_obj._Data__df.index # Use the index from the main _Data object
        try:
            # Attempt to access the column(s) from the full DataFrame and then slice
            return self.__original_df.loc[full_index[:current_len], key]
        except KeyError as e:
            # Handle MultiIndex case for single key access when only one ticker exists
            if isinstance(self.__original_df.columns, pd.MultiIndex) and \
               isinstance(key, str) and \
               len(self.__data_obj.tickers) == 1:
                ticker = self.__data_obj.the_ticker
                if (ticker, key) in self.__original_df.columns:
                    # Access the specific column tuple and slice
                    key_to_use = (ticker, key)
                # else: key might be a new column name, handled below? Or let KeyError happen?
                # Let's assume if it's a string and single ticker, we try the tuple format.

            # Attempt to access the column(s) from the full DataFrame and then slice
            result_slice = self.__original_df.loc[full_index[:current_len], key_to_use]
            # This simplification should happen regardless of how the slice was obtained
            return self._simplify_df_columns(result_slice) 

        # This except block might be unreachable if the try block handles all cases or raises appropriately.
        # Let's simplify the logic.

    def __getitem__(self, key):
        current_len = self.__data_obj._Data__len
        # Use the index corresponding to the current view length
        current_view_index = self.__data_obj.index 

        try:
            if isinstance(key, slice):
                # Apply slice to the current view's index to get target labels
                target_index = current_view_index[key]
                result_slice = self.__original_df.loc[target_index, :]
            elif isinstance(key, str):
                 # Handle string key (potential column access)
                 if isinstance(self.__original_df.columns, pd.MultiIndex) and \
                    len(self.__data_obj.tickers) == 1:
                     # Try accessing as (ticker, key) for single-asset MultiIndex
                     ticker = self.__data_obj.the_ticker
                     try:
                         # Try accessing the specific tuple key first
                         result_slice = self.__original_df.loc[current_view_index, (ticker, key)]
                     except KeyError:
                         # If (ticker, key) fails, try accessing the key directly.
                         result_slice = self.__original_df.loc[current_view_index, key] # This might raise KeyError again
                 else:
                     # Standard single-level column access or multi-asset access (requires tuple key)
                     result_slice = self.__original_df.loc[current_view_index, key]
            else:
                 # Handle other key types (e.g., list of columns, boolean array for rows)
                 # Assume key applies to rows if it's not a column label type pandas recognizes for columns
                 result_slice = self.__original_df.loc[current_view_index, key]

            # Simplify columns if it's a single-asset context and result is a DataFrame
            return self._simplify_df_columns(result_slice)

        except KeyError as e:
             # Reraise if the key truly doesn't exist after trying various access methods
             raise KeyError(f"Key '{key}' not found in DataFrame index or columns") from e
        # except Exception as e: # Catch other potential indexing errors? Be careful not to mask useful errors.
        #     raise RuntimeError(f"Error during DataFrame access with key '{key}'") from e


    def __setitem__(self, key, value):
        # Modify the original DataFrame directly
        full_index = self.__data_obj._Data__df.index # Use the index from the main _Data object

        # Ensure value has the full length expected for the original DataFrame
        if len(value) != len(self.__original_df):
            if np.isscalar(value):
                value = np.repeat(value, len(self.__original_df))
            else:
                # Attempt to align if value is a Series/array with matching index subset
                try:
                    # Use the current slice's index for alignment first
                    aligned_value = pd.Series(value, index=self.__data_obj.index)
                    # Then reindex to the full DataFrame's index
                    aligned_value = aligned_value.reindex(full_index)
                    value = aligned_value.values
                except Exception as e:
                    raise ValueError(f"Length mismatch or alignment error when setting column '{key}'. Expected {len(self.__original_df)}, got {len(value)}.") from e

        if isinstance(self.__original_df.columns, pd.MultiIndex):
            # Handle single string key for single-ticker case
            if len(self.__data_obj.tickers) == 1 and isinstance(key, str):
                ticker = self.__data_obj.the_ticker
                # Use .loc with the full index for assignment
                self.__original_df.loc[:, (ticker, key)] = value
            elif isinstance(key, tuple) and len(key) == 2: # Assigning (ticker, column)
                self.__original_df.loc[:, key] = value
            else:
                raise ValueError(f"Cannot assign key '{key}' to multi-asset DataFrame via .df accessor. Use tuple key (ticker, column) or ensure single asset.")
        else: # Single-level DataFrame (should not occur with current Backtest init logic)
             self.__original_df[key] = value

        # Update the _Data object's arrays and cache
        self.__data_obj._update()

    # Delegate other DataFrame methods/attributes
    def __getattr__(self, name):
        # Handle direct attribute access for standard OHLCV columns in single-asset context
        if name in ['Open', 'High', 'Low', 'Close', 'Volume'] and len(self.__data_obj.tickers) == 1:
            ticker = self.__data_obj.the_ticker
            key_to_use = (ticker, name)
            if key_to_use in self.__original_df.columns:
                # Return the column as a Series, sliced to the current length
                current_len = self.__data_obj._Data__len
                full_index = self.__data_obj._Data__df.index
                # .loc access with tuple key on MultiIndex DF returns a Series
                series_slice = self.__original_df.loc[full_index[:current_len], key_to_use]
                return series_slice
            # If (ticker, name) is not found for some reason, fall through to standard delegation

        # Always delegate attribute access directly to the original DataFrame for other attributes.
        # Slicing for indexers (like .iloc) happens upon use.
        try:
            original_attr = getattr(self.__original_df, name)
            # Intercept indexers like iloc, loc to simplify their results
            if name in ['iloc', 'loc']:
                # Return a custom indexer wrapper that simplifies results
                return _IndexerWrapper(self, original_attr)
            else:
                # Delegate other attributes directly
                return original_attr
        except AttributeError:
            # Raise AttributeError consistent with normal object behavior
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None

    def __repr__(self):
        # Represent the current slice of the full DataFrame
        current_len = self.__data_obj._Data__len
        full_index = self.__data_obj._Data__df.index
        # Use .loc with the full index for robust slicing
        return repr(self.__original_df.loc[full_index[:current_len]])


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
        """
        Converts a DataFrame to shared memory. Handles MultiIndex columns.
        Returns a tuple containing:
            - A tuple of tuples, where each inner tuple represents a column (or index)
              and contains: (column_name, shm_name, shape, dtype).  column_name
              will be a tuple for MultiIndex columns.
            - A tuple containing the names of the columns.  For MultiIndex, this will
              be a tuple of tuples.
        """
        column_names = tuple(df.columns)
        data = tuple(
            (column, *self.arr2shm(values))
            for column, values in chain([(self._DF_INDEX_COL, df.index)], df.items())
        )
        return data, column_names

    @staticmethod
    def shm2s(shm, shape, dtype) -> pd.Series:
        arr = np.ndarray(shape, dtype=dtype.base, buffer=shm.buf)
        arr.setflags(write=False)
        return pd.Series(arr, dtype=dtype)

    _DF_INDEX_COL = '__bt_index'

    @staticmethod
    def shm2df(data_shm, column_names):
        """
        Reconstructs a DataFrame from shared memory, handling MultiIndex columns.
        Args:
            data_shm: The tuple returned by df2shm.
            column_names: The tuple of column names returned by df2shm.
        Returns:
            A tuple containing the reconstructed DataFrame and a list of SharedMemory
            objects that need to be unlinked.
        """
        shm_list = [SharedMemory(name=name, create=False, track=False) for _, name, _, _ in data_shm]
        
        # Reconstruct the DataFrame data.  Skip the index column.
        data = {}
        shms = iter(shm_list)
        index_shm = next(shms) # first shm is the index
        index_col, _, index_shape, index_dtype = data_shm[0]

        for col in column_names:
            shm = next(shms)
            col_data = next(x for x in data_shm if x[0] == col)
            _, _, shape, dtype = col_data
            data[col] = SharedMemoryManager.shm2s(shm, shape, dtype)

        df = pd.DataFrame(data)

        #Reconstruct the index
        df.index = SharedMemoryManager.shm2s(index_shm, index_shape, index_dtype)
        df.index.name = None
        
        # Set the index name if it was set
        if index_col != SharedMemoryManager._DF_INDEX_COL:
            df.index.name = index_col

        return df, shm_list



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

    def __init__(self, obj: Union[pd.DataFrame, pd.Series]):
        if obj.empty:
            return
        self._obj = obj
        self._is_series = isinstance(obj, pd.Series)
        self._is_dataframe = isinstance(obj, pd.DataFrame)
        self.__tickers = []
        self.__indicators = {}
        self.__indicator = None # For single DataFrame or Series

        if self._is_dataframe:
            if self._obj.columns.nlevels == 2:
                self.__tickers = list(self._obj.columns.levels[0])
                self.__indicators = {ticker: PicklableAnalysisIndicators(obj[ticker]) for ticker in self.__tickers}
            elif self._obj.columns.nlevels == 1:
                self.__indicator = PicklableAnalysisIndicators(obj)
            else:
                raise AttributeError(
                    f'DataFrame columns can have at most 2 levels, got {self._obj.columns.nlevels}')
        elif self._is_series:
            # pandas_ta seems to expect a DataFrame even when initialized with a Series.
            # Convert the Series to a DataFrame, ensuring a simple string column name
            # to avoid AttributeError: Can only use .str accessor with Index, not MultiIndex
            # which occurs if df.columns is a MultiIndex.
            series_name = getattr(obj, 'name', None)
            if isinstance(series_name, tuple) and len(series_name) == 2:
                # Use the second element of the tuple (e.g., 'Close' from ('GOOG', 'Close'))
                col_name = series_name[1]
            elif isinstance(series_name, str):
                col_name = series_name
            else:
                # Default column name if the original name is complex or missing
                col_name = 'Close' # pandas_ta often defaults to looking for 'close'

            df_from_series = obj.to_frame(name=col_name)
            self.__indicator = PicklableAnalysisIndicators(df_from_series)
        else:
            raise TypeError("Input must be a pandas DataFrame or Series")


    def __ta(self, method, *args, columns=None, **kwargs):
        if self.__tickers: # Multi-asset DataFrame
            dir_ = {ticker: getattr(indicator, method)(*args, **kwargs)
                    for ticker, indicator in self.__indicators.items()}
            if columns: # Rename columns if requested
                for _, df in dir_.items():
                    df.columns = columns
            # Return concatenated DataFrame or single DataFrame if only one ticker
            return pd.concat(dir_, axis=1) if len(dir_) > 1 else dir_[self.__tickers[0]]
        elif self.__indicator: # Single-asset DataFrame or Series
            return getattr(self.__indicator, method)(*args, **kwargs)
        else: # Should not happen if __init__ worked
             raise RuntimeError("TA object not properly initialized.")

    def __getattr__(self, method: str):
        # Allow calling pandas_ta methods directly, e.g., df.ta.sma()
        if self.__indicator or self.__indicators:
            return partial(self.__ta, method)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{method}' or is uninitialized")

    def apply(self, func, *args, **kwargs):
        """Apply a custom function to the underlying data."""
        if self.__tickers: # Multi-asset DataFrame
            dir_ = {ticker: func(self._obj[ticker], *args, **kwargs) for ticker in self.__tickers}
            return pd.concat(dir_, axis=1)
        elif self._is_dataframe or self._is_series: # Single DataFrame or Series
            return func(self._obj, *args, **kwargs)
        else:
             raise RuntimeError("TA object not properly initialized.")

    def join(self, other, lsuffix='', rsuffix=''):
        """Join with another DataFrame or Series."""
        if self._is_series:
            # Series.join needs 'other' to be Series or list of Series
            return self._obj.join(other, lsuffix=lsuffix, rsuffix=rsuffix)
        elif self.__tickers: # Multi-asset DataFrame
            if not isinstance(other, pd.DataFrame) or other.columns.nlevels != 2 or \
               set(self.__tickers) != set(other.columns.levels[0]):
                raise ValueError("For multi-asset join, 'other' must be a DataFrame with matching tickers.")
            dir_ = {ticker: self._obj[ticker].join(other[ticker], lsuffix=lsuffix, rsuffix=rsuffix)
                    for ticker in self.__tickers}
            return pd.concat(dir_, axis=1)
        elif self._is_dataframe: # Single DataFrame
            return self._obj.join(other, lsuffix=lsuffix, rsuffix=rsuffix)
        else:
             raise RuntimeError("TA object not properly initialized.")
