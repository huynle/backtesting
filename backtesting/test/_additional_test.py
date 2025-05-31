import sys
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
import pytest

import pandas as pd

from backtesting import Backtest, Strategy, Allocation
from backtesting._util import _Array
from backtesting.test import GOOG, SPY
from pandas.testing import assert_series_equal

SHORT_DATA = GOOG.iloc[:20]  # Short data for fast tests with no indicator lag

@contextmanager
def _tempfile():
    with NamedTemporaryFile(suffix='.html') as f:
        if sys.platform.startswith('win'):
            f.close()
        yield f.name

class TestAllocation:
    @pytest.fixture
    def a(self):
        return Allocation(['A', 'B', 'C', 'D'])

    def test_call_assume_first(self, a):
        assert_series_equal(a.previous_weights, pd.Series([0.0, 0.0, 0.0, 0.0], index=['A', 'B', 'C', 'D']))
        with pytest.raises(RuntimeError):
            a.weights
        with pytest.raises(RuntimeError):
            a.weights = pd.Series([0.3, 0.3, 0.4, 0.0], index=['A', 'B', 'C', 'D'])
        with pytest.raises(RuntimeError):
            a.bucket
        with pytest.raises(RuntimeError):
            a.unallocated
        with pytest.raises(RuntimeError):
            a.normalize()

    @pytest.fixture
    def b(self) -> Allocation:
        alloc = Allocation(['A', 'B', 'C', 'D'])
        alloc.assume_zero()
        return alloc

    def test_assume_zero(self, b):
        assert_series_equal(b.weights, pd.Series([0.0, 0.0, 0.0, 0.0], index=['A', 'B', 'C', 'D']))
        assert b.unallocated == 1.0
        assert len(b.bucket) == 0

    def test_individual_weight_assignment(self, b):
        b.weights['A'] = 0.5
        b.weights['B'] = 0.3
        assert_series_equal(b.weights, pd.Series([0.5, 0.3, 0.0, 0.0], index=['A', 'B', 'C', 'D']))
        assert pytest.approx(b.unallocated) == 0.2
        with pytest.raises(AssertionError):
            b.weights['C'] = 0.3
            b.weights
        with pytest.raises(AssertionError):
            b.weights['X'] = 0.1
            b.weights

    def test_entire_weight_assignment(self, b):
        b.weights = pd.Series([0.4, 0.6], index=['A', 'B'])
        assert_series_equal(b.weights, pd.Series([0.4, 0.6, 0.0, 0.0], index=['A', 'B', 'C', 'D']))
        assert b.unallocated == 0.0

    def test_bucket_creation(self, b):
        bucket = b.bucket['test']
        assert isinstance(bucket, Allocation.Bucket)

    def test_append_assets(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        assert bucket.tickers == ['A', 'B']

    def test_append_boolean_series(self, b):
        bucket = b.bucket['test']
        series = pd.Series([True, False, True, False], index=['A', 'B', 'C', 'D'])
        bucket.append(series)
        assert bucket.tickers == ['A', 'C']

    def test_append_non_boolean_series(self, b):
        bucket = b.bucket['test']
        series = pd.Series([0, 0, 0.5, 1], index=['A', 'B', 'C', 'D'])
        bucket.append(series)
        assert bucket.tickers == ['A', 'B', 'C', 'D']

    def test_append_mixed_conditions(self, b):
        bucket = b.bucket['test']
        series = pd.Series([0, 0.3, 0.5, 1], index=['A', 'B', 'C', 'D'])
        bucket.append(series, series > 0, series < 1, ['A', 'C'])
        assert bucket.tickers == ['C']

    def test_multiple_append(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.append(['C', 'D'])
        assert bucket.tickers == ['A', 'B', 'C', 'D']

    def test_multiple_append_with_duplicates_and_mixed_order(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.append(['C', 'D', 'A'])
        assert bucket.tickers == ['A', 'B', 'C', 'D']

    def test_remove_assets(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        bucket.remove(['A', 'C'])
        assert bucket.tickers == ['B']

    def test_trim_bucket(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C', 'D'])
        bucket.trim(2)
        assert bucket.tickers == ['A', 'B']

    def test_weight_empty_bucket(self, b):
        b.weights = pd.Series([0.25] * 4, index=['A', 'B', 'C', 'D'])
        bucket = b.bucket['test']
        bucket.weight_explicitly(0.5).apply('overwrite')
        assert bucket.weights.empty
        assert_series_equal(b.weights, pd.Series([0.25] * 4, index=['A', 'B', 'C', 'D']))
        bucket.weight_equally()
        assert bucket.weights.empty
        assert_series_equal(b.weights, pd.Series([0.25] * 4, index=['A', 'B', 'C', 'D']))
        bucket.weight_proportionally([])
        assert bucket.weights.empty
        assert_series_equal(b.weights, pd.Series([0.25] * 4, index=['A', 'B', 'C', 'D']))

    def test_weight_explicitly_single_value(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.weight_explicitly(0.5)
        assert_series_equal(bucket.weights, pd.Series([0.5, 0.5], index=['A', 'B']))
        for value in [-0.01, 1.01]:
            with pytest.raises(AssertionError):
                bucket.weight_explicitly(value)

    def test_weight_explicitly_list(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.weight_explicitly([0.2, 0.8])
        assert_series_equal(bucket.weights, pd.Series([0.2, 0.8], index=['A', 'B']))
        for value in [-0.01, 1.01, 0.9]:
            with pytest.raises(AssertionError):
                bucket.weight_explicitly([0.2, value])

    def test_weight_explicitly_short_list(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.weight_explicitly([0.2])
        assert_series_equal(bucket.weights, pd.Series([0.2, 0.], index=['A', 'B']))

    def test_weight_explicitly_long_list(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        bucket.weight_explicitly([0.2, 0.3, 0.4])
        assert_series_equal(bucket.weights, pd.Series([0.2, 0.3], index=['A', 'B']))

    def test_weight_explicitly_series(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        series = pd.Series([0.2, 0.8], index=['A', 'B'])
        bucket.weight_explicitly(series)
        assert_series_equal(bucket.weights, series)
        for value in [-0.01, 1.01]:
            with pytest.raises(AssertionError):
                bucket.weight_explicitly(pd.Series([0.2, value], index=['A', 'B']))

    def test_weight_explicitly_short_series(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        series = pd.Series([0.2], index=['A'])
        bucket.weight_explicitly(series)
        assert_series_equal(bucket.weights, pd.Series([0.2, 0.], index=['A', 'B']))

    def test_weight_explicitly_long_series(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B'])
        series = pd.Series([0.2, 0.3, 0.4], index=['A', 'B', 'C'])
        bucket.weight_explicitly(series)
        assert_series_equal(bucket.weights, pd.Series([0.2, 0.3], index=['A', 'B']))

    def test_weight_equally(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        bucket.weight_equally()
        assert_series_equal(bucket.weights, pd.Series([1/3] * 3, index=['A', 'B', 'C']))
        for value in [-0.01, 1.01]:
            with pytest.raises(AssertionError):
                bucket.weight_equally(value)

    def test_weight_equally_sum(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        bucket.weight_equally(sum_=0.3)
        assert_series_equal(bucket.weights, pd.Series([0.3/3] * 3, index=['A', 'B', 'C']))

    def test_weight_proportionally(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        bucket.weight_proportionally([1, 2, 3])
        assert_series_equal(bucket.weights, pd.Series([1/6, 1/3, 1/2], index=['A', 'B', 'C']))
        with pytest.raises(AssertionError):
            bucket.weight_proportionally([1, 2, -9])
        with pytest.raises(AssertionError):
            bucket.weight_proportionally([1, 2])

    def test_weight_proportionally_sum(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        bucket.weight_proportionally([1, 2, 3], sum_=0.6)
        assert_series_equal(bucket.weights, pd.Series([0.6/6, 0.6/3, 0.6/2], index=['A', 'B', 'C']))

    def test_apply_before_weight_assignment(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C'])
        with pytest.raises(RuntimeError):
            bucket.apply('accumulate')

    def test_apply_patch_method(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C', 'D'])
        bucket.weight_equally().apply()
        assert_series_equal(b.weights, pd.Series([1/4] * 4, index=['A', 'B', 'C', 'D']))
        bucket2 = b.bucket['test2']
        bucket2.append(['A', 'B'])
        bucket2.weight_explicitly(0.2).apply('update')
        assert_series_equal(b.weights, pd.Series([0.2, 0.2, 0.25, 0.25], index=['A', 'B', 'C', 'D']))

    def test_apply_replace_method(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C', 'D'])
        bucket.weight_equally().apply('overwrite')
        assert_series_equal(b.weights, pd.Series([0.25] * 4, index=['A', 'B', 'C', 'D']))
        bucket2 = b.bucket['test2']
        bucket2.append(['A', 'B'])
        bucket2.weight_explicitly(0.2).apply('overwrite')
        assert_series_equal(b.weights, pd.Series([0.2, 0.2, 0., 0.], index=['A', 'B', 'C', 'D']))

    def test_apply_sum_method(self, b):
        bucket = b.bucket['test']
        bucket.append(['A', 'B', 'C', 'D'])
        bucket.weight_equally(0.8).apply('accumulate')
        assert_series_equal(b.weights, pd.Series([0.2] * 4, index=['A', 'B', 'C', 'D']))
        bucket2 = b.bucket['test2']
        bucket2.append(['A', 'B'])
        bucket2.weight_equally().apply('accumulate')
        assert_series_equal(b.weights, pd.Series([0.3, 0.3, 0.2, 0.2], index=['A', 'B', 'C', 'D']))
        bucket3 = b.bucket['test3']
        bucket3.append(b.weights)
        bucket3.weight_explicitly(b.weights).apply('accumulate')
        with pytest.raises(AssertionError):
            assert_series_equal(b.weights, pd.Series([0.6, 0.6, 0.4, 0.4], index=['A', 'B', 'C', 'D']))
        b.normalize()
        assert_series_equal(b.weights, pd.Series([0.3, 0.3, 0.2, 0.2], index=['A', 'B', 'C', 'D']))

    @pytest.fixture
    def c(self) -> Allocation:
        alloc = Allocation(['A', 'B', 'C', 'D'])
        alloc.assume_zero()
        bucket = alloc.bucket['test']
        bucket.append(['A', 'B', 'C', 'D'])
        bucket.weight_equally(0.5).apply('overwrite')
        alloc._next()
        alloc.assume_previous()
        return alloc

    def test_assume_previous(self, c):
        assert_series_equal(c.previous_weights, c.weights)
        assert len(c.bucket) == 0

    def test_next(self, c):
        c._next()
        assert_series_equal(c.previous_weights, pd.Series([0.5/4] * 4, index=c.tickers))
        with pytest.raises(RuntimeError):
            c.weights

    def test_normalize(self, c):
        c.normalize()
        assert_series_equal(c.weights, pd.Series([1/4] * 4, index=c.tickers))
        assert c.unallocated == 0.0


class RotateBuying(Strategy):
    def init(self):
        pass

    def next(self):
        self.alloc.assume_zero()
        index = len(self.data) % len(self.alloc.tickers)
        self.alloc.weights.iloc[index] = 1
        self.rebalance(cash_reserve=0.5)


class TestBacktest:
    @pytest.fixture
    def data(self):
        df = pd.DataFrame({
            'Open': [1, 2, 3, 4, 5],
            'High': [2, 3, 4, 5, 6],
            'Low': [3, 4, 5, 6, 7],
            'Close': [4, 5, 6, 7, 8]
        }, index=pd.date_range('2020-01-01', periods=5))
        df = pd.concat([df]*3, axis=1, keys=['A', 'B', 'C'])
        return df

    def test_rotate_buying(self, data):
        bt = Backtest(data, RotateBuying, finalize_trades=False)
        result = bt.run()
        assert result['# Trades'] == 2

        bt = Backtest(data, RotateBuying, finalize_trades=True)
        result = bt.run()
        assert result['# Trades'] == 4


    def test_talib_df(self):
        """integrated TA lib in a dataframe
        """
        class S(Strategy):
            def init(self):
                self.I(self.data.ta.macd)

            def next(self):
                pass

        bt = Backtest(GOOG, S)
        bt.run()
        with _tempfile() as f:
            bt.plot(filename=f,
                    plot_drawdown=False, plot_equity=False, plot_pl=False, plot_volume=False,
                    open_browser=False)

    def test_talib_series(self):
        """integrated TA lib in a dataframe
        """
        class S(Strategy):
            def init(self):
                self.I(self.data.Close.s.ta.ema(10))

            def next(self):
                pass

        bt = Backtest(GOOG, S)
        bt.run()
        with _tempfile() as f:
            bt.plot(filename=f,
                    plot_drawdown=False, plot_equity=False, plot_pl=False, plot_volume=False,
                    open_browser=False)


# Create a multi-asset dataframe from the imported GOOG and SPY data
MULTI_ASSET_DATA = {
    "GOOG": GOOG.copy(),
    "SPY": SPY.copy()
}


def calculate_ema(data, period, smoothing=2):
    """
    Calculate Exponential Moving Average
    
    Parameters:
    data (array-like): Price data series
    period (int): EMA period (e.g., 10, 20, 50)
    smoothing (int): Smoothing factor, typically 2 for standard EMA
    
    Returns:
    array-like: EMA values
    """
    ema = [0] * len(data)
    # Start with SMA for the initial EMA value
    ema[period-1] = sum(data[:period]) / period
    
    # Calculate multiplier
    multiplier = smoothing / (period + 1)
    
    # Calculate EMA for remaining data points
    for i in range(period, len(data)):
        ema[i] = (data.iloc[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


class TestDataConditioning:


    @pytest.fixture
    def data(self):
        df = pd.DataFrame({
            'Open': [1, 2, 3, 4, 5],
            'High': [2, 3, 4, 5, 6],
            'Low': [3, 4, 5, 6, 7],
            'Close': [4, 5, 6, 7, 8]
        }, index=pd.date_range('2020-01-01', periods=5))
        df = pd.concat([df]*3, axis=1, keys=['A', 'B', 'C'])
        return df

    def test_fill_volume(self, data):
        """ Test that multiindex columns can have volumes be properly filled out
        """
        class Simple(Strategy):
            def init(self):
                pass
            def next(self):
                pass

        bt = Backtest(data, Simple)
        assert bt

    def test_multi_asset_data_access(self):
        class Simple(Strategy):
            def init(self):
                assert isinstance(self.data['GOOG'][-5:], pd.DataFrame)
                assert isinstance(self.data['GOOG'], pd.DataFrame)
                with pytest.raises(ValueError):
                    self.data.Open
                
                with pytest.raises(ValueError):
                    self.data.High
                
                with pytest.raises(ValueError):
                    self.data.Low
                
                with pytest.raises(ValueError):
                    self.data.Close
                
                with pytest.raises(ValueError):
                    self.data.Volume

            def next(self):
                pass

        bt = Backtest(MULTI_ASSET_DATA, Simple)
        bt.run()

    def test_single_asset_data_access(self):
        class Simple(Strategy):
            def init(self):
                assert isinstance(self.data[-5:], pd.DataFrame)
                assert isinstance(self.data.df, pd.DataFrame)
                assert isinstance(self.data.df['High'], pd.Series)
                assert isinstance(self.data.High, _Array)

            def next(self):
                pass
        bt = Backtest(GOOG, Simple)
        bt.run()




    

class TestBacktestMulti(object):

    def test_multi_asset_run(self):
        class MultiAssetStrategy(Strategy):
            """
            A strategy that trades GOOG and SPY based on simple moving average crossovers.

            This strategy buys GOOG when its closing price is above its 10-day SMA and
            sells SPY when its closing price is below its 10-day SMA.
            """

            def init(self):
                # Calculate 10-day SMA for GOOG
                self.ema_goog = self.I(calculate_ema, self.data["GOOG", "Close"].s, 10)

                # Calculate 10-day SMA for SPY
                self.ema_spy = self.I(calculate_ema, self.data["SPY", "Close"].s, 10)

                # Calculate EMAs for SPY to determine market health
                self.ema10_spy = self.I(
                    calculate_ema, self.data["SPY"].Close, 10, name="SPY_EMA10"
                )
                self.ema20_spy = self.I(
                    calculate_ema, self.data["SPY"].Close, 20, name="SPY_EMA20"
                )
                self.ema50_spy = self.I(
                    calculate_ema, self.data["SPY", "Close"].s, 50, name="SPY_EMA50"
                )

                # Calculate EMAs for GOOG
                self.ema20_goog = self.I(
                    calculate_ema, self.data["GOOG", "Close"].s, 20, name="GOOG_EMA20"
                )

            def next(self):
                # Check if market is healthy: SPY 10 EMA > 20 EMA and price > 50 EMA
                market_healthy = (self.ema10_spy[-1] > self.ema20_spy[-1]) and (
                    self.data["SPY", "Close"][-1] > self.ema50_spy[-1]
                )

                # If market is healthy and GOOG is above its 10-day SMA, buy GOOG
                if (
                    market_healthy
                    and self.data["GOOG"].Close[-1] > self.ema_goog[-1]
                ):
                    if not self.get_position("GOOG"):
                        # self.buy(ticker="GOOG", size=0.1)
                        self.buy(ticker="GOOG")

                # If GOOG falls below its 20 EMA, liquidate 50% of position
                if (
                    self.get_position("GOOG")
                    and self.data["GOOG", "Close"][-1] < self.ema20_goog[-1]
                ):
                    current_size = self.get_position("GOOG").size
                    # self.position("GOOG").close(portion=0.5)
                    self.get_position("GOOG").close()

        bt = Backtest(MULTI_ASSET_DATA, MultiAssetStrategy, cash=1000000)
        stats = bt.run()
        bt.plot()
        assert stats["# Trades"] == 82
        assert stats["_positions"] == {"GOOG": 7127, "SPY": 0, "Cash": 30}

    def test_multi_asset_rebalance(self):
        class RebalanceStrategy(Strategy):
            def init(self):
                pass

            def next(self):
                if len(self.data) % 10 == 0:  # Rebalance every 10 days
                    self.alloc.assume_zero()
                    self.alloc.weights["GOOG"] = 1.0
                    self.alloc.weights["SPY"] = 0
                    # Set cash_reserve to 0 for closer target allocation in test
                    self.rebalance(cash_reserve=0)

        bt = Backtest(MULTI_ASSET_DATA, RebalanceStrategy, cash=1_000_000)
        stats = bt.run()
        assert stats["# Trades"] ==  0
        # bt.plot()
        # # Check if final equity distribution is roughly 60/40
        # final_equity = stats._equity_curve.iloc[-1]
        # goog_value = final_equity["GOOG"]
        # SPY_value = final_equity["SPY"]
        # total_value = goog_value + SPY_value
        # assert goog_value / total_value == pytest.approx(0.6, abs=0.1)
        
    def test_multistrategy(self):
        class MultiStrategy(Strategy):

            lookback = 10

            def init(self):
                # Define ROC indicator for each asset ticker separately
                self.roc = {}
                for ticker in self.data.tickers:
                    roc_series = self.data[ticker, "Close"].s.ta.roc(self.lookback)
                    # Store each indicator, possibly naming it per ticker for clarity in plots
                    self.roc[ticker] = self.I(lambda s=roc_series: s, name=f'ROC_{ticker}') # Use lambda to pass series

            def next(self):
                self.alloc.assume_zero()                                #2
                # Build a Series of current ROC values indexed by ticker
                current_roc = pd.Series({ticker: ind[-1] for ticker, ind in self.roc.items()}) #3
                (self.alloc.bucket['equity']                            #4
                    .append(current_roc.sort_values(ascending=False), current_roc > 0)  #5 Use current_roc
                    .trim(3)                                            #6
                    .weight_explicitly(1/3)                             #7
                    .apply())                                           #8
                self.rebalance(cash_reserve=0.01)                       #9

        bt = Backtest(MULTI_ASSET_DATA, MultiStrategy, cash=1_000_000)
        bt.run()
        # bt.plot()
    

