# run this using `python2 strategies/sma-cross.py` from the main directory
from backtesting import Backtest, Strategy
from backtesting.indicators import SMA
from backtesting.provider.ibkr import InteractiveBrokers  # Importing the InteractiveBrokers class
from backtesting.lib import crossover
import talib
import pandas as pd


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        # self.ma1 = self.I(SMA, price, 10)
        # self.ma2 = self.I(SMA, price, 20)
        self.ma1 = self.I(talib.EMA, price, 10)
        self.ma2 = self.I(talib.EMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


def test_sma_cross_with_ibkr_data():
    # Initialize Interactive Brokers connection
    ib = InteractiveBrokers()

    # Fetch SPY data from Interactive Brokers
    data = ib.get_data(symbol="SPY", bar_size="1 day", duration="1 Y")
    # Disconnect from Interactive Brokers
    ib.disconnect()

    # Ensure data was successfully retrieved before proceeding
    assert data is not None, "Failed to retrieve SPY data from Interactive Brokers."

    if data is not None:
        bt = Backtest(data, SmaCross, commission=0.002, exclusive_orders=True)
        stats = bt.run()

        # Assert that the stats object is not None
        assert stats is not None, "Backtest run failed to produce stats."

        # Add assertions to check key performance metrics
        assert isinstance(stats, pd.Series), "Stats should be a Pandas Series"
        assert "Equity Final [$]" in stats.index, "Equity Final not in stats"
        assert "Return [%]" in stats.index, "Return not in stats"
        assert "Win Rate [%]" in stats.index, "Win Rate not in stats"

        # You can add more specific assertions based on expected performance
        assert stats["Return [%]"] > -100, "Return should be greater than -100%"  # Example

        print(stats)
        bt.plot()  # Consider commenting out plot during automated testing
    else:
        print("Failed to retrieve SPY data from Interactive Brokers. Exiting.")
