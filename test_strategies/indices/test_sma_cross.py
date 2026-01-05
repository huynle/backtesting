# run this using `python2 strategies/sma-cross.py` from the main directory
import os

import pandas as pd
import pytest
import talib

from backtesting import Backtest, Strategy
from test_strategies.conftest import END_DATE, START_DATE, TEMP_DIR, minimum_strategy_check


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma = self.I(talib.SMA, price, 250)  # Calculate 200-day moving average
        self.bought = False

    def next(self):
        if not self.bought and self.data.Close > self.ma:
            self.buy()  # Buy if closing price is above 200-day MA
            self.bought = True
        elif self.bought and self.data.Close < self.ma:
            self.sell()  # Sell if closing price is below 200-day MA
            self.bought = False


@pytest.mark.parametrize(
    "ibkr_data",
    [
        {
            "symbol": "SPY",
            "bar_size": "1 day",
            "start_date": START_DATE,
            "end_date": END_DATE,
        },
        # {
        #     "symbol": "QQQ",
        #     "bar_size": "1 day",
        #     "start_date": START_DATE,
        #     "end_date": END_DATE,
        # },
    ],
    indirect=True,
)
def test_sma_cross_with_ibkr_data(ibkr_data):
    request_parameterized = ibkr_data[0]
    ibkr_data = ibkr_data[1]

    if ibkr_data is not None:
        # Extract parameters for filename
        symbol = request_parameterized.param.get("symbol")
        bar_size = request_parameterized.param.get("bar_size")
        start_date = request_parameterized.param.get("start_date")
        end_date = request_parameterized.param.get("end_date")
        strategy_name = "SmaCross"  # Hardcoded strategy name, adjust if needed

        # Construct the filename
        filename = f"{strategy_name}-{symbol}-{bar_size}-{start_date}-{end_date}".replace(" ", "_")
        filepath = os.path.join(TEMP_DIR, filename)

        bt = Backtest(ibkr_data, SmaCross, commission=0.002, exclusive_orders=True)
        stats = bt.run()
        bt.plot(
            filename=filepath, open_browser=False
        )  # Consider commenting out plot during automated testing

        # Assert that the stats object is not None
        assert stats is not None, "Backtest run failed to produce stats."

        # Add assertions to check key performance metrics
        assert isinstance(stats, pd.Series), "Stats should be a Pandas Series"
        assert "Equity Final [$]" in stats.index, "Equity Final not in stats"
        assert "Return [%]" in stats.index, "Return not in stats"
        assert "Win Rate [%]" in stats.index, "Win Rate not in stats"

        minimum_strategy_check(stats)

        # You can add more specific assertions based on expected performance
        assert stats["Return [%]"] > -100, "Return should be greater than -100%"  # Example
    else:
        print("Failed to retrieve SPY data from Interactive Brokers. Exiting.")
    # Initialize Interactive Brokers connection
