from enum import Enum
import shutil
import os

import pytest

from backtesting.provider.ibkr import InteractiveBrokers  # Importing the InteractiveBrokers class


class StatsColumns(Enum):
    START = "Start"
    END = "End"
    DURATION = "Duration"
    EXPOSURE_TIME = "Exposure Time [%]"
    EQUITY_FINAL = "Equity Final [$]"
    EQUITY_PEAK = "Equity Peak [$]"
    COMMISSIONS = "Commissions [$]"
    RETURN = "Return [%]"
    BUY_AND_HOLD_RETURN = "Buy & Hold Return [%]"
    RETURN_ANNUALIZED = "Return (Ann.) [%]"
    VOLATILITY_ANNUALIZED = "Volatility (Ann.) [%]"
    CAGR = "CAGR [%]"
    SHARPE_RATIO = "Sharpe Ratio"
    SORTINO_RATIO = "Sortino Ratio"
    CALMAR_RATIO = "Calmar Ratio"
    ALPHA = "Alpha [%]"
    BETA = "Beta"
    MAX_DRAWDOWN = "Max. Drawdown [%]"
    AVG_DRAWDOWN = "Avg. Drawdown [%]"
    MAX_DRAWDOWN_DURATION = "Max. Drawdown Duration"
    AVG_DRAWDOWN_DURATION = "Avg. Drawdown Duration"
    NUM_TRADES = "# Trades"
    WIN_RATE = "Win Rate [%]"
    BEST_TRADE = "Best Trade [%]"
    WORST_TRADE = "Worst Trade [%]"
    AVG_TRADE = "Avg. Trade [%]"
    MAX_TRADE_DURATION = "Max. Trade Duration"
    AVG_TRADE_DURATION = "Avg. Trade Duration"
    PROFIT_FACTOR = "Profit Factor"
    EXPECTANCY = "Expectancy [%]"
    SQN = "SQN"
    KELLY_CRITERION = "Kelly Criterion"
    STRATEGY = "_strategy"
    EQUITY_CURVE = "_equity_curve"
    TRADES = "_trades"


@pytest.fixture(scope="session")
def ibkr_data(request):
    """
    Fetches data from Interactive Brokers based on start and end dates.
    """
    symbol = request.param.get("symbol", "SPY")
    bar_size = request.param.get("bar_size", "1 day")
    start_date = request.param.get("start_date")  # Ensure these are strings in 'YYYYMMDD' format
    end_date = request.param.get("end_date")  # Ensure these are strings in 'YYYYMMDD' format

    ib = InteractiveBrokers()
    data = ib.get_data(symbol=symbol, bar_size=bar_size, start_date=start_date, end_date=end_date)
    ib.disconnect()  # Disconnect after fetching data
    assert data is not None, f"Failed to retrieve {symbol} data from Interactive Brokers."
    return (request, data)


# Define a temporary directory for test files
TEMP_DIR = "outputs"


# Ensure the temporary directory exists and is empty before each test session
def pytest_sessionstart(session):
    # if os.path.exists(TEMP_DIR):
    #     shutil.rmtree(TEMP_DIR)  # Remove if exists
    os.makedirs(TEMP_DIR, exist_ok=True)


def pytest_sessionfinish(session):
    # Clean up the temporary directory after the test session
    pass
    # if os.path.exists(TEMP_DIR):
    #     shutil.rmtree(TEMP_DIR)
