from functools import partial
import time
from typing import assert_never, assert_type
import pandas as pd
import websocket

from backtesting import Strategy
from backtesting.backtesting import Backtest
from backtesting.live import LiveMarketOhlcv, LiveTrade
from backtesting.livetrading import executor
from backtesting.livetrading.broker import Broker, Contract
from backtesting.livetrading.converter import ohlcv_to_dataframe
from backtesting.livetrading.event import Bar, Contract, Symbol, Ticker, TickerEvent
# from backtesting.livetrading.config import config


def SMA(arr: pd.Series, n: int) -> pd.Series:
    """
    Returns `n`-period simple moving average of array `arr`.
    """
    return pd.Series(arr).rolling(n).mean()


class LiveStrategy(Strategy):
    n1 = 10
    n2 = 20

    def __init__(self, broker, data, params):
        super().__init__(broker=broker, data=data, params=params)

    def init(self):
        sma1 = self.I(SMA, self.data.Close, self.n1)
        sma2 = self.I(SMA, self.data.Close, self.n2)

    def set_atr_periods(self):
        if len(self.data) > 1:
            print(self.data.High, self.data.Low)

    def next(self):
        print(self.data)


class PositionManager:
    def __init__(self, exchange, position_amount):
        assert position_amount > 0
        self.exchange = exchange
        self.position_amount = position_amount

        # "`data` must be a pandas.DataFrame with columns "
        # "'Open', 'High', 'Low', 'Close', and (optionally) 'Volume'"
        self._backtesting = Backtest(
            data=pd.DataFrame(
                {
                    "Open": [1],
                    "High": [1],
                    "Low": [1],
                    "Close": [1],
                    "Volume": [100],
                },
                index=[pd.Timestamp.utcnow()],  # Set the index to a list containing the timestamp
            ),
            strategy=LiveStrategy,
        )

        self.live_trade = LiveTrade(self._backtesting)
        self.live_trade.init()

    def on_event(self, event):
        # react on event from websocket
        assert isinstance(event, TickerEvent)

        self.live_trade.on_bar(
            LiveMarketOhlcv(
                symbol=event.data.contract.symbol,
                open_price=event.data.Open,
                high_price=event.data.High,
                low_price=event.data.Low,
                close_price=event.data.Close,
                total_qty=event.data.Volume,
                timestamp_second=pd.Timestamp(event.data.Date, unit="s"),
            )
        )


def test_live():
    # websocket.enableTrace(False)
    from configloader import ConfigLoader

    config = ConfigLoader()
    config.update_from_json_file("./env")

    event_dis = executor.EventDispatcher(LiveStrategy)

    exchange = Broker(event_dis, config=config)

    # pair_info = exchange.get_pair_info("BTC-USD")

    position_mgr = PositionManager(exchange, 0.8)

    strategy = LiveStrategy(exchange, [], {})

    # exchange.subscribe_to_ticker_events(
    #     Pair(base_symbol="UTC", quote_symbol="SDT"), "3m", position_mgr.on_event
    # )
    exchange.subscribe_to_ticker_events(
        Symbol(symbol="SPY"),
        "1d",
        position_mgr.on_event,
    )

    event_dis.set_strategy(strategy)

    event_dis.set_backtesting_partial(cash=100000)

    event_dis.run()
