import time
import pandas as pd
import websocket

from backtesting import Strategy
from backtesting.livetrading import executor
from backtesting.livetrading.broker import Broker, Pair
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

    def on_event(self, bar_event):
        # react on event from websocket
        pass


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

    exchange.subscribe_to_ticker_events(
        Pair(base_symbol="UTC", quote_symbol="SDT"), "3m", position_mgr.on_event
    )

    event_dis.set_strategy(strategy)

    event_dis.set_backtesting_partial(cash=100000)

    event_dis.run()
