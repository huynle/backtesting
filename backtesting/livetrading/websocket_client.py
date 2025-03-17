import logging
import time
from numpy import random
import websocket, json, _thread
import datetime

from typing import Dict, List, Set

from backtesting.livetrading.event import EventSource, EventProducer

logger = logging.getLogger(__name__)


class WSClient(EventProducer, websocket.WebSocket):
    """ "Class for channel based web socket clients.
    :param config: Config settings for exchange.
    """

    def __init__(self, config):
        super(WSClient, self).__init__(config["ws_url"])
        self.event_sources: Dict[str, EventSource] = {}
        self.pending_subscriptions: Set[str] = set()
        self.timeout = config["ws_timeout"]
        self.on_open = lambda ws: self.subscribe_msg()
        self.on_message = lambda ws, msg: self.handle_message(json.loads(msg))
        self.on_error = lambda ws, e: logger.warning(f"Error: {e}")
        self.on_close = self.on_close
        self._running = False
        self.thread = None

    def set_channel_event_source(self, channel: str, event_source: EventSource):
        assert channel not in self.event_sources, "channel already registered"
        self.event_sources[channel] = event_source
        self.pending_subscriptions.add(channel)

    def subscribe_msg(self):
        self.pending_subscriptions.update(self.event_sources.keys())
        channels = list(self.pending_subscriptions)
        self.subscribe_to_channels(channels)

    def on_close(self):
        self.pending_subscriptions = set()

    def main(self):
        if not self._running:
            self.thread = _thread.start_new_thread(self.run_forever, ())
            self._running = True

    def subscribe_to_channels(self, channels: List[str]):
        sub_msg = {"type": "subscribe", "product_ids": ["ETH-USD", "BTC-USD"], "channels": channels}
        self.send(json.dumps(sub_msg))
        logger.info(f"Subscribed to channels: {channels}")

    def handle_message(self, message: dict) -> None:
        channel = message.get("type")
        event_source = self.event_sources.get(channel)
        if event_source:
            event_source.push_to_queue(message)


class MockDataGenerator:
    """
    Generates mock data for testing purposes.
    """

    def __init__(self, pair: str = "ETH-USD"):
        """
        Initializes the MockDataGenerator.

        :param pair: The trading pair to generate data for.
        """
        self.pair = pair

    def generate_bar_data(self) -> Dict:
        """
        Generates mock bar data.

        :return: A dictionary containing mock bar data.
        """
        now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        open_price = random.uniform(100, 200)
        high_price = open_price + random.uniform(0, 10)
        low_price = open_price - random.uniform(0, 10)
        close_price = random.uniform(low_price, high_price)
        volume = random.uniform(10, 100)

        return {
            "t": int(time.time() * 1000),  # Current timestamp in milliseconds
            "E": int(time.time() * 1000),  # Event time
            "s": self.pair.replace("-", ""),  # Symbol
            "k": {
                "t": int(time.time() * 1000),
                "o": str(open_price),
                "h": str(high_price),
                "l": str(low_price),
                "c": str(close_price),
                "v": str(volume),
                "x": True,  # Kline is closed
            },
        }

    def generate_ticker_data(self) -> Dict:
        """
        Generates mock ticker data.

        :return: A dictionary containing mock ticker data.
        """
        price = random.uniform(100, 200)
        volume = random.uniform(10, 100)
        now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        return {
            "type": "ticker",
            "sequence": random.randint(1, 100),
            "product_id": self.pair,
            "price": str(price),
            "open_24h": str(random.uniform(90, 110)),
            "high_24h": str(price + random.uniform(0, 5)),
            "low_24h": str(price - random.uniform(0, 5)),
            "volume_24h": str(volume),
            "time": now.isoformat(),
        }


class MockWSClient(EventProducer):
    """
    A WebSocket client that generates mock data for testing purposes.

    This class extends both the EventProducer and websocket.WebSocket classes to provide
    a mock WebSocket client that generates random data and pushes it to the appropriate
    event sources.
    """

    def __init__(self, config):
        """
        Initializes the MockWSClient.

        :param config: Configuration settings for the WebSocket client, including the WebSocket URL and timeout.
        """
        # super().__init__(config["ws_url"])
        self.event_sources: Dict[str, EventSource] = {}
        self.pending_subscriptions: Set[str] = set()
        self.timeout = config["ws_timeout"]
        # self.on_open = lambda ws: self.subscribe_msg()  # No need to subscribe in mock
        # self.on_message = lambda ws, msg: self.handle_message(json.loads(msg)) # Override handle_message
        self.on_error = lambda ws, e: logger.warning(f"Error: {e}")
        self.on_close = self.on_close
        self._running = False
        self.thread = None
        self.data_generator = MockDataGenerator()  # Initialize the mock data generator

    def set_channel_event_source(self, channel: str, event_source: EventSource):
        """
        Sets the event source for a given channel.

        :param channel: The channel to set the event source for.
        :param event_source: The event source to associate with the channel.
        """
        assert channel not in self.event_sources, "channel already registered"
        self.event_sources[channel] = event_source
        self.pending_subscriptions.add(channel)

    def subscribe_msg(self):
        """
        Subscribes to all pending channels.
        """
        self.pending_subscriptions.update(self.event_sources.keys())
        channels = list(self.pending_subscriptions)
        self.subscribe_to_channels(channels)

    def on_close(self):
        """
        Handles the WebSocket close event.
        """
        self.pending_subscriptions = set()

    def main(self):
        """
        Starts the mock data generation and pushing process in a separate thread.
        """
        if not self._running:
            self.thread = _thread.start_new_thread(self.run_mock_data, ())
            self._running = True

    def subscribe_to_channels(self, channels: List[str]):
        """
        Mocks the subscription to the given channels.

        :param channels: A list of channels to subscribe to.
        """
        sub_msg = {"type": "subscribe", "product_ids": ["ETH-USD", "BTC-USD"], "channels": channels}
        # self.send(json.dumps(sub_msg))  # No need to send in mock
        logger.info(f"Subscribed to channels: {channels}")

    def handle_message(self, message: dict) -> None:
        """
        Handles incoming messages from the WebSocket.

        :param message: A dictionary containing the message data.
        """
        channel = message.get("type")
        event_source = self.event_sources.get(channel)
        if event_source:
            event_source.push_to_queue(message)

    def run_mock_data(self):
        """
        Generates and pushes mock data to the subscribed channels at regular intervals.
        """
        while self._running:
            for channel, event_source in self.event_sources.items():
                if "kline" in channel:
                    message = {"data": self.data_generator.generate_bar_data()}
                    event_source.push_to_queue(message)
                elif channel == "ticker":
                    message = self.data_generator.generate_ticker_data()
                    event_source.push_to_queue(message)
            time.sleep(1)  # Simulate real-time data updates
