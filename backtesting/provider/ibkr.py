from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import time
import os


# Assuming _Data is defined elsewhere in your backtesting framework
# Example:
class _Data:
    def __init__(self, df):
        self.df = df  # DataFrame holding the OHLCV data


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}  # Store historical data

    def historicalData(self, reqId: int, bar):
        """
        Callback function to receive historical data.
        """
        if reqId not in self.data:
            self.data[reqId] = []
        self.data[reqId].append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """
        Callback function to signal the end of historical data retrieval.
        """
        print(f"Historical data for reqId {reqId} finished.")
        # Convert data to Pandas DataFrame and store it
        df = pd.DataFrame(
            self.data[reqId], columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        self.data[reqId] = df  # Store DataFrame instead of raw list

    def error(self, reqId, errorCode, errorString):
        print(f"Error: reqId={reqId}, code={errorCode}, string={errorString}")


class InteractiveBrokers:
    def __init__(self, client_id=0, port=4002, host="127.0.0.1"):
        self.app = IBapi()
        self.app.connect(host, port, client_id)
        self.app.nextOrderId = 0

        # Start the IB API thread
        import threading

        self.ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        self.ib_thread.start()
        time.sleep(1)  # Give the connection time to establish

    def run_loop(self):
        self.app.run()

    def create_contract(self, symbol, secType="STK", exchange="SMART", currency="USD"):
        """
        Creates a contract object for the given symbol.
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = secType
        contract.exchange = exchange
        contract.currency = currency
        return contract

    def get_data(
        self,
        symbol,
        bar_size,
        duration,
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        from_cache=True,
    ):
        """
        Retrieves historical data from Interactive Brokers and returns a _Data object.

        Args:
            symbol (str): The ticker symbol (e.g., "GOOG").
            bar_size (str): The size of each bar (e.g., "1 day", "1 hour", "5 mins").  Legal values are 1 secs, 5 secs, 10 secs, 15 secs, 30 secs, 1 min, 2 mins, 3 mins, 5 mins, 10 mins, 15 mins, 20 mins, 30 mins, 1 hour, 2 hours, 3 hours, 4 hours, 8 hours, 1 day, 1W, 1M
            duration (str): The duration of the historical data to retrieve (e.g., "1 Y", "6 M", "1 W"). Legal values are S, D, W, M, Y
            secType (str): Security Type (e.g., "STK", "FUT", "OPT").
            exchange (str): Exchange (e.g., "SMART", "ARCA").
            currency (str): Currency (e.g., "USD").
            from_cache (bool): If True, try to load data from a local CSV file.  Defaults to True.

        Returns:
            _Data: A _Data object containing the historical data.  Returns None on error.
        """
        # Construct the cache file path
        cache_dir = os.path.expanduser("~/.backtest/data")
        cache_file = os.path.join(cache_dir, f"{symbol}-{bar_size}-{duration}.csv")

        # Check if the data exists in the cache
        if from_cache and os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
                print(f"Data loaded from cache: {cache_file}")
                return _Data(df).df
            except Exception as e:
                print(f"Error loading data from cache: {e}")
                # If there's an error loading from cache, fall back to fetching from IB.
                pass

        contract = self.create_contract(symbol, sec_type, exchange, currency)
        req_id = self.app.nextOrderId
        self.app.nextOrderId += 1

        self.app.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime="",  # Empty string for current time
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",  # Or "MIDPOINT", "BID_ASK", etc.
            useRTH=1,  # 1 for regular trading hours, 0 for all hours
            formatDate=1,  # 1 for YYYYMMDD HH:MM:SS, 2 for epoch seconds
            keepUpToDate=False,  # True to keep receiving updates
            chartOptions=[],
        )

        # Wait for the data to be retrieved (you might need to adjust the sleep time)
        start_time = time.time()
        while (
            req_id not in self.app.data and time.time() - start_time < 10
        ):  # Timeout after 10 seconds
            time.sleep(0.1)

        if req_id in self.app.data:
            df = self.app.data.pop(req_id)  # Get the DataFrame and remove it from the dictionary

            # Save the data to the cache
            os.makedirs(cache_dir, exist_ok=True)  # Ensure the directory exists
            df.to_csv(cache_file)
            print(f"Data saved to cache: {cache_file}")

            return _Data(df).df
        else:
            print(f"Error: Could not retrieve data for {symbol} after timeout.")
            return None

    def disconnect(self):
        self.app.disconnect()


if __name__ == "__main__":
    # Example Usage:
    ib = InteractiveBrokers()  # Connect to IB Gateway/TWS

    # Get daily data for GOOG for the last year
    data = ib.get_data(symbol="GOOG", bar_size="1 day", duration="1 Y")

    if data:
        print(data.head())  # Print the first few rows of the DataFrame
    else:
        print("Failed to retrieve GOOG data.")

    # Get 5 minute data for AAPL for the last week
    data = ib.get_data(symbol="AAPL", bar_size="5 mins", duration="1 W")

    if data:
        print(data.head())  # Print the first few rows of the DataFrame
    else:
        print("Failed to retrieve AAPL data.")

    # Demonstrate loading from cache:
    data_from_cache = ib.get_data(symbol="GOOG", bar_size="1 day", duration="1 Y", from_cache=True)
    if data_from_cache is not None:
        print("Successfully loaded GOOG data from cache.")
        print(data_from_cache.head())

    # Demonstrate forcing a fresh download (bypassing cache):
    data_no_cache = ib.get_data(symbol="GOOG", bar_size="1 day", duration="1 Y", from_cache=False)
    if data_no_cache is not None:
        print("Successfully loaded GOOG data without cache.")
        print(data_no_cache.head())

    ib.disconnect()  # Disconnect from IB
