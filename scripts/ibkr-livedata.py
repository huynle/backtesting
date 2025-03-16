import datetime
import threading
import time

from ibapi.client import *
from ibapi.ticktype import TickTypeEnum
from ibapi.wrapper import *

port = 4002


class TestApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.orderId = 0

    def nextValidId(self, orderId: OrderId):
        self.orderId = orderId

    def nextId(self):
        self.orderId += 1
        return self.orderId

    def error(self, reqId, errorCode, errorString, advancedOrderReject=""):
        print(
            f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}"
        )

    def tickPrice(self, reqId, tickType, price, attrib):
        print(
            f"reqId: {reqId}, tickType: {TickTypeEnum.to_str(tickType)}, price: {price}, attrib: {attrib}"
        )

    def tickSize(self, reqId, tickType, size):
        print(f"reqId: {reqId}, tickType: {TickTypeEnum.to_str(tickType)}, size: {size}")


app = TestApp()
app.connect("127.0.0.1", port, 0)
threading.Thread(target=app.run).start()
time.sleep(1)

mycontract = Contract()
mycontract.symbol = "AAPL"
mycontract.secType = "STK"
mycontract.exchange = "SMART"
mycontract.currency = "USD"

app.reqMarketDataType(3)
app.reqMktData(app.nextId(), mycontract, "232", False, False, [])
