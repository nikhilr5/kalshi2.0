from ib_async import *

#class to handle connecting to interactive broker md through IB Gateway
class IBKRMd: 

    def __init__(self, callback):
        self.port = 4001
        self.host = '127.0.0.1'
        self.clientId = 1
        self.gateway = IB()
        self.callback = callback

    def connect(self):
        self.gateway.connect('127.0.0.1', 4001, clientId=2)
        self.gateway.reqMarketDataType(3)  # Switch to 1 when live data activates

    def subscribe(self):
        # Get front month gold
        details = self.gateway.reqContractDetails(Future('GC', exchange='COMEX'))
        details.sort(key=lambda x: x.contract.lastTradeDateOrContractMonth)
        gold = details[0].contract
        print(f"Streaming: {gold.localSymbol}")

        # Subscribe to streaming updates
        self.gateway.pendingTickersEvent += self.callback
        self.gateway.reqMktData(gold)

    def run(self):
        # Run forever
        self.gateway.run()