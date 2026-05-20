import queue
import threading

class Strategy2:

    def __init__(self, ticker, 
                 strike, edge_bid, 
                 edge_ask, size_bid, 
                 size_ask, max_position, 
                 osm):
        self.ticker = ticker
        self.strike = strike
        self.edge_bid = edge_bid
        self.edge_ask = edge_ask
        self.size_bid = size_bid
        self.size_ask = size_ask
        self.max_position = max_position
        self.queue = queue.Queue()
        self.running = False
        self.osm = osm

        self.theo = 0
        self.best_bid = 0
        self.best_ask = 0

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()


    def stop(self):
        print("Stopping Strategy :)")
        self.running = False

    def update_theo(self, theo):
        self.queue.put(('THEO', theo))
    
    def update_params(self, settings):
        self.queue.put(('SETTINGS', settings))
    

    #actually run the loop and continuously pull from the queue
    def _run(self):
        while self.running:
            try: 
                header, payload =  self.queue.get(timeout=0.01)
                if header == 'BBO':
                    self._bbo_update(payload)
                elif header == "THEO":
                    self._theo_update(payload)
                elif header == "SETTINGS":
                    self._settings_update(payload)
            except queue.Empty:
                pass
            finally:
                pass

    def _theo_update(self, theo):
        if self.theo == theo:
            return #nothing has changed no need to adjust anything
        self.theo = theo

        desired_bid = self.theo - self.edge_bid
        if desired_bid <= self.best_bid:
            self.osm.ensure_bid(desired_bid)
        else:
            self.osm.ensure_bid(self.best_bid)

        desired_ask = self.theo + self.edge_ask
        if desired_ask >= self.best_ask: 
            self.osm.ensure_ask(desired_ask)
        else:
            self.osm.ensure_ask(self.best_ask)

    def _bbo_update(self, bbo):
        self.best_bid, self.best_ask = bbo

        #if not theo don't place an order
        if self.theo <= 0:
            return
        
        #cancelled and replace if we are the lone BBO
        #if bid moved up check to see if we can move up if theo support
        if self.osm.has_bid():
            if self.osm.bid > self.best_bid:
                self.osm.ensure_bid(self.best_bid)
            elif  self.theo - self.edge_bid > self.osm.bid:
                self.osm.ensure_bid(self.theo - self.edge_bid)

        #same thing for ask
        if self.osm.has_ask():
            if self.osm.ask < self.best_ask:
                self.osm.ensure_ask(self.best_ask)
            elif self.theo + self.edge_ask < self.osm.ask:
                self.osm.ensure_ask(self.theo + self.edge_ask)

    #TODO
    def _settings_update(self):
        pass