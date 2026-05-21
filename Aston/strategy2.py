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
        self.position = 0

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()


    def stop(self):
        print("Stopping Strategy :)")
        # Synchronous cancel BEFORE flipping `running`.  This guarantees
        # resting orders are killed on Kalshi (or timed out) before
        # OSM's worker can exit and abandon them.  Without the sync
        # version, the CANCEL_ALL message could sit in OSM's queue
        # while the worker exits — leaving live orders on the previous
        # market through a series switch.
        self.osm.cancel_all_sync(timeout=2.0)
        self.running = False

    def update_theo(self, theo):
        self.queue.put(('THEO', theo))
    
    def update_params(self, edge_bid, edge_ask, size_bid, size_ask,
                    max_position, tolerance=0.01):
      self.queue.put(('SETTINGS', {
          "edge_bid": edge_bid, "edge_ask": edge_ask,
          "size_bid": size_bid, "size_ask": size_ask,
          "max_position": max_position, "tolerance": tolerance,
      })) 

    def update_bbo(self, bbo: tuple):
        self.queue.put(('BBO', bbo))
    
    def cancel_all_orders_local(self) -> list:
        """Tell OSM to cancel both sides.  Returns [] since OSM owns the
        order IDs now (app.py used to batch-cancel via the returned list)."""
        self.osm.cancel_all()
        return []

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

    # ------------------------------------------------------------------
    # Position-cap helpers — same accounting as legacy Strategy:
    #   effective_long  = current long + resting bid (would add to long)
    #   effective_short = current short + resting ask (would add to short)
    # `*_remaining` is what we can still place on each side without
    # breaching max_position.
    # ------------------------------------------------------------------
    def _remaining_bid_capacity(self) -> int:
        # Position is sourced from OSM (which is the one process that
        # dedupes fills by trade_id) so a WS replay can't double-count.
        pos = self.osm.position
        resting_bid_size = (
            self.osm.resting_bid.size if self.osm.resting_bid else 0
        )
        effective_long = max(pos, 0) + resting_bid_size
        return max(self.max_position - effective_long, 0)

    def _remaining_ask_capacity(self) -> int:
        pos = self.osm.position
        resting_ask_size = (
            self.osm.resting_ask.size if self.osm.resting_ask else 0
        )
        effective_short = max(-pos, 0) + resting_ask_size
        return max(self.max_position - effective_short, 0)

    def _place_or_cancel_bid(self, price: float):
        """Place a bid sized within remaining capacity, or cancel if no
        room left.  Single entry point so the max_position cap always
        runs before reaching OSM."""
        sz = min(self.size_bid, self._remaining_bid_capacity())
        if sz <= 0:
            if self.osm.has_bid:
                self.osm.cancel_bid()
            return
        self.osm.ensure_bid(price, sz)

    def _place_or_cancel_ask(self, price: float):
        sz = min(self.size_ask, self._remaining_ask_capacity())
        if sz <= 0:
            if self.osm.has_ask:
                self.osm.cancel_ask()
            return
        self.osm.ensure_ask(price, sz)

    def _theo_update(self, theo):
        if self.theo == theo:
            return #nothing has changed no need to adjust anything
        self.theo = theo

        desired_bid = self.theo - self.edge_bid
        if desired_bid <= self.best_bid:
            self._place_or_cancel_bid(desired_bid)
        else:
            self._place_or_cancel_bid(self.best_bid)

        desired_ask = self.theo + self.edge_ask
        if desired_ask >= self.best_ask:
            self._place_or_cancel_ask(desired_ask)
        else:
            self._place_or_cancel_ask(self.best_ask)

    def _bbo_update(self, bbo):
        self.best_bid, self.best_ask = bbo

        #if not theo don't place an order
        if self.theo <= 0:
            return

        #cancelled and replace if we are the lone BBO
        #if bid moved up check to see if we can move up if theo support
        if self.osm.has_bid:
            if self.osm.bid > self.best_bid:
                self._place_or_cancel_bid(self.best_bid)
            elif self.theo - self.edge_bid > self.osm.bid:
                self._place_or_cancel_bid(self.theo - self.edge_bid)

        #same thing for ask
        if self.osm.has_ask:
            if self.osm.ask < self.best_ask:
                self._place_or_cancel_ask(self.best_ask)
            elif self.theo + self.edge_ask < self.osm.ask:
                self._place_or_cancel_ask(self.theo + self.edge_ask)

    def _settings_update(self, payload):
      self.edge_bid     = payload["edge_bid"]
      self.edge_ask     = payload["edge_ask"]
      self.size_bid     = payload["size_bid"]
      self.size_ask     = payload["size_ask"]
      self.max_position = payload["max_position"]
      # tolerance lives in OSM — forward
      self.osm.update_tolerance(payload["tolerance"])