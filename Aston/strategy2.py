import queue
import threading

class Strategy2:

    def __init__(self, ticker,
                 strike, edge_bid,
                 edge_ask, size_bid,
                 size_ask, osm,
                 bid_enabled=True, ask_enabled=True,
                 range_min_bid=0.0, range_max_bid=1.0,
                 range_min_ask=0.0, range_max_ask=1.0):
        self.ticker = ticker
        self.strike = strike
        self.edge_bid = edge_bid
        self.edge_ask = edge_ask
        self.size_bid = size_bid
        self.size_ask = size_ask
        # Per-side enable + price-range gates.  A side only quotes when
        # it's enabled AND its quote price (theo ± edge) is within
        # [range_min, range_max] for that side.  Defaults reproduce the
        # always-on, full-range [0,1] behavior.
        self.bid_enabled = bid_enabled
        self.ask_enabled = ask_enabled
        self.range_min_bid = range_min_bid
        self.range_max_bid = range_max_bid
        self.range_min_ask = range_min_ask
        self.range_max_ask = range_max_ask
        self.queue = queue.Queue()
        self.running = False
        # OSM owns position + max_position + capacity clamping.
        # Strategy2 only signals desired price + desired size; OSM
        # decides whether there's room and clamps before placing.
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
        # Step 1: flip running OFF and wait for the worker thread to
        # exit so Strategy2 stops pushing ENSURE_* into OSM's queue.
        # Without this, the worker could process one last BBO/THEO
        # event and call osm.ensure_bid() AFTER cancel_all_sync had
        # already cleared OSM's state — leaving a freshly-placed
        # order on the about-to-roll market.
        self.running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        # Step 2: OSM-side teardown — waits for in-flight places to
        # land, then cancels them along with anything resting.
        self.osm.cancel_all_sync(timeout=2.0)

    def update_theo(self, theo):
        self.queue.put(('THEO', theo))

    def update_params(self, edge_bid, edge_ask, size_bid, size_ask,
                    max_position, tolerance=0.01, dwell_s=1.0,
                    bid_enabled=True, ask_enabled=True,
                    range_min_bid=0.0, range_max_bid=1.0,
                    range_min_ask=0.0, range_max_ask=1.0):
      self.queue.put(('SETTINGS', {
          "edge_bid": edge_bid, "edge_ask": edge_ask,
          "size_bid": size_bid, "size_ask": size_ask,
          "max_position": max_position, "tolerance": tolerance,
          "dwell_s": dwell_s,
          "bid_enabled": bid_enabled, "ask_enabled": ask_enabled,
          "range_min_bid": range_min_bid, "range_max_bid": range_max_bid,
          "range_min_ask": range_min_ask, "range_max_ask": range_max_ask,
      }))

    def update_bbo(self, bbo: tuple):
        self.queue.put(('BBO', bbo))

    def cancel_all_orders_local(self) -> list:
        """Tell OSM to cancel both sides.  Returns [] since OSM owns the
        order IDs now (app.py used to batch-cancel via the returned list)."""
        self.osm.cancel_all()
        return []
    
    def on_fill(self, action, price, count, side):
        # No-op.  app.py's _on_ws_fill calls this for legacy parity;
        # Strategy2 receives fills via OSM's fill-channel handler and
        # OSM's strategy_queue("FILL", ...) forwarding instead.
        pass

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

    # ------------------------------------------------------------------
    # Pricing — pure function of (theo, best_bid, best_ask, edges).
    # No OSM state read.  Size is just `self.size_*`; OSM clamps it to
    # remaining capacity (and cancels if no room) on its single-threaded
    # worker, so no race between cap check and place.
    # ------------------------------------------------------------------

    def _theo_update(self, theo):
        if self.theo == theo:
            return #nothing has changed no need to adjust anything
        self.theo = theo
        self._repost()

    def _bbo_update(self, bbo):
        self.best_bid, self.best_ask = bbo
        if self.theo <= 0:
            return
        self._repost()

    def _repost(self):
        # Desired prices: post at fair, capped at BBO (never lonely-
        # at-BBO).
        desired_bid = min(self.theo - self.edge_bid, self.best_bid)
        desired_ask = max(self.theo + self.edge_ask, self.best_ask)

        # ---- Per-side enable + range gate ------------------------
        # Quote price for the gate is the un-clamped theo ± edge (what
        # the user dials in), NOT the BBO-clamped desired price.  A side
        # is gated off when disabled OR when its quote price falls
        # outside [range_min, range_max] for that side; gated-off means
        # pull any resting order on that side (same as the risk guards).
        ask_quote = self.theo + self.edge_ask
        bid_quote = self.theo - self.edge_bid
        ask_gated = (not self.ask_enabled
                     or not (self.range_min_ask <= ask_quote <= self.range_max_ask))
        bid_gated = (not self.bid_enabled
                     or not (self.range_min_bid <= bid_quote <= self.range_max_bid))

        # ---- Ask-side guards ------------------------------------
        # Tail-market: yes_bid ≥ 98¢ means the contract is virtually
        # certain to settle YES.  Adverse-selection bait — no edge to
        # extract by selling near-certain YES.  Pull instead.
        # Fair-range: a sell at ≥ $1 will be rejected by Kalshi
        # post-only anyway; cancel instead of churning the API.
        if ask_gated or self.best_bid >= 0.98 or desired_ask >= 1.0:
            self.osm.cancel_ask()
        else:
            self.osm.ensure_ask(desired_ask, self.size_ask)

        # ---- Bid-side guards (mirror) ----------------------------
        if bid_gated or self.best_ask <= 0.02 or desired_bid <= 0.0:
            self.osm.cancel_bid()
        else:
            self.osm.ensure_bid(desired_bid, self.size_bid)

    def _settings_update(self, payload):
      self.edge_bid     = payload["edge_bid"]
      self.edge_ask     = payload["edge_ask"]
      self.size_bid     = payload["size_bid"]
      self.size_ask     = payload["size_ask"]
      self.bid_enabled    = payload.get("bid_enabled", True)
      self.ask_enabled    = payload.get("ask_enabled", True)
      self.range_min_bid  = payload.get("range_min_bid", 0.0)
      self.range_max_bid  = payload.get("range_max_bid", 1.0)
      self.range_min_ask  = payload.get("range_min_ask", 0.0)
      self.range_max_ask  = payload.get("range_max_ask", 1.0)
      # tolerance + max_position live in OSM — forward via its queue.
      self.osm.update_tolerance(payload["tolerance"])
      self.osm.update_max_position(payload["max_position"])
      self.osm.update_dwell(payload.get("dwell_s", 1.0))
      # Gate may have flipped a side on/off or moved its range — re-run
      # the quoting decision so the change takes effect without waiting
      # for the next theo/BBO tick.
      if self.theo > 0:
          self._repost()