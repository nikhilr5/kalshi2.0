import queue
import threading
import uuid
from dataclasses import dataclass


@dataclass
class Quote:
    order_id: str
    price: float
    size: int


@dataclass
class PendingOp:
    request_id: str
    kind: str       # "place" | "cancel"
    side: str       # "bid" | "ask"
    price: float    # for "place"; not used for "cancel"


class OSM:

    def __init__(self, ticker, tolerance, api):
        self.ticker = ticker
        self.tolerance = tolerance
        self.api = api
        self.queue = queue.Queue(maxsize=1024)
        self.running = False

        # What Strategy wants
        self.desired_bid_price: float | None = None
        self.desired_ask_price: float | None = None

        # What's actually on Kalshi
        self.resting_bid: Quote | None = None
        self.resting_ask: Quote | None = None

        # In-flight API ops keyed by request_id
        self.pending_ops: dict[str, PendingOp] = {}