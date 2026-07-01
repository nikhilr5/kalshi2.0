"""Discover which Kalshi sports series have settled markets with liquidity, and
inspect their structure (binary win? result field?). Probes a curated league list.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Aston"))
from kalshi_api import KalshiAPI

api = KalshiAPI()
CANDS = [
    "KXMLBGAME", "KXMLB", "KXNBAGAME", "KXNBA", "KXWNBAGAME", "KXWNBA",
    "KXNHLGAME", "KXNHL", "KXNFLGAME", "KXNFL", "KXEPLGAME", "KXUCL",
    "KXUFCFIGHT", "KXUFC", "KXATPMATCH", "KXWTAMATCH", "KXATP", "KXWTA",
    "KXTENNIS", "KXCS2MAP", "KXLOLGAME", "KXMLS", "KXSOCCER", "KXNCAAFB",
    "KXNCAABB", "KXF1", "KXGOLF", "KXPGA", "KXWIMBLEDON", "KXMLBWIN",
]
_f = lambda x: float(x) if x not in (None, "") else 0.0

print(f"{'series':16} {'n_mkt':>6} {'settled':>7} {'tot_vol':>10} {'sample ticker / result'}")
for s in CANDS:
    try:
        mk = api.get_markets(series_ticker=s)
    except Exception:
        mk = []
    if not mk:
        continue
    settled = [m for m in mk if m.get("result") in ("yes", "no")]
    vol = sum(_f(m.get("volume_fp")) for m in mk)
    ex = settled[0] if settled else mk[0]
    print(f"{s:16} {len(mk):>6} {len(settled):>7} {vol:>10.0f} "
          f"{ex.get('ticker','')[:34]} -> {ex.get('result')} | \"{(ex.get('title') or '')[:30]}\"")
