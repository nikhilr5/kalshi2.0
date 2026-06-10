"""Print live resting orders on Kalshi.  Run: python3 check_orders.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Aston"))
from kalshi_api import KalshiAPI

orders = KalshiAPI().get_orders("resting")
print(f"{len(orders)} live orders\n")
for o in sorted(orders, key=lambda x: (x.get("ticker",""), x.get("yes_price_dollars",0))):
    print(f"  {o.get('ticker',''):30s} {o.get('action',''):4s} {o.get('side','')}  "
          f"${float(o.get('yes_price_dollars',0)):.3f} x {o.get('remaining_count',0):>3d}  "
          f"{o.get('order_id','')[:8]}")
