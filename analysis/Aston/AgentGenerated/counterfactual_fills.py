"""Counterfactual fill simulation — what would fill count and EV look
like at wider edges?

For each `placed` event:
  • t_placed, t_end (next cancel/fill event for same client_order_id)
  • theo_at_placed (merge_asof on theo_state)
  • For candidate edge Δ, hypothetical price P_hyp:
      buy:   P_hyp = theo − Δ
      sell:  P_hyp = theo + Δ
  • Check trade tape between [t_placed, t_end] on this ticker:
      buy  fills iff any trade with yes_price ≤ P_hyp and YES-bid hit
      sell fills iff any trade with yes_price ≥ P_hyp and YES-ask hit
  • Realized = (settlement − P_hyp) × sgn × 100  (cents)

Caveats:
  - Upper bound on fill probability — ignores queue position behind
    other resting orders at the same price.
  - Assumes the trade tape would be unchanged if I were quoting at
    a different edge.  My 1-lot quotes barely move the market, so
    this is fine for first-pass.
  - Wider edges only.  For narrower hypotheticals the inequality
    direction would need to flip and assumptions get messier.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import fetch_settlements_from_api, load_all_data  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI  # noqa: E402


SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"
CACHE_DIR     = (Path(__file__).resolve().parent.parent.parent
                 / "backtesting" / "_trades_cache")
EDGE_LEVELS_C = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


# =============================================================================
# Load placements + lifetimes + theo
# =============================================================================
theo, _book, _spot, fills, events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)

events  = events.sort_values('ts').reset_index(drop=True)
placed  = (events[events['event_type'] == 'placed']
           .drop_duplicates('client_order_id', keep='first')
           .reset_index(drop=True))
placed  = placed[placed['side'] == 'yes'].reset_index(drop=True)
placed['sgn'] = np.where(placed['action'] == 'buy', +1, -1)

end_events = events[events['event_type'].isin(['cancelled', 'filled'])]
end_per_order = (end_events.sort_values('ts')
                  .drop_duplicates('client_order_id', keep='first')
                  [['client_order_id', 'ts']]
                  .rename(columns={'ts': 't_end'}))
placed = placed.merge(end_per_order, on='client_order_id', how='left')
# Fallback: if no cancel/fill event, give it 15 min from placement.
placed['t_end'] = placed['t_end'].fillna(
    placed['ts'] + pd.Timedelta(seconds=900))

theo_lookup = theo[['ts', 'ticker', 'theo']].sort_values('ts')
placed = placed.sort_values('ts')
placed = pd.merge_asof(
    placed, theo_lookup, on='ts', by='ticker', direction='backward')
placed['theo_at_placed'] = placed['theo']
placed = placed.dropna(subset=['theo_at_placed']).reset_index(drop=True)

print(f"[sim] {len(placed):,} placements with valid theo")


# =============================================================================
# Settlements
# =============================================================================
api = KalshiAPI()
tickers = sorted(set(placed['ticker'].dropna().unique()))
settlements = fetch_settlements_from_api(
    tickers, api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json")
placed['outcome'] = placed['ticker'].map(settlements)


# =============================================================================
# Load trade tape, classify YES bid vs ask hits
# =============================================================================
def load_trades_for_tickers(tickers: list[str]) -> pd.DataFrame:
    chunks = []
    for t in tickers:
        p = CACHE_DIR / f"{t}.json"
        if not p.exists():
            continue
        with p.open() as f:
            data = json.load(f)
        if not data:
            continue
        df = pd.DataFrame(data)
        chunks.append(df)
    out = pd.concat(chunks, ignore_index=True)
    out['yes_price'] = pd.to_numeric(out['yes_price_dollars'], errors='coerce')
    out['ts'] = pd.to_datetime(out['created_time'], utc=True)
    # Verified empirically by matching real fills to the tape:
    #   (no, ask)  = trade printed with YES-bid-side maker  → "YES bid hit"
    #   (yes, bid) = trade printed with YES-ask-side maker  → "YES ask hit"
    out['yes_bid_hit'] = (
        (out['taker_outcome_side'] == 'no') & (out['taker_book_side'] == 'ask'))
    out['yes_ask_hit'] = (
        (out['taker_outcome_side'] == 'yes') & (out['taker_book_side'] == 'bid'))
    return out[['ticker', 'ts', 'yes_price', 'yes_bid_hit', 'yes_ask_hit']]


trades = load_trades_for_tickers(tickers)
trades = trades.sort_values(['ticker', 'ts']).reset_index(drop=True)
print(f"[sim] {len(trades):,} trades loaded")


# =============================================================================
# Per-ticker fill-probe — vectorised per (placement × edge_level)
# =============================================================================
trades_by_ticker = {t: g.reset_index(drop=True) for t, g in trades.groupby('ticker')}

# Pre-allocate outputs: filled_buy[i, e] and filled_sell[i, e]
n_p   = len(placed)
n_ed  = len(EDGE_LEVELS_C)
edges_arr = np.array(EDGE_LEVELS_C) / 100.0

filled = np.zeros((n_p, n_ed), dtype=bool)
P_hyp_grid = np.full((n_p, n_ed), np.nan)

theo_vals = placed['theo_at_placed'].to_numpy()
sgn_vals  = placed['sgn'].to_numpy()
# P_hyp = theo - sgn * edge  (buy sgn=+1 → P=theo-edge; sell sgn=-1 → P=theo+edge)
P_hyp_grid = theo_vals[:, None] - sgn_vals[:, None] * edges_arr[None, :]

placed_ts_ns = placed['ts'].astype('int64').to_numpy()
placed_end_ns = placed['t_end'].astype('int64').to_numpy()
placed_tickers = placed['ticker'].to_numpy()
placed_actions = placed['action'].to_numpy()

print("[sim] running probe across placements...")
for i in range(n_p):
    tk = placed_tickers[i]
    g = trades_by_ticker.get(tk)
    if g is None or len(g) == 0:
        continue
    g_ns = g['ts'].astype('int64').to_numpy()
    lo = np.searchsorted(g_ns, placed_ts_ns[i], side='left')
    hi = np.searchsorted(g_ns, placed_end_ns[i], side='right')
    if hi <= lo:
        continue
    yp = g['yes_price'].to_numpy()[lo:hi]
    if placed_actions[i] == 'buy':
        mask_side = g['yes_bid_hit'].to_numpy()[lo:hi]
        # Fill iff any trade with yes_price ≤ P_hyp AND yes_bid_hit
        for e_idx in range(n_ed):
            if ((yp <= P_hyp_grid[i, e_idx]) & mask_side).any():
                filled[i, e_idx] = True
    else:
        mask_side = g['yes_ask_hit'].to_numpy()[lo:hi]
        for e_idx in range(n_ed):
            if ((yp >= P_hyp_grid[i, e_idx]) & mask_side).any():
                filled[i, e_idx] = True

print("[sim] probe complete")


# =============================================================================
# Realized P&L per placement × edge
# =============================================================================
# realized_c[i, e] = (outcome - P_hyp) * sgn * 100, where sgn is direction in $-PnL terms.
# For buy yes: pnl = outcome - P_hyp_buy.  For sell yes: pnl = P_hyp_sell - outcome.
# Note: theo - sgn*edge with sgn=+1 (buy) gives buy P_hyp; sgn=-1 (sell) gives sell P_hyp.
# Direction sign for P&L: buy=+1, sell=-1 — same as our sgn convention.
outcomes = placed['outcome'].to_numpy()
realized_c = (outcomes[:, None] - P_hyp_grid) * sgn_vals[:, None] * 100.0

# Mask unfilled and unsettled
realized_c = np.where(filled, realized_c, np.nan)
realized_c = np.where(np.isnan(outcomes)[:, None], np.nan, realized_c)


# =============================================================================
# Summary
# =============================================================================
n_days = (placed['ts'].max() - placed['ts'].min()).total_seconds() / 86400.0
print(f"[sim] window spans {n_days:.2f} days\n")

rows = []
for side, action in [('buy', 'buy'), ('sell', 'sell')]:
    mask_side = (placed['action'] == action).to_numpy()
    for e_idx, e in enumerate(EDGE_LEVELS_C):
        n_pl = int(mask_side.sum())
        f_col = filled[:, e_idx] & mask_side
        n_fi = int(f_col.sum())
        rv = realized_c[mask_side, e_idx]
        rv = rv[~np.isnan(rv)]
        mean_r = float(rv.mean()) if len(rv) else np.nan
        ev_c   = (n_fi / n_pl) * (mean_r if mean_r == mean_r else 0) if n_pl else 0
        # Daily P&L: sum of realized cents in this side / n_days, ÷100 to dollars
        total_d = (rv.sum() / 100.0) / n_days if n_days > 0 else 0
        rows.append({
            'side': side, 'edge_c': e,
            'n_placed': n_pl, 'n_filled': n_fi,
            'fill_pct': 100 * n_fi / n_pl if n_pl else 0,
            'realized_c_per_fill': mean_r,
            'EV_c_per_placement': ev_c,
            'dollars_per_day': total_d,
        })

out = pd.DataFrame(rows)
pd.set_option('display.float_format', '{:+.3f}'.format)
pd.set_option('display.width', 200)
print("=== Counterfactual fill simulation (yes-side placements) ===")
print(out.to_string(index=False))

# Combined daily P&L per edge pairing (assuming both sides set at same edge for simplicity)
print("\n=== Combined daily P&L per edge level (both sides) ===")
combo = (out.groupby('edge_c')[['n_filled', 'dollars_per_day']]
            .sum().rename(columns={'dollars_per_day': 'combined_$_per_day'}))
print(combo)

# Sanity check vs actuals
real_fills = fills[fills['side'] == 'yes'] if 'side' in fills.columns else fills
real_fills = real_fills.copy()
real_fills['outcome'] = real_fills['ticker'].map(settlements)
real_fills['fsgn'] = np.where(real_fills['action'] == 'buy', +1, -1)
real_fills['realized_c'] = (real_fills['outcome'] - real_fills['price']) * real_fills['fsgn'] * 100
real_actuals = (real_fills.dropna(subset=['realized_c'])
                          .groupby('action')['realized_c']
                          .agg(['count', 'sum', 'mean']))
real_actuals['dollars_per_day'] = real_actuals['sum'] / 100.0 / n_days
print("\n=== Actuals (real fills, real edges) for sanity check ===")
print(real_actuals)
