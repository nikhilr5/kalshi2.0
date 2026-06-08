"""Classify today's stale-fill cohort (5-30s resting duration) into
the four candidate causes. See `stale_fill_classification.md` for the
written summary.

Run: python3 analysis/Aston/AgentGenerated/stale_fill_classification.py
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

DB = Path("~/Desktop/Kalshi2.0/analysis/backtesting/data/KXETH15M-26MAY22.db").expanduser()
TOLERANCE = 0.01    # $0.01 = 1¢ from aston_settings.json
STALE_LO, STALE_HI = 5.0, 30.0
CHURN_WINDOW_S = 2.0


def main():
    con = sqlite3.connect(DB)
    fills = pd.read_sql("SELECT client_order_id, kalshi_ts, ts, ticker, action, "
                        "price, count FROM fills", con,
                        parse_dates=["kalshi_ts", "ts"])
    placed = pd.read_sql("SELECT client_order_id, kalshi_ts AS placed_kalshi_ts, "
                         "ts AS placed_ts, price AS placed_price, side, action "
                         "FROM order_events WHERE event_type='placed'", con,
                         parse_dates=["placed_kalshi_ts", "placed_ts"])
    cancelled = pd.read_sql("SELECT client_order_id, kalshi_ts AS cancel_kalshi_ts "
                            "FROM order_events WHERE event_type='cancelled'", con,
                            parse_dates=["cancel_kalshi_ts"])
    theo = pd.read_sql("SELECT ts, theo FROM theo_state", con,
                       parse_dates=["ts"]).rename(columns={"ts": "theo_ts"})
    theo = theo.dropna(subset=["theo_ts"]).sort_values("theo_ts")
    book = pd.read_sql("SELECT ts FROM kalshi_book", con,
                       parse_dates=["ts"])
    book = book.dropna(subset=["ts"]).sort_values("ts")
    con.close()

    # (a) join fills <- placed on client_order_id
    df = fills.merge(placed[["client_order_id", "placed_kalshi_ts", "placed_ts",
                             "placed_price", "side"]],
                     on="client_order_id", how="left")
    n_unmatched = df["placed_kalshi_ts"].isna().sum()
    df = df.dropna(subset=["placed_kalshi_ts"]).copy()

    # (b) resting duration in seconds
    df["rest_s"] = (df["kalshi_ts"] - df["placed_kalshi_ts"]).dt.total_seconds()
    df = df[df["rest_s"] >= 0]

    # (c) stale cohort
    stale = df[(df["rest_s"] >= STALE_LO) & (df["rest_s"] <= STALE_HI)].copy()
    print(f"total fills: {len(fills)}  unmatched (no placed event): {n_unmatched}")
    print(f"matched fills: {len(df)}")
    print(f"rest_s percentiles: "
          f"p50={df['rest_s'].median():.2f}s  "
          f"p90={df['rest_s'].quantile(0.9):.2f}s  "
          f"p99={df['rest_s'].quantile(0.99):.2f}s")
    print(f"stale cohort (5s <= rest_s <= 30s): n={len(stale)}\n")

    # (d) attach theo at placement + at fill via asof
    stale = stale.sort_values("placed_kalshi_ts")
    stale = pd.merge_asof(stale, theo, left_on="placed_kalshi_ts",
                          right_on="theo_ts", direction="backward")
    stale = stale.rename(columns={"theo": "theo_at_placed",
                                  "theo_ts": "theo_placed_ts"})
    stale = stale.sort_values("kalshi_ts")
    stale = pd.merge_asof(stale, theo, left_on="kalshi_ts",
                          right_on="theo_ts", direction="backward")
    stale = stale.rename(columns={"theo": "theo_at_fill",
                                  "theo_ts": "theo_fill_ts"})

    # book-churn count in [fill - 2s, fill]
    book_sorted = book["ts"].sort_values().reset_index(drop=True)
    fill_ts = stale["kalshi_ts"].values
    lo = stale["kalshi_ts"] - pd.Timedelta(seconds=CHURN_WINDOW_S)
    book_ns = book_sorted.values.astype("datetime64[ns]")
    idx_hi = np.searchsorted(book_ns, fill_ts.astype("datetime64[ns]"), side="right")
    idx_lo = np.searchsorted(book_ns, lo.values.astype("datetime64[ns]"), side="left")
    stale["book_churn_2s"] = idx_hi - idx_lo

    # (e) signed theo drift: positive = drifted AGAINST our position.
    # Buy/bid resting: we got long. Theo dropping (drift negative) is
    # adverse; flip sign so positive = adverse.
    # Sell/ask resting: we got short. Theo rising (drift positive) is
    # adverse; keep sign.
    raw = stale["theo_at_fill"] - stale["theo_at_placed"]
    is_buy = stale["action"].str.lower() == "buy"
    stale["theo_drift_signed"] = np.where(is_buy, -raw, raw)
    stale["theo_drift_abs"] = raw.abs()

    # (f) cancelled-event lookup for case 2
    cancelled_ids = set(cancelled["client_order_id"].dropna())
    stale["has_cancel_event"] = stale["client_order_id"].isin(cancelled_ids)

    # Bucketing (mutually exclusive, priority order: case 3 first
    # because "no drift" overrides everything — those orders should
    # have sat regardless of churn or limbo).
    drift_adv = stale["theo_drift_signed"] > TOLERANCE
    case3 = stale["theo_drift_abs"] <= TOLERANCE  # within tolerance either direction
    # The cohort is "stale fills" — i.e. they DID fill. So a 'cancelled'
    # event for the same client_order_id is rare (fill precedes cancel
    # ack). What we care about for case 2 is: strategy WANTED to cancel
    # (drift>tol, adverse) AND no cancel ever materialized.
    case2 = (~case3) & drift_adv & (~stale["has_cancel_event"])

    # case 1 — high book churn in the 2s pre-fill, top quartile of cohort
    churn_q75 = stale["book_churn_2s"].quantile(0.75)
    case1 = (~case3) & (~case2) & (stale["book_churn_2s"] >= churn_q75)
    case4 = ~(case1 | case2 | case3)

    stale["bucket"] = np.select(
        [case3, case2, case1, case4],
        ["3_no_reprice_needed", "2_pending_cancel_limbo",
         "1_queue_backup_high_churn", "4_unknown"],
        default="4_unknown")

    print("=== Cohort stats ===")
    print(f"book_churn_2s: p25={stale['book_churn_2s'].quantile(.25):.0f}  "
          f"p50={stale['book_churn_2s'].median():.0f}  "
          f"p75={churn_q75:.0f}  "
          f"p90={stale['book_churn_2s'].quantile(.9):.0f}  "
          f"max={stale['book_churn_2s'].max():.0f}")
    print(f"theo_drift_signed: p50={stale['theo_drift_signed'].median()*100:+.2f}¢  "
          f"p90={stale['theo_drift_signed'].quantile(.9)*100:+.2f}¢  "
          f"max={stale['theo_drift_signed'].max()*100:+.2f}¢")
    print(f"theo_drift_abs: p50={stale['theo_drift_abs'].median()*100:.2f}¢  "
          f"p90={stale['theo_drift_abs'].quantile(.9)*100:.2f}¢")
    print(f"frac with cancel event ever: "
          f"{stale['has_cancel_event'].mean()*100:.1f}%")
    print(f"frac with drift > tolerance (signed adverse): "
          f"{drift_adv.mean()*100:.1f}%\n")

    counts = stale["bucket"].value_counts().sort_index()
    print("=== Bucket breakdown ===")
    for b, n in counts.items():
        print(f"  {b:36s} n={n:4d}  ({n/len(stale)*100:.1f}%)")
    print()

    print("=== Example rows per bucket (up to 3 each) ===")
    cols = ["client_order_id", "kalshi_ts", "rest_s", "action",
            "placed_price", "price", "theo_at_placed", "theo_at_fill",
            "theo_drift_signed", "book_churn_2s", "has_cancel_event"]
    for b in sorted(stale["bucket"].unique()):
        print(f"\n-- {b} --")
        sub = stale[stale["bucket"] == b][cols].head(3)
        with pd.option_context("display.max_columns", None,
                               "display.width", 200,
                               "display.float_format", "{:.4f}".format):
            print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
