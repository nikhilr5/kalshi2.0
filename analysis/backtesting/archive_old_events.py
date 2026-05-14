"""
Archive event DBs that closed more than N days ago to S3, then delete locally.

Event tickers look like KXBTCD-26MAY0517 → year/month/day/hour.  Anything
whose close_time is more than --days-old days in the past is uploaded to
the configured S3 path and removed from disk.

Dry run by default — pass --apply to actually upload + delete.

Usage:
    # Set bucket via env or arg
    export KALSHI_ARCHIVE_BUCKET=my-kalshi-archive
    python archive_old_events.py
    python archive_old_events.py --apply
    python archive_old_events.py --bucket my-bucket --days-old 2 --apply
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Pattern: KXBTCD-26MAY0517  →  year=2026, month=5, day=5, hour=17
EVENT_RE = re.compile(r"^KXBTCD-(\d{2})([A-Z]{3})(\d{2})(\d{2})$")


def parse_event_close(event_ticker: str):
    """Return UTC close datetime, or None if ticker doesn't parse."""
    m = EVENT_RE.match(event_ticker)
    if not m:
        return None
    yy, mon, dd, hh = m.groups()
    month = MONTH_MAP.get(mon)
    if not month:
        return None
    try:
        return datetime(2000 + int(yy), month, int(dd), int(hh), 0, 0,
                        tzinfo=timezone.utc)
    except Exception:
        return None


def s3_key_for(event_ticker: str, prefix: str) -> str:
    """Bucket-relative key — yyyy/mm/<event_ticker>.db for nice browsing."""
    close = parse_event_close(event_ticker)
    if close is None:
        return f"{prefix}misc/{event_ticker}.db"
    return f"{prefix}{close:%Y/%m}/{event_ticker}.db"


def human(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=os.environ.get("KALSHI_ARCHIVE_BUCKET", ""),
                   help="S3 bucket name (or set KALSHI_ARCHIVE_BUCKET).")
    p.add_argument("--prefix", default="kalshi-events/",
                   help="Key prefix inside the bucket. Trailing / required.")
    p.add_argument("--days-old", type=float, default=2.0,
                   help="Archive events whose close was more than this many days ago.")
    p.add_argument("--apply", action="store_true",
                   help="Actually upload + delete (default is dry-run).")
    args = p.parse_args()

    if not args.bucket:
        print("ERROR: pass --bucket or set KALSHI_ARCHIVE_BUCKET", file=sys.stderr)
        sys.exit(1)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=args.days_old)
    print(f"Archive cutoff: events with close < {cutoff.isoformat()} UTC")
    print(f"Target: s3://{args.bucket}/{args.prefix}")
    print(f"Mode:   {'APPLY (upload + delete)' if args.apply else 'DRY-RUN'}")
    print()

    candidates = []
    for path in sorted(DATA_DIR.glob("*.db")):
        if path.name == "recorder.db":
            continue
        event_ticker = path.stem
        close = parse_event_close(event_ticker)
        if close is None:
            print(f"  [skip] {path.name}: can't parse close from ticker")
            continue
        if close >= cutoff:
            print(f"  [keep] {path.name}: closes {close.isoformat()} (within window)")
            continue
        candidates.append((path, event_ticker, close))

    if not candidates:
        print("\nNothing to archive.")
        return

    total_bytes = sum(p.stat().st_size for p, _, _ in candidates)
    print(f"\n{len(candidates)} event(s) to archive ({human(total_bytes)} total):")
    for path, et, close in candidates:
        size = human(path.stat().st_size)
        key = s3_key_for(et, args.prefix)
        print(f"  {path.name}  ({size})  →  s3://{args.bucket}/{key}")

    if not args.apply:
        print("\n(dry-run — re-run with --apply to actually do it)")
        return

    print("\nUploading + deleting...")
    failures = 0
    freed = 0
    for path, et, close in candidates:
        key = s3_key_for(et, args.prefix)
        s3_uri = f"s3://{args.bucket}/{key}"
        print(f"  → {path.name}: uploading {human(path.stat().st_size)}...", flush=True)
        result = subprocess.run(
            ["aws", "s3", "cp", str(path), s3_uri,
             "--storage-class", "STANDARD_IA"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"    UPLOAD FAILED: {result.stderr.strip()}")
            failures += 1
            continue
        # Verify the upload exists on S3 before deleting
        verify = subprocess.run(
            ["aws", "s3api", "head-object",
             "--bucket", args.bucket, "--key", key],
            capture_output=True, text=True,
        )
        if verify.returncode != 0:
            print(f"    VERIFY FAILED — leaving local file intact")
            failures += 1
            continue
        size = path.stat().st_size
        path.unlink()
        freed += size
        print(f"    deleted local. freed {human(size)}.")

    print(f"\nDone. Freed {human(freed)} locally; {failures} failure(s).")


if __name__ == "__main__":
    main()
