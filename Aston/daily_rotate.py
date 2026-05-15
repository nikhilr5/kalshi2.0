"""Daily rotation of recorder DBs to S3.

Finds every `KX*15M-*.db` file in analysis/backtesting/data/ whose
mtime is older than 24 hours, uploads it (plus its `-wal` / `-shm`
sidecars if present) to s3://kalshibtc/archive/, and removes the local
copies on successful upload.

Designed to be run from launchd once per day.  Idempotent — re-runs
upload only files that still exist locally, so a missed schedule day
just catches up the next time.

Usage:
    python3 daily_rotate.py             # do it
    python3 daily_rotate.py --dry-run   # log what would happen
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

DATA_DIR = (Path(__file__).resolve().parent.parent
            / "analysis" / "backtesting" / "data")
S3_PREFIX = "s3://kalshibtc/archive"
AGE_THRESHOLD_S = 24 * 3600
DB_GLOB = "KX*15M-*.db"


def aws_cp(local: Path, dry_run: bool) -> bool:
    """Upload one file. Returns True on success.  Streams aws CLI
    output to stdout so the launchd log captures progress."""
    remote = f"{S3_PREFIX}/{local.name}"
    if dry_run:
        print(f"[dry-run] would upload {local} → {remote}")
        return True
    cmd = ["aws", "s3", "cp", str(local), remote, "--only-show-errors"]
    print(f"[up] {local.name}")
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print("[ERR] aws CLI not found on PATH — skipping rotate")
        return False
    if r.returncode != 0:
        print(f"[ERR] upload failed for {local.name}: "
              f"{r.stderr.strip() or r.stdout.strip()}")
        return False
    # Verify the object exists on S3 before trusting the upload.
    verify = subprocess.run(
        ["aws", "s3", "ls", remote], capture_output=True, text=True)
    if verify.returncode != 0 or not verify.stdout.strip():
        print(f"[ERR] post-upload verify failed for {local.name}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not DATA_DIR.exists():
        print(f"[skip] {DATA_DIR} does not exist — nothing to rotate")
        return 0

    now = time.time()
    cutoff = now - AGE_THRESHOLD_S
    candidates = sorted(p for p in DATA_DIR.glob(DB_GLOB)
                        if p.stat().st_mtime < cutoff)

    if not candidates:
        print("[idle] no DBs older than 24h")
        return 0

    print(f"[rotate] {len(candidates)} DB(s) eligible "
          f"(cutoff = mtime < {time.strftime('%Y-%m-%d %H:%M', time.localtime(cutoff))})")

    failures = 0
    for db in candidates:
        # Bundle the sidecars so the DB stays restorable from S3.  Some
        # may not exist (if the DB has been fully checkpointed) — that's
        # fine, just skip.
        siblings = [db]
        for ext in (".db-wal", ".db-shm"):
            sib = db.with_suffix(ext)
            if sib.exists() and sib.stat().st_mtime < cutoff:
                siblings.append(sib)

        ok = True
        for f in siblings:
            if not aws_cp(f, args.dry_run):
                ok = False
                break
        if not ok:
            failures += 1
            continue

        if not args.dry_run:
            for f in siblings:
                try:
                    f.unlink()
                    print(f"[rm] {f.name}")
                except Exception as e:
                    print(f"[ERR] could not remove {f.name}: {e}")
                    failures += 1

    if failures:
        print(f"[done] {failures} failure(s)")
        return 1
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
