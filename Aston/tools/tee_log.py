"""Tee stdout/stderr to the terminal AND a daily-rotated log file.

Usage (top of main(), before anything prints):

    from tools.tee_log import install
    install("aston", Path(__file__).parents[1] / "logs/Aston")

Writes logs/Aston/{prefix}-{YYMONDD}_{HHMMSS}.log where the timestamp is
when this file was opened — app launch, or the UTC-midnight rotation on
multi-day runs.  Each restart therefore gets its own file.  Logs older
than retain_days are deleted on each rotation and at startup.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class TeeLog:
    def __init__(self, stream, log_dir: Path, prefix: str, retain_days: int):
        self.stream = stream
        self.log_dir = Path(log_dir)
        self.prefix = prefix
        self.retain_days = retain_days
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._day: str | None = None
        self._fh = None

    def _rotate(self, day: str):
        if self._fh:
            self._fh.close()
        stamp = datetime.now(timezone.utc).strftime("%H%M%S")
        self._fh = open(self.log_dir / f"{self.prefix}-{day}_{stamp}.log",
                        "a", buffering=1)
        self._day = day
        cutoff = time.time() - self.retain_days * 86400
        for f in self.log_dir.glob(f"{self.prefix}-*.log"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass

    def write(self, text):
        self.stream.write(text)
        day = datetime.now(timezone.utc).strftime("%y%b%d").upper()
        if day != self._day:
            self._rotate(day)
        try:
            self._fh.write(text)
        except OSError:
            pass  # disk issues must never take down the trading process

    def flush(self):
        self.stream.flush()
        if self._fh:
            try:
                self._fh.flush()
            except OSError:
                pass

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()


def install(prefix: str, log_dir, retain_days: int = 2):
    """Route stdout+stderr through TeeLog. Call once, before printing."""
    sys.stdout = TeeLog(sys.stdout, log_dir, prefix, retain_days)
    sys.stderr = TeeLog(sys.stderr, log_dir, prefix, retain_days)
