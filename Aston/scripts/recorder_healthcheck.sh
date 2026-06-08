#!/bin/bash
# Recorder health check. Runs every 5 minutes via LaunchAgent.
#
# Logic: find today's KXETH15M recorder DB. Check the most recent
# row's timestamp across fills, kalshi_book, spot_ticks. If the
# newest row is older than 5 minutes, fire a macOS banner.
#
# A "stale" recorder = the process might still be alive, but it's
# not writing data. That's what we care about.

set -u
DATA_DIR="/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting/data"
LOG_FILE="$HOME/Library/Logs/aston-recorder-healthcheck.log"
THRESHOLD_S=300   # 5 minutes
SERIES_PREFIX="KXETH15M"

# Map current UTC date to YYMONDD
month_abbr() {
  case "$1" in
    01) echo JAN;; 02) echo FEB;; 03) echo MAR;; 04) echo APR;;
    05) echo MAY;; 06) echo JUN;; 07) echo JUL;; 08) echo AUG;;
    09) echo SEP;; 10) echo OCT;; 11) echo NOV;; 12) echo DEC;;
  esac
}

now_epoch=$(date -u +%s)
yy=$(date -u +%y)
mm=$(date -u +%m)
dd=$(date -u +%d)
suffix="${yy}$(month_abbr "$mm")${dd}"
db_path="$DATA_DIR/${SERIES_PREFIX}-${suffix}.db"

EMAIL_SCRIPT="/Users/nikhilr5/Desktop/Kalshi2.0/Aston/scripts/send_alert_email.py"

notify() {
  local title="$1"
  local body="$2"
  osascript -e "display notification \"$body\" with title \"$title\""
  /usr/bin/python3 "$EMAIL_SCRIPT" "$title: $body" "$body" >> "$LOG_FILE" 2>&1 || true
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

if [ ! -f "$db_path" ]; then
  log "ALERT: today's DB missing: $db_path"
  notify "Aston recorder" "Today's DB missing ($suffix)"
  exit 0
fi

# Newest ts across the three tables. SQLite stores ts as ISO 8601;
# convert to epoch via `date -j -f`. Use stderr suppression so a
# missing table doesn't blow up the query.
newest=$(sqlite3 "$db_path" "
  SELECT MAX(ts) FROM (
    SELECT ts FROM fills
    UNION ALL SELECT ts FROM kalshi_book
    UNION ALL SELECT ts FROM spot_ticks
  );
" 2>/dev/null)

if [ -z "$newest" ] || [ "$newest" = "" ]; then
  log "ALERT: DB has no rows in fills/kalshi_book/spot_ticks: $db_path"
  notify "Aston recorder" "DB exists but no rows yet ($suffix)"
  exit 0
fi

# ts looks like 2026-06-08T12:34:56.789+00:00. Strip subseconds and
# normalize tz to feed `date -j -f`.
ts_clean=$(echo "$newest" | sed -E 's/\.[0-9]+//' | sed -E 's/\+00:00$/Z/')
newest_epoch=$(date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$ts_clean" +%s 2>/dev/null)

if [ -z "$newest_epoch" ]; then
  log "WARN: could not parse newest ts: $newest"
  exit 0
fi

age=$((now_epoch - newest_epoch))
log "newest_ts=$newest age_s=$age db=$(basename "$db_path")"

if [ "$age" -gt "$THRESHOLD_S" ]; then
  mins=$((age / 60))
  notify "Aston recorder" "No writes in ${mins}m (last: $ts_clean)"
  log "ALERT: stale ${age}s threshold ${THRESHOLD_S}s"
fi
