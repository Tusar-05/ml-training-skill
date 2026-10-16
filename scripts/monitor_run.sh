#!/usr/bin/env bash
# Live view of a training run directory (metrics + checkpoints + log + GPU).
# Usage: bash monitor_run.sh <run-dir> [--watch [N]]   (N = refresh seconds, default 5)

set -u

RUN=""
WATCH=0
INTERVAL=5

usage() {
  cat >&2 <<'EOF'
usage: bash monitor_run.sh <run-dir> [--watch [N]]
  one-shot snapshot of a runs/<name> directory; --watch refreshes every N seconds
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --watch)
      WATCH=1; shift
      if [ $# -gt 0 ] && printf '%s' "$1" | grep -Eq '^[0-9]+$'; then
        INTERVAL="$1"; shift
      fi ;;
    -h|--help) usage; exit 0 ;;
    -*)        echo "unknown option: $1" >&2; usage; exit 1 ;;
    *)         [ -z "$RUN" ] && RUN="$1"; shift ;;
  esac
done

if [ -z "$RUN" ]; then
  usage
  exit 1
fi
if [ ! -d "$RUN" ]; then
  echo "not a directory: $RUN" >&2
  exit 1
fi

snapshot() {
  echo "=== $(date '+%Y-%m-%d %H:%M:%S')  run: $RUN ==="

  if [ -f "$RUN/metrics.csv" ]; then
    rows=$(($(wc -l < "$RUN/metrics.csv") - 1))
    echo "-- metrics.csv ($rows rows), last 3:"
    tail -3 "$RUN/metrics.csv" | sed 's/^/   /'
  else
    echo "-- metrics.csv: missing"
  fi

  if [ -d "$RUN/checkpoints" ]; then
    echo "-- checkpoints:"
    if ls "$RUN/checkpoints"/* >/dev/null 2>&1; then
      ls -l --time-style=+%H:%M:%S "$RUN/checkpoints" | tail -n +2 | sed 's/^/   /'
    else
      echo "   (empty)"
    fi
  else
    echo "-- checkpoints/: missing"
  fi

  if [ -f "$RUN/logs/train.log" ]; then
    echo "-- log tail:"
    tail -3 "$RUN/logs/train.log" | sed 's/^/   /'
  fi

  if [ -f "$RUN/logs/pid.txt" ]; then
    pid=$(awk '{print $2}' "$RUN/logs/pid.txt" 2>/dev/null | head -1)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      et=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
      echo "-- pid $pid: ALIVE (elapsed $et)"
    else
      echo "-- pid ${pid:-<unreadable>}: DEAD (run finished or crashed; check log tail)"
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1 \
     && nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader >/dev/null 2>&1; then
    echo "-- gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu \
      --format=csv,noheader | head -1)"
  else
    echo "-- gpu: nvidia-smi unavailable (CPU-only run)"
  fi
}

if [ "$WATCH" -eq 1 ]; then
  while true; do
    clear
    snapshot
    sleep "$INTERVAL"
  done
else
  snapshot
fi
