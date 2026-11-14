#!/bin/bash
# ppl_ladder.sh - perplexity ladder with the protocol of PROTOCOL.md.
#
#   ./ppl_ladder.sh -f wiki.test.raw -c 80 base=base-f16.gguf:20 k2=k2.gguf:99
#
# Each arm is label=path[:ngl]; -ngl defaults to 99. The -ngl distinction matters: a 15 GB f16
# 7B does not fit a 12 GB card and must be run partially offloaded, which is exact (see PROTOCOL.md)
# but must be stated next to the number.
#
# Output (also appended to $LOG): one line per arm, label and the engine's own final estimate.
set -uo pipefail

BIN=${BIN:-llama-perplexity}
FILE=${FILE:-wiki.test.raw}
CHUNKS=${CHUNKS:-80}
CTX=${CTX:-512}
THREADS=${THREADS:-6}
LOG=${LOG:-ppl_ladder.log}

usage() { sed -n '2,9p' "$0"; exit 1; }

while getopts "b:f:c:t:T:l:h" o; do
  case $o in
    b) BIN=$OPTARG ;;
    f) FILE=$OPTARG ;;
    c) CHUNKS=$OPTARG ;;
    t) CTX=$OPTARG ;;
    T) THREADS=$OPTARG ;;
    l) LOG=$OPTARG ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || usage
[ -f "$FILE" ] || { echo "no such text file: $FILE" >&2; exit 1; }

echo "=== $(date +%F_%H:%M) $BIN | $FILE | ctx $CTX | chunks $CHUNKS" | tee -a "$LOG"
for arm in "$@"; do
  label=${arm%%=*}
  rest=${arm#*=}
  model=${rest%:*}
  ngl=99
  [ "$model" != "$rest" ] && ngl=${rest##*:}
  [ -f "$model" ] || { echo "  $label: missing $model" | tee -a "$LOG"; continue; }
  printf "%-18s " "$label" | tee -a "$LOG"
  timeout 21600 "$BIN" -m "$model" -f "$FILE" -c "$CTX" --chunks "$CHUNKS" \
      -ngl "$ngl" -t "$THREADS" 2>&1 \
    | grep -oE "n_ctx=[0-9]+ = [0-9.]+ \+/- [0-9.]+" | tail -1 | tee -a "$LOG"
done
echo "=== done $(date +%F_%H:%M)" | tee -a "$LOG"
