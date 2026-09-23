#!/usr/bin/env bash
# Runs the simplification benchmark once over all BiGG models, in the order the
# benchmark itself uses, and stores everything under experiments/:
#
#   experiments/<timestamp>/
#       meta.txt              machine, commit and settings of the run
#       benchmark.log         stdout and stderr together, in order
#       summary.tsv           one line per model, parsed from the log
#       simplified_bigg/      the exported simplified polytopes
#   experiments/latest -> <timestamp>
#
# Usage:
#   ./run_benchmarks.sh
#
# Settings (environment variables):
#   BUILD_DIR   the CMake build directory        (default /project/test/build)
#   OUT_ROOT    where the experiments are stored  (default $BUILD_DIR/experiments)
#   NO_BUILD=1  skips rebuilding the benchmark

set -uo pipefail

BUILD_DIR="${BUILD_DIR:-/project/test/build}"
BIN_NAME="benchmark_simplification"
BIN="$BUILD_DIR/$BIN_NAME"
OUT_ROOT="${OUT_ROOT:-$BUILD_DIR/experiments}"

# Builds the benchmark so the run always matches the current sources.
if [[ "${NO_BUILD:-0}" != "1" ]]; then
    echo "building $BIN_NAME ..."
    if ! make -C "$BUILD_DIR" "$BIN_NAME" >/dev/null; then
        echo "error: build failed" >&2
        exit 1
    fi
fi

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not found or not executable" >&2
    exit 1
fi

# Prepares the run directory.
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUT_ROOT/$STAMP"
LOG="$RUN_DIR/benchmark.log"
SUMMARY="$RUN_DIR/summary.tsv"
mkdir -p "$RUN_DIR"
ln -sfn "$STAMP" "$OUT_ROOT/latest"

{
    echo "date:       $(date -Iseconds)"
    echo "host:       $(hostname)"
    echo "cpu:        $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
    echo "cores:      $(nproc)"
    echo "commit:     $(git -C "$BUILD_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "dirty:      $(git -C "$BUILD_DIR" status --porcelain 2>/dev/null | grep -q . && echo yes || echo no)"
    echo "binary:     $BIN"
} > "$RUN_DIR/meta.txt"

echo "results in $RUN_DIR"
echo

# Runs from inside the run directory, so the exports land in simplified_bigg/ there.
# stdout and stderr go to one log so the timings stay under their model header, and
# stdbuf keeps both line buffered so an interrupted run still leaves a complete log.
start=$(date +%s)
( cd "$RUN_DIR" && stdbuf -oL -eL "$BIN" ) 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
end=$(date +%s)

echo "exit code:  $rc" >> "$RUN_DIR/meta.txt"
echo "wall time:  $((end-start))s" >> "$RUN_DIR/meta.txt"

# Builds the summary: the model name comes from the " --- NAME (...) ---" header,
# the numbers from the result row that follows it.
printf "model\tbounds\tdims\tstatus\ttime_s\tskinniness\n" > "$SUMMARY"
awk '
    $1 == "---" { if (name != "" && !done) print name "\tNA\tNA\tNO_RESULT\tNA\tNA";
                  name = $2; done = 0; next }
    $1 == "clarkson" && name != "" {
        t = $5; sub(/s$/, "", t);
        print name "\t" $2 "\t" $3 "\t" $4 "\t" t "\t" $6;
        done = 1
    }
    END { if (name != "" && !done) print name "\tNA\tNA\tNO_RESULT\tNA\tNA" }
' "$LOG" >> "$SUMMARY"

echo
echo "summary ($SUMMARY):"
column -t -s $'\t' "$SUMMARY"