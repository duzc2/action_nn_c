#!/usr/bin/env bash
set -e

# Unified demo runner for action_nn_c
# Usage: bash scripts/run_demo.sh <demo> [phase]
#   demo:  move, sevenseg, target, transformer, mnist, mnist_cnn,
#          nested_nav, road_graph_nav, cnn_rnn_react, hybrid_route
#   phase: generate, train, infer, all (default: all)

DEMO=${1:-}
PHASE=${2:-all}

if [ -z "$DEMO" ]; then
    echo "Usage: bash scripts/run_demo.sh <demo> [phase]"
    echo "  demo:  move, sevenseg, target, transformer, mnist, mnist_cnn,"
    echo "         nested_nav, road_graph_nav, cnn_rnn_react, hybrid_route"
    echo "  phase: generate, train, infer, all (default: all)"
    exit 1
fi

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) EXE=".exe" ;;
    *)                    EXE="" ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Debug}"

if [ ! -d "$ROOT/demo/$DEMO" ]; then
    echo "Error: demo '$DEMO' not found at $ROOT/demo/$DEMO"
    exit 1
fi

build_and_run() {
    local demo=$1 phase=$2
    local build_dir="$ROOT/build/demo/$demo/$phase"
    local exe_path="$build_dir/$BUILD_TYPE/${demo}_${phase}${EXE}"

    echo "[$demo] configure + build $phase"
    cmake -S "$ROOT/demo/$demo/$phase" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build "$build_dir" --config "$BUILD_TYPE"

    echo "[$demo] run $phase"
    if [ ! -f "$exe_path" ]; then
        # Fallback: try without the config subdirectory (Ninja single-config)
        exe_path="$build_dir/${demo}_${phase}${EXE}"
    fi
    "$exe_path"
}

if [ "$PHASE" = "all" ]; then
    build_and_run "$DEMO" generate
    build_and_run "$DEMO" train
    build_and_run "$DEMO" infer
else
    build_and_run "$DEMO" "$PHASE"
fi

echo "[$DEMO] demo completed successfully"
