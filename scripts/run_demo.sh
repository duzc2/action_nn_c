#!/usr/bin/env bash
set -e

# Unified demo runner for action_nn_c
# Usage: bash scripts/run_demo.sh <demo> [phase]
#   demo:  move, sevenseg, target, transformer, mnist, mnist_cnn,
#          nested_nav, road_graph_nav, cnn_rnn_react, hybrid_route, weather
#   phase: generate, train, infer, all (default: all)

DEMO=${1:-}
PHASE=${2:-all}

if [ -z "$DEMO" ]; then
    echo "Usage: bash scripts/run_demo.sh <demo> [phase]"
    echo "  demo:  move, sevenseg, target, transformer, mnist, mnist_cnn,"
    echo "         nested_nav, road_graph_nav, cnn_rnn_react, hybrid_route, weather"
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

# Shared generated-code directory (generate phase writes here,
# train and infer read from here).
data_dir() {
    echo "$ROOT/build/demo/$1/data"
}

build_and_run() {
    local demo=$1 phase=$2
    local build_dir="$ROOT/build/demo/$demo/$phase"
    local exe_path="$build_dir/$BUILD_TYPE/${demo}_${phase}${EXE}"
    local extra_args=""

    # Point train/infer to the shared generated-code directory.
    if [ "$phase" != "generate" ]; then
        extra_args="-DACTION_C_GENERATED_DIR=$(data_dir "$demo")"
    fi

    echo "[$demo] configure + build $phase"
    # shellcheck disable=SC2086
    cmake -S "$ROOT/demo/$demo/$phase" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" $extra_args
    cmake --build "$build_dir" --config "$BUILD_TYPE"

    echo "[$demo] run $phase"
    if [ ! -f "$exe_path" ]; then
        # Fallback: try without the config subdirectory (Ninja single-config)
        exe_path="$build_dir/${demo}_${phase}${EXE}"
    fi

    # Create the local "data" directory that the executable expects
    # (it uses ../data relative to the exe directory after CWD change).
    local exe_data_dir="$build_dir/$BUILD_TYPE/../data"
    mkdir -p "$exe_data_dir"

    # Before infer, copy weights from shared dir to infer's local data dir
    if [ "$phase" = "infer" ]; then
        local shared_data="$(data_dir "$demo")"
        for wf in "$shared_data"/weights*.bin; do
            if [ -f "$wf" ]; then
                cp "$wf" "$exe_data_dir/"
            fi
        done
    fi

    "$exe_path"

    # After generate, copy generated files to the shared data directory
    if [ "$phase" = "generate" ]; then
        local shared_data="$(data_dir "$demo")"
        # The generate executable writes to "data/" relative to its location
        # (which is build/demo/<name>/generate/<config>/ after CWD change).
        # Also try build/demo/<name>/generate/data/ as fallback.
        local src_dir=""
        if [ -d "$exe_data_dir" ] && [ "$(ls -A "$exe_data_dir" 2>/dev/null)" ]; then
            src_dir="$exe_data_dir"
        elif [ -d "$build_dir/data" ] && [ "$(ls -A "$build_dir/data" 2>/dev/null)" ]; then
            src_dir="$build_dir/data"
        fi
        if [ -n "$src_dir" ]; then
            echo "[$demo] copying generated files to shared data dir"
            rm -rf "$shared_data" 2>/dev/null || true
            mkdir -p "$shared_data"
            cp -r "$src_dir"/* "$shared_data"/
        fi
    fi

    # After train, copy weights to the shared data directory
    if [ "$phase" = "train" ]; then
        local shared_data="$(data_dir "$demo")"
        mkdir -p "$shared_data"
        for wf in "$exe_data_dir"/weights*.bin; do
            if [ -f "$wf" ]; then
                cp "$wf" "$shared_data/"
            fi
        done
        # Also try alternate locations
        for wf in "$build_dir/data"/weights*.bin; do
            if [ -f "$wf" ]; then
                cp "$wf" "$shared_data/"
            fi
        done
    fi
}

if [ "$PHASE" = "all" ]; then
    build_and_run "$DEMO" generate
    build_and_run "$DEMO" train
    build_and_run "$DEMO" infer
else
    build_and_run "$DEMO" "$PHASE"
fi

echo "[$DEMO] demo completed successfully"
