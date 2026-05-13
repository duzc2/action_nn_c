#!/usr/bin/env bash
set -e

# Weather prediction demo runner
# Usage: bash run_demo.sh [phase]
#   phase: generate, train, infer, all (default: all)

PHASE=${1:-all}

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) EXE=".exe" ;;
    *)                    EXE="" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Debug}"

data_dir() {
    echo "$ROOT/build/demo/weather/data"
}

build_and_run() {
    local phase=$1
    local build_dir="$ROOT/build/demo/weather/$phase"
    local exe_path="$build_dir/$BUILD_TYPE/weather_${phase}${EXE}"
    local extra_args=""

    if [ "$phase" != "generate" ]; then
        extra_args="-DACTION_C_GENERATED_DIR=$(data_dir)"
    fi

    echo "[weather] configure + build $phase"
    cmake -S "$SCRIPT_DIR/$phase" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" $extra_args
    cmake --build "$build_dir" --config "$BUILD_TYPE"

    echo "[weather] run $phase"
    if [ ! -f "$exe_path" ]; then
        exe_path="$build_dir/weather_${phase}${EXE}"
    fi

    local exe_data_dir="$build_dir/$BUILD_TYPE/../data"
    mkdir -p "$exe_data_dir"

    # Before infer, copy all weight files from shared data dir
    if [ "$phase" = "infer" ]; then
        local shared_data="$(data_dir)"
        for wf in "$shared_data"/weights*.bin; do
            if [ -f "$wf" ]; then
                cp "$wf" "$exe_data_dir/"
            fi
        done
    fi

    "$exe_path"

    # After generate, copy generated files to shared data directory
    if [ "$phase" = "generate" ]; then
        local shared_data="$(data_dir)"
        local src_dir=""
        if [ -d "$exe_data_dir" ] && [ "$(ls -A "$exe_data_dir" 2>/dev/null)" ]; then
            src_dir="$exe_data_dir"
        elif [ -d "$build_dir/data" ] && [ "$(ls -A "$build_dir/data" 2>/dev/null)" ]; then
            src_dir="$build_dir/data"
        fi
        if [ -n "$src_dir" ]; then
            echo "[weather] copying generated files to shared data dir"
            rm -rf "$shared_data" 2>/dev/null || true
            mkdir -p "$shared_data"
            cp -r "$src_dir"/* "$shared_data"/
        fi
    fi

    # After train, copy all weight files to shared data directory
    if [ "$phase" = "train" ]; then
        local shared_data="$(data_dir)"
        mkdir -p "$shared_data"
        for wf in "$exe_data_dir"/weights*.bin; do
            if [ -f "$wf" ]; then
                cp "$wf" "$shared_data/"
            fi
        done
        for wf in "$build_dir/data"/weights*.bin; do
            if [ -f "$wf" ]; then
                cp "$wf" "$shared_data/"
            fi
        done
    fi
}

if [ "$PHASE" = "all" ]; then
    build_and_run generate
    build_and_run train
    build_and_run infer
else
    build_and_run "$PHASE"
fi

echo "[weather] demo completed successfully"
