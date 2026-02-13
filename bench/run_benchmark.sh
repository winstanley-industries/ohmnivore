#!/usr/bin/env bash
set -e

# CMOS Inverter Chain Benchmark
# Compares ohmnivore (GPU and CPU) against ngspice across different chain sizes.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CIRCUIT_DIR="$SCRIPT_DIR/circuits"
BINARY="$PROJECT_DIR/target/release/ohmnivore"
GENERATOR="$SCRIPT_DIR/generate_chain.py"

SIZES=(10 50 100 200)

# --- Preflight checks ---

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found in PATH" >&2
    exit 1
fi

if ! command -v ngspice &>/dev/null; then
    echo "WARNING: ngspice not found in PATH; ngspice columns will show N/A" >&2
    HAS_NGSPICE=0
else
    HAS_NGSPICE=1
fi

# --- Build ---

echo "Building ohmnivore in release mode..."
(cd "$PROJECT_DIR" && cargo build --release --quiet)

if [ ! -x "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY" >&2
    exit 1
fi

# --- Generate netlists ---

echo "Generating netlists..."
mkdir -p "$CIRCUIT_DIR"
for n in "${SIZES[@]}"; do
    python3 "$GENERATOR" "$n"
done

# --- Timing helper ---
# Runs a command, captures wall-clock seconds to 3 decimal places.
# Result is stored in $ELAPSED.
time_cmd() {
    local start end
    start=$(python3 -c 'import time; print(f"{time.time():.6f}")')
    "$@" >/dev/null 2>&1 || true
    end=$(python3 -c 'import time; print(f"{time.time():.6f}")')
    ELAPSED=$(python3 -c "print(f'{$end - $start:.3f}')")
}

# --- Create ngspice-compatible netlist ---
# ohmnivore uses ".DC" for DC op, ngspice uses ".op" plus a .print line.
make_ngspice_netlist() {
    local src="$1"
    local tmp="$2"
    local analysis="$3"

    if [ "$analysis" = "dc" ]; then
        # Replace .DC with .op, add .print dc v(out_1) before .END
        sed -e 's/^\.DC$/.op/' \
            -e 's/^\.END/.print dc v(out_1)\n.END/' \
            "$src" > "$tmp"
    else
        # TRAN: add .print tran v(out_1) before .END
        sed -e 's/^\.END/.print tran v(out_1)\n.END/' \
            "$src" > "$tmp"
    fi
}

# --- Run benchmarks ---

echo "Running benchmarks..."
echo ""

# Arrays to store results: "size|analysis|gpu|cpu|ngspice"
declare -a RESULTS

for n in "${SIZES[@]}"; do
    for analysis in dc tran; do
        netlist="$CIRCUIT_DIR/inverter_chain_${n}_${analysis}.spice"
        if [ ! -f "$netlist" ]; then
            echo "WARNING: $netlist not found, skipping" >&2
            continue
        fi

        # GPU
        time_cmd "$BINARY" "$netlist"
        gpu_time="$ELAPSED"

        # CPU
        time_cmd "$BINARY" --cpu "$netlist"
        cpu_time="$ELAPSED"

        # ngspice
        if [ "$HAS_NGSPICE" -eq 1 ]; then
            tmp_netlist=$(mktemp /tmp/ohmnivore_bench_XXXXXX.spice)
            make_ngspice_netlist "$netlist" "$tmp_netlist" "$analysis"
            time_cmd ngspice -b "$tmp_netlist"
            ngspice_time="$ELAPSED"
            rm -f "$tmp_netlist"
        else
            ngspice_time="N/A"
        fi

        RESULTS+=("${n}|${analysis^^}|${gpu_time}|${cpu_time}|${ngspice_time}")
        echo "  ${n} stages / ${analysis^^}: GPU=${gpu_time}s  CPU=${cpu_time}s  ngspice=${ngspice_time}s"
    done
done

# --- Print results table ---

echo ""
echo "CMOS Inverter Chain Benchmark"
echo "============================="
echo ""
printf "%-6s | %-8s | %9s | %9s | %11s\n" "Size" "Analysis" "GPU (s)" "CPU (s)" "ngspice (s)"
printf "%-6s-+-%-8s-+-%-9s-+-%-9s-+-%-11s\n" "------" "--------" "---------" "---------" "-----------"

for row in "${RESULTS[@]}"; do
    IFS='|' read -r size analysis gpu cpu ngspice <<< "$row"
    printf "%5s  | %-8s | %9s | %9s | %11s\n" "$size" "$analysis" "$gpu" "$cpu" "$ngspice"
done

echo ""
echo "Done."
