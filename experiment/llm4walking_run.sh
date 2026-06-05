#!/bin/bash
# LLM4Walking - Philemon-TSH Graph Walking Experiment Pipeline
# Upstream: https://github.com/dylanyunlon/philemon-TSH
# Strategy: mv upstream skeleton + 20% algorithmic modification + breakpoint debug
#
# Mirrors the LLM4CardGame pipeline structure:
#   setup → clone → build → generate → run → eval
#
# Dependencies: g++ (>=9), cmake (>=3.20), numactl (optional), conda (optional)

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   LLM4Walking — Philemon-TSH Experiment Pipeline         ║"
echo "║   Graph Walking: BFS, SSSP, PageRank, WCC, TC            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
SRC_DIR="${SRC_DIR:-$PROJECT_DIR/src}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
BIN_DIR="${BIN_DIR:-$PROJECT_DIR/bin}"

# Graph parameters (can override via env)
NUM_VERTICES="${NUM_VERTICES:-10000}"
NUM_EDGES="${NUM_EDGES:-50000}"
SEED="${SEED:-42}"
DEBUG_LEVEL="${DEBUG_LEVEL:-2}"     # 0=silent 1=summary 2=per-step 3=per-edge
NUM_THREADS="${NUM_THREADS:-1}"
PR_ITERS="${PR_ITERS:-10}"
DAMPING="${DAMPING:-0.85}"

# Algorithms to run
ALGORITHMS=("bfs" "sssp" "pagerank" "wcc" "tc" "microbench")

# Real datasets (SNAP format)
DATASETS_SMALL=("email-Enron" "wiki-Vote" "p2p-Gnutella31")
DATASETS_LARGE=("soc-LiveJournal1" "com-Orkut" "roadNet-CA")

# Conda env
CONDA_ENV_NAME="llm4walking"

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

timestamp() {
    date +%Y%m%d_%H%M%S
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"

    # CPU
    echo "CPU Info:"
    echo "  Cores:    $(nproc)"
    echo "  Model:    $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'N/A')"
    echo "  Arch:     $(uname -m)"
    echo ""

    # Memory
    echo "Memory:"
    free -h 2>/dev/null | head -2 || echo "  N/A"
    echo ""

    # GPU (optional)
    echo "GPU Status:"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
                   --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
        echo ""
        echo "CUDA Version:"
        nvidia-smi | grep "CUDA Version" | awk '{print "  "$9}' 2>/dev/null || echo "  N/A"
    else
        echo "  No nvidia-smi (CPU-only mode, OK for graph walking)"
    fi
    echo ""

    # NUMA
    echo "NUMA Topology:"
    if command -v numactl &>/dev/null; then
        numactl --hardware 2>/dev/null | head -5 || echo "  numactl failed"
    else
        echo "  numactl not available (will run without NUMA binding)"
    fi
    echo ""

    # Compiler
    echo "Compiler:"
    if check_command g++; then
        g++ --version | head -1
    else
        echo "  FATAL: g++ not found. Install via: apt install g++ or conda install gxx_linux-64"
        exit 1
    fi
    echo ""

    # CMake (optional, for full build)
    echo "CMake:"
    cmake --version 2>/dev/null | head -1 || echo "  Not found (optional, we use direct g++ compilation)"
    echo ""

    echo "✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Build Environment"

    # Check if conda is available
    if command -v conda &>/dev/null; then
        echo "Conda detected. Setting up environment: $CONDA_ENV_NAME"

        if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
            echo "Environment '${CONDA_ENV_NAME}' already exists."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
        else
            echo "Creating conda environment..."
            conda create -n ${CONDA_ENV_NAME} python=3.10 -y
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}

            # Install Python deps for data analysis / plotting
            pip install --upgrade pip
            pip install numpy scipy pandas matplotlib seaborn
            pip install networkx  # For graph data generation & validation
        fi
    else
        echo "No conda found. Using system compiler directly."
    fi

    # Ensure build tools
    echo "Checking build tools..."
    check_command g++ || { echo "FATAL: g++ required"; exit 1; }

    # Create directories
    mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$BIN_DIR" "$PROJECT_DIR/configs"

    echo "✓ Environment ready"
}

# ===========================================
# Clone Upstream Repository
# ===========================================

clone_upstream() {
    print_step "Cloning Upstream Philemon-TSH Repository"

    UPSTREAM_DIR="$PROJECT_DIR/upstream_repo"

    if [ -d "$UPSTREAM_DIR" ]; then
        echo "Upstream repo already exists: $UPSTREAM_DIR"
        cd "$UPSTREAM_DIR"
        git pull 2>/dev/null || echo "  (pull skipped)"
        cd "$SCRIPT_DIR"
    else
        git clone https://github.com/dylanyunlon/philemon-TSH.git "$UPSTREAM_DIR"
    fi

    echo "✓ Upstream at: $UPSTREAM_DIR"
    echo ""
    echo "File inventory (src/):"
    find "$UPSTREAM_DIR/src" -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" | wc -l
    echo " source files"
}

# ===========================================
# Build — Compile the Walking Experiments
# ===========================================

build_experiments() {
    print_step "Building Walking Experiments"

    CXX="${CXX:-g++}"
    CXXFLAGS="-std=c++17 -O2 -pthread -Wall -Wextra -Wno-unused-parameter"
    DEBUG_FLAGS="-DPHILE_DEBUG=1 -DWALKING_DEBUG=1"

    mkdir -p "$BIN_DIR"

    # ── 1. Synthetic experiment (small/medium, all algorithms) ──
    echo "[1/3] Building walking_experiment (synthetic)..."
    $CXX $CXXFLAGS $DEBUG_FLAGS \
        -o "$BIN_DIR/walking_experiment" \
        "$SRC_DIR/walking_experiment.cpp"
    echo "      OK ($(du -h "$BIN_DIR/walking_experiment" | cut -f1))"

    # ── 2. Real-scale experiment (large datasets) ──
    echo "[2/3] Building walking_realscale..."
    $CXX $CXXFLAGS $DEBUG_FLAGS \
        -o "$BIN_DIR/walking_realscale" \
        "$SRC_DIR/walking_realscale.cpp"
    echo "      OK ($(du -h "$BIN_DIR/walking_realscale" | cut -f1))"

    # ── 3. Debug inspector (state dump utility) ──
    echo "[3/4] Building walking_inspector..."
    $CXX $CXXFLAGS $DEBUG_FLAGS \
        -o "$BIN_DIR/walking_inspector" \
        "$SRC_DIR/walking_inspector.cpp"
    echo "      OK ($(du -h "$BIN_DIR/walking_inspector" | cut -f1))"

    # ── 4. GPU Tree Traversal (.cu → nvcc or g++ fallback) ──
    echo "[4/4] Building walking_gpu_tree..."
    if command -v nvcc &>/dev/null; then
        echo "      nvcc detected → real CUDA build"
        NVCC="${NVCC:-nvcc}"
        NVFLAGS="-std=c++17 -O2 -DWALKING_CUDA=1"
        # Detect GPU arch
        GPU_ARCH="${GPU_ARCH:-sm_86}"  # Default A6000; override for H100 (sm_90)
        $NVCC $NVFLAGS -arch=$GPU_ARCH \
            -o "$BIN_DIR/walking_gpu_tree" \
            "$SRC_DIR/walking_gpu_tree.cu"
    else
        echo "      nvcc not found → CPU fallback (g++ -x c++)"
        $CXX $CXXFLAGS $DEBUG_FLAGS -DWALKING_CUDA=0 \
            -x c++ -o "$BIN_DIR/walking_gpu_tree" \
            "$SRC_DIR/walking_gpu_tree.cu"
    fi
    echo "      OK ($(du -h "$BIN_DIR/walking_gpu_tree" | cut -f1))"

    # ── 5. Warp-Cooperative GPU Tree (M080-M082) ──
    echo "[5/5] Building walking_warp_cooperative..."
    if command -v nvcc &>/dev/null; then
        echo "      nvcc detected → real CUDA build"
        NVCC="${NVCC:-nvcc}"
        NVFLAGS="-std=c++17 -O2 -DWALKING_CUDA=1"
        GPU_ARCH="${GPU_ARCH:-sm_86}"
        $NVCC $NVFLAGS -arch=$GPU_ARCH \
            -o "$BIN_DIR/walking_warp_cooperative" \
            "$SRC_DIR/walking_warp_cooperative.cu"
    else
        echo "      nvcc not found → CPU fallback (g++ -x c++)"
        $CXX $CXXFLAGS $DEBUG_FLAGS -DWALKING_CUDA=0 \
            -x c++ -o "$BIN_DIR/walking_warp_cooperative" \
            "$SRC_DIR/walking_warp_cooperative.cu"
    fi
    echo "      OK ($(du -h "$BIN_DIR/walking_warp_cooperative" | cut -f1))"

    # ── 6. TemGraph GPU Temporal Queries (M083-M085) ──
    echo "[6/6] Building walking_temgraph_gpu..."
    if command -v nvcc &>/dev/null; then
        echo "      nvcc detected → real CUDA build"
        NVCC="${NVCC:-nvcc}"
        NVFLAGS="-std=c++17 -O2 -DWALKING_CUDA=1"
        GPU_ARCH="${GPU_ARCH:-sm_86}"
        $NVCC $NVFLAGS -arch=$GPU_ARCH \
            -o "$BIN_DIR/walking_temgraph_gpu" \
            "$SRC_DIR/walking_temgraph_gpu.cu"
    else
        echo "      nvcc not found → CPU fallback (g++ -x c++)"
        $CXX $CXXFLAGS $DEBUG_FLAGS -DWALKING_CUDA=0 \
            -x c++ -o "$BIN_DIR/walking_temgraph_gpu" \
            "$SRC_DIR/walking_temgraph_gpu.cu"
    fi
    echo "      OK ($(du -h "$BIN_DIR/walking_temgraph_gpu" | cut -f1))"

    echo ""
    echo "✓ All binaries in: $BIN_DIR/"
    ls -lh "$BIN_DIR/"
}

# ===========================================
# Generate Synthetic Graph Data
# ===========================================

generate_data() {
    print_step "Generating Graph Data"

    mkdir -p "$DATA_DIR/synthetic" "$DATA_DIR/real"

    # ── Synthetic data ──
    echo "Generating synthetic graph: V=$NUM_VERTICES E=$NUM_EDGES seed=$SEED"
    python3 - <<'PYEOF'
import sys, os, random, json
V = int(os.environ.get("NUM_VERTICES", 10000))
E = int(os.environ.get("NUM_EDGES", 50000))
seed = int(os.environ.get("SEED", 42))
data_dir = os.environ.get("DATA_DIR", "./data")

random.seed(seed)
out_path = os.path.join(data_dir, "synthetic", f"graph_V{V}_E{E}.txt")

with open(out_path, 'w') as f:
    f.write(f"# Synthetic graph: V={V} E={E} seed={seed}\n")
    f.write(f"# Format: source destination weight\n")
    edges = set()
    while len(edges) < E:
        s = random.randint(0, V-1)
        d = random.randint(0, V-1)
        if s != d and (s,d) not in edges:
            edges.add((s,d))
            w = round(1.0 + random.random() * 9.0, 2)
            f.write(f"{s}\t{d}\t{w}\n")

# Metadata
meta = {"vertices": V, "edges": E, "seed": seed, "path": out_path}
with open(os.path.join(data_dir, "synthetic", "metadata.json"), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"[DATA·GEN] Wrote {E} edges → {out_path}")
print(f"[DATA·GEN] Metadata → {os.path.dirname(out_path)}/metadata.json")
PYEOF

    # ── Download real datasets (SNAP) ──
    echo ""
    echo "Downloading real-world datasets (SNAP)..."
    for ds in "email-Enron"; do
        DSFILE="$DATA_DIR/real/${ds}.txt"
        if [ -f "$DSFILE" ]; then
            echo "  $ds: already exists"
        else
            echo "  $ds: downloading..."
            wget -q "https://snap.stanford.edu/data/${ds}.txt.gz" -O "${DSFILE}.gz" 2>/dev/null && \
                gunzip -f "${DSFILE}.gz" && \
                echo "    OK: $(wc -l < "$DSFILE") lines" || \
                echo "    WARN: download failed (will use synthetic data)"
        fi
    done

    echo ""
    echo "✓ Data in: $DATA_DIR/"
    find "$DATA_DIR" -type f | head -20
}

# ===========================================
# Run Synthetic Experiment
# ===========================================

run_synthetic() {
    print_step "Running Synthetic Graph Experiment"

    BIN="$BIN_DIR/walking_experiment"
    if [ ! -f "$BIN" ]; then
        echo "Binary not found, building..."
        build_experiments
    fi

    TIMESTAMP=$(timestamp)
    LOG="$LOG_DIR/synthetic_${TIMESTAMP}.log"
    CFG="$PROJECT_DIR/experiment/walking.cfg"

    echo "Config:  $CFG"
    echo "Log:     $LOG"
    echo "Debug:   level=$DEBUG_LEVEL"
    echo ""

    # Choose run method
    RUN_CMD="$BIN"
    if [ -f "$CFG" ]; then
        RUN_CMD="$BIN $CFG"
    fi

    echo "Command: $RUN_CMD"
    echo ""

    if command -v numactl &>/dev/null; then
        echo "[NUMA] Binding to node 0"
        numactl --cpunodebind=0 --membind=0 $RUN_CMD 2>&1 | tee "$LOG"
    else
        $RUN_CMD 2>&1 | tee "$LOG"
    fi

    echo ""
    echo "=== Post-run Metrics ==="
    grep -E '^\[TIMER·END\]' "$LOG" 2>/dev/null || echo "(no timer data)"
    echo ""
    grep -E '^\[.*·RESULT\]' "$LOG" 2>/dev/null || echo "(no result data)"
    echo ""
    grep -E '^\[TIER·ACCESS\]' "$LOG" 2>/dev/null || echo "(no tier data)"
    echo ""
    grep -E '^\[BP·' "$LOG" 2>/dev/null | wc -l | xargs -I{} echo "Breakpoints hit: {}"
}

# ===========================================
# Run Real-Scale Experiment
# ===========================================

run_realscale() {
    print_step "Running Real-Scale Graph Experiment"

    BIN="$BIN_DIR/walking_realscale"
    if [ ! -f "$BIN" ]; then
        echo "Binary not found, building..."
        build_experiments
    fi

    # Find dataset
    EDGE_FILE="${EDGE_FILE:-}"
    if [ -z "$EDGE_FILE" ]; then
        # Auto-detect: prefer real data, fall back to synthetic
        for ds in "email-Enron" "wiki-Vote" "p2p-Gnutella31"; do
            if [ -f "$DATA_DIR/real/${ds}.txt" ]; then
                EDGE_FILE="$DATA_DIR/real/${ds}.txt"
                break
            fi
        done
        if [ -z "$EDGE_FILE" ]; then
            # Fall back to synthetic
            EDGE_FILE=$(find "$DATA_DIR/synthetic" -name "*.txt" | head -1)
        fi
    fi

    if [ -z "$EDGE_FILE" ] || [ ! -f "$EDGE_FILE" ]; then
        echo "No graph data found. Run 'generate' first."
        return 1
    fi

    TIMESTAMP=$(timestamp)
    LOG="$LOG_DIR/realscale_${TIMESTAMP}.log"

    echo "Dataset: $EDGE_FILE"
    echo "Debug:   $DEBUG_LEVEL"
    echo "Log:     $LOG"
    echo ""

    RUN_CMD="$BIN $EDGE_FILE $DEBUG_LEVEL $PR_ITERS"
    echo "Command: $RUN_CMD"
    echo ""

    if command -v numactl &>/dev/null; then
        echo "[NUMA] Binding to node 0"
        numactl --cpunodebind=0 --membind=0 $RUN_CMD 2>&1 | tee "$LOG"
    else
        $RUN_CMD 2>&1 | tee "$LOG"
    fi

    echo ""
    echo "=== Post-run Analysis ==="
    grep -E '^\[T·END\]' "$LOG" 2>/dev/null || echo "(no timer data)"
    echo ""
    grep -E '^\[.*·RESULT\]' "$LOG" 2>/dev/null || echo "(no results)"
    echo ""
    echo "Breakpoints hit: $(grep -c '^\[BP·' "$LOG" 2>/dev/null || echo 0)"
}

# ===========================================
# Run Debug Inspector
# ===========================================

run_inspector() {
    print_step "Running Walking Inspector (State Dump)"

    BIN="$BIN_DIR/walking_inspector"
    if [ ! -f "$BIN" ]; then
        echo "Binary not found, building..."
        build_experiments
    fi

    TIMESTAMP=$(timestamp)
    LOG="$LOG_DIR/inspector_${TIMESTAMP}.log"

    echo "This tool steps through each algorithm phase and dumps full state."
    echo "Log: $LOG"
    echo ""

    $BIN "$NUM_VERTICES" "$NUM_EDGES" "$DEBUG_LEVEL" 2>&1 | tee "$LOG"

    echo ""
    echo "=== Inspector Summary ==="
    grep -c '^\[INSPECT\]' "$LOG" 2>/dev/null | xargs -I{} echo "State dumps: {}"
    grep -c '^\[ASSERT\]' "$LOG" 2>/dev/null | xargs -I{} echo "Assertions checked: {}"
}

# ===========================================
# Run GPU Tree Traversal Experiment
# ===========================================

run_gpu_tree() {
    print_step "Running GPU Tree Traversal Experiment"

    BIN="$BIN_DIR/walking_gpu_tree"
    if [ ! -f "$BIN" ]; then
        echo "Binary not found, building..."
        build_experiments
    fi

    NUM_KEYS="${NUM_KEYS:-100000}"
    NUM_QUERIES="${NUM_QUERIES:-50000}"

    TIMESTAMP=$(timestamp)
    LOG="$LOG_DIR/gpu_tree_${TIMESTAMP}.log"

    echo "Keys:     $NUM_KEYS"
    echo "Queries:  $NUM_QUERIES"
    echo "Debug:    $DEBUG_LEVEL"
    echo "Log:      $LOG"
    echo ""

    if command -v nvidia-smi &>/dev/null; then
        echo "=== GPU Status ==="
        nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu \
                   --format=csv,noheader 2>/dev/null || echo "  (query failed)"
        echo ""
    fi

    RUN_CMD="$BIN $NUM_KEYS $NUM_QUERIES $DEBUG_LEVEL"
    echo "Command: $RUN_CMD"
    echo ""

    if command -v numactl &>/dev/null; then
        echo "[NUMA] Binding to node 0"
        numactl --cpunodebind=0 --membind=0 $RUN_CMD 2>&1 | tee "$LOG"
    else
        $RUN_CMD 2>&1 | tee "$LOG"
    fi

    echo ""
    echo "=== Post-run Metrics ==="
    grep -E '^\[T·END\]' "$LOG" 2>/dev/null || echo "(no timer data)"
    echo ""
    grep -E '^\[ASSERT' "$LOG" 2>/dev/null | tail -5
    echo ""
    echo "Inspections: $(grep -c '^\[INSPECT' "$LOG" 2>/dev/null || echo 0)"
}

# ===========================================
# Evaluation — Parse Results + Compare
# ===========================================

run_evaluation() {
    print_step "Running Evaluation & Comparison"

    echo "Parsing experiment logs..."

    python3 - <<'PYEOF'
import os, json, re, glob

log_dir = os.environ.get("LOG_DIR", "./logs")
out_dir = os.environ.get("OUTPUT_DIR", "./results")
os.makedirs(out_dir, exist_ok=True)

results = {"synthetic": [], "realscale": [], "inspector": []}

for kind in ["synthetic", "realscale", "inspector"]:
    logs = sorted(glob.glob(os.path.join(log_dir, f"{kind}_*.log")))
    for lf in logs[-3:]:  # last 3 runs
        run = {"log": lf, "timers": {}, "results": {}, "breakpoints": 0}
        with open(lf, 'r') as f:
            for line in f:
                # Timers
                m = re.match(r'\[TIMER·END\]\s+(\S+)\s+→\s+([\d.]+)\s+ms', line)
                if not m:
                    m = re.match(r'\[T·END\]\s+(\S+)\s+→\s+([\d.]+)\s+ms', line)
                if m:
                    run["timers"][m.group(1)] = float(m.group(2))
                # Results
                m = re.match(r'\[(\w+)·RESULT\]\s+(.*)', line)
                if m:
                    run["results"][m.group(1)] = m.group(2).strip()
                # Breakpoints
                if line.startswith("[BP·"):
                    run["breakpoints"] += 1
        results[kind].append(run)

out_file = os.path.join(out_dir, "eval_summary.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== Evaluation Summary ===")
for kind, runs in results.items():
    if not runs:
        continue
    latest = runs[-1]
    print(f"\n[{kind.upper()}] (from {latest['log']})")
    for k, v in latest["timers"].items():
        print(f"  {k}: {v:.1f} ms")
    for k, v in latest["results"].items():
        print(f"  {k}: {v}")
    print(f"  breakpoints_hit: {latest['breakpoints']}")

print(f"\nFull results: {out_file}")
PYEOF

    echo ""
    echo "✓ Evaluation complete"
}

# ===========================================
# Full Pipeline: all steps
# ===========================================

run_all() {
    check_system
    setup_environment
    build_experiments
    generate_data
    run_synthetic
    run_realscale
    run_inspector
    run_gpu_tree
    run_evaluation
}

# ===========================================
# Help
# ===========================================

show_help() {
    cat << 'EOF'
LLM4Walking — Philemon-TSH Graph Walking Pipeline

Usage: ./llm4walking_run.sh [command]

Commands:
  check         - Check system requirements
  setup         - Setup build environment
  clone         - Clone upstream repository
  build         - Compile experiment binaries
  generate      - Generate graph data (synthetic + download SNAP)
  synthetic     - Run synthetic graph experiment
  realscale     - Run real-scale experiment
  inspector     - Run debug state inspector
  gpu_tree      - Run GPU tree traversal (ART + Interval, nvcc or CPU fallback)
  eval          - Parse logs and evaluate results
  all           - Full pipeline (check→build→generate→run→eval)

Environment Variables:
  NUM_VERTICES  - Vertices in synthetic graph (default: 10000)
  NUM_EDGES     - Edges in synthetic graph (default: 50000)
  SEED          - Random seed (default: 42)
  DEBUG_LEVEL   - 0=silent 1=summary 2=per-step 3=per-edge (default: 2)
  NUM_THREADS   - Thread count (default: 1)
  PR_ITERS      - PageRank iterations (default: 10)
  EDGE_FILE     - Path to edge list for realscale (auto-detect if unset)
  DATA_DIR      - Data directory (default: ./data)
  OUTPUT_DIR    - Results directory (default: ./results)

Upstream: https://github.com/dylanyunlon/philemon-TSH
EOF
}

# ===========================================
# Main Dispatch
# ===========================================

main() {
    COMMAND=${1:-"help"}

    case $COMMAND in
        check)
            check_system
            ;;
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_upstream
            ;;
        build)
            build_experiments
            ;;
        generate|gen)
            generate_data
            ;;
        synthetic|synth)
            run_synthetic
            ;;
        realscale|real)
            run_realscale
            ;;
        inspector|debug)
            run_inspector
            ;;
        gpu_tree|gpu|tree)
            run_gpu_tree
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        all)
            run_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4walking_run.sh help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
