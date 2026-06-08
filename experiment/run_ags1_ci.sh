#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# philemon-TSH ags1 CI Runner
# 自动编译 → 多规模运行 → 日志 git push → 下一位Claude读取并迭代
#
# 用法:
#   ./experiment/run_ags1_ci.sh                    # 运行全部实验
#   ./experiment/run_ags1_ci.sh m145_m146           # 只运行指定实验
#   SCALES="1000 10000" ./experiment/run_ags1_ci.sh # 自定义规模
#
# 环境变量:
#   SCALES     - 空格分隔的测试规模 (默认: "1000 10000 100000 1000000")
#   THREADS    - 线程数 (默认: 32, 即半个NUMA node)
#   NUMA_NODE  - NUMA节点 (默认: 1, GPU所在节点)
#   AUTO_PUSH  - 是否自动git push (默认: 1)
#   CXX        - 编译器 (默认: g++)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ── 配置 ──────────────────────────────────────────────────────────
SCALES="${SCALES:-1000 10000 100000 1000000}"
THREADS="${THREADS:-32}"
NUMA_NODE="${NUMA_NODE:-1}"
AUTO_PUSH="${AUTO_PUSH:-1}"
CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -fopenmp -march=znver4"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="experiment/logs/${TIMESTAMP}"
SUMMARY_FILE="${LOG_DIR}/SUMMARY.md"

# ── 颜色输出 ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date +%H:%M:%S) $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date +%H:%M:%S) $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date +%H:%M:%S) $*"; }

# ── 系统信息采集 ──────────────────────────────────────────────────
collect_sysinfo() {
    local outfile="$1"
    {
        echo "# System Info — $(date)"
        echo "## Hardware"
        echo '```'
        lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture" 2>/dev/null || true
        echo ""
        free -h 2>/dev/null || cat /proc/meminfo | head -5
        echo ""
        nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "No GPU"
        echo '```'
        echo ""
        echo "## Software"
        echo '```'
        uname -r
        ${CXX} --version | head -1
        echo "NUMA node: ${NUMA_NODE}"
        echo "Threads: ${THREADS}"
        echo "Scales: ${SCALES}"
        echo '```'
    } > "$outfile"
}

# ── 发现实验文件 ──────────────────────────────────────────────────
discover_experiments() {
    local filter="${1:-}"
    local experiments=()
    for f in experiment/*_experiment.cpp; do
        [ -f "$f" ] || continue
        local name=$(basename "$f" .cpp)
        if [ -z "$filter" ] || [[ "$name" == *"$filter"* ]]; then
            experiments+=("$name")
        fi
    done
    echo "${experiments[@]}"
}

# ── 编译 ──────────────────────────────────────────────────────────
compile_experiment() {
    local src="$1"
    local bin="$2"
    log_info "Compiling: ${src} → ${bin}"
    
    local compile_log="${LOG_DIR}/compile_$(basename "$src" .cpp).log"
    if ${CXX} ${CXXFLAGS} -o "$bin" "$src" 2>"$compile_log"; then
        log_info "Compile OK: $(wc -c < "$bin") bytes"
        return 0
    else
        log_error "Compile FAILED — see ${compile_log}"
        cat "$compile_log"
        return 1
    fi
}

# ── 运行单个实验 ──────────────────────────────────────────────────
run_single() {
    local bin="$1"
    local scale="$2"
    local name=$(basename "$bin")
    local run_log="${LOG_DIR}/${name}_scale${scale}.log"
    local debug_log="${LOG_DIR}/${name}_scale${scale}_debug.log"
    
    log_info "Running: ${name} --scale ${scale} --threads ${THREADS}"
    
    local start_ns=$(date +%s%N)
    
    if command -v numactl &>/dev/null; then
        numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \
            "$bin" --scale "$scale" --threads "$THREADS" \
            >"$run_log" 2>"$debug_log"
        local exit_code=$?
    else
        "$bin" --scale "$scale" --threads "$THREADS" \
            >"$run_log" 2>"$debug_log"
        local exit_code=$?
    fi
    
    local end_ns=$(date +%s%N)
    local elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    
    if [ $exit_code -eq 0 ]; then
        # 提取结果摘要
        local pass=$(grep -c "PASS" "$run_log" || echo 0)
        local fail=$(grep -c "FAIL" "$run_log" || echo 0)
        log_info "  DONE: ${pass} PASS, ${fail} FAIL, ${elapsed_ms}ms total"
        echo "${name},${scale},${pass},${fail},${elapsed_ms},OK" >> "${LOG_DIR}/results.csv"
    else
        log_error "  CRASHED (exit=${exit_code}) — see ${run_log}"
        echo "${name},${scale},0,0,${elapsed_ms},CRASH(${exit_code})" >> "${LOG_DIR}/results.csv"
    fi
    
    return $exit_code
}

# ── 生成摘要 ──────────────────────────────────────────────────────
generate_summary() {
    {
        echo "# CI Run Summary — ${TIMESTAMP}"
        echo ""
        echo "## Results"
        echo ""
        echo "| Experiment | Scale | PASS | FAIL | Time(ms) | Status |"
        echo "|-----------|-------|------|------|----------|--------|"
        
        if [ -f "${LOG_DIR}/results.csv" ]; then
            while IFS=, read -r name scale pass fail ms status; do
                echo "| ${name} | ${scale} | ${pass} | ${fail} | ${ms} | ${status} |"
            done < "${LOG_DIR}/results.csv"
        fi
        
        echo ""
        echo "## Algorithm Performance (from logs)"
        echo ""
        
        # 提取性能表格
        for f in "${LOG_DIR}"/*_scale*.log; do
            [ -f "$f" ] || continue
            if grep -q "Algorithm.*Time" "$f"; then
                echo "### $(basename "$f" .log)"
                echo '```'
                grep -A 10 "┌─────" "$f" || true
                echo '```'
                echo ""
            fi
        done
        
        echo "## For Next Claude"
        echo ""
        echo "读取以下文件获取详细日志:"
        echo '```'
        ls -la "${LOG_DIR}"/*.log 2>/dev/null || echo "No logs"
        echo '```'
        echo ""
        echo "下一步任务: M147-M148 (main.cpp + wrapper.h + driver.h)"
    } > "$SUMMARY_FILE"
}

# ── Git push ──────────────────────────────────────────────────────
git_push_logs() {
    if [ "${AUTO_PUSH}" != "1" ]; then
        log_warn "AUTO_PUSH=0, skipping git push"
        return
    fi
    
    log_info "Git: adding logs and pushing..."
    cd "$REPO_DIR"
    
    git add "${LOG_DIR}/" 2>/dev/null || true
    git add experiment/logs/ 2>/dev/null || true
    
    local msg="CI run ${TIMESTAMP}: "
    if [ -f "${LOG_DIR}/results.csv" ]; then
        local total_pass=$(awk -F, '{s+=$3}END{print s+0}' "${LOG_DIR}/results.csv")
        local total_fail=$(awk -F, '{s+=$4}END{print s+0}' "${LOG_DIR}/results.csv")
        msg+="${total_pass} PASS, ${total_fail} FAIL"
    else
        msg+="no results"
    fi
    
    git commit -m "$msg" --author="dylanyunlon <dogechat@163.com>" 2>/dev/null || true
    
    if git push origin main 2>/dev/null; then
        log_info "Git push OK — Claude小弟可以读取日志了"
    else
        log_warn "Git push failed (可能需要先 git pull)"
        git pull --rebase origin main 2>/dev/null && git push origin main 2>/dev/null || \
            log_error "Push still failed, 手动解决冲突"
    fi
}

# ── 主流程 ────────────────────────────────────────────────────────
main() {
    local filter="${1:-}"
    
    echo "═══════════════════════════════════════════════════════"
    echo " philemon-TSH ags1 CI Runner"
    echo " Time: $(date)"
    echo " Scales: ${SCALES}"
    echo " NUMA: node${NUMA_NODE}, ${THREADS} threads"
    echo "═══════════════════════════════════════════════════════"
    
    # 创建日志目录
    mkdir -p "$LOG_DIR"
    echo "experiment,scale,pass,fail,time_ms,status" > "${LOG_DIR}/results.csv"
    
    # 采集系统信息
    collect_sysinfo "${LOG_DIR}/sysinfo.md"
    
    # 发现实验
    local experiments=($(discover_experiments "$filter"))
    if [ ${#experiments[@]} -eq 0 ]; then
        log_error "No experiments found (filter='${filter}')"
        exit 1
    fi
    log_info "Found ${#experiments[@]} experiment(s): ${experiments[*]}"
    
    # 编译 + 运行
    local build_dir="/tmp/philemon_build_${TIMESTAMP}"
    mkdir -p "$build_dir"
    
    local any_fail=0
    for exp in "${experiments[@]}"; do
        local src="experiment/${exp}.cpp"
        local bin="${build_dir}/${exp}"
        
        if ! compile_experiment "$src" "$bin"; then
            any_fail=1
            continue
        fi
        
        for scale in ${SCALES}; do
            run_single "$bin" "$scale" || any_fail=1
        done
    done
    
    # 生成摘要
    generate_summary
    log_info "Summary: ${SUMMARY_FILE}"
    cat "$SUMMARY_FILE"
    
    # Git push
    git_push_logs
    
    # 清理
    rm -rf "$build_dir"
    
    echo ""
    echo "═══════════════════════════════════════════════════════"
    if [ $any_fail -eq 0 ]; then
        echo " ALL PASS — 日志已push，等待下一位Claude"
    else
        echo " SOME FAILURES — 检查 ${LOG_DIR}/ 中的日志"
    fi
    echo "═══════════════════════════════════════════════════════"
    
    return $any_fail
}

main "$@"
