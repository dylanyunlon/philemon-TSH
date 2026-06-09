#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_sota_comparison.sh — 运行 SOTA 对比实验
# M198: 所有系统同机同条件对比
#
# 输出: experiment/results/sota_comparison.csv
#       experiment/results/sota_gpu_tiered.csv
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/experiment/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/experiment/logs/${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# ─── 系统信息 ───
echo "# System Info — $(date)" > "${LOG_DIR}/sysinfo.md"
echo "## Hardware" >> "${LOG_DIR}/sysinfo.md"
lscpu | grep -E "Model name|CPU\(s\):|Socket|Core|Thread|NUMA" >> "${LOG_DIR}/sysinfo.md"
free -h >> "${LOG_DIR}/sysinfo.md"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
    --format=csv,noheader >> "${LOG_DIR}/sysinfo.md" 2>/dev/null

# ─── 数据集配置 ───
DATASETS=(
    "rmat:14:262144"
    "rmat:16:1048576"
    "rmat:18:4194304"
    "rmat:20:16777216"
    "rmat:22:67108864"
)
ALGORITHMS=("BFS" "PageRank" "SSSP" "WCC")
SEEDS=(42 123 456)
NUM_THREADS=32

# ─── CSV 头 ───
OUTFILE="${RESULTS_DIR}/sota_comparison.csv"
echo "# SOTA Comparison — ${TIMESTAMP}" > "${OUTFILE}"
echo "# Generated on ags1 (H100+2xA6000+1.5TB DRAM)" >> "${OUTFILE}"
echo "timestamp,dataset,scale,edges,system,algorithm,seed,threads,latency_ms,throughput_meps,memory_mb,p50_ms,p99_ms" >> "${OUTFILE}"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  SOTA Comparison Experiment — ${TIMESTAMP}                ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ─── 运行各系统 ───

run_philemon() {
    local dataset=$1 scale=$2 edges=$3 algo=$4 seed=$5
    echo "  [Philemon] ${dataset}:${scale} ${algo} seed=${seed}..."

    local binary="${PROJECT_ROOT}/build/philemon_bench"
    if [ ! -f "${binary}" ]; then
        echo "    SKIP: binary not found"
        return
    fi

    local result
    result=$(timeout 300 "${binary}" \
        --scale "${scale}" \
        --algorithm "${algo}" \
        --seed "${seed}" \
        --threads "${NUM_THREADS}" \
        --tier-config "hbm:gpu2,gddr:gpu0,dram:cpu" \
        --json 2>&1) || { echo "    TIMEOUT/ERROR"; return; }

    # 解析输出 (假设 benchmark 输出 JSON 或 key=value)
    local lat_ms mem_mb
    lat_ms=$(echo "${result}" | grep -oP 'latency_ms=\K[0-9.]+' || echo "0")
    mem_mb=$(echo "${result}" | grep -oP 'memory_mb=\K[0-9.]+' || echo "0")

    echo "${TIMESTAMP},${dataset},${scale},${edges},Philemon,${algo},${seed},${NUM_THREADS},${lat_ms},0,${mem_mb},0,0" >> "${OUTFILE}"
}

run_rapidstore() {
    local dataset=$1 scale=$2 edges=$3 algo=$4 seed=$5
    echo "  [RapidStore] ${dataset}:${scale} ${algo} seed=${seed}..."

    local binary="${PROJECT_ROOT}/baselines/RapidStore/build/wrapper/neo_wrapper.out"
    if [ ! -f "${binary}" ]; then
        echo "    SKIP: binary not found"
        return
    fi

    # RapidStore 使用 DynamicGraphStorage 测试框架的配置格式
    local result
    result=$(timeout 300 "${binary}" \
        --workload_type "algorithm" \
        --dataset_type "rmat" \
        --rmat_scale "${scale}" \
        --algorithm "${algo}" \
        --num_threads "${NUM_THREADS}" \
        2>&1) || { echo "    TIMEOUT/ERROR"; return; }

    local lat_ms mem_mb
    lat_ms=$(echo "${result}" | grep -oP 'avg_latency.*?([0-9.]+)' | grep -oP '[0-9.]+$' || echo "0")
    mem_mb=$(echo "${result}" | grep -oP 'memory.*?([0-9.]+)' | grep -oP '[0-9.]+$' || echo "0")

    echo "${TIMESTAMP},${dataset},${scale},${edges},RapidStore,${algo},${seed},${NUM_THREADS},${lat_ms},0,${mem_mb},0,0" >> "${OUTFILE}"
}

run_csr_baseline() {
    local dataset=$1 scale=$2 edges=$3 algo=$4 seed=$5
    echo "  [CSR] ${dataset}:${scale} ${algo} seed=${seed}..."

    local binary="${PROJECT_ROOT}/baselines/DynamicGraphStorage/build/csr_wrapper"
    if [ ! -f "${binary}" ]; then
        # 尝试 RapidStore 的 CSR wrapper
        binary="${PROJECT_ROOT}/baselines/RapidStore/build/wrapper/csr_wrapper.out"
    fi
    if [ ! -f "${binary}" ]; then
        echo "    SKIP: CSR binary not found"
        return
    fi

    local result
    result=$(timeout 300 "${binary}" \
        --workload_type "algorithm" \
        --dataset_type "rmat" \
        --rmat_scale "${scale}" \
        --algorithm "${algo}" \
        --num_threads "${NUM_THREADS}" \
        2>&1) || { echo "    TIMEOUT/ERROR"; return; }

    local lat_ms mem_mb
    lat_ms=$(echo "${result}" | grep -oP 'avg_latency.*?([0-9.]+)' | grep -oP '[0-9.]+$' || echo "0")
    mem_mb=$(echo "${result}" | grep -oP 'memory.*?([0-9.]+)' | grep -oP '[0-9.]+$' || echo "0")

    echo "${TIMESTAMP},${dataset},${scale},${edges},CSR,${algo},${seed},${NUM_THREADS},${lat_ms},0,${mem_mb},0,0" >> "${OUTFILE}"
}

# ─── 主循环 ───
for ds_spec in "${DATASETS[@]}"; do
    IFS=: read -r ds_type ds_scale ds_edges <<< "${ds_spec}"
    for algo in "${ALGORITHMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_csr_baseline "${ds_type}" "${ds_scale}" "${ds_edges}" "${algo}" "${seed}"
            run_philemon "${ds_type}" "${ds_scale}" "${ds_edges}" "${algo}" "${seed}"
            run_rapidstore "${ds_type}" "${ds_scale}" "${ds_edges}" "${algo}" "${seed}"
        done
    done
done

# ─── 汇总 ───
echo ""
echo ">>> Results saved to: ${OUTFILE}"
echo ">>> Logs saved to: ${LOG_DIR}/"
wc -l "${OUTFILE}"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Experiment complete. Run auto_push_results.sh to push.   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
