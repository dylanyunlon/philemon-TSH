#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run_gpu_tiered.sh — 真实 GPU 三级内存实验
# M199-M204: H100 HBM / A6000 GDDR / CPU DRAM
#
# 硬件映射:
#   Tier 0 (HBM):  GPU2 = H100 NVL 96GB, sm_90, NUMA1
#   Tier 1 (GDDR): GPU0 = A6000 49GB, sm_86, NUMA1
#   Tier 2 (DRAM): CPU host memory, NUMA1 (~774GB)
#
# 关键: 所有 GPU 在 NUMA1，用 numactl --cpunodebind=1 --membind=1
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/experiment/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/experiment/logs/gpu_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  GPU Tiered Memory Experiment — ${TIMESTAMP}              ║"
echo "║  H100(HBM) + A6000(GDDR) + DRAM                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ─── 验证 GPU 可用 ───
echo ""
echo ">>> Verifying GPUs..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

# ─── 检查 CUDA compute capability ───
NVCC_ARCHS=$(nvcc --version 2>/dev/null | tail -1 || echo "unknown")
echo "  CUDA: ${NVCC_ARCHS}"
echo "  需要 sm_86 (A6000) + sm_90 (H100)"

# ─── 配置 ───
BINARY="${PROJECT_ROOT}/build/philemon_bench"
if [ ! -f "${BINARY}" ]; then
    echo "ERROR: ${BINARY} not found. Run cmake --build build first."
    exit 1
fi

# 实验参数
SCALES=(14 16 18 20)
WINDOWS=("narrow:1000" "medium:50000" "wide:500000" "full:1000000")
SEEDS=(42 123 456)
N_POINTS=2000

# ─── RQ1: Index 预测 selection cost ───
echo ""
echo "═══ RQ1: Index Prediction vs Measured Selection Cost ═══"
RQ1_OUT="${RESULTS_DIR}/rq1_index_prediction.csv"
echo "scale,window,seed,predicted_cost_us,measured_cost_us,ratio" > "${RQ1_OUT}"

for scale in "${SCALES[@]}"; do
    for win_spec in "${WINDOWS[@]}"; do
        IFS=: read -r win_name win_size <<< "${win_spec}"
        for seed in "${SEEDS[@]}"; do
            echo "  scale=${scale} window=${win_name} seed=${seed}..."
            numactl --cpunodebind=1 --membind=1 \
                "${BINARY}" \
                --mode rq1_index \
                --scale "${scale}" \
                --window-size "${win_size}" \
                --seed "${seed}" \
                --tier-config "hbm:2,gddr:0,dram:host" \
                --output "${RQ1_OUT}" \
                2>>"${LOG_DIR}/rq1.log" || echo "  FAILED"
        done
    done
done

# ─── RQ2: Tier Placement vs Latency ───
echo ""
echo "═══ RQ2: Tier Placement vs Query Latency ═══"
RQ2_OUT="${RESULTS_DIR}/rq2_tier_latency.csv"
echo "scale,tier_config,algorithm,seed,latency_us,qps" > "${RQ2_OUT}"

TIER_CONFIGS=("tiered:hbm+gddr+dram" "hbm_only:hbm" "dram_only:dram")
for scale in 18 20; do
    for tc_spec in "${TIER_CONFIGS[@]}"; do
        IFS=: read -r tc_name tc_tiers <<< "${tc_spec}"
        for seed in "${SEEDS[@]}"; do
            echo "  scale=${scale} config=${tc_name} seed=${seed}..."
            numactl --cpunodebind=1 --membind=1 \
                "${BINARY}" \
                --mode rq2_tier \
                --scale "${scale}" \
                --tier-config "${tc_tiers}" \
                --seed "${seed}" \
                --n-points "${N_POINTS}" \
                --output "${RQ2_OUT}" \
                2>>"${LOG_DIR}/rq2.log" || echo "  FAILED"
        done
    done
done

# ─── RQ3: Scan/Selection Speedup ───
echo ""
echo "═══ RQ3: Intra-partition Scan + Selection Speedup ═══"
RQ3_OUT="${RESULTS_DIR}/rq3_speedup.csv"
echo "scale,window,method,seed,scan_us,selection_us,speedup_scan,speedup_sel" > "${RQ3_OUT}"

for scale in "${SCALES[@]}"; do
    for win_spec in "${WINDOWS[@]}"; do
        IFS=: read -r win_name win_size <<< "${win_spec}"
        for seed in "${SEEDS[@]}"; do
            echo "  scale=${scale} window=${win_name} seed=${seed}..."
            numactl --cpunodebind=1 --membind=1 \
                "${BINARY}" \
                --mode rq3_speedup \
                --scale "${scale}" \
                --window-size "${win_size}" \
                --seed "${seed}" \
                --output "${RQ3_OUT}" \
                2>>"${LOG_DIR}/rq3.log" || echo "  FAILED"
        done
    done
done

# ─── RQ4: Scaling to 100M edges ───
echo ""
echo "═══ RQ4: Large-Scale Scaling (up to 2^24) ═══"
RQ4_OUT="${RESULTS_DIR}/rq4_scaling.csv"
echo "scale,edges,tier_config,algorithm,seed,latency_ms,memory_mb,insert_meps" > "${RQ4_OUT}"

for scale in 14 16 18 20 22 24; do
    for seed in "${SEEDS[@]}"; do
        echo "  scale=${scale} seed=${seed}..."
        numactl --cpunodebind=1 --membind=1 \
            timeout 600 "${BINARY}" \
            --mode rq4_scaling \
            --scale "${scale}" \
            --seed "${seed}" \
            --tier-config "hbm:2,gddr:0,dram:host" \
            --output "${RQ4_OUT}" \
            2>>"${LOG_DIR}/rq4.log" || echo "  FAILED (timeout or error)"
    done
done

# ─── RQ5: Streaming + Compaction ───
echo ""
echo "═══ RQ5: Streaming Insertion + LSM Compaction ═══"
RQ5_OUT="${RESULTS_DIR}/rq5_streaming.csv"
echo "flush_id,total_edges,flush_lat_us,compact,compact_lat_ms,mismatches" > "${RQ5_OUT}"

numactl --cpunodebind=1 --membind=1 \
    "${BINARY}" \
    --mode rq5_streaming \
    --scale 20 \
    --n-flushes 256 \
    --edges-per-flush 50000 \
    --output "${RQ5_OUT}" \
    2>>"${LOG_DIR}/rq5.log" || echo "  FAILED"

# ─── RQ6: Concurrent Queries During Migration ───
echo ""
echo "═══ RQ6: Query-Under-Migration Concurrency ═══"
RQ6_OUT="${RESULTS_DIR}/rq6_concurrent.csv"
echo "phase,n_queries,n_migrations,query_lat_us,migration_lat_us,races_detected" > "${RQ6_OUT}"

numactl --cpunodebind=1 --membind=1 \
    "${BINARY}" \
    --mode rq6_concurrent \
    --scale 20 \
    --query-threads 16 \
    --migration-threads 4 \
    --duration-sec 30 \
    --output "${RQ6_OUT}" \
    2>>"${LOG_DIR}/rq6.log" || echo "  FAILED"

# ─── 汇总 ───
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  GPU Tiered Experiment Complete                           ║"
echo "╠═══════════════════════════════════════════════════════════╣"
for f in "${RQ1_OUT}" "${RQ2_OUT}" "${RQ3_OUT}" "${RQ4_OUT}" "${RQ5_OUT}" "${RQ6_OUT}"; do
    echo "║  $(basename "$f"): $(wc -l < "$f") lines"
done
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo ">>> Auto-pushing results..."
bash "${SCRIPT_DIR}/auto_push_results.sh" "experiment: GPU tiered RQ1-RQ6 ${TIMESTAMP}"
