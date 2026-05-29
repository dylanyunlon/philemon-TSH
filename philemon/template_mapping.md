# Fill-In Mapping — DES-LOC template role → Philemon-TSH concept

The DES-LOC paper is the skeleton. Each row gives a *role* in that skeleton (the
thing a placeholder stands for) and the **Philemon-TSH** concept that fills it.
This is the contract that keeps the five sections coherent across Claudes; it is
the Philemon-TSH analogue of the template repo's `mapping_database.json`.

## Top-level narrative roles (drive Abstract + §1)

| DES-LOC role | Philemon-TSH fill |
|---|---|
| System name | **Philemon-TSH** (Temporal Subgraph on Heterogeneous memory) |
| Bandwidth-limited baseline (DDP) | **Capacity-limited single-tier store**: pinning the whole temporal graph in HBM |
| "synchronize parameters only" (Local SGD/FedAvg) | **single-tier interval index** (TEM-Graph) / per-device cache pooling (PyTorch/NCCL) — no hierarchy |
| Heuristic that keeps state local / resets it | **naive demotion**: spill cold data to DRAM with no temporal-locality awareness |
| Provably-convergent-but-expensive (Local Adam) | **all-HBM with a full index**: correct + fast but does not fit billion-edge graphs |
| Per-parameter states (1st/2nd moment u,v) | **temporal partitions** carrying access-frequency / hotness state |
| Independent sync periods K_x / K_u / K_v | **independent residency tiers HBM / GDDR / DRAM**, assigned by access frequency & temporal density |
| Half-life timescale separation τ₀.₅(β)=ln0.5/lnβ | **hot/cold timescale separation**: dense, recently-touched intervals are "fast-changing" → HBM; sparse, stale intervals → DRAM |
| "Parameters dominate the asymptotic rate" | **partition *selection* dominates query cost** (the O(P) inter-partition sweep) |
| "2nd moment needs ≥ infrequent sync (β₂<1 ⇒ pv>0)" | **streaming needs ≥ an incremental index** (segmented/LSM) or rebuild cost explodes |
| Convergence guarantee | **query-correctness guarantee** (indexed result ≡ exhaustive scan) |
| Larger stable step size from frequent momentum sync | **lower query latency from a finer, indexed partition selection** |
| 170× vs DDP / 2× vs Local Adam | **43.9× intra-partition scan vs linear**; **up to 4.1× partition selection vs full sweep** |
| 1.3–2.1× wall-clock speedup | **sub-millisecond temporal subgraph retrieval** on the H100+A6000+DRAM hierarchy |
| Robust to worker failures | **race-free / torn-read-free concurrency** (seqlock; TSan-clean under 2.1M concurrent queries) |

## Component roles (seed §2 The System)

| DES-LOC role | Philemon-TSH fill | Pattern source ("from C, the good example") |
|---|---|---|
| Core algorithm (DES-LOC update) | **TieredAllocator** waterfall HBM→GDDR→DRAM | NCCL `ncclMemAlloc`→`cuMemCreate`/`cuMemMap`; Megatron MultiGroupMemPoolAllocator |
| State-class partitioning of the model | **TemporalBridge**: interval → subgraph partition → tier placement | RapidStore `wrapper::snapshot_edges`; TEM-Graph `build_index` |
| Synchronization schedule / cadence | **MigrationScheduler** background hotness sweep | NCCL topo ring/tree; Megatron deferred batched param hooks |
| Cheap, non-blocking state read | **SeqLock** wait-free optimistic reads (read_begin/read_retry) | Linux seqlock / NCCL seq_num; replaces abseil ReaderLock & PyTorch COWDeleter shared_mutex |
| Memory-state container | **SlabAllocator** per-tier size-class pools, O(1) bitmask slots, `compact()` | NCCL `freeMask`+`popFirstOneBit`; PyTorch `Block`/`try_merge_blocks`; TF Arena bump alloc |
| Intra-partition fast lookup | **IntervalIndex** dual sorted views (by_start/by_end), O(log N + k) | PyTorch `BlockPool` dual `std::set`; LevelDB `Seek`; Thrust `lower_bound` |
| Inter-partition fast selection | **PartitionSkipList** (span-max(ts_hi) augmentation) → O(log P + k) | Pugh skip list + CLRS §14.3 interval-tree subtree-max |
| Incremental schedule under streaming | **SegmentedPartitionIndex** (one immutable segment per flush, compaction at threshold 8) | LSM-tree / Lucene segments |
| Overlap of background work with hot path | **AsyncMigrationEngine + TierPtr** (RAII scope-safe handle, double-buffer, event fence) | PyTorch `CUDAStreamGuard`; cudaMemcpyAsync double-buffering |

## Theory roles (seed §3 Guarantees)

| DES-LOC role | Philemon-TSH fill |
|---|---|
| Assumptions 1–3 (lower-bound, L-smoothness, bounded heterogeneity) | **Invariants**: partitions sorted by ts_lo on flush; intervals closed; span-max ≥ every member ts_hi |
| Theorem 1 (O(1/√T) rate, ψ factor) | **Theorem (selection cost)**: augmented skip-list `overlaps(lo,hi)` is O(log P + k); pruning `span_max < lo` is sound |
| ψ-factor boundary analysis (px², linear pu) | **Amortization analysis**: per-flush O(M log M) segment build; compaction at threshold ⇒ amortized O(log P) build, no O(N²log N) cliff |
| "high-probability guarantee needs pv>0" | **Streaming-correctness**: segmented walk ≡ whole-list ≡ brute force (three-way cross-check) |
| Toy-problem verification (Fig. 1) | **Validation harness**: 10M exhaustive + 32k random/adversarial + 20k three-way cross-checks; 900/900 runtime samples by alloc_id; 2.1M-query TSan run, no race |
