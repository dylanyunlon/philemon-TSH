# Philemon-TSH Paper — Multi-Claude Fill-In Plan

**Template:** `dylanyunlon/Desynced-Low-Communication` (the DES-LOC / ICLR-2026
paper, abstracted into a fill-in skeleton: `template.txt` + `mapping_database.json`,
worked example in `des_loc_reconstructed.tex` which fills the skeleton for a
*second* system, "Neuron_SP").

**Fill content:** `dylanyunlon/walpurgis-WTFGG` — the **Philemon-TSH** system
("Walpurgis": Workload-Aware GNN training on mixed-generation GPUs; temporal
subgraph processing on heterogeneous memory HBM/GDDR/DRAM).

**Output:** `philemon_tsh.tex` — a self-contained NeurIPS-style paper, structurally
1:1 with the DES-LOC template's five main body sections (appendix is **out of scope**).

---

## The five "big" sections (template → Philemon-TSH)

| # | DES-LOC template section | Philemon-TSH section | Claude |
|---|--------------------------|----------------------|--------|
| §1 | Introduction | Introduction | **#1 (this)** |
| §2 | DES-LOC Algorithm (method) | The Philemon-TSH System (tiered allocator + temporal bridge + migration scheduler + seqlock + slab) | #2 |
| §3 | Convergence Guarantees (theory) | Correctness & Complexity Guarantees (skip-list span-max pruning, O(log P + k) selection, segmented-LSM amortization, seqlock read consistency) | #3 |
| §4 | Experimental Design | Experimental Design (RQs, workloads LDBC + synthetic, tiers, baselines, metrics, hardware) | #4 |
| §5 | Evaluation (RQ1–RQ6) | Evaluation (scan speedup, selection speedup, latency/QPS/memory/migration curves, scalability) | #5 |

Related Work + Conclusion are short and **fold into Claude #5's pass** (appended
after Evaluation), matching DES-LOC §6–§7 length. They are not "big" sections.

---

## Mandatory norms (every later Claude obeys)

1. **Structural 1:1 with the template.** Mirror the DES-LOC section/subsection
   skeleton (abstract → intro → centered research question → 4 bold contributions
   → method subsections → assumptions/theorems → experimental setup → RQ-keyed
   evaluation subsections). Do not invent block types the template doesn't have.
2. **Content is Philemon-TSH, grounded in the repo.** Every technical claim, number,
   and pattern-lineage statement must trace to `walpurgis-WTFGG` (source headers,
   `DEVELOPMENT_PLAN.md`, `REVIEW_M*.md`, or the `philemon_*_2000.json` data).
   Do not import DES-LOC's optimizer numbers; do not fabricate results.
3. **Chained edits.** Claude-N edits the file at the HEAD left by Claude-(N-1).
   Re-read the section stubs and the glossary (`template_mapping.md`) before writing.
   Keep the embedded `filecontents*` bib in sync — add a `\bibitem` / `.bib` entry
   for every new `\cite` key.
4. **Compiles standalone.** `pdflatex philemon_tsh.tex` (+ bibtex) must succeed with
   no external style files. Preamble is self-contained (`article` + amsmath/amsthm/
   algorithm/booktabs/hyperref/geometry). Verify before handing off.
5. **The fill-in mapping is the contract.** `template_mapping.md` records
   DES-LOC-skeleton-token → Philemon-TSH-concept. When a later section needs a new
   abstract slot, append it there; do not silently overload an existing symbol.

---

## Status

- **Claude #1 — DONE.** Title, author block, abstract, full §1 Introduction
  (+ 4 contributions), self-contained preamble + theorem envs + embedded bib,
  stub headers for §2–§5, this plan, and `template_mapping.md`. PDF compiles.
  - ⚠️ For Claude #2: §2 must open with a "Separation of Access Timescales"
    subsection (template analogue of DES-LOC §2.1 "Separation of Optimizer-State
    Timescales") before the architecture/algorithm subsection. Use the
    half-life-style intuition but for *memory residency* (hot intervals → HBM),
    not optimizer momenta. Cite the tiered-allocator waterfall + the "from C, the
    good example" lineage (RapidStore `snapshot_edges` → TieredAllocator).
- Claude #2 — **§2 The Philemon-TSH System — DONE.** Replaced the §2 stub with
  two subsections: §2.1 "Separation of Access Timescales" (temporal density
  $\rho(P)$ + access frequency as the hot/cold residency signal; flush-time
  density-adaptive partitioning + runtime hotness migration; RapidStore →
  TieredAllocator lineage) and §2.2 "The Philemon-TSH Architecture" (tiered
  allocator waterfall + lock-free touch + slab; temporal bridge ingest/flush +
  dual-sorted interval index + LevelDB/Thrust scan; augmented + segmented
  selection index O(log P + k); seqlock wait-free read path + async migration),
  with **Algorithm 1** (Flush / Query / MigrateSweep). Added `leveldb` + `cccl`
  bib keys. PDF compiles (6 pages), no undefined refs.
  - ⚠️ For Claude #3: §3 states the *correctness & complexity* guarantees that
    §2 asserts operationally. Lead with the invariants (partitions start-sorted
    on flush; closed intervals; span-max ≥ every member ts_hi), then the
    selection theorem (`overlaps` is O(log P + k); the `span_max < lo` prune is
    sound), the segmented-index amortization (per-flush O(M log M), compaction at
    threshold 8 ⇒ no O(N² log N) cliff), the seqlock read-consistency statement,
    and the validation harness (10M / 32k / 20k cross-checks; 900/900 runtime
    samples; 2.1M-query TSan run). Use the `theorem`/`lemma`/`assumption` envs
    already in the preamble. Cross-reference Algorithm 1 line numbers where useful.
- Claude #3 — §3 Correctness & Complexity Guarantees — PENDING.
- Claude #4 — §4 Experimental Design — PENDING.
- Claude #5 — §5 Evaluation (+ short Related Work & Conclusion) — PENDING.
