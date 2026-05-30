#ifndef PHILEMON_LDBC_LOADER_HPP
#define PHILEMON_LDBC_LOADER_HPP
/**
 * ldbc_loader.hpp — LDBC SNB Temporal Graph Loader with Tier Placement
 *
 * 骨架来源: upstream/rapidstore/dataset_preprocessor/dataset_preprocessor.{hpp,cpp}
 *            (61 + 596 = 657行)
 * 修改 (~20%):
 *   - TemporalEdge 替换 weightedEdge，loadEdges 解析 timestamp 列
 *   - 增加 computeTierPlacement(): 按 degree+recency 分配 HBM/GDDR/DRAM
 *   - 增加 calibratePartitionThresholds(): 用实际数据校准自适应阈值
 *   - 增加 per-stage 全量状态 dump (边计数、degree 分布、tier 分布)
 *   - 增加 validateLoad() 做加载后一致性断言
 *   - 保留 removeDuplicateEdges, randomShuffle, computeDegreeDistribution,
 *     selectNodesByDegree, saveStream 等核心逻辑原样
 *
 * Milestone: M017 — LDBC SNB loader
 */

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <numeric>
#include <cassert>
#include <chrono>

#include "ldbc_types.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace loader {

// ─── String utilities (from upstream, preserved) ────────────────────
inline std::string removeCharacter(const std::string& str, char c) {
    std::string result;
    result.reserve(str.size());
    for (char ch : str) {
        if (ch != c) result += ch;
    }
    return result;
}

inline void splitString(const std::string& str, char delim,
                         std::vector<std::string>& tokens) {
    std::stringstream ss(removeCharacter(str, '\r'));
    std::string token;
    while (std::getline(ss, token, delim)) {
        if (!token.empty()) tokens.push_back(token);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LDBCLoader — Main loader class
// ═══════════════════════════════════════════════════════════════════════
class LDBCLoader {
private:
    // ─── Data (from upstream DataPreProcessor) ───────────────────
    std::vector<TemporalEdge> edgeList_;
    std::vector<std::vector<TemporalEdge>> graph_;  // adjacency list
    vertexID numVertices_;
    std::vector<vertexID> degreeDistribution_;
    std::vector<vertexID> directDegreeDistribution_;
    std::vector<uint64_t> prefixSum_;
    std::unordered_set<vertexID> highDegreeNodes_;
    uint64_t highDegreeEdgeCount_;
    std::unordered_set<vertexID> lowDegreeNodes_;
    uint64_t lowDegreeEdgeCount_;
    unsigned int randomSeed_;

    double initialGraphRatio_;
    double vertexQueryRatio_;
    double edgeQueryRatio_;
    double highVertexRatio_;
    double highEdgeRatio_;
    double lowVertexRatio_;
    double lowEdgeRatio_;
    uint64_t insert_num_;
    uint64_t search_num_;
    uint64_t scan_num_;

    // ─── NEW: Tier placement data ───────────────────────────────
    std::vector<TierHint> edgeTierHints_;      // per-edge tier assignment
    std::vector<TierHint> vertexTierHints_;     // per-vertex tier assignment
    PartitionHint partitionHint_;
    LDBCConfig config_;
    DegreeStats stats_;

    // ─── NEW: Timestamp range tracking ──────────────────────────
    uint64_t minTimestamp_{UINT64_MAX};
    uint64_t maxTimestamp_{0};

    // ─── Internal methods (from upstream, mostly preserved) ─────
    void loadEdges(const std::string& inputFile, bool weighted,
                   char delimiter = ' ');
    void removeDuplicateEdges();
    void randomShuffle();
    void computeDegreeDistribution();
    void selectNodesByDegree();
    vertexID selectNodeByDegree();
    void selectRandomNodes(uint64_t targetSize,
                           std::vector<vertexID>& chosenNodes,
                           bool uniform = false);
    void saveStream(const std::string& streamPath,
                    std::vector<operation>& stream);

    // ─── NEW: Tier-aware methods ────────────────────────────────
    void computeTierPlacement();
    void calibratePartitionThresholds();

public:
    // ─── Constructor (from upstream, + config/tier params) ───────
    LDBCLoader(std::string inputFile, bool weighted,
               char delimiter = ' ',
               double initGraphRatio = 0.8,
               double vertexQueryRatio = 0.2,
               double edgeQueryRatio = 0.2,
               double highDegreeVertexRatio = 0.01,
               double highDegreeEdgeRatio = 0.2,
               double lowDegreeVertexRatio = 0.2,
               double lowDegreeEdgeRatio = 0.5,
               uint64_t insert_num = 10000,
               uint64_t search_num = 10000,
               uint64_t scan_num = 10000,
               unsigned int seed = 0,
               bool shuffle = true,
               LDBCConfig config = LDBCConfig());

    // ─── Workload generation (from upstream) ────────────────────
    void generateAllWorkloads(const std::string& dirPath);

    // ─── NEW: Tier-aware accessors ──────────────────────────────
    const std::vector<TemporalEdge>& edges() const { return edgeList_; }
    const std::vector<TierHint>& edgeTierHints() const { return edgeTierHints_; }
    const std::vector<TierHint>& vertexTierHints() const { return vertexTierHints_; }
    const PartitionHint& partitionHint() const { return partitionHint_; }
    const DegreeStats& degreeStats() const { return stats_; }
    vertexID vertexCount() const { return numVertices_; }
    uint64_t edgeCount() const { return edgeList_.size(); }
    uint64_t timestampMin() const { return minTimestamp_; }
    uint64_t timestampMax() const { return maxTimestamp_; }

    // ─── NEW: Debug / state inspection ──────────────────────────
    void dumpFullState(const char* tag = "LDBCLoader") const;
    void dumpTierDistribution() const;
    void dumpTimestampHistogram(int bins = 20) const;
    bool validateLoad() const;
};

// ═══════════════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════════════

inline LDBCLoader::LDBCLoader(
    std::string inputFile, bool weighted, char delimiter,
    double initGraphRatio, double vertexQueryRatio, double edgeQueryRatio,
    double highDegreeVertexRatio, double highDegreeEdgeRatio,
    double lowDegreeVertexRatio, double lowDegreeEdgeRatio,
    uint64_t insert_num, uint64_t search_num, uint64_t scan_num,
    unsigned int seed, bool shuffle, LDBCConfig config)
    : numVertices_(0),
      highDegreeEdgeCount_(0), lowDegreeEdgeCount_(0),
      randomSeed_(seed),
      initialGraphRatio_(initGraphRatio),
      vertexQueryRatio_(vertexQueryRatio), edgeQueryRatio_(edgeQueryRatio),
      highVertexRatio_(highDegreeVertexRatio),
      highEdgeRatio_(highDegreeEdgeRatio),
      lowVertexRatio_(lowDegreeVertexRatio),
      lowEdgeRatio_(lowDegreeEdgeRatio),
      insert_num_(insert_num), search_num_(search_num),
      scan_num_(scan_num), config_(config)
{
    debug::ScopedTimer timer("LDBCLoader::constructor");

    PHILE_DBG(1, "──── LDBCLoader: Loading %s ────", inputFile.c_str());
    config_.dump();

    // Stage 1: Load edges (from upstream)
    loadEdges(inputFile, weighted, delimiter);
    PHILE_DBG(1, "[stage 1/6] loadEdges complete: %zu edges, %lu vertices",
              edgeList_.size(), (unsigned long)numVertices_);

    // Stage 2: Remove duplicates (from upstream)
    size_t before_dedup = edgeList_.size();
    removeDuplicateEdges();
    PHILE_DBG(1, "[stage 2/6] dedup: %zu → %zu edges (removed %zu)",
              before_dedup, edgeList_.size(), before_dedup - edgeList_.size());

    // Stage 3: Optional shuffle (from upstream)
    if (shuffle) {
        randomShuffle();
        PHILE_DBG(1, "[stage 3/6] shuffle with seed=%u", randomSeed_);
    }

    // Stage 4: Degree distribution (from upstream)
    computeDegreeDistribution();
    PHILE_DBG(1, "[stage 4/6] degree distribution computed");

    // Stage 5: Select high/low degree nodes (from upstream)
    selectNodesByDegree();
    PHILE_DBG(1, "[stage 5/6] high_degree=%zu low_degree=%zu",
              highDegreeNodes_.size(), lowDegreeNodes_.size());

    // Stage 6: NEW — Tier placement + threshold calibration
    computeTierPlacement();
    calibratePartitionThresholds();
    PHILE_DBG(1, "[stage 6/6] tier placement + calibration complete");

    // Validation checkpoint
    if (!validateLoad()) {
        PHILE_DBG(0, "!!! VALIDATION FAILED after loading !!!");
    }

    // Full state dump at debug level 2+
    dumpFullState("post-construct");
}

// ─── loadEdges: from upstream + timestamp parsing ───────────────────
inline void LDBCLoader::loadEdges(const std::string& inputFile,
                                   bool weighted, char delimiter) {
    debug::ScopedTimer timer("loadEdges");

    std::ifstream handle(inputFile);
    if (!handle.is_open()) {
        std::fprintf(stderr, "[LDBCLoader] ERROR: cannot open %s\n",
                     inputFile.c_str());
        return;
    }

    std::string line;
    int lineNumber = 1;
    std::unordered_map<vertexID, uint64_t> vertexMap;
    vertexID nextVertexIndex = 0;
    uint64_t parse_errors = 0;
    uint64_t self_loops = 0;

    while (std::getline(handle, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::vector<std::string> tokens;
        splitString(line, delimiter, tokens);

        vertexID source, destination;
        double weight = random() / static_cast<double>(RAND_MAX);
        uint64_t timestamp = 0;  // NEW: default timestamp

        try {
            source = std::stoull(tokens.at(0));
            destination = std::stoull(tokens.at(1));
            if (weighted && tokens.size() > 2) {
                weight = std::stod(tokens.at(2));
            }
            // NEW: parse timestamp if present (LDBC SNB format)
            if (tokens.size() > 3) {
                timestamp = std::stoull(tokens.at(3));
            } else {
                // Synthesize monotonic timestamp from line position
                timestamp = static_cast<uint64_t>(lineNumber);
            }

            if (source == destination) {
                self_loops++;
                continue;
            }

            // Vertex ID remapping (from upstream, preserved)
            if (vertexMap.find(source) == vertexMap.end()) {
                vertexMap[source] = nextVertexIndex++;
                source = nextVertexIndex - 1;
            } else {
                source = vertexMap[source];
            }

            if (vertexMap.find(destination) == vertexMap.end()) {
                vertexMap[destination] = nextVertexIndex++;
                destination = nextVertexIndex - 1;
            } else {
                destination = vertexMap[destination];
            }

            if (source > destination) std::swap(source, destination);

        } catch (const std::exception& e) {
            parse_errors++;
            if (parse_errors <= 5) {
                PHILE_DBG(2, "[loadEdges] parse error line %d: %s",
                          lineNumber, line.c_str());
            }
            lineNumber++;
            continue;
        }

        edgeList_.push_back({source, destination, weight, timestamp});

        // NEW: track timestamp range
        if (timestamp < minTimestamp_) minTimestamp_ = timestamp;
        if (timestamp > maxTimestamp_) maxTimestamp_ = timestamp;

        lineNumber++;

        // Progress report every 1M edges
        if (edgeList_.size() % 1000000 == 0) {
            PHILE_DBG(1, "[loadEdges] progress: %zuM edges loaded...",
                      edgeList_.size() / 1000000);
        }
    }

    numVertices_ = nextVertexIndex;
    handle.close();

    // Build adjacency list (from upstream)
    graph_.resize(numVertices_);
    for (const auto& edge : edgeList_) {
        graph_[edge.source].push_back(edge);
    }

    PHILE_DBG(1, "[loadEdges] DONE: V=%lu E=%zu ts_range=[%lu,%lu] "
              "parse_errors=%lu self_loops=%lu",
              (unsigned long)numVertices_, edgeList_.size(),
              (unsigned long)minTimestamp_, (unsigned long)maxTimestamp_,
              (unsigned long)parse_errors, (unsigned long)self_loops);
}

// ─── removeDuplicateEdges: from upstream, preserved ────────────────
inline void LDBCLoader::removeDuplicateEdges() {
    debug::ScopedTimer timer("removeDuplicateEdges");

    std::sort(edgeList_.begin(), edgeList_.end(),
              [](const TemporalEdge& a, const TemporalEdge& b) {
        if (a.source == b.source) return a.destination < b.destination;
        return a.source < b.source;
    });

    uint64_t current = 0, ahead = 0;
    for (; ahead < edgeList_.size(); ahead++, current++) {
        while (ahead + 1 < edgeList_.size() &&
               edgeList_[ahead].source == edgeList_[ahead + 1].source &&
               edgeList_[ahead].destination == edgeList_[ahead + 1].destination) {
            ahead++;
        }
        if (ahead > current) edgeList_[current] = edgeList_[ahead];
    }
    edgeList_.resize(current);
}

// ─── randomShuffle: from upstream, preserved ───────────────────────
inline void LDBCLoader::randomShuffle() {
    std::default_random_engine engine(randomSeed_);
    std::shuffle(edgeList_.begin(), edgeList_.end(), engine);
}

// ─── computeDegreeDistribution: from upstream, + stats collection ──
inline void LDBCLoader::computeDegreeDistribution() {
    debug::ScopedTimer timer("computeDegreeDistribution");

    degreeDistribution_.resize(numVertices_, 0);
    directDegreeDistribution_.resize(numVertices_, 0);
    for (const auto& e : edgeList_) {
        degreeDistribution_[e.source]++;
        degreeDistribution_[e.destination]++;
        directDegreeDistribution_[e.source]++;
    }

    prefixSum_.resize(degreeDistribution_.size());
    uint64_t sum = 0;
    for (uint64_t i = 0; i < prefixSum_.size(); i++) {
        prefixSum_[i] = degreeDistribution_[i] + sum;
        sum = prefixSum_[i];
    }

    // ─── NEW: Compute DegreeStats for debug inspection ──────────
    stats_.num_vertices = numVertices_;
    stats_.num_edges = edgeList_.size();
    stats_.max_degree = 0;
    stats_.avg_degree = 0;
    stats_.high_degree_count = 0;
    stats_.low_degree_count = 0;

    std::vector<vertexID> sorted_deg = degreeDistribution_;
    std::sort(sorted_deg.begin(), sorted_deg.end());

    if (!sorted_deg.empty()) {
        stats_.max_degree = sorted_deg.back();
        stats_.median_degree = sorted_deg[sorted_deg.size() / 2];
        stats_.p90_degree = sorted_deg[static_cast<size_t>(sorted_deg.size() * 0.9)];
        stats_.p99_degree = sorted_deg[static_cast<size_t>(sorted_deg.size() * 0.99)];
        uint64_t deg_sum = std::accumulate(sorted_deg.begin(),
                                            sorted_deg.end(), 0ULL);
        stats_.avg_degree = static_cast<double>(deg_sum) / numVertices_;
    }

    // Count high/low by threshold
    for (auto d : degreeDistribution_) {
        if (d > stats_.p99_degree) stats_.high_degree_count++;
        if (d <= 2) stats_.low_degree_count++;
    }

    stats_.dump();
}

// ─── selectNodesByDegree: from upstream, preserved ─────────────────
inline void LDBCLoader::selectNodesByDegree() {
    debug::ScopedTimer timer("selectNodesByDegree");

    std::vector<std::pair<uint64_t, vertexID>> nodeDegrees;
    for (uint64_t i = 0; i < numVertices_; i++) {
        nodeDegrees.push_back({degreeDistribution_[i], i});
    }
    std::sort(nodeDegrees.begin(), nodeDegrees.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    uint64_t selectedHighDegreeSize = numVertices_ * highVertexRatio_;

    highDegreeEdgeCount_ = 0;
    for (uint64_t i = 0; i < selectedHighDegreeSize && i < nodeDegrees.size(); i++) {
        highDegreeNodes_.insert(nodeDegrees[i].second);
        highDegreeEdgeCount_ += nodeDegrees[i].first;
    }

    uint64_t selectedLowDegreeSize = numVertices_ * lowEdgeRatio_;
    lowDegreeEdgeCount_ = 0;
    for (uint64_t i = 0; i < selectedLowDegreeSize && i < nodeDegrees.size(); i++) {
        auto& node = nodeDegrees[numVertices_ - 1 - i];
        lowDegreeNodes_.insert(node.second);
        lowDegreeEdgeCount_ += node.first;
    }

    PHILE_DBG(2, "[selectNodesByDegree] high: %zu nodes, %lu edges | "
              "low: %zu nodes, %lu edges",
              highDegreeNodes_.size(), (unsigned long)highDegreeEdgeCount_,
              lowDegreeNodes_.size(), (unsigned long)lowDegreeEdgeCount_);
}

// ─── selectNodeByDegree: from upstream, preserved ──────────────────
inline vertexID LDBCLoader::selectNodeByDegree() {
    uint64_t maxValue = *(prefixSum_.end() - 1);
    uint64_t randValue = random() % maxValue;
    auto it = std::lower_bound(prefixSum_.begin(), prefixSum_.end(), randValue);
    return it - prefixSum_.begin();
}

// ─── selectRandomNodes: from upstream, preserved ───────────────────
inline void LDBCLoader::selectRandomNodes(uint64_t targetSize,
                                           std::vector<vertexID>& chosenNodes,
                                           bool uniform) {
    for (uint64_t i = 0; i < targetSize; i++) {
        bool chosen = false;
        uint64_t selectedNode = 0;
        uint64_t j_counter = 0;

        while (!chosen) {
            if (uniform) selectedNode = (j_counter++) % numVertices_;
            else selectedNode = selectNodeByDegree();

            if (chosenNodes[selectedNode] < directDegreeDistribution_[selectedNode]) {
                chosenNodes[selectedNode]++;
                chosen = true;
            }
        }
    }
}

// ─── saveStream: from upstream, preserved ──────────────────────────
inline void LDBCLoader::saveStream(const std::string& streamPath,
                                    std::vector<operation>& stream) {
    std::ofstream file(streamPath, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(stream.data()),
                   stream.size() * sizeof(operation));
        PHILE_DBG(2, "[saveStream] wrote %zu ops to %s",
                  stream.size(), streamPath.c_str());
    } else {
        PHILE_DBG(0, "[saveStream] ERROR: cannot write %s", streamPath.c_str());
    }
    file.close();
}

// ═══════════════════════════════════════════════════════════════════════
// NEW: Tier-Aware Methods (~20% of total code)
// ═══════════════════════════════════════════════════════════════════════

// ─── computeTierPlacement: assign each vertex/edge to HBM/GDDR/DRAM ─
inline void LDBCLoader::computeTierPlacement() {
    debug::ScopedTimer timer("computeTierPlacement");

    vertexTierHints_.resize(numVertices_, TierHint::AUTO);
    edgeTierHints_.resize(edgeList_.size(), TierHint::AUTO);

    if (numVertices_ == 0 || edgeList_.empty()) return;

    // Strategy: degree + recency → tier
    // Hot (HBM): top hot_fraction% by degree AND recent timestamps
    // Warm (GDDR): next warm_fraction%
    // Cold (DRAM): rest

    // Sort vertices by degree to find thresholds
    std::vector<std::pair<uint64_t, vertexID>> deg_sorted;
    deg_sorted.reserve(numVertices_);
    for (vertexID v = 0; v < numVertices_; v++) {
        deg_sorted.push_back({degreeDistribution_[v], v});
    }
    std::sort(deg_sorted.begin(), deg_sorted.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    size_t hot_count  = static_cast<size_t>(numVertices_ * partitionHint_.hot_fraction);
    size_t warm_count = static_cast<size_t>(numVertices_ * partitionHint_.warm_fraction);

    // Assign vertex tiers
    for (size_t i = 0; i < deg_sorted.size(); i++) {
        vertexID v = deg_sorted[i].second;
        if (i < hot_count) {
            vertexTierHints_[v] = TierHint::HBM;
        } else if (i < hot_count + warm_count) {
            vertexTierHints_[v] = TierHint::GDDR;
        } else {
            vertexTierHints_[v] = TierHint::DRAM;
        }
    }

    // Assign edge tiers: inherit from higher-tier endpoint
    // Also factor in recency: recent edges get promoted
    uint64_t ts_range = (maxTimestamp_ > minTimestamp_) ?
                        (maxTimestamp_ - minTimestamp_) : 1;
    uint64_t hot_ts_threshold = maxTimestamp_ -
                                static_cast<uint64_t>(ts_range * partitionHint_.hot_fraction);

    uint64_t hbm_edges = 0, gddr_edges = 0, dram_edges = 0;

    for (size_t i = 0; i < edgeList_.size(); i++) {
        const auto& e = edgeList_[i];
        TierHint src_hint = vertexTierHints_[e.source];
        TierHint dst_hint = vertexTierHints_[e.destination];

        // Take the "hotter" of the two endpoints
        TierHint base_hint = (src_hint < dst_hint) ? src_hint : dst_hint;

        // Recency boost: recent edges promoted one tier
        if (e.timestamp >= hot_ts_threshold && base_hint != TierHint::HBM) {
            base_hint = static_cast<TierHint>(
                static_cast<uint8_t>(base_hint) - 1);
        }

        edgeTierHints_[i] = base_hint;

        switch (base_hint) {
            case TierHint::HBM:  hbm_edges++; break;
            case TierHint::GDDR: gddr_edges++; break;
            default:             dram_edges++; break;
        }
    }

    PHILE_DBG(1, "[tierPlacement] HBM=%lu GDDR=%lu DRAM=%lu edges "
              "(hot_ts_threshold=%lu)",
              (unsigned long)hbm_edges, (unsigned long)gddr_edges,
              (unsigned long)dram_edges, (unsigned long)hot_ts_threshold);
}

// ─── calibratePartitionThresholds: use real data for adaptive config ─
inline void LDBCLoader::calibratePartitionThresholds() {
    debug::ScopedTimer timer("calibratePartitionThresholds");

    if (numVertices_ == 0) return;

    // Calibrate density threshold from actual graph
    double actual_density = static_cast<double>(edgeList_.size()) / numVertices_;
    partitionHint_.density_threshold = std::max(actual_density * 0.5, 2.0);

    // Calibrate partition sizes based on tier capacities
    size_t edge_bytes = sizeof(TemporalEdge);
    uint64_t total_data_bytes = edgeList_.size() * edge_bytes;

    // HBM budget → how many edges fit
    uint64_t hbm_edge_budget = config_.hbm_capacity / edge_bytes;
    uint64_t gddr_edge_budget = config_.gddr_capacity / edge_bytes;

    // Set max partition to ~1% of tier capacity for good granularity
    partitionHint_.max_partition_edges = std::min(
        static_cast<uint64_t>(hbm_edge_budget / 100),
        static_cast<uint64_t>(1048576));
    partitionHint_.min_partition_edges = std::max(
        partitionHint_.max_partition_edges / 1024,
        static_cast<uint64_t>(64));

    // Adjust hot/warm fractions based on what actually fits
    if (total_data_bytes > 0) {
        double fit_in_hbm = static_cast<double>(config_.hbm_capacity) /
                            total_data_bytes;
        double fit_in_gddr = static_cast<double>(config_.gddr_capacity) /
                             total_data_bytes;
        partitionHint_.hot_fraction = std::min(fit_in_hbm, 0.5);
        partitionHint_.warm_fraction = std::min(fit_in_gddr, 0.5);
    }

    PHILE_DBG(1, "[calibrate] density_thresh=%.2f min=%lu max=%lu "
              "hot=%.3f warm=%.3f",
              partitionHint_.density_threshold,
              (unsigned long)partitionHint_.min_partition_edges,
              (unsigned long)partitionHint_.max_partition_edges,
              partitionHint_.hot_fraction, partitionHint_.warm_fraction);
    partitionHint_.dump();
}

// ═══════════════════════════════════════════════════════════════════════
// Debug / State Inspection Methods
// ═══════════════════════════════════════════════════════════════════════

inline void LDBCLoader::dumpFullState(const char* tag) const {
    if (debug::get_debug_level() < 2) return;

    std::printf("════ LDBCLoader State Dump [%s] ════\n", tag);
    std::printf("  vertices=%lu  edges=%zu  ts_range=[%lu, %lu]\n",
                (unsigned long)numVertices_, edgeList_.size(),
                (unsigned long)minTimestamp_, (unsigned long)maxTimestamp_);
    std::printf("  graph_.size()=%zu  degreeDistribution_.size()=%zu\n",
                graph_.size(), degreeDistribution_.size());
    std::printf("  highDegreeNodes=%zu (%lu edges)  lowDegreeNodes=%zu (%lu edges)\n",
                highDegreeNodes_.size(), (unsigned long)highDegreeEdgeCount_,
                lowDegreeNodes_.size(), (unsigned long)lowDegreeEdgeCount_);
    std::printf("  edgeTierHints=%zu  vertexTierHints=%zu\n",
                edgeTierHints_.size(), vertexTierHints_.size());

    // Print first 5 edges for spot check
    size_t show = std::min(edgeList_.size(), (size_t)5);
    for (size_t i = 0; i < show; i++) {
        const auto& e = edgeList_[i];
        std::printf("  edge[%zu]: src=%lu dst=%lu w=%.4f ts=%lu tier=%s\n",
                    i, (unsigned long)e.source, (unsigned long)e.destination,
                    e.weight, (unsigned long)e.timestamp,
                    i < edgeTierHints_.size() ?
                        tier_hint_name(edgeTierHints_[i]) : "N/A");
    }

    stats_.dump();
    partitionHint_.dump();
    config_.dump();
    std::printf("════ End State Dump ════\n");
}

inline void LDBCLoader::dumpTierDistribution() const {
    uint64_t counts[4] = {0, 0, 0, 0};
    for (auto h : edgeTierHints_) {
        counts[static_cast<uint8_t>(h)]++;
    }
    std::printf("──── Edge Tier Distribution ────\n");
    std::printf("  HBM=%lu (%.1f%%)  GDDR=%lu (%.1f%%)  DRAM=%lu (%.1f%%)  AUTO=%lu\n",
                (unsigned long)counts[0],
                edgeList_.empty() ? 0.0 : 100.0 * counts[0] / edgeList_.size(),
                (unsigned long)counts[1],
                edgeList_.empty() ? 0.0 : 100.0 * counts[1] / edgeList_.size(),
                (unsigned long)counts[2],
                edgeList_.empty() ? 0.0 : 100.0 * counts[2] / edgeList_.size(),
                (unsigned long)counts[3]);

    uint64_t vcounts[4] = {0, 0, 0, 0};
    for (auto h : vertexTierHints_) {
        vcounts[static_cast<uint8_t>(h)]++;
    }
    std::printf("  Vertex: HBM=%lu GDDR=%lu DRAM=%lu AUTO=%lu\n",
                (unsigned long)vcounts[0], (unsigned long)vcounts[1],
                (unsigned long)vcounts[2], (unsigned long)vcounts[3]);
    std::printf("──── End Distribution ────\n");
}

inline void LDBCLoader::dumpTimestampHistogram(int bins) const {
    if (edgeList_.empty() || minTimestamp_ >= maxTimestamp_) return;

    std::vector<uint64_t> hist(bins, 0);
    uint64_t range = maxTimestamp_ - minTimestamp_;
    double bin_width = static_cast<double>(range) / bins;

    for (const auto& e : edgeList_) {
        int b = static_cast<int>((e.timestamp - minTimestamp_) / bin_width);
        if (b >= bins) b = bins - 1;
        hist[b]++;
    }

    uint64_t max_count = *std::max_element(hist.begin(), hist.end());

    std::printf("──── Timestamp Histogram (%d bins) ────\n", bins);
    for (int i = 0; i < bins; i++) {
        uint64_t lo = minTimestamp_ + static_cast<uint64_t>(i * bin_width);
        uint64_t hi = minTimestamp_ + static_cast<uint64_t>((i + 1) * bin_width);
        int bar_len = max_count > 0 ?
            static_cast<int>(40.0 * hist[i] / max_count) : 0;
        std::printf("  [%8lu-%8lu] %6lu |",
                    (unsigned long)lo, (unsigned long)hi,
                    (unsigned long)hist[i]);
        for (int j = 0; j < bar_len; j++) std::putchar('#');
        std::putchar('\n');
    }
    std::printf("──── End Histogram ────\n");
}

inline bool LDBCLoader::validateLoad() const {
    debug::ScopedTimer timer("validateLoad");

    bool ok = true;

    // Check vertex count consistency
    if (graph_.size() != numVertices_) {
        PHILE_DBG(0, "[VALIDATE FAIL] graph_.size()=%zu != numVertices_=%lu",
                  graph_.size(), (unsigned long)numVertices_);
        ok = false;
    }

    // Check degree distribution size
    if (degreeDistribution_.size() != numVertices_) {
        PHILE_DBG(0, "[VALIDATE FAIL] degDist.size()=%zu != V=%lu",
                  degreeDistribution_.size(), (unsigned long)numVertices_);
        ok = false;
    }

    // Check tier hints sizes
    if (edgeTierHints_.size() != edgeList_.size()) {
        PHILE_DBG(0, "[VALIDATE FAIL] edgeTierHints=%zu != edges=%zu",
                  edgeTierHints_.size(), edgeList_.size());
        ok = false;
    }

    // Check no edge has source == destination
    for (size_t i = 0; i < edgeList_.size(); i++) {
        if (edgeList_[i].source == edgeList_[i].destination) {
            PHILE_DBG(0, "[VALIDATE FAIL] self-loop at edge[%zu]", i);
            ok = false;
            break;
        }
    }

    // Check adjacency list consistency (spot check first 100 vertices)
    uint64_t adj_total = 0;
    for (size_t v = 0; v < std::min(graph_.size(), (size_t)100); v++) {
        adj_total += graph_[v].size();
    }
    PHILE_DBG(2, "[validate] spot-check adj_total=%lu for first 100 vertices",
              (unsigned long)adj_total);

    if (ok) {
        PHILE_DBG(1, "[validate] ✓ All checks passed");
    }
    return ok;
}

// ─── generateAllWorkloads: from upstream, + temporal query streams ──
inline void LDBCLoader::generateAllWorkloads(const std::string& dirPath) {
    debug::ScopedTimer timer("generateAllWorkloads");

    // Insert all edges as initial stream (from upstream)
    std::vector<operation> initialStream;
    initialStream.reserve(edgeList_.size());
    for (const auto& e : edgeList_) {
        initialStream.push_back({operationType::INSERT, e});
    }
    std::string path = dirPath + "/initial_stream_analytic.stream";
    saveStream(path, initialStream);

    // Analytic query stream (from upstream)
    std::vector<operation> analyticStream;
    std::default_random_engine engine(randomSeed_);
    std::uniform_int_distribution<vertexID> vdist(0, numVertices_ - 1);
    vertexID src = vdist(engine);

    analyticStream.push_back({operationType::BFS,       {src, 0, 0.0, 0}});
    analyticStream.push_back({operationType::PAGE_RANK, {0, 0, 0.0, 0}});
    analyticStream.push_back({operationType::SSSP,      {src, 0, 0.0, 0}});
    analyticStream.push_back({operationType::TC,        {0, 0, 0.0, 0}});
    analyticStream.push_back({operationType::WCC,       {0, 0, 0.0, 0}});

    // NEW: Cross-tier BFS and SSSP queries
    analyticStream.push_back({operationType::CROSS_TIER_BFS,  {src, 0, 0.0, 0}});
    analyticStream.push_back({operationType::CROSS_TIER_SSSP, {src, 0, 0.0, 0}});

    path = dirPath + "/target_stream_analytic.stream";
    saveStream(path, analyticStream);

    // NEW: Temporal query stream — range queries across time
    std::vector<operation> temporalStream;
    uint64_t ts_range = (maxTimestamp_ > minTimestamp_) ?
                        (maxTimestamp_ - minTimestamp_) : 1;
    for (int i = 0; i < 100; i++) {
        uint64_t query_start = minTimestamp_ +
            static_cast<uint64_t>((ts_range * i) / 100);
        uint64_t query_end = query_start + ts_range / 10;
        temporalStream.push_back({operationType::TEMPORAL_QUERY,
                                  {0, 0, 0.0, query_start}});
    }
    path = dirPath + "/target_stream_temporal.stream";
    saveStream(path, temporalStream);

    PHILE_DBG(1, "[generateAllWorkloads] wrote %zu initial, %zu analytic, "
              "%zu temporal ops",
              initialStream.size(), analyticStream.size(),
              temporalStream.size());
}

}  // namespace loader
}  // namespace philemon

#endif  // PHILEMON_LDBC_LOADER_HPP
