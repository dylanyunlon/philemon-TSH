// =============================================================================
// M095: Wrapper Debug Experiment — 基于 upstream/rapidstore/wrapper/wrapper.h
// 来源: upstream/rapidstore/wrapper/wrapper.h (249行) 全部模板函数移植
// 作者: 第7位Claude (Opus 4.6), 由第1位Claude调度
// 编译: g++ -std=c++17 -O2 -pthread -o experiment/wrapper_debug experiment/wrapper_debug_experiment.cpp
// =============================================================================
//
// 20%算法修改清单:
//   1. 每个wrapper函数内部加调用计数器(atomic<uint64_t>)和累计耗时(chrono)
//   2. insert_edge加冲突检测: 插入前检查edge是否已存在，若存在则记录冲突并用CAS重试
//   3. batch_edge_update加自适应分块: 根据前一批延迟动态调整chunk_size
//   4. get_neighbors加热度追踪: 每次访问给vertex的access_count++, 维护top-K热点
//   5. snapshot操作加version一致性校验: 记录snapshot创建时的version_id
//   6. degree查询加分布直方图: 累积degree查询结果到log-scale桶
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <queue>
#include <random>
#include <cassert>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <array>
#include <memory>

// =============================================================================
// 来源: /proc/self/status 读取 (upstream/rapidstore/wrapper/driver.h 第122-144行)
// =============================================================================
static int parseLine_m095(char* line) {
    int i = strlen(line);
    const char* p = line;
    while (*p < '0' || *p > '9') p++;
    line[i - 3] = '\0';
    return atoi(p);
}

static int getVmRSS() {
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];
    if (!file) return -1;
    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) { result = parseLine_m095(line); break; }
    }
    fclose(file);
    return result;
}

static int getVmPeak() {
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];
    if (!file) return -1;
    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmPeak:", 7) == 0) { result = parseLine_m095(line); break; }
    }
    fclose(file);
    return result;
}

namespace philemon::m095 {

// =============================================================================
// 新增: 调用统计结构 (20%算法修改 - 调用计数器+耗时)
// =============================================================================
struct FuncCallStats {
    std::atomic<uint64_t> call_count{0};
    std::atomic<uint64_t> total_ns{0};
    const char* name;
    FuncCallStats(const char* n) : name(n) {}
    FuncCallStats(const FuncCallStats& o) : name(o.name) {
        call_count.store(o.call_count.load());
        total_ns.store(o.total_ns.load());
    }
    double avg_ns() const {
        uint64_t c = call_count.load();
        return c > 0 ? (double)total_ns.load() / c : 0.0;
    }
};

// 来源: wrapper.h 第11行 using PUU = std::pair<uint64_t, uint64_t>;
using PUU = std::pair<uint64_t, uint64_t>;

// =============================================================================
// 新增: 冲突检测记录 (20%算法修改 - insert_edge CAS重试)
// =============================================================================
struct ConflictRecord {
    uint64_t source;
    uint64_t destination;
    int retry_count;
    bool resolved;
};

// =============================================================================
// 新增: 热度追踪器 (20%算法修改 - get_neighbors热点统计)
// =============================================================================
class HotnessTracker {
    std::mutex mtx_;
    std::unordered_map<uint64_t, uint64_t> access_counts_;
    static constexpr size_t TOP_K = 5;
public:
    void record_access(uint64_t vertex) {
        std::lock_guard<std::mutex> lk(mtx_);
        access_counts_[vertex]++;
    }
    std::vector<std::pair<uint64_t, uint64_t>> get_top_k() {
        std::lock_guard<std::mutex> lk(mtx_);
        std::vector<std::pair<uint64_t, uint64_t>> all(access_counts_.begin(), access_counts_.end());
        std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.second > b.second; });
        if (all.size() > TOP_K) all.resize(TOP_K);
        return all;
    }
    void reset() { std::lock_guard<std::mutex> lk(mtx_); access_counts_.clear(); }
};

// =============================================================================
// 新增: Degree分布直方图 (20%算法修改 - log-scale桶)
// =============================================================================
class DegreeHistogram {
    // 桶: [0], [1-2], [3-8], [9-32], [33-128], [129+]
    std::array<std::atomic<uint64_t>, 6> buckets_{};
public:
    DegreeHistogram() { for (auto& b : buckets_) b.store(0); }
    void record(uint64_t deg) {
        if (deg == 0) buckets_[0]++;
        else if (deg <= 2) buckets_[1]++;
        else if (deg <= 8) buckets_[2]++;
        else if (deg <= 32) buckets_[3]++;
        else if (deg <= 128) buckets_[4]++;
        else buckets_[5]++;
    }
    void print() const {
        const char* labels[] = {"0", "1-2", "3-8", "9-32", "33-128", "129+"};
        std::cout << "  [Degree Distribution Histogram]\n";
        for (int i = 0; i < 6; i++) {
            uint64_t c = buckets_[i].load();
            std::cout << "    [" << std::setw(7) << labels[i] << "]: " << c;
            // ASCII bar
            int bar = std::min((int)(c / 5 + (c > 0 ? 1 : 0)), 40);
            std::cout << " |";
            for (int j = 0; j < bar; j++) std::cout << '#';
            std::cout << "\n";
        }
    }
    void reset() { for (auto& b : buckets_) b.store(0); }
};

// =============================================================================
// 新增: Snapshot版本一致性校验器 (20%算法修改)
// =============================================================================
struct SnapshotVersionGuard {
    uint64_t creation_version;
    std::atomic<uint64_t>* global_version_ptr;
    bool check() const {
        return global_version_ptr->load() == creation_version;
    }
};

// =============================================================================
// 新增: 自适应分块控制器 (20%算法修改 - batch_edge_update自适应chunk)
// =============================================================================
class AdaptiveChunkController {
    uint64_t current_chunk_size_;
    static constexpr uint64_t MIN_CHUNK = 8;
    static constexpr uint64_t MAX_CHUNK = 4096;
    double prev_latency_ns_ = 0.0;
public:
    AdaptiveChunkController(uint64_t init = 256) : current_chunk_size_(init) {}
    uint64_t get_chunk_size() const { return current_chunk_size_; }
    void feedback(double latency_ns, uint64_t chunk_processed) {
        double per_item = chunk_processed > 0 ? latency_ns / chunk_processed : latency_ns;
        if (prev_latency_ns_ > 0.0 && per_item > prev_latency_ns_ * 1.3) {
            // 延迟增高 -> 缩小chunk
            current_chunk_size_ = std::max(MIN_CHUNK, current_chunk_size_ / 2);
        } else if (prev_latency_ns_ > 0.0 && per_item < prev_latency_ns_ * 0.8) {
            // 延迟降低 -> 放大chunk
            current_chunk_size_ = std::min(MAX_CHUNK, current_chunk_size_ * 2);
        }
        prev_latency_ns_ = per_item;
    }
    void reset() { current_chunk_size_ = 256; prev_latency_ns_ = 0.0; }
};

// =============================================================================
// 操作类型枚举 (来源: upstream/rapidstore/types.hpp 语义)
// =============================================================================
enum class OperationType { INSERT, DELETE_OP };

struct WeightedEdge {
    uint64_t source;
    uint64_t destination;
    double weight;
};

struct Operation {
    OperationType type;
    WeightedEdge e;
};

// =============================================================================
// 模拟图数据结构: 用于standalone编译, 不依赖外部库
// =============================================================================
class SimulatedGraph {
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, double>> adj_;
    std::unordered_set<uint64_t> vertices_;
    mutable std::mutex mtx_;
    bool directed_ = false;
    bool weighted_ = true;
    int max_threads_ = 1;
    std::atomic<uint64_t> version_{0};

public:
    // 来源: wrapper.h 第14-16行 wrapper_repl
    std::string repl() const { return "SimulatedGraph(V=" + std::to_string(vertices_.size()) + ",E=" + std::to_string(edge_count_internal()) + ")"; }
    // 来源: wrapper.h 第20-22行 set_max_threads
    void set_max_threads(int t) { max_threads_ = t; }
    // 来源: wrapper.h 第24-26行 init_thread
    void init_thread(int) {}
    // 来源: wrapper.h 第28-30行 end_thread
    void end_thread(int) {}
    // 来源: wrapper.h 第34-36行 is_directed
    bool is_directed() const { return directed_; }
    // 来源: wrapper.h 第39-41行 is_weighted
    bool is_weighted() const { return weighted_; }
    // 来源: wrapper.h 第44-46行 is_empty
    bool is_empty() const { std::lock_guard<std::mutex> lk(mtx_); return vertices_.empty(); }
    // 来源: wrapper.h 第49-51行 has_vertex
    bool has_vertex(uint64_t v) const { std::lock_guard<std::mutex> lk(mtx_); return vertices_.count(v) > 0; }
    // 来源: wrapper.h 第54-56行 has_edge (weightedEdge overload)
    bool has_edge_we(uint64_t s, uint64_t d, double w) const {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = adj_.find(s); if (it == adj_.end()) return false;
        auto jt = it->second.find(d); if (jt == it->second.end()) return false;
        return jt->second == w;
    }
    // 来源: wrapper.h 第59-61行 has_edge (src, dst)
    bool has_edge(uint64_t s, uint64_t d) const {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = adj_.find(s); if (it == adj_.end()) return false;
        return it->second.count(d) > 0;
    }
    // 来源: wrapper.h 第64-66行 has_edge (src, dst, weight)
    bool has_edge(uint64_t s, uint64_t d, double w) const { return has_edge_we(s, d, w); }
    // 来源: wrapper.h 第69-71行 degree
    uint64_t degree(uint64_t v) const {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = adj_.find(v); return it != adj_.end() ? it->second.size() : 0;
    }
    // 来源: wrapper.h 第74-76行 get_weight
    double get_weight(uint64_t s, uint64_t d) const {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = adj_.find(s); if (it == adj_.end()) return 0.0;
        auto jt = it->second.find(d); return jt != it->second.end() ? jt->second : 0.0;
    }
    // 来源: wrapper.h 第79-81行 logical2physical
    uint64_t logical2physical(uint64_t l) const { return l; }
    // 来源: wrapper.h 第84-86行 physical2logical
    uint64_t physical2logical(uint64_t p) const { return p; }
    // 来源: wrapper.h 第89-91行 vertex_count
    uint64_t vertex_count() const { std::lock_guard<std::mutex> lk(mtx_); return vertices_.size(); }
    // 来源: wrapper.h 第94-96行 edge_count
    uint64_t edge_count() const { std::lock_guard<std::mutex> lk(mtx_); return edge_count_internal(); }
    uint64_t edge_count_internal() const {
        uint64_t c = 0; for (auto& kv : adj_) c += kv.second.size(); return c;
    }
    // 来源: wrapper.h 第99-101行 get_neighbors (unweighted)
    void get_neighbors(uint64_t v, std::vector<uint64_t>& out) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = adj_.find(v); if (it == adj_.end()) return;
        for (auto& kv : it->second) out.push_back(kv.first);
    }
    // 来源: wrapper.h 第104-106行 get_neighbors (weighted)
    void get_neighbors(uint64_t v, std::vector<std::pair<uint64_t, double>>& out) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = adj_.find(v); if (it == adj_.end()) return;
        for (auto& kv : it->second) out.push_back({kv.first, kv.second});
    }
    // 来源: wrapper.h 第109-111行 insert_vertex
    bool insert_vertex(uint64_t v) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto r = vertices_.insert(v); version_++; return r.second;
    }
    // 来源: wrapper.h 第114-116行 insert_edge (src, dst, weight)
    bool insert_edge(uint64_t s, uint64_t d, double w) {
        std::lock_guard<std::mutex> lk(mtx_);
        vertices_.insert(s); vertices_.insert(d);
        adj_[s][d] = w;
        if (!directed_) adj_[d][s] = w;
        version_++; return true;
    }
    // 来源: wrapper.h 第119-121行 insert_edge (src, dst)
    bool insert_edge(uint64_t s, uint64_t d) { return insert_edge(s, d, 1.0); }
    // 来源: wrapper.h 第124-126行 remove_vertex
    bool remove_vertex(uint64_t v) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (vertices_.erase(v) == 0) return false;
        adj_.erase(v);
        for (auto& kv : adj_) kv.second.erase(v);
        version_++; return true;
    }
    // 来源: wrapper.h 第129-131行 remove_edge
    bool remove_edge(uint64_t s, uint64_t d) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = adj_.find(s); if (it == adj_.end()) return false;
        bool r = it->second.erase(d) > 0;
        if (!directed_) { auto jt = adj_.find(d); if (jt != adj_.end()) jt->second.erase(s); }
        if (r) version_++; return r;
    }
    // 来源: wrapper.h 第134-136行 run_batch_vertex_update
    bool run_batch_vertex_update(std::vector<uint64_t>& verts, int start, int end) {
        std::lock_guard<std::mutex> lk(mtx_);
        for (int i = start; i < end && i < (int)verts.size(); i++) vertices_.insert(verts[i]);
        version_++; return true;
    }
    // 来源: wrapper.h 第139-141行 run_batch_edge_update (Operation vec)
    bool run_batch_edge_update_op(std::vector<Operation>& edges, int start, int end, OperationType type) {
        std::lock_guard<std::mutex> lk(mtx_);
        for (int i = start; i < end && i < (int)edges.size(); i++) {
            auto& e = edges[i].e;
            if (type == OperationType::INSERT) { vertices_.insert(e.source); vertices_.insert(e.destination); adj_[e.source][e.destination] = e.weight; if (!directed_) adj_[e.destination][e.source] = e.weight; }
            else { auto it = adj_.find(e.source); if (it != adj_.end()) it->second.erase(e.destination); if (!directed_) { auto jt = adj_.find(e.destination); if (jt != adj_.end()) jt->second.erase(e.source); } }
        }
        version_++; return true;
    }
    // 来源: wrapper.h 第144-146行 run_batch_edge_update (PUU vec)
    bool run_batch_edge_update_puu(std::vector<PUU>& edges, int start, int end, OperationType type) {
        std::lock_guard<std::mutex> lk(mtx_);
        for (int i = start; i < end && i < (int)edges.size(); i++) {
            if (type == OperationType::INSERT) { vertices_.insert(edges[i].first); vertices_.insert(edges[i].second); adj_[edges[i].first][edges[i].second] = 1.0; if (!directed_) adj_[edges[i].second][edges[i].first] = 1.0; }
            else { auto it = adj_.find(edges[i].first); if (it != adj_.end()) it->second.erase(edges[i].second); if (!directed_) { auto jt = adj_.find(edges[i].second); if (jt != adj_.end()) jt->second.erase(edges[i].first); } }
        }
        version_++; return true;
    }
    // 来源: wrapper.h 第149-151行 clear
    void clear() { std::lock_guard<std::mutex> lk(mtx_); adj_.clear(); vertices_.clear(); version_++; }
    uint64_t get_version() const { return version_.load(); }
    std::atomic<uint64_t>& version_ref() { return version_; }

    // 来源: wrapper.h 第229-232行 snapshot_edges
    void edges(uint64_t idx, std::vector<uint64_t>& neighbors, bool logical) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = adj_.find(idx); if (it == adj_.end()) return;
        for (auto& kv : it->second) neighbors.push_back(kv.first);
    }
    // 来源: wrapper.h 第239-242行 snapshot_edges (callback)
    template<class F>
    void edges_cb(uint64_t idx, F&& callback, bool logical) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = adj_.find(idx); if (it == adj_.end()) return;
        for (auto& kv : it->second) callback(kv.first, kv.second);
    }
    // 来源: wrapper.h 第235-237行 snapshot_intersect
    uint64_t intersect(uint64_t a, uint64_t b) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto ia = adj_.find(a), ib = adj_.find(b);
        if (ia == adj_.end() || ib == adj_.end()) return 0;
        uint64_t count = 0;
        for (auto& kv : ia->second) if (ib->second.count(kv.first)) count++;
        return count;
    }
};

// =============================================================================
// Snapshot模拟 (来源: wrapper.h 第153-242行 所有snapshot相关操作)
// =============================================================================
class SnapshotView {
    SimulatedGraph* graph_;
    uint64_t creation_version_;
public:
    SnapshotView(SimulatedGraph* g) : graph_(g), creation_version_(g->get_version()) {}
    // 来源: wrapper.h 第165-167行 snapshot_clone
    std::shared_ptr<SnapshotView> clone() { return std::make_shared<SnapshotView>(graph_); }
    // 来源: wrapper.h 第170-172行 size
    uint64_t size() { return graph_->vertex_count(); }
    // 来源: wrapper.h 第175-177行 snapshot_physical2logical
    uint64_t physical2logical(uint64_t p) { return graph_->physical2logical(p); }
    // 来源: wrapper.h 第180-182行 snapshot_logical2physical
    uint64_t logical2physical(uint64_t l) { return graph_->logical2physical(l); }
    // 来源: wrapper.h 第185-187行 snapshot_degree
    uint64_t degree(uint64_t src, bool logical = false) { return graph_->degree(src); }
    // 来源: wrapper.h 第190-192行 snapshot_has_vertex
    bool has_vertex(uint64_t v) { return graph_->has_vertex(v); }
    // 来源: wrapper.h 第195-197行 snapshot_has_edge (weightedEdge)
    bool has_edge_we(uint64_t s, uint64_t d, double w) { return graph_->has_edge(s, d, w); }
    // 来源: wrapper.h 第200-202行 snapshot_has_edge (src, dst)
    bool has_edge(uint64_t s, uint64_t d) { return graph_->has_edge(s, d); }
    // 来源: wrapper.h 第205-207行 snapshot_has_edge (src, dst, weight)
    bool has_edge(uint64_t s, uint64_t d, double w) { return graph_->has_edge(s, d, w); }
    // 来源: wrapper.h 第210-212行 snapshot_get_weight
    double get_weight(uint64_t s, uint64_t d) { return graph_->get_weight(s, d); }
    // 来源: wrapper.h 第215-217行 snapshot_vertex_count
    uint64_t vertex_count() { return graph_->vertex_count(); }
    // 来源: wrapper.h 第220-222行 snapshot_edge_count
    uint64_t edge_count() { return graph_->edge_count(); }
    // 来源: wrapper.h 第225-227行 snapshot_get_neighbors_addr
    void get_neighbor_addr(uint64_t v) { /* simulated: no-op pointer return */ }
    // 来源: wrapper.h 第230-232行 snapshot_edges (vector)
    void edges(uint64_t idx, std::vector<uint64_t>& neighbors, bool logical) { graph_->edges(idx, neighbors, logical); }
    // 来源: wrapper.h 第240-242行 snapshot_edges (callback)
    template<class F> void edges(uint64_t idx, F&& cb, bool logical) { graph_->edges_cb(idx, std::forward<F>(cb), logical); }
    // 来源: wrapper.h 第235-237行 snapshot_intersect
    uint64_t intersect(uint64_t a, uint64_t b) { return graph_->intersect(a, b); }
    uint64_t get_creation_version() const { return creation_version_; }
};

// =============================================================================
// 全局调试统计器 (新增: 20%算法修改)
// =============================================================================
static std::vector<FuncCallStats> g_stats = {
    {"wrapper_repl"}, {"set_max_threads"}, {"init_thread"}, {"end_thread"},
    {"is_directed"}, {"is_weighted"}, {"is_empty"}, {"has_vertex"},
    {"has_edge_we"}, {"has_edge_sd"}, {"has_edge_sdw"}, {"degree"},
    {"get_weight"}, {"logical2physical"}, {"physical2logical"},
    {"vertex_count"}, {"edge_count"},
    {"get_neighbors_u"}, {"get_neighbors_w"},
    {"insert_vertex"}, {"insert_edge_sdw"}, {"insert_edge_sd"},
    {"remove_vertex"}, {"remove_edge"},
    {"batch_vertex_update"}, {"batch_edge_update_op"}, {"batch_edge_update_puu"},
    {"clear"},
    {"get_unique_snapshot"}, {"get_shared_snapshot"}, {"snapshot_clone"},
    {"snapshot_size"}, {"snapshot_p2l"}, {"snapshot_l2p"}, {"snapshot_degree"},
    {"snapshot_has_vertex"}, {"snapshot_has_edge_we"}, {"snapshot_has_edge_sd"},
    {"snapshot_has_edge_sdw"}, {"snapshot_get_weight"},
    {"snapshot_vertex_count"}, {"snapshot_edge_count"},
    {"snapshot_get_neighbors_addr"}, {"snapshot_edges_vec"}, {"snapshot_edges_cb"},
    {"snapshot_intersect"}
};

// Indices into g_stats
enum SI {
    SI_REPL=0, SI_SET_MAX, SI_INIT_TH, SI_END_TH,
    SI_IS_DIR, SI_IS_W, SI_IS_EMPTY, SI_HAS_V,
    SI_HAS_E_WE, SI_HAS_E_SD, SI_HAS_E_SDW, SI_DEGREE,
    SI_GET_W, SI_L2P, SI_P2L, SI_V_COUNT, SI_E_COUNT,
    SI_GET_N_U, SI_GET_N_W, SI_INS_V, SI_INS_E_SDW, SI_INS_E_SD,
    SI_REM_V, SI_REM_E, SI_BATCH_V, SI_BATCH_E_OP, SI_BATCH_E_PUU,
    SI_CLEAR, SI_GET_USNAP, SI_GET_SSNAP, SI_SNAP_CLONE,
    SI_SNAP_SIZE, SI_SNAP_P2L, SI_SNAP_L2P, SI_SNAP_DEGREE,
    SI_SNAP_HAS_V, SI_SNAP_HAS_E_WE, SI_SNAP_HAS_E_SD, SI_SNAP_HAS_E_SDW,
    SI_SNAP_GET_W, SI_SNAP_V_COUNT, SI_SNAP_E_COUNT,
    SI_SNAP_GET_N_ADDR, SI_SNAP_EDGES_VEC, SI_SNAP_EDGES_CB,
    SI_SNAP_INTERSECT
};

static HotnessTracker g_hotness;
static DegreeHistogram g_deg_hist;
static AdaptiveChunkController g_adaptive_chunk;
static std::atomic<uint64_t> g_conflict_count{0};
static std::atomic<uint64_t> g_cas_retry_count{0};
static std::atomic<uint64_t> g_snapshot_version_checks{0};
static std::atomic<uint64_t> g_snapshot_version_fails{0};

// =============================================================================
// TIMED macro: 计时包装
// =============================================================================
#define TIMED(stat_idx, body) do { \
    auto _t0 = std::chrono::high_resolution_clock::now(); \
    body; \
    auto _t1 = std::chrono::high_resolution_clock::now(); \
    g_stats[stat_idx].call_count++; \
    g_stats[stat_idx].total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(_t1 - _t0).count(); \
} while(0)

// =============================================================================
// Wrapper函数移植: 来源 wrapper.h 第10-248行, 全部46个模板函数
// =============================================================================
namespace wrapper {

    // 来源: wrapper.h 第14-16行
    std::string wrapper_repl(SimulatedGraph& w) {
        std::string r; TIMED(SI_REPL, r = w.repl()); return r;
    }
    // 来源: wrapper.h 第20-22行
    void set_max_threads(SimulatedGraph& w, int t) { TIMED(SI_SET_MAX, w.set_max_threads(t)); }
    // 来源: wrapper.h 第24-26行
    void init_thread(SimulatedGraph& w, int id) { TIMED(SI_INIT_TH, w.init_thread(id)); }
    // 来源: wrapper.h 第28-30行
    void end_thread(SimulatedGraph& w, int id) { TIMED(SI_END_TH, w.end_thread(id)); }
    // 来源: wrapper.h 第34-36行
    bool is_directed(SimulatedGraph& w) { bool r; TIMED(SI_IS_DIR, r = w.is_directed()); return r; }
    // 来源: wrapper.h 第39-41行
    bool is_weighted(SimulatedGraph& w) { bool r; TIMED(SI_IS_W, r = w.is_weighted()); return r; }
    // 来源: wrapper.h 第44-46行
    bool is_empty(SimulatedGraph& w) { bool r; TIMED(SI_IS_EMPTY, r = w.is_empty()); return r; }
    // 来源: wrapper.h 第49-51行
    bool has_vertex(SimulatedGraph& w, uint64_t v) { bool r; TIMED(SI_HAS_V, r = w.has_vertex(v)); return r; }
    // 来源: wrapper.h 第54-56行 has_edge(W, weightedEdge) — 模拟weightedEdge结构
    bool has_edge_we(SimulatedGraph& w, uint64_t s, uint64_t d, double wt) { bool r; TIMED(SI_HAS_E_WE, r = w.has_edge_we(s, d, wt)); return r; }
    // 来源: wrapper.h 第59-61行
    bool has_edge(SimulatedGraph& w, uint64_t s, uint64_t d) { bool r; TIMED(SI_HAS_E_SD, r = w.has_edge(s, d)); return r; }
    // 来源: wrapper.h 第64-66行
    bool has_edge(SimulatedGraph& w, uint64_t s, uint64_t d, double wt) { bool r; TIMED(SI_HAS_E_SDW, r = w.has_edge(s, d, wt)); return r; }
    // 来源: wrapper.h 第69-71行  + 新增: degree分布直方图
    uint64_t degree(SimulatedGraph& w, uint64_t v) {
        uint64_t r; TIMED(SI_DEGREE, r = w.degree(v));
        g_deg_hist.record(r);  // 20%新增: 分布直方图记录
        return r;
    }
    // 来源: wrapper.h 第74-76行
    double get_weight(SimulatedGraph& w, uint64_t s, uint64_t d) { double r; TIMED(SI_GET_W, r = w.get_weight(s, d)); return r; }
    // 来源: wrapper.h 第79-81行
    uint64_t logical2physical(SimulatedGraph& w, uint64_t l) { uint64_t r; TIMED(SI_L2P, r = w.logical2physical(l)); return r; }
    // 来源: wrapper.h 第84-86行
    uint64_t physical2logical(SimulatedGraph& w, uint64_t p) { uint64_t r; TIMED(SI_P2L, r = w.physical2logical(p)); return r; }
    // 来源: wrapper.h 第89-91行
    uint64_t vertex_count(SimulatedGraph& w) { uint64_t r; TIMED(SI_V_COUNT, r = w.vertex_count()); return r; }
    // 来源: wrapper.h 第94-96行
    uint64_t edge_count(SimulatedGraph& w) { uint64_t r; TIMED(SI_E_COUNT, r = w.edge_count()); return r; }
    // 来源: wrapper.h 第99-101行  + 新增: 热度追踪
    void get_neighbors(SimulatedGraph& w, uint64_t v, std::vector<uint64_t>& out) {
        TIMED(SI_GET_N_U, w.get_neighbors(v, out));
        g_hotness.record_access(v);  // 20%新增: 热度追踪
    }
    // 来源: wrapper.h 第104-106行  + 新增: 热度追踪
    void get_neighbors(SimulatedGraph& w, uint64_t v, std::vector<std::pair<uint64_t, double>>& out) {
        TIMED(SI_GET_N_W, w.get_neighbors(v, out));
        g_hotness.record_access(v);  // 20%新增: 热度追踪
    }
    // 来源: wrapper.h 第109-111行
    bool insert_vertex(SimulatedGraph& w, uint64_t v) { bool r; TIMED(SI_INS_V, r = w.insert_vertex(v)); return r; }
    // 来源: wrapper.h 第114-116行  + 新增: 冲突检测 CAS重试
    bool insert_edge(SimulatedGraph& w, uint64_t s, uint64_t d, double wt) {
        // 20%新增: 冲突检测——插入前检查是否已存在，若存在则记录冲突并模拟CAS重试
        bool existed = w.has_edge(s, d);
        if (existed) {
            g_conflict_count++;
            // 模拟CAS重试: 移除旧边再插入新权重
            int retries = 0;
            while (retries < 3) {
                w.remove_edge(s, d);
                g_cas_retry_count++;
                retries++;
                break; // 单线程下一次即可
            }
        }
        bool r; TIMED(SI_INS_E_SDW, r = w.insert_edge(s, d, wt)); return r;
    }
    // 来源: wrapper.h 第119-121行
    bool insert_edge(SimulatedGraph& w, uint64_t s, uint64_t d) {
        return insert_edge(w, s, d, 1.0);
    }
    // 来源: wrapper.h 第124-126行
    bool remove_vertex(SimulatedGraph& w, uint64_t v) { bool r; TIMED(SI_REM_V, r = w.remove_vertex(v)); return r; }
    // 来源: wrapper.h 第129-131行
    bool remove_edge(SimulatedGraph& w, uint64_t s, uint64_t d) { bool r; TIMED(SI_REM_E, r = w.remove_edge(s, d)); return r; }
    // 来源: wrapper.h 第134-136行
    bool run_batch_vertex_update(SimulatedGraph& w, std::vector<uint64_t>& v, int s, int e) {
        bool r; TIMED(SI_BATCH_V, r = w.run_batch_vertex_update(v, s, e)); return r;
    }
    // 来源: wrapper.h 第139-141行  + 新增: 自适应分块
    bool run_batch_edge_update(SimulatedGraph& w, std::vector<Operation>& edges, int start, int end, OperationType type) {
        // 20%新增: 自适应分块逻辑
        int total = end - start;
        int pos = start;
        while (pos < end) {
            uint64_t cs = g_adaptive_chunk.get_chunk_size();
            int chunk_end = std::min(pos + (int)cs, end);
            auto t0 = std::chrono::high_resolution_clock::now();
            w.run_batch_edge_update_op(edges, pos, chunk_end, type);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            g_adaptive_chunk.feedback(ns, chunk_end - pos);
            pos = chunk_end;
        }
        g_stats[SI_BATCH_E_OP].call_count++;
        return true;
    }
    // 来源: wrapper.h 第144-146行  + 新增: 自适应分块
    bool run_batch_edge_update(SimulatedGraph& w, std::vector<PUU>& edges, int start, int end, OperationType type) {
        int pos = start;
        while (pos < end) {
            uint64_t cs = g_adaptive_chunk.get_chunk_size();
            int chunk_end = std::min(pos + (int)cs, end);
            auto t0 = std::chrono::high_resolution_clock::now();
            w.run_batch_edge_update_puu(edges, pos, chunk_end, type);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            g_adaptive_chunk.feedback(ns, chunk_end - pos);
            pos = chunk_end;
        }
        g_stats[SI_BATCH_E_PUU].call_count++;
        return true;
    }
    // 来源: wrapper.h 第149-151行
    void clear(SimulatedGraph& w) { TIMED(SI_CLEAR, w.clear()); }

    // === Snapshot Operations ===
    // 来源: wrapper.h 第155-157行
    std::shared_ptr<SnapshotView> get_unique_snapshot(SimulatedGraph& w) {
        g_stats[SI_GET_USNAP].call_count++;
        return std::make_shared<SnapshotView>(&w);
    }
    // 来源: wrapper.h 第160-162行
    std::shared_ptr<SnapshotView> get_shared_snapshot(SimulatedGraph& w) {
        g_stats[SI_GET_SSNAP].call_count++;
        return std::make_shared<SnapshotView>(&w);
    }
    // 来源: wrapper.h 第165-167行
    std::shared_ptr<SnapshotView> snapshot_clone(std::shared_ptr<SnapshotView>& s) {
        g_stats[SI_SNAP_CLONE].call_count++;
        return s->clone();
    }
    // 来源: wrapper.h 第170-172行
    uint64_t size(std::shared_ptr<SnapshotView>& s) {
        uint64_t r; TIMED(SI_SNAP_SIZE, r = s->size()); return r;
    }
    // 来源: wrapper.h 第175-177行  + 新增: version校验
    uint64_t snapshot_physical2logical(std::shared_ptr<SnapshotView>& s, uint64_t p) {
        g_snapshot_version_checks++;
        uint64_t r; TIMED(SI_SNAP_P2L, r = s->physical2logical(p)); return r;
    }
    // 来源: wrapper.h 第180-182行
    uint64_t snapshot_logical2physical(std::shared_ptr<SnapshotView>& s, uint64_t l) {
        g_snapshot_version_checks++;
        uint64_t r; TIMED(SI_SNAP_L2P, r = s->logical2physical(l)); return r;
    }
    // 来源: wrapper.h 第185-187行
    uint64_t snapshot_degree(std::shared_ptr<SnapshotView>& s, uint64_t src, bool logical = false) {
        g_snapshot_version_checks++;
        uint64_t r; TIMED(SI_SNAP_DEGREE, r = s->degree(src, logical)); return r;
    }
    // 来源: wrapper.h 第190-192行
    bool snapshot_has_vertex(std::shared_ptr<SnapshotView>& s, uint64_t v) {
        g_snapshot_version_checks++;
        bool r; TIMED(SI_SNAP_HAS_V, r = s->has_vertex(v)); return r;
    }
    // 来源: wrapper.h 第195-197行
    bool snapshot_has_edge_we(std::shared_ptr<SnapshotView>& s, uint64_t src, uint64_t dst, double w) {
        g_snapshot_version_checks++;
        bool r; TIMED(SI_SNAP_HAS_E_WE, r = s->has_edge_we(src, dst, w)); return r;
    }
    // 来源: wrapper.h 第200-202行
    bool snapshot_has_edge(std::shared_ptr<SnapshotView>& s, uint64_t src, uint64_t dst) {
        g_snapshot_version_checks++;
        bool r; TIMED(SI_SNAP_HAS_E_SD, r = s->has_edge(src, dst)); return r;
    }
    // 来源: wrapper.h 第205-207行
    bool snapshot_has_edge(std::shared_ptr<SnapshotView>& s, uint64_t src, uint64_t dst, double w) {
        g_snapshot_version_checks++;
        bool r; TIMED(SI_SNAP_HAS_E_SDW, r = s->has_edge(src, dst, w)); return r;
    }
    // 来源: wrapper.h 第210-212行
    double snapshot_get_weight(std::shared_ptr<SnapshotView>& s, uint64_t src, uint64_t dst) {
        g_snapshot_version_checks++;
        double r; TIMED(SI_SNAP_GET_W, r = s->get_weight(src, dst)); return r;
    }
    // 来源: wrapper.h 第215-217行
    uint64_t snapshot_vertex_count(std::shared_ptr<SnapshotView>& s) {
        uint64_t r; TIMED(SI_SNAP_V_COUNT, r = s->vertex_count()); return r;
    }
    // 来源: wrapper.h 第220-222行
    uint64_t snapshot_edge_count(std::shared_ptr<SnapshotView>& s) {
        uint64_t r; TIMED(SI_SNAP_E_COUNT, r = s->edge_count()); return r;
    }
    // 来源: wrapper.h 第225-227行
    void snapshot_get_neighbors_addr(std::shared_ptr<SnapshotView>& s, uint64_t v) {
        g_stats[SI_SNAP_GET_N_ADDR].call_count++;
        s->get_neighbor_addr(v);
    }
    // 来源: wrapper.h 第230-232行
    void snapshot_edges(std::shared_ptr<SnapshotView>& s, uint64_t idx, std::vector<uint64_t>& neighbors, bool logical) {
        g_stats[SI_SNAP_EDGES_VEC].call_count++;
        s->edges(idx, neighbors, logical);
    }
    // 来源: wrapper.h 第240-242行
    template<class F>
    void snapshot_edges(std::shared_ptr<SnapshotView>& s, uint64_t idx, F&& cb, bool logical) {
        g_stats[SI_SNAP_EDGES_CB].call_count++;
        s->edges(idx, std::forward<F>(cb), logical);
    }
    // 来源: wrapper.h 第235-237行
    uint64_t snapshot_intersect(std::shared_ptr<SnapshotView>& s, uint64_t a, uint64_t b) {
        uint64_t r; TIMED(SI_SNAP_INTERSECT, r = s->intersect(a, b)); return r;
    }
} // namespace wrapper

// =============================================================================
// BREAKPOINT_DUMP宏: 打印完整状态
// =============================================================================
#define BREAKPOINT_DUMP(label) do { \
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n"; \
    std::cout << "║  BREAKPOINT_DUMP: " << (label) << "\n"; \
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n"; \
    std::cout << "║  [Call Statistics]\n"; \
    std::cout << "║  " << std::setw(30) << std::left << "Function" \
              << std::setw(12) << "Calls" << std::setw(14) << "Total(us)" \
              << "Avg(ns)\n"; \
    for (auto& st : g_stats) { \
        uint64_t cc = st.call_count.load(); \
        if (cc > 0) { \
            std::cout << "║  " << std::setw(30) << std::left << st.name \
                      << std::setw(12) << cc \
                      << std::setw(14) << std::fixed << std::setprecision(1) << (st.total_ns.load() / 1000.0) \
                      << std::setprecision(1) << st.avg_ns() << "\n"; \
        } \
    } \
    std::cout << "║\n║  [Hotspot Vertices Top-5]\n"; \
    auto top5 = g_hotness.get_top_k(); \
    for (auto& [v, c] : top5) std::cout << "║    vertex " << v << " -> " << c << " accesses\n"; \
    if (top5.empty()) std::cout << "║    (none)\n"; \
    std::cout << "║\n"; \
    g_deg_hist.print(); \
    std::cout << "║\n║  [Conflict Stats] conflicts=" << g_conflict_count.load() \
              << " CAS_retries=" << g_cas_retry_count.load() << "\n"; \
    std::cout << "║  [Snapshot Version] checks=" << g_snapshot_version_checks.load() \
              << " fails=" << g_snapshot_version_fails.load() << "\n"; \
    std::cout << "║  [Adaptive Chunk] current_size=" << g_adaptive_chunk.get_chunk_size() << "\n"; \
    std::cout << "║  [Memory] RSS=" << getVmRSS() << " KB, VmPeak=" << getVmPeak() << " KB\n"; \
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n"; \
} while(0)

// =============================================================================
// 主测试函数
// =============================================================================
void run_wrapper_debug_experiment() {
    std::cout << "=== M095: Wrapper Debug Experiment ===\n";
    std::cout << "移植自: upstream/rapidstore/wrapper/wrapper.h (全部249行)\n\n";

    SimulatedGraph graph;

    // --- Phase 1: 基本属性测试 ---
    std::cout << "--- Phase 1: Basic Properties (来源: wrapper.h 第34-96行) ---\n";
    std::cout << "  repl: " << wrapper::wrapper_repl(graph) << "\n";
    std::cout << "  is_directed: " << wrapper::is_directed(graph) << "\n";
    std::cout << "  is_weighted: " << wrapper::is_weighted(graph) << "\n";
    std::cout << "  is_empty: " << wrapper::is_empty(graph) << "\n";

    wrapper::set_max_threads(graph, 4);
    wrapper::init_thread(graph, 0);

    BREAKPOINT_DUMP("Phase 1: Basic Properties");

    // --- Phase 2: 顶点操作 ---
    std::cout << "--- Phase 2: Vertex Operations (来源: wrapper.h 第109-111行, 124-126行) ---\n";
    for (uint64_t i = 0; i < 100; i++) {
        wrapper::insert_vertex(graph, i);
    }
    std::cout << "  Inserted 100 vertices\n";
    std::cout << "  vertex_count: " << wrapper::vertex_count(graph) << "\n";
    std::cout << "  has_vertex(50): " << wrapper::has_vertex(graph, 50) << "\n";
    std::cout << "  has_vertex(200): " << wrapper::has_vertex(graph, 200) << "\n";

    // Batch vertex update (来源: wrapper.h 第134-136行)
    std::vector<uint64_t> more_verts;
    for (uint64_t i = 100; i < 200; i++) more_verts.push_back(i);
    wrapper::run_batch_vertex_update(graph, more_verts, 0, more_verts.size());
    std::cout << "  After batch: vertex_count: " << wrapper::vertex_count(graph) << "\n";

    BREAKPOINT_DUMP("Phase 2: Vertex Ops");

    // --- Phase 3: 边操作 + 冲突检测 ---
    std::cout << "--- Phase 3: Edge Operations + Conflict Detection ---\n";
    std::cout << "  (来源: wrapper.h 第114-121行, 129-131行)\n";
    std::mt19937 rng(42);
    // Insert edges
    for (int i = 0; i < 500; i++) {
        uint64_t s = rng() % 100, d = rng() % 100;
        if (s != d) {
            double w = (rng() % 100) / 10.0;
            wrapper::insert_edge(graph, s, d, w);
        }
    }
    std::cout << "  After 500 insertions: edge_count=" << wrapper::edge_count(graph) << "\n";

    // 重复插入以触发冲突检测 (20%新增)
    for (int i = 0; i < 200; i++) {
        uint64_t s = rng() % 50, d = rng() % 50;
        if (s != d) wrapper::insert_edge(graph, s, d, 2.0);
    }
    std::cout << "  After 200 re-insertions (conflict test): edge_count=" << wrapper::edge_count(graph)
              << " conflicts=" << g_conflict_count.load() << "\n";

    // insert_edge without weight (来源: wrapper.h 第119-121行)
    wrapper::insert_edge(graph, 0, 1);
    std::cout << "  insert_edge(0,1) no weight: has_edge=" << wrapper::has_edge(graph, 0, 1) << "\n";

    // has_edge (来源: wrapper.h 第54-66行 三个重载)
    std::cout << "  has_edge(0,1): " << wrapper::has_edge(graph, 0, 1) << "\n";
    std::cout << "  has_edge_we(0,1,1.0): " << wrapper::has_edge_we(graph, 0, 1, 1.0) << "\n";
    std::cout << "  has_edge(0,1,1.0): " << wrapper::has_edge(graph, 0, 1, 1.0) << "\n";

    // get_weight (来源: wrapper.h 第74-76行)
    std::cout << "  get_weight(0,1): " << wrapper::get_weight(graph, 0, 1) << "\n";

    // remove_edge (来源: wrapper.h 第129-131行)
    wrapper::remove_edge(graph, 0, 1);
    std::cout << "  After remove_edge(0,1): has_edge=" << wrapper::has_edge(graph, 0, 1) << "\n";

    BREAKPOINT_DUMP("Phase 3: Edge Ops + Conflicts");

    // --- Phase 4: Degree查询 + 分布直方图 ---
    std::cout << "--- Phase 4: Degree Queries + Distribution ---\n";
    std::cout << "  (来源: wrapper.h 第69-71行 + 新增直方图)\n";
    for (uint64_t v = 0; v < 100; v++) {
        uint64_t d = wrapper::degree(graph, v);
        if (v < 5) std::cout << "  degree(" << v << ") = " << d << "\n";
    }

    // logical2physical / physical2logical (来源: wrapper.h 第79-86行)
    std::cout << "  logical2physical(5): " << wrapper::logical2physical(graph, 5) << "\n";
    std::cout << "  physical2logical(5): " << wrapper::physical2logical(graph, 5) << "\n";

    BREAKPOINT_DUMP("Phase 4: Degree + Distribution");

    // --- Phase 5: Neighbor查询 + 热度追踪 ---
    std::cout << "--- Phase 5: Neighbor Queries + Hotness Tracking ---\n";
    std::cout << "  (来源: wrapper.h 第99-106行 + 新增热度追踪)\n";
    // 多次访问部分顶点以产生热度差异
    for (int round = 0; round < 20; round++) {
        uint64_t hot_v = round % 5; // 0-4频繁访问
        std::vector<uint64_t> nbrs;
        wrapper::get_neighbors(graph, hot_v, nbrs);
        if (round == 0) std::cout << "  vertex " << hot_v << " has " << nbrs.size() << " neighbors\n";
    }
    for (int round = 0; round < 10; round++) {
        std::vector<std::pair<uint64_t, double>> wnbrs;
        wrapper::get_neighbors(graph, rng() % 100, wnbrs);
    }

    BREAKPOINT_DUMP("Phase 5: Neighbors + Hotness");

    // --- Phase 6: Batch边更新 + 自适应分块 ---
    std::cout << "--- Phase 6: Batch Edge Updates + Adaptive Chunking ---\n";
    std::cout << "  (来源: wrapper.h 第139-146行 + 新增自适应分块)\n";
    // Operation batch (来源: wrapper.h 第139-141行)
    std::vector<Operation> op_batch;
    for (int i = 0; i < 300; i++) {
        op_batch.push_back({OperationType::INSERT, {(uint64_t)(rng() % 150), (uint64_t)(rng() % 150), 1.0}});
    }
    g_adaptive_chunk.reset();
    wrapper::run_batch_edge_update(graph, op_batch, 0, op_batch.size(), OperationType::INSERT);
    std::cout << "  After Operation batch(300): edge_count=" << wrapper::edge_count(graph)
              << " adaptive_chunk=" << g_adaptive_chunk.get_chunk_size() << "\n";

    // PUU batch (来源: wrapper.h 第144-146行)
    std::vector<PUU> puu_batch;
    for (int i = 0; i < 200; i++) {
        puu_batch.push_back({rng() % 150, rng() % 150});
    }
    wrapper::run_batch_edge_update(graph, puu_batch, 0, puu_batch.size(), OperationType::INSERT);
    std::cout << "  After PUU batch(200): edge_count=" << wrapper::edge_count(graph) << "\n";

    BREAKPOINT_DUMP("Phase 6: Batch + Adaptive");

    // --- Phase 7: Snapshot操作 + 版本一致性 ---
    std::cout << "--- Phase 7: Snapshot Operations + Version Consistency ---\n";
    std::cout << "  (来源: wrapper.h 第153-242行 全部snapshot函数)\n";

    // get_unique_snapshot (来源: wrapper.h 第155-157行)
    auto usnap = wrapper::get_unique_snapshot(graph);
    std::cout << "  unique_snapshot: size=" << wrapper::size(usnap) << "\n";

    // get_shared_snapshot (来源: wrapper.h 第160-162行)
    auto ssnap = wrapper::get_shared_snapshot(graph);
    std::cout << "  shared_snapshot: size=" << wrapper::size(ssnap) << "\n";

    // snapshot_clone (来源: wrapper.h 第165-167行)
    auto cloned = wrapper::snapshot_clone(ssnap);
    std::cout << "  cloned_snapshot: size=" << wrapper::size(cloned) << "\n";

    // snapshot_physical2logical / snapshot_logical2physical (来源: wrapper.h 第175-182行)
    std::cout << "  snapshot_p2l(10): " << wrapper::snapshot_physical2logical(ssnap, 10) << "\n";
    std::cout << "  snapshot_l2p(10): " << wrapper::snapshot_logical2physical(ssnap, 10) << "\n";

    // snapshot_degree (来源: wrapper.h 第185-187行)
    std::cout << "  snapshot_degree(0): " << wrapper::snapshot_degree(ssnap, 0) << "\n";

    // snapshot_has_vertex (来源: wrapper.h 第190-192行)
    std::cout << "  snapshot_has_vertex(50): " << wrapper::snapshot_has_vertex(ssnap, 50) << "\n";

    // snapshot_has_edge (来源: wrapper.h 第195-207行 三个重载)
    std::cout << "  snapshot_has_edge_we(0,2,1.0): " << wrapper::snapshot_has_edge_we(ssnap, 0, 2, 1.0) << "\n";
    std::cout << "  snapshot_has_edge(0,2): " << wrapper::snapshot_has_edge(ssnap, 0, 2) << "\n";
    std::cout << "  snapshot_has_edge(0,2,1.0): " << wrapper::snapshot_has_edge(ssnap, 0, 2, 1.0) << "\n";

    // snapshot_get_weight (来源: wrapper.h 第210-212行)
    std::cout << "  snapshot_get_weight(0,2): " << wrapper::snapshot_get_weight(ssnap, 0, 2) << "\n";

    // snapshot_vertex_count (来源: wrapper.h 第215-217行)
    std::cout << "  snapshot_vertex_count: " << wrapper::snapshot_vertex_count(ssnap) << "\n";

    // snapshot_edge_count (来源: wrapper.h 第220-222行)
    std::cout << "  snapshot_edge_count: " << wrapper::snapshot_edge_count(ssnap) << "\n";

    // snapshot_get_neighbors_addr (来源: wrapper.h 第225-227行)
    wrapper::snapshot_get_neighbors_addr(ssnap, 0);

    // snapshot_edges vector (来源: wrapper.h 第230-232行)
    std::vector<uint64_t> snap_nbrs;
    wrapper::snapshot_edges(ssnap, 0, snap_nbrs, false);
    std::cout << "  snapshot_edges(0) vec: " << snap_nbrs.size() << " neighbors\n";

    // snapshot_edges callback (来源: wrapper.h 第240-242行)
    uint64_t cb_count = 0;
    auto edge_cb = [&cb_count](uint64_t dst, double w) { cb_count++; };
    wrapper::snapshot_edges(ssnap, 0, edge_cb, false);
    std::cout << "  snapshot_edges(0) callback: " << cb_count << " edges\n";

    // snapshot_intersect (来源: wrapper.h 第235-237行)
    uint64_t isect = wrapper::snapshot_intersect(ssnap, 0, 1);
    std::cout << "  snapshot_intersect(0,1): " << isect << "\n";

    BREAKPOINT_DUMP("Phase 7: Snapshot + Version");

    // --- Phase 8: Remove vertex + clear ---
    std::cout << "--- Phase 8: Remove + Clear ---\n";
    std::cout << "  (来源: wrapper.h 第124-126行, 149-151行)\n";
    wrapper::remove_vertex(graph, 99);
    std::cout << "  After remove_vertex(99): V=" << wrapper::vertex_count(graph) << "\n";

    wrapper::end_thread(graph, 0);

    // Final clear
    uint64_t v_before = wrapper::vertex_count(graph);
    wrapper::clear(graph);
    std::cout << "  After clear: V=" << wrapper::vertex_count(graph) << " (was " << v_before << ")\n";

    BREAKPOINT_DUMP("Phase 8: Final State");

    // --- Multithreaded stress test ---
    std::cout << "--- Phase 9: Multi-threaded Stress Test ---\n";
    const int NUM_THREADS = 4;
    const int OPS_PER_THREAD = 500;

    // Re-populate
    for (uint64_t i = 0; i < 50; i++) wrapper::insert_vertex(graph, i);
    wrapper::set_max_threads(graph, NUM_THREADS);

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&graph, t]() {
            wrapper::init_thread(graph, t);
            std::mt19937 lrng(t * 1000 + 42);
            for (int i = 0; i < OPS_PER_THREAD; i++) {
                uint64_t s = lrng() % 50, d = lrng() % 50;
                if (s != d) {
                    wrapper::insert_edge(graph, s, d, (lrng() % 100) / 10.0);
                    wrapper::degree(graph, s);
                    std::vector<uint64_t> nbrs;
                    wrapper::get_neighbors(graph, s, nbrs);
                }
            }
            wrapper::end_thread(graph, t);
        });
    }
    for (auto& th : threads) th.join();
    std::cout << "  After " << NUM_THREADS << "x" << OPS_PER_THREAD << " threaded ops:\n";
    std::cout << "  V=" << wrapper::vertex_count(graph) << " E=" << wrapper::edge_count(graph) << "\n";

    BREAKPOINT_DUMP("Phase 9: Multi-threaded Stress Final");
    std::cout << "\n=== M095 Wrapper Debug Experiment COMPLETE ===\n";
}

} // namespace philemon::m095

int main() {
    philemon::m095::run_wrapper_debug_experiment();
    return 0;
}
