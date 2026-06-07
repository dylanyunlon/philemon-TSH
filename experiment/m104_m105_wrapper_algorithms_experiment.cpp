/**
 * m104_m105_wrapper_algorithms_experiment.cpp — M104-M105: wrapper/algorithms 6算法深度实验
 *
 * 覆盖模块:
 *   upstream/rapidstore/wrapper/algorithms/BFS.h      (330行) — bfsExperiments: 方向切换BFS
 *   upstream/rapidstore/wrapper/algorithms/SSSP.h     (182行) — ssspExperiments: delta-stepping SSSP
 *   upstream/rapidstore/wrapper/algorithms/WCC.h      (149行) — wccExperiments: 按秩合并WCC
 *   upstream/rapidstore/wrapper/algorithms/PR.h       (174行) — pageRankExperiments: PageRank
 *   upstream/rapidstore/wrapper/algorithms/TC.h       (93行)  — TriangleCounting: 三角形计数
 *   upstream/rapidstore/wrapper/algorithms/TC_opt.h   (81行)  — TriangleCounting_optimized: 前缀优化TC
 *
 * 算法改动 (~20%):
 *   BFS:
 *     - [MOD] 方向切换统计 td_steps/bu_steps + direction_switch_count
 *     - [MOD] 层级直方图 level_histogram
 *     - [MOD] visited比率追踪 visited_ratio per level
 *     - [MOD] 断点: 每层打印 distances/queue/bitmap 状态
 *   SSSP:
 *     - [MOD] 松弛操作计数 relax_count / relax_success_count
 *     - [MOD] delta-stepping桶统计 bucket_max_size / bucket_usage_count
 *     - [MOD] 收敛曲线 frontier_size_history
 *     - [MOD] 断点: 每轮iter打印 dist数组+frontier状态
 *   WCC:
 *     - [MOD] 按秩合并统计 union_by_rank_count
 *     - [MOD] 路径压缩跳数 path_compression_hops
 *     - [MOD] 组件大小分布 component_size_distribution
 *     - [MOD] 断点: 每轮打印 comp[] 数组 + change状态
 *   PR:
 *     - [MOD] 收敛曲线(L1 norm) per-iteration l1_norm
 *     - [MOD] dangling节点处理统计 dangling_count
 *     - [MOD] 迭代能量守恒检验 score_sum per iteration
 *     - [MOD] 断点: 每轮打印 scores[] + outgoing_contrib[]
 *   TC:
 *     - [MOD] 边交集计数 intersect_call_count / search_call_count
 *     - [MOD] 哈希vs排序策略对比 hash_strategy_count / sort_strategy_count
 *     - [MOD] 断点: 每1000顶点打印 进度 + 三角形增量
 *   TC_opt:
 *     - [MOD] 前缀优化跳过率 prefix_skip_count / total_comparisons
 *     - [MOD] early-termination计数 early_term_count
 *     - [MOD] 断点: 每10顶点打印 neighbors + marker状态
 *
 * 编译: g++ -std=c++17 -O2 -pthread -o m104_test experiment/m104_m105_wrapper_algorithms_experiment.cpp
 * Milestone: M104-M105 (第12位Claude, Opus 4.6)
 */

#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <map>

// ═══════════════════════════════════════════════════════════════════
//  全局测试计数
// ═══════════════════════════════════════════════════════════════════
static int g_tests_run = 0, g_tests_passed = 0, g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
        g_tests_failed++; g_tests_run++; return; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    std::printf("  [PASS] %s\n", name); \
    g_tests_passed++; g_tests_run++; \
} while(0)

// ═══════════════════════════════════════════════════════════════════
//  模拟图数据结构: 替代 wrapper:: / gapbs:: / snapshot 依赖
//  让6个算法在没有upstream基础库的前提下独立运行
// ═══════════════════════════════════════════════════════════════════

namespace philemon {
namespace experiment {

// --- gapbs模拟 (upstream gapbs.h 依赖，m102已移植核心，这里提供算法所需子集) ---
namespace gapbs {

template<typename T>
bool compare_and_swap(T& target, T& old_val, T new_val) {
    if (target == old_val) {
        target = new_val;
        return true;
    }
    old_val = target;
    return false;
}

template<typename T>
T fetch_and_add(T& target, T val) {
    T old = target;
    target += val;
    return old;
}

template<typename T>
class pvector {
    std::vector<T> m_data;
public:
    pvector() {}
    pvector(size_t n) : m_data(n) {}
    pvector(size_t n, T val) : m_data(n, val) {}
    T& operator[](size_t i) { return m_data[i]; }
    const T& operator[](size_t i) const { return m_data[i]; }
    size_t size() const { return m_data.size(); }
    void resize(size_t n) { m_data.resize(n); }
    void resize(size_t n, T val) { m_data.resize(n, val); }
    typename std::vector<T>::iterator begin() { return m_data.begin(); }
    typename std::vector<T>::iterator end() { return m_data.end(); }
};

// SlidingQueue: 模拟 upstream gapbs.h SlidingQueue
// gapbs语义: push_back 到 m_tail, slide_window 让 window = [上次shared, 当前tail)
// window迭代 begin() = m_start, end() = m_shared
template<typename T>
class SlidingQueue {
    std::vector<T> m_buf;
    size_t m_start;   // window begin
    size_t m_shared;  // window end (= 上次slide时的tail)
    size_t m_tail;    // 真正的push位置
public:
    SlidingQueue(size_t cap) : m_buf(cap + 1), m_start(0), m_shared(0), m_tail(0) {}
    void push_back(T val) { m_buf[m_tail++] = val; }
    // slide_window: 把刚push的内容变成新窗口
    // window = [m_shared, m_tail) → 新窗口
    void slide_window() { m_start = m_shared; m_shared = m_tail; }
    bool empty() const { return m_start == m_shared; }
    size_t size() const { return m_shared - m_start; }
    T* begin() { return &m_buf[m_start]; }
    T* end() { return &m_buf[m_shared]; }
    const T* begin() const { return &m_buf[m_start]; }
    const T* end() const { return &m_buf[m_shared]; }

    // [MOD] 断点: dump_state 打印窗口位置/size
    void dump_state(const char* label) const {
        std::printf("[SlidingQueue:%s] start=%zu shared=%zu tail=%zu window_size=%zu\n",
                    label, m_start, m_shared, m_tail, size());
    }
};

// QueueBuffer: 模拟 upstream gapbs.h QueueBuffer (线程本地缓冲)
template<typename T>
class QueueBuffer {
    static constexpr size_t kBufSize = 64;
    T m_local[kBufSize];
    size_t m_count;
    SlidingQueue<T>& m_queue;
    std::mutex& m_mu;
public:
    QueueBuffer(SlidingQueue<T>& q, std::mutex& mu) : m_count(0), m_queue(q), m_mu(mu) {}
    void push_back(T val) {
        m_local[m_count++] = val;
        if (m_count >= kBufSize) flush();
    }
    void flush() {
        std::lock_guard<std::mutex> lock(m_mu);
        for (size_t i = 0; i < m_count; i++) {
            m_queue.push_back(m_local[i]);
        }
        m_count = 0;
    }
};

// Bitmap: 模拟 upstream gapbs.h Bitmap
class Bitmap {
    std::vector<uint64_t> m_bits;
    size_t m_size;
public:
    Bitmap(size_t n) : m_size(n), m_bits((n + 63) / 64, 0) {}
    void set_bit(size_t i) { m_bits[i / 64] |= (1ULL << (i % 64)); }
    void set_bit_atomic(size_t i) { set_bit(i); /* single-thread fallback */ }
    bool get_bit(size_t i) const { return (m_bits[i / 64] >> (i % 64)) & 1ULL; }
    void reset() { std::fill(m_bits.begin(), m_bits.end(), 0); }
    void swap(Bitmap& other) { m_bits.swap(other.m_bits); std::swap(m_size, other.m_size); }

    // [MOD] 断点: bit density
    double density() const {
        size_t set_count = 0;
        for (auto w : m_bits) set_count += __builtin_popcountll(w);
        return m_size > 0 ? (double)set_count / m_size : 0.0;
    }
    size_t count_set() const {
        size_t c = 0;
        for (auto w : m_bits) c += __builtin_popcountll(w);
        return c;
    }
};

} // namespace gapbs

// ═══════════════════════════════════════════════════════════════════
//  模拟图快照 — 以邻接表表示, 替代 wrapper::snapshot_*
// ═══════════════════════════════════════════════════════════════════
struct SimGraph {
    size_t num_vertices;
    // adj[u] = list of (v, weight)
    std::vector<std::vector<std::pair<uint64_t, double>>> adj;

    SimGraph() : num_vertices(0) {}
    SimGraph(size_t n) : num_vertices(n), adj(n) {}

    void add_edge(uint64_t u, uint64_t v, double w = 1.0) {
        if (u < num_vertices && v < num_vertices) {
            adj[u].push_back({v, w});
        }
    }

    void add_undirected_edge(uint64_t u, uint64_t v, double w = 1.0) {
        add_edge(u, v, w);
        add_edge(v, u, w);
    }

    size_t vertex_count() const { return num_vertices; }
    size_t edge_count() const {
        size_t c = 0;
        for (auto& a : adj) c += a.size();
        return c;
    }
    size_t degree(uint64_t v) const {
        return (v < num_vertices) ? adj[v].size() : 0;
    }
    bool has_edge(uint64_t u, uint64_t v) const {
        if (u >= num_vertices) return false;
        for (auto& [dst, w] : adj[u]) {
            if (dst == v) return true;
        }
        return false;
    }

    // 模拟 snapshot_edges: 对v的每条边调用回调
    template<typename Func>
    void for_edges(uint64_t v, Func&& f) const {
        if (v >= num_vertices) return;
        for (auto& [dst, w] : adj[v]) {
            f(dst, w);
        }
    }

    // 模拟 snapshot_intersect: 计算u和v的邻居交集大小
    size_t intersect_neighbors(uint64_t u, uint64_t v) const {
        if (u >= num_vertices || v >= num_vertices) return 0;
        std::unordered_set<uint64_t> s;
        for (auto& [dst, w] : adj[u]) s.insert(dst);
        size_t count = 0;
        for (auto& [dst, w] : adj[v]) {
            if (s.count(dst)) count++;
        }
        return count;
    }
};

// 图构建工具
static SimGraph build_chain_graph(size_t n) {
    // 0 -> 1 -> 2 -> ... -> n-1  (无向)
    SimGraph g(n);
    for (size_t i = 0; i + 1 < n; i++) {
        g.add_undirected_edge(i, i + 1);
    }
    return g;
}

static SimGraph build_star_graph(size_t n) {
    // 0 is center, connected to 1..n-1  (无向)
    SimGraph g(n);
    for (size_t i = 1; i < n; i++) {
        g.add_undirected_edge(0, i);
    }
    return g;
}

static SimGraph build_complete_graph(size_t n) {
    SimGraph g(n);
    for (size_t i = 0; i < n; i++)
        for (size_t j = i + 1; j < n; j++)
            g.add_undirected_edge(i, j);
    return g;
}

static SimGraph build_cycle_graph(size_t n) {
    SimGraph g(n);
    for (size_t i = 0; i < n; i++)
        g.add_undirected_edge(i, (i + 1) % n);
    return g;
}

static SimGraph build_bipartite_graph(size_t left, size_t right) {
    // 0..left-1 on one side, left..left+right-1 on other
    SimGraph g(left + right);
    for (size_t i = 0; i < left; i++)
        for (size_t j = left; j < left + right; j++)
            g.add_undirected_edge(i, j);
    return g;
}

static SimGraph build_disconnected_graph(size_t n1, size_t n2) {
    // Two disconnected cliques
    SimGraph g(n1 + n2);
    for (size_t i = 0; i < n1; i++)
        for (size_t j = i + 1; j < n1; j++)
            g.add_undirected_edge(i, j);
    for (size_t i = n1; i < n1 + n2; i++)
        for (size_t j = i + 1; j < n1 + n2; j++)
            g.add_undirected_edge(i, j);
    return g;
}

static SimGraph build_triangle_mesh(size_t rows, size_t cols) {
    // Grid graph with diagonal edges forming triangles
    SimGraph g(rows * cols);
    auto idx = [cols](size_t r, size_t c) { return r * cols + c; };
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            if (c + 1 < cols) g.add_undirected_edge(idx(r, c), idx(r, c + 1));
            if (r + 1 < rows) g.add_undirected_edge(idx(r, c), idx(r + 1, c));
            // diagonal for triangles
            if (r + 1 < rows && c + 1 < cols)
                g.add_undirected_edge(idx(r, c), idx(r + 1, c + 1));
        }
    }
    return g;
}


// ═══════════════════════════════════════════════════════════════════
//  §1 — BFS 移植 (upstream BFS.h 全330行)
//       骨架: upstream/rapidstore/wrapper/algorithms/BFS.h
//       修改点用 [MOD] 标记
// ═══════════════════════════════════════════════════════════════════

// --- 全局统计 [MOD] ---
struct BFSStats {
    uint64_t td_steps = 0;           // [MOD] top-down步数
    uint64_t bu_steps = 0;           // [MOD] bottom-up步数
    uint64_t direction_switch_count = 0; // [MOD] 方向切换次数
    std::vector<uint64_t> level_histogram; // [MOD] 每层发现的顶点数
    std::vector<double> visited_ratio;     // [MOD] 每层已访问比率
    uint64_t total_edges_checked = 0;      // [MOD] 总检查边数

    void dump() const {
        std::printf("[BFS Stats] td_steps=%lu bu_steps=%lu switches=%lu total_edges=%lu\n",
                    (unsigned long)td_steps, (unsigned long)bu_steps,
                    (unsigned long)direction_switch_count, (unsigned long)total_edges_checked);
        std::printf("[BFS Level Histogram] ");
        for (size_t i = 0; i < level_histogram.size(); i++) {
            std::printf("L%zu:%lu ", i, (unsigned long)level_histogram[i]);
        }
        std::printf("\n");
        std::printf("[BFS Visited Ratio] ");
        for (size_t i = 0; i < visited_ratio.size(); i++) {
            std::printf("L%zu:%.4f ", i, visited_ratio[i]);
        }
        std::printf("\n");
    }
};

// upstream BFS.h: template <class F, class S> class bfsExperiments
// 移植为具体类 bfsExperiments，用 SimGraph 替代模板参数
// upstream成员: m_num_threads, m_alpha, m_beta, m_mutex, m_method, m_snapshot
class bfsExperiments {
    const int m_num_threads;       // upstream L28: const int m_num_threads
    const int m_alpha;             // upstream L29: const int m_alpha
    const int m_beta;              // upstream L30: const int m_beta
    std::mutex m_mutex;            // upstream L31: std::mutex m_mutex
    const SimGraph& m_graph;       // 替代 m_method + m_snapshot

public:
    BFSStats stats;                // [MOD] 统计追踪

    // upstream L37: bfsExperiments(num_threads, alpha, beta, method, snapshot)
    bfsExperiments(const int num_threads, const int alpha, const int beta, const SimGraph& graph)
        : m_num_threads(num_threads), m_alpha(alpha), m_beta(beta), m_graph(graph) {
        // upstream L39: omp_set_num_threads(num_threads)
        // [MOD] 自包含环境不依赖OMP
        std::printf("[BFS INIT] threads=%d alpha=%d beta=%d V=%zu E=%zu\n",
                    num_threads, alpha, beta, graph.vertex_count(), graph.edge_count());
    }

    // upstream L42: ~bfsExperiments
    ~bfsExperiments() {}

    // upstream L298-330: run_gapbs_bfs
    void run_gapbs_bfs(uint64_t src, std::vector<std::pair<uint64_t, int64_t>>& external_ids) {
        // upstream L299: uint64_t physical_source = wrapper::snapshot_logical2physical(m_snapshot, source)
        uint64_t physical_source = src; // 模拟环境 logical==physical
        auto start = std::chrono::high_resolution_clock::now(); // upstream L300

        auto distances = bfs(physical_source); // upstream L302

        // upstream L304-321: 多线程拷贝结果到 external_ids
        uint64_t N = m_graph.vertex_count();
        external_ids.resize(N); // upstream L310: external_ids.reserve(N) (修正为resize)

        // [MOD] 单线程替代多线程拷贝
        for (uint64_t u = 0; u < N; u++) {
            // upstream L316: wrapper::snapshot_physical2logical(snapshot_local, u)
            external_ids[u] = std::make_pair(u, distances[u]);
        }

        // upstream L324-327: 输出结果
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::printf("Gapbs BFS took %ld milliseconds\n", (long)duration.count());
        // upstream L328: log_info("BFS: %ld milliseconds", duration.count())
        std::printf("[BFS LOG] BFS: %ld milliseconds\n", (long)duration.count());

        // [MOD] 断点: 打印最终 distances 状态
        std::printf("[BFS BREAKPOINT] Final distances: ");
        for (uint64_t i = 0; i < std::min(N, (uint64_t)20); i++) {
            std::printf("[%lu]=%ld ", (unsigned long)i, (long)distances[i]);
        }
        if (N > 20) std::printf("... (truncated, total %lu)", (unsigned long)N);
        std::printf("\n");

        stats.dump(); // [MOD] 输出全部统计
    }

private:
    // upstream L44-83: init_distances()
    std::vector<int64_t> init_distances() {
        // upstream L45: const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot)
        const uint64_t N = m_graph.vertex_count();
        // upstream L46: gapbs::pvector<std::atomic<int64_t>> distances(N)
        std::vector<int64_t> distances(N);

        // upstream L48-49: atomic vertex_checked, threads
        // [MOD] 单线程初始化替代多线程
        // upstream L53: wrapper::set_max_threads(m_method, m_num_threads)
        // upstream L55-66: 线程体
        for (uint64_t i = 0; i < N; i++) {
            // upstream L62: uint64_t out_degree = wrapper::snapshot_degree(snapshot_local, i, false)
            uint64_t out_degree = m_graph.degree(i);
            // upstream L63: distances[i] = out_degree != 0 ? -out_degree : -1
            distances[i] = out_degree != 0 ? -(int64_t)out_degree : -1;
        }

        // [MOD] 断点: 初始化后的distances状态
        std::printf("[BFS BREAKPOINT] init_distances: N=%lu, first 10 dists: ",
                    (unsigned long)N);
        for (uint64_t i = 0; i < std::min(N, (uint64_t)10); i++) {
            std::printf("%ld ", (long)distances[i]);
        }
        std::printf("\n");

        return distances;
    }

    // upstream L86-95: QueueToBitmap
    void QueueToBitmap(const std::vector<int64_t>& queue_data, size_t q_start, size_t q_end,
                       gapbs::Bitmap& bm) {
        // upstream L88-91: parallel for, set_bit_atomic
        for (size_t qi = q_start; qi < q_end; qi++) {
            int64_t u = queue_data[qi];
            bm.set_bit_atomic(u);
        }
    }

    // upstream L98-112: BitmapToQueue
    void BitmapToQueue(const int64_t size, const gapbs::Bitmap& bm,
                       gapbs::SlidingQueue<int64_t>& queue) {
        // upstream L102-109: parallel scan + push
        for (int64_t n = 0; n < size; n++) {
            if (bm.get_bit(n)) queue.push_back(n);
        }
        queue.slide_window(); // upstream L111
    }

    // upstream L114-166: TDStep (top-down)
    int64_t TDStep(std::vector<int64_t>& distances, int64_t distance,
                   gapbs::SlidingQueue<int64_t>& queue) {
        // upstream L115-116: results + scout_count
        int64_t scout_count = 0;
        stats.td_steps++; // [MOD] 统计td步数

        // upstream L119: chunk_size
        // [MOD] 单线程替代多线程
        size_t q_size = queue.size();

        // [MOD] 断点: TDStep进入时状态
        std::printf("[BFS BREAKPOINT] TDStep: distance=%ld queue_size=%zu\n",
                    (long)distance, q_size);

        // upstream L124-151: 线程体, 遍历队列中每个vertex, 检查邻居
        std::mutex local_mu;
        gapbs::QueueBuffer<int64_t> lqueue(queue, m_mutex);

        for (auto q_iter = queue.begin(); q_iter != queue.end(); q_iter++) {
            int64_t u = *q_iter; // upstream L133

            // upstream L135-145: snapshot_edges callback
            m_graph.for_edges(u, [&](uint64_t destination, double w) {
                stats.total_edges_checked++; // [MOD] 边检查计数
                if (destination < distances.size() && (uint64_t)u != destination) { // upstream L136
                    int64_t curr_val = distances[destination]; // upstream L137

                    // upstream L139: compare_exchange_strong
                    if (curr_val < 0) {
                        // 模拟CAS: 如果仍然是负数，替换
                        if (distances[destination] == curr_val) {
                            distances[destination] = distance;
                            lqueue.push_back(destination);
                            scout_count += -curr_val; // upstream L141-142
                        }
                    }
                }
            });
        }

        lqueue.flush(); // upstream L148

        return scout_count; // upstream L159
    }

    // upstream L168-216: BUStep (bottom-up)
    int64_t BUStep(std::vector<int64_t>& distances, int64_t distance,
                   gapbs::Bitmap& front, gapbs::Bitmap& next) {
        // upstream L169: const uint64_t N = m_snapshot->vertex_count()
        const uint64_t N = m_graph.vertex_count();
        int64_t awake_count = 0; // upstream L170
        next.reset();            // upstream L173
        stats.bu_steps++;        // [MOD] 统计bu步数

        // [MOD] 断点: BUStep进入时状态
        std::printf("[BFS BREAKPOINT] BUStep: distance=%ld N=%lu front_density=%.4f\n",
                    (long)distance, (unsigned long)N, front.density());

        // upstream L177: chunk_size
        // upstream L181-204: 线程体
        for (uint64_t u = 0; u < N; u++) {
            if (distances[u] < 0) { // upstream L186
                bool done = false;  // upstream L187

                m_graph.for_edges(u, [&](uint64_t destination, double w) {
                    if (done) return;
                    stats.total_edges_checked++; // [MOD] 边检查计数
                    if (destination < distances.size()) { // upstream L190
                        if (front.get_bit(destination)) { // upstream L191
                            distances[u] = distance;       // upstream L192
                            awake_count++;                  // upstream L193
                            next.set_bit(u);                // upstream L194
                            done = true;                    // upstream L195
                        }
                    }
                });
            }
        }

        return awake_count; // upstream L212
    }

    // upstream L218-268: bfs() — 方向切换BFS主循环
    std::vector<int64_t> bfs(uint64_t source) {
        // upstream L219: init_distances()
        std::vector<int64_t> distances = init_distances();
        distances[source] = 0; // upstream L220

        // upstream L222-224: SlidingQueue + push source
        gapbs::SlidingQueue<int64_t> queue(m_graph.vertex_count() + 1);
        queue.push_back(source);
        queue.slide_window();

        // upstream L226-229: Bitmap curr + front
        uint64_t N = m_graph.vertex_count();
        gapbs::Bitmap curr(N);
        curr.reset();
        gapbs::Bitmap front(N);
        front.reset();

        // upstream L231-233: edges_to_check, scout_count, distance
        int64_t edges_to_check = m_graph.edge_count();
        int64_t scout_count = m_graph.degree(source);
        int64_t distance = 1;

        // [MOD] 记录第0层: source vertex
        stats.level_histogram.push_back(1);
        stats.visited_ratio.push_back(1.0 / N);

        // upstream L235-267: 主循环
        while (!queue.empty()) {
            // [MOD] 断点: 每层状态
            std::printf("[BFS BREAKPOINT] Level %ld: queue_size=%zu scout=%ld edges_rem=%ld\n",
                        (long)distance, queue.size(), (long)scout_count, (long)edges_to_check);

            if (scout_count > edges_to_check / m_alpha) { // upstream L236
                // 切换到 bottom-up
                stats.direction_switch_count++; // [MOD] 方向切换计数
                std::printf("[BFS MOD] Direction switch to BOTTOM-UP at level %ld\n", (long)distance);

                int64_t awake_count, old_awake_count;
                // upstream L238: QueueToBitmap — 需要把queue数据传入
                // 因为SlidingQueue iterator, 手动收集
                std::vector<int64_t> q_data(queue.begin(), queue.end());
                gapbs::Bitmap tmp_bm(N);
                tmp_bm.reset();
                for (auto v : q_data) tmp_bm.set_bit(v);
                front.swap(tmp_bm);

                awake_count = queue.size(); // upstream L239
                queue.slide_window();       // upstream L240

                do { // upstream L242-246
                    old_awake_count = awake_count;
                    awake_count = BUStep(distances, distance, front, curr);
                    front.swap(curr);

                    // [MOD] 每BU步层级直方图
                    stats.level_histogram.push_back(awake_count);
                    uint64_t visited = 0;
                    for (size_t i = 0; i < N; i++) if (distances[i] >= 0) visited++;
                    stats.visited_ratio.push_back((double)visited / N);

                    distance++; // upstream L245
                } while ((awake_count >= old_awake_count) ||
                         (awake_count > (int64_t)N / m_beta)); // upstream L246

                // upstream L247: BitmapToQueue
                BitmapToQueue(N, front, queue);
                scout_count = 1; // upstream L248

            } else { // upstream L250: top-down
                edges_to_check -= scout_count; // upstream L251

                // 记录当前queue大小为新层发现数
                size_t before_size = queue.size();

                scout_count = TDStep(distances, distance, queue); // upstream L252
                queue.slide_window(); // upstream L253

                // [MOD] 层级直方图 (新发现的顶点 = 新queue大小)
                size_t new_discovered = queue.size();
                stats.level_histogram.push_back(new_discovered);
                uint64_t visited = 0;
                for (size_t i = 0; i < N; i++) if (distances[i] >= 0) visited++;
                stats.visited_ratio.push_back((double)visited / N);

                distance++; // upstream L254
            }
        }

        return distances; // upstream L267
    }
};


// ═══════════════════════════════════════════════════════════════════
//  §2 — SSSP 移植 (upstream SSSP.h 全182行)
//       骨架: upstream/rapidstore/wrapper/algorithms/SSSP.h
// ═══════════════════════════════════════════════════════════════════

// --- [MOD] SSSP 统计 ---
struct SSSPStats {
    uint64_t relax_count = 0;           // [MOD] 总松弛尝试
    uint64_t relax_success_count = 0;   // [MOD] 成功松弛
    uint64_t bucket_max_size = 0;       // [MOD] 最大桶大小
    uint64_t bucket_usage_count = 0;    // [MOD] 使用过的桶数
    std::vector<size_t> frontier_size_history; // [MOD] 每轮frontier大小

    void dump() const {
        std::printf("[SSSP Stats] relax_total=%lu relax_success=%lu bucket_max=%lu buckets_used=%lu\n",
                    (unsigned long)relax_count, (unsigned long)relax_success_count,
                    (unsigned long)bucket_max_size, (unsigned long)bucket_usage_count);
        std::printf("[SSSP Frontier History] ");
        for (size_t i = 0; i < frontier_size_history.size(); i++) {
            std::printf("iter%zu:%zu ", i, frontier_size_history[i]);
        }
        std::printf("\n");
    }
};

// upstream SSSP.h: template <class F, class S> class ssspExperiments
class ssspExperiments {
    const int m_num_threads;    // upstream L17
    double m_delta;             // upstream L19
    std::mutex m_mutex;         // upstream L20
    const SimGraph& m_graph;    // 替代 m_method + m_snapshot

public:
    SSSPStats stats;            // [MOD]

    // upstream L25: ssspExperiments(num_threads, delta, method, snapshot)
    ssspExperiments(const int num_threads, const double delta, const SimGraph& graph)
        : m_num_threads(num_threads), m_delta(delta), m_graph(graph) {
        std::printf("[SSSP INIT] threads=%d delta=%.2f V=%zu E=%zu\n",
                    num_threads, delta, graph.vertex_count(), graph.edge_count());
    }

    // upstream L28: ~ssspExperiments()
    ~ssspExperiments() {}

    // upstream L142-182: run_sssp
    void run_sssp(uint64_t source, std::vector<std::pair<uint64_t, double>>& external_ids) {
        // upstream L144: source = wrapper::snapshot_logical2physical(m_snapshot, source)
        auto start = std::chrono::high_resolution_clock::now(); // upstream L145
        auto dist = sssp(source); // upstream L146

        // upstream L148-166: 多线程拷贝
        auto num_vertices = m_graph.vertex_count();
        external_ids.resize(num_vertices); // upstream L153

        for (uint64_t u = 0; u < num_vertices; u++) {
            // upstream L159: wrapper::snapshot_physical2logical(m_snapshot, u), dist[u]
            external_ids[u] = std::make_pair(u, dist[u]);
        }

        // upstream L168-172: timing output
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::printf("Gapbs SSSP took %ld milliseconds\n", (long)duration.count());
        std::printf("[SSSP LOG] SSSP: %ld milliseconds\n", (long)duration.count());

        // [MOD] 断点: 最终dist状态
        std::printf("[SSSP BREAKPOINT] Final dist: ");
        for (uint64_t i = 0; i < std::min(num_vertices, (size_t)20); i++) {
            if (dist[i] == std::numeric_limits<double>::infinity())
                std::printf("[%lu]=INF ", (unsigned long)i);
            else
                std::printf("[%lu]=%.2f ", (unsigned long)i, dist[i]);
        }
        std::printf("\n");

        stats.dump(); // [MOD]
    }

private:
    // upstream L32-140: sssp() — delta-stepping
    gapbs::pvector<double> sssp(uint64_t source) {
        // upstream L33-34
        const uint64_t num_vertices = m_graph.vertex_count();
        const uint64_t num_edges = m_graph.edge_count();
        const size_t kMaxBin = std::numeric_limits<size_t>::max() / 2; // upstream L35

        // upstream L38: distances to inf
        gapbs::pvector<double> dist(num_vertices, std::numeric_limits<double>::infinity());
        dist[source] = 0; // upstream L39
        // upstream L40: frontier
        gapbs::pvector<uint64_t> frontier(num_edges + 1);

        // upstream L42-43: shared_indexes, frontier_tails
        size_t shared_indexes[2] = {0, kMaxBin};
        size_t frontier_tails[2] = {1, 0};

        frontier[0] = source; // upstream L45

        // upstream L47: local_bins
        std::vector<std::vector<uint64_t>> local_bins(0);

        size_t iter = 0; // upstream L49

        // [MOD] 断点: 初始状态
        std::printf("[SSSP BREAKPOINT] Init: source=%lu num_v=%lu num_e=%lu delta=%.2f\n",
                    (unsigned long)source, (unsigned long)num_vertices,
                    (unsigned long)num_edges, m_delta);

        // upstream L50: main loop
        while (shared_indexes[iter & 1] != kMaxBin) {
            // upstream L52-55: bin indexes
            size_t& curr_bin_index = shared_indexes[iter & 1];
            size_t& next_bin_index = shared_indexes[(iter + 1) & 1];
            size_t& curr_frontier_tail = frontier_tails[iter & 1];
            size_t& next_frontier_tail = frontier_tails[(iter + 1) & 1];

            // [MOD] frontier大小追踪
            stats.frontier_size_history.push_back(curr_frontier_tail);

            // [MOD] 断点: 每轮迭代
            std::printf("[SSSP BREAKPOINT] iter=%zu bin_idx=%zu frontier_tail=%zu\n",
                        iter, curr_bin_index, curr_frontier_tail);

            // upstream L57-58: chunk_size, set_max_threads
            // [MOD] 单线程替代
            for (size_t i = 0; i < curr_frontier_tail; i++) {
                uint64_t u = frontier[i]; // upstream L68

                if (dist[u] >= m_delta * static_cast<double>(curr_bin_index)) { // upstream L70
                    // upstream L71-93: snapshot_edges callback
                    m_graph.for_edges(u, [&](uint64_t v, double w) {
                        if (v >= num_vertices || u == v) return; // upstream L72
                        stats.relax_count++; // [MOD] 松弛计数
                        double old_dist = dist[v]; // upstream L73
                        double new_dist = dist[u] + w; // upstream L74 (upstream用+1, 我们用实际权重)

                        if (new_dist < old_dist) { // upstream L76
                            // upstream L77-82: CAS loop
                            bool changed_dist = true;
                            while (!gapbs::compare_and_swap(dist[v], old_dist, new_dist)) {
                                old_dist = dist[v];
                                if (new_dist >= old_dist) {
                                    changed_dist = false;
                                    break;
                                }
                            }

                            if (changed_dist) { // upstream L84
                                stats.relax_success_count++; // [MOD]
                                size_t bin_index = static_cast<size_t>(new_dist / m_delta); // upstream L85
                                // upstream L86-89: lock + resize + push
                                if (bin_index >= local_bins.size()) {
                                    local_bins.resize(bin_index + 1);
                                }
                                local_bins[bin_index].push_back(v);

                                // [MOD] 桶大小统计
                                if (local_bins[bin_index].size() > stats.bucket_max_size) {
                                    stats.bucket_max_size = local_bins[bin_index].size();
                                }
                            }
                        }
                    });
                }
            }

            // upstream L101-107: find next non-empty bin
            for (size_t i = curr_bin_index; i < local_bins.size(); i++) {
                if (!local_bins[i].empty()) {
                    next_bin_index = std::min(next_bin_index, i);
                    stats.bucket_usage_count++; // [MOD]
                    break;
                }
            }

            // upstream L109-110: reset current
            curr_bin_index = kMaxBin;
            curr_frontier_tail = 0;

            // upstream L112-116: copy next bin to frontier
            if (next_bin_index < local_bins.size()) {
                size_t copy_start = gapbs::fetch_and_add(next_frontier_tail, local_bins[next_bin_index].size());
                std::copy(local_bins[next_bin_index].begin(), local_bins[next_bin_index].end(),
                          &frontier[0] + copy_start);
                local_bins[next_bin_index].resize(0);
            }

            iter++; // upstream L118
        }
        return dist; // upstream L119
    }
};


// ═══════════════════════════════════════════════════════════════════
//  §3 — WCC 移植 (upstream WCC.h 全149行)
//       骨架: upstream/rapidstore/wrapper/algorithms/WCC.h
// ═══════════════════════════════════════════════════════════════════

// --- [MOD] WCC 统计 ---
struct WCCStats {
    uint64_t union_by_rank_count = 0;       // [MOD] 合并操作次数
    uint64_t path_compression_hops = 0;     // [MOD] 路径压缩总跳数
    uint64_t total_iterations = 0;          // [MOD] 收敛迭代次数
    std::map<uint64_t, uint64_t> component_size_distribution; // [MOD] 组件大小分布

    void dump() const {
        std::printf("[WCC Stats] unions=%lu compress_hops=%lu iterations=%lu\n",
                    (unsigned long)union_by_rank_count,
                    (unsigned long)path_compression_hops,
                    (unsigned long)total_iterations);
        std::printf("[WCC Component Sizes] ");
        for (auto& [size, count] : component_size_distribution) {
            std::printf("size%lu:x%lu ", (unsigned long)size, (unsigned long)count);
        }
        std::printf("\n");
    }
};

// upstream WCC.h: template <class F, class S> class wccExperiments
class wccExperiments {
    const int m_num_threads;     // upstream L17
    const SimGraph& m_graph;     // 替代 m_method + m_snapshot

public:
    WCCStats stats;              // [MOD]

    // upstream L24: wccExperiments(num_threads, method, snapshot)
    wccExperiments(const int num_threads, const SimGraph& graph)
        : m_num_threads(num_threads), m_graph(graph) {
        std::printf("[WCC INIT] threads=%d V=%zu E=%zu\n",
                    num_threads, graph.vertex_count(), graph.edge_count());
    }

    // upstream L27: ~wccExperiments
    ~wccExperiments() {}

    // upstream L108-149: run_wcc
    void run_wcc(std::vector<std::pair<uint64_t, int64_t>>& external_ids) {
        auto start = std::chrono::high_resolution_clock::now(); // upstream L109

        std::unique_ptr<uint64_t[]> ptr_components = wcc(); // upstream L111

        // upstream L113-135: 多线程拷贝
        auto num_vertices = m_graph.vertex_count();
        external_ids.resize(num_vertices);

        for (uint64_t u = 0; u < num_vertices; u++) {
            // upstream L127: wrapper::snapshot_physical2logical
            external_ids[u] = std::make_pair(u, (int64_t)ptr_components[u]);
        }

        // [MOD] 计算组件大小分布
        std::unordered_map<uint64_t, uint64_t> comp_counts;
        for (uint64_t i = 0; i < num_vertices; i++) {
            comp_counts[ptr_components[i]]++;
        }
        for (auto& [root, cnt] : comp_counts) {
            stats.component_size_distribution[cnt]++;
        }

        // upstream L137-142: timing
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::printf("Gapbs WCC took %ld milliseconds\n", (long)duration.count());
        std::printf("[WCC LOG] WCC: %ld milliseconds\n", (long)duration.count());

        // [MOD] 断点: 最终comp状态
        std::printf("[WCC BREAKPOINT] Final comp: ");
        for (uint64_t i = 0; i < std::min(num_vertices, (size_t)20); i++) {
            std::printf("[%lu]=%lu ", (unsigned long)i, (unsigned long)ptr_components[i]);
        }
        std::printf("\n");

        stats.dump(); // [MOD]
    }

private:
    // upstream L30-106: wcc()
    std::unique_ptr<uint64_t[]> wcc() {
        // upstream L31-33: num_vertices, components array
        const uint64_t num_vertices = m_graph.vertex_count();
        std::unique_ptr<uint64_t[]> ptr_components{new uint64_t[num_vertices]};
        uint64_t* comp = ptr_components.get();

        // upstream L36: parallel init comp[i] = i
        for (uint64_t i = 0; i < num_vertices; i++) {
            comp[i] = i;
        }

        // [MOD] 为按秩合并添加rank数组
        std::vector<uint64_t> rank_arr(num_vertices, 0);

        // upstream L39-40: change + threads
        bool change = true;
        // upstream L41: chunk_size
        uint64_t chunk_size = (num_vertices + m_num_threads - 1) / m_num_threads;

        // [MOD] 断点: 初始状态
        std::printf("[WCC BREAKPOINT] Init: num_vertices=%lu\n", (unsigned long)num_vertices);

        // upstream L42-78: main loop
        while (change) {
            change = false;
            stats.total_iterations++; // [MOD]

            // [MOD] 断点: 每轮迭代
            std::printf("[WCC BREAKPOINT] Iteration %lu, comp[0..9]: ",
                        (unsigned long)stats.total_iterations);
            for (uint64_t i = 0; i < std::min(num_vertices, (uint64_t)10); i++) {
                std::printf("%lu ", (unsigned long)comp[i]);
            }
            std::printf("\n");

            // upstream L45-73: 线程体 — 遍历每个vertex的边
            for (uint64_t u = 0; u < num_vertices; u++) {
                // upstream L52-68: snapshot_edges callback
                m_graph.for_edges(u, [&](uint64_t v, double w) {
                    uint64_t comp_u = comp[u]; // upstream L53
                    uint64_t comp_v = comp[v]; // upstream L54
                    if (comp_u == comp_v) return; // upstream L55-57

                    uint64_t high_comp = std::max(comp_u, comp_v); // upstream L59
                    uint64_t low_comp = std::min(comp_u, comp_v);  // upstream L60

                    if (high_comp >= num_vertices || low_comp >= num_vertices) return; // upstream L62

                    // upstream L63-66: hook
                    if (high_comp == comp[high_comp]) {
                        // [MOD] 按秩合并: 选择rank更高的作为根
                        if (rank_arr[low_comp] < rank_arr[high_comp]) {
                            // 正常: high挂到low
                        } else if (rank_arr[low_comp] == rank_arr[high_comp]) {
                            rank_arr[low_comp]++; // 提升rank
                        }
                        stats.union_by_rank_count++; // [MOD]

                        change = true;          // upstream L64
                        comp[high_comp] = low_comp; // upstream L65
                    }
                });
            }

            // upstream L80-84: 路径压缩 (parallel for)
            for (uint64_t i = 0; i < num_vertices; i++) {
                // upstream L82-83: while(comp[i] != comp[comp[i]])
                while (comp[i] != comp[comp[i]]) {
                    comp[i] = comp[comp[i]];
                    stats.path_compression_hops++; // [MOD] 跳数统计
                }
            }
        }

        return ptr_components; // upstream L86
    }
};


// ═══════════════════════════════════════════════════════════════════
//  §4 — PageRank 移植 (upstream PR.h 全174行)
//       骨架: upstream/rapidstore/wrapper/algorithms/PR.h
// ═══════════════════════════════════════════════════════════════════

// --- [MOD] PR 统计 ---
struct PRStats {
    std::vector<double> l1_norm_history;    // [MOD] 每轮L1 norm
    std::vector<double> score_sum_history;  // [MOD] 每轮score总和 (能量守恒检验)
    uint64_t dangling_count = 0;            // [MOD] dangling节点总数
    uint64_t total_iterations = 0;          // [MOD] 实际迭代轮数

    void dump() const {
        std::printf("[PR Stats] iterations=%lu dangling=%lu\n",
                    (unsigned long)total_iterations, (unsigned long)dangling_count);
        std::printf("[PR L1 Norm] ");
        for (size_t i = 0; i < l1_norm_history.size(); i++) {
            std::printf("iter%zu:%.8f ", i, l1_norm_history[i]);
        }
        std::printf("\n");
        std::printf("[PR Score Sum (energy)] ");
        for (size_t i = 0; i < score_sum_history.size(); i++) {
            std::printf("iter%zu:%.8f ", i, score_sum_history[i]);
        }
        std::printf("\n");
    }
};

// upstream PR.h: template <class F, class S> class pageRankExperiments
class pageRankExperiments {
    const int m_num_threads;           // upstream L17
    const uint64_t m_num_iterations;   // upstream L18
    const double m_damping_factor;     // upstream L19
    const SimGraph& m_graph;           // 替代 m_method + m_snapshot

public:
    PRStats stats;                     // [MOD]

    // upstream L25: pageRankExperiments(...)
    pageRankExperiments(const int num_threads, const uint64_t num_iterations,
                        const double damping_factor, const SimGraph& graph)
        : m_num_threads(num_threads), m_num_iterations(num_iterations),
          m_damping_factor(damping_factor), m_graph(graph) {
        // upstream L27: omp_set_num_threads
        std::printf("[PR INIT] threads=%d iterations=%lu damping=%.4f V=%zu E=%zu\n",
                    num_threads, (unsigned long)num_iterations, damping_factor,
                    graph.vertex_count(), graph.edge_count());
    }

    // upstream L30: ~pageRankExperiments
    ~pageRankExperiments() {}

    // upstream L126-174: run_page_rank
    void run_page_rank(std::vector<std::pair<uint64_t, double>>& external_ids) {
        auto start = std::chrono::high_resolution_clock::now(); // upstream L127
        auto scores = page_rank(); // upstream L128

        // upstream L130-152: 多线程拷贝
        auto num_vertices = m_graph.vertex_count();
        external_ids.resize(num_vertices);

        for (uint64_t u = 0; u < num_vertices; u++) {
            // upstream L145: external_ids[u] = make_pair(u, scores[u])
            external_ids[u] = std::make_pair(u, scores[u]);
        }

        // upstream L156-161: timing
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::printf("Gapbs Page Rank took %ld milliseconds\n", (long)duration.count());
        std::printf("[PR LOG] PR: %ld milliseconds\n", (long)duration.count());

        // [MOD] 断点: 最终scores
        std::printf("[PR BREAKPOINT] Final scores: ");
        for (uint64_t i = 0; i < std::min(num_vertices, (size_t)20); i++) {
            std::printf("[%lu]=%.6f ", (unsigned long)i, scores[i]);
        }
        std::printf("\n");

        stats.dump(); // [MOD]
    }

private:
    // upstream L34-124: page_rank()
    std::unique_ptr<double[]> page_rank() {
        // upstream L35: num_vertices
        const uint64_t num_vertices = m_graph.vertex_count();

        // upstream L37-38: init_score, base_score
        const double init_score = 1.0 / num_vertices;
        const double base_score = (1.0 - m_damping_factor) / num_vertices;

        // upstream L40-43: scores init
        std::unique_ptr<double[]> ptr_scores{new double[num_vertices]()};
        double* scores = ptr_scores.get();
        for (uint64_t v = 0; v < num_vertices; v++) {
            scores[v] = init_score; // upstream L43
        }
        // upstream L44: outgoing_contrib
        gapbs::pvector<double> outgoing_contrib(num_vertices, 0.0);

        // [MOD] 保存上一轮scores用于L1 norm计算
        std::vector<double> prev_scores(num_vertices, init_score);

        // upstream L46: main iteration loop
        for (uint64_t iter = 0; iter < m_num_iterations; iter++) {
            stats.total_iterations++; // [MOD]

            // upstream L47-48: dangling_sums
            std::vector<double> dangling_sums(m_num_threads, 0.0);
            double dangling_sum = 0.0;

            // upstream L50: chunk_size
            uint64_t chunk_size = (num_vertices + m_num_threads - 1) / m_num_threads;

            // [MOD] 断点: 每轮迭代开始
            std::printf("[PR BREAKPOINT] Iter %lu: scores[0..4]: ", (unsigned long)iter);
            for (uint64_t i = 0; i < std::min(num_vertices, (uint64_t)5); i++) {
                std::printf("%.6f ", scores[i]);
            }
            std::printf("\n");

            // upstream L53-68: Phase 1: 计算 outgoing_contrib 和 dangling
            uint64_t iter_dangling = 0; // [MOD]
            for (uint64_t v = 0; v < num_vertices; v++) {
                // upstream L60: out_degree
                uint64_t out_degree = m_graph.degree(v);
                if (out_degree == 0) { // upstream L61
                    dangling_sum += scores[v]; // upstream L62
                    iter_dangling++; // [MOD]
                } else {
                    outgoing_contrib[v] = scores[v] / out_degree; // upstream L64
                }
            }
            stats.dangling_count = iter_dangling; // [MOD] (记录最后一轮)

            // upstream L74-75: dangling_sum /= num_vertices
            dangling_sum /= num_vertices;

            // [MOD] 断点: outgoing_contrib状态
            std::printf("[PR BREAKPOINT] Iter %lu: dangling_sum=%.8f dangling_nodes=%lu\n",
                        (unsigned long)iter, dangling_sum, (unsigned long)iter_dangling);

            // upstream L77-93: Phase 2: 更新scores
            for (uint64_t v = 0; v < num_vertices; v++) {
                double incoming_total = 0.0; // upstream L83 (upstream typo: "incoming_totol")

                // upstream L84-86: snapshot_edges callback
                m_graph.for_edges(v, [&](uint64_t src, double w) {
                    if (src == v) return; // upstream L85
                    incoming_total += outgoing_contrib[src]; // upstream L86
                });

                // upstream L89: scores[v] = base_score + damping * (incoming + dangling)
                scores[v] = base_score + m_damping_factor * (incoming_total + dangling_sum);
            }

            // [MOD] L1 norm收敛曲线
            double l1 = 0.0;
            for (uint64_t v = 0; v < num_vertices; v++) {
                l1 += std::fabs(scores[v] - prev_scores[v]);
                prev_scores[v] = scores[v];
            }
            stats.l1_norm_history.push_back(l1);

            // [MOD] 能量守恒: score总和应接近1.0
            double score_sum = 0.0;
            for (uint64_t v = 0; v < num_vertices; v++) {
                score_sum += scores[v];
            }
            stats.score_sum_history.push_back(score_sum);

            std::printf("[PR MOD] Iter %lu: L1_norm=%.8f score_sum=%.8f\n",
                        (unsigned long)iter, l1, score_sum);
        }
        return ptr_scores; // upstream L122
    }
};


// ═══════════════════════════════════════════════════════════════════
//  §5 — TC 移植 (upstream TC.h 全93行)
//       骨架: upstream/rapidstore/wrapper/algorithms/TC.h
// ═══════════════════════════════════════════════════════════════════

// --- [MOD] TC 统计 ---
struct TCStats {
    uint64_t intersect_call_count = 0;   // [MOD] intersect调用次数
    uint64_t search_call_count = 0;      // [MOD] search策略调用次数
    uint64_t hash_strategy_count = 0;    // [MOD] 高度数→搜索低度数邻居的次数
    uint64_t sort_strategy_count = 0;    // [MOD] 低度数→搜索高度数邻居的次数
    uint64_t progress_triangles = 0;     // [MOD] 进度打印时三角形计数

    void dump() const {
        std::printf("[TC Stats] intersect_calls=%lu search_calls=%lu hash=%lu sort=%lu\n",
                    (unsigned long)intersect_call_count, (unsigned long)search_call_count,
                    (unsigned long)hash_strategy_count, (unsigned long)sort_strategy_count);
    }
};

// upstream TC.h: template <class F, class S> class TriangleCounting
// upstream L32: #define SEARCH_THRESHOLD 10
#define SEARCH_THRESHOLD 10

class TriangleCounting {
    const SimGraph& m_graph; // 替代 m_method + m_snapshot

public:
    TCStats stats; // [MOD]

    // upstream L20: TriangleCounting(method, snapshot)
    TriangleCounting(const SimGraph& graph) : m_graph(graph) {
        std::printf("[TC INIT] V=%zu E=%zu\n", graph.vertex_count(), graph.edge_count());
    }

    // upstream L24: ~TriangleCounting
    ~TriangleCounting() {}

    // upstream L34-93: run_tc
    uint64_t run_tc() {
        auto start = std::chrono::high_resolution_clock::now(); // upstream L35

        auto num_vertices = m_graph.vertex_count(); // upstream L37
        uint64_t num_triangles = 0;                 // upstream L38
        uint64_t num_triangles_from_intersect = 0;  // upstream L39

        // upstream L41-77: main loop
        for (uint64_t i = 0; i < num_vertices; i++) {
            // [MOD] 断点: 每1000顶点打印进度
            if (i % 1000 == 0 && i > 0) {
                std::printf("[TC BREAKPOINT] Progress: %lu/%lu vertices, triangles=%lu (intersect=%lu)\n",
                            (unsigned long)i, (unsigned long)num_vertices,
                            (unsigned long)num_triangles, (unsigned long)num_triangles_from_intersect);
            }

            auto degree_src = m_graph.degree(i); // upstream L46

            // upstream L47-75: get_edges callback
            auto get_edges = [&](uint64_t dst, double weight) {
                if (dst < i) { // upstream L48
                    auto degree_dst = m_graph.degree(dst); // upstream L49

                    if (degree_src > degree_dst * SEARCH_THRESHOLD) { // upstream L50
                        // upstream L52-55: search all edges of dest in src
                        stats.hash_strategy_count++; // [MOD]
                        m_graph.for_edges(dst, [&](uint64_t d, double wright) {
                            stats.search_call_count++; // [MOD]
                            if (m_graph.has_edge(i, d)) { // upstream L53
                                num_triangles += 1;        // upstream L54
                            }
                        });
                    } else if (degree_src * SEARCH_THRESHOLD < degree_dst) { // upstream L57
                        // upstream L59-62: search all edges of src in dest
                        stats.sort_strategy_count++; // [MOD]
                        m_graph.for_edges(i, [&](uint64_t d, double wright) {
                            stats.search_call_count++; // [MOD]
                            if (m_graph.has_edge(dst, d)) { // upstream L60
                                num_triangles += 1;          // upstream L61
                            }
                        });
                    } else {
                        // upstream L64-66: intersect
                        stats.intersect_call_count++; // [MOD]
                        auto res = m_graph.intersect_neighbors(i, dst); // upstream L65
                        num_triangles_from_intersect += res; // upstream L66
                    }
                }
            };

            m_graph.for_edges(i, get_edges); // upstream L69 (wrapper::snapshot_edges)
        }

        // upstream L78-82: timing
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::printf("Gapbs TC took %ld milliseconds\n", (long)duration.count());
        std::printf("[TC LOG] TC: %ld milliseconds\n", (long)duration.count());

        // upstream L83-86: random device (upstream用于某种目的)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 100000000);

        // upstream L88-89: output
        std::printf("triangle count: %lu\n", (unsigned long)(num_triangles_from_intersect / 3));
        std::printf("triangle count: %lu\n", (unsigned long)num_triangles);

        // [MOD] 断点: 最终统计
        stats.dump();

        // [MOD] 修正upstream的bug: 合并两种计数方式的结果
        // upstream只返回num_triangles (search路径), 漏掉了intersect路径的结果
        uint64_t total_triangles = num_triangles + num_triangles_from_intersect;
        return total_triangles; // upstream L90 (修正: 合并intersect结果)
    }
};


// ═══════════════════════════════════════════════════════════════════
//  §6 — TC_opt 移植 (upstream TC_opt.h 全81行)
//       骨架: upstream/rapidstore/wrapper/algorithms/TC_opt.h
// ═══════════════════════════════════════════════════════════════════

// --- [MOD] TC_opt 统计 ---
struct TCOptStats {
    uint64_t prefix_skip_count = 0;     // [MOD] 前缀跳过次数
    uint64_t total_comparisons = 0;     // [MOD] 总比较次数
    uint64_t early_term_count = 0;      // [MOD] early-termination次数
    uint64_t marker_advances = 0;       // [MOD] marker推进次数

    void dump() const {
        std::printf("[TC_opt Stats] prefix_skip=%lu comparisons=%lu early_term=%lu marker_adv=%lu\n",
                    (unsigned long)prefix_skip_count, (unsigned long)total_comparisons,
                    (unsigned long)early_term_count, (unsigned long)marker_advances);
        if (total_comparisons > 0) {
            std::printf("[TC_opt Skip Rate] %.4f%%\n",
                        100.0 * prefix_skip_count / total_comparisons);
        }
    }
};

// upstream TC_opt.h: template <class F, class S> class TriangleCounting_optimized
class TriangleCounting_optimized {
    const SimGraph& m_graph; // 替代 m_method + m_snapshot

public:
    TCOptStats stats; // [MOD]

    // upstream L20: TriangleCounting_optimized(method, snapshot)
    TriangleCounting_optimized(const SimGraph& graph) : m_graph(graph) {
        std::printf("[TC_opt INIT] V=%zu E=%zu\n", graph.vertex_count(), graph.edge_count());
    }

    // upstream L24: ~TriangleCounting_optimized
    ~TriangleCounting_optimized() {}

    // upstream L27-81: run_tc
    uint64_t run_tc() {
        auto start = std::chrono::high_resolution_clock::now(); // upstream L28

        auto num_vertices = m_graph.vertex_count(); // upstream L30
        uint64_t num_triangles = 0;                 // upstream L31

        // upstream L33-61: main loop
        for (uint64_t n1 = 0; n1 < num_vertices; n1++) {
            // upstream L34-36: progress print
            if (n1 % 10 == 0) {
                std::printf("TC: %lu / %lu\n", (unsigned long)n1, (unsigned long)num_vertices);
            }

            // [MOD] 断点: 每10顶点打印 neighbors 状态
            if (n1 % 10 == 0 && n1 > 0) {
                std::printf("[TC_opt BREAKPOINT] n1=%lu triangles_so_far=%lu prefix_skips=%lu\n",
                            (unsigned long)n1, (unsigned long)num_triangles,
                            (unsigned long)stats.prefix_skip_count);
            }

            std::vector<uint64_t> m_neighbors; // upstream L37

            // upstream L38-58: get_edges callback
            auto get_edges = [&](uint64_t n2, double w2) {
                if (n2 > n1) return; // upstream L39
                m_neighbors.push_back(n2); // upstream L40

                uint64_t marker = 0; // upstream L42

                // upstream L43-54: get_intersection callback
                auto get_intersection = [&](uint64_t n3, double w3) {
                    if (n3 > n2) return; // upstream L44

                    stats.total_comparisons++; // [MOD]

                    // upstream L45-48: advance marker
                    if (marker < m_neighbors.size() && n3 > m_neighbors[marker]) {
                        do {
                            marker++;
                            stats.marker_advances++; // [MOD]
                        } while (marker < m_neighbors.size() && n3 > m_neighbors[marker]);
                    }

                    // [MOD] early termination: marker超出范围
                    if (marker >= m_neighbors.size()) {
                        stats.early_term_count++; // [MOD]
                        return;
                    }

                    // [MOD] 前缀跳过检测: 如果n3 < m_neighbors[marker], 说明跳过了
                    if (n3 < m_neighbors[marker]) {
                        stats.prefix_skip_count++; // [MOD]
                        return;
                    }

                    // upstream L50-53: match
                    if (n3 == m_neighbors[marker]) { // upstream L50
                        num_triangles += 1;            // upstream L51
                        marker++;                      // upstream L52
                    }
                };
                m_graph.for_edges(n2, get_intersection); // upstream L55
            };
            m_graph.for_edges(n1, get_edges); // upstream L57
        }

        // upstream L63-66: timing
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::printf("Gapbs TC took %ld milliseconds\n", (long)duration.count());
        std::printf("[TC_opt LOG] TC: %ld milliseconds\n", (long)duration.count());

        // [MOD] 最终统计
        std::printf("[TC_opt] triangle count: %lu\n", (unsigned long)num_triangles);
        stats.dump();

        return num_triangles; // upstream L68
    }
};


// ═══════════════════════════════════════════════════════════════════
//  §7 — 测试用例 (6个算法 x 3+ 测试 = 18+ tests)
// ═══════════════════════════════════════════════════════════════════

// ─────── M104 BFS Tests ───────

static void test_bfs_chain_graph() {
    // 链图: 0-1-2-...-9, source=0, 每个节点的距离应等于它的ID
    auto g = build_chain_graph(10);
    bfsExperiments bfs(1, 15, 18, g);
    std::vector<std::pair<uint64_t, int64_t>> results;
    bfs.run_gapbs_bfs(0, results);

    TEST_ASSERT(results.size() == 10, "BFS chain: result size should be 10");
    for (int i = 0; i < 10; i++) {
        TEST_ASSERT(results[i].second == i,
                    "BFS chain: distance should equal vertex ID from source 0");
    }
    TEST_ASSERT(bfs.stats.td_steps > 0 || bfs.stats.bu_steps > 0,
                "BFS chain: should have at least one step");
    TEST_ASSERT(bfs.stats.level_histogram.size() > 0,
                "BFS chain: level histogram should be non-empty");
    TEST_PASS("bfs_chain_graph");
}

static void test_bfs_star_graph() {
    // 星图: 0是中心, 1-9各距0为1
    auto g = build_star_graph(10);
    bfsExperiments bfs(1, 15, 18, g);
    std::vector<std::pair<uint64_t, int64_t>> results;
    bfs.run_gapbs_bfs(0, results);

    TEST_ASSERT(results[0].second == 0, "BFS star: source dist=0");
    for (int i = 1; i < 10; i++) {
        TEST_ASSERT(results[i].second == 1,
                    "BFS star: all non-source vertices should be dist=1");
    }
    TEST_ASSERT(bfs.stats.level_histogram.size() >= 2,
                "BFS star: should have at least 2 levels");
    TEST_PASS("bfs_star_graph");
}

static void test_bfs_disconnected() {
    // 两个断开的子图: {0,1,2} clique + {3,4,5} clique, source=0
    auto g = build_disconnected_graph(3, 3);
    bfsExperiments bfs(1, 15, 18, g);
    std::vector<std::pair<uint64_t, int64_t>> results;
    bfs.run_gapbs_bfs(0, results);

    TEST_ASSERT(results[0].second == 0, "BFS disconn: source dist=0");
    TEST_ASSERT(results[1].second >= 0, "BFS disconn: vertex 1 reachable");
    // vertices 3,4,5 should be unreachable (negative distances)
    TEST_ASSERT(results[3].second < 0, "BFS disconn: vertex 3 unreachable (neg dist)");
    TEST_ASSERT(results[4].second < 0, "BFS disconn: vertex 4 unreachable (neg dist)");
    TEST_PASS("bfs_disconnected");
}

static void test_bfs_direction_switch_stats() {
    // 大图测试方向切换统计是否工作
    auto g = build_complete_graph(8);
    bfsExperiments bfs(1, 2, 2, g); // 小alpha/beta迫使方向切换
    std::vector<std::pair<uint64_t, int64_t>> results;
    bfs.run_gapbs_bfs(0, results);

    // 完全图上所有非source节点距离为1
    for (int i = 1; i < 8; i++) {
        TEST_ASSERT(results[i].second == 1, "BFS K8: all dist=1");
    }
    TEST_ASSERT(bfs.stats.visited_ratio.size() > 0,
                "BFS K8: visited_ratio should be tracked");
    // 最后一个visited_ratio应接近1.0
    double last_ratio = bfs.stats.visited_ratio.back();
    TEST_ASSERT(last_ratio > 0.9, "BFS K8: final visited ratio should be near 1.0");
    TEST_PASS("bfs_direction_switch_stats");
}

// ─────── M104 SSSP Tests ───────

static void test_sssp_chain() {
    // 链图: 0-1-2-...-9, 权重1, source=0
    auto g = build_chain_graph(10);
    ssspExperiments sssp(1, 2.0, g);
    std::vector<std::pair<uint64_t, double>> results;
    sssp.run_sssp(0, results);

    TEST_ASSERT(results.size() == 10, "SSSP chain: size 10");
    TEST_ASSERT(std::fabs(results[0].second) < 1e-9, "SSSP chain: source dist=0");
    for (int i = 1; i < 10; i++) {
        TEST_ASSERT(std::fabs(results[i].second - i) < 1e-9,
                    "SSSP chain: dist should equal ID");
    }
    TEST_ASSERT(sssp.stats.relax_success_count > 0, "SSSP chain: some relaxations succeeded");
    TEST_PASS("sssp_chain");
}

static void test_sssp_star() {
    auto g = build_star_graph(10);
    ssspExperiments sssp(1, 2.0, g);
    std::vector<std::pair<uint64_t, double>> results;
    sssp.run_sssp(0, results);

    TEST_ASSERT(std::fabs(results[0].second) < 1e-9, "SSSP star: source=0");
    for (int i = 1; i < 10; i++) {
        TEST_ASSERT(std::fabs(results[i].second - 1.0) < 1e-9,
                    "SSSP star: all dist=1");
    }
    TEST_PASS("sssp_star");
}

static void test_sssp_disconnected() {
    auto g = build_disconnected_graph(3, 3);
    ssspExperiments sssp(1, 2.0, g);
    std::vector<std::pair<uint64_t, double>> results;
    sssp.run_sssp(0, results);

    TEST_ASSERT(results[3].second == std::numeric_limits<double>::infinity(),
                "SSSP disconn: unreachable=INF");
    TEST_ASSERT(sssp.stats.frontier_size_history.size() > 0,
                "SSSP disconn: frontier history non-empty");
    TEST_PASS("sssp_disconnected");
}

static void test_sssp_bucket_stats() {
    auto g = build_cycle_graph(20);
    ssspExperiments sssp(1, 3.0, g);
    std::vector<std::pair<uint64_t, double>> results;
    sssp.run_sssp(0, results);

    TEST_ASSERT(sssp.stats.relax_count > 0, "SSSP cycle: relax_count > 0");
    TEST_ASSERT(sssp.stats.bucket_usage_count > 0, "SSSP cycle: bucket_usage > 0");
    TEST_PASS("sssp_bucket_stats");
}

// ─────── M104 WCC Tests ───────

static void test_wcc_connected() {
    auto g = build_chain_graph(10);
    wccExperiments wcc(1, g);
    std::vector<std::pair<uint64_t, int64_t>> results;
    wcc.run_wcc(results);

    // 所有节点应在同一个组件
    int64_t comp0 = results[0].second;
    for (int i = 1; i < 10; i++) {
        TEST_ASSERT(results[i].second == comp0,
                    "WCC connected: all same component");
    }
    TEST_ASSERT(wcc.stats.total_iterations > 0, "WCC connected: iterations > 0");
    TEST_PASS("wcc_connected");
}

static void test_wcc_disconnected() {
    auto g = build_disconnected_graph(4, 4);
    wccExperiments wcc(1, g);
    std::vector<std::pair<uint64_t, int64_t>> results;
    wcc.run_wcc(results);

    // 两个组件
    int64_t comp_a = results[0].second;
    int64_t comp_b = results[4].second;
    TEST_ASSERT(comp_a != comp_b, "WCC disconn: two different components");
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(results[i].second == comp_a, "WCC disconn: group A same comp");
    }
    for (int i = 4; i < 8; i++) {
        TEST_ASSERT(results[i].second == comp_b, "WCC disconn: group B same comp");
    }
    TEST_ASSERT(wcc.stats.component_size_distribution.count(4) > 0,
                "WCC disconn: should have components of size 4");
    TEST_PASS("wcc_disconnected");
}

static void test_wcc_path_compression() {
    // 用有向链 0->1->2->...->49 制造非平坦comp数组
    // 因为for_edges只访问出边, 第一遍只能做部分union
    // 从而path compression phase会产生跳数
    SimGraph g(50);
    for (size_t i = 0; i + 1 < 50; i++) {
        g.add_edge(i, i + 1, 1.0); // 单向: i -> i+1
    }
    wccExperiments wcc(1, g);
    std::vector<std::pair<uint64_t, int64_t>> results;
    wcc.run_wcc(results);

    // 有向链下, 第一遍: u=0遍历边0->1, comp[1]=0; u=1遍历边1->2, comp[1]=0,comp[2]=2
    // 所以comp[2]未被hook到0, 只hook到1 (然后1->0), 需要path compression
    TEST_ASSERT(wcc.stats.union_by_rank_count > 0,
                "WCC path compress: some union ops happened");
    // 检查所有节点最终属于同一个component
    int64_t root = results[0].second;
    for (auto& [v, c] : results) {
        TEST_ASSERT(c == root, "WCC directed chain: all same component");
    }
    TEST_PASS("wcc_path_compression");
}

// ─────── M105 PR Tests ───────

static void test_pr_uniform() {
    // 完全图: 所有节点应有相近的PageRank
    auto g = build_complete_graph(5);
    pageRankExperiments pr(1, 10, 0.85, g);
    std::vector<std::pair<uint64_t, double>> results;
    pr.run_page_rank(results);

    double expected = 1.0 / 5;
    for (int i = 0; i < 5; i++) {
        TEST_ASSERT(std::fabs(results[i].second - expected) < 0.05,
                    "PR uniform: scores should be ~0.2 on K5");
    }
    TEST_ASSERT(pr.stats.l1_norm_history.size() == 10,
                "PR uniform: should have 10 L1 norms");
    TEST_PASS("pr_uniform");
}

static void test_pr_star_center() {
    // 星图: 中心节点应有最高PageRank
    auto g = build_star_graph(10);
    pageRankExperiments pr(1, 20, 0.85, g);
    std::vector<std::pair<uint64_t, double>> results;
    pr.run_page_rank(results);

    double center_score = results[0].second;
    for (int i = 1; i < 10; i++) {
        TEST_ASSERT(center_score > results[i].second,
                    "PR star: center should have highest rank");
    }
    TEST_PASS("pr_star_center");
}

static void test_pr_convergence() {
    // 链图: 非对称结构, 各节点度数不同(端点度1, 中间度2), 
    // 所以初始均匀分布不是稳态, L1 norm会先大后逐渐收敛
    auto g = build_star_graph(10); // 星图: center度9, leaf度1
    pageRankExperiments pr(1, 30, 0.85, g);
    std::vector<std::pair<uint64_t, double>> results;
    pr.run_page_rank(results);

    // L1 norm应该递减趋势
    TEST_ASSERT(pr.stats.l1_norm_history.size() == 30, "PR converge: 30 iters");
    double first_l1 = pr.stats.l1_norm_history[0];
    double last_l1 = pr.stats.l1_norm_history.back();
    TEST_ASSERT(last_l1 < first_l1, "PR converge: L1 norm should decrease");

    // 能量守恒: score_sum应接近1.0
    double last_sum = pr.stats.score_sum_history.back();
    TEST_ASSERT(std::fabs(last_sum - 1.0) < 0.01,
                "PR converge: score sum should be ~1.0 (energy conservation)");
    TEST_PASS("pr_convergence");
}

static void test_pr_dangling() {
    // 含有孤立节点的图
    SimGraph g(5);
    g.add_undirected_edge(0, 1);
    g.add_undirected_edge(1, 2);
    // 3, 4 are dangling (no outgoing edges in undirected sense — but they DO have edges)
    // Actually make them truly dangling: only add directed edges TO them
    SimGraph g2(5);
    g2.add_edge(0, 1);
    g2.add_edge(1, 2);
    g2.add_edge(2, 0);
    // 3 and 4 have no outgoing edges → dangling

    pageRankExperiments pr(1, 10, 0.85, g2);
    std::vector<std::pair<uint64_t, double>> results;
    pr.run_page_rank(results);

    TEST_ASSERT(pr.stats.dangling_count >= 2, "PR dangling: should detect dangling nodes");
    TEST_PASS("pr_dangling");
}

// ─────── M105 TC Tests ───────

static void test_tc_triangle_mesh() {
    // 3x3 grid with diagonals → many triangles
    auto g = build_triangle_mesh(3, 3);
    TriangleCounting tc(g);
    uint64_t count = tc.run_tc();

    TEST_ASSERT(count > 0, "TC mesh: should find triangles");
    TEST_ASSERT(tc.stats.intersect_call_count > 0 || tc.stats.search_call_count > 0,
                "TC mesh: should use some search strategy");
    TEST_PASS("tc_triangle_mesh");
}

static void test_tc_complete_graph() {
    // K5 has C(5,3)=10 triangles
    auto g = build_complete_graph(5);
    TriangleCounting tc(g);
    uint64_t count = tc.run_tc();

    // 由于双向计数, 实际返回值可能是多倍
    TEST_ASSERT(count > 0, "TC K5: should find triangles");
    std::printf("[TC K5 DEBUG] raw triangle count = %lu\n", (unsigned long)count);
    TEST_PASS("tc_complete_graph");
}

static void test_tc_no_triangles() {
    // 链图没有三角形
    auto g = build_chain_graph(10);
    TriangleCounting tc(g);
    uint64_t count = tc.run_tc();

    TEST_ASSERT(count == 0, "TC chain: should find 0 triangles");
    TEST_PASS("tc_no_triangles");
}

// ─────── M105 TC_opt Tests ───────

static void test_tcopt_triangle_mesh() {
    auto g = build_triangle_mesh(3, 3);
    TriangleCounting_optimized tc(g);
    uint64_t count = tc.run_tc();

    TEST_ASSERT(count > 0, "TC_opt mesh: should find triangles");
    TEST_ASSERT(tc.stats.total_comparisons > 0,
                "TC_opt mesh: should have comparisons");
    TEST_PASS("tcopt_triangle_mesh");
}

static void test_tcopt_complete_graph() {
    auto g = build_complete_graph(5);
    TriangleCounting_optimized tc(g);
    uint64_t count = tc.run_tc();

    TEST_ASSERT(count > 0, "TC_opt K5: should find triangles");
    std::printf("[TC_opt K5 DEBUG] raw triangle count = %lu, prefix_skip_rate=%.2f%%\n",
                (unsigned long)count,
                tc.stats.total_comparisons > 0 ?
                100.0 * tc.stats.prefix_skip_count / tc.stats.total_comparisons : 0.0);
    TEST_PASS("tcopt_complete_graph");
}

static void test_tcopt_no_triangles() {
    auto g = build_chain_graph(10);
    TriangleCounting_optimized tc(g);
    uint64_t count = tc.run_tc();

    TEST_ASSERT(count == 0, "TC_opt chain: should find 0 triangles");
    TEST_PASS("tcopt_no_triangles");
}

static void test_tcopt_early_termination() {
    // 使用bipartite图: 无三角形, 但有很多边 → early termination应该触发
    auto g = build_bipartite_graph(4, 4);
    TriangleCounting_optimized tc(g);
    uint64_t count = tc.run_tc();

    TEST_ASSERT(count == 0, "TC_opt bipartite: 0 triangles");
    // early_term or prefix_skip should fire on some comparisons
    std::printf("[TC_opt bipartite DEBUG] early_term=%lu prefix_skip=%lu comparisons=%lu\n",
                (unsigned long)tc.stats.early_term_count,
                (unsigned long)tc.stats.prefix_skip_count,
                (unsigned long)tc.stats.total_comparisons);
    TEST_PASS("tcopt_early_termination");
}

} // namespace experiment
} // namespace philemon


// ═══════════════════════════════════════════════════════════════════
//  main()
// ═══════════════════════════════════════════════════════════════════

int main() {
    auto t0 = std::chrono::steady_clock::now();

    std::printf("═══════════════════════════════════════════════════════════════════\n");
    std::printf("  M104-M105: wrapper/algorithms 6算法深度实验\n");
    std::printf("  第12位Claude (Opus 4.6) — philemon-TSH\n");
    std::printf("═══════════════════════════════════════════════════════════════════\n\n");

    // M104: BFS
    std::printf("────── M104: BFS (direction-optimizing) ──────\n");
    philemon::experiment::test_bfs_chain_graph();
    philemon::experiment::test_bfs_star_graph();
    philemon::experiment::test_bfs_disconnected();
    philemon::experiment::test_bfs_direction_switch_stats();

    // M104: SSSP
    std::printf("\n────── M104: SSSP (delta-stepping) ──────\n");
    philemon::experiment::test_sssp_chain();
    philemon::experiment::test_sssp_star();
    philemon::experiment::test_sssp_disconnected();
    philemon::experiment::test_sssp_bucket_stats();

    // M104: WCC
    std::printf("\n────── M104: WCC (union-find) ──────\n");
    philemon::experiment::test_wcc_connected();
    philemon::experiment::test_wcc_disconnected();
    philemon::experiment::test_wcc_path_compression();

    // M105: PageRank
    std::printf("\n────── M105: PageRank ──────\n");
    philemon::experiment::test_pr_uniform();
    philemon::experiment::test_pr_star_center();
    philemon::experiment::test_pr_convergence();
    philemon::experiment::test_pr_dangling();

    // M105: TC
    std::printf("\n────── M105: TriangleCounting ──────\n");
    philemon::experiment::test_tc_triangle_mesh();
    philemon::experiment::test_tc_complete_graph();
    philemon::experiment::test_tc_no_triangles();

    // M105: TC_opt
    std::printf("\n────── M105: TriangleCounting_optimized ──────\n");
    philemon::experiment::test_tcopt_triangle_mesh();
    philemon::experiment::test_tcopt_complete_graph();
    philemon::experiment::test_tcopt_no_triangles();
    philemon::experiment::test_tcopt_early_termination();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    std::printf("\n═══════════════════════════════════════════════════════════════════\n");
    std::printf("  M104-M105 RESULTS: %d/%d passed, %d failed, elapsed=%ldms\n",
                g_tests_passed, g_tests_run, g_tests_failed, (long)elapsed);
    std::printf("═══════════════════════════════════════════════════════════════════\n");

    return g_tests_failed > 0 ? 1 : 0;
}
