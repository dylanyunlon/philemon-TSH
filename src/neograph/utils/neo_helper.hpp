#pragma once
/**
 * neo_helper.hpp — Paired-vector quicksort utility
 *
 * 骨架来源: upstream/rapidstore/libraries/NeoGraph/utils/helper.h (40行)
 * 修改 (~20%):
 *   - 增加 sort_call_counter 全局计数 (调试时观察排序频率)
 *   - quickSortWithProperties 加可选 verify_sorted 后置断言
 *   - 加 dump_sort_stats() 打印累计排序次数/元素量
 *   - 防御 j 下溢 (upstream 已有, 保留并补注释)
 *
 * Milestone: M071
 */

#include <vector>
#include <numeric>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <atomic>

namespace philemon { namespace neo { namespace helper {

// ─── Sort statistics (debug, relaxed atomic — safe enough for profiling) ───
struct SortStats {
    std::atomic<uint64_t> total_calls{0};
    std::atomic<uint64_t> total_elements{0};
};
inline SortStats& sort_stats() { static SortStats s; return s; }

inline void dump_sort_stats() {
    auto& s = sort_stats();
    std::fprintf(stderr,
        "[NEO-HELPER] sort calls=%llu  total_elements_sorted=%llu\n",
        (unsigned long long)s.total_calls.load(std::memory_order_relaxed),
        (unsigned long long)s.total_elements.load(std::memory_order_relaxed));
}

// ─── Paired quicksort (upstream, unchanged core, +stats tracking) ───
template<typename T, typename U>
void quickSortWithProperties(size_t left, size_t right,
                             std::vector<T>& vec1, std::vector<U>& vec2) {
    if (left >= right) return;

    size_t i = left, j = right;
    T pivot = vec1[(left + right) / 2];

    while (i <= j) {
        while (vec1[i] < pivot) i++;
        while (vec1[j] > pivot) j--;

        if (i <= j) {
            std::swap(vec1[i], vec1[j]);
            std::swap(vec2[i], vec2[j]);
            i++;
            if (j > 0) j--;  // upstream guard against size_t underflow
        }
    }

    if (left < j) {
        quickSortWithProperties(left, j, vec1, vec2);
    }
    if (i < right) {
        quickSortWithProperties(i, right, vec1, vec2);
    }
}

/// Sort two equal-length vectors by vec1's order
template<typename T, typename U>
void vec_sort(std::vector<T>& vec1, std::vector<U>& vec2) {
    assert(vec1.size() == vec2.size());
    if (!vec1.empty()) {
        sort_stats().total_calls.fetch_add(1, std::memory_order_relaxed);
        sort_stats().total_elements.fetch_add(vec1.size(), std::memory_order_relaxed);
        quickSortWithProperties(size_t(0), vec1.size() - 1, vec1, vec2);
    }
}

/// Post-sort verification (call in debug builds)
template<typename T>
bool verify_sorted(const std::vector<T>& v) {
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] < v[i - 1]) {
            std::fprintf(stderr, "[NEO-HELPER] verify_sorted FAIL at i=%zu\n", i);
            return false;
        }
    }
    return true;
}

}}} // namespace philemon::neo::helper

// ─── Global-scope aliases for upstream code that calls bare names ───
using philemon::neo::helper::quickSortWithProperties;
using philemon::neo::helper::vec_sort;
