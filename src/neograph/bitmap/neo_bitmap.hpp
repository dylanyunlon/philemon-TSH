#pragma once
/**
 * neo_bitmap.hpp — Bitmap with popcount profiling and tier-aware scanning
 *
 * 骨架来源: upstream/.../bitmap/include/bitmap.h (230行) + bitmap.cpp (空)
 * 修改 (~20%):
 *   - popcount() 求总设置位数 — 用于快速判定segment利用率
 *   - for_each 增加 early-exit 重载 (回调返回false即停)
 *   - lower_bound 内循环加 __builtin_popcountll 批量跳过零块
 *   - dump_bitmap() 打印各块占用情况
 *   - density() 计算占用率 (迁移调度判定阈值用)
 *
 * Milestone: M071
 */

#include <array>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cstdio>

namespace container {

template<size_t BLOCK_NUM>
struct Bitmap {
    std::array<uint64_t, BLOCK_NUM> data{};

    Bitmap() = default;
    Bitmap(const Bitmap&) = default;
    Bitmap(Bitmap&&) = default;
    Bitmap& operator=(const Bitmap&) = default;

    void set(uint64_t index) {
        const uint64_t block = index / 64;
        const uint64_t offset = index % 64;
        data[block] |= 1ULL << offset;
    }

    void reset(uint64_t index) {
        const uint64_t block = index / 64;
        const uint64_t offset = index % 64;
        data[block] &= ~(1ULL << offset);
    }

    [[nodiscard]] bool get(uint64_t index) const {
        const uint64_t block = index / 64;
        const uint64_t offset = index % 64;
        return data[block] & (1ULL << offset);
    }

    // ─── NEW: total set bits via popcount ───
    [[nodiscard]] uint64_t popcount() const {
        uint64_t total = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++)
            total += __builtin_popcountll(data[i]);
        return total;
    }

    // ─── NEW: occupancy ratio [0.0, 1.0] ───
    [[nodiscard]] double density() const {
        return static_cast<double>(popcount()) / (BLOCK_NUM * 64);
    }

    [[nodiscard]] uint64_t at(uint64_t pos_idx) const {
        uint16_t count = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            // Skip empty blocks fast with popcount
            uint64_t block_pop = __builtin_popcountll(mask);
            if (count + block_pop <= pos_idx) {
                count += block_pop;
                continue;
            }
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                if (count == pos_idx) return index + i * 64;
                count++;
                mask ^= t;
            }
        }
        return 0;
    }

    [[nodiscard]] bool empty() const {
        return std::all_of(data.begin(), data.end(),
                           [](uint64_t i) { return i == 0; });
    }

    [[nodiscard]] uint64_t find_first() {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            if (data[i]) return __builtin_ctzll(data[i]) + i * 64;
        }
        return std::numeric_limits<uint64_t>::max();
    }

    [[nodiscard]] uint64_t find_first(uint64_t begin) {
        const uint64_t begin_block = begin / 64;
        for (size_t i = begin_block; i < BLOCK_NUM; i++) {
            if (data[i]) return __builtin_ctzll(data[i]) + i * 64;
        }
        return std::numeric_limits<uint64_t>::max();
    }

    [[nodiscard]] uint64_t lower_bound(uint64_t element, uint64_t prefix) const {
        uint64_t target = element & 0xFF;
        uint64_t res = 0;
        if ((element & ~0xFFULL) == prefix) {
            for (size_t i = 0; i < BLOCK_NUM; i++) {
                uint64_t mask = data[i];
                while (mask) {
                    uint64_t t = mask & -mask;
                    uint64_t index = __builtin_ctzll(mask);
                    if ((index + i * 64) >= target) break;
                    res++;
                    mask ^= t;
                }
            }
        } else {
            for (size_t i = 0; i < BLOCK_NUM; i++) {
                uint64_t mask = data[i];
                // Fast skip: if all bits in block are below element, count them all
                if (mask && ((63 + i * 64) | prefix) < element) {
                    res += __builtin_popcountll(mask);
                    continue;
                }
                while (mask) {
                    uint64_t t = mask & -mask;
                    uint64_t index = __builtin_ctzll(mask);
                    if (((index + i * 64) | prefix) >= element) break;
                    res++;
                    mask ^= t;
                }
            }
        }
        return res;
    }

    [[nodiscard]] uint64_t consume() {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            if (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                data[i] = mask ^ t;
                return index + i * 64;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    [[nodiscard]] uint64_t consume(uint64_t begin) {
        const uint64_t begin_block = begin / 64;
        for (size_t i = begin_block; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            if (mask) {
                uint64_t t = mask & -mask;
                data[i] = mask ^ t;
                return __builtin_ctzll(mask) + i * 64;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    // ─── for_each (upstream, unchanged) ───
    template<typename F>
    void for_each(F&& f) const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                f(__builtin_ctzll(mask) + i * 64);
                mask ^= t;
            }
        }
    }

    template<typename F>
    void for_each(F&& f, uint64_t begin, uint64_t end) const {
        uint64_t count = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                if (count >= begin && count < end) f(index + i * 64);
                else if (count >= end) return;
                count++;
                mask ^= t;
            }
        }
    }

    // ─── NEW: for_each with early-exit (callback returns false to stop) ───
    template<typename F>
    bool for_each_until(F&& f) const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                if (!f(__builtin_ctzll(mask) + i * 64)) return false;
                mask ^= t;
            }
        }
        return true;
    }

    template<typename F>
    void for_each_zero(F&& f) const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = ~data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                f(__builtin_ctzll(mask) + i * 64);
                mask ^= t;
            }
        }
    }

    // ─── NEW: dump ───
    void dump(const char* label = "") const {
        std::fprintf(stderr, "[BITMAP:%s] blocks=%zu pop=%llu density=%.3f\n",
                     label, BLOCK_NUM, (unsigned long long)popcount(), density());
    }
};

} // namespace container
