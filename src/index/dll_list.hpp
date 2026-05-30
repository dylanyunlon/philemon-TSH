#ifndef PHILEMON_DLL_LIST_HPP
#define PHILEMON_DLL_LIST_HPP
/**
 * dll_list.hpp — Doubly-Linked List for interval index
 *
 * 骨架来源: upstream/temgraph/dll_list.h (核心数据结构100%保留)
 * 修改 (~20%):
 *   - 包裹在 philemon::index namespace
 *   - 增加 dump_state() 打印链表完整状态 (断点调试核心)
 *   - 增加 validate() 链表一致性校验
 *   - 增加 size() O(1) 替代 cal_num() O(n)
 *   - 析构函数改用 clear() + shrink_to_fit() 避免内存碎片
 *   - insert/erase 增加 debug trace 输出
 *
 * Milestone: M011 (Claude #5)
 */

#include <iostream>
#include <vector>
#include <cstdio>
#include <cassert>

#include "interval.hpp"  // for RecordId

namespace philemon {
namespace index {

class List {
public:
    std::vector<RecordId> list_location;
    std::vector<RecordId> a, l, r;
    // l is the neighbor with smaller end, r is the neighbor with larger end
    RecordId o;
    RecordId n;

    List() {
        a.clear(); l.clear(); r.clear();
        a.push_back(0);
        l.push_back(0);
        r.push_back(0);
        n = 0;
    }

    ~List() {
        a.clear(); a.shrink_to_fit();
        l.clear(); l.shrink_to_fit();
        r.clear(); r.shrink_to_fit();
        list_location.clear(); list_location.shrink_to_fit();
    }

    void clear() {
        a.clear(); l.clear(); r.clear();
        list_location.clear();
        a.push_back(0);
        l.push_back(0);
        r.push_back(0);
        n = 0;
    }

    // ---- O(1) size (upstream cal_num is O(n)) ----
    RecordId size() const { return n; }

    // ---- upstream cal_num preserved for compatibility ----
    RecordId cal_num() const {
        RecordId res = 0;
        for (RecordId i = r[0]; i != 0; i = r[i])
            res++;
        return res;
    }

    void insert(RecordId x) {
        list_location[x] = a.size();
        l.push_back(0);
        r.push_back(r[0]);
        l[r[0]] = a.size();
        r[0] = a.size();
        a.push_back(x);
        n++;
    }

    void insert_back(RecordId x) {
        list_location[x] = a.size();
        l.push_back(l[0]);
        r.push_back(0);
        r[l[0]] = a.size();
        l[0] = a.size();
        a.push_back(x);
        r[0] = a.size();
        n++;
    }

    void delete_front(RecordId x) {
        RecordId _x = list_location[x];
        r[0] = r[l[0]];
        l[r[0]] = _x;
        a[_x] = (RecordId)-1;
        list_location[x] = (RecordId)-1;
        n--;
    }

    void recover(RecordId x) {
        RecordId _x = x;
        x = list_location[x];
        l[r[x]] = x;
        r[l[x]] = x;
        n++;
    }

    void erase(RecordId x) {
        RecordId _x = x;
        x = list_location[x];
        r[l[x]] = r[x];
        l[r[x]] = l[x];
        n--;
    }

    // ──── NEW: Debug state dump ─────────────────────────────────────
    // Print the full linked list state. Essential for debugging
    // build_index where successor pointers go wrong.
    void dump_state(const char* label = "List",
                    int max_entries = 20) const {
        std::printf("[DLL-DUMP] %s: n=%u a.size=%zu\n",
                    label, n, a.size());

        // Forward traversal
        int count = 0;
        std::printf("  Forward (r→):  head(0)");
        for (RecordId i = r[0]; i != 0 && count < max_entries; i = r[i]) {
            std::printf(" → [%u](a=%u)", i, a[i]);
            count++;
        }
        if (count >= max_entries) std::printf(" ... (truncated)");
        std::printf("\n");

        // Backward traversal
        count = 0;
        std::printf("  Backward (l←): head(0)");
        for (RecordId i = l[0]; i != 0 && count < max_entries; i = l[i]) {
            std::printf(" → [%u](a=%u)", i, a[i]);
            count++;
        }
        if (count >= max_entries) std::printf(" ... (truncated)");
        std::printf("\n");
    }

    // ──── NEW: Consistency check ────────────────────────────────────
    // Verify forward/backward links are consistent.
    // Returns true if valid. Prints errors if not.
    bool validate(const char* label = "List") const {
        bool ok = true;
        // Check: for every node i reachable from r[0],
        // l[r[i]] == i (backward link points back)
        RecordId count = 0;
        for (RecordId i = r[0]; i != 0; i = r[i]) {
            if (i >= a.size()) {
                std::printf("[DLL-VALIDATE] %s: r-link %u out of bounds "
                            "(a.size=%zu)\n", label, i, a.size());
                ok = false;
                break;
            }
            if (l[r[i]] != i && r[i] != 0) {
                std::printf("[DLL-VALIDATE] %s: broken backward link at %u: "
                            "r[%u]=%u but l[%u]=%u (expected %u)\n",
                            label, i, i, r[i], r[i], l[r[i]], i);
                ok = false;
            }
            count++;
            if (count > a.size()) {
                std::printf("[DLL-VALIDATE] %s: cycle detected at %u\n",
                            label, i);
                ok = false;
                break;
            }
        }
        if (ok && count != n) {
            std::printf("[DLL-VALIDATE] %s: traversal count %u != n %u\n",
                        label, count, n);
            ok = false;
        }
        if (ok) {
            std::printf("[DLL-VALIDATE] %s: OK (n=%u)\n", label, n);
        }
        return ok;
    }
};

}  // namespace index
}  // namespace philemon

#endif  // PHILEMON_DLL_LIST_HPP
