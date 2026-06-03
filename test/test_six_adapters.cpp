/**
 * test_six_adapters.cpp — 编译验证 + self_test for all 6 independent adapters
 *
 * CSR / Sortledton / Teseo / Aspen / LiveGraph / Neo
 */
#include <cstdio>
#include <cstdlib>

#include "../src/wrapper/apps/csr_tiered_adapter.hpp"
#include "../src/wrapper/apps/sortledton_tiered_adapter.hpp"
#include "../src/wrapper/apps/teseo_tiered_adapter.hpp"
#include "../src/wrapper/apps/aspen_tiered_adapter.hpp"
#include "../src/wrapper/apps/livegraph_tiered_adapter.hpp"
#include "../src/wrapper/apps/neo_tiered_adapter.hpp"

int main() {
    std::fprintf(stderr, "\n╔══════════════════════════════════════════════╗\n");
    std::fprintf(stderr,   "║  Philemon-TSH: 6 Independent Adapter Tests  ║\n");
    std::fprintf(stderr,   "╚══════════════════════════════════════════════╝\n\n");

    int pass = 0;

    // 1. CSR
    try {
        philemon::adapters::csr::csr_adapter_self_test();
        pass++;
        std::fprintf(stderr, "[1/6] CSR ............. PASS\n");
    } catch (...) {
        std::fprintf(stderr, "[1/6] CSR ............. FAIL\n");
    }

    // 2. Sortledton
    try {
        philemon::adapters::sortledton::sortledton_adapter_self_test();
        pass++;
        std::fprintf(stderr, "[2/6] Sortledton ...... PASS\n");
    } catch (...) {
        std::fprintf(stderr, "[2/6] Sortledton ...... FAIL\n");
    }

    // 3. Teseo
    try {
        philemon::adapters::teseo::teseo_adapter_self_test();
        pass++;
        std::fprintf(stderr, "[3/6] Teseo ........... PASS\n");
    } catch (...) {
        std::fprintf(stderr, "[3/6] Teseo ........... FAIL\n");
    }

    // 4. Aspen
    try {
        philemon::adapters::aspen::aspen_adapter_self_test();
        pass++;
        std::fprintf(stderr, "[4/6] Aspen ........... PASS\n");
    } catch (...) {
        std::fprintf(stderr, "[4/6] Aspen ........... FAIL\n");
    }

    // 5. LiveGraph
    try {
        philemon::adapters::livegraph::livegraph_adapter_self_test();
        pass++;
        std::fprintf(stderr, "[5/6] LiveGraph ....... PASS\n");
    } catch (...) {
        std::fprintf(stderr, "[5/6] LiveGraph ....... FAIL\n");
    }

    // 6. Neo
    try {
        philemon::adapters::neo::neo_adapter_self_test();
        pass++;
        std::fprintf(stderr, "[6/6] Neo ............. PASS\n");
    } catch (...) {
        std::fprintf(stderr, "[6/6] Neo ............. FAIL\n");
    }

    std::fprintf(stderr, "\n══════════════════════════════════════════════\n");
    std::fprintf(stderr, "  Result: %d/6 PASSED\n", pass);
    std::fprintf(stderr, "══════════════════════════════════════════════\n\n");

    return (pass == 6) ? 0 : 1;
}
