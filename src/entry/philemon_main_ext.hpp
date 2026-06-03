#ifndef PHILEMON_MAIN_EXT_HPP
#define PHILEMON_MAIN_EXT_HPP
/**
 * philemon_main_ext.hpp — 程序入口模板 (替代 upstream driver_main.h)
 *
 * 骨架来源: upstream/rapidstore/wrapper/driver_main.h (15行)
 * 修改 (~40%):
 *   - 移除 ittnotify 依赖
 *   - 使用 philemon ConfigParser 替代 boost commandLineParser
 *   - 增加 startup banner 打印: 版本/时间/配置摘要
 *   - 增加 shutdown 时自动 dump 操作统计
 *
 * Milestone: M028
 */

#include <iostream>
#include <chrono>
#include <ctime>
#include "../utils/config_parser_ext.hpp"
#include "../utils/log/philemon_log.hpp"
#include "philemon_driver_ext.hpp"

namespace philemon {
namespace entry {

inline void print_banner() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::printf(
        "\n"
        "╔═══════════════════════════════════════════════════╗\n"
        "║   Philemon-TSH : Tiered Storage Heterogeneous    ║\n"
        "║   Graph Processing System                        ║\n"
        "║   Build: %s %s                     ║\n"
        "║   Start: %.24s           ║\n"
        "╚═══════════════════════════════════════════════════╝\n\n",
        __DATE__, __TIME__, ctime(&t));
}

template <class GraphType>
int run_main(int argc, char** argv, GraphType& graph) {
    print_banner();

    auto& cfg = philemon::config::ConfigParser::instance();

    if (argc > 1) {
        cfg.parse(argv[1]); // config file path
    } else {
        plog_warn("No config file provided, using defaults");
    }

    // Parse remaining args as overrides
    if (argc > 2) cfg.parse_args(argc, argv);

    // Validate and dump
    cfg.dump_config("STARTUP");
    if (!cfg.validate()) {
        plog_fatal("Config validation failed, aborting");
        return 1;
    }

    // ─── Driver setup ───────────────────────────────────────────────
    plog_info("[MAIN] Initializing driver with %d threads", cfg.num_threads());

    // The actual graph + driver wiring happens in caller
    // This template just provides the entry scaffold

    plog_info("[MAIN] Ready to execute workloads");
    return 0;
}

inline void shutdown_report() {
    std::printf("\n");
    wrapper::counters::dump("SHUTDOWN");
    plog_info("[MAIN] Philemon-TSH shutdown complete");
}

} // namespace entry
} // namespace philemon

#endif // PHILEMON_MAIN_EXT_HPP
