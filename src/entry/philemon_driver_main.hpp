/**
 * philemon_driver_main.hpp — 驱动入口
 *
 * 骨架来源: upstream/rapidstore/wrapper/driver_main.h (15行)
 * 修改 (~20%):
 *   - [MOD] 移除ittnotify依赖 → 自带Timer计时
 *   - [MOD] commandLineParser → ConfigEngine
 *   - [NEW] 启动时打印系统信息(CPU/内存/编译器)
 *   - [NEW] BREAKPOINT_ENTRY(): 入口调试点
 *   - [KEEP] main()结构保留: parse config → execute
 *
 * Milestone: M098
 */
#ifndef PHILEMON_DRIVER_MAIN_HPP
#define PHILEMON_DRIVER_MAIN_HPP

#include <iostream>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/resource.h>

#include "../utils/philemon_cli_engine.hpp"
#include "../utils/philemon_timer.hpp"

namespace philemon {
namespace entry {

// ─── [NEW] 系统信息打印 ─────────────────────────────────────────────
inline void print_system_info() {
    std::fprintf(stderr, "╔═══════════════════════════════════════════════╗\n");
    std::fprintf(stderr, "║  Philemon-TSH Driver Entry                   ║\n");
    std::fprintf(stderr, "╠═══════════════════════════════════════════════╣\n");

    // CPU info
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* colon = strchr(line, ':');
                if (colon) {
                    std::fprintf(stderr, "║ CPU: %s", colon + 2);
                }
                break;
            }
        }
        fclose(f);
    }

    // Memory info
    f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemTotal", 8) == 0) {
                uint64_t kb = 0;
                sscanf(line, "MemTotal: %lu kB", &kb);
                std::fprintf(stderr, "║ RAM: %.1f GB\n", kb / (1024.0 * 1024.0));
                break;
            }
        }
        fclose(f);
    }

    // 编译器
    std::fprintf(stderr, "║ Compiler: %s\n",
#ifdef __clang__
                 "Clang " __clang_version__
#elif defined(__GNUC__)
                 "GCC " __VERSION__
#else
                 "Unknown"
#endif
    );
    std::fprintf(stderr, "║ C++ Standard: %ld\n", __cplusplus);
    std::fprintf(stderr, "║ PID: %d\n", getpid());
    std::fprintf(stderr, "╚═══════════════════════════════════════════════╝\n\n");
}

// ─── 入口函数 (upstream driver_main.h 重写) ────────────────────────
inline int driver_main(int argc, char** argv) {
    print_system_info();

    philemon::utils::Timer startup_timer;

    // 解析配置
    auto& engine = philemon::config::ConfigEngine::get_instance();
    
    if (argc >= 2) {
        std::fprintf(stderr, "[ENTRY] Parsing config: %s\n", argv[1]);
        engine.parse(argv[1]);
    } else {
        std::fprintf(stderr, "[ENTRY] No config file, using defaults\n");
        engine.use_defaults();
    }

    // [BREAKPOINT] 入口调试点
    engine.dump_config();

    double parse_time = startup_timer.elapsed_and_reset();
    std::fprintf(stderr, "[ENTRY] Config parsed in %.4f sec\n", parse_time);

    // 校验
    int errors = engine.validate();
    if (errors > 0) {
        std::fprintf(stderr, "[ENTRY] Config validation failed with %d errors\n", errors);
        return 1;
    }

    std::fprintf(stderr, "[ENTRY] Ready to execute (workload=%s, threads=%d)\n",
                 philemon::config::op_type_name(engine.get_workload_type()),
                 engine.get_num_threads());

    // 这里可以调用 wrapper::execute(engine.get_driver_config())
    // 但在实验模式下，我们在experiment文件中直接调用

    return 0;
}

#define BREAKPOINT_ENTRY() do { \
    std::fprintf(stderr, "[BREAKPOINT_ENTRY] pid=%d\n", getpid()); \
    philemon::entry::print_system_info(); \
    BREAKPOINT_CONFIG(); \
} while(0)

} // namespace entry
} // namespace philemon

#endif // PHILEMON_DRIVER_MAIN_HPP
