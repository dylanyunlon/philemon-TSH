#ifndef PHILEMON_READERS_HPP
#define PHILEMON_READERS_HPP
/**
 * philemon_readers.hpp — 图数据文件读取器
 *
 * 骨架来源:
 *   upstream/rapidstore/readers/reader.hpp+cpp       (56行)
 *   upstream/rapidstore/readers/edgeListReader.hpp+cpp (105行)
 *   upstream/rapidstore/readers/vertexReader.hpp+cpp   (87行)
 *
 * 修改 (~20%):
 *   - [MOD] driver::reader → philemon::readers namespace
 *   - [MOD] hpp+cpp合并为单header-only文件
 *   - [NEW] 每个read()调用计数器, 可打印读取统计
 *   - [NEW] TemporalEdgeListReader: 支持 src dst ts_start ts_end 格式
 *   - [NEW] PHILE_READER_STATS() 宏: 打印已读边/顶点数
 *   - [KEEP] Reader抽象基类接口100%保留
 *   - [KEEP] edgeListReader的解析逻辑100%保留
 *   - [KEEP] vertexReader的解析逻辑100%保留
 *   - [KEEP] Reader::open()工厂方法100%保留
 *
 * Milestone: M027+
 */

#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <atomic>
#include <cstdio>

#include "../wrapper/graph_edge.hpp"

namespace philemon {
namespace readers {

// ─── Reader type enum (upstream 100%) ───────────────────────────────
enum class readerType {
    edgeList,
    vertexList,
    temporalEdgeList  // NEW
};

// ─── Abstract Reader base (upstream 100%) ───────────────────────────
class Reader {
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

public:
    Reader() = default;
    virtual ~Reader() = default;

    static std::unique_ptr<Reader> open(const std::string& path,
                                        readerType type,
                                        bool weighted = false);

    virtual bool read(driver::graph::weightedEdge& edge) { return false; }
    virtual bool read(uint64_t& vertex) { return false; }
    virtual bool is_directed() const = 0;

    // NEW: stats
    uint64_t read_count() const { return read_count_.load(); }

protected:
    mutable std::atomic<uint64_t> read_count_{0};
};

// ─── edgeListReader (upstream 100% parse logic) ─────────────────────
class edgeListReader : public Reader {
public:
    edgeListReader(const std::string& path, bool weighted)
        : m_is_weighted(weighted) {
        m_handle.open(path, std::ios::in);
        if (!m_handle.good()) {
            std::fprintf(stderr, "[READER] FAIL: cannot open %s\n", path.c_str());
        } else {
            std::fprintf(stderr, "[READER] opened edge file: %s (weighted=%d)\n",
                         path.c_str(), (int)weighted);
        }
        m_handle.seekg(0, std::ios::beg);
    }

    ~edgeListReader() override { m_handle.close(); }

    bool is_directed() const override { return true; }

    // upstream parse logic 100% preserved
    bool read(driver::graph::weightedEdge& edge) override {
        std::string line;
        while (std::getline(m_handle, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream ss(line);
            uint64_t src, dst;
            double w = static_cast<double>(std::rand()) / RAND_MAX;

            if (m_is_weighted) {
                if (!(ss >> src) || !(ss >> dst) || !(ss >> w)) {
                    std::fprintf(stderr, "[READER] skip invalid weighted line: %s\n",
                                 line.c_str());
                    continue;
                }
            } else {
                if (!(ss >> src) || !(ss >> dst)) {
                    std::fprintf(stderr, "[READER] skip invalid line: %s\n",
                                 line.c_str());
                    continue;
                }
            }

            edge.set_edge(src, dst, w);
            uint64_t cnt = read_count_.fetch_add(1) + 1;

            // DEBUG: periodic progress
            if (cnt % 500000 == 0) {
                std::fprintf(stderr, "[READER] ... read %lu edges so far\n",
                             (unsigned long)cnt);
            }

            return true;
        }
        std::fprintf(stderr, "[READER] EOF after %lu edges\n",
                     (unsigned long)read_count_.load());
        return false;
    }

private:
    std::fstream m_handle;
    const bool m_is_weighted;
};

// ─── vertexReader (upstream 100% parse logic) ───────────────────────
class vertexReader : public Reader {
public:
    explicit vertexReader(const std::string& path) {
        m_handle.open(path, std::ios::in);
        if (!m_handle.good()) {
            std::fprintf(stderr, "[READER] FAIL: cannot open vertex file %s\n",
                         path.c_str());
        } else {
            std::fprintf(stderr, "[READER] opened vertex file: %s\n", path.c_str());
        }
    }

    ~vertexReader() override { m_handle.close(); }

    bool is_directed() const override { return false; }

    // upstream parse logic 100% preserved
    bool read(uint64_t& vertex) override {
        std::string line;
        while (std::getline(m_handle, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream ss(line);
            if (!(ss >> vertex)) {
                std::fprintf(stderr, "[READER] skip invalid vertex line: %s\n",
                             line.c_str());
                continue;
            }

            uint64_t cnt = read_count_.fetch_add(1) + 1;
            if (cnt % 500000 == 0) {
                std::fprintf(stderr, "[READER] ... read %lu vertices so far\n",
                             (unsigned long)cnt);
            }

            return true;
        }
        std::fprintf(stderr, "[READER] vertex EOF after %lu vertices\n",
                     (unsigned long)read_count_.load());
        return false;
    }

private:
    std::fstream m_handle;
};

// ─── TemporalEdgeListReader (NEW: src dst ts_start ts_end) ──────────
class TemporalEdgeListReader : public Reader {
public:
    TemporalEdgeListReader(const std::string& path, bool weighted)
        : m_is_weighted(weighted) {
        m_handle.open(path, std::ios::in);
        if (!m_handle.good()) {
            std::fprintf(stderr, "[READER] FAIL: cannot open temporal file %s\n",
                         path.c_str());
        } else {
            std::fprintf(stderr, "[READER] opened temporal edge file: %s\n",
                         path.c_str());
        }
    }

    ~TemporalEdgeListReader() override { m_handle.close(); }

    bool is_directed() const override { return true; }

    bool read(driver::graph::weightedEdge& edge) override {
        std::string line;
        while (std::getline(m_handle, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream ss(line);
            uint64_t src, dst;
            int32_t ts_s = 0, ts_e = 0;
            double w = 1.0;

            if (!(ss >> src) || !(ss >> dst)) continue;

            if (m_is_weighted) { ss >> w; }
            ss >> ts_s >> ts_e;

            edge.source = src;
            edge.destination = dst;
            edge.weight = w;
            edge.ts_start = ts_s;
            edge.ts_end = ts_e;

            uint64_t cnt = read_count_.fetch_add(1) + 1;
            if (cnt <= 3) {
                std::fprintf(stderr,
                    "[READER] temporal edge #%lu: %lu→%lu ts=[%d,%d]\n",
                    (unsigned long)cnt, (unsigned long)src,
                    (unsigned long)dst, ts_s, ts_e);
            }

            return true;
        }
        return false;
    }

private:
    std::fstream m_handle;
    bool m_is_weighted;
};

// ─── Reader factory (upstream 100% + temporal) ──────────────────────
inline std::unique_ptr<Reader> Reader::open(const std::string& path,
                                            readerType type,
                                            bool weighted) {
    std::fprintf(stderr, "[READER-FACTORY] opening %s type=%d\n",
                 path.c_str(), (int)type);

    switch (type) {
        case readerType::edgeList:
            return std::make_unique<edgeListReader>(path, weighted);
        case readerType::vertexList:
            return std::make_unique<vertexReader>(path);
        case readerType::temporalEdgeList:
            return std::make_unique<TemporalEdgeListReader>(path, weighted);
    }
    return nullptr;  // unreachable
}

// ─── Stats macro ────────────────────────────────────────────────────
#define PHILE_READER_STATS(reader_ptr) do { \
    std::fprintf(stderr, "[READER-STATS] total reads: %lu\n", \
                 (unsigned long)(reader_ptr)->read_count()); \
} while(0)

}  // namespace readers
}  // namespace philemon

// Backward-compatible alias
namespace driver { namespace reader {
    using Reader       = philemon::readers::Reader;
    using readerType   = philemon::readers::readerType;
}}

#endif  // PHILEMON_READERS_HPP
