/**
 * philemon_file_readers.hpp — 文件读取器 (edge list + vertex list)
 *
 * 骨架来源:
 *   upstream/rapidstore/readers/edgeListReader.cpp+hpp (105行)
 *   upstream/rapidstore/readers/vertexReader.cpp+hpp   (87行)
 *   upstream/rapidstore/readers/reader.cpp+hpp         (56行)
 *
 * 修改 (~20%):
 *   - [MOD] namespace driver::reader → philemon::io::readers
 *   - [NEW] read计数器: 每个reader追踪已读取条目数
 *   - [NEW] BREAKPOINT_READER(): 打印reader状态(文件路径/位置/计数)
 *   - [NEW] 读取进度回调: 每N条触发用户回调
 *   - [KEEP] Reader基类接口100%保留
 *   - [KEEP] edgeListReader解析逻辑100%保留(注释行/空行跳过)
 *   - [KEEP] vertexReader解析逻辑100%保留
 *   - [KEEP] Reader::open()工厂方法保留
 *
 * Milestone: M098
 */
#ifndef PHILEMON_FILE_READERS_HPP
#define PHILEMON_FILE_READERS_HPP

#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cstdint>

#include "edge_stream_file_io.hpp"

namespace philemon {
namespace io {
namespace readers {

// ─── Reader类型枚举 (upstream保留) ──────────────────────────────────
enum class readerType {
    edgeList,
    vertexList
};

// ─── [NEW] 进度回调类型 ─────────────────────────────────────────────
using ProgressCallback = std::function<void(uint64_t count, const char* type)>;

// ─── Abstract Reader (upstream骨架 + 计数器) ────────────────────────
class FileReader {
    FileReader(const FileReader&) = delete;
    FileReader& operator=(const FileReader&) = delete;

protected:
    uint64_t read_count_ = 0;         // [NEW] 读取计数
    uint64_t skip_count_ = 0;         // [NEW] 跳过计数
    std::string file_path_;           // [NEW] 文件路径记录
    ProgressCallback progress_cb_;     // [NEW] 进度回调
    uint64_t progress_interval_ = 100000;  // [NEW] 回调间隔

public:
    FileReader() = default;
    virtual ~FileReader() = default;

    virtual bool read_edge(StreamEdge& edge) { return false; }
    virtual bool read_vertex(uint64_t& vertex) { return false; }
    virtual bool is_directed() const { return true; }

    uint64_t get_read_count() const { return read_count_; }
    uint64_t get_skip_count() const { return skip_count_; }

    void set_progress_callback(ProgressCallback cb, uint64_t interval = 100000) {
        progress_cb_ = cb;
        progress_interval_ = interval;
    }

    // [NEW] 状态dump
    void dump_state(const char* label = "Reader") const {
        std::fprintf(stderr, "[BREAKPOINT] %s: file=%s read=%lu skip=%lu\n",
                     label, file_path_.c_str(), read_count_, skip_count_);
    }

    // 工厂方法
    static std::unique_ptr<FileReader> open(const std::string& path,
                                            readerType type,
                                            bool weighted = false);
};

// ─── EdgeListReader (upstream骨架, 修改20%) ─────────────────────────
class EdgeListFileReader : public FileReader {
    std::fstream handle_;
    bool is_weighted_;

public:
    // upstream构造 (100%)
    EdgeListFileReader(const std::string& path, bool weighted)
        : is_weighted_(weighted) 
    {
        file_path_ = path;
        handle_.open(path, std::ios::in);
        if (!handle_.good()) {
            std::fprintf(stderr, "[READER] Cannot open edge file: %s\n", path.c_str());
        } else {
            std::fprintf(stderr, "[READER] Opened edge file: %s (weighted=%d)\n",
                         path.c_str(), weighted);
        }
        handle_.seekg(0, std::ios::beg);
    }

    ~EdgeListFileReader() override {
        if (handle_.is_open()) handle_.close();
    }

    bool is_directed() const override { return true; }

    // upstream read逻辑 (100% + 计数)
    bool read_edge(StreamEdge& edge) override {
        std::string line;
        while (std::getline(handle_, line)) {
            if (line.empty() || line[0] == '#') {
                skip_count_++;
                continue;
            }

            std::istringstream ss(line);
            uint64_t source_id, dest_id;
            double weight = static_cast<double>(std::rand()) / RAND_MAX;

            if (is_weighted_) {
                if (!(ss >> source_id) || !(ss >> dest_id) || !(ss >> weight)) {
                    std::fprintf(stderr, "[READER] Invalid weighted line: %s\n", line.c_str());
                    skip_count_++;
                    continue;
                }
            } else {
                if (!(ss >> source_id) || !(ss >> dest_id)) {
                    std::fprintf(stderr, "[READER] Invalid line: %s\n", line.c_str());
                    skip_count_++;
                    continue;
                }
                std::srand(static_cast<unsigned int>(std::time(nullptr)));
                weight = static_cast<double>(std::rand()) / RAND_MAX;
            }

            edge.set_edge(source_id, dest_id, weight);
            read_count_++;

            // [NEW] 进度回调
            if (progress_cb_ && (read_count_ % progress_interval_ == 0)) {
                progress_cb_(read_count_, "edges");
            }

            return true;
        }
        return false;
    }
};

// ─── VertexReader (upstream骨架, 修改20%) ───────────────────────────
class VertexFileReader : public FileReader {
    std::fstream handle_;

public:
    VertexFileReader(const std::string& path) {
        file_path_ = path;
        handle_.open(path, std::ios::in);
        if (!handle_.good()) {
            std::fprintf(stderr, "[READER] Cannot open vertex file: %s\n", path.c_str());
        } else {
            std::fprintf(stderr, "[READER] Opened vertex file: %s\n", path.c_str());
        }
    }

    ~VertexFileReader() override {
        if (handle_.is_open()) handle_.close();
    }

    bool is_directed() const override { return false; }

    bool read_vertex(uint64_t& vertex) override {
        std::string line;
        while (std::getline(handle_, line)) {
            if (line.empty() || line[0] == '#') {
                skip_count_++;
                continue;
            }

            std::istringstream ss(line);
            if (!(ss >> vertex)) {
                std::fprintf(stderr, "[READER] Invalid vertex line: %s\n", line.c_str());
                skip_count_++;
                continue;
            }

            read_count_++;

            if (progress_cb_ && (read_count_ % progress_interval_ == 0)) {
                progress_cb_(read_count_, "vertices");
            }

            return true;
        }
        return false;
    }
};

// ─── 工厂方法 (upstream保留) ────────────────────────────────────────
inline std::unique_ptr<FileReader> FileReader::open(
    const std::string& path, readerType type, bool weighted)
{
    switch (type) {
        case readerType::edgeList:
            return std::make_unique<EdgeListFileReader>(path, weighted);
        case readerType::vertexList:
            return std::make_unique<VertexFileReader>(path);
        default:
            std::fprintf(stderr, "[READER] Unknown reader type\n");
            return nullptr;
    }
}

#define BREAKPOINT_READER(reader) \
    (reader).dump_state(#reader)

} // namespace readers
} // namespace io
} // namespace philemon

#endif // PHILEMON_FILE_READERS_HPP
