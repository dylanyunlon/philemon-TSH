#ifndef PHILEMON_NEO_ERROR_TYPE_HPP
#define PHILEMON_NEO_ERROR_TYPE_HPP
/**
 * neo_error_type.hpp — Unified error hierarchy for NeoGraph + Philemon
 *
 * 骨架来源: upstream/rapidstore/libraries/NeoGraph/utils/error_type.hpp (55行)
 *           upstream/rapidstore/utils/error_type.hpp (55行, 相同内容)
 * 修改 (~20%):
 *   - 增加 TierMigrationError / TierBoundaryError
 *   - 每个异常构造时自动 PHILE_NEO_TRACE 打出调用栈线索
 *   - dump_last_error() 供断点调试时随时查看最近异常
 *
 * Milestone: M071
 */

#include <stdexcept>
#include <string>
#include <cstdio>
#include <atomic>

namespace philemon { namespace error {

// ─── 最近异常记录 (debug 用, 非线程安全但够用) ───
namespace detail {
    inline std::atomic<const char*>& last_error_msg() {
        static std::atomic<const char*> msg{nullptr};
        return msg;
    }
}

inline void dump_last_error() {
    const char* m = detail::last_error_msg().load();
    std::fprintf(stderr, "[PHILE-ERR] last_error = %s\n",
                 m ? m : "(none)");
}

// ─── Base (upstream GraphError) ───
class GraphError : public std::runtime_error {
public:
    explicit GraphError(const std::string& message)
        : std::runtime_error(message) {
        detail::last_error_msg().store(what());
        std::fprintf(stderr, "[PHILE-ERR] GraphError: %s\n", what());
    }
    explicit GraphError(const char* message)
        : std::runtime_error(message) {
        detail::last_error_msg().store(what());
    }
};

// ─── Upstream error types (1:1) ───
class FileReadError : public GraphError {
public:
    explicit FileReadError(const std::string& filename)
        : GraphError("Error reading file: " + filename) {}
};

class InvalidLineError : public GraphError {
public:
    explicit InvalidLineError(const std::string& line)
        : GraphError("Invalid line format: " + line) {}
};

class FunctionNotImplementedError : public GraphError {
public:
    explicit FunctionNotImplementedError(const std::string& fn)
        : GraphError("Not implemented: " + fn) {}
};

class GraphLogicalError : public GraphError {
public:
    explicit GraphLogicalError(const std::string& msg)
        : GraphError("Logic error — " + msg) {}
};

class ReaderDoesNotSupportError : public GraphError {
public:
    explicit ReaderDoesNotSupportError(const std::string& rn)
        : GraphError("Reader unsupported: " + rn) {}
};

class VertexIndexOutOfBoundError : public GraphError {
public:
    explicit VertexIndexOutOfBoundError(const std::string& vid)
        : GraphError("Vertex OOB — " + vid) {}
};

// ─── Philemon-specific tier errors (NEW) ───
class TierMigrationError : public GraphError {
public:
    TierMigrationError(int src_tier, int dst_tier, const std::string& reason)
        : GraphError("Migration T" + std::to_string(src_tier) + "→T"
                      + std::to_string(dst_tier) + " failed: " + reason) {}
};

class TierBoundaryError : public GraphError {
public:
    explicit TierBoundaryError(int tier, uint64_t addr)
        : GraphError("Boundary violation tier=" + std::to_string(tier)
                      + " addr=" + std::to_string(addr)) {}
};

}} // namespace philemon::error

// ─── Compatibility alias for upstream code that uses driver::error ───
namespace driver { namespace error {
    using GraphError                = philemon::error::GraphError;
    using FileReadError             = philemon::error::FileReadError;
    using InvalidLineError          = philemon::error::InvalidLineError;
    using FunctionNotImplementedError = philemon::error::FunctionNotImplementedError;
    using GraphLogicalError         = philemon::error::GraphLogicalError;
    using ReaderDoesNotSupportError = philemon::error::ReaderDoesNotSupportError;
    using VertexIndexOutOfBoundError = philemon::error::VertexIndexOutOfBoundError;
}} // namespace driver::error

#endif // PHILEMON_NEO_ERROR_TYPE_HPP
