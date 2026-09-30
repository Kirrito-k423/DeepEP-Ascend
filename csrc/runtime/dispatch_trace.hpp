#pragma once
#include <akl/trace/capture.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <vector>

namespace deep_ep {
// Each process captures invocation 11 of each kernel; export after both kernels finish.
struct DispatchTrace {
    const char* root = std::getenv("EP_DEBUG_CLOCK_DIR");
    bool enabled = root && *root;
    int mode = enabled ? (std::getenv("EP_DEBUG_CLOCK_EMPTY") ? 2 : 1) : 0;
    uint8_t* data = nullptr;
    struct Pending {
        std::filesystem::path root;
        int rank;
        std::unique_ptr<akl::Capture<32>> capture;
    };
    inline static std::mutex mutex;
    inline static std::map<std::string, uint64_t> calls;
    inline static std::vector<Pending> pending;
    DispatchTrace(const char* name, int blocks, int rank) {
        if (!enabled) return;
        std::lock_guard<std::mutex> lock(mutex);
        if (++calls[name] != 11) return;
        auto capture = std::make_unique<akl::Capture<32>>(blocks, c10_npu::getCurrentNPUStream().stream());
        data = capture->Data();
        pending.push_back({std::filesystem::path(root) / name, rank, std::move(capture)});
    }
    static void Export() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& item : pending) item.capture->Export(item.root, item.rank);
        pending.clear();
    }
};
}
