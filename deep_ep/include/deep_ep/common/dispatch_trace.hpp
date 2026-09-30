#pragma once
#include <akl/trace/semantic.h>
#ifndef EP_DEBUG_CLOCK
#define EP_DEBUG_CLOCK 0
#endif
#if EP_DEBUG_CLOCK == 2
#undef AKL_DEBUG_CLOCK
#define AKL_DEBUG_CLOCK(recorder, ...) do {} while (false)
#endif

namespace deep_ep {
template<bool Enabled>
__aicore__ inline void finish_dispatch_trace(akl::Recorder<Enabled, 32>& clock, __gm__ uint8_t* output) {
    if constexpr (Enabled) {
        if (!output) return;
        // Reuse UB only after business pipelines drain; no extra cross-core barrier.
        AscendC::PipeBarrier<PIPE_ALL>();
        AKL_DEBUG_CLOCK(clock, "dispatch", "pipes-drained");
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, (8 + 2 * 32) * sizeof(uint64_t));
        clock.Flush(output, buffer.Get<uint64_t>(), 0);
    }
}
}
