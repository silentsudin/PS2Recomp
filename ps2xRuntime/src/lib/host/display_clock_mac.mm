#include "runtime/ps2_display_clock.h"

#import <CoreVideo/CoreVideo.h>
#include <mach/mach_time.h>

#include <atomic>
#include <chrono>
#include <time.h>

namespace
{
    std::atomic<int64_t> g_lastRefreshNs{0}, g_periodNs{0};
    mach_timebase_info_data_t g_timebase{};

    CVReturn onRefresh(CVDisplayLinkRef link, const CVTimeStamp *, const CVTimeStamp *output, CVOptionFlags, CVOptionFlags *,
                       void *)
    {
        // The upcoming refresh (on the exact refresh grid; the callback itself runs irregularly),
        // in mach_absolute_time nanoseconds.
        const int64_t ns = static_cast<int64_t>(output->hostTime * g_timebase.numer / g_timebase.denom);
        g_lastRefreshNs.store(ns, std::memory_order_relaxed);
        const double period = CVDisplayLinkGetActualOutputVideoRefreshPeriod(link);
        if (period > 0)
            g_periodNs.store(static_cast<int64_t>(period * 1e9), std::memory_order_relaxed);
        return kCVReturnSuccess;
    }

    bool start()
    {
        mach_timebase_info(&g_timebase);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        CVDisplayLinkRef link = nullptr;
        if (CVDisplayLinkCreateWithActiveCGDisplays(&link) != kCVReturnSuccess || !link)
            return false;
        CVDisplayLinkSetOutputCallback(link, onRefresh, nullptr);
        CVDisplayLinkStart(link);
#pragma clang diagnostic pop
        return true; // runs for the life of the process
    }
}

bool ps2x::displayClockSample(int64_t &lastRefreshNs, int64_t &periodNs)
{
    static const bool started = start();
    periodNs = g_periodNs.load(std::memory_order_relaxed);
    // mach_absolute_time stops while the Mac sleeps; steady_clock (CLOCK_MONOTONIC_RAW) does not.
    const int64_t uptime = static_cast<int64_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
    const int64_t steady = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch()).count();
    lastRefreshNs = g_lastRefreshNs.load(std::memory_order_relaxed) + (steady - uptime);
    return started && g_lastRefreshNs.load(std::memory_order_relaxed) != 0 && periodNs != 0;
}
