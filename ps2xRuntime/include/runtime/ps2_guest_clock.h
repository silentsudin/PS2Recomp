#pragma once

// The guest's wall clock (RTC, memory card and file times). Normally the host's clock;
// RT_FAKE_CLOCK=<unix seconds> makes it start at that time (UTC, no timezone) and advance with guest
// vblanks instead, so test runs see the same dates every time.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <ctime>

namespace ps2_guest_clock
{
    // Guest vblanks since boot; the EE scheduler advances it.
    inline std::atomic<uint64_t> g_vblanks{0};

    inline bool fake()
    {
        static const bool enabled = std::getenv("RT_FAKE_CLOCK") != nullptr;
        return enabled;
    }

    inline std::time_t now()
    {
        if (!fake())
            return std::time(nullptr);
        static const std::time_t epoch = static_cast<std::time_t>(std::strtoll(std::getenv("RT_FAKE_CLOCK"), nullptr, 10));
        // NTSC: 59.94 fields per second.
        return epoch + static_cast<std::time_t>(g_vblanks.load(std::memory_order_relaxed) * 1001u / 60000u);
    }

    // Broken-down local time (UTC when faked).
    inline bool toLocal(std::time_t t, std::tm &out)
    {
#ifdef _WIN32
        return (fake() ? gmtime_s(&out, &t) : localtime_s(&out, &t)) == 0;
#else
        return (fake() ? gmtime_r(&t, &out) : localtime_r(&t, &out)) != nullptr;
#endif
    }
}
