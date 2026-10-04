#pragma once

// The display's refresh clock (macOS: CVDisplayLink on the main display): when the last refresh
// happened, in std::chrono::steady_clock nanoseconds, and the refresh period. Used to keep the
// guest's 60 Hz frames in phase with a 60/120/240 Hz display, so each frame is shown for the same
// number of refreshes (no judder).

#include <cstdint>

namespace ps2x
{
    // False when unavailable (other platforms, or before the first refresh was seen).
    bool displayClockSample(int64_t &lastRefreshNs, int64_t &periodNs);

    // The same refresh clock in the host's media time (mach_absolute_time in nanoseconds, the
    // base of Metal's presentAtTime and VK_GOOGLE_display_timing on MoltenVK), and that clock now.
    bool displayClockSampleHost(int64_t &lastRefreshHostNs, int64_t &periodNs);
    int64_t hostTimeNowNs();
}
