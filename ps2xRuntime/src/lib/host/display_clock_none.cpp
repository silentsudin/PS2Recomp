#include "runtime/ps2_display_clock.h"

#if !defined(__APPLE__)
bool ps2x::displayClockSample(int64_t &, int64_t &) { return false; }
#endif
