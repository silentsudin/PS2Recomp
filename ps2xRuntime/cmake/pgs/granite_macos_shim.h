// Force-included into Granite's granite-util on macOS: Darwin has no clock_nanosleep /
// TIMER_ABSTIME, which Granite's util/timer.cpp uses for sleep_until_nsecs().
#pragma once
#if defined(__APPLE__)
#include <errno.h>
#include <time.h>

#ifndef TIMER_ABSTIME
#define TIMER_ABSTIME 1

static inline int clock_nanosleep(clockid_t clock, int flags, const struct timespec *request,
                                  struct timespec *remain)
{
    struct timespec duration = *request;
    if (flags & TIMER_ABSTIME)
    {
        struct timespec now;
        if (clock_gettime(clock, &now) != 0)
            return errno;
        long long delta = (request->tv_sec - now.tv_sec) * 1000000000ll + (request->tv_nsec - now.tv_nsec);
        if (delta <= 0)
            return 0;
        duration.tv_sec = (time_t)(delta / 1000000000ll);
        duration.tv_nsec = (long)(delta % 1000000000ll);
        remain = NULL; // absolute sleeps are simply re-issued by the caller on EINTR
    }
    return nanosleep(&duration, remain) == 0 ? 0 : errno;
}
#endif
#endif
