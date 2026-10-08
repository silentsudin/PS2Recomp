#pragma once

#include <algorithm>
#include <string_view>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WINAPI
#define WINAPI __stdcall
#endif

extern "C"
{
    typedef void *HANDLE;
    typedef void *HMODULE;
    typedef const wchar_t *PCWSTR;
    typedef long HRESULT;

    __declspec(dllimport) HMODULE WINAPI GetModuleHandleW(const wchar_t *lpModuleName);
    __declspec(dllimport) void *WINAPI GetProcAddress(HMODULE hModule, const char *lpProcName);
    __declspec(dllimport) HANDLE WINAPI GetCurrentThread(void);
}
#elif defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#endif
#if defined(__ANDROID__)
#include <cstdio>
#include <cstdlib>
#include <sched.h>
#endif

namespace ThreadNaming
{
    inline void SetCurrentThreadName(std::string_view name)
    {
#if defined(_WIN32) 
        using SetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PCWSTR);

        HMODULE kernel32 = ::GetModuleHandleW(L"Kernel32.dll");
        auto setThreadDescription =
            reinterpret_cast<SetThreadDescriptionFn>(::GetProcAddress(kernel32, "SetThreadDescription"));

        if (setThreadDescription)
        {
            std::wstring wname(name.begin(), name.end());
            setThreadDescription(::GetCurrentThread(), wname.c_str());
        }

#elif defined(__APPLE__)
        pthread_setname_np(name.data());

#elif defined(__linux__)
        pthread_setname_np(pthread_self(), name.data());
#endif
    }

    // The emulation's hot threads (game, VIF1/VU1, GS): the scheduler keeps them on performance
    // cores. With default QoS macOS may move them to efficiency cores when the machine is busy,
    // and a game frame that only just fits in 16.7 ms then misses its vblank (30 fps).
    // On Android the scheduler moves them as it likes: on the AYN Thor the GS thread spent ~9% of
    // its time on the little cores (2.0 GHz against 2.8-3.2), where a frame takes twice as long,
    // and a few such frames in a row were late enough to pause frame generation. There they are
    // kept on the cores faster than the slowest cluster (RT_THREAD_AFFINITY=0: anywhere).
    inline void SetCurrentThreadInteractive()
    {
#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#elif defined(__ANDROID__)
        static const cpu_set_t *fast = []() -> const cpu_set_t * {
            const char *e = std::getenv("RT_THREAD_AFFINITY");
            if (e && *e == '0')
                return nullptr;
            long maxFreq[CPU_SETSIZE] = {};
            long lowest = 0, highest = 0;
            int cpus = 0;
            for (int c = 0; c < 64; ++c)
            {
                char path[96];
                std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
                FILE *f = std::fopen(path, "r");
                if (!f)
                    break;
                if (std::fscanf(f, "%ld", &maxFreq[c]) != 1)
                    maxFreq[c] = 0;
                std::fclose(f);
                cpus = c + 1;
                if (maxFreq[c] > 0 && (lowest == 0 || maxFreq[c] < lowest))
                    lowest = maxFreq[c];
                highest = std::max(highest, maxFreq[c]);
            }
            if (cpus == 0 || lowest == highest) // one cluster (or unknown): nothing to keep off
                return nullptr;
            static cpu_set_t set;
            CPU_ZERO(&set);
            int count = 0;
            for (int c = 0; c < cpus; ++c)
                if (maxFreq[c] > lowest)
                {
                    CPU_SET(c, &set);
                    ++count;
                }
            std::fprintf(stderr, "[threads] game, VU1 and GS threads on the %d faster cores of %d\n", count, cpus);
            return &set;
        }();
        if (fast)
            sched_setaffinity(0, sizeof(cpu_set_t), fast);
#endif
    }
}
