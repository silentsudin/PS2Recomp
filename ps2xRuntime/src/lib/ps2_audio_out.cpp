#include "ps2_audio_out.h"
#include "runtime/ps2_audio_suspend.h"
#include "runtime/ps2_test_harness.h"

#include "raylib.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace
{
    constexpr unsigned kSampleRate = 48000;
    constexpr size_t kCapacityFrames = kSampleRate / 2;   // 500 ms
    constexpr size_t kMaxQueuedFrames = kSampleRate / 5;  // drop beyond 200 ms to bound latency
    // The queue is held near 50 ms by playing it a little faster or slower (linear interpolation):
    // the SPU2 runs on the guest clock and the device on its own, and the difference (about 0.02% on
    // the Thor) otherwise grows the queue until the cap drops frames, which crackles.
    constexpr double kTargetFrames = kSampleRate * 0.05;
    double g_frac = 0.0; // the read position's fraction between g_read and the next frame
    // After the queue ran dry (the game paused under a host menu or in the background, a stall),
    // play silence until it holds half the target again: resuming on a near-empty queue
    // underran every few callbacks (crackles) while the rate matching took ~10 s to build it up.
    constexpr size_t kPrimeFrames = static_cast<size_t>(kTargetFrames * 0.5);
    bool g_priming = true;

    std::mutex g_mutex;
    std::vector<int16_t> g_ring(kCapacityFrames * 2);
    size_t g_read = 0, g_write = 0, g_count = 0; // in frames
    AudioStream g_stream{};
    std::atomic<bool> g_started{false};
    // RT_AUDIO_STATS=1: underruns (the device asked for more than was queued: silence) and drops
    // (the SPU2 was over 200 ms ahead: the oldest frames go), logged every 5 s. Both click.
    std::atomic<uint64_t> g_underrunFrames{0}, g_underruns{0}, g_droppedFrames{0}, g_calls{0}, g_fillSum{0};

    void callback(void *buffer, unsigned int frames)
    {
        auto *out = static_cast<int16_t *>(buffer);
        std::lock_guard<std::mutex> lock(g_mutex);
        g_calls.fetch_add(1, std::memory_order_relaxed);
        g_fillSum.fetch_add(g_count, std::memory_order_relaxed);
        // Frames consumed per frame played: up to 0.5% slower when short, up to 1% faster when
        // well over (a backlog at start drains in seconds); too small a change to hear.
        const double error = (static_cast<double>(g_count) - kTargetFrames) / kTargetFrames;
        const double ratio = 1.0 + std::clamp(error * 0.004, -0.005, 0.01);
        if (g_priming && g_count >= kPrimeFrames)
            g_priming = false;
        size_t n = 0;
        if (g_priming)
            n = frames; // (silence below, not counted as an underrun)
        for (; n < frames && g_count >= 2; ++n)
        {
            const size_t next = (g_read + 1) % kCapacityFrames;
            for (int c = 0; c < 2; ++c)
            {
                const double a = g_ring[g_read * 2 + c], b = g_ring[next * 2 + c];
                out[n * 2 + c] = static_cast<int16_t>(std::lround(a + (b - a) * g_frac));
            }
            g_frac += ratio;
            while (g_frac >= 1.0 && g_count >= 2)
            {
                g_read = (g_read + 1) % kCapacityFrames;
                --g_count;
                g_frac -= 1.0;
            }
        }
        if (g_priming)
            std::memset(out, 0, frames * 2 * sizeof(int16_t));
        else if (n < frames)
        {
            std::memset(out + n * 2, 0, (frames - n) * 2 * sizeof(int16_t)); // underrun: silence
            g_underruns.fetch_add(1, std::memory_order_relaxed);
            g_underrunFrames.fetch_add(frames - n, std::memory_order_relaxed);
            g_priming = g_count < 2; // ran dry: build the queue up again before playing on
        }
        static const bool stats = [] { const char *e = std::getenv("RT_AUDIO_STATS"); return e && *e == '1'; }();
        static auto windowStart = std::chrono::steady_clock::now();
        if (stats && std::chrono::steady_clock::now() - windowStart > std::chrono::seconds(5))
        {
            windowStart = std::chrono::steady_clock::now();
            const uint64_t calls = g_calls.exchange(0);
            std::fprintf(stderr, "[audio] 5 s: %llu callbacks of %u frames, %llu underruns (%llu frames), %llu frames dropped, "
                                 "average queue %llu frames\n",
                         (unsigned long long)calls, frames, (unsigned long long)g_underruns.exchange(0),
                         (unsigned long long)g_underrunFrames.exchange(0), (unsigned long long)g_droppedFrames.exchange(0),
                         (unsigned long long)(calls ? g_fillSum.exchange(0) / calls : 0));
        }
    }
}

void ps2AudioOutStart()
{
    // RT_AUDIO=0 turns sound off.
    const char *enabled = std::getenv("RT_AUDIO");
    if (enabled && *enabled == '0')
        return;
    if (g_started.exchange(true) || !IsAudioDeviceReady())
        return;
    SetAudioStreamBufferSizeDefault(1024);
    g_stream = LoadAudioStream(kSampleRate, 16, 2);
    SetAudioStreamCallback(g_stream, callback);
    PlayAudioStream(g_stream);
}

void ps2AudioOutStop()
{
    if (!g_started.exchange(false))
        return;
    StopAudioStream(g_stream);
    UnloadAudioStream(g_stream);
}

namespace
{
    std::atomic<float> g_gain{1.0f};
    std::atomic<bool> g_mono{false};
}

void ps2AudioOutSetMix(float gain, bool mono)
{
    g_gain = std::clamp(gain, 0.0f, 1.0f);
    g_mono = mono;
}

void ps2AudioOutFlush()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_read = g_write = g_count = 0;
    g_frac = 0.0;
    g_priming = true;
}

void ps2AudioOutSubmit(const int16_t *interleavedStereo, size_t frames)
{
    // RT_AUDIO_DUMP=<file>: also write everything as raw s16le 48 kHz stereo (diagnostics).
    static FILE *dump = [] {
        const char *path = std::getenv("RT_AUDIO_DUMP");
        return path && *path ? std::fopen(path, "wb") : nullptr;
    }();
    if (dump)
    {
        std::fwrite(interleavedStereo, sizeof(int16_t) * 2, frames, dump);
        std::fflush(dump);
    }
    ps2_test::onAudio(interleavedStereo, frames);
    if (!g_started.load(std::memory_order_relaxed))
        return;
    const float gain = g_gain.load(std::memory_order_relaxed);
    const bool mono = g_mono.load(std::memory_order_relaxed);
    const bool mix = mono || gain < 0.999f;
    std::lock_guard<std::mutex> lock(g_mutex);
    for (size_t i = 0; i < frames; ++i)
    {
        if (g_count >= kMaxQueuedFrames)
        {
            // Producer is ahead of real time: drop the oldest frame.
            g_read = (g_read + 1) % kCapacityFrames;
            --g_count;
            g_droppedFrames.fetch_add(1, std::memory_order_relaxed);
        }
        int16_t l = interleavedStereo[i * 2], r = interleavedStereo[i * 2 + 1];
        if (mix)
        {
            float fl = l * gain, fr = r * gain;
            if (mono)
                fl = fr = (fl + fr) * 0.5f;
            l = static_cast<int16_t>(std::clamp(fl, -32768.0f, 32767.0f));
            r = static_cast<int16_t>(std::clamp(fr, -32768.0f, 32767.0f));
        }
        g_ring[g_write * 2] = l;
        g_ring[g_write * 2 + 1] = r;
        g_write = (g_write + 1) % kCapacityFrames;
        ++g_count;
    }
}

void ps2AudioOutSuspend(bool suspend)
{
    static std::mutex m;
    static int depth = 0;
    static bool wasStarted = false;
    std::lock_guard<std::mutex> lock(m);
    if (suspend)
    {
        if (depth++ > 0)
            return;
        wasStarted = g_started.load();
        ps2AudioOutStop();
        if (IsAudioDeviceReady())
            CloseAudioDevice();
    }
    else
    {
        if (depth == 0 || --depth > 0)
            return;
        if (!wasStarted)
            return;
        InitAudioDevice();
        ps2AudioOutStart();
    }
}
