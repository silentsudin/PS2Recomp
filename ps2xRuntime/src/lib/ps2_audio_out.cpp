#include "ps2_audio_out.h"
#include "runtime/ps2_test_harness.h"

#include "raylib.h"

#include <algorithm>
#include <atomic>
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

    std::mutex g_mutex;
    std::vector<int16_t> g_ring(kCapacityFrames * 2);
    size_t g_read = 0, g_write = 0, g_count = 0; // in frames
    AudioStream g_stream{};
    std::atomic<bool> g_started{false};

    void callback(void *buffer, unsigned int frames)
    {
        auto *out = static_cast<int16_t *>(buffer);
        std::lock_guard<std::mutex> lock(g_mutex);
        const size_t n = std::min<size_t>(frames, g_count);
        for (size_t i = 0; i < n; ++i)
        {
            out[i * 2] = g_ring[g_read * 2];
            out[i * 2 + 1] = g_ring[g_read * 2 + 1];
            g_read = (g_read + 1) % kCapacityFrames;
        }
        g_count -= n;
        if (n < frames)
            std::memset(out + n * 2, 0, (frames - n) * 2 * sizeof(int16_t)); // underrun: silence
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
    std::lock_guard<std::mutex> lock(g_mutex);
    for (size_t i = 0; i < frames; ++i)
    {
        if (g_count >= kMaxQueuedFrames)
        {
            // Producer is ahead of real time: drop the oldest frame.
            g_read = (g_read + 1) % kCapacityFrames;
            --g_count;
        }
        g_ring[g_write * 2] = interleavedStereo[i * 2];
        g_ring[g_write * 2 + 1] = interleavedStereo[i * 2 + 1];
        g_write = (g_write + 1) % kCapacityFrames;
        ++g_count;
    }
}
