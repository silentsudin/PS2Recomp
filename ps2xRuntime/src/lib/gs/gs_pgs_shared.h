#pragma once

// Shared between the paraLLEl-GS backend and the Vulkan presenter (gs_pgs_presenter.cpp): one
// Vulkan device, one lock for everything recorded on it, and the newest scanout image.

#if defined(PS2X_HAVE_PGS)

#include "device.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>

namespace ps2x
{
    class HostPresenter;
}

namespace ps2x::gs
{
    struct PgsShared
    {
        Vulkan::Device *device = nullptr;
        // Held for every use of the device (the GS thread mirroring packets, the render thread
        // presenting). The device has one thread index, so every holder registers index 0.
        std::mutex mutex;
        // The newest picture, in SHADER_READ_ONLY_OPTIMAL (guarded by `mutex`).
        Vulkan::ImageHandle scanout;
        // The raw Z behind it (R32F, same size) when wantDepth is set (guarded by `mutex`).
        Vulkan::ImageHandle depth;
        std::atomic<bool> wantDepth{false};
        // A paraLLEl-GS backend renders on this device (otherwise the CPU GS: pictures are uploaded).
        bool attached = false;
        // Submits the GS's open command buffers (with `mutex` held). A frame context only advances
        // once every command buffer requested in it is submitted, so the presenter calls this
        // before starting a swapchain frame.
        std::function<void()> flushLocked;
    };

    // The shared device of a presenter made by createPgsPresenter, or nullptr.
    PgsShared *pgsShared(HostPresenter *presenter);

    // Registers Granite thread index 0 for the calling thread (once per thread).
    void pgsRegisterThread();
}

#endif
