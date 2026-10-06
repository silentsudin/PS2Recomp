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
        // Per-pixel screen motion (RG16F, GS pixels; wantMotion needs wantDepth too).
        Vulkan::ImageHandle motion;
        std::atomic<bool> wantMotion{false};
        // The UI mask (R8: 1 where the HUD / 2D screens drew), so post-processing spares the UI.
        Vulkan::ImageHandle ui;
        std::atomic<bool> wantUi{false};
        // The game's map (course map, town minimap) drawn on its own (a second screen shows it):
        // with wantMap set, a backend that can (the hardware GS) draws the map's batches into `map`
        // instead of the frame (an opaque image the size of the frame buffer), and mapUv is the part
        // the map covered in the last frame that drew it (u0, v0, u1, v1), mapAspect its width over
        // height on a 4:3 TV. `map` is null when the last frames drew no map (guarded by `mutex`).
        std::atomic<bool> wantMap{false};
        Vulkan::ImageHandle map;
        float mapUv[4] = {};
        float mapAspect = 1.0f;
        // Re-rendered frame generation: shadow GS instances replay the frame with the 3D moved on by
        // (i + 1) / (wantShadows + 1) of a frame; their pictures, scanned out with the real one.
        std::atomic<uint32_t> wantShadows{0};
        Vulkan::ImageHandle shadowScanout[3];
        // Which real frame each shadow picture belongs to (presentSerial counts real scanouts).
        uint64_t presentSerial = 0;
        uint64_t shadowSerial[3] = {};
        // A paraLLEl-GS backend renders on this device (otherwise the CPU GS: pictures are uploaded).
        bool attached = false;
        // How many scanout images the backend cycles through (0 = unknown): with 3 or more, a
        // picture stays untouched for two more presents, so the presenter can keep it as history
        // instead of copying it.
        uint32_t scanoutRing = 0;
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
