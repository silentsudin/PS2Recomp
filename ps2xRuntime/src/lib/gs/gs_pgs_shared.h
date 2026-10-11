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
        // The game's map (course map, town minimap, field maps) drawn on its own (a second screen, or
        // the app's own map on the main screen, shows it). wantMap is read once per game frame, at
        // the frame's first draw (`mapLatched` says what the frame being drawn does): with it set, a
        // backend that can (the hardware GS) draws that frame's map batches into a layer of their
        // own instead of the frame, and publishes it when the game moves on to its next frame: the
        // layer (`map`, opaque, the frame buffer's width by 256 lines, its GS coordinates), the map
        // itself in it (mapUv: u0, v0, u1, v1; the town bezel left out), everything the map drew
        // (mapOuterUv: with the bezel) and where that was in the picture scanned out (mapPicture,
        // 0..1 of it), its width over height on a 4:3 TV, its kind and a serial, all at once (under
        // `mutex`; the presenter takes one copy per host frame). `map` is null while none is shown:
        // cleared at once when wantMap turns off, and when frames come without a map.
        // mapFramesSeen counts the game frames that drew a map, diverted or not (the game's own
        // signal that a map is up: town, field, highway or race).
        std::atomic<bool> wantMap{false};
        std::atomic<bool> mapLatched{false};
        std::atomic<uint64_t> mapFramesSeen{0};
        Vulkan::ImageHandle map;
        float mapUv[4] = {};
        float mapOuterUv[4] = {};
        float mapPicture[4] = {};
        float mapAspect = 1.0f;
        uint32_t mapKind = 0; // 1 the town's window (scissored, with its bezel), 2 another map (a course's)
        uint64_t mapSerial = 0;
        // Re-rendered frame generation: shadow GS instances replay the frame with the 3D moved on by
        // (i + 1) / (wantShadows + 1) of a frame; their pictures, scanned out with the real one.
        std::atomic<uint32_t> wantShadows{0};
        Vulkan::ImageHandle shadowScanout[3];
        // Which real frame each shadow picture belongs to (presentSerial counts real scanouts).
        uint64_t presentSerial = 0;
        // The game's 3D frame on show (GSPresentationRequest::frame3D). Temporal passes (TAA, GSR 2,
        // MetalFX) run once per picture, and scale the motion (one frame's) by how many frames the
        // picture is past their history: with the game ahead of the presenter (a slow present,
        // lockstep tests), the history is several frames back.
        uint64_t pictureSerial = 0;
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
        // The backend records and submits only while holding `mutex` (no command buffer stays open
        // outside it), so the presenter may let it go while it waits for a swapchain image.
        bool submitsUnderLock = false;
    };

    // The shared device of a presenter made by createPgsPresenter, or nullptr.
    PgsShared *pgsShared(HostPresenter *presenter);

    // Registers Granite thread index 0 for the calling thread (once per thread).
    void pgsRegisterThread();
}

#endif
