#pragma once

// Presentation through Metal on macOS (Road Trip recomp). The Vulkan presenter renders the
// finished picture into an offscreen image; this copies it into a drawable of the window's
// CAMetalLayer and presents it, on MoltenVK's own Metal queue (so after the Vulkan work), without
// the device lock the GS needs. MoltenVK's swapchain fetches its drawable lazily while encoding,
// under that lock, and at two presents per guest frame the wait cost the game frames.
// All pointers are Objective-C objects passed as void *.

#include <cstdint>
#include <functional>
#include <memory>

namespace ps2x::gs
{
    class MetalPresent
    {
    public:
        // `layer` is the window's CAMetalLayer (SDL_Metal_GetLayer); nullptr if Metal is unavailable.
        static std::unique_ptr<MetalPresent> create(void *mtlDevice, void *layer, bool vsync);
        virtual ~MetalPresent() = default;

        // Copies `texture` (BGRA8 unorm, width x height) into the next drawable and presents it at
        // `atSeconds` (host media time, CACurrentMediaTime base; 0 = at the next refresh). Waits for
        // a drawable (call it with no locks held). `onPresented` gets the time it reached the
        // screen (0 if it never did), from a Metal thread.
        virtual bool present(void *queue, void *texture, uint32_t width, uint32_t height, double atSeconds,
                             std::function<void(double presentedSeconds)> onPresented) = 0;
        virtual void setVsync(bool vsync) = 0;
    };
}
