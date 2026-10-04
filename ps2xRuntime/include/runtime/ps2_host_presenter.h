#pragma once

// How the runtime shows the game: the host window, the newest GS picture letterboxed to 4:3,
// the host UI (Dear ImGui) on top, and presentation. PS2Runtime drives it from run() on the
// thread that called run().
//
//  - Raylib (default): raylib/OpenGL window. The GS picture is read back to the CPU and uploaded
//    into a GL texture each frame.
//  - paraLLEl-GS Vulkan (createPgsPresenter, gs_pgs_backend.h): an SDL3 window with a Vulkan
//    swapchain on the GS's own device. The picture stays on the GPU.
//
// Headless runs (RT_HEADLESS=1) use no presenter.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

class PS2Runtime;

namespace ps2x
{
    class HostPresenter
    {
    public:
        virtual ~HostPresenter() = default;
        virtual const char *name() const = 0;

        // Opens the window. False (with a message on stderr) on failure.
        virtual bool open(const char *title, int width, int height) = 0;
        virtual void close() = 0;
        // The window's close button (or the OS) asked to quit.
        virtual bool closeRequested() = 0;

        // Host UI: an ImGui context and its platform/renderer backends (no-ops without the debug
        // UI). uiBegin/uiEnd bracket the ImGui frame inside frame()'s drawUi.
        virtual void uiInit() {}
        virtual void uiShutdown() {}
        virtual void uiBegin() {}
        virtual void uiEnd() {}

        // One host frame: the newest GS picture (latched when the guest has shown a new one, or a
        // test capture is pending), then drawUi(), then present.
        virtual void frame(PS2Runtime &runtime, const std::function<void()> &drawUi) = 0;

        // Widescreen: the picture's shape on screen (4:3 by default). Frames the GS marks as
        // 2D-backed (title, menus; GS::lastFrameWas2D) are still shown 4:3, pillarboxed.
        void setDisplayAspect(float aspect) { m_displayAspect = aspect; }
        float displayAspect() const { return m_displayAspect; }

        // Post-processing of the game picture (the Vulkan presenter): edge anti-aliasing at the
        // render resolution, then scaling to the window.
        struct PostProcess
        {
            enum class AntiAliasing : uint8_t { None, Fxaa, Smaa, Taa };
            enum class Scaling : uint8_t { Bilinear, Fsr1, MetalFxSpatial, MetalFxTemporal };
            AntiAliasing aa = AntiAliasing::None;
            Scaling scaling = Scaling::Bilinear;
            float sharpness = 0.5f; // FSR 1 RCAS, 0 (soft) .. 1 (sharpest)
        };
        virtual bool supportsPostProcess() const { return false; }
        virtual void setPostProcess(const PostProcess &post) { (void)post; }

        // Frame generation for displays faster than the game's 60 Hz: `factor` frames are
        // presented per guest frame (2 at 120 Hz, 4 at 240 Hz), the real one first. 1 = off.
        struct FrameGeneration
        {
            uint32_t factor = 1;
            // Re-rendered (default): a shadow GS renders the frame again with every object moved
            // on along its motion (no added latency, real geometry). Otherwise the picture is
            // warped with the motion vectors: interpolated (one refresh of latency) or
            // extrapolated (edge artefacts).
            bool rerender = true;
            bool extrapolate = false;
        };
        virtual void setFrameGeneration(const FrameGeneration &fg) { (void)fg; }
        virtual uint32_t frameGenerationFactor() const { return 1; }
        // Presents per guest frame, set by the run loop before each frame(): the display's
        // refreshes per guest frame (2 at 120 Hz), so each present holds exactly one refresh. The
        // extra presents repeat the picture (no reprocessing) or carry generated frames.
        virtual void setPresentsPerFrame(uint32_t n) { (void)n; }

        // The SDL_Window of this presenter, or nullptr (raylib keeps its own).
        virtual void *sdlWindow() { return nullptr; }

        // Writes the next presented frame (with the UI) to a PNG. False if unsupported.
        virtual bool captureWindow(const std::string &pngPath)
        {
            (void)pngPath;
            return false;
        }

    protected:
        float m_displayAspect = 4.0f / 3.0f;
    };

    // The aspect to draw the newest picture at: the display aspect, or 4:3 for 2D-backed frames.
    float pictureAspect(PS2Runtime &runtime, const HostPresenter &presenter);

    std::unique_ptr<HostPresenter> createRaylibPresenter();
}
