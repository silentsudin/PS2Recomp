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
            enum class AntiAliasing : uint8_t { None, Fxaa, Smaa };
            enum class Scaling : uint8_t { Bilinear, Fsr1, MetalFxSpatial };
            AntiAliasing aa = AntiAliasing::None;
            Scaling scaling = Scaling::Bilinear;
            float sharpness = 0.5f; // FSR 1 RCAS, 0 (soft) .. 1 (sharpest)
        };
        virtual bool supportsPostProcess() const { return false; }
        virtual void setPostProcess(const PostProcess &post) { (void)post; }

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
