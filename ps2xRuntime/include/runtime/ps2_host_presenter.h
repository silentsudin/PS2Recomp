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

        // The SDL_Window of this presenter, or nullptr (raylib keeps its own).
        virtual void *sdlWindow() { return nullptr; }

        // Writes the next presented frame (with the UI) to a PNG. False if unsupported.
        virtual bool captureWindow(const std::string &pngPath)
        {
            (void)pngPath;
            return false;
        }
    };

    std::unique_ptr<HostPresenter> createRaylibPresenter();
}
