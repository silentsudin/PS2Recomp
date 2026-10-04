// The raylib/OpenGL presenter (see ps2_host_presenter.h): the GS picture is latched, copied to
// the CPU and uploaded into a GL texture, drawn letterboxed to 4:3, then the host UI (rlImGui).

#include "runtime/ps2_host_presenter.h"

#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"
#include "ps2_host_backend.h"
#include "rlgl.h"
#include "runtime/ps2_test_harness.h"
#include "runtime/ee_scheduler.h"

#if defined(PS2X_ENABLE_DEBUG_UI) && !defined(PLATFORM_VITA)
#include "imgui.h"
#include "rlImGui.h"
#define PS2X_RAYLIB_PRESENTER_UI 1
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
    constexpr int FB_WIDTH = 640;
    constexpr int FB_HEIGHT = 512;
    constexpr int DEFAULT_DISPLAY_HEIGHT = 448;
    constexpr uint32_t DEFAULT_FB_SIZE = FB_WIDTH * FB_HEIGHT * 4;

void UploadFrame(Texture2D &tex, PS2Runtime *rt, uint32_t &outWidth, uint32_t &outHeight)
{
    static uint64_t s_lastPresentationTick = std::numeric_limits<uint64_t>::max();
    static bool s_hasLatchedInitialFrame = false;
    static uint32_t s_lastDisplayFbp = std::numeric_limits<uint32_t>::max();
    static uint32_t s_lastSourceFbp = std::numeric_limits<uint32_t>::max();
    static bool s_lastPreferred = false;
    static uint32_t s_lastWidth = 0u;
    static uint32_t s_lastHeight = 0u;
    static bool s_hasUploadedFrame = false;
    static std::vector<uint8_t> s_scratch;
    static std::vector<uint8_t> s_uploadBuffer(DEFAULT_FB_SIZE, 0u);
    // The texture starts at 640x512 and grows to fit larger pictures (e.g. the GPU GS's
    // high-resolution progressive scanout).
    static int s_texWidth = FB_WIDTH;
    static int s_texHeight = FB_HEIGHT;

    const uint64_t currentTick = rt->eeScheduler().currentVSyncTick();
    const bool capture = ps2_test::frameCaptureRequested();
    // Headless test runs only need pictures that a test asks for; latching every frame would read
    // back each one from the GPU and hold up the GS thread.
    static const bool headless = [] { const char *e = std::getenv("RT_HEADLESS"); return e && *e == '1'; }();
    if (headless && !capture)
    {
        outWidth = s_lastWidth ? s_lastWidth : FB_WIDTH;
        outHeight = s_lastHeight ? s_lastHeight : DEFAULT_DISPLAY_HEIGHT;
        return;
    }
    const bool needsLatch = !s_hasLatchedInitialFrame || currentTick != s_lastPresentationTick || capture;
    if (needsLatch)
    {
        rt->gsUnsynced().latchHostPresentationFrame();
        s_lastPresentationTick = currentTick;
        s_hasLatchedInitialFrame = true;
    }
    else if (s_hasUploadedFrame)
    {
        outWidth = (s_lastWidth != 0u) ? s_lastWidth : FB_WIDTH;
        outHeight = (s_lastHeight != 0u) ? s_lastHeight : DEFAULT_DISPLAY_HEIGHT;
        return;
    }

    s_scratch.clear();
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;
    bool usedPreferredDisplaySource = false;
    const bool haveFrame = rt->gsUnsynced().copyLatchedHostPresentationFrame(s_scratch,
                                                                    width,
                                                                    height,
                                                                    &displayFbp,
                                                                    &sourceFbp,
                                                                    &usedPreferredDisplaySource);
    if (capture)
        ps2_test::deliverFrameCapture(haveFrame ? s_scratch : std::vector<uint8_t>{}, haveFrame ? width : 0u,
                                      haveFrame ? height : 0u);
    if (!haveFrame)
    {
        Image blank = GenImageColor(s_texWidth, s_texHeight, MAGENTA);
        UpdateTexture(tex, blank.data);
        UnloadImage(blank);
        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        s_lastWidth = outWidth;
        s_lastHeight = outHeight;
        s_hasUploadedFrame = true;
        return;
    }

    PS2_IF_AGRESSIVE_LOGS({
        static uint32_t s_uploadDebugCount = 0u;
        if (s_uploadDebugCount < 128u ||
            displayFbp != s_lastDisplayFbp ||
            sourceFbp != s_lastSourceFbp ||
            usedPreferredDisplaySource != s_lastPreferred ||
            width != s_lastWidth ||
            height != s_lastHeight)
        {
            std::cout << "[frame:upload] idx=" << s_uploadDebugCount
                      << " tick=" << currentTick
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " size=" << width << "x" << height
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u)
                      << std::endl;
        }
        ++s_uploadDebugCount;
    });
    s_lastDisplayFbp = displayFbp;
    s_lastSourceFbp = sourceFbp;
    s_lastPreferred = usedPreferredDisplaySource;
    s_lastWidth = width;
    s_lastHeight = height;

    if (static_cast<int>(width) > s_texWidth || static_cast<int>(height) > s_texHeight)
    {
        s_texWidth = std::max<int>(s_texWidth, static_cast<int>(width));
        s_texHeight = std::max<int>(s_texHeight, static_cast<int>(height));
        UnloadTexture(tex);
        Image blank = GenImageColor(s_texWidth, s_texHeight, BLANK);
        tex = LoadTextureFromImage(blank);
        UnloadImage(blank);
        SetTextureFilter(tex, TEXTURE_FILTER_BILINEAR);
        s_uploadBuffer.assign(static_cast<size_t>(s_texWidth) * s_texHeight * 4u, 0u);
    }

    std::fill(s_uploadBuffer.begin(), s_uploadBuffer.end(), 0u);
    if (!s_scratch.empty() && width != 0u && height != 0u)
    {
        const uint32_t copyWidth = std::min<uint32_t>(width, s_texWidth);
        const uint32_t copyHeight = std::min<uint32_t>(height, s_texHeight);
        const size_t srcRowBytes = static_cast<size_t>(width) * 4u;
        const size_t dstRowBytes = static_cast<size_t>(s_texWidth) * 4u;
        const size_t copyRowBytes = static_cast<size_t>(copyWidth) * 4u;
        for (uint32_t y = 0; y < copyHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * srcRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * dstRowBytes;
            if (srcOffset + copyRowBytes > s_scratch.size() ||
                dstOffset + copyRowBytes > s_uploadBuffer.size())
            {
                break;
            }
            std::memcpy(s_uploadBuffer.data() + dstOffset, s_scratch.data() + srcOffset, copyRowBytes);
        }
    }

    UpdateTexture(tex, s_uploadBuffer.data());
    outWidth = width;
    outHeight = height;
    s_hasUploadedFrame = true;
}

    class RaylibPresenter final : public ps2x::HostPresenter
    {
    public:
        const char *name() const override { return "raylib"; }

        bool open(const char *title, int width, int height) override
        {
#if !defined(PLATFORM_VITA)
            SetConfigFlags(FLAG_WINDOW_RESIZABLE);
#endif
            InitWindow(width, height, title);
            if (!IsWindowReady())
            {
                std::cerr << "[presenter] raylib could not open a window" << std::endl;
                return false;
            }
#if defined(PLATFORM_VITA)
            SetTargetFPS(60);
#else
            if (const char *hostFps = std::getenv("RT_HOST_FPS"))
                SetTargetFPS(std::atoi(hostFps));
            else
                SetTargetFPS(0); // paced by guest vblanks in run()
#endif
            Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLANK);
            m_tex = LoadTextureFromImage(blank);
            UnloadImage(blank);
            SetTextureFilter(m_tex, TEXTURE_FILTER_BILINEAR);
            m_open = true;
            return true;
        }

        void close() override
        {
            if (!m_open)
                return;
            UnloadTexture(m_tex);
            CloseWindow();
            m_open = false;
        }

        bool closeRequested() override { return WindowShouldClose(); }

        void uiInit() override
        {
#if defined(PS2X_RAYLIB_PRESENTER_UI)
            if (!ImGui::GetCurrentContext())
                rlImGuiSetup(true);
#endif
        }
        void uiShutdown() override
        {
#if defined(PS2X_RAYLIB_PRESENTER_UI)
            if (ImGui::GetCurrentContext())
                rlImGuiShutdown();
#endif
        }
        void uiBegin() override
        {
#if defined(PS2X_RAYLIB_PRESENTER_UI)
            rlImGuiBegin();
#endif
        }
        void uiEnd() override
        {
#if defined(PS2X_RAYLIB_PRESENTER_UI)
            rlImGuiEnd();
#endif
        }

        bool captureWindow(const std::string &pngPath) override
        {
            m_capturePath = pngPath;
            return true;
        }

        void frame(PS2Runtime &runtime, const std::function<void()> &drawUi) override
        {
            uint32_t presentWidth = FB_WIDTH;
            uint32_t presentHeight = DEFAULT_DISPLAY_HEIGHT;
            UploadFrame(m_tex, &runtime, presentWidth, presentHeight);

            BeginDrawing();
            ClearBackground(BLACK);
            const float srcWidth = static_cast<float>(std::max<uint32_t>(1u, presentWidth));
            const float srcHeight = static_cast<float>(std::max<uint32_t>(1u, presentHeight));
            const float screenWidth = static_cast<float>(GetScreenWidth());
            const float screenHeight = static_cast<float>(GetScreenHeight());
            // The PS2 always drives a 4:3 TV: a 640x224 field buffer or a 512x448 frame buffer
            // both fill the whole picture. Fit a 4:3 box instead of scaling pixels 1:1.
            constexpr float kDisplayAspect = 4.0f / 3.0f;
            const float dstWidth = std::min(screenWidth, screenHeight * kDisplayAspect);
            const float dstHeight = dstWidth / kDisplayAspect;
            const Rectangle srcRect{0.0f, 0.0f, srcWidth, srcHeight};
            const Rectangle dstRect{(screenWidth - dstWidth) * 0.5f, (screenHeight - dstHeight) * 0.5f, dstWidth, dstHeight};
            DrawTexturePro(m_tex, srcRect, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
            if (drawUi)
                drawUi();
            if (!m_capturePath.empty())
            {
                rlDrawRenderBatchActive(); // draw what is batched, so the read sees this frame
                Image image = LoadImageFromScreen();
                ExportImage(image, m_capturePath.c_str());
                UnloadImage(image);
                m_capturePath.clear();
            }
            EndDrawing();
        }

    private:
        Texture2D m_tex{};
        bool m_open = false;
        std::string m_capturePath;
    };
}

std::unique_ptr<ps2x::HostPresenter> ps2x::createRaylibPresenter()
{
    return std::make_unique<RaylibPresenter>();
}
