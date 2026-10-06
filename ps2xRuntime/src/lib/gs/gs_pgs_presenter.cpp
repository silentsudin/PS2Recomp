// The Vulkan presenter (see ps2_host_presenter.h): an SDL3 window with a Granite WSI swapchain on
// the device paraLLEl-GS renders with. Each host frame draws the newest scanout image letterboxed
// to 4:3 and the host UI (Dear ImGui, drawn here through Granite with ImGui's own shaders), then
// presents. Nothing is read back to the CPU except for test captures and screenshots.

#include "runtime/ps2_display_clock.h"
#include "runtime/gs/gs_pgs_backend.h"
#include "runtime/ps2_host_presenter.h"

#if defined(PS2X_HAVE_PGS) && defined(PS2X_PGS_PRESENTER)

#include "gs_pgs_shared.h"
#include "runtime/gs/gs_motion.h"
#include "imgui_spirv.h"
#if defined(__APPLE__)
#include "gs_metalfx.h"
#include "gs_metal_present.h"
#endif
#include "post/post_spirv.h"

// FidelityFX FSR 1 constants, computed on the CPU (post/ffx, MIT).
#define A_CPU 1
#include "post/ffx/ffx_a.h"
#include "post/ffx/ffx_fsr1.h"
// SMAA's precomputed area and search textures (post/smaa, MIT).
#include "post/smaa/AreaTex.h"
#include "post/smaa/SearchTex.h"

#include "ps2_runtime.h"
#include "runtime/ee_scheduler.h"
#include "runtime/ps2_test_harness.h"

#include "context.hpp"
#include "device.hpp"
#include "wsi.hpp"

#include <SDL3/SDL_metal.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#if defined(__APPLE__)
#include <dlfcn.h>
#endif

#if defined(PS2X_ENABLE_DEBUG_UI)
#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#define PS2X_PGS_PRESENTER_UI 1
#endif

#include "raylib.h" // ExportImage for screenshots

#if defined(__ANDROID__)
#include <android/native_window.h>
#include <vulkan/vulkan_android.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>
#include <thread>
#include <vector>

namespace ps2x::gs
{
    namespace
    {
        // Present-at-time (VK_GOOGLE_display_timing; MoltenVK turns it into Metal's presentAtTime):
        // Granite presents through its device table, so its vkQueuePresentKHR is wrapped to add the
        // desired time the presenter set for this present.
        PFN_vkQueuePresentKHR g_realQueuePresent = nullptr;
        uint64_t g_desiredPresentNs = 0; // host media time; 0 = as soon as possible (presenter thread)
        uint64_t g_targetRefreshNs = 0;  // the refresh that present is aimed at
        uint32_t g_presentTimingId = 0;
        VkSwapchainKHR g_timedSwapchain = VK_NULL_HANDLE;
        // Recent presents: id -> the refresh it was aimed at (feedback for the present lead).
        struct TimedPresent
        {
            uint32_t id = 0;
            int64_t target = 0;
        };
        TimedPresent g_timedPresents[16];

        VKAPI_ATTR VkResult VKAPI_CALL queuePresentAtTime(VkQueue queue, const VkPresentInfoKHR *info)
        {
            const uint64_t desired = g_desiredPresentNs;
            g_desiredPresentNs = 0;
            if (!desired || !info || info->swapchainCount != 1)
                return g_realQueuePresent(queue, info);
            g_timedSwapchain = info->pSwapchains[0];
            const VkPresentTimeGOOGLE time = {++g_presentTimingId, desired};
            g_timedPresents[g_presentTimingId % 16] = {g_presentTimingId, static_cast<int64_t>(g_targetRefreshNs)};
            VkPresentTimesInfoGOOGLE times = {VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE, info->pNext, 1, &time};
            VkPresentInfoKHR withTime = *info;
            withTime.pNext = &times;
            return g_realQueuePresent(queue, &withTime);
        }

        bool presentAtTimeWanted()
        {
#if defined(__APPLE__)
            static const bool on = [] {
                const char *e = std::getenv("RT_PRESENT_AT_TIME");
                return !(e && *e == '0');
            }();
            return on;
#else
            return false;
#endif
        }

        class SdlPlatform final : public Vulkan::WSIPlatform
        {
        public:
            std::vector<const char *> get_device_extensions() override
            {
                if (presentAtTimeWanted())
                    return {"VK_KHR_swapchain", "VK_GOOGLE_display_timing"};
                return {"VK_KHR_swapchain"};
            }

            SdlPlatform(SDL_Window *window, bool surface) : m_window(window), m_surface(surface) {}

            VkSurfaceKHR create_surface(VkInstance instance, VkPhysicalDevice) override
            {
                VkSurfaceKHR surface = VK_NULL_HANDLE;
                if (!m_surface) // presenting through Metal: no Vulkan swapchain on this window
                    return VK_NULL_HANDLE;
                if (!SDL_Vulkan_CreateSurface(m_window, instance, nullptr, &surface))
                {
                    std::cerr << "[presenter] SDL_Vulkan_CreateSurface: " << SDL_GetError() << std::endl;
                    return VK_NULL_HANDLE;
                }
                return surface;
            }

            void destroy_surface(VkInstance instance, VkSurfaceKHR surface) override
            {
                SDL_Vulkan_DestroySurface(instance, surface, nullptr);
            }

            std::vector<const char *> get_instance_extensions() override
            {
                Uint32 count = 0;
                const char *const *names = SDL_Vulkan_GetInstanceExtensions(&count);
                return names ? std::vector<const char *>(names, names + count) : std::vector<const char *>{};
            }

            uint32_t get_surface_width() override
            {
                int w = 0, h = 0;
                SDL_GetWindowSizeInPixels(m_window, &w, &h);
                return static_cast<uint32_t>(std::max(w, 1));
            }

            uint32_t get_surface_height() override
            {
                int w = 0, h = 0;
                SDL_GetWindowSizeInPixels(m_window, &w, &h);
                return static_cast<uint32_t>(std::max(h, 1));
            }

            bool alive(Vulkan::WSI &) override { return true; }
            void poll_input() override {}
            void poll_input_async(Granite::InputTrackerHandler *) override {}

            void requestResize() { resize = true; }

        private:
            SDL_Window *m_window;
            bool m_surface;
        };

        // ImGui's vertex layout, also used for the game picture's quad.
        struct Vertex
        {
            float x, y, u, v;
            uint32_t colour;
        };

        class PgsPresenter final : public HostPresenter
        {
        public:
            explicit PgsPresenter(PgsPresenterOptions options) : m_options(std::move(options)) {}
            ~PgsPresenter() override { close(); }

            const char *name() const override { return "vulkan"; }

            PgsShared *shared() { return &m_shared; }

            bool open(const char *title, int width, int height) override
            {
                if (m_window)
                    return true; // opened early (the app's setup screen), now handed to the runtime
                if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
                    return fail("SDL video", SDL_GetError());
                if (!SDL_Vulkan_LoadLibrary(m_options.vulkanLibrary.empty() ? nullptr : m_options.vulkanLibrary.c_str()))
                    return fail("SDL_Vulkan_LoadLibrary", SDL_GetError());
#if defined(__APPLE__)
                // macOS presents through Metal (gs_metal_present.h); RT_METAL_PRESENT=0 uses the
                // Vulkan swapchain (MoltenVK) as other platforms do.
                if (const char *e = std::getenv("RT_METAL_PRESENT"); !(e && *e == '0'))
                    m_metalPresent = true;
#endif
                SDL_WindowFlags flags = (m_metalPresent ? SDL_WINDOW_METAL : SDL_WINDOW_VULKAN) | SDL_WINDOW_RESIZABLE |
                                        SDL_WINDOW_HIGH_PIXEL_DENSITY;
#if defined(__ANDROID__)
                flags |= SDL_WINDOW_FULLSCREEN; // immersive: no status or navigation bar
#endif
                m_window = SDL_CreateWindow(title, width, height, flags);
                if (!m_window)
                    return fail("SDL_CreateWindow", SDL_GetError());

                // Granite and SDL use the same Vulkan library (SDL's loader).
                auto getProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
                if (!getProc || !Vulkan::Context::init_loader(getProc))
                    return fail("Vulkan loader", "vkGetInstanceProcAddr unavailable");

                pgsRegisterThread();
                m_platform = std::make_unique<SdlPlatform>(m_window, !m_metalPresent);
                m_wsi.set_platform(m_platform.get());
                m_wsi.set_present_mode(m_options.vsync ? Vulkan::PresentMode::SyncToVBlank
                                                       : Vulkan::PresentMode::UnlockedMaybeTear);
                m_wsi.set_backbuffer_format(Vulkan::BackbufferFormat::UNORM);
                // begin_frame waits (VK_KHR_present_wait) until the present this many frames back is
                // on screen, under the device lock the GS thread needs. At one present per display
                // refresh, 1 makes that wait about a refresh long and the game misses frames.
                m_wsi.set_present_wait_latency(2);
                Vulkan::Context::SystemHandles handles = {};
                // As the GS's own device: push descriptors, no descriptor buffers/heaps.
                if (!m_wsi.init_context_from_platform(1, handles, Vulkan::CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT,
                                                      Vulkan::CONTEXT_CREATION_ENABLE_DESCRIPTOR_BUFFER_BIT |
                                                          Vulkan::CONTEXT_CREATION_ENABLE_DESCRIPTOR_HEAP_BIT))
                    return fail("Vulkan", "instance/device creation failed");
                if (!m_wsi.init_device() || (!m_metalPresent && !m_wsi.init_surface_swapchain()))
                    return fail("Vulkan", "swapchain creation failed");
                if (presentAtTimeWanted() && m_metalPresent)
                    m_presentAtTime = true; // Metal's presentDrawable:atTime:
                else if (presentAtTimeWanted())
                {
                    auto &table = const_cast<VolkDeviceTable &>(m_wsi.get_context().get_device_table());
                    g_realQueuePresent = table.vkQueuePresentKHR;
                    table.vkQueuePresentKHR = queuePresentAtTime;
                    m_presentAtTime = true;
                }
                m_shared.device = &m_wsi.get_device();
                // paraLLEl-GS advances a frame context on every flush, and the swapchain on every
                // frame: with Granite's default of 2 the game thread waits for the GPU inside a
                // flush. 4, as the GS's own device had.
                m_shared.device->init_frame_contexts(8);

                Vulkan::ResourceLayout vert = {};
                vert.input_mask = 0x7;
                vert.output_mask = 0x3;
                vert.push_constant_size = 16;
                Vulkan::ResourceLayout frag = {};
                frag.input_mask = 0x3;
                frag.output_mask = 0x1;
                frag.sets[0].sampled_image_mask = 0x1;
                frag.sets[0].fp_mask = 0x1;
                m_program = m_shared.device->request_program(imgui_spirv::__glsl_shader_vert_spv,
                                                             sizeof(imgui_spirv::__glsl_shader_vert_spv),
                                                             imgui_spirv::__glsl_shader_frag_spv,
                                                             sizeof(imgui_spirv::__glsl_shader_frag_spv), &vert, &frag);
                if (!m_program)
                    return fail("Vulkan", "presenter shaders");
                // Post-processing: a fullscreen triangle and one fragment shader per pass.
                Vulkan::ResourceLayout fsVert = {};
                fsVert.output_mask = 0x1;
                auto post = [&](const uint32_t *code, size_t size, uint32_t push, uint32_t inputs, uint32_t textures = 0x1) {
                    Vulkan::ResourceLayout frag = {};
                    frag.input_mask = inputs;
                    frag.output_mask = 0x1;
                    frag.push_constant_size = push;
                    frag.sets[0].sampled_image_mask = textures;
                    frag.sets[0].fp_mask = textures;
                    return m_shared.device->request_program(post_spirv::fullscreen_vert, sizeof(post_spirv::fullscreen_vert),
                                                            code, size, &fsVert, &frag);
                };
                m_fxaa = post(post_spirv::fxaa_frag, sizeof(post_spirv::fxaa_frag), 8, 0x1);
                m_easu = post(post_spirv::fsr_easu_frag, sizeof(post_spirv::fsr_easu_frag), 80, 0);
                m_rcas = post(post_spirv::fsr_rcas_frag, sizeof(post_spirv::fsr_rcas_frag), 32, 0);
                m_smaaEdges = post(post_spirv::smaa_edges_frag, sizeof(post_spirv::smaa_edges_frag), 16, 0x1, 0x1);
                m_smaaWeights = post(post_spirv::smaa_weights_frag, sizeof(post_spirv::smaa_weights_frag), 16, 0x1, 0x7);
                m_smaaBlend = post(post_spirv::smaa_blend_frag, sizeof(post_spirv::smaa_blend_frag), 16, 0x1, 0x3);
                m_depthView = post(post_spirv::depth_view_frag, sizeof(post_spirv::depth_view_frag), 4, 0x1);
                m_motionView = post(post_spirv::motion_view_frag, sizeof(post_spirv::motion_view_frag), 4, 0x1);
                m_taa = post(post_spirv::taa_frag, sizeof(post_spirv::taa_frag), 32, 0x1, 0x7);
                m_depthNormalize = post(post_spirv::depth_normalize_frag, sizeof(post_spirv::depth_normalize_frag), 0, 0x1);
                m_uiComposite = post(post_spirv::ui_composite_frag, sizeof(post_spirv::ui_composite_frag), 0, 0x1, 0x7);
                m_frameGen = post(post_spirv::frame_gen_frag, sizeof(post_spirv::frame_gen_frag), 32, 0x1, 0xF);
                m_copy = post(post_spirv::copy_frag, sizeof(post_spirv::copy_frag), 0, 0x1);
                m_flicker = post(post_spirv::flicker_frag, sizeof(post_spirv::flicker_frag), 0, 0x1, 0x7);
                m_flickerWeight = post(post_spirv::flicker_weight_frag, sizeof(post_spirv::flicker_weight_frag), 8, 0x1, 0xF);
                m_flickerGrid = post(post_spirv::flicker_grid_frag, sizeof(post_spirv::flicker_grid_frag), 0, 0x1, 0x3);
                m_flickerCut = post(post_spirv::flicker_cut_frag, sizeof(post_spirv::flicker_cut_frag), 0, 0x0, 0x1);
                if (const char *e = std::getenv("RT_SHOW_MOTION"); e && *e == '1')
                {
                    m_showMotion = true;
                    m_shared.wantDepth = true;
                    m_shared.wantMotion = true;
                    MotionTracker::instance().setEnabled(true);
                }
                if (const char *e = std::getenv("RT_SHOW_DEPTH"); e && *e == '1')
                {
                    m_showDepth = true;
                    m_shared.wantDepth = true;
                }
#if defined(__APPLE__)
                // MetalFX through MoltenVK's Metal objects.
                {
                    // MoltenVK's own functions: from the library SDL loaded (they are not instance
                    // commands, so vkGetInstanceProcAddr may not know them).
                    void *lib = dlopen(m_options.vulkanLibrary.empty() ? "libMoltenVK.dylib" : m_options.vulkanLibrary.c_str(),
                                       RTLD_NOW | RTLD_NOLOAD);
                    auto sym = [&](const char *name) -> void * {
                        void *f = lib ? dlsym(lib, name) : nullptr;
                        return f ? f : reinterpret_cast<void *>(getProc(m_wsi.get_context().get_instance(), name));
                    };
                    using GetDevice = void (*)(VkPhysicalDevice, void **);
                    auto getDevice = reinterpret_cast<GetDevice>(sym("vkGetMTLDeviceMVK"));
                    m_getTexture = reinterpret_cast<GetTexture>(sym("vkGetMTLTextureMVK"));
                    m_getQueue = reinterpret_cast<GetQueue>(sym("vkGetMTLCommandQueueMVK"));
                    void *mtlDevice = nullptr;
                    if (getDevice && m_getTexture && m_getQueue)
                        getDevice(m_wsi.get_context().get_gpu(), &mtlDevice);
                    m_mtlDevice = mtlDevice;
                    if (mtlDevice)
                    {
                        m_metalfx = MetalFxSpatial::create(mtlDevice);
                        m_metalfxTemporal = MetalFxTemporal::create(mtlDevice);
                    }
                    if (m_metalPresent)
                    {
                        m_metalView = SDL_Metal_CreateView(m_window);
                        if (mtlDevice && m_metalView)
                            m_metal = MetalPresent::create(mtlDevice, SDL_Metal_GetLayer(m_metalView), m_options.vsync);
                        if (!m_metal)
                            return fail("Metal", "could not present through the window's CAMetalLayer");
                    }
                    std::cout << "[presenter] MetalFX spatial "
                              << (m_metalfx ? "available"
                                            : !getDevice ? "unavailable (no MoltenVK Metal interop)"
                                            : !mtlDevice ? "unavailable (no Metal device)"
                                                         : "unavailable (not supported on this GPU/OS)")
                              << std::endl;
                }
#endif
                if (!m_fxaa || !m_easu || !m_rcas || !m_smaaEdges || !m_smaaWeights || !m_smaaBlend)
                    return fail("Vulkan", "post-processing shaders");
                {
                    auto area = Vulkan::ImageCreateInfo::immutable_2d_image(AREATEX_WIDTH, AREATEX_HEIGHT, VK_FORMAT_R8G8_UNORM);
                    Vulkan::ImageInitialData areaData = {areaTexBytes, 0, 0};
                    m_smaaArea = m_shared.device->create_image(area, &areaData);
                    auto search = Vulkan::ImageCreateInfo::immutable_2d_image(SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, VK_FORMAT_R8_UNORM);
                    Vulkan::ImageInitialData searchData = {searchTexBytes, 0, 0};
                    m_smaaSearch = m_shared.device->create_image(search, &searchData);
                }
                std::cout << "[presenter] " << (m_metalPresent ? "Metal layer " : "Vulkan swapchain ") << m_platform->get_surface_width()
                          << "x" << m_platform->get_surface_height() << (m_options.vsync ? " (vsync)" : "") << std::endl;
                return true;
            }

            void close() override
            {
                // No GPU waits here: when the loop has stopped, the GS thread may hold the device lock
                // waiting on a frame context that only presenting advances. The device and its
                // resources go with the presenter (or the process).
                m_window = nullptr;
            }

            bool closeRequested() override { return m_closeRequested; }

            void *sdlWindow() override { return m_window; }

            void uiInit() override
            {
#if defined(PS2X_PGS_PRESENTER_UI)
                if (m_ui)
                    return; // already (the setup screen)
                if (!ImGui::GetCurrentContext())
                    ImGui::CreateContext();
                ImGui_ImplSDL3_InitForVulkan(m_window);
                ImGuiIO &io = ImGui::GetIO();
                io.BackendRendererName = "ps2x_granite";
                io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures;
                m_ui = true;
#endif
            }

            void uiShutdown() override
            {
#if defined(PS2X_PGS_PRESENTER_UI)
                if (!m_ui)
                    return;
                ImGui_ImplSDL3_Shutdown();
                ImGui::DestroyContext();
                m_ui = false;
#endif
            }

            void uiBegin() override
            {
#if defined(PS2X_PGS_PRESENTER_UI)
                ImGui_ImplSDL3_NewFrame();
                ImGui::NewFrame();
#endif
            }

            void uiEnd() override
            {
#if defined(PS2X_PGS_PRESENTER_UI)
                ImGui::Render();
                m_uiFrame = true;
#endif
            }

            bool captureWindow(const std::string &pngPath) override
            {
                m_capturePath = pngPath;
                return true;
            }

            void frame(PS2Runtime &runtime, const std::function<void()> &drawUi) override
            {
                pumpEvents();

                updateTemporal(runtime);
                latch(runtime);
                m_pictureAspect = pictureAspect(runtime, *this);
                m_frame2D = runtime.gsUnsynced().lastFrameWas2D();
                m_uiFrame = false;
                if (drawUi)
                    drawUi();
                waitForDrawable();
                render();
            }

            void frameUi(const std::function<void()> &drawUi) override
            {
                pumpEvents();
                m_uiFrame = false;
                if (drawUi)
                    drawUi();
                waitForDrawable();
                render(); // no GS yet: a black picture under the UI
            }

        private:
            bool fail(const char *what, const char *why)
            {
                std::cerr << "[presenter] " << what << ": " << (why ? why : "") << std::endl;
                return false;
            }

            void pumpEvents()
            {
                SDL_Event e;
                while (SDL_PollEvent(&e))
                {
#if defined(PS2X_PGS_PRESENTER_UI)
                    if (m_ui)
                        ImGui_ImplSDL3_ProcessEvent(&e);
#endif
                    if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
                        m_closeRequested = true;
                    if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
                        m_platform->requestResize();
#if defined(__ANDROID__)
                    // Android takes the window's surface away while another activity is in front
                    // (the system file picker, the home screen): drop the Vulkan surface and
                    // swapchain with it, and make new ones when the app comes back. SDL reports it
                    // as the window being minimized and restored (its app background/foreground
                    // events don't reach this queue).
                    if ((e.type == SDL_EVENT_WINDOW_MINIMIZED || e.type == SDL_EVENT_WILL_ENTER_BACKGROUND) && !m_background)
                    {
                        std::lock_guard<std::mutex> lock(m_shared.mutex);
                        pgsRegisterThread();
                        m_shared.device->wait_idle();
                        m_wsi.deinit_surface_and_swapchain();
                        m_background = true;
                    }
                    if ((e.type == SDL_EVENT_WINDOW_RESTORED || e.type == SDL_EVENT_DID_ENTER_FOREGROUND) && m_background)
                    {
                        std::lock_guard<std::mutex> lock(m_shared.mutex);
                        pgsRegisterThread();
                        const VkSurfaceKHR surface =
                            m_platform->create_surface(m_shared.device->get_instance(), m_shared.device->get_physical_device());
                        if (surface != VK_NULL_HANDLE)
                        {
                            m_wsi.reinit_surface_and_swapchain(surface);
                            m_background = false;
                        }
                    }
#endif
                }
            }

            // The newest picture: kept on the GPU by the paraLLEl-GS backend on this device, or
            // (CPU GS) uploaded. Test captures also read it back.
            void latch(PS2Runtime &runtime)
            {
                const uint64_t tick = runtime.eeScheduler().currentVSyncTick();
                const bool capture = ps2_test::frameCaptureRequested();
                if (m_latched && tick == m_lastTick && !capture)
                    return;
                m_latched = true;
                m_lastTick = tick;
                const bool gpu = m_shared.attached;
                runtime.gsUnsynced().latchHostPresentationFrame(gpu, capture || !gpu);
                if (!capture && gpu)
                    return;
                std::vector<uint8_t> pixels;
                uint32_t w = 0, h = 0;
                const bool ok = runtime.gsUnsynced().copyLatchedHostPresentationFrame(pixels, w, h, nullptr, nullptr, nullptr);
                if (capture)
                    ps2_test::deliverFrameCapture(ok ? pixels : std::vector<uint8_t>{}, ok ? w : 0u, ok ? h : 0u);
                if (!gpu && ok && w && h)
                {
                    std::lock_guard<std::mutex> lock(m_shared.mutex);
                    pgsRegisterThread();
                    auto info = Vulkan::ImageCreateInfo::immutable_2d_image(w, h, VK_FORMAT_R8G8B8A8_UNORM);
                    Vulkan::ImageInitialData init = {pixels.data(), 0, 0};
                    m_cpuFrame = m_shared.device->create_image(info, &init);
                }
            }

            void bindState(Vulkan::CommandBuffer &cmd, float fbWidth, float fbHeight, float originX = 0, float originY = 0)
            {
                cmd.set_program(m_program);
                cmd.set_opaque_state();
                cmd.set_depth_test(false, false);
                cmd.set_cull_mode(VK_CULL_MODE_NONE);
                cmd.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
                cmd.set_blend_enable(true);
                cmd.set_blend_factors(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);
                cmd.set_blend_op(VK_BLEND_OP_ADD);
                cmd.set_vertex_attrib(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, x));
                cmd.set_vertex_attrib(1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, u));
                cmd.set_vertex_attrib(2, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(Vertex, colour));
                const float pc[4] = {2.0f / fbWidth, 2.0f / fbHeight, -1.0f - originX * 2.0f / fbWidth,
                                     -1.0f - originY * 2.0f / fbHeight};
                cmd.push_constants(pc, 0, sizeof(pc));
            }

            // The game picture, letterboxed to 4:3 (the PS2 always drives a 4:3 TV) or, with
            // widescreen, to the display aspect.
            // Where the picture goes on the swapchain: the letterbox rectangle (in whole pixels).
            VkRect2D pictureRect(float fw, float fh) const
            {
                const float w = std::floor(std::min(fw, fh * m_pictureAspect)), h = std::floor(w / m_pictureAspect);
                return {{static_cast<int32_t>((fw - w) * 0.5f), static_cast<int32_t>((fh - h) * 0.5f)},
                        {static_cast<uint32_t>(w), static_cast<uint32_t>(h)}};
            }

            // Renders a fullscreen pass from `src` into `dst` (re-created at w x h when needed).
            struct PassInput
            {
                const Vulkan::Image *image;
                Vulkan::StockSampler sampler;
            };

            void offscreenPass(Vulkan::CommandBuffer &cmd, Vulkan::ImageHandle &dst, uint32_t w, uint32_t h,
                               Vulkan::Program *program, const Vulkan::Image &src, const void *push, uint32_t pushSize)
            {
                const PassInput input[1] = {{&src, Vulkan::StockSampler::LinearClamp}};
                offscreenPass(cmd, dst, w, h, program, input, 1, push, pushSize, false);
            }

            void offscreenPass(Vulkan::CommandBuffer &cmd, Vulkan::ImageHandle &dst, uint32_t w, uint32_t h,
                               Vulkan::Program *program, const PassInput *inputs, uint32_t inputCount, const void *push,
                               uint32_t pushSize, bool clear, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM)
            {
                if (!dst || dst->get_width() != w || dst->get_height() != h || dst->get_format() != format)
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(w, h, format);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    dst = m_shared.device->create_image(info);
                }
                cmd.image_barrier(*dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                Vulkan::RenderPassInfo rp = {};
                rp.num_color_attachments = 1;
                rp.color_attachments[0] = &dst->get_view();
                rp.store_attachments = 1;
                rp.clear_attachments = clear ? 1 : 0;
                rp.clear_color[0] = {};
                cmd.begin_render_pass(rp);
                cmd.set_program(program);
                cmd.set_opaque_state();
                cmd.set_depth_test(false, false);
                cmd.set_cull_mode(VK_CULL_MODE_NONE);
                for (uint32_t i = 0; i < inputCount; ++i)
                    cmd.set_texture(0, i, inputs[i].image->get_view(), inputs[i].sampler);
                if (pushSize)
                    cmd.push_constants(push, 0, pushSize);
                cmd.draw(3);
                cmd.end_render_pass();
                cmd.image_barrier(*dst, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }

            // Before the swapchain pass: anti-aliasing at the render resolution, then FSR 1 EASU to
            // the picture's size on screen. Leaves the image the final pass draws in m_final.
            void preparePicture(Vulkan::CommandBufferHandle &cmdHandle, float fw, float fh)
            {
                Vulkan::CommandBuffer &cmd = *cmdHandle;
                const Vulkan::Image *image = sourceImage();
                m_final = image;
                m_finalRcas = false;
                if (!image)
                    return;
                const uint32_t sw = image->get_width(), sh = image->get_height();
                if (m_showMotion && m_shared.motion)
                {
                    const float scale = 1.0f / 32.0f; // 16 px of motion: fully red/green
                    const PassInput mv[1] = {{m_shared.motion.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_aaImage, sw, sh, m_motionView, mv, 1, &scale, sizeof(scale), false);
                    m_final = m_aaImage.get();
                    return;
                }
                if (m_showDepth && m_shared.depth)
                {
                    const float rcpMax = 1.0f / 16777216.0f;
                    const PassInput depth[1] = {{m_shared.depth.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_aaImage, sw, sh, m_depthView, depth, 1, &rcpMax, sizeof(rcpMax), false);
                    m_final = m_aaImage.get();
                    return;
                }
                const bool metalfxTemporal = m_post.scaling == PostProcess::Scaling::MetalFxTemporal;
                if (m_post.aa == PostProcess::AntiAliasing::Taa && !metalfxTemporal)
                {
                    if (m_noTemporal)
                        temporalAntiAliasingShadow(cmd, sw, sh);
                    else
                        temporalAntiAliasing(cmd, sw, sh);
                }
                else if (m_post.aa == PostProcess::AntiAliasing::Smaa)
                {
                    const float metrics[4] = {1.0f / static_cast<float>(sw), 1.0f / static_cast<float>(sh), static_cast<float>(sw),
                                              static_cast<float>(sh)};
                    const PassInput edges[1] = {{m_final, Vulkan::StockSampler::LinearClamp}};
                    offscreenPass(cmd, m_smaaEdgeImage, sw, sh, m_smaaEdges, edges, 1, metrics, sizeof(metrics), true);
                    const PassInput weights[3] = {{m_smaaEdgeImage.get(), Vulkan::StockSampler::LinearClamp},
                                                  {m_smaaArea.get(), Vulkan::StockSampler::LinearClamp},
                                                  {m_smaaSearch.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_smaaWeightImage, sw, sh, m_smaaWeights, weights, 3, metrics, sizeof(metrics), true);
                    const PassInput blend[2] = {{m_final, Vulkan::StockSampler::LinearClamp},
                                                {m_smaaWeightImage.get(), Vulkan::StockSampler::LinearClamp}};
                    offscreenPass(cmd, m_aaImage, sw, sh, m_smaaBlend, blend, 2, metrics, sizeof(metrics), false);
                    m_final = m_aaImage.get();
                }
                else if (m_post.aa == PostProcess::AntiAliasing::Fxaa)
                {
                    const float rcp[2] = {1.0f / static_cast<float>(sw), 1.0f / static_cast<float>(sh)};
                    offscreenPass(cmd, m_aaImage, sw, sh, m_fxaa, *m_final, rcp, sizeof(rcp));
                    m_final = m_aaImage.get();
                }
                const VkRect2D rect = pictureRect(fw, fh);
#if defined(__APPLE__)
                // Temporal needs a progressive picture: progressive fields, or 4x supersampling and
                // up (full frames). Interlaced fields below that alternate every frame: spatial.
                const bool progressive = sh >= 600 || m_progressiveFields;
                if (metalfxTemporal && progressive && m_metalfxTemporal && rect.extent.width > sw && m_shared.motion &&
                    m_shared.depth && m_shared.motion->get_width() == sw && m_shared.depth->get_width() == sw)
                {
                    // Beyond the scaler's largest ratio (640x448 to a Retina window is about 3.5x),
                    // it upscales as far as it can and the presentation scales the rest.
                    const float k = std::min({m_metalfxTemporal->maxScale(), float(rect.extent.width) / float(sw),
                                              float(rect.extent.height) / float(sh)});
                    upscaleMetalFxTemporal(cmdHandle, sw, sh, uint32_t(float(sw) * k), uint32_t(float(sh) * k));
                    return;
                }
                if (metalfxTemporal)
                    m_temporalValid = m_temporalValidShadow = false; // a 2D screen: start the histories again afterwards
                if ((m_post.scaling == PostProcess::Scaling::MetalFxSpatial || (metalfxTemporal && !progressive)) && m_metalfx &&
                    rect.extent.width > sw)
                {
                    upscaleMetalFx(cmdHandle, sw, sh, rect.extent.width, rect.extent.height);
                    return;
                }
#endif
                if (m_post.scaling == PostProcess::Scaling::Fsr1 && rect.extent.width > sw)
                {
                    struct
                    {
                        AU1 con[16];
                        uint32_t origin[4];
                    } push = {};
                    FsrEasuCon(push.con, push.con + 4, push.con + 8, push.con + 12, static_cast<AF1>(sw), static_cast<AF1>(sh),
                               static_cast<AF1>(sw), static_cast<AF1>(sh), static_cast<AF1>(rect.extent.width),
                               static_cast<AF1>(rect.extent.height));
                    offscreenPass(cmd, m_upImage, rect.extent.width, rect.extent.height, m_easu, *m_final, &push, sizeof(push));
                    m_final = m_upImage.get();
                    m_finalRcas = true;
                }
            }

            // TAA needs the game's depth/motion and a jittered camera: switched on and off with it.
            void updateTemporal(PS2Runtime &runtime)
            {
                m_progressiveFields = runtime.gsUnsynced().progressiveFields();
                const bool on = m_post.aa == PostProcess::AntiAliasing::Taa ||
                                m_post.scaling == PostProcess::Scaling::MetalFxTemporal;
                const bool motion = on || m_fg.factor > 1; // frame generation needs motion and depth too
                m_shared.wantShadows = m_fg.factor > 1 && m_fg.rerender ? m_fg.factor - 1 : 0u;
                if (!m_showMotion && !m_showDepth)
                {
                    m_shared.wantDepth = motion;
                    m_shared.wantMotion = motion;
                    MotionTracker::instance().setEnabled(motion);
                }
                // Jitter across one picture pixel: the frame buffer is 640 x 224 (fields), the
                // picture 2x or more of it.
                const Vulkan::Image *pic = m_shared.scanout.get();
                const float sx = pic ? 640.0f / static_cast<float>(pic->get_width()) : 0.5f;
                const float sy = pic ? 224.0f / static_cast<float>(pic->get_height()) : 0.25f;
                runtime.gsUnsynced().setTemporalJitter(on, sx, sy);
                if (on)
                    runtime.gsUnsynced().snapshotJitter(m_jitter[0], m_jitter[1], m_jitter[2], m_jitter[3]);
                if (!on)
                    m_taaValid = false;
                // Any post-processing (and frame generation) spares the UI (the GS marks where the HUD and 2D
                // screens drew).
                m_shared.wantUi = m_post.aa != PostProcess::AntiAliasing::None || m_post.scaling != PostProcess::Scaling::Bilinear ||
                                  m_fg.factor > 1; // generated frames keep the current frame's HUD
            }

            // A shadow frame (frame generation, half a frame ahead) blended with the real frame's
            // TAA result, reprojected half a frame with the real frame's motion; it does not become
            // history (the real frames' history stays a sequence of real frames).
            void temporalAntiAliasingShadow(Vulkan::CommandBuffer &cmd, uint32_t sw, uint32_t sh)
            {
                const Vulkan::ImageHandle &history = m_taaHistory[m_taaIndex ^ 1]; // the real frame's output
                if (!m_taaValid || !history || history->get_width() != sw || history->get_height() != sh || !m_shared.motion ||
                    m_shared.motion->get_width() != sw || m_shared.motion->get_height() != sh)
                    return;
                struct
                {
                    float motionToUv[2];
                    float jitterDelta[2];
                    float rcpSize[2];
                    float blend;
                    float historyValid;
                } push = {{0.5f / 640.0f, 0.5f / 224.0f}, {m_jitter[0] - m_jitter[2], m_jitter[1] - m_jitter[3]},
                          {1.0f / static_cast<float>(sw), 1.0f / static_cast<float>(sh)}, 0.1f, 1.0f};
                // Half the motion: a point at x in the shadow (frame N + 1/2) was at x - m/2 in
                // frame N, whose TAA output is the history here.
                const PassInput inputs[3] = {{m_final, Vulkan::StockSampler::NearestClamp},
                                             {history.get(), Vulkan::StockSampler::LinearClamp},
                                             {m_shared.motion.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(cmd, m_taaShadowImage, sw, sh, m_taa, inputs, 3, &push, sizeof(push), false);
                m_final = m_taaShadowImage.get();
            }

            void temporalAntiAliasing(Vulkan::CommandBuffer &cmd, uint32_t sw, uint32_t sh)
            {
                if (!m_shared.motion || m_shared.motion->get_width() != sw || m_shared.motion->get_height() != sh)
                {
                    m_taaValid = false; // a 2D screen (no 3D motion): nothing to accumulate
                    return;
                }
                Vulkan::ImageHandle &out = m_taaHistory[m_taaIndex];
                const Vulkan::ImageHandle &history = m_taaHistory[m_taaIndex ^ 1];
                const bool valid = m_taaValid && history && history->get_width() == sw && history->get_height() == sh;
                struct
                {
                    float motionToUv[2];
                    float jitterDelta[2];
                    float rcpSize[2];
                    float blend;
                    float historyValid;
                } push = {{1.0f / 640.0f, 1.0f / 224.0f}, {m_jitter[0] - m_jitter[2], m_jitter[1] - m_jitter[3]},
                          {1.0f / static_cast<float>(sw), 1.0f / static_cast<float>(sh)}, 0.1f, valid ? 1.0f : 0.0f};
                const PassInput inputs[3] = {{m_final, Vulkan::StockSampler::NearestClamp},
                                             {valid ? history.get() : m_final, Vulkan::StockSampler::LinearClamp},
                                             {m_shared.motion.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(cmd, out, sw, sh, m_taa, inputs, 3, &push, sizeof(push), false);
                m_final = out.get();
                m_taaIndex ^= 1;
                m_taaValid = true;
            }

#if defined(__APPLE__)
            // The passes so far go to the GPU and finish; MetalFX then scales m_final into the
            // window-sized m_upImage on the same Metal queue; the frame continues in a new command
            // buffer. (MoltenVK image layouts are no-ops in Metal, so the transition below only
            // keeps Vulkan's bookkeeping consistent.)
            void upscaleMetalFx(Vulkan::CommandBufferHandle &cmd, uint32_t sw, uint32_t sh, uint32_t w, uint32_t h)
            {
                Vulkan::Device &dev = *m_shared.device;
                if (!m_upImage || m_upImage->get_width() != w || m_upImage->get_height() != h ||
                    !(m_upImage->get_create_info().usage & VK_IMAGE_USAGE_STORAGE_BIT))
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(w, h, VK_FORMAT_R8G8B8A8_UNORM);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    m_upImage = dev.create_image(info);
                }
                cmd->image_barrier(*m_upImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                // Submitted (MoltenVK commits it to the Metal queue); MetalFX encodes after it on
                // that queue, so no CPU wait under the device lock.
                dev.submit(cmd);
                void *in = nullptr, *out = nullptr, *queue = nullptr;
                m_getTexture(m_final->get_image(), &in);
                m_getTexture(m_upImage->get_image(), &out);
                m_getQueue(m_wsi.get_context().get_queue_info().queues[Vulkan::QUEUE_INDEX_GRAPHICS], &queue);
                if (in && out && queue && m_metalfx->upscale(queue, in, sw, sh, out, w, h))
                    m_final = m_upImage.get();
                cmd = dev.request_command_buffer();
            }
#endif

#if defined(__APPLE__)
            // MetalFX temporal: depth normalised for it, then like the spatial scaler, with motion
            // (pointing back to last frame, in input pixels) and this frame's jitter.
            void upscaleMetalFxTemporal(Vulkan::CommandBufferHandle &cmd, uint32_t sw, uint32_t sh, uint32_t w, uint32_t h)
            {
                Vulkan::Device &dev = *m_shared.device;
                // Shadow frames (frame generation) have their own scaler and history: they arrive
                // one guest frame apart, like the real ones, half a frame later.
                if (m_noTemporal && !m_metalfxTemporalShadow && m_mtlDevice)
                    m_metalfxTemporalShadow = MetalFxTemporal::create(m_mtlDevice);
                MetalFxTemporal *scaler = m_noTemporal ? m_metalfxTemporalShadow.get() : m_metalfxTemporal.get();
                Vulkan::ImageHandle &upImage = m_noTemporal ? m_upImageShadow : m_upImage;
                bool &valid = m_noTemporal ? m_temporalValidShadow : m_temporalValid;
                if (!scaler)
                    return;
                const PassInput depthIn[1] = {{m_shared.depth.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(*cmd, m_depthNorm, sw, sh, m_depthNormalize, depthIn, 1, nullptr, 0, false, VK_FORMAT_R32_SFLOAT);
                if (!upImage || upImage->get_width() != w || upImage->get_height() != h ||
                    !(upImage->get_create_info().usage & VK_IMAGE_USAGE_STORAGE_BIT))
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(w, h, VK_FORMAT_R8G8B8A8_UNORM);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    upImage = dev.create_image(info);
                }
                cmd->image_barrier(*upImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                // Submitted (MoltenVK commits it to the Metal queue); MetalFX encodes after it on
                // that queue, so no CPU wait under the device lock.
                dev.submit(cmd);
                MetalFxTemporal::Frame f = {};
                m_getTexture(m_final->get_image(), &f.color);
                m_getTexture(m_depthNorm->get_image(), &f.depth);
                m_getTexture(m_shared.motion->get_image(), &f.motion);
                m_getTexture(upImage->get_image(), &f.output);
                void *queue = nullptr;
                m_getQueue(m_wsi.get_context().get_queue_info().queues[Vulkan::QUEUE_INDEX_GRAPHICS], &queue);
                const float px = static_cast<float>(sw) / 640.0f, py = static_cast<float>(sh) / 224.0f;
                f.inW = sw;
                f.inH = sh;
                f.outW = w;
                f.outH = h;
                f.jitterX = m_jitter[0] * px;
                f.jitterY = m_jitter[1] * py;
                f.motionScaleX = -px; // ours: current minus previous, GS pixels
                f.motionScaleY = -py;
                f.reset = !valid;
                if (f.color && f.depth && f.motion && f.output && queue && scaler->upscale(queue, f))
                {
                    m_final = upImage.get();
                    valid = true;
                }
                cmd = dev.request_command_buffer();
            }
#endif

            // The guest's picture: the scanout (or the CPU GS's upload), or with progressive fields
            // the scanout with alternating pixels blended (blendFlicker).
            const Vulkan::Image *sourceImage() const
            {
                if (m_sourceOverride)
                    return m_sourceOverride;
                if (m_flickerOut)
                    return m_flickerOut.get();
                const Vulkan::ImageHandle &raw = m_shared.attached ? m_shared.scanout : m_cpuFrame;
                return raw.get();
            }

            // Progressive fields show every field whole, so effects that alternate two pictures
            // and count on an interlaced TV to blend them would strobe (flicker.frag). Once per
            // guest frame: the raw picture against the two before it.
            void blendFlicker(Vulkan::CommandBuffer &cmd)
            {
                m_flickerOut.reset();
                const Vulkan::ImageHandle &raw = m_shared.attached ? m_shared.scanout : m_cpuFrame;
                // RT_FLICKER=0: no flicker blending (A/B measurements).
                static const bool flickerOff = [] {
                    const char *e = std::getenv("RT_FLICKER");
                    return e && *e == '0';
                }();
                if (!raw || !m_progressiveFields || m_showMotion || m_showDepth || flickerOff)
                {
                    m_flickerValid = 0;
                    return;
                }
                const uint32_t w = raw->get_width(), h = raw->get_height();
                Vulkan::ImageHandle &prev = m_flickerHist[m_flickerIndex], &before = m_flickerHist[m_flickerIndex ^ 1u];
                const bool valid = m_flickerValid >= 1 && prev && prev->get_width() == w && prev->get_height() == h;
                const bool haveBefore = valid && m_flickerValid >= 2 && before && before->get_width() == w &&
                                        before->get_height() == h;
                if (valid)
                {
                    // The cut: a 32x32 grid of drastic changes, then its share (no 2048 scattered reads
                    // in one pixel).
                    const PassInput gridIn[2] = {{raw.get(), Vulkan::StockSampler::NearestClamp},
                                                 {prev.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_flickerGridImage, 32, 32, m_flickerGrid, gridIn, 2, nullptr, 0, false);
                    const PassInput cutIn[1] = {{m_flickerGridImage.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_flickerCutImage, 1, 1, m_flickerCut, cutIn, 1, nullptr, 0, false);
                    // The weight at the game's resolution (a supersampled picture is 2x or 4x it), then
                    // the blend at full resolution.
                    const uint32_t ds = std::max(1u, w / 640u);
                    const uint32_t lw = std::max(1u, w / ds), lh = std::max(1u, h / ds);
                    const float rcp[2] = {1.0f / static_cast<float>(lw), 1.0f / static_cast<float>(lh)};
                    const PassInput weightIn[4] = {{raw.get(), Vulkan::StockSampler::NearestClamp},
                                                   {prev.get(), Vulkan::StockSampler::NearestClamp},
                                                   {haveBefore ? before.get() : prev.get(), Vulkan::StockSampler::NearestClamp},
                                                   {m_flickerCutImage.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_flickerWeightImage, lw, lh, m_flickerWeight, weightIn, 4, rcp, sizeof(rcp), false,
                                  VK_FORMAT_R8_UNORM);
                    const PassInput in[3] = {{raw.get(), Vulkan::StockSampler::NearestClamp},
                                             {prev.get(), Vulkan::StockSampler::NearestClamp},
                                             {m_flickerWeightImage.get(), Vulkan::StockSampler::LinearClamp}};
                    offscreenPass(cmd, m_flickerImage, w, h, m_flicker, in, 3, nullptr, 0, false);
                }
                // The raw picture becomes "previous"; the old previous becomes "before last".
                m_flickerIndex ^= 1u;
                if (m_shared.scanoutRing >= 3u && m_shared.attached)
                    m_flickerHist[m_flickerIndex] = raw; // a ring of 3+: kept as it is until overwritten
                else
                {
                    const PassInput copyIn[1] = {{raw.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_flickerHist[m_flickerIndex], w, h, m_copy, copyIn, 1, nullptr, 0, false);
                }
                m_flickerValid = std::min(m_flickerValid + 1u, 2u);
                if (valid)
                    m_flickerOut = m_flickerImage;
            }

            // Frame generation: the guest frame's picture is prepared once and kept (with the one
            // before it); each display refresh shows it or a frame made between/after them.
            void generatedPicture(Vulkan::CommandBufferHandle &cmd, float fw, float fh, bool fresh)
            {
                if (m_fg.rerender)
                {
                    // Re-rendered: the real frame, then the shadow GS's frames of the same guest
                    // frame with the 3D moved on (gs_pgs_backend.cpp). 2D screens repeat the real one.
                    if (fresh || !m_lastPicture)
                    {
                        preparePicture(cmd, fw, fh);
                        m_lastPicture = m_final;
                        m_lastRcas = m_finalRcas;
                        m_realSerial = m_shared.presentSerial;
                        return;
                    }
                    const uint32_t step = m_subframe * m_fg.factor / std::max(m_presents, m_fg.factor);
                    const Vulkan::Image *raw = (m_shared.attached ? m_shared.scanout : m_cpuFrame).get();
                    const Vulkan::Image *shadow = step >= 1 && step <= 3 ? m_shared.shadowScanout[step - 1].get() : nullptr;
                    // Only this real frame's own shadow (the worker may still be on it).
                    if (shadow && m_shared.shadowSerial[step - 1] != m_realSerial)
                        shadow = nullptr;
                    if (shadow && raw && !m_frame2D && shadow->get_width() == raw->get_width() &&
                        shadow->get_height() == raw->get_height())
                    {
                        m_sourceOverride = shadow; // until the next render: spareUi composites from it too
                        m_noTemporal = true;
                        preparePicture(cmd, fw, fh);
                        m_noTemporal = false;
                    }
                    else
                    {
                        m_final = m_lastPicture;
                        m_finalRcas = m_lastRcas;
                    }
                    return;
                }
                if (fresh || !m_genHistory[m_genIndex])
                {
                    preparePicture(cmd, fw, fh);
                    if (!m_final)
                        return;
                    m_genIndex ^= 1u;
                    m_genPrevValid = static_cast<bool>(m_genHistory[m_genIndex ^ 1u]);
                    const PassInput in[1] = {{m_final, Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(*cmd, m_genHistory[m_genIndex], m_final->get_width(), m_final->get_height(), m_copy, in, 1,
                                  nullptr, 0, false);
                    m_genRcas = m_finalRcas;
                    m_genJitterDelta[0] = m_jitter[0] - m_jitter[2];
                    m_genJitterDelta[1] = m_jitter[1] - m_jitter[3];
                }
                const Vulkan::Image *cur = m_genHistory[m_genIndex].get();
                const Vulkan::Image *prev = m_genHistory[m_genIndex ^ 1u].get();
                m_final = cur;
                m_finalRcas = m_genRcas;
                // The generated frame this present shows (presents beyond the factor repeat it).
                const uint32_t step = m_subframe * m_fg.factor / std::max(m_presents, m_fg.factor);
                const float k = static_cast<float>(m_fg.factor);
                const float t = m_fg.extrapolate ? static_cast<float>(step) / k
                                                 : static_cast<float>(std::min(step + 1u, m_fg.factor)) / k;
                const bool haveMotion = m_shared.motion && m_shared.depth && m_shared.motion->get_width() == m_shared.depth->get_width();
                const bool samePrev = prev && m_genPrevValid && prev->get_width() == cur->get_width() &&
                                      prev->get_height() == cur->get_height();
                const bool generate = haveMotion && (m_fg.extrapolate ? t > 0.0f : (t < 1.0f && samePrev));
                if (!generate)
                    return;
                struct
                {
                    float motionToUv[2];
                    float jitterDelta[2];
                    float rcpMotionSize[2];
                    float t;
                    float mode;
                } push = {{1.0f / 640.0f, 1.0f / 224.0f}, {m_genJitterDelta[0], m_genJitterDelta[1]},
                          {1.0f / static_cast<float>(m_shared.motion->get_width()), 1.0f / static_cast<float>(m_shared.motion->get_height())},
                          t, m_fg.extrapolate ? 1.0f : 0.0f};
                const PassInput in[4] = {{samePrev ? prev : cur, Vulkan::StockSampler::LinearClamp},
                                         {cur, Vulkan::StockSampler::LinearClamp},
                                         {m_shared.motion.get(), Vulkan::StockSampler::NearestClamp},
                                         {m_shared.depth.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(*cmd, m_genImage, cur->get_width(), cur->get_height(), m_frameGen, in, 4, &push, sizeof(push), false);
                m_final = m_genImage.get();
                (void)fw;
                (void)fh;
            }

            // Puts the UI back as the game drew it over the post-processed picture: the processed
            // image (finished at the picture's size on screen, RCAS included) is mixed with the
            // original by the GS's UI mask.
            void spareUi(Vulkan::CommandBuffer &cmd, float fw, float fh)
            {
                // A shadow frame takes its HUD from the real frame (same mask, same HUD).
                const Vulkan::Image *original = m_sourceOverride ? (m_flickerOut ? m_flickerOut.get()
                                                                                 : (m_shared.attached ? m_shared.scanout : m_cpuFrame).get())
                                                                 : sourceImage();
                if (!m_final || !original || m_final == original || !m_shared.ui ||
                    m_shared.ui->get_width() != original->get_width() || m_shared.ui->get_height() != original->get_height())
                    return;
                const VkRect2D rect = pictureRect(fw, fh);
                if (m_finalRcas)
                {
                    struct
                    {
                        AU1 con[4];
                        uint32_t origin[4];
                    } push = {};
                    FsrRcasCon(push.con, (1.0f - std::clamp(m_post.sharpness, 0.0f, 1.0f)) * 2.0f);
                    const PassInput in[1] = {{m_final, Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_rcasImage, rect.extent.width, rect.extent.height, m_rcas, in, 1, &push, sizeof(push), false);
                    m_final = m_rcasImage.get();
                    m_finalRcas = false;
                }
                const PassInput in[3] = {{m_final, Vulkan::StockSampler::LinearClamp},
                                         {original, Vulkan::StockSampler::LinearClamp},
                                         {m_shared.ui.get(), Vulkan::StockSampler::LinearClamp}};
                offscreenPass(cmd, m_uiImage, rect.extent.width, rect.extent.height, m_uiComposite, in, 3, nullptr, 0, false);
                m_final = m_uiImage.get();
            }

            void drawGame(Vulkan::CommandBuffer &cmd, float fw, float fh)
            {
                if (!m_final)
                    return;
                const VkRect2D rect = pictureRect(fw, fh);
                if (m_finalRcas)
                {
                    // FSR 1 RCAS, sharpening into the picture's rectangle on the swapchain.
                    struct
                    {
                        AU1 con[4];
                        uint32_t origin[4];
                    } push = {};
                    FsrRcasCon(push.con, (1.0f - std::clamp(m_post.sharpness, 0.0f, 1.0f)) * 2.0f);
                    push.origin[0] = static_cast<uint32_t>(rect.offset.x);
                    push.origin[1] = static_cast<uint32_t>(rect.offset.y);
                    cmd.set_program(m_rcas);
                    cmd.set_opaque_state();
                    cmd.set_depth_test(false, false);
                    cmd.set_cull_mode(VK_CULL_MODE_NONE);
                    cmd.set_viewport({static_cast<float>(rect.offset.x), static_cast<float>(rect.offset.y),
                                      static_cast<float>(rect.extent.width), static_cast<float>(rect.extent.height), 0.0f, 1.0f});
                    cmd.set_scissor(rect);
                    cmd.set_texture(0, 0, m_final->get_view(), Vulkan::StockSampler::NearestClamp);
                    cmd.push_constants(&push, 0, sizeof(push));
                    cmd.draw(3);
                    cmd.set_viewport({0.0f, 0.0f, fw, fh, 0.0f, 1.0f});
                    return;
                }
                const Vulkan::Image *image = m_final;
                const float w = static_cast<float>(rect.extent.width), h = static_cast<float>(rect.extent.height);
                const float x0 = static_cast<float>(rect.offset.x), y0 = static_cast<float>(rect.offset.y);
                bindState(cmd, fw, fh);
                cmd.set_blend_enable(false);
                cmd.set_scissor({{0, 0}, {static_cast<uint32_t>(fw), static_cast<uint32_t>(fh)}});
                auto *v = static_cast<Vertex *>(cmd.allocate_vertex_data(0, 4 * sizeof(Vertex), sizeof(Vertex)));
                v[0] = {x0, y0, 0, 0, 0xFFFFFFFFu};
                v[1] = {x0 + w, y0, 1, 0, 0xFFFFFFFFu};
                v[2] = {x0 + w, y0 + h, 1, 1, 0xFFFFFFFFu};
                v[3] = {x0, y0 + h, 0, 1, 0xFFFFFFFFu};
                auto *i = static_cast<uint16_t *>(cmd.allocate_index_data(6 * sizeof(uint16_t), VK_INDEX_TYPE_UINT16));
                const uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
                std::memcpy(i, idx, sizeof(idx));
                cmd.set_texture(0, 0, image->get_view(), Vulkan::StockSampler::LinearClamp);
                cmd.draw_indexed(6);
            }

        public:
            bool supportsPostProcess() const override { return true; }
            void setPostProcess(const PostProcess &post) override { m_post = post; }
            void setFrameGeneration(const FrameGeneration &fg) override
            {
                m_fg = fg;
                m_fg.factor = std::clamp<uint32_t>(fg.factor, 1u, 4u);
                // More presents in flight per guest frame: keep begin_frame's present-wait short.
                m_wsi.set_present_wait_latency(m_fg.factor > 1 ? 3u : 2u);
            }
            uint32_t frameGenerationFactor() const override { return m_fg.factor; }
            void setPresentsPerFrame(uint32_t n) override { m_presents = std::max(n, 1u); }

        private:

#if defined(PS2X_PGS_PRESENTER_UI)
            void updateTextures(ImDrawData *dd)
            {
                if (!dd->Textures)
                    return;
                for (ImTextureData *tex : *dd->Textures)
                {
                    if (tex->Status == ImTextureStatus_OK)
                        continue;
                    if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0)
                    {
                        m_textures.erase(tex->GetTexID());
                        tex->SetTexID(ImTextureID_Invalid);
                        tex->SetStatus(ImTextureStatus_Destroyed);
                        continue;
                    }
                    if (tex->Status != ImTextureStatus_WantCreate && tex->Status != ImTextureStatus_WantUpdates)
                        continue;
                    // (Re)upload the whole texture: atlas updates are rare.
                    std::vector<uint8_t> rgba;
                    const uint8_t *pixels = static_cast<const uint8_t *>(tex->GetPixels());
                    if (tex->Format == ImTextureFormat_Alpha8)
                    {
                        rgba.resize(static_cast<size_t>(tex->Width) * tex->Height * 4u);
                        for (int p = 0; p < tex->Width * tex->Height; ++p)
                        {
                            rgba[p * 4 + 0] = rgba[p * 4 + 1] = rgba[p * 4 + 2] = 0xFF;
                            rgba[p * 4 + 3] = pixels[p];
                        }
                        pixels = rgba.data();
                    }
                    auto info = Vulkan::ImageCreateInfo::immutable_2d_image(tex->Width, tex->Height, VK_FORMAT_R8G8B8A8_UNORM);
                    Vulkan::ImageInitialData init = {pixels, 0, 0};
                    ImTextureID id = tex->GetTexID();
                    if (id == ImTextureID_Invalid)
                        id = static_cast<ImTextureID>(++m_nextTexture);
                    m_textures[id] = m_shared.device->create_image(info, &init);
                    tex->SetTexID(id);
                    tex->SetStatus(ImTextureStatus_OK);
                }
            }

            void drawUi(Vulkan::CommandBuffer &cmd, float fw, float fh)
            {
                if (m_uiFrame)
                    drawImGui(cmd, ImGui::GetDrawData(), fw, fh);
            }

            // An ImGui texture: the presenter's own (kPictureTexture) or one of the font atlas's.
            const Vulkan::ImageView *uiTexture(ImTextureID id)
            {
                if (id == static_cast<ImTextureID>(kMapTexture))
                    return m_shared.map ? &m_shared.map->get_view() : nullptr;
                if (id == static_cast<ImTextureID>(kPictureTexture))
                {
                    const Vulkan::Image *picture = m_shared.attached ? m_shared.scanout.get() : m_cpuFrame.get();
                    return picture ? &picture->get_view() : nullptr;
                }
                auto it = m_textures.find(id);
                return it == m_textures.end() || !it->second ? nullptr : &it->second->get_view();
            }

            void drawImGui(Vulkan::CommandBuffer &cmd, ImDrawData *dd, float fw, float fh)
            {
                if (!dd || dd->TotalVtxCount == 0)
                    return;
                const ImVec2 scale = dd->FramebufferScale;
                // ImGui works in window points; the swapchain is in pixels.
                bindState(cmd, dd->DisplaySize.x, dd->DisplaySize.y, dd->DisplayPos.x, dd->DisplayPos.y);
                auto *v = static_cast<ImDrawVert *>(
                    cmd.allocate_vertex_data(0, static_cast<VkDeviceSize>(dd->TotalVtxCount) * sizeof(ImDrawVert), sizeof(ImDrawVert)));
                auto *ix = static_cast<ImDrawIdx *>(cmd.allocate_index_data(
                    static_cast<VkDeviceSize>(dd->TotalIdxCount) * sizeof(ImDrawIdx),
                    sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32));
                for (const ImDrawList *list : dd->CmdLists)
                {
                    std::memcpy(v, list->VtxBuffer.Data, list->VtxBuffer.Size * sizeof(ImDrawVert));
                    std::memcpy(ix, list->IdxBuffer.Data, list->IdxBuffer.Size * sizeof(ImDrawIdx));
                    v += list->VtxBuffer.Size;
                    ix += list->IdxBuffer.Size;
                }
                uint32_t vtxBase = 0, idxBase = 0;
                for (const ImDrawList *list : dd->CmdLists)
                {
                    for (const ImDrawCmd &c : list->CmdBuffer)
                    {
                        if (c.UserCallback)
                            continue;
                        const float x0 = std::max(0.0f, (c.ClipRect.x - dd->DisplayPos.x) * scale.x);
                        const float y0 = std::max(0.0f, (c.ClipRect.y - dd->DisplayPos.y) * scale.y);
                        const float x1 = std::min(fw, (c.ClipRect.z - dd->DisplayPos.x) * scale.x);
                        const float y1 = std::min(fh, (c.ClipRect.w - dd->DisplayPos.y) * scale.y);
                        if (x1 <= x0 || y1 <= y0)
                            continue;
                        const Vulkan::ImageView *view = uiTexture(c.GetTexID());
                        if (!view)
                            continue;
                        cmd.set_scissor({{static_cast<int32_t>(x0), static_cast<int32_t>(y0)},
                                         {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}});
                        cmd.set_texture(0, 0, *view,
                                        m_nearestTextures.count(static_cast<uint64_t>(c.GetTexID())) ||
                                                c.GetTexID() == static_cast<ImTextureID>(kMapTexture)
                                            ? Vulkan::StockSampler::NearestClamp
                                                                                                    : Vulkan::StockSampler::LinearClamp);
                        cmd.draw_indexed(c.ElemCount, 1, idxBase + c.IdxOffset, static_cast<int32_t>(vtxBase + c.VtxOffset), 0);
                    }
                    vtxBase += static_cast<uint32_t>(list->VtxBuffer.Size);
                    idxBase += static_cast<uint32_t>(list->IdxBuffer.Size);
                }
            }
#else
            void drawUi(Vulkan::CommandBuffer &, float, float) {}
#endif

            // ---- The second screen (see HostPresenter::setSecondScreen) ----
        public:
            void setSecondScreen(void *nativeWindow) override
            {
#if defined(__ANDROID__)
                if (nativeWindow)
                    ANativeWindow_acquire(static_cast<ANativeWindow *>(nativeWindow));
                std::lock_guard<std::mutex> lock(m_second.pendingMutex);
                if (m_second.pendingSet && m_second.pending)
                    ANativeWindow_release(static_cast<ANativeWindow *>(m_second.pending));
                m_second.pending = nativeWindow;
                m_second.pendingSet = true;
#else
                (void)nativeWindow;
#endif
            }

            bool secondScreenSize(int &width, int &height) const override
            {
                const uint64_t size = m_second.size.load();
                if (!size)
                    return false;
                width = static_cast<int>(size >> 32);
                height = static_cast<int>(size & 0xFFFFFFFFu);
                return true;
            }

            void submitSecondScreenUi(void *imguiContext) override { m_second.ui = imguiContext; }

            void setWantMap(bool want) override { m_shared.wantMap = want; }

            bool mapRegion(float uv[4], float &aspect) const override
            {
                std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(m_shared.mutex));
                if (!m_shared.map)
                    return false;
                std::memcpy(uv, m_shared.mapUv, sizeof(m_shared.mapUv));
                aspect = m_shared.mapAspect;
                return true;
            }

            uint64_t createUiTexture(const uint8_t *rgba, int width, int height, bool nearest) override
            {
                if (!rgba || width <= 0 || height <= 0 || !m_shared.device)
                    return 0;
                std::lock_guard<std::mutex> lock(m_shared.mutex);
                pgsRegisterThread();
                auto info = Vulkan::ImageCreateInfo::immutable_2d_image(static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                                                                        VK_FORMAT_R8G8B8A8_UNORM);
                Vulkan::ImageInitialData init = {rgba, 0, 0};
                const uint64_t id = ++m_nextTexture;
                m_textures[id] = m_shared.device->create_image(info, &init);
                if (nearest)
                    m_nearestTextures.insert(id);
                return id;
            }

        private:
            struct SecondScreen
            {
                std::mutex pendingMutex;
                void *pending = nullptr; // a new ANativeWindow (acquired), or nullptr
                bool pendingSet = false;
                void *window = nullptr; // the ANativeWindow in use (acquired)
                VkSurfaceKHR surface = VK_NULL_HANDLE;
                VkSwapchainKHR swapchain = VK_NULL_HANDLE;
                VkExtent2D extent = {};
                bool outOfDate = false;
                std::vector<Vulkan::ImageHandle> images;
                std::vector<Vulkan::ImageViewHandle> views;
                std::vector<Vulkan::Semaphore> release; // per image, until it is acquired again
                std::atomic<uint64_t> size{0};          // width << 32 | height, 0 = none
                void *ui = nullptr;                     // this frame's ImGui context, if submitted
            } m_second;

            void destroySecondSwapchain()
            {
                Vulkan::Device &dev = *m_shared.device;
                if (m_second.swapchain == VK_NULL_HANDLE)
                    return;
                dev.wait_idle();
                m_second.views.clear();
                m_second.images.clear();
                m_second.release.clear();
                dev.get_device_table().vkDestroySwapchainKHR(dev.get_device(), m_second.swapchain, nullptr);
                m_second.swapchain = VK_NULL_HANDLE;
                m_second.size = 0;
            }

            void destroySecondScreen()
            {
                destroySecondSwapchain();
#if defined(__ANDROID__)
                if (m_second.surface != VK_NULL_HANDLE)
                    vkDestroySurfaceKHR(m_shared.device->get_instance(), m_second.surface, nullptr);
                m_second.surface = VK_NULL_HANDLE;
                if (m_second.window)
                    ANativeWindow_release(static_cast<ANativeWindow *>(m_second.window));
                m_second.window = nullptr;
#endif
            }

            bool createSecondSwapchain()
            {
                Vulkan::Device &dev = *m_shared.device;
                const VkPhysicalDevice gpu = dev.get_physical_device();
                const auto &queues = m_wsi.get_context().get_queue_info();
                VkBool32 supported = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(gpu, queues.family_indices[Vulkan::QUEUE_INDEX_GRAPHICS], m_second.surface,
                                                     &supported);
                if (!supported)
                    return fail("second screen", "the graphics queue can't present to it");
                VkSurfaceCapabilitiesKHR caps = {};
                vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, m_second.surface, &caps);
                VkExtent2D extent = caps.currentExtent;
#if defined(__ANDROID__)
                if (extent.width == 0xFFFFFFFFu)
                    extent = {static_cast<uint32_t>(ANativeWindow_getWidth(static_cast<ANativeWindow *>(m_second.window))),
                              static_cast<uint32_t>(ANativeWindow_getHeight(static_cast<ANativeWindow *>(m_second.window)))};
#endif
                if (!extent.width || !extent.height)
                    return false;
                uint32_t count = 0;
                vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, m_second.surface, &count, nullptr);
                std::vector<VkSurfaceFormatKHR> formats(count);
                vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, m_second.surface, &count, formats.data());
                if (formats.empty())
                    return false;
                VkSurfaceFormatKHR format = formats[0];
                for (const auto &f : formats)
                    if (f.format == VK_FORMAT_R8G8B8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_UNORM)
                    {
                        format = f;
                        break;
                    }
                // Never wait for this screen: MAILBOX when there is one, and the acquire below
                // doesn't block either way.
                vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, m_second.surface, &count, nullptr);
                std::vector<VkPresentModeKHR> modes(count);
                vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, m_second.surface, &count, modes.data());
                const bool mailbox = std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end();
                VkSwapchainCreateInfoKHR info = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
                info.surface = m_second.surface;
                info.minImageCount = std::max(caps.minImageCount, 3u);
                if (caps.maxImageCount)
                    info.minImageCount = std::min(info.minImageCount, caps.maxImageCount);
                info.imageFormat = format.format;
                info.imageColorSpace = format.colorSpace;
                info.imageExtent = extent;
                info.imageArrayLayers = 1;
                info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                // The compositor turns the picture with the display; no rotated rendering here.
                info.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                                        ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                                        : caps.currentTransform;
                info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
                for (VkCompositeAlphaFlagBitsKHR a : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
                                                      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR})
                    if (caps.supportedCompositeAlpha & a)
                    {
                        info.compositeAlpha = a;
                        break;
                    }
                info.presentMode = mailbox ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
                info.clipped = VK_TRUE;
                const auto &table = dev.get_device_table();
                if (table.vkCreateSwapchainKHR(dev.get_device(), &info, nullptr, &m_second.swapchain) != VK_SUCCESS)
                {
                    m_second.swapchain = VK_NULL_HANDLE;
                    return fail("second screen", "vkCreateSwapchainKHR failed");
                }
                table.vkGetSwapchainImagesKHR(dev.get_device(), m_second.swapchain, &count, nullptr);
                std::vector<VkImage> images(count);
                table.vkGetSwapchainImagesKHR(dev.get_device(), m_second.swapchain, &count, images.data());
                auto imageInfo = Vulkan::ImageCreateInfo::render_target(extent.width, extent.height, format.format);
                imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                for (VkImage image : images)
                {
                    m_second.images.push_back(dev.wrap_image(imageInfo, image));
                    Vulkan::ImageViewCreateInfo view = {};
                    view.image = m_second.images.back().get();
                    view.format = format.format;
                    view.view_type = VK_IMAGE_VIEW_TYPE_2D;
                    view.levels = 1;
                    view.layers = 1;
                    m_second.views.push_back(dev.create_image_view(view));
                }
                m_second.release.resize(images.size());
                m_second.extent = extent;
                m_second.outOfDate = false;
                m_second.size = (static_cast<uint64_t>(extent.width) << 32) | extent.height;
                std::cout << "[presenter] second screen " << extent.width << "x" << extent.height
                          << (mailbox ? " (mailbox)" : " (fifo)") << std::endl;
                return true;
            }

            // With the device lock held, after the main screen's frame.
            void renderSecondScreen()
            {
                Vulkan::Device &dev = *m_shared.device;
#if defined(__ANDROID__)
                {
                    std::lock_guard<std::mutex> lock(m_second.pendingMutex);
                    if (m_second.pendingSet)
                    {
                        destroySecondScreen();
                        m_second.window = m_second.pending;
                        m_second.pending = nullptr;
                        m_second.pendingSet = false;
                        if (m_second.window)
                        {
                            auto create = reinterpret_cast<PFN_vkCreateAndroidSurfaceKHR>(
                                vkGetInstanceProcAddr(dev.get_instance(), "vkCreateAndroidSurfaceKHR"));
                            VkAndroidSurfaceCreateInfoKHR info = {VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
                            info.window = static_cast<ANativeWindow *>(m_second.window);
                            if (!create || create(dev.get_instance(), &info, nullptr, &m_second.surface) != VK_SUCCESS)
                            {
                                m_second.surface = VK_NULL_HANDLE;
                                fail("second screen", "vkCreateAndroidSurfaceKHR failed");
                            }
                        }
                    }
                }
#endif
                void *ui = m_second.ui;
                m_second.ui = nullptr;
                if (m_second.surface == VK_NULL_HANDLE)
                    return;
                if (m_second.outOfDate)
                    destroySecondSwapchain();
                if (m_second.swapchain == VK_NULL_HANDLE && !createSecondSwapchain())
                    return;
#if defined(PS2X_PGS_PRESENTER_UI)
                if (!ui)
                    return;
                Vulkan::Semaphore acquire = dev.request_semaphore(VK_SEMAPHORE_TYPE_BINARY);
                uint32_t index = 0;
                const auto &table = dev.get_device_table();
                const VkResult acquired = table.vkAcquireNextImageKHR(dev.get_device(), m_second.swapchain, 0,
                                                                      acquire->get_semaphore(), VK_NULL_HANDLE, &index);
                if (acquired == VK_ERROR_OUT_OF_DATE_KHR || acquired == VK_ERROR_SURFACE_LOST_KHR)
                {
                    m_second.outOfDate = true;
                    return;
                }
                if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR)
                    return; // no image free yet (the screen is behind): skip it this frame
                acquire->signal_external();
                dev.add_wait_semaphore(Vulkan::CommandBuffer::Type::Generic, std::move(acquire),
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, true);

                ImGuiContext *previous = ImGui::GetCurrentContext();
                ImGui::SetCurrentContext(static_cast<ImGuiContext *>(ui));
                ImDrawData *dd = ImGui::GetDrawData();
                if (dd)
                    updateTextures(dd);
                const Vulkan::Image &image = *m_second.images[index];
                const float fw = static_cast<float>(m_second.extent.width), fh = static_cast<float>(m_second.extent.height);
                auto cmd = dev.request_command_buffer();
                cmd->image_barrier(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                Vulkan::RenderPassInfo rp = {};
                rp.num_color_attachments = 1;
                rp.color_attachments[0] = m_second.views[index].get();
                rp.clear_attachments = 1;
                rp.store_attachments = 1;
                cmd->begin_render_pass(rp);
                drawImGui(*cmd, dd, fw, fh);
                cmd->end_render_pass();
                ImGui::SetCurrentContext(previous);
                cmd->image_barrier(image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_NONE, 0);
                Vulkan::Semaphore release;
                dev.submit(cmd, nullptr, 1, &release);

                const VkSemaphore wait = release->get_semaphore();
                VkResult result = VK_SUCCESS;
                VkPresentInfoKHR present = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
                present.waitSemaphoreCount = 1;
                present.pWaitSemaphores = &wait;
                present.swapchainCount = 1;
                present.pSwapchains = &m_second.swapchain;
                present.pImageIndices = &index;
                present.pResults = &result;
                const VkQueue queue = m_wsi.get_context().get_queue_info().queues[Vulkan::QUEUE_INDEX_GRAPHICS];
                const PFN_vkQueuePresentKHR presentFn = g_realQueuePresent ? g_realQueuePresent : table.vkQueuePresentKHR;
                dev.external_queue_lock();
                const VkResult overall = presentFn(queue, &present);
                dev.external_queue_unlock();
                release->wait_external(); // consumed by the present, even when it fails
                m_second.release[index] = std::move(release);
                // (SUBOPTIMAL is expected: the compositor turns the unrotated picture.)
                if (overall == VK_ERROR_OUT_OF_DATE_KHR || overall == VK_ERROR_SURFACE_LOST_KHR)
                    m_second.outOfDate = true;
#else
                (void)ui;
#endif
            }

            void drawScene(Vulkan::CommandBuffer &cmd, float fw, float fh)
            {
                drawGame(cmd, fw, fh);
                drawUi(cmd, fw, fh);
            }

            void render()
            {
                std::unique_lock<std::mutex> lock(m_shared.mutex);
                pgsRegisterThread();
                Vulkan::Device &dev = *m_shared.device;
#if defined(PS2X_PGS_PRESENTER_UI)
                if (m_uiFrame)
                    updateTextures(ImGui::GetDrawData());
#endif
                if (m_shared.flushLocked)
                    m_shared.flushLocked();
                const Vulkan::Image *backImage = nullptr;
                if (m_metalPresent)
                {
                    // Into an offscreen backbuffer (a ring of three: Metal may still be copying the
                    // one before), presented through Metal after the lock is released.
                    dev.next_frame_context();
                    int pw = 0, ph = 0;
                    SDL_GetWindowSizeInPixels(m_window, &pw, &ph);
                    const uint32_t bw = static_cast<uint32_t>(std::max(pw, 1)), bh = static_cast<uint32_t>(std::max(ph, 1));
                    m_backIndex = (m_backIndex + 1) % 3;
                    Vulkan::ImageHandle &bb = m_backbuffers[m_backIndex];
                    if (!bb || bb->get_width() != bw || bb->get_height() != bh)
                    {
                        auto info = Vulkan::ImageCreateInfo::render_target(bw, bh, VK_FORMAT_B8G8R8A8_UNORM);
                        info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                        info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                        bb = dev.create_image(info);
                    }
                    backImage = bb.get();
                }
                else
                {
                    if (m_background || !m_wsi.begin_frame())
                        return;
                    backImage = &dev.get_swapchain_view().get_image();
                }
                const Vulkan::Image &back = *backImage;
                const float fw = static_cast<float>(back.get_width()), fh = static_cast<float>(back.get_height());
                auto cmd = dev.request_command_buffer();
                // A new guest frame, or another display refresh of the same one.
                m_sourceOverride = nullptr;
                const bool fresh = m_lastTick != m_renderedTick;
                m_renderedTick = m_lastTick;
                m_subframe = fresh ? 0u : m_subframe + 1u;
                if (fresh)
                    blendFlicker(*cmd);
                if (m_fg.factor > 1)
                    generatedPicture(cmd, fw, fh, fresh);
                else if (fresh || !m_lastPicture)
                {
                    preparePicture(cmd, fw, fh);
                    m_lastPicture = m_final;
                    m_lastRcas = m_finalRcas;
                }
                else
                {
                    // Another refresh of the same guest frame: the same picture (TAA and MetalFX
                    // history must see each guest frame once).
                    m_final = m_lastPicture;
                    m_finalRcas = m_lastRcas;
                }
                spareUi(*cmd, fw, fh);
#if defined(__APPLE__)
                if (m_metalPresent)
                {
                    cmd->image_barrier(back, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, 0,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                    Vulkan::RenderPassInfo rp = {};
                    rp.num_color_attachments = 1;
                    rp.color_attachments[0] = &back.get_view();
                    rp.clear_attachments = 1;
                    rp.store_attachments = 1;
                    cmd->begin_render_pass(rp);
                    drawScene(*cmd, fw, fh);
                    cmd->end_render_pass();
                    cmd->image_barrier(back, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                       VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                    dev.submit(cmd); // MoltenVK commits it to the Metal queue before returning
                    if (!m_capturePath.empty())
                        captureLocked(fw, fh);
                    schedulePresent();
                    void *texture = nullptr, *queue = nullptr;
                    m_getTexture(back.get_image(), &texture);
                    m_getQueue(m_wsi.get_context().get_queue_info().queues[Vulkan::QUEUE_INDEX_GRAPHICS], &queue);
                    const uint64_t desired = g_desiredPresentNs, target = g_targetRefreshNs;
                    g_desiredPresentNs = 0;
                    lock.unlock();
                    // Waits for a drawable here, holding nothing the GS thread needs.
                    m_metal->present(queue, texture, back.get_width(), back.get_height(), desired ? static_cast<double>(desired) / 1e9 : 0.0,
                                     [this, target](double presented) {
                                         if (presented > 0.0 && target)
                                             notePresented(static_cast<int64_t>(target), static_cast<int64_t>(presented * 1e9));
                                     });
                    return;
                }
#endif
                auto rp = dev.get_swapchain_render_pass(Vulkan::SwapchainRenderPass::ColorOnly);
                rp.clear_color[0] = {};
                cmd->begin_render_pass(rp);
                drawScene(*cmd, fw, fh);
                cmd->end_render_pass();
                dev.submit(cmd);
                if (!m_capturePath.empty())
                    captureLocked(fw, fh);
                schedulePresent();
                m_wsi.end_frame();
                renderSecondScreen();
            }

            // Metal's presented times (from a Metal thread) for the adaptive present lead.
            void notePresented(int64_t target, int64_t presented)
            {
                std::lock_guard<std::mutex> lock(m_presentedMutex);
                if (m_presented.size() < 64)
                    m_presented.push_back({target, presented});
            }

            // With present-at-time, presents wait in the queue for their refresh, so the swapchain's
            // images (Metal keeps at most 3 drawables) can all be in flight at 120 presents per
            // second, and acquiring the next one would block inside render(), under the device
            // lock the GS thread needs (the game then drops to 30 fps). Wait here instead, outside
            // the lock, until the present two back is on screen and its image is free again.
            void waitForDrawable()
            {
                if (m_metalPresent)
                    return; // Metal fetches its drawable after the device lock is released
                const int64_t due = m_presentTargets[(m_presentCount + 1) % 3]; // two presents back
                if (!m_presentAtTime || !due)
                    return;
                const int64_t wait = due + 1000000 - ps2x::hostTimeNowNs();
                if (wait > 0)
                    std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<int64_t>(wait, 50000000)));
            }

            // Puts each present on a fixed grid of display refreshes: one every
            // refresh-rate / present-rate refreshes (every 2nd at 60 fps on a 120 Hz panel), so
            // every frame is on screen for the same time. Left alone, presents land anywhere in a
            // refresh and frames alternate between 1 and 3 refreshes (judder).
            void schedulePresent()
            {
                int64_t lastRefresh = 0, period = 0;
                if (!m_presentAtTime || !ps2x::displayClockSampleHost(lastRefresh, period) || ps2_test::paused())
                {
                    m_presentTarget = 0;
                    m_presentTargets[0] = m_presentTargets[1] = m_presentTargets[2] = 0;
                    return;
                }
                const double presentRate = 60.0 * static_cast<double>(std::max(m_presents, 1u));
                const int64_t step = period * std::max<int64_t>(1, std::llround(1e9 / static_cast<double>(period) / presentRate));
                // The earliest refresh this frame can make: the GPU still has the frame's work to
                // do and Core Animation needs it before that refresh. The lead adapts to what
                // this machine needs (on-screen times from VK_GOOGLE_display_timing): two misses
                // close together add 1 ms, every 30 frames on time take a quarter off (2 ms to one
                // refresh). A fixed half refresh was too little for an 8x race on an M3 Max; a
                // refresh and a half cost 8 ms of latency.
                updatePresentLead(period);
                const int64_t earliest = ps2x::hostTimeNowNs() + m_presentLead;
                auto onGrid = [&](int64_t t) { return lastRefresh + ((t - lastRefresh + period - 1) / period) * period; };
                int64_t target = m_presentTarget ? m_presentTarget + step : onGrid(earliest);
                if (target < earliest || target > earliest + 2 * step)
                    target = onGrid(earliest); // late (a frame was slow) or drifted: start the grid again
                m_presentTarget = target;
                m_presentTargets[m_presentCount++ % 3] = target;
                // Half a refresh early: Metal shows it at the first refresh at or after this time.
                g_desiredPresentNs = static_cast<uint64_t>(target - period / 2);
                g_targetRefreshNs = static_cast<uint64_t>(target);
            }

            void updatePresentLead(int64_t period)
            {
                const int64_t minLead = 2000000, maxLead = period;
                if (m_presentLead == 0)
                    m_presentLead = period / 2;
                // (target refresh, time on screen) of recent presents: from Metal's presented
                // handlers, or VK_GOOGLE_display_timing on the swapchain path.
                std::vector<std::pair<int64_t, int64_t>> shown;
                if (m_metalPresent)
                {
                    std::lock_guard<std::mutex> lock(m_presentedMutex);
                    shown.swap(m_presented);
                }
                else
                {
                    if (!m_pastTiming && g_timedSwapchain)
                        m_pastTiming = reinterpret_cast<PFN_vkGetPastPresentationTimingGOOGLE>(
                            vkGetDeviceProcAddr(m_wsi.get_context().get_device(), "vkGetPastPresentationTimingGOOGLE"));
                    if (!m_pastTiming || !g_timedSwapchain)
                        return;
                    VkPastPresentationTimingGOOGLE past[16];
                    uint32_t count = 16;
                    const VkResult r = m_pastTiming(m_wsi.get_context().get_device(), g_timedSwapchain, &count, past);
                    if (r != VK_SUCCESS && r != VK_INCOMPLETE)
                        return;
                    for (uint32_t i = 0; i < count; ++i)
                    {
                        const TimedPresent &sent = g_timedPresents[past[i].presentID % 16];
                        if (sent.id == past[i].presentID && sent.target && past[i].actualPresentTime)
                            shown.push_back({sent.target, static_cast<int64_t>(past[i].actualPresentTime)});
                    }
                }
                for (const auto &[sentTarget, actual] : shown)
                {
                    // Metal's presented time sits a steady few ms after the display link's refresh
                    // time; a miss shows up a whole refresh later than that offset.
                    const int64_t offset = actual - sentTarget;
                    m_offsetWindowMin = std::min(m_offsetWindowMin, offset);
                    if (++m_offsetSamples >= 240)
                    {
                        m_presentOffset = m_offsetWindowMin;
                        m_offsetWindowMin = INT64_MAX;
                        m_offsetSamples = 0;
                    }
                    if (m_presentOffset == INT64_MAX)
                        m_presentOffset = offset;
                    if (offset > m_presentOffset + period / 2)
                    {
                        // Missed its refresh. A lone miss is usually a stall elsewhere (a slow
                        // guest frame, the system); two within 60 presents mean the lead is short.
                        if (m_presentsSinceMiss < 60)
                            m_presentLead = std::min(m_presentLead + 1000000, maxLead);
                        m_presentsSinceMiss = 0;
                        m_presentsOnTime = 0;
                    }
                    else
                    {
                        m_presentsSinceMiss = std::min(m_presentsSinceMiss + 1, 1000u);
                        if (++m_presentsOnTime >= 30)
                        {
                            m_presentLead = std::max(m_presentLead - 250000, minLead);
                            m_presentsOnTime = 0;
                        }
                    }
                }
                if (std::getenv("RT_PRESENT_DEBUG"))
                {
                    static int n = 0;
                    if ((n++ % 120) == 0)
                        std::fprintf(stderr, "[present] lead %.2f ms\n", static_cast<double>(m_presentLead) / 1e6);
                }
            }

            // Draws the frame again into an image and saves it as a PNG.
            void captureLocked(float fw, float fh)
            {
                Vulkan::Device &dev = *m_shared.device;
                const uint32_t w = static_cast<uint32_t>(fw), h = static_cast<uint32_t>(fh);
                auto target = dev.create_image(Vulkan::ImageCreateInfo::render_target(w, h, VK_FORMAT_R8G8B8A8_UNORM));
                Vulkan::BufferCreateInfo binfo = {};
                binfo.size = static_cast<VkDeviceSize>(w) * h * 4u;
                binfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                binfo.domain = Vulkan::BufferDomain::CachedHost;
                auto buffer = dev.create_buffer(binfo);

                auto cmd = dev.request_command_buffer();
                Vulkan::RenderPassInfo rp = {};
                rp.num_color_attachments = 1;
                rp.color_attachments[0] = &target->get_view();
                rp.clear_attachments = 1;
                rp.store_attachments = 1;
                cmd->begin_render_pass(rp);
                drawScene(*cmd, fw, fh);
                cmd->end_render_pass();
                cmd->image_barrier(*target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                cmd->copy_image_to_buffer(*buffer, *target, 0, {}, {w, h, 1}, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
                cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                             VK_ACCESS_2_HOST_READ_BIT);
                Vulkan::Fence fence;
                dev.submit(cmd, &fence);
                fence->wait();
                std::vector<uint8_t> pixels(binfo.size);
                const void *src = dev.map_host_buffer(*buffer, Vulkan::MEMORY_ACCESS_READ_BIT);
                std::memcpy(pixels.data(), src, pixels.size());
                dev.unmap_host_buffer(*buffer, Vulkan::MEMORY_ACCESS_READ_BIT);
                // Encode the PNG off the render thread: it holds the device lock the GS thread
                // needs, and a full-window encode takes long enough to stall the game (and audio).
                std::thread([pixels = std::move(pixels), w, h, path = std::move(m_capturePath)]() mutable {
                    for (size_t p = 3; p < pixels.size(); p += 4)
                        pixels[p] = 0xFF;
                    Image image = {pixels.data(), static_cast<int>(w), static_cast<int>(h), 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
                    // Written under another name, then renamed: the file appears complete.
                    const std::string part = path + ".part.png";
                    if (ExportImage(image, part.c_str()))
                        std::rename(part.c_str(), path.c_str());
                }).detach();
                m_capturePath.clear();
            }

            PgsPresenterOptions m_options;
            SDL_Window *m_window = nullptr;
            std::unique_ptr<SdlPlatform> m_platform;
            PgsShared m_shared;
            Vulkan::WSI m_wsi; // after m_shared: destroyed first, with the device
            Vulkan::Program *m_program = nullptr;
            Vulkan::Program *m_fxaa = nullptr, *m_easu = nullptr, *m_rcas = nullptr;
            PostProcess m_post;
            Vulkan::ImageHandle m_aaImage, m_upImage; // intermediate pictures
            Vulkan::Program *m_depthView = nullptr, *m_motionView = nullptr, *m_taa = nullptr;
            Vulkan::ImageHandle m_taaHistory[2];
            uint32_t m_taaIndex = 0;
            bool m_taaValid = false;
            float m_jitter[4] = {}; // this frame's and last frame's camera jitter (GS pixels)
            Vulkan::Program *m_depthNormalize = nullptr;
            Vulkan::ImageHandle m_depthNorm;
            Vulkan::Program *m_uiComposite = nullptr;
            Vulkan::ImageHandle m_rcasImage, m_uiImage;
            bool m_temporalValid = false;
            Vulkan::ImageHandle m_taaShadowImage;
            // Interlace flicker blending.
            Vulkan::Program *m_flicker = nullptr, *m_flickerCut = nullptr, *m_flickerGrid = nullptr, *m_flickerWeight = nullptr;
            Vulkan::ImageHandle m_flickerHist[2], m_flickerImage, m_flickerOut, m_flickerCutImage, m_flickerGridImage,
                m_flickerWeightImage;
            uint32_t m_flickerIndex = 0, m_flickerValid = 0;
            // Frame generation.
            FrameGeneration m_fg;
            Vulkan::Program *m_frameGen = nullptr, *m_copy = nullptr;
            Vulkan::ImageHandle m_genHistory[2], m_genImage;
            uint32_t m_genIndex = 0, m_subframe = 0;
            bool m_genPrevValid = false, m_genRcas = false;
            float m_genJitterDelta[2] = {};
            uint64_t m_renderedTick = ~0ull;
            uint32_t m_presents = 1;
            // Presenting through Metal (macOS).
            bool m_metalPresent = false;
#if defined(__APPLE__)
            SDL_MetalView m_metalView = nullptr;
            std::unique_ptr<MetalPresent> m_metal;
#endif
            Vulkan::ImageHandle m_backbuffers[3];
            uint32_t m_backIndex = 0;
            std::mutex m_presentedMutex;
            std::vector<std::pair<int64_t, int64_t>> m_presented; // (target refresh, on screen), host ns
            bool m_presentAtTime = false;
            int64_t m_presentTarget = 0; // host ns of the refresh the last present was aimed at
            int64_t m_presentTargets[3] = {}; // the last three presents' refreshes (ring)
            int64_t m_presentLead = 0;        // how far ahead of now a present is aimed (adaptive)
            uint32_t m_presentsOnTime = 0, m_presentsSinceMiss = 1000;
            PFN_vkGetPastPresentationTimingGOOGLE m_pastTiming = nullptr;
            int64_t m_presentOffset = INT64_MAX, m_offsetWindowMin = INT64_MAX; // presented time - target refresh
            uint32_t m_offsetSamples = 0;
            uint32_t m_presentCount = 0;
            const Vulkan::Image *m_lastPicture = nullptr; // the guest frame's picture, for repeats
            bool m_lastRcas = false;
            const Vulkan::Image *m_sourceOverride = nullptr; // a shadow frame shown instead of the scanout
            bool m_noTemporal = false;                       // processing a shadow frame: no TAA/MetalFX history
            bool m_frame2D = false;
            uint64_t m_realSerial = 0; // PgsShared::presentSerial of the real frame on show
            bool m_progressiveFields = true; // GS::progressiveFields, read each frame
            bool m_showDepth = false, m_showMotion = false;
            Vulkan::Program *m_smaaEdges = nullptr, *m_smaaWeights = nullptr, *m_smaaBlend = nullptr;
            Vulkan::ImageHandle m_smaaArea, m_smaaSearch, m_smaaEdgeImage, m_smaaWeightImage;
            const Vulkan::Image *m_final = nullptr;   // what the final pass draws this frame
            bool m_finalRcas = false;
#if defined(__APPLE__)
            using GetTexture = void (*)(VkImage, void **);
            using GetQueue = void (*)(VkQueue, void **);
            GetTexture m_getTexture = nullptr;
            GetQueue m_getQueue = nullptr;
            std::unique_ptr<MetalFxSpatial> m_metalfx;
            std::unique_ptr<MetalFxTemporal> m_metalfxTemporal;
            std::unique_ptr<MetalFxTemporal> m_metalfxTemporalShadow; // frame generation's shadow frames
            Vulkan::ImageHandle m_upImageShadow;
            bool m_temporalValidShadow = false;
            void *m_mtlDevice = nullptr;
#endif
            Vulkan::ImageHandle m_cpuFrame; // CPU GS: the uploaded picture
            std::unordered_map<uint64_t, Vulkan::ImageHandle> m_textures; // by ImTextureID
            uint64_t m_nextTexture = 0;
            std::unordered_set<uint64_t> m_nearestTextures; // createUiTexture(..., nearest)
            float m_pictureAspect = 4.0f / 3.0f;
            uint64_t m_lastTick = 0;
            bool m_latched = false;
            bool m_closeRequested = false;
            bool m_background = false; // Android: no surface while another activity is in front
            bool m_ui = false;
            bool m_uiFrame = false;
            std::string m_capturePath;
        };
    }

    PgsShared *pgsShared(HostPresenter *presenter)
    {
        auto *p = dynamic_cast<PgsPresenter *>(presenter);
        return p ? p->shared() : nullptr;
    }

    std::unique_ptr<HostPresenter> createPgsPresenter(const PgsPresenterOptions &options, std::string &error)
    {
        if (!pgsAvailable())
        {
            error = "built without paraLLEl-GS";
            return nullptr;
        }
        return std::make_unique<PgsPresenter>(options);
    }
}

#elif defined(PS2X_HAVE_PGS)

#include "gs_pgs_shared.h"

namespace ps2x::gs
{
    PgsShared *pgsShared(HostPresenter *) { return nullptr; }

    std::unique_ptr<HostPresenter> createPgsPresenter(const PgsPresenterOptions &, std::string &error)
    {
        error = "the Vulkan presenter needs the SDL3 host platform";
        return nullptr;
    }
}

#endif
