// The Vulkan presenter (see ps2_host_presenter.h): an SDL3 window with a Granite WSI swapchain on
// the device paraLLEl-GS renders with. Each host frame draws the newest scanout image letterboxed
// to 4:3 and the host UI (Dear ImGui, drawn here through Granite with ImGui's own shaders), then
// presents. Nothing is read back to the CPU except for test captures and screenshots.

#include "runtime/gs/gs_pgs_backend.h"
#include "runtime/ps2_host_presenter.h"

#if defined(PS2X_HAVE_PGS) && defined(PS2X_PGS_PRESENTER)

#include "gs_pgs_shared.h"
#include "runtime/gs/gs_motion.h"
#include "imgui_spirv.h"
#if defined(__APPLE__)
#include "gs_metalfx.h"
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <thread>
#include <vector>

namespace ps2x::gs
{
    namespace
    {
        class SdlPlatform final : public Vulkan::WSIPlatform
        {
        public:
            explicit SdlPlatform(SDL_Window *window) : m_window(window) {}

            VkSurfaceKHR create_surface(VkInstance instance, VkPhysicalDevice) override
            {
                VkSurfaceKHR surface = VK_NULL_HANDLE;
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
                if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
                    return fail("SDL video", SDL_GetError());
                if (!SDL_Vulkan_LoadLibrary(m_options.vulkanLibrary.empty() ? nullptr : m_options.vulkanLibrary.c_str()))
                    return fail("SDL_Vulkan_LoadLibrary", SDL_GetError());
                m_window = SDL_CreateWindow(title, width, height,
                                            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
                if (!m_window)
                    return fail("SDL_CreateWindow", SDL_GetError());

                // Granite and SDL use the same Vulkan library (SDL's loader).
                auto getProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
                if (!getProc || !Vulkan::Context::init_loader(getProc))
                    return fail("Vulkan loader", "vkGetInstanceProcAddr unavailable");

                pgsRegisterThread();
                m_platform = std::make_unique<SdlPlatform>(m_window);
                m_wsi.set_platform(m_platform.get());
                m_wsi.set_present_mode(m_options.vsync ? Vulkan::PresentMode::SyncToVBlank
                                                       : Vulkan::PresentMode::UnlockedMaybeTear);
                m_wsi.set_backbuffer_format(Vulkan::BackbufferFormat::UNORM);
                Vulkan::Context::SystemHandles handles = {};
                // As the GS's own device: push descriptors, no descriptor buffers/heaps.
                if (!m_wsi.init_context_from_platform(1, handles, Vulkan::CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT,
                                                      Vulkan::CONTEXT_CREATION_ENABLE_DESCRIPTOR_BUFFER_BIT |
                                                          Vulkan::CONTEXT_CREATION_ENABLE_DESCRIPTOR_HEAP_BIT))
                    return fail("Vulkan", "instance/device creation failed");
                if (!m_wsi.init_device() || !m_wsi.init_surface_swapchain())
                    return fail("Vulkan", "swapchain creation failed");
                m_shared.device = &m_wsi.get_device();
                // paraLLEl-GS advances a frame context on every flush, and the swapchain on every
                // frame: with Granite's default of 2 the game thread waits for the GPU inside a
                // flush. 4, as the GS's own device had.
                m_shared.device->init_frame_contexts(4);

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
                    if (mtlDevice)
                    {
                        m_metalfx = MetalFxSpatial::create(mtlDevice);
                        m_metalfxTemporal = MetalFxTemporal::create(mtlDevice);
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
                std::cout << "[presenter] Vulkan swapchain " << m_platform->get_surface_width() << "x"
                          << m_platform->get_surface_height() << (m_options.vsync ? " (vsync)" : "") << std::endl;
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
                m_uiFrame = false;
                if (drawUi)
                    drawUi();
                render();
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
                const Vulkan::ImageHandle &image = m_shared.attached ? m_shared.scanout : m_cpuFrame;
                m_final = image.get();
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
                    temporalAntiAliasing(cmd, sw, sh);
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
                    m_temporalValid = false; // a 2D screen: start the history again afterwards
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
                if (!m_showMotion && !m_showDepth)
                {
                    m_shared.wantDepth = on;
                    m_shared.wantMotion = on;
                    MotionTracker::instance().setEnabled(on);
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
                // Any post-processing spares the UI (the GS marks where the HUD and 2D screens drew).
                m_shared.wantUi = m_post.aa != PostProcess::AntiAliasing::None || m_post.scaling != PostProcess::Scaling::Bilinear;
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
                Vulkan::Fence fence;
                dev.submit(cmd, &fence);
                fence->wait();
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
                const PassInput depthIn[1] = {{m_shared.depth.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(*cmd, m_depthNorm, sw, sh, m_depthNormalize, depthIn, 1, nullptr, 0, false, VK_FORMAT_R32_SFLOAT);
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
                Vulkan::Fence fence;
                dev.submit(cmd, &fence);
                fence->wait();
                MetalFxTemporal::Frame f = {};
                m_getTexture(m_final->get_image(), &f.color);
                m_getTexture(m_depthNorm->get_image(), &f.depth);
                m_getTexture(m_shared.motion->get_image(), &f.motion);
                m_getTexture(m_upImage->get_image(), &f.output);
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
                f.reset = !m_temporalValid;
                if (f.color && f.depth && f.motion && f.output && queue && m_metalfxTemporal->upscale(queue, f))
                {
                    m_final = m_upImage.get();
                    m_temporalValid = true;
                }
                cmd = dev.request_command_buffer();
            }
#endif

            // Puts the UI back as the game drew it over the post-processed picture: the processed
            // image (finished at the picture's size on screen, RCAS included) is mixed with the
            // original by the GS's UI mask.
            void spareUi(Vulkan::CommandBuffer &cmd, float fw, float fh)
            {
                const Vulkan::ImageHandle &original = m_shared.attached ? m_shared.scanout : m_cpuFrame;
                if (!m_final || !original || m_final == original.get() || !m_shared.ui ||
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
                                         {original.get(), Vulkan::StockSampler::LinearClamp},
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
                ImDrawData *dd = ImGui::GetDrawData();
                if (!m_uiFrame || !dd || dd->TotalVtxCount == 0)
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
                        auto it = m_textures.find(c.GetTexID());
                        if (it == m_textures.end() || !it->second)
                            continue;
                        cmd.set_scissor({{static_cast<int32_t>(x0), static_cast<int32_t>(y0)},
                                         {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}});
                        cmd.set_texture(0, 0, it->second->get_view(), Vulkan::StockSampler::LinearClamp);
                        cmd.draw_indexed(c.ElemCount, 1, idxBase + c.IdxOffset, static_cast<int32_t>(vtxBase + c.VtxOffset), 0);
                    }
                    vtxBase += static_cast<uint32_t>(list->VtxBuffer.Size);
                    idxBase += static_cast<uint32_t>(list->IdxBuffer.Size);
                }
            }
#else
            void drawUi(Vulkan::CommandBuffer &, float, float) {}
#endif

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
                if (!m_wsi.begin_frame())
                    return;
                const Vulkan::Image &back = dev.get_swapchain_view().get_image();
                const float fw = static_cast<float>(back.get_width()), fh = static_cast<float>(back.get_height());
                auto cmd = dev.request_command_buffer();
                preparePicture(cmd, fw, fh);
                spareUi(*cmd, fw, fh);
                auto rp = dev.get_swapchain_render_pass(Vulkan::SwapchainRenderPass::ColorOnly);
                rp.clear_color[0] = {};
                cmd->begin_render_pass(rp);
                drawScene(*cmd, fw, fh);
                cmd->end_render_pass();
                dev.submit(cmd);
                if (!m_capturePath.empty())
                    captureLocked(fw, fh);
                m_wsi.end_frame();
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
                    ExportImage(image, path.c_str());
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
#endif
            Vulkan::ImageHandle m_cpuFrame; // CPU GS: the uploaded picture
            std::unordered_map<uint64_t, Vulkan::ImageHandle> m_textures; // by ImTextureID
            uint64_t m_nextTexture = 0;
            float m_pictureAspect = 4.0f / 3.0f;
            uint64_t m_lastTick = 0;
            bool m_latched = false;
            bool m_closeRequested = false;
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
