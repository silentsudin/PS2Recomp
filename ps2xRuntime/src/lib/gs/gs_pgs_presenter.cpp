// The Vulkan presenter (see ps2_host_presenter.h): an SDL3 window with a Granite WSI swapchain on
// the device paraLLEl-GS renders with. Each host frame draws the newest scanout image letterboxed
// to 4:3 and the host UI (Dear ImGui, drawn here through Granite with ImGui's own shaders), then
// presents. Nothing is read back to the CPU except for test captures and screenshots.

#include "runtime/gs/gs_pgs_backend.h"
#include "runtime/ps2_host_presenter.h"

#if defined(PS2X_HAVE_PGS) && defined(PS2X_PGS_PRESENTER)

#include "gs_pgs_shared.h"
#include "imgui_spirv.h"

#include "ps2_runtime.h"
#include "runtime/ee_scheduler.h"
#include "runtime/ps2_test_harness.h"

#include "context.hpp"
#include "device.hpp"
#include "wsi.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#if defined(PS2X_ENABLE_DEBUG_UI)
#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#define PS2X_PGS_PRESENTER_UI 1
#endif

#include "raylib.h" // ExportImage for screenshots

#include <algorithm>
#include <cstring>
#include <iostream>
#include <unordered_map>
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
                latch(runtime);
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

            // The game picture, letterboxed to 4:3 (the PS2 always drives a 4:3 TV).
            void drawGame(Vulkan::CommandBuffer &cmd, float fw, float fh)
            {
                const Vulkan::ImageHandle &image = m_shared.attached ? m_shared.scanout : m_cpuFrame;
                if (!image)
                    return;
                constexpr float kAspect = 4.0f / 3.0f;
                const float w = std::min(fw, fh * kAspect), h = w / kAspect;
                const float x0 = (fw - w) * 0.5f, y0 = (fh - h) * 0.5f;
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
                for (size_t p = 3; p < pixels.size(); p += 4)
                    pixels[p] = 0xFF;
                Image image = {pixels.data(), static_cast<int>(w), static_cast<int>(h), 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
                ExportImage(image, m_capturePath.c_str());
                m_capturePath.clear();
            }

            PgsPresenterOptions m_options;
            SDL_Window *m_window = nullptr;
            std::unique_ptr<SdlPlatform> m_platform;
            PgsShared m_shared;
            Vulkan::WSI m_wsi; // after m_shared: destroyed first, with the device
            Vulkan::Program *m_program = nullptr;
            Vulkan::ImageHandle m_cpuFrame; // CPU GS: the uploaded picture
            std::unordered_map<uint64_t, Vulkan::ImageHandle> m_textures; // by ImTextureID
            uint64_t m_nextTexture = 0;
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
