#include "runtime/gs/gs_pgs_backend.h"
#include "runtime/gs/gs_display_phase.h"

#if defined(PS2X_HAVE_PGS)

#include "runtime/gs/gs_cpu_backend.h"

#include "context.hpp"
#include "device.hpp"
#include "gs_interface.hpp"
#include "thread_id.hpp"
#include "gs_pgs_shared.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>

namespace ps2x::gs
{
    namespace
    {
        // Largest picture handed to the host; bigger scanouts (high SSAA rates) are box-filtered down.
        constexpr uint32_t kMaxHostFrameWidth = 2048u;
        constexpr uint32_t kMaxHostFrameHeight = 2048u;
        constexpr uint32_t kTrxDirLocalToHost = 1u;

        // The runtime's libgraph stubs express DISPLAY.DH as the frame-buffer height (what the
        // CPU backend expects). On hardware, in interlaced field mode (SMODE2 INT=1, FFMD=1) DH
        // counts raster lines of the whole frame, i.e. twice the field buffer's height.
        uint64_t displayToHardware(uint64_t display, uint64_t smode2)
        {
            const bool interlacedFieldMode = (smode2 & 0x3u) == 0x3u;
            if (!interlacedFieldMode)
                return display;
            const uint64_t dh = (display >> 44) & 0x7FFu;
            const uint64_t hwDh = std::min<uint64_t>((dh + 1u) * 2u - 1u, 0x7FFu);
            return (display & ~(0x7FFull << 44)) | (hwDh << 44);
        }

        template <typename Bits>
        void setPrivReg(Bits &dst, uint64_t value)
        {
            // Copy the 64-bit hardware register image into paraLLEl-GS's bitfield view. Some views
            // are padded past 64 bits (SMODE2Bits is 12 bytes); the fields live in the low bytes.
            std::memset(&dst, 0, sizeof(Bits));
            std::memcpy(&dst, &value, std::min(sizeof(Bits), sizeof(value)));
        }

        bool envEquals(const char *name, const char *value)
        {
            const char *e = std::getenv(name);
            return e && std::strcmp(e, value) == 0;
        }

        ParallelGS::SuperSampling superSamplingFromEnv(uint32_t fallback)
        {
            uint32_t rate = fallback;
            if (const char *e = std::getenv("RT_GS_SSAA"))
                rate = static_cast<uint32_t>(std::strtoul(e, nullptr, 10));
            switch (rate)
            {
            case 2: return ParallelGS::SuperSampling::X2;
            case 4: return ParallelGS::SuperSampling::X4;
            case 8: return ParallelGS::SuperSampling::X8;
            case 16: return ParallelGS::SuperSampling::X16;
            default: return ParallelGS::SuperSampling::X1;
            }
        }

        ParallelGS::SuperSampling superSamplingFromCount(uint32_t rate)
        {
            switch (rate)
            {
            case 2: return ParallelGS::SuperSampling::X2;
            case 4: return ParallelGS::SuperSampling::X4;
            case 8: return ParallelGS::SuperSampling::X8;
            case 16: return ParallelGS::SuperSampling::X16;
            default: return ParallelGS::SuperSampling::X1;
            }
        }

        class PgsBackend final : public GSRasterBackend, public GSPacketMirror, public PgsControl
        {
        public:
            ~PgsBackend() override
            {
                if (!m_shared)
                    return;
                const auto lock = lockDevice();
                m_iface.flush();
                m_shared->flushLocked = nullptr;
                m_shared->attached = false;
                m_shared->scanout.reset();
            }

            bool init(const PgsOptions &options, std::string &error)
            {
                if (PgsShared *shared = pgsShared(options.presenter))
                {
                    // The presenter's device: draw into its swapchain, no readback per frame.
                    m_shared = shared;
                    m_dev = shared->device;
                    m_mtx = &shared->mutex;
                    shared->attached = true;
                    shared->flushLocked = [this] { m_iface.flush(); };
                }
                else
                {
                    if (!options.vulkanLibrary.empty())
                        setenv("GRANITE_VULKAN_LIBRARY", options.vulkanLibrary.c_str(), 1);

                    if (!Vulkan::Context::init_loader(nullptr))
                    {
                        error = "could not load a Vulkan library (MoltenVK)";
                        return false;
                    }

                    m_context.set_num_thread_indices(1);
                    if (!m_context.init_instance_and_device(nullptr, 0, nullptr, 0,
                                                            Vulkan::CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT))
                    {
                        error = "Vulkan instance/device creation failed";
                        return false;
                    }

                    m_dev->set_context(m_context);
                    m_dev->init_frame_contexts(4);
                }
                const auto initLock = lockDevice();

                ParallelGS::GSOptions gsOptions = {};
                // The game draws 224-line fields. Super-sampling plus a high-resolution progressive
                // scanout (see present()) turns them into a full 448-line picture with no
                // interlacing artefacts. RT_GS_SSAA=1|2|4|8|16 (default 4); RT_GS_PROGRESSIVE=0
                // keeps the plain field deinterlacer.
                m_progressive = !envEquals("RT_GS_PROGRESSIVE", "0");
                gsOptions.super_sampling = superSamplingFromEnv(m_progressive ? 4u : 1u);
                if (!m_iface.init(m_dev, gsOptions))
                {
                    error = "paraLLEl-GS init failed (missing Vulkan features?)";
                    return false;
                }

                const auto &props = m_dev->get_gpu_properties();
                m_info.name = props.deviceName;
                m_info.vendorId = props.vendorID;
                m_info.deviceId = props.deviceID;
                m_info.apiVersion = props.apiVersion;
                // The interface clamps the rate to what the device supports (8x and 16x need
                // compute subgroup size control for 8/16-wide groups).
                m_info.maxSuperSampling = static_cast<uint32_t>(m_iface.get_max_supported_super_sampling());
                m_samples = std::min(static_cast<uint32_t>(gsOptions.super_sampling), m_info.maxSuperSampling);
                std::cout << "[gs] paraLLEl-GS on " << props.deviceName << ", super-sampling " << m_samples << "x (max "
                          << m_info.maxSuperSampling << "x)" << std::endl;
                return true;
            }

            // ---------------------------------------------------------------- PgsControl
            PgsDeviceInfo deviceInfo() const override { return m_info; }

            void setSuperSampling(uint32_t samples) override
            {
                const auto lock = lockDevice();
                const ParallelGS::SuperSampling rate = superSamplingFromCount(samples);
                if (static_cast<uint32_t>(rate) == m_samples)
                    return;
                m_iface.set_super_sampling_rate(rate, true, false);
                m_samples = std::min(static_cast<uint32_t>(rate), m_info.maxSuperSampling);
            }

            uint32_t superSampling() const override { return m_samples; }

            void setSharpTextures(bool on) override
            {
                const auto lock = lockDevice();
                m_hacks.disable_mipmaps = on;
                m_iface.set_hacks(m_hacks);
            }

            bool WantsVertexSideband() const override
            {
                return m_shared && (m_shared->wantMotion.load(std::memory_order_relaxed) ||
                                    m_shared->wantUi.load(std::memory_order_relaxed));
            }

            bool WantsDepthSnapshot() const override
            {
                return m_shared && m_shared->wantDepth.load(std::memory_order_relaxed);
            }

            void SnapshotDepth(uint32_t zbp, uint32_t fbw) override
            {
                const auto lock = lockDevice();
                m_iface.flush();
                m_iface.snapshot_depth(zbp, fbw, 512);
            }

            // ---------------------------------------------------------------- GSPacketMirror
            void MirrorGifPacketWithMotion(uint32_t pathIndex, const uint8_t *data, uint32_t sizeBytes, const uint32_t *motion,
                                           uint32_t motionCount) override
            {
                if (!data || sizeBytes < 16u || pathIndex > 3u)
                    return;
                const auto lock = lockDevice();
                const bool wanted = WantsVertexSideband();
                if (wanted != m_motionEnabled)
                {
                    m_iface.set_motion_enabled(wanted);
                    m_motionEnabled = wanted;
                }
                if (wanted)
                    m_iface.set_vertex_motion(motion, motionCount);
                m_iface.gif_transfer(pathIndex, data, sizeBytes);
                m_iface.set_vertex_motion(nullptr, 0);
            }

            void MirrorGifPacket(uint32_t pathIndex, const uint8_t *data, uint32_t sizeBytes) override
            {
                if (!data || sizeBytes < 16u || pathIndex > 3u)
                    return;
                const auto lock = lockDevice();
                m_iface.gif_transfer(pathIndex, data, sizeBytes);
                m_gpuVramNewer = true;
            }

            void MirrorRegisterWrite(uint8_t regAddr, uint64_t value) override
            {
                const auto lock = lockDevice();
                m_iface.write_register(static_cast<ParallelGS::RegisterAddr>(regAddr), value);
                m_gpuVramNewer = true;
            }

            // ---------------------------------------------------------------- GSRasterBackend
            bool WantsPrimitives() const override { return false; }

            void Initialize(uint8_t *vram, uint32_t vramSize) override
            {
                m_vram = vram;
                m_vramSize = vramSize;
                m_shadow.Initialize(vram, vramSize);
                // Seed the GPU copy with whatever the previous backend left in local memory.
                const auto lock = lockDevice();
                uploadWholeVramLocked();
            }

            void Reset() override
            {
                const auto lock = lockDevice();
                m_iface.reset_context_state();
                m_shadow.Reset();
            }

            // Rendering, CLUT loads and uploads all reach the GPU through the mirrored stream.
            void Submit(const GSPrimitiveBatch &) override {}
            void LoadClut(const GSTex0Reg &, const GSTexClutReg &) override {}
            void UploadImage(const uint8_t *, uint32_t) override {}
            void Flush() override {}
            void TextureFlush() override {}
            void Sync(GSSyncReason) override {}

            void BeginTransfer(const GSTransferCommand &command) override
            {
                // Local->host readbacks are served by the CPU shadow from fresh GPU VRAM.
                if (command.direction == kTrxDirLocalToHost)
                    pullVramFromGpu();
                m_shadow.BeginTransfer(command);
            }

            bool ClearFramebuffer(const GSContext &, uint32_t) override
            {
                // The swap stub follows this fast path with the real clear sprite, which reaches
                // the GPU as mirrored register writes.
                return true;
            }

            uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override
            {
                return m_shadow.ConsumeLocalToHostBytes(dst, maxBytes);
            }

            uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override
            {
                const_cast<PgsBackend *>(this)->pullVramFromGpu();
                return m_shadow.ReadVram(psm, base, bw, x, y);
            }

            void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override
            {
                pullVramFromGpu();
                m_shadow.WriteVram(psm, base, bw, x, y, value);
                const auto lock = lockDevice();
                uploadWholeVramLocked();
            }

            void SnapshotVram(std::vector<uint8_t> &out) const override
            {
                const_cast<PgsBackend *>(this)->pullVramFromGpu();
                m_shadow.SnapshotVram(out);
            }

            GSTransferSnapshot GetTransferSnapshot() const override { return m_shadow.GetTransferSnapshot(); }

            PresentationFrame Present(const GSPresentationRequest &request) override
            {
                // Record and submit under the lock, but wait for the GPU outside it: the game
                // thread keeps streaming GIF packets while this frame finishes rendering.
                Vulkan::Fence fence;
                uint32_t w = 0, h = 0;
                {
                    const auto lock = lockDevice();

                    auto &priv = m_iface.get_priv_register_state();
                    setPrivReg(priv.pmode, request.pmode);
                    // Progressive fields: the game no longer offsets every other field, so the
                    // field is scanned out as a 224-line progressive (double-strike) picture: no
                    // deinterlacing, no field-aware line shift, the same image every vblank.
                    const uint64_t smode2 = request.progressiveFields ? (request.smode2 & ~0x3ull) : request.smode2;
                    setPrivReg(priv.smode2, smode2);
                    setPrivReg(priv.dispfb1, request.dispfb1);
                    setPrivReg(priv.display1, displayToHardware(request.display1, smode2));
                    setPrivReg(priv.dispfb2, request.dispfb2);
                    setPrivReg(priv.display2, displayToHardware(request.display2, smode2));
                    setPrivReg(priv.bgcolor, request.bgcolor);
                    priv.smode1.CMOD = 2; // NTSC
                    priv.smode1.LC = 32;

                    ParallelGS::VSyncInfo vsync = {};
                    // Field phase of the buffer on display, from the guest's vblank timeline
                    // (see gs_display_phase.h).
                    vsync.phase = g_displayFieldPhase.load(std::memory_order_relaxed);
                    const bool onGpu = request.keepOnGpu && m_shared;
                    if (onGpu)
                    {
                        // Sampled by the presenter; a readback transitions it and back.
                        vsync.dst_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        vsync.dst_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                        vsync.dst_access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    }
                    else
                    {
                        vsync.dst_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        vsync.dst_stage = VK_PIPELINE_STAGE_2_COPY_BIT;
                        vsync.dst_access = VK_ACCESS_2_TRANSFER_READ_BIT;
                    }
                    vsync.adapt_to_internal_horizontal_resolution = true;
                    // Road Trip renders 640x224 fields (INT=1, FFMD=1). Progressive mode scans out
                    // the super-sampled field at double height instead of bobbing between fields.
                    if (m_progressive)
                    {
                        vsync.force_progressive = true;
                        vsync.anti_blur = true;
                        // Needs at least 4 samples per pixel (paraLLEl-GS scans out 2x2 of them).
                        vsync.high_resolution_scanout = m_samples >= 4u;
                        // Progressive fields: twice as many lines as columns from the samples
                        // (2x/4x: 640x448, 8x/16x: 1280x896), so the picture keeps its detail
                        // vertically too.
                        if (request.progressiveFields)
                        {
                            vsync.progressive_field_scanout = true;
                            vsync.high_resolution_scanout = m_samples >= 2u;
                        }
                    }

                    if (onGpu && request.depthValid && m_shared->wantDepth.load(std::memory_order_relaxed))
                    {
                        vsync.scanout_depth = true;
                        vsync.depth_zbp = request.depthZbp;
                        vsync.depth_psm = 0x30u | (request.depthPsm & 0xFu); // PSMZ32/24/16/16S
                        vsync.scanout_motion = m_shared->wantMotion.load(std::memory_order_relaxed);
                        static int dbg = 0;
                        if (std::getenv("RT_SHOW_DEPTH") && (dbg++ % 120) == 0)
                            std::fprintf(stderr, "[depth] zbp=%u psm=0x%x dispfb fbp=%u fbw=%u\n", request.depthZbp, vsync.depth_psm,
                                         unsigned(request.dispfb1 & 0x1FF), unsigned((request.dispfb1 >> 9) & 0x3F));
                    }
                    if (onGpu && m_motionEnabled && m_shared->wantUi.load(std::memory_order_relaxed))
                        vsync.scanout_ui = true;
                    m_iface.flush();
                    ParallelGS::ScanoutResult scanout = m_iface.vsync(vsync);
                    if (onGpu)
                    {
                        m_shared->depth = scanout.depth;
                        m_shared->motion = scanout.motion;
                        m_shared->ui = scanout.ui;
                    }
                    if (scanout.image)
                    {
                        w = scanout.image->get_width();
                        h = scanout.image->get_height();
                        if (onGpu)
                            m_shared->scanout = scanout.image;
                        if (!onGpu || request.readback)
                            fence = submitReadbackLocked(*scanout.image, onGpu);
                    }
                    // With a presenter, its swapchain frames advance the frame contexts.
                    if (!m_shared)
                        m_dev->next_frame_context();
                }
                if (onGpuOnly(request))
                {
                    PresentationFrame frame{};
                    frame.width = w;
                    frame.height = h;
                    return frame;
                }

                PresentationFrame frame{};
                if (!fence)
                    return frame;
                fence->wait();
                const auto lock = lockDevice();
                return copyReadbackLocked(w, h);
            }

        private:
            bool m_progressive = true;
            std::atomic<uint32_t> m_samples{1};
            ParallelGS::Hacks m_hacks = {}; // every hack set so far (set_hacks replaces them all)
            bool m_motionEnabled = false;

            // The device has one thread index (set_num_thread_indices(1)) and every use holds
            // m_mutex, so each calling thread (EE, VU1, render) uses index 0. Registering it stops
            // Granite logging an error per call from an unregistered thread.
            std::unique_lock<std::mutex> lockDevice() const
            {
                pgsRegisterThread();
                return std::unique_lock<std::mutex>(*m_mtx);
            }
            PgsDeviceInfo m_info;
            bool onGpuOnly(const GSPresentationRequest &request) const { return request.keepOnGpu && m_shared && !request.readback; }

            Vulkan::Fence submitReadbackLocked(const Vulkan::Image &image, bool sampledLayout)
            {
                const uint32_t w = image.get_width();
                const uint32_t h = image.get_height();

                Vulkan::BufferCreateInfo info = {};
                info.size = static_cast<VkDeviceSize>(w) * h * 4u;
                info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                info.domain = Vulkan::BufferDomain::CachedHost;
                if (!m_readback || m_readback->get_create_info().size < info.size)
                    m_readback = m_dev->create_buffer(info);

                auto cmd = m_dev->request_command_buffer();
                if (sampledLayout)
                    cmd->image_barrier(image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT,
                                       VK_ACCESS_2_TRANSFER_READ_BIT);
                cmd->copy_image_to_buffer(*m_readback, image, 0, {}, {w, h, 1}, 0, 0,
                                          {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
                if (sampledLayout)
                    cmd->image_barrier(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
                Vulkan::Fence fence;
                m_dev->submit(cmd, &fence);
                return fence;
            }

            PresentationFrame copyReadbackLocked(uint32_t w, uint32_t h)
            {
                const auto *src = static_cast<const uint8_t *>(
                    m_dev->map_host_buffer(*m_readback, Vulkan::MEMORY_ACCESS_READ_BIT));

                // The picture goes to the host at its native size (the high-resolution scanout is
                // 1280x896 at 4x SSAA); only very large ones are box-filtered down.
                const uint32_t fx = (w + kMaxHostFrameWidth - 1u) / kMaxHostFrameWidth;
                const uint32_t fy = (h + kMaxHostFrameHeight - 1u) / kMaxHostFrameHeight;
                PresentationFrame frame{};
                frame.width = w / fx;
                frame.height = h / fy;
                frame.stride = frame.width;
                frame.pixels.resize(static_cast<size_t>(frame.width) * frame.height * 4u);
                const uint32_t area = fx * fy;
                for (uint32_t y = 0; y < frame.height; ++y)
                {
                    uint8_t *dst = frame.pixels.data() + static_cast<size_t>(y) * frame.width * 4u;
                    if (area == 1u)
                    {
                        std::memcpy(dst, src + static_cast<size_t>(y) * w * 4u, static_cast<size_t>(frame.width) * 4u);
                        for (uint32_t x = 0; x < frame.width; ++x)
                            dst[x * 4u + 3u] = 0xFF;
                        continue;
                    }
                    for (uint32_t x = 0; x < frame.width; ++x)
                    {
                        uint32_t sum[3] = {};
                        for (uint32_t sy = 0; sy < fy; ++sy)
                        {
                            const uint8_t *p = src + (static_cast<size_t>(y * fy + sy) * w + x * fx) * 4u;
                            for (uint32_t sx = 0; sx < fx; ++sx, p += 4)
                            {
                                sum[0] += p[0];
                                sum[1] += p[1];
                                sum[2] += p[2];
                            }
                        }
                        dst[x * 4u + 0u] = static_cast<uint8_t>(sum[0] / area);
                        dst[x * 4u + 1u] = static_cast<uint8_t>(sum[1] / area);
                        dst[x * 4u + 2u] = static_cast<uint8_t>(sum[2] / area);
                        dst[x * 4u + 3u] = 0xFF;
                    }
                }
                m_dev->unmap_host_buffer(*m_readback, Vulkan::MEMORY_ACCESS_READ_BIT);
                return frame;
            }

            void pullVramFromGpu()
            {
                const auto lock = lockDevice();
                if (!m_gpuVramNewer || !m_vram)
                    return;
                m_iface.flush();
                if (const void *gpu = m_iface.map_vram_read(0, m_vramSize))
                    std::memcpy(m_vram, gpu, m_vramSize);
                m_gpuVramNewer = false;
            }

            void uploadWholeVramLocked()
            {
                if (!m_vram || m_vramSize == 0u)
                    return;
                if (void *gpu = m_iface.map_vram_write(0, m_vramSize))
                {
                    std::memcpy(gpu, m_vram, m_vramSize);
                    m_iface.end_vram_write(0, m_vramSize);
                }
                m_gpuVramNewer = false;
            }

            Vulkan::Context m_context; // own device (no presenter) ...
            Vulkan::Device m_device;
            Vulkan::Device *m_dev = &m_device; // ... or the presenter's
            PgsShared *m_shared = nullptr;
            ParallelGS::GSInterface m_iface;
            Vulkan::BufferHandle m_readback;

            GSCpuBackend m_shadow; // transfer/readback logic over the shared local memory
            uint8_t *m_vram = nullptr;
            uint32_t m_vramSize = 0u;
            bool m_gpuVramNewer = false;
            mutable std::mutex m_mutex;
            std::mutex *m_mtx = &m_mutex;
        };
    }

    bool pgsAvailable() { return true; }

    void pgsRegisterThread()
    {
        static thread_local bool registered = false;
        if (!registered)
        {
            Util::register_thread_index(0);
            registered = true;
        }
    }

    std::unique_ptr<GSRasterBackend> createPgsBackend(const PgsOptions &options, std::string &error,
                                                      PgsControl **control)
    {
        auto backend = std::make_unique<PgsBackend>();
        if (!backend->init(options, error))
            return nullptr;
        if (control)
            *control = backend.get();
        return backend;
    }
}

#else

namespace ps2x::gs
{
    bool pgsAvailable() { return false; }

    std::unique_ptr<HostPresenter> createPgsPresenter(const PgsPresenterOptions &, std::string &error)
    {
        error = "built without paraLLEl-GS (PS2X_ENABLE_PGS=OFF)";
        return nullptr;
    }

    std::unique_ptr<GSRasterBackend> createPgsBackend(const PgsOptions &, std::string &error, PgsControl **)
    {
        error = "built without paraLLEl-GS (PS2X_ENABLE_PGS=OFF)";
        return nullptr;
    }
}

#endif
