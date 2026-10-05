#include "runtime/gs/gs_pgs_backend.h"
#include "runtime/gs/gs_display_phase.h"

#if defined(PS2X_HAVE_PGS)

#include "runtime/gs/gs_cpu_backend.h"

#include "context.hpp"
#include "device.hpp"
#include "gs_interface.hpp"
#include "thread_id.hpp"
#include "gs_pgs_shared.h"
#include "ThreadNaming.h"
#include "gs_texture_tools.h"

#include <algorithm>
#include <deque>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

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
                {
                    std::lock_guard<std::mutex> jobs(m_shadowJobMutex);
                    m_shadowStop = true;
                }
                m_shadowJobCv.notify_all();
                if (m_shadowWorker.joinable())
                    m_shadowWorker.join();
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
                    // Before each presenter frame (which advances the device's frame context and so
                    // waits for every open command buffer): the real GS and its shadows.
                    shared->flushLocked = [this] {
                        m_iface.flush();
                        for (auto &shadow : m_shadowIf)
                            if (shadow)
                                shadow->flush();
                    };
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
                for (auto &shadow : m_shadowIf)
                    if (shadow)
                        shadow->set_super_sampling_rate(rate, true, false);
                m_samples = std::min(static_cast<uint32_t>(rate), m_info.maxSuperSampling);
            }

            uint32_t superSampling() const override { return m_samples; }

            void setTextures(const std::string &dumpDir, const std::string &packDir) override
            {
                m_textures.configure(dumpDir, packDir);
                const auto lock = lockDevice();
                m_iface.set_texture_readback(m_textures.active());
                if (packDir != m_packDir)
                {
                    // Another pack (or none): the old one's images must not stay bound.
                    m_packDir = packDir;
                    m_iface.drop_texture_replacements();
                    for (auto &shadow : m_shadowIf)
                        if (shadow)
                            shadow->drop_texture_replacements();
                    m_hdImages.clear();
                    m_hdByImage.clear();
                    m_pendingImages.clear();
                    m_pendingOrder.clear();
                    m_shadowReplacements.clear();
                    m_hdBytes = 0;
                    m_predicted.clear();
                    m_replaced.clear();
                }
            }

            TexturePackStats texturePackStats() const override
            {
                const auto lock = lockDevice();
                return {m_textures.packSize(), m_replaced.size()};
            }

            void setAnisotropy(uint32_t level) override
            {
                const auto lock = lockDevice();
                m_iface.set_anisotropy(level);
                for (auto &shadow : m_shadowIf)
                    if (shadow)
                        shadow->set_anisotropy(level);
            }

            void setSharpTextures(bool on) override
            {
                const auto lock = lockDevice();
                m_hacks.disable_mipmaps = on;
                m_iface.set_hacks(m_hacks);
                for (auto &shadow : m_shadowIf)
                    if (shadow)
                        shadow->set_hacks(m_hacks);
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
                recordShadowGifLocked(pathIndex, data, sizeBytes);
                textureToolsIdleLocked();
            }

            void MirrorGifPacket(uint32_t pathIndex, const uint8_t *data, uint32_t sizeBytes) override
            {
                if (!data || sizeBytes < 16u || pathIndex > 3u)
                    return;
                const auto lock = lockDevice();
                m_iface.gif_transfer(pathIndex, data, sizeBytes);
                m_gpuVramNewer = true;
                recordShadowGifLocked(pathIndex, data, sizeBytes);
                textureToolsIdleLocked();
            }

            uint32_t ShadowFrames() const override { return m_shadowCount; }

            void SetShadowVariant(uint32_t index, const uint8_t *data, uint32_t sizeBytes) override
            {
                if (index < kMaxShadows)
                {
                    m_variant[index] = data;
                    m_variantSize[index] = sizeBytes;
                }
            }

            void MirrorRegisterWrite(uint8_t regAddr, uint64_t value) override
            {
                const auto lock = lockDevice();
                m_iface.write_register(static_cast<ParallelGS::RegisterAddr>(regAddr), value);
                m_gpuVramNewer = true;
                for (uint32_t i = 0; i < m_shadowCount; ++i)
                {
                    auto &out = m_shadowStream[i];
                    out.push_back(1); // register write
                    out.push_back(regAddr);
                    const size_t at = out.size();
                    out.resize(at + 8);
                    std::memcpy(out.data() + at, &value, 8);
                }
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
                        // After the real frame's work: its shadows (re-rendered frames).
                        presentShadowsLocked(priv, vsync);
                    }
                    textureToolsLocked();
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

            // ---------------------------------------------------------------- shadow frames
            // Re-rendered frame generation. Everything mirrored to the GS between two vsyncs is
            // recorded per shadow (PATH1 3D packets in their re-projected version) and replayed
            // into that shadow's own paraLLEl-GS after the real frame's vsync, so the shadow work
            // queues behind the real frame instead of delaying it.
            static constexpr uint32_t kMaxShadows = 3;

            void recordShadowGifLocked(uint32_t pathIndex, const uint8_t *data, uint32_t sizeBytes)
            {
                for (uint32_t i = 0; i < m_shadowCount; ++i)
                {
                    const uint8_t *src = m_variant[i] ? m_variant[i] : data;
                    const uint32_t size = m_variant[i] ? m_variantSize[i] : sizeBytes;
                    auto &out = m_shadowStream[i];
                    out.push_back(0); // GIF packet
                    out.push_back(static_cast<uint8_t>(pathIndex));
                    const size_t at = out.size();
                    out.resize(at + 4 + size);
                    std::memcpy(out.data() + at, &size, 4);
                    std::memcpy(out.data() + at + 4, src, size);
                }
                for (auto &v : m_variant)
                    v = nullptr;
            }

            void presentShadowsLocked(const ParallelGS::PrivRegisterState &priv, const ParallelGS::VSyncInfo &realVsync)
            {
                ++m_shared->presentSerial;
                const uint32_t want = std::min(m_shared->wantShadows.load(std::memory_order_relaxed), kMaxShadows);
                if (want != m_shadowCount)
                {
                    // Start (or stop) at a frame boundary; a new shadow begins with the real VRAM.
                    {
                        std::lock_guard<std::mutex> jobs(m_shadowJobMutex);
                        ++m_shadowGeneration; // the worker drops what it was replaying
                        for (auto &job : m_shadowJobs)
                        {
                            job.stream.clear();
                            job.pending = false;
                        }
                    }
                    for (uint32_t i = 0; i < kMaxShadows; ++i)
                    {
                        m_shadowStream[i].clear();
                        m_shared->shadowScanout[i].reset();
                        m_shared->shadowSerial[i] = 0;
                        if (i >= want)
                            m_shadowIf[i].reset();
                        else if (!m_shadowIf[i])
                        {
                            auto shadow = std::make_unique<ParallelGS::GSInterface>();
                            ParallelGS::GSOptions options = {};
                            options.super_sampling = superSamplingFromCount(m_samples);
                            if (!shadow->init(m_dev, options))
                                break;
                            shadow->set_owns_frame_contexts(false);
                            shadow->set_hacks(m_hacks);
                            m_iface.flush();
                            const void *real = m_iface.map_vram_read(0, m_vramSize);
                            void *dst = shadow->map_vram_write(0, m_vramSize);
                            if (real && dst)
                            {
                                std::memcpy(dst, real, m_vramSize);
                                shadow->end_vram_write(0, m_vramSize);
                            }
                            m_shadowIf[i] = std::move(shadow);
                        }
                    }
                    m_shadowCount = want;
                    if (want && !m_shadowWorker.joinable())
                        m_shadowWorker = std::thread([this] { shadowWorker(); });
                    return;
                }
                if (!m_shadowCount)
                    return;
                // Hand this frame's recording to the worker (appended if it is still behind: a
                // shadow must see every packet the real GS saw).
                {
                    std::lock_guard<std::mutex> jobs(m_shadowJobMutex);
                    for (uint32_t i = 0; i < m_shadowCount; ++i)
                    {
                        ShadowJob &job = m_shadowJobs[i];
                        job.stream.insert(job.stream.end(), m_shadowStream[i].begin(), m_shadowStream[i].end());
                        m_shadowStream[i].clear();
                        job.priv = priv;
                        job.vsync = realVsync;
                        job.vsync.scanout_depth = job.vsync.scanout_motion = job.vsync.scanout_ui = false;
                        job.serial = m_shared->presentSerial;
                        job.pending = true;
                    }
                }
                m_shadowJobCv.notify_all();
            }

            // Replays shadow frames off the presenter thread, one packet per hold of the device
            // lock, so the game's own GS work is never held up for long.
            void shadowWorker()
            {
                ThreadNaming::SetCurrentThreadName("GsShadow");
                std::vector<uint8_t> stream;
                for (;;)
                {
                    uint32_t index = kMaxShadows, generation = 0;
                    ParallelGS::PrivRegisterState priv;
                    ParallelGS::VSyncInfo vsync;
                    uint64_t serial = 0;
                    {
                        std::unique_lock<std::mutex> jobs(m_shadowJobMutex);
                        m_shadowJobCv.wait(jobs, [&] {
                            if (m_shadowStop)
                                return true;
                            for (const auto &job : m_shadowJobs)
                                if (job.pending)
                                    return true;
                            return false;
                        });
                        if (m_shadowStop)
                            return;
                        for (uint32_t i = 0; i < kMaxShadows; ++i)
                            if (m_shadowJobs[i].pending)
                            {
                                index = i;
                                break;
                            }
                        ShadowJob &job = m_shadowJobs[index];
                        stream.swap(job.stream);
                        job.stream.clear();
                        job.pending = false;
                        priv = job.priv;
                        vsync = job.vsync;
                        serial = job.serial;
                        generation = m_shadowGeneration;
                    }
                    auto current = [&] {
                        std::lock_guard<std::mutex> jobs(m_shadowJobMutex);
                        return generation == m_shadowGeneration;
                    };
                    // In slices of at most half a millisecond under the lock: per packet, the lock
                    // changed hands hundreds of times a frame and the handovers held up the game.
                    size_t at = 0;
                    bool dropped = false;
                    while (at + 2 <= stream.size() && !dropped)
                    {
                        const auto lock = lockDevice();
                        if (!current() || !m_shadowIf[index])
                        {
                            dropped = true;
                            break;
                        }
                        ParallelGS::GSInterface &shadow = *m_shadowIf[index];
                        const auto sliceEnd = std::chrono::steady_clock::now() + std::chrono::microseconds(500);
                        while (at + 2 <= stream.size())
                        {
                            const uint8_t kind = stream[at], arg = stream[at + 1];
                            at += 2;
                            if (kind == 0)
                            {
                                uint32_t size = 0;
                                std::memcpy(&size, stream.data() + at, 4);
                                shadow.gif_transfer(arg, stream.data() + at + 4, size);
                                at += 4 + size;
                            }
                            else
                            {
                                uint64_t value = 0;
                                std::memcpy(&value, stream.data() + at, 8);
                                shadow.write_register(static_cast<ParallelGS::RegisterAddr>(arg), value);
                                at += 8;
                            }
                            if (std::chrono::steady_clock::now() >= sliceEnd)
                                break;
                        }
                        // Submit before letting go of the lock: the real GS's flushes advance the
                        // device's frame context, which waits for every open command buffer (a
                        // deadlock if this one stayed open while we wait for the lock).
                        shadow.flush();
                    }
                    stream.clear();
                    if (dropped)
                        continue;
                    const auto lock = lockDevice();
                    if (!current() || !m_shadowIf[index])
                        continue;
                    ParallelGS::GSInterface &shadow = *m_shadowIf[index];
                    shadow.get_priv_register_state() = priv;
                    shadow.flush();
                    m_shared->shadowScanout[index] = shadow.vsync(vsync).image;
                    m_shared->shadowSerial[index] = serial;
                }
            }

            // Texture dumps and packs: hand finished readbacks to the worker; apply replacements.
            // Without presentation (headless), vsync runs only when a picture is asked for: the
            // texture tools are then serviced from the GIF path once no vsync has for 50 ms.
            void textureToolsIdleLocked()
            {
                if (!m_textures.active())
                    return;
                const auto now = std::chrono::steady_clock::now();
                if (now - m_texturesServiced < std::chrono::milliseconds(50))
                    return;
                m_iface.flush();
                m_iface.close_texture_readbacks();
                textureToolsLocked();
            }

            struct HdImage
            {
                Vulkan::ImageHandle image;
                size_t bytes = 0;
                uint64_t lastUsed = 0;
                std::vector<uint64_t> stableKeys; // predictions binding it
            };
            struct PendingBind
            {
                uint64_t stableKey, cacheKey;
                bool recolor;
                float transform[20];
            };
            struct PendingImage
            {
                uint32_t width = 0, height = 0;
                std::vector<uint8_t> rgba;
                std::vector<PendingBind> binds;
                bool queued = false;
            };
            static constexpr size_t kCreateBytesPerFrame = 24u << 20;

            void predictLocked(uint64_t stableKey, Vulkan::ImageHandle image, const float *recolor = nullptr)
            {
                m_iface.set_texture_prediction(stableKey, image, recolor);
                for (auto &shadow : m_shadowIf)
                    if (shadow)
                        shadow->set_texture_prediction(stableKey, image, recolor);
            }

            // recolor: another palette of the texture (TextureTools::Replacement::transform), or null.
            void bindLocked(HdImage &hd, uint64_t contentKey, uint64_t stableKey, uint64_t cacheKey, const float *recolor)
            {
                // From now on every decode of this description binds the replacement (textures
                // whose palette is reloaded every frame get a new cache key every frame).
                hd.lastUsed = m_textureFrame;
                m_recolored += recolor != nullptr;
                m_replaced.insert(contentKey);
                m_iface.replace_cached_texture(cacheKey, hd.image);
                predictLocked(stableKey, hd.image, recolor);
                if (m_predicted.insert(stableKey).second)
                    hd.stableKeys.push_back(stableKey);
                m_shadowReplacements[cacheKey] = {hd.image, 120u};
            }

            void evictLocked()
            {
                std::vector<std::pair<uint64_t, uint64_t>> byAge; // last used, content key
                for (auto &[key, hd] : m_hdImages)
                    if (m_textureFrame - hd.lastUsed > 120) // not drawn for two seconds
                        byAge.emplace_back(hd.lastUsed, key);
                std::sort(byAge.begin(), byAge.end());
                const size_t target = m_hdBudget / 5 * 4;
                size_t dropped = 0;
                for (auto [age, key] : byAge)
                {
                    if (m_hdBytes <= target)
                        break;
                    auto it = m_hdImages.find(key);
                    for (uint64_t stable : it->second.stableKeys)
                        if (m_predicted.erase(stable))
                            predictLocked(stable, {});
                    for (auto sit = m_shadowReplacements.begin(); sit != m_shadowReplacements.end();)
                        sit = sit->second.first == it->second.image ? m_shadowReplacements.erase(sit) : std::next(sit);
                    m_hdByImage.erase(it->second.image.get());
                    m_hdBytes -= it->second.bytes;
                    m_replaced.erase(key);
                    m_textures.forget(key);
                    m_hdImages.erase(it);
                    ++dropped;
                }
                if (dropped)
                {
                    // Cached textures still hold the dropped images: decode everything again
                    // (predictions rebind the images that stay at once).
                    m_iface.invalidate_texture_cache();
                    for (auto &shadow : m_shadowIf)
                        if (shadow)
                            shadow->invalidate_texture_cache();
                    m_evicted += dropped;
                }
            }

            void textureToolsLocked()
            {
                if (!m_textures.active())
                    return;
                m_texturesServiced = std::chrono::steady_clock::now();
                m_readbacks.clear();
                m_iface.collect_texture_readbacks(m_readbacks);
                for (auto &rb : m_readbacks)
                {
                    const size_t size = size_t(rb.width) * rb.height * 4;
                    const auto *src = static_cast<const uint8_t *>(m_dev->map_host_buffer(*rb.buffer, Vulkan::MEMORY_ACCESS_READ_BIT));
                    if (!src)
                        continue;
                    std::vector<uint8_t> rgba(src, src + size);
                    m_dev->unmap_host_buffer(*rb.buffer, Vulkan::MEMORY_ACCESS_READ_BIT);
                    m_textures.submit(rb.hash, rb.stable, rb.width, rb.height, rb.psm, std::move(rgba));
                }
                const auto started = std::chrono::steady_clock::now();
                ++m_textureFrame;
                m_replacements.clear();
                m_textures.collect(m_replacements);
                for (auto &r : m_replacements)
                {
                    if (!r.hit)
                    {
                        // This description now decodes to something else: stop predicting it.
                        if (m_predicted.erase(r.stableKey))
                            predictLocked(r.stableKey, {});
                        continue;
                    }
                    auto it = m_hdImages.find(r.contentKey);
                    if (it != m_hdImages.end())
                    {
                        bindLocked(it->second, r.contentKey, r.stableKey, r.cacheKey, r.recolor ? r.transform : nullptr);
                        continue;
                    }
                    // Not an image yet: created below, within a per-frame budget (a 4x pack has
                    // images of 50 MB with mips; creating several at once would stall a frame).
                    auto &pending = m_pendingImages[r.contentKey];
                    if (!r.rgba.empty())
                    {
                        pending.width = r.width;
                        pending.height = r.height;
                        pending.rgba = std::move(r.rgba);
                        if (!pending.queued)
                        {
                            pending.queued = true;
                            m_pendingOrder.push_back(r.contentKey);
                        }
                    }
                    PendingBind bind{r.stableKey, r.cacheKey, r.recolor, {}};
                    if (r.recolor)
                        std::memcpy(bind.transform, r.transform, sizeof(bind.transform));
                    pending.binds.push_back(bind);
                }
                size_t created = 0;
                while (!m_pendingOrder.empty() && created < kCreateBytesPerFrame)
                {
                    const uint64_t key = m_pendingOrder.front();
                    m_pendingOrder.pop_front();
                    auto pit = m_pendingImages.find(key);
                    if (pit == m_pendingImages.end() || pit->second.rgba.empty())
                        continue;
                    PendingImage pending = std::move(pit->second);
                    m_pendingImages.erase(pit);
                    // Sampled only (no storage usage): paraLLEl-GS recognises it as a replacement
                    // and keeps it out of its image pool. A full mip chain, for trilinear and
                    // anisotropic sampling.
                    auto info = Vulkan::ImageCreateInfo::immutable_2d_image(pending.width, pending.height, VK_FORMAT_R8G8B8A8_UNORM, true);
                    Vulkan::ImageInitialData init = {pending.rgba.data(), 0, 0};
                    HdImage hd;
                    hd.image = m_dev->create_image(info, &init);
                    if (!hd.image)
                        continue;
                    hd.bytes = pending.rgba.size() * 4 / 3;
                    created += pending.rgba.size();
                    m_hdBytes += hd.bytes;
                    m_hdByImage[hd.image.get()] = key;
                    auto &stored = m_hdImages[key] = std::move(hd);
                    for (const PendingBind &b : pending.binds)
                        bindLocked(stored, key, b.stableKey, b.cacheKey, b.recolor ? b.transform : nullptr);
                }
                // Least recently used images over the memory budget go (they load again when needed).
                m_usedImages.clear();
                m_iface.collect_used_replacements(m_usedImages);
                for (const Vulkan::Image *image : m_usedImages)
                {
                    auto u = m_hdByImage.find(image);
                    if (u != m_hdByImage.end())
                        m_hdImages[u->second].lastUsed = m_textureFrame;
                }
                if (m_hdBytes > m_hdBudget)
                    evictLocked();
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                m_textureWorstMs = std::max(m_textureWorstMs, ms);
                if (m_textureFrame % 300 == 0 && !m_hdImages.empty())
                {
                    const auto ts = m_textures.stats();
                    std::fprintf(stderr, "[textures] pack: %zu images, %zu MB (budget %zu), %zu pending, %zu evicted, worst %.1f ms/frame; "
                                         "palettes: %zu patterns, %zu/%zu fitted, %zu recoloured binds\n",
                                 m_hdImages.size(), m_hdBytes >> 20, m_hdBudget >> 20, m_pendingOrder.size(), m_evicted,
                                 m_textureWorstMs, ts.patterns, ts.fitted, ts.fits, m_recolored);
                    m_textureWorstMs = 0.0;
                }
                // Shadow GS instances decode a frame later: keep trying for two seconds.
                for (auto it = m_shadowReplacements.begin(); it != m_shadowReplacements.end();)
                {
                    for (auto &shadow : m_shadowIf)
                        if (shadow)
                            shadow->replace_cached_texture(it->first, it->second.first);
                    if (--it->second.second == 0)
                        it = m_shadowReplacements.erase(it);
                    else
                        ++it;
                }
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
            std::unique_ptr<ParallelGS::GSInterface> m_shadowIf[kMaxShadows];
            std::vector<uint8_t> m_shadowStream[kMaxShadows];
            const uint8_t *m_variant[kMaxShadows] = {};
            uint32_t m_variantSize[kMaxShadows] = {};
            uint32_t m_shadowCount = 0;
            TextureTools m_textures;
            std::vector<ParallelGS::GSRenderer::TextureReadback> m_readbacks;
            std::vector<TextureTools::Replacement> m_replacements;
            std::unordered_map<uint64_t, HdImage> m_hdImages; // content key -> replacement image
            std::unordered_map<const Vulkan::Image *, uint64_t> m_hdByImage;
            std::unordered_map<uint64_t, PendingImage> m_pendingImages;
            std::deque<uint64_t> m_pendingOrder;
            std::vector<const Vulkan::Image *> m_usedImages;
            size_t m_hdBytes = 0, m_evicted = 0, m_recolored = 0;
            size_t m_hdBudget = [] {
                const char *mb = std::getenv("RT_TEXTURE_PACK_BUDGET_MB");
                return size_t(mb ? std::strtoull(mb, nullptr, 10) : 2048) << 20;
            }();
            uint64_t m_textureFrame = 0;
            double m_textureWorstMs = 0.0;
            std::unordered_set<uint64_t> m_replaced; // content keys of pack images in use
            std::string m_packDir;
            std::unordered_map<uint64_t, std::pair<Vulkan::ImageHandle, uint32_t>> m_shadowReplacements;
            std::unordered_set<uint64_t> m_predicted; // stable keys with a prediction set
            struct ShadowJob
            {
                std::vector<uint8_t> stream;
                ParallelGS::PrivRegisterState priv;
                ParallelGS::VSyncInfo vsync;
                uint64_t serial = 0;
                bool pending = false;
            };
            ShadowJob m_shadowJobs[kMaxShadows];
            std::mutex m_shadowJobMutex;
            std::condition_variable m_shadowJobCv;
            std::thread m_shadowWorker;
            uint32_t m_shadowGeneration = 0;
            bool m_shadowStop = false;
            Vulkan::BufferHandle m_readback;

            GSCpuBackend m_shadow; // transfer/readback logic over the shared local memory
            uint8_t *m_vram = nullptr;
            uint32_t m_vramSize = 0u;
            bool m_gpuVramNewer = false;
            std::chrono::steady_clock::time_point m_texturesServiced{};
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
