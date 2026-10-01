#include "runtime/gs/gs_pgs_backend.h"

#if defined(PS2X_HAVE_PGS)

#include "runtime/gs/gs_cpu_backend.h"

#include "context.hpp"
#include "device.hpp"
#include "gs_interface.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>

namespace ps2x::gs
{
    namespace
    {
        constexpr uint32_t kHostFrameWidth = 640u;  // matches GS::copyLatchedHostPresentationFrame
        constexpr uint32_t kHostFrameHeight = 512u;
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

        class PgsBackend final : public GSRasterBackend, public GSPacketMirror
        {
        public:
            bool init(const PgsOptions &options, std::string &error)
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

                m_device.set_context(m_context);
                m_device.init_frame_contexts(4);

                ParallelGS::GSOptions gsOptions = {};
                if (!m_iface.init(&m_device, gsOptions))
                {
                    error = "paraLLEl-GS init failed (missing Vulkan features?)";
                    return false;
                }

                const auto &props = m_device.get_gpu_properties();
                std::cout << "[gs] paraLLEl-GS on " << props.deviceName << std::endl;
                return true;
            }

            // ---------------------------------------------------------------- GSPacketMirror
            void MirrorGifPacket(uint32_t pathIndex, const uint8_t *data, uint32_t sizeBytes) override
            {
                if (!data || sizeBytes < 16u || pathIndex > 3u)
                    return;
                std::lock_guard<std::mutex> lock(m_mutex);
                m_iface.gif_transfer(pathIndex, data, sizeBytes);
                m_gpuVramNewer = true;
            }

            void MirrorRegisterWrite(uint8_t regAddr, uint64_t value) override
            {
                std::lock_guard<std::mutex> lock(m_mutex);
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
                std::lock_guard<std::mutex> lock(m_mutex);
                uploadWholeVramLocked();
            }

            void Reset() override
            {
                std::lock_guard<std::mutex> lock(m_mutex);
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
                std::lock_guard<std::mutex> lock(m_mutex);
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
                std::lock_guard<std::mutex> lock(m_mutex);

                auto &priv = m_iface.get_priv_register_state();
                setPrivReg(priv.pmode, request.pmode);
                setPrivReg(priv.smode2, request.smode2);
                setPrivReg(priv.dispfb1, request.dispfb1);
                setPrivReg(priv.display1, displayToHardware(request.display1, request.smode2));
                setPrivReg(priv.dispfb2, request.dispfb2);
                setPrivReg(priv.display2, displayToHardware(request.display2, request.smode2));
                setPrivReg(priv.bgcolor, request.bgcolor);
                priv.smode1.CMOD = 2; // NTSC
                priv.smode1.LC = 32;

                ParallelGS::VSyncInfo vsync = {};
                vsync.phase = static_cast<uint32_t>(request.vsyncTick & 1u);
                vsync.dst_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                vsync.dst_stage = VK_PIPELINE_STAGE_2_COPY_BIT;
                vsync.dst_access = VK_ACCESS_2_TRANSFER_READ_BIT;
                vsync.adapt_to_internal_horizontal_resolution = true;
                // Road Trip renders 640x224 fields (INT=1, FFMD=1); paraLLEl-GS's default
                // deinterlacer reconstructs the 448-line picture from them.

                m_iface.flush();
                ParallelGS::ScanoutResult scanout = m_iface.vsync(vsync);
                PresentationFrame frame{};
                if (scanout.image)
                    frame = readbackLocked(*scanout.image);
                m_device.next_frame_context();
                return frame;
            }

        private:
            PresentationFrame readbackLocked(const Vulkan::Image &image)
            {
                const uint32_t w = image.get_width();
                const uint32_t h = image.get_height();

                Vulkan::BufferCreateInfo info = {};
                info.size = static_cast<VkDeviceSize>(w) * h * 4u;
                info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                info.domain = Vulkan::BufferDomain::CachedHost;
                if (!m_readback || m_readback->get_create_info().size < info.size)
                    m_readback = m_device.create_buffer(info);

                auto cmd = m_device.request_command_buffer();
                cmd->copy_image_to_buffer(*m_readback, image, 0, {}, {w, h, 1}, 0, 0,
                                          {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
                cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
                Vulkan::Fence fence;
                m_device.submit(cmd, &fence);
                fence->wait();

                const auto *src = static_cast<const uint8_t *>(
                    m_device.map_host_buffer(*m_readback, Vulkan::MEMORY_ACCESS_READ_BIT));

                PresentationFrame frame{};
                frame.width = std::min(w, kHostFrameWidth);
                frame.height = std::min(h, kHostFrameHeight);
                frame.pixels.assign(static_cast<size_t>(kHostFrameWidth) * kHostFrameHeight * 4u, 0u);
                for (uint32_t y = 0; y < frame.height; ++y)
                {
                    uint8_t *dst = frame.pixels.data() + static_cast<size_t>(y) * kHostFrameWidth * 4u;
                    std::memcpy(dst, src + static_cast<size_t>(y) * w * 4u, static_cast<size_t>(frame.width) * 4u);
                    for (uint32_t x = 0; x < frame.width; ++x)
                        dst[x * 4u + 3u] = 0xFF;
                }
                m_device.unmap_host_buffer(*m_readback, Vulkan::MEMORY_ACCESS_READ_BIT);
                return frame;
            }

            void pullVramFromGpu()
            {
                std::lock_guard<std::mutex> lock(m_mutex);
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

            Vulkan::Context m_context;
            Vulkan::Device m_device;
            ParallelGS::GSInterface m_iface;
            Vulkan::BufferHandle m_readback;

            GSCpuBackend m_shadow; // transfer/readback logic over the shared local memory
            uint8_t *m_vram = nullptr;
            uint32_t m_vramSize = 0u;
            bool m_gpuVramNewer = false;
            std::mutex m_mutex;
        };
    }

    bool pgsAvailable() { return true; }

    std::unique_ptr<GSRasterBackend> createPgsBackend(const PgsOptions &options, std::string &error)
    {
        auto backend = std::make_unique<PgsBackend>();
        if (!backend->init(options, error))
            return nullptr;
        return backend;
    }
}

#else

namespace ps2x::gs
{
    bool pgsAvailable() { return false; }

    std::unique_ptr<GSRasterBackend> createPgsBackend(const PgsOptions &, std::string &error)
    {
        error = "built without paraLLEl-GS (PS2X_ENABLE_PGS=OFF)";
        return nullptr;
    }
}

#endif
