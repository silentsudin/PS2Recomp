// Hardware GS (see runtime/gs/gs_hw_backend.h).
//
// The GS thread turns the frontend's primitives into batches of triangles (one per run of equal
// draw state) and resolves each batch's texture against the CPU copy of GS memory at that
// moment. Present records the whole run of batches into one command buffer, under the device
// lock once per frame, then scans out the display buffer.
//
// Render targets: one colour image per frame buffer (FBP), RGBA8 at the render scale (vertical
// scale doubled for progressive fields), alpha kept as A/128; one D32F depth image per Z buffer.
// Textures that live inside a render target (the post-pass sampling its own frame buffer, the
// loading-screen blur) are read from a snapshot of that target.

#include "runtime/gs/gs_hw_backend.h"

#if defined(PS2X_HAVE_PGS)

#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "../gs_pgs_shared.h"
#include "../post/post_spirv.h"
#include "hw_spirv.h"

#include "context.hpp"
#include "device.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ps2x::gs
{
    namespace
    {
        constexpr uint32_t kPages = 512; // 4 MB of GS memory in 8 KB pages

        struct HwVertex
        {
            float x, y, z;
            uint32_t rgba;
            float s, t, q;
            float fog;
        };
        static_assert(sizeof(HwVertex) == 32);

        struct Push
        {
            float posScale[4];
            float texNorm[4];
            float region[4];
            float fogColor[4];
            float lod[4];
            float texa[4];
            uint32_t mode[4];
            float texOffset[4];
        };
        static_assert(sizeof(Push) == 128);

        enum : uint32_t
        {
            F_TME = 1u,
            F_TCC = 8u,
            F_FGE = 16u,
            F_MIP = 1024u,
            F_LINEAR = 2048u,
            F_REPEAT_IN_SHADER = 4096u,
            F_DATE = 8192u,
            F_DATM = 16384u,
        };

        // Page geometry of a pixel format: page width/height in pixels.
        void pageSize(uint32_t psm, uint32_t &w, uint32_t &h)
        {
            switch (psm)
            {
            case GS_PSM_CT16:
            case GS_PSM_CT16S:
            case GS_PSM_Z16:
            case GS_PSM_Z16S:
                w = 64, h = 64;
                break;
            case GS_PSM_T8:
                w = 128, h = 64;
                break;
            case GS_PSM_T4:
                w = 128, h = 128;
                break;
            default: // 32-bit formats and the T8H/T4HL/T4HH views of them
                w = 64, h = 32;
                break;
            }
        }

        // The pages a rectangle of a buffer (base in blocks, width in 64-pixel units) touches.
        void pageRange(uint32_t bp, uint32_t bw, uint32_t psm, uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                       uint32_t &first, uint32_t &last)
        {
            uint32_t pw, ph;
            pageSize(psm, pw, ph);
            const uint32_t perRow = std::max(1u, (std::max(bw, 1u) * 64u) / pw);
            const uint32_t px0 = x / pw, py0 = y / ph;
            const uint32_t px1 = (x + std::max(w, 1u) - 1u) / pw, py1 = (y + std::max(h, 1u) - 1u) / ph;
            first = bp / 32u + py0 * perRow + px0;
            last = bp / 32u + py1 * perRow + px1;
        }

        bool isIndexed(uint32_t psm)
        {
            return psm == GS_PSM_T8 || psm == GS_PSM_T4 || psm == GS_PSM_T8H || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
        }
        bool isFourBit(uint32_t psm) { return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH; }

        uint32_t rgba5551(uint16_t c)
        {
            return ((c & 0x1Fu) << 3) | (((c >> 5) & 0x1Fu) << 11) | (((c >> 10) & 0x1Fu) << 19) | (((c >> 15) & 1u) << 31);
        }

        // As the CPU GS (gs_cpu_backend.cpp applyTexa).
        uint32_t applyTexa(const GSTexaReg &texa, uint32_t psm, uint32_t texel)
        {
            if (psm == GS_PSM_CT32)
                return texel;
            const bool rgbZero = (texel & 0x00FFFFFFu) == 0u;
            uint32_t a = texel >> 24;
            if (psm == GS_PSM_CT24)
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            else if (psm == GS_PSM_CT16 || psm == GS_PSM_CT16S)
                a = (a & 0x80u) ? texa.ta1 : ((texa.aem && rgbZero) ? 0u : texa.ta0);
            return (texel & 0x00FFFFFFu) | (a << 24);
        }

        uint64_t mix(uint64_t h, uint64_t v)
        {
            h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
            return h;
        }

        void unsupported(const char *what)
        {
            static std::mutex m;
            static std::unordered_set<std::string> seen;
            std::lock_guard<std::mutex> lock(m);
            if (seen.insert(what).second)
                std::cerr << "[hwgs] unsupported: " << what << "\n";
        }

        struct Batch
        {
            GSDrawState state{};
            uint32_t firstVertex = 0, vertexCount = 0;
            Vulkan::ImageHandle texture; // decoded texture, or null
            bool fromTarget = false;     // texture read from a render target
            uint32_t targetFbp = 0;
            float texOffsetX = 0, texOffsetY = 0;
        };

        struct Target
        {
            uint32_t fbp = 0, fbw = 0, psm = 0;
            uint32_t width = 0, height = 0; // GS pixels
            uint32_t sx = 1, sy = 1;        // device pixels per GS pixel
            Vulkan::ImageHandle color, snapshot;
            uint64_t drawSerial = 1, snapshotSerial = 0;
        };

        class HwBackend final : public GSRasterBackend, public PgsControl
        {
        public:
            ~HwBackend() override
            {
                if (!m_shared || m_own)
                {
                    if (m_own)
                    {
                        const auto lock = lockDevice();
                        m_dev->wait_idle();
                    }
                    return;
                }
                const auto lock = lockDevice();
                m_shared->flushLocked = nullptr;
                m_shared->attached = false;
                m_shared->scanout.reset();
            }

            bool init(const HwOptions &options, std::string &error)
            {
                m_shared = pgsShared(options.presenter);
                if (!m_shared || !m_shared->device)
                {
                    // A device of its own (headless runs, tests): pictures go back to the CPU.
                    if (!options.vulkanLibrary.empty())
                        setenv("GRANITE_VULKAN_LIBRARY", options.vulkanLibrary.c_str(), 1);
                    if (!Vulkan::Context::init_loader(nullptr))
                    {
                        error = "could not load a Vulkan library";
                        return false;
                    }
                    m_context.set_num_thread_indices(1);
                    if (!m_context.init_instance_and_device(nullptr, 0, nullptr, 0,
                                                            Vulkan::CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT))
                    {
                        error = "Vulkan instance/device creation failed";
                        return false;
                    }
                    m_ownDevice.set_context(m_context);
                    m_ownDevice.init_frame_contexts(4);
                    m_ownShared.device = &m_ownDevice;
                    m_shared = &m_ownShared;
                    m_own = true;
                }
                m_dev = m_shared->device;
                if (const char *s = std::getenv("RT_GS_SCALE"))
                    m_scale = std::clamp<uint32_t>(static_cast<uint32_t>(std::atoi(s)), 1u, 6u);

                const auto lock = lockDevice();
                Vulkan::ResourceLayout vert = {};
                vert.input_mask = 0x1F;
                vert.output_mask = 0x7;
                vert.push_constant_size = sizeof(Push);
                Vulkan::ResourceLayout frag = {};
                frag.input_mask = 0x7;
                frag.output_mask = 0x1;
                frag.push_constant_size = sizeof(Push);
                frag.sets[0].sampled_image_mask = 0x3;
                frag.sets[0].fp_mask = 0x3;
                m_drawProgram = m_dev->request_program(hw_spirv::draw_vert, sizeof(hw_spirv::draw_vert), hw_spirv::draw_frag,
                                                       sizeof(hw_spirv::draw_frag), &vert, &frag);
                if (!m_drawProgram)
                {
                    error = "shaders";
                    return false;
                }
                // A 1x1 texture for untextured draws (the shader always declares one).
                const uint32_t white = 0xFFFFFFFFu;
                Vulkan::ImageInitialData init = {&white, 0, 0};
                m_white = m_dev->create_image(Vulkan::ImageCreateInfo::immutable_2d_image(1, 1, VK_FORMAT_R8G8B8A8_UNORM), &init);

                m_shared->attached = true;
                m_shared->flushLocked = [] {};
                const auto &props = m_dev->get_gpu_properties();
                m_info.name = props.deviceName;
                m_info.vendorId = props.vendorID;
                m_info.deviceId = props.deviceID;
                m_info.apiVersion = props.apiVersion;
                m_info.maxSuperSampling = 16;
                std::cout << "[hwgs] hardware GS on " << m_info.name << ", render scale " << m_scale << "x\n";
                return true;
            }

            // ---------------------------------------------------------------- PgsControl
            PgsDeviceInfo deviceInfo() const override { return m_info; }
            void setSuperSampling(uint32_t samples) override
            {
                const uint32_t scale = samples >= 16 ? 4 : samples >= 4 ? 2 : 1;
                if (scale != m_scale.load())
                {
                    m_scale = scale;
                    m_rescale = true;
                }
            }
            uint32_t superSampling() const override { return m_scale * m_scale; }
            void setSharpTextures(bool) override {}
            void setTextures(const std::string &, const std::string &) override {}
            TexturePackStats texturePackStats() const override { return {}; }
            void setAnisotropy(uint32_t) override {}

            // ---------------------------------------------------------------- GSRasterBackend
            bool WantsPrimitives() const override { return true; }

            void Initialize(uint8_t *vram, uint32_t vramSize) override
            {
                m_vram = vram;
                m_vramSize = vramSize;
                m_cpu.Initialize(vram, vramSize);
                ++m_epoch;
            }

            void Reset() override
            {
                m_cpu.Reset();
                ++m_epoch;
            }

            void Submit(const GSPrimitiveBatch &batch) override
            {
                const GSDrawState &state = batch.state;
                const uint32_t type = state.prim.type;
                if (type == GS_PRIM_POINT || type == GS_PRIM_LINE || type == GS_PRIM_LINESTRIP)
                {
                    unsupported("points and lines");
                    return;
                }
                const uint32_t needed = type == GS_PRIM_SPRITE ? 2u : 3u;
                if (batch.vertexCount < needed)
                    return;

                // A new batch when the state or the texture contents changed (decided and resolved
                // without the frame lock: decoding a texture takes the device lock).
                bool fresh;
                {
                    std::lock_guard<std::mutex> lock(m_frameMutex);
                    fresh = !m_haveCurrent || m_currentEpoch != m_epoch ||
                            std::memcmp(&m_current.state, &state, sizeof(state)) != 0;
                    static const bool stats = std::getenv("RT_HWGS_STATS") != nullptr;
                    if (stats && fresh && m_haveCurrent)
                    {
                        if (m_currentEpoch != m_epoch)
                            ++m_breakEpoch;
                        else
                        {
                            const auto *x = reinterpret_cast<const uint8_t *>(&m_current.state);
                            const auto *y = reinterpret_cast<const uint8_t *>(&state);
                            size_t i = 0;
                            while (i < sizeof(state) && x[i] == y[i])
                                ++i;
                            ++m_breakAt[i];
                        }
                    }
                    if (fresh && m_open)
                        closeBatch();
                }
                if (fresh)
                {
                    m_current = Batch{};
                    // memcpy, not assignment: batches are compared with memcmp, padding included.
                    std::memcpy(&m_current.state, &state, sizeof(state));
                    if (state.prim.tme)
                        resolveTexture(m_current);
                    m_haveCurrent = true;
                    m_currentEpoch = m_epoch;
                }

                bool flush = false;
                std::unique_lock<std::mutex> lock(m_frameMutex);
                if (!m_open)
                {
                    // A buffer flip (the other frame buffer) or a long run: record what's pending.
                    flush = !m_batches.empty() && (m_batches.back().state.context.frame.fbp != state.context.frame.fbp ||
                                                   m_vertices.size() > 200000u || m_batches.size() > 4000u);
                    Batch b = m_current;
                    b.firstVertex = static_cast<uint32_t>(m_vertices.size());
                    b.vertexCount = 0;
                    m_batches.push_back(std::move(b));
                    m_open = true;
                }

                const GSContext &ctx = state.context;
                const float ofx = ctx.xyoffset.ofx / 16.0f, ofy = ctx.xyoffset.ofy / 16.0f;
                const double zScale = zUnit(ctx.zbuf.psm);
                const float tw = static_cast<float>(1u << std::min<uint32_t>(ctx.tex0.tw, 10u));
                const float th = static_cast<float>(1u << std::min<uint32_t>(ctx.tex0.th, 10u));
                auto convert = [&](const GSVertex &v, const GSVertex &colorFrom) {
                    HwVertex o;
                    o.x = v.x - ofx;
                    o.y = v.y - ofy;
                    o.z = static_cast<float>(std::min(v.z * zScale, 1.0));
                    o.rgba = colorFrom.r | (colorFrom.g << 8) | (colorFrom.b << 16) | (static_cast<uint32_t>(colorFrom.a) << 24);
                    if (state.prim.fst)
                    {
                        o.s = v.u / 16.0f / tw;
                        o.t = v.v / 16.0f / th;
                        o.q = 1.0f;
                    }
                    else
                    {
                        o.s = v.s;
                        o.t = v.t;
                        o.q = v.q;
                    }
                    o.fog = v.fog;
                    return o;
                };

                if (type == GS_PRIM_SPRITE)
                {
                    const GSVertex &a = batch.vertices[0], &b = batch.vertices[1];
                    HwVertex v0 = convert(a, b), v1 = convert(b, b);
                    v0.z = v1.z; // a sprite takes Z (and colour, fog) from its second vertex
                    v0.fog = v1.fog;
                    HwVertex v2 = v0, v3 = v0;
                    v2.x = v1.x, v2.s = v1.s; // top-right
                    v3.y = v1.y, v3.t = v1.t; // bottom-left
                    const HwVertex quad[6] = {v0, v2, v3, v2, v1, v3};
                    m_vertices.insert(m_vertices.end(), quad, quad + 6);
                    m_batches.back().vertexCount += 6;
                }
                else
                {
                    // Flat shading takes the colour of the last vertex.
                    const GSVertex &flat = batch.vertices[2];
                    for (int i = 0; i < 3; ++i)
                        m_vertices.push_back(convert(batch.vertices[i], state.prim.iip ? batch.vertices[i] : flat));
                    m_batches.back().vertexCount += 3;
                }
                lock.unlock();
                if (flush)
                    flushPending();
            }

            // Records and submits what the GS thread has queued (the device lock before the frame lock,
            // as in Present, so batches are recorded in order).
            void flushPending()
            {
                const auto device = lockDevice();
                std::vector<Batch> batches;
                std::vector<HwVertex> vertices;
                {
                    std::lock_guard<std::mutex> lock(m_frameMutex);
                    if (m_open)
                    {
                        // Keep the open batch's last vertices with it: move the whole open batch.
                        Batch open = m_batches.back();
                        m_batches.pop_back();
                        batches.swap(m_batches);
                        std::vector<HwVertex> tail(m_vertices.begin() + open.firstVertex, m_vertices.end());
                        vertices.swap(m_vertices);
                        open.firstVertex = 0;
                        m_batches.push_back(std::move(open));
                        m_vertices = std::move(tail);
                    }
                    else
                    {
                        batches.swap(m_batches);
                        vertices.swap(m_vertices);
                    }
                }
                if (batches.empty())
                    return;
                auto cmd = m_dev->request_command_buffer();
                record(*cmd, batches, vertices);
                m_dev->submit(cmd);
                // A device of its own: a frame context per flush (waits for the one a few flushes
                // back), so the GPU can't fall behind and resources are recycled.
                if (m_own)
                    m_dev->next_frame_context();
            }

            void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override
            {
                // The game reloads the CLUT with every TEX0, mostly with the same colours: only a
                // real change starts new batches.
                const auto before = m_clut;
                loadClut(tex0, texclut);
                if (m_clut == before)
                    return;
                std::lock_guard<std::mutex> lock(m_frameMutex);
                if (m_open)
                    closeBatch();
                ++m_epoch;
            }

            void BeginTransfer(const GSTransferCommand &command) override
            {
                std::lock_guard<std::mutex> lock(m_frameMutex);
                if (m_open)
                    closeBatch();
                if (command.direction == 0u || command.direction == 2u)
                {
                    uint32_t first, last;
                    pageRange(command.bitbltbuf.dbp, command.bitbltbuf.dbw, command.bitbltbuf.dpsm, command.trxpos.dsax,
                              command.trxpos.dsay, command.trxreg.rrw, command.trxreg.rrh, first, last);
                    for (uint32_t p = first; p <= last; ++p)
                        ++m_pageVersion[p % kPages];
                    if (command.direction == 2u)
                    {
                        pageRange(command.bitbltbuf.sbp, command.bitbltbuf.sbw, command.bitbltbuf.spsm, command.trxpos.ssax,
                                  command.trxpos.ssay, command.trxreg.rrw, command.trxreg.rrh, first, last);
                        if (findTargetForPages(first, last))
                            unsupported("local-to-local copy out of a render target");
                    }
                }
                else if (command.direction == 1u)
                    unsupported("local-to-host readback (render targets aren't written back)");
                m_cpu.BeginTransfer(command);
                ++m_epoch;
            }

            void UploadImage(const uint8_t *data, uint32_t sizeBytes) override
            {
                m_cpu.UploadImage(data, sizeBytes);
                m_pageCache.Invalidate();
            }

            void Flush() override {}
            void TextureFlush() override {}
            void Sync(GSSyncReason) override {}

            bool ClearFramebuffer(const GSContext &, uint32_t) override
            {
                // The game's clear is followed by the real clear sprite, which draws normally.
                return true;
            }

            uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override
            {
                return m_cpu.ConsumeLocalToHostBytes(dst, maxBytes);
            }
            uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override
            {
                return m_cpu.ReadVram(psm, base, bw, x, y);
            }
            void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override
            {
                m_cpu.WriteVram(psm, base, bw, x, y, value);
                std::lock_guard<std::mutex> lock(m_frameMutex);
                ++m_epoch;
                for (auto &v : m_pageVersion)
                    ++v;
            }
            void SnapshotVram(std::vector<uint8_t> &out) const override { m_cpu.SnapshotVram(out); }
            GSTransferSnapshot GetTransferSnapshot() const override { return m_cpu.GetTransferSnapshot(); }

            PresentationFrame Present(const GSPresentationRequest &request) override
            {
                Vulkan::Fence fence;
                uint32_t outW = 0, outH = 0;
                {
                    const auto lock = lockDevice();
                    std::vector<Batch> batches;
                    std::vector<HwVertex> vertices;
                    {
                        std::lock_guard<std::mutex> frameLock(m_frameMutex);
                        if (m_open)
                            closeBatch();
                        batches.swap(m_batches);
                        vertices.swap(m_vertices);
                    }
                    if (m_rescale.exchange(false))
                        m_targets.clear(), m_depths.clear();
                    m_yScale = request.progressiveFields ? 2u : 1u;
                    auto cmd = m_dev->request_command_buffer();
                    // The presenter may still be sampling an older scanout image.
                    cmd->barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
                    record(*cmd, batches, vertices);
                    const Vulkan::Image *scan = scanout(*cmd, request);
                    if (scan)
                    {
                        outW = scan->get_width();
                        outH = scan->get_height();
                        m_shared->scanout = m_scanout[m_scanoutIndex];
                        ++m_shared->presentSerial;
                    }
                    if (scan && request.readback)
                        readbackCopy(*cmd, *scan);
                    m_dev->submit(cmd, request.readback && scan ? &fence : nullptr);
                    if (m_own)
                        m_dev->next_frame_context();
                }
                static const bool stats = std::getenv("RT_HWGS_STATS") != nullptr;
                if (stats && ++m_stats.presents == 120)
                {
                    const double n = 120.0;
                    std::fprintf(stderr, "[hwgs] per present: %.0f batches (%.0f DATE), %.0f vertices, %.1f passes, %.1f snapshots, %.1f texture decodes\n",
                                 m_stats.batches / n, m_stats.dateBatches / n, m_stats.vertices / n, m_stats.passes / n,
                                 m_stats.snapshots / n, m_stats.decodes / n);
                    m_stats = {};
                    std::fprintf(stderr, "[hwgs] batch breaks: epoch %llu;", static_cast<unsigned long long>(m_breakEpoch));
                    for (auto &[at, n] : m_breakAt)
                        std::fprintf(stderr, " @%zu:%llu", at, static_cast<unsigned long long>(n));
                    std::fprintf(stderr, " (prim @%zu, texa @%zu, fog @%zu, context.tex0 @%zu, context.frame @%zu)\n",
                                 offsetof(GSDrawState, prim), offsetof(GSDrawState, texa), offsetof(GSDrawState, fogR),
                                 offsetof(GSDrawState, context) + offsetof(GSContext, tex0),
                                 offsetof(GSDrawState, context) + offsetof(GSContext, frame));
                    m_breakAt.clear();
                    m_breakEpoch = 0;
                }
                PresentationFrame frame{};
                if (fence && outW && outH)
                {
                    fence->wait();
                    const auto lock = lockDevice();
                    frame = readbackFrame(outW, outH);
                }
                return frame;
            }

        private:
            // ---------------------------------------------------------------- batching (GS thread)
            static double zUnit(uint32_t zpsm)
            {
                switch (zpsm & 0xFu)
                {
                case 0u: return 1.0 / 4294967296.0;   // Z32
                case 1u: return 1.0 / 16777216.0;     // Z24
                default: return 1.0 / 65536.0;        // Z16, Z16S
                }
            }

            void closeBatch()
            {
                m_open = false;
                if (!m_batches.empty() && m_batches.back().vertexCount == 0)
                    m_batches.pop_back();
            }

            bool findTargetForPages(uint32_t first, uint32_t last) const
            {
                std::lock_guard<std::mutex> lock(m_targetPagesMutex);
                for (const auto &[fbp, span] : m_targetPages)
                    if (first <= span.second && last >= span.first)
                        return true;
                return false;
            }

            // Which frame buffer (if any) a texture's pages lie in.
            bool textureInTarget(const GSTex0Reg &tex, uint32_t &fbp, uint32_t &fbw)
            {
                if (tex.psm != GS_PSM_CT32 && tex.psm != GS_PSM_CT24)
                    return false;
                uint32_t first, last;
                pageRange(tex.tbp0, tex.tbw, tex.psm, 0, 0, 1u << tex.tw, 1u << tex.th, first, last);
                // The target starting closest below the texture (targets sized by a 512-line scissor
                // overlap the next frame buffer's pages).
                std::lock_guard<std::mutex> lock(m_targetPagesMutex);
                bool found = false;
                for (auto &[base, range] : m_targetPages)
                    if (first >= range.first && first <= range.second && (!found || base > fbp))
                    {
                        fbp = base;
                        fbw = range.fbw;
                        found = true;
                    }
                return found;
            }

            void resolveTexture(Batch &b)
            {
                const GSContext &ctx = b.state.context;
                const GSTex0Reg &tex = ctx.tex0;
                uint32_t fbp = 0, targetFbw = 1;
                if (textureInTarget(tex, fbp, targetFbw))
                {
                    b.fromTarget = true;
                    b.targetFbp = fbp;
                    // Texture origin inside the target, in whole pages.
                    const uint32_t pageOffset = tex.tbp0 / 32u - fbp;
                    const uint32_t fbw = std::max<uint32_t>(targetFbw, 1u);
                    b.texOffsetX = static_cast<float>((pageOffset % fbw) * 64u);
                    b.texOffsetY = static_cast<float>((pageOffset / fbw) * 32u);
                    if (tex.tbp0 % 32u)
                        unsupported("texture starting mid-page inside a render target");
                    return;
                }
                b.texture = decodedTexture(b.state);
            }

            // ---------------------------------------------------------------- CLUT and textures
            void loadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
            {
                // CLD conditions and CSM1 layout as the CPU GS (gs_cpu_backend.cpp LoadClut).
                if (!m_vram || !isIndexed(tex0.psm))
                    return;
                switch (tex0.cld)
                {
                case 1u: break;
                case 2u: m_clutCbp[0] = tex0.cbp; break;
                case 3u: m_clutCbp[1] = tex0.cbp; break;
                case 4u: if (m_clutCbp[0] == tex0.cbp) return; m_clutCbp[0] = tex0.cbp; break;
                case 5u: if (m_clutCbp[1] == tex0.cbp) return; m_clutCbp[1] = tex0.cbp; break;
                default: return;
                }
                const bool fourBit = isFourBit(tex0.psm);
                const bool sixteen = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
                const bool thirtyTwo = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
                if (!sixteen && !thirtyTwo)
                    return;
                const uint32_t entries = fourBit ? 16u : 256u;
                const uint32_t dstBase = (static_cast<uint32_t>(tex0.csa) & (sixteen ? 0x1Fu : 0x0Fu)) << 4u;
                const bool csm1Suffix = tex0.csm == 0u && thirtyTwo && !fourBit;
                for (uint32_t e = csm1Suffix ? dstBase : 0u; e < entries; ++e)
                {
                    uint32_t sx, sy, sw = 1u;
                    if (tex0.csm == 0u)
                    {
                        const uint32_t i = (e & ~0x18u) | ((e & 0x08u) << 1u) | ((e & 0x10u) >> 1u);
                        sx = i & 0x0Fu;
                        sy = i >> 4u;
                    }
                    else
                    {
                        sw = texclut.cbw ? texclut.cbw : 1u;
                        sx = (static_cast<uint32_t>(texclut.cou) << 4u) + e;
                        sy = texclut.cov;
                    }
                    const uint32_t raw = GSMem::ReadTexture(m_pageCache, m_vram, tex0.cpsm, tex0.cbp, sw, sx, sy);
                    const uint32_t dst = (csm1Suffix ? e : dstBase + e) & (sixteen ? 0x1FFu : 0x0FFu);
                    if (sixteen)
                        m_clut[dst] = static_cast<uint16_t>(raw);
                    else
                    {
                        m_clut[dst] = static_cast<uint16_t>(raw & 0xFFFFu);
                        m_clut[dst + 256u] = static_cast<uint16_t>(raw >> 16u);
                    }
                }
            }

            uint32_t clutColor(const GSDrawState &s, uint32_t index) const
            {
                const GSTex0Reg &tex = s.context.tex0;
                const bool sixteen = tex.cpsm == GS_PSM_CT16 || tex.cpsm == GS_PSM_CT16S;
                const uint32_t base = (static_cast<uint32_t>(tex.csa) & (sixteen ? 0x1Fu : 0x0Fu)) << 4u;
                const uint32_t src = isFourBit(tex.psm) ? (index & 0x0Fu) : index;
                uint32_t i = (base + src) & (sixteen ? 0x1FFu : 0x0FFu);
                if (!sixteen && tex.csm == 0u && !isFourBit(tex.psm))
                    i = std::min((src & 0xF0u) + base, 240u) + (src & 0x0Fu);
                if (sixteen)
                    return applyTexa(s.texa, tex.cpsm, rgba5551(m_clut[i]));
                const uint32_t raw = m_clut[i] | (static_cast<uint32_t>(m_clut[i + 256u]) << 16u);
                return applyTexa(s.texa, tex.cpsm, tex.cpsm == GS_PSM_CT24 ? (raw & 0x00FFFFFFu) : raw);
            }

            uint32_t texel(const GSDrawState &s, uint32_t psm, uint32_t tbp, uint32_t tbw, uint32_t x, uint32_t y)
            {
                const uint32_t raw = GSMem::ReadTexture(m_pageCache, m_vram, psm, tbp, tbw, x, y);
                switch (psm)
                {
                case GS_PSM_CT32:
                case GS_PSM_CT24:
                    return applyTexa(s.texa, psm, raw);
                case GS_PSM_CT16:
                case GS_PSM_CT16S:
                    return applyTexa(s.texa, psm, rgba5551(static_cast<uint16_t>(raw)));
                default:
                    if (isIndexed(psm))
                        return clutColor(s, raw & 0xFFu);
                    return 0xFFFF00FFu;
                }
            }

            Vulkan::ImageHandle decodedTexture(const GSDrawState &s)
            {
                const GSContext &ctx = s.context;
                const GSTex0Reg &tex = ctx.tex0;
                const uint32_t tw = std::min<uint32_t>(tex.tw, 10u), th = std::min<uint32_t>(tex.th, 10u);
                uint32_t levels = 1;
                const uint32_t mxl = (ctx.tex1 >> 2) & 7u;
                if (mxl > 0 && ctx.miptbp1 != 0)
                    levels = std::min<uint32_t>(mxl, 6u) + 1u;
                else if (mxl > 0)
                    unsupported("mipmaps without MIPTBP (automatic base addresses)");

                struct Level
                {
                    uint32_t tbp, tbw;
                };
                Level lv[7] = {{tex.tbp0, tex.tbw}};
                for (uint32_t l = 1; l < levels; ++l)
                {
                    const uint64_t reg = l <= 3 ? ctx.miptbp1 : ctx.miptbp2;
                    const uint32_t shift = ((l - 1) % 3) * 20u;
                    lv[l] = {static_cast<uint32_t>((reg >> shift) & 0x3FFFu), static_cast<uint32_t>((reg >> (shift + 14)) & 0x3Fu)};
                }

                // Key: where and how the texture is read, the pages' contents, the palette, TEXA.
                uint64_t key = mix(0, tex.psm | (tw << 8) | (th << 12) | (levels << 16) | (static_cast<uint64_t>(tex.cpsm) << 20));
                for (uint32_t l = 0; l < levels; ++l)
                {
                    key = mix(key, lv[l].tbp | (static_cast<uint64_t>(lv[l].tbw) << 16));
                    uint32_t first, last;
                    pageRange(lv[l].tbp, lv[l].tbw, tex.psm, 0, 0, std::max(1u << tw >> l, 1u), std::max(1u << th >> l, 1u), first, last);
                    for (uint32_t p = first; p <= last; ++p)
                        key = mix(key, (static_cast<uint64_t>(p) << 32) | m_pageVersion[p % kPages]);
                }
                key = mix(key, s.texa.ta0 | (s.texa.aem << 8) | (s.texa.ta1 << 16));
                if (isIndexed(tex.psm))
                {
                    const uint32_t n = isFourBit(tex.psm) ? 16u : 256u;
                    key = mix(key, tex.csa | (tex.csm << 8));
                    for (uint32_t i = 0; i < n; ++i)
                        key = mix(key, clutColor(s, i));
                }

                const uint64_t now = m_frame.load();
                auto it = m_textures.find(key);
                if (it != m_textures.end())
                {
                    it->second.lastUse = now;
                    return it->second.image;
                }
                // Textures not used for a while (palette fades leave many behind).
                if (m_textures.size() > 256u && now != m_lastEviction)
                {
                    m_lastEviction = now;
                    for (auto i = m_textures.begin(); i != m_textures.end();)
                        i = now - i->second.lastUse > 600u ? m_textures.erase(i) : std::next(i);
                }

                ++m_stats.decodes;
                std::vector<std::vector<uint32_t>> data(levels);
                std::array<Vulkan::ImageInitialData, 7> init{};
                for (uint32_t l = 0; l < levels; ++l)
                {
                    const uint32_t w = std::max(1u, (1u << tw) >> l), h = std::max(1u, (1u << th) >> l);
                    data[l].resize(static_cast<size_t>(w) * h);
                    for (uint32_t y = 0; y < h; ++y)
                        for (uint32_t x = 0; x < w; ++x)
                            data[l][static_cast<size_t>(y) * w + x] = texel(s, tex.psm, lv[l].tbp, lv[l].tbw, x, y);
                    init[l] = {data[l].data(), 0, 0};
                }
                auto info = Vulkan::ImageCreateInfo::immutable_2d_image(1u << tw, 1u << th, VK_FORMAT_R8G8B8A8_UNORM);
                info.levels = levels;
                Vulkan::ImageHandle image;
                {
                    const auto lock = lockDevice();
                    image = m_dev->create_image(info, init.data());
                }
                m_textures[key] = {image, now};
                return image;
            }

            // ---------------------------------------------------------------- recording (device lock held)
            Target &target(uint32_t fbp, uint32_t fbw, uint32_t psm, uint32_t minHeight, Vulkan::CommandBuffer &cmd,
                           bool &passOpen)
            {
                Target &t = m_targets[fbp];
                const uint32_t width = std::max<uint32_t>(fbw, 1u) * 64u;
                const uint32_t height = std::max<uint32_t>((minHeight + 31u) & ~31u, 32u);
                if (!t.color || t.width != width || t.height < height || t.sx != m_scale || t.sy != m_scale * m_yScale)
                {
                    endPass(cmd, passOpen);
                    auto info = Vulkan::ImageCreateInfo::render_target(width * m_scale, height * m_scale * m_yScale,
                                                                       VK_FORMAT_R8G8B8A8_UNORM);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    auto image = m_dev->create_image(info);
                    cmd.image_barrier(*image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                                      0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                    VkClearValue zero = {};
                    cmd.clear_image(*image, zero);
                    if (t.color && t.width == width && t.sx == m_scale && t.sy == m_scale * m_yScale)
                    {
                        // Grown: keep what was drawn.
                        cmd.image_barrier(*t.color, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                          VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                        cmd.barrier(VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                                    VK_ACCESS_2_TRANSFER_WRITE_BIT);
                        cmd.copy_image(*image, *t.color, {0, 0, 0}, {0, 0, 0},
                                       {t.color->get_width(), t.color->get_height(), 1}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                       {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
                    }
                    cmd.image_barrier(*image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                    t.color = image;
                    t.fbp = fbp;
                    t.fbw = fbw;
                    t.psm = psm;
                    t.width = width;
                    t.height = height;
                    t.sx = m_scale;
                    t.sy = m_scale * m_yScale;
                    t.snapshot.reset();
                    ++t.drawSerial;
                    std::lock_guard<std::mutex> lock(m_targetPagesMutex);
                    uint32_t first, last;
                    pageRange(fbp * 32u, fbw, GS_PSM_CT32, 0, 0, width, height, first, last);
                    m_targetPages[fbp] = {first, last, fbw};
                }
                return t;
            }

            Vulkan::ImageHandle &depth(uint32_t zbp, const Target &t, Vulkan::CommandBuffer &cmd, bool &passOpen)
            {
                Vulkan::ImageHandle &d = m_depths[zbp];
                const uint32_t w = t.color->get_width(), h = t.color->get_height();
                if (!d || d->get_width() != w || d->get_height() != h)
                {
                    endPass(cmd, passOpen);
                    auto info = Vulkan::ImageCreateInfo::render_target(w, h, VK_FORMAT_D32_SFLOAT);
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    d = m_dev->create_image(info);
                    cmd.image_barrier(*d, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                                      0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                    VkClearValue zero = {};
                    cmd.clear_image(*d, zero, VK_IMAGE_ASPECT_DEPTH_BIT);
                    cmd.image_barrier(*d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
                }
                return d;
            }

            void endPass(Vulkan::CommandBuffer &cmd, bool &passOpen)
            {
                if (passOpen)
                {
                    cmd.end_render_pass();
                    passOpen = false;
                    m_passColor = nullptr;
                    m_passDepth = nullptr;
                }
            }

            // A copy of a target to sample from (refreshed when the target was drawn since).
            const Vulkan::Image &snapshot(Target &t, Vulkan::CommandBuffer &cmd, bool &passOpen)
            {
                if (t.snapshot && t.snapshotSerial == t.drawSerial)
                    return *t.snapshot;
                endPass(cmd, passOpen);
                ++m_stats.snapshots;
                if (!t.snapshot || t.snapshot->get_width() != t.color->get_width() ||
                    t.snapshot->get_height() != t.color->get_height())
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(t.color->get_width(), t.color->get_height(),
                                                                       VK_FORMAT_R8G8B8A8_UNORM);
                    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    t.snapshot = m_dev->create_image(info);
                }
                cmd.image_barrier(*t.color, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                cmd.image_barrier(*t.snapshot, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT,
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT);
                cmd.copy_image(*t.snapshot, *t.color);
                cmd.image_barrier(*t.snapshot, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                cmd.image_barrier(*t.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                t.snapshotSerial = t.drawSerial;
                return *t.snapshot;
            }

            const Vulkan::Sampler &sampler(bool magLinear, bool minLinear, bool mipLinear, bool repeatU, bool repeatV)
            {
                const uint32_t key = magLinear | (minLinear << 1) | (mipLinear << 2) | (repeatU << 3) | (repeatV << 4);
                auto &s = m_samplers[key];
                if (!s)
                {
                    Vulkan::SamplerCreateInfo info = {};
                    info.mag_filter = magLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
                    info.min_filter = minLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
                    info.mipmap_mode = mipLinear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    info.address_mode_u = repeatU ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                    info.address_mode_v = repeatV ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                    info.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                    info.max_lod = VK_LOD_CLAMP_NONE;
                    s = m_dev->create_sampler(info);
                }
                return *s;
            }

            // PS2 blending (A - B) * C / 128 + D as Vulkan fixed-function blend. Returns false when
            // it can't be expressed (logged).
            static bool blendFactors(uint64_t alpha, VkBlendFactor &src, VkBlendFactor &dst, VkBlendOp &op)
            {
                const uint32_t a = alpha & 3u, b = (alpha >> 2) & 3u, c = (alpha >> 4) & 3u, d = (alpha >> 6) & 3u;
                const VkBlendFactor f = c == 0u ? VK_BLEND_FACTOR_SRC_ALPHA : c == 1u ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_CONSTANT_ALPHA;
                const VkBlendFactor oneMinusF = c == 0u   ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
                                                : c == 1u ? VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA
                                                          : VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
                // Coefficient of Cs and Cd: (A==X) C - (B==X) C + (D==X), X in {Cs (0), Cd (1)}.
                auto coef = [&](uint32_t x, VkBlendFactor &factor, int &sign) {
                    const int cTerms = (a == x ? 1 : 0) - (b == x ? 1 : 0);
                    const bool one = d == x;
                    sign = 1;
                    if (cTerms == 0)
                        factor = one ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_ZERO;
                    else if (cTerms == 1)
                    {
                        if (one)
                            return false; // 1 + C
                        factor = f;
                    }
                    else if (one)
                        factor = oneMinusF;
                    else
                    {
                        factor = f;
                        sign = -1;
                    }
                    return true;
                };
                int ss, ds;
                if (!coef(0u, src, ss) || !coef(1u, dst, ds))
                    return false;
                if (ss > 0 && ds > 0)
                    op = VK_BLEND_OP_ADD;
                else if (ss > 0)
                    op = VK_BLEND_OP_SUBTRACT;
                else if (ds > 0)
                    op = VK_BLEND_OP_REVERSE_SUBTRACT;
                else
                {
                    src = dst = VK_BLEND_FACTOR_ZERO;
                    op = VK_BLEND_OP_ADD;
                }
                return true;
            }

            void record(Vulkan::CommandBuffer &cmd, const std::vector<Batch> &batches, const std::vector<HwVertex> &vertices)
            {
                ++m_frame;
                bool passOpen = false;
                m_stats.batches += batches.size();
                m_stats.vertices += vertices.size();
                for (const Batch &b : batches)
                {
                    const GSDrawState &s = b.state;
                    if ((s.context.test >> 14) & 1u)
                        ++m_stats.dateBatches;
                    const GSContext &ctx = s.context;
                    const uint32_t fpsm = ctx.frame.psm;
                    if (fpsm != GS_PSM_CT32 && fpsm != GS_PSM_CT24)
                    {
                        unsupported("frame buffer format other than CT32/CT24");
                        continue;
                    }
                    Target &t = target(ctx.frame.fbp, ctx.frame.fbw, fpsm, static_cast<uint32_t>(ctx.scissor.y1) + 1u, cmd, passOpen);

                    const bool zte = (ctx.test >> 16) & 1u;
                    const uint32_t ztst = (ctx.test >> 17) & 3u;
                    const bool zwrite = !ctx.zbuf.zmask;
                    const bool useDepth = (zte && ztst != 1u) || zwrite;
                    const Vulkan::Image *depthImage = nullptr;
                    if (useDepth)
                        depthImage = depth(ctx.zbuf.zbp, t, cmd, passOpen).get();

                    // The texture: decoded, or a snapshot of a render target.
                    const Vulkan::ImageView *texView = &m_white->get_view();
                    float extentW = 1, extentH = 1, alphaScale = 255.0f;
                    bool ct24Target = false;
                    if (s.prim.tme)
                    {
                        if (b.fromTarget)
                        {
                            auto it = m_targets.find(b.targetFbp);
                            if (it != m_targets.end() && it->second.color)
                            {
                                const Vulkan::Image &snap = snapshot(it->second, cmd, passOpen);
                                texView = &snap.get_view();
                                extentW = static_cast<float>(it->second.width);
                                extentH = static_cast<float>(it->second.height);
                                alphaScale = 128.0f;
                                ct24Target = ctx.tex0.psm == GS_PSM_CT24;
                            }
                        }
                        else if (b.texture)
                        {
                            texView = &b.texture->get_view();
                            extentW = static_cast<float>(b.texture->get_width());
                            extentH = static_cast<float>(b.texture->get_height());
                        }
                    }

                    // DATE reads the target as it was before this draw.
                    const uint64_t test = ctx.test;
                    const bool date = (test >> 14) & 1u;
                    const Vulkan::ImageView *destView = &m_white->get_view();
                    if (date)
                        destView = &snapshot(t, cmd, passOpen).get_view();

                    if (passOpen && (m_passColor != t.color.get() || m_passDepth != depthImage))
                        endPass(cmd, passOpen);
                    if (!passOpen)
                    {
                        Vulkan::RenderPassInfo rp = {};
                        rp.num_color_attachments = 1;
                        rp.color_attachments[0] = &t.color->get_view();
                        rp.load_attachments = 1;
                        rp.store_attachments = 1;
                        if (depthImage)
                        {
                            rp.depth_stencil = &depthImage->get_view();
                            rp.op_flags = Vulkan::RENDER_PASS_OP_LOAD_DEPTH_STENCIL_BIT | Vulkan::RENDER_PASS_OP_STORE_DEPTH_STENCIL_BIT;
                        }
                        cmd.begin_render_pass(rp);
                        ++m_stats.passes;
                        passOpen = true;
                        m_passColor = t.color.get();
                        m_passDepth = depthImage;
                    }
                    ++t.drawSerial;

                    const float devW = static_cast<float>(t.color->get_width()), devH = static_cast<float>(t.color->get_height());
                    const float sx = static_cast<float>(t.sx), sy = static_cast<float>(t.sy);

                    cmd.set_program(m_drawProgram);
                    cmd.set_opaque_state();
                    cmd.set_cull_mode(VK_CULL_MODE_NONE);
                    cmd.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
                    cmd.set_vertex_attrib(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(HwVertex, x));
                    cmd.set_vertex_attrib(1, 0, VK_FORMAT_R32_SFLOAT, offsetof(HwVertex, z));
                    cmd.set_vertex_attrib(2, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(HwVertex, rgba));
                    cmd.set_vertex_attrib(3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(HwVertex, s));
                    cmd.set_vertex_attrib(4, 0, VK_FORMAT_R32_SFLOAT, offsetof(HwVertex, fog));

                    // Depth.
                    if (depthImage)
                    {
                        static const VkCompareOp ops[4] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_ALWAYS,
                                                           VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_GREATER};
                        cmd.set_depth_test(true, zwrite);
                        cmd.set_depth_compare(zte ? ops[ztst] : VK_COMPARE_OP_ALWAYS);
                    }
                    else
                        cmd.set_depth_test(false, false);

                    // Blending.
                    if (s.prim.abe)
                    {
                        VkBlendFactor src, dst;
                        VkBlendOp op;
                        if (!blendFactors(ctx.alpha, src, dst, op))
                        {
                            unsupported("blend equation with a (1 + C) term");
                            src = VK_BLEND_FACTOR_ONE, dst = VK_BLEND_FACTOR_ZERO, op = VK_BLEND_OP_ADD;
                        }
                        cmd.set_blend_enable(true);
                        cmd.set_blend_factors(src, VK_BLEND_FACTOR_ONE, dst, VK_BLEND_FACTOR_ZERO);
                        cmd.set_blend_op(op, VK_BLEND_OP_ADD);
                        const float fix = static_cast<float>((ctx.alpha >> 32) & 0xFFu) / 128.0f;
                        const float constants[4] = {0, 0, 0, std::min(fix, 1.0f)};
                        cmd.set_blend_constants(constants);
                        if (s.pabe)
                            unsupported("PABE");
                    }
                    else
                        cmd.set_blend_enable(false);

                    // Colour write mask: whole channels of FBMSK; a CT24 target keeps its alpha.
                    uint32_t mask = 0;
                    for (uint32_t c = 0; c < 4; ++c)
                    {
                        const uint32_t m = (ctx.frame.fbmsk >> (c * 8)) & 0xFFu;
                        if (m != 0xFFu)
                            mask |= 1u << c;
                        if (m != 0u && m != 0xFFu)
                            unsupported("partial FBMSK");
                    }
                    if (fpsm == GS_PSM_CT24)
                        mask &= 0x7u;

                    // Scissor.
                    VkRect2D sc;
                    sc.offset.x = static_cast<int32_t>(ctx.scissor.x0 * sx);
                    sc.offset.y = static_cast<int32_t>(ctx.scissor.y0 * sy);
                    sc.extent.width = static_cast<uint32_t>(std::max(0, (ctx.scissor.x1 + 1 - ctx.scissor.x0)) * sx);
                    sc.extent.height = static_cast<uint32_t>(std::max(0, (ctx.scissor.y1 + 1 - ctx.scissor.y0)) * sy);
                    cmd.set_scissor(sc);

                    Push p = {};
                    p.posScale[0] = 2.0f * sx / devW;
                    p.posScale[1] = 2.0f * sy / devH;
                    p.posScale[2] = 1.0f / devW - 1.0f;
                    p.posScale[3] = 1.0f / devH - 1.0f;
                    uint32_t flags = 0;
                    const uint64_t tex1 = ctx.tex1;
                    const bool mmag = (tex1 >> 5) & 1u;
                    const uint32_t mmin = (tex1 >> 6) & 7u;
                    const uint32_t wrapS = ctx.clamp & 3u, wrapT = (ctx.clamp >> 2) & 3u;
                    if (s.prim.tme)
                    {
                        flags |= F_TME | (static_cast<uint32_t>(ctx.tex0.tfx & 3u) << 1) | (wrapS << 5) | (wrapT << 7);
                        if (ctx.tex0.tcc)
                            flags |= F_TCC;
                        const bool linear = mmag || mmin == 1u || mmin >= 4u;
                        if (linear)
                            flags |= F_LINEAR;
                        const uint32_t mxl = (tex1 >> 2) & 7u;
                        if (mxl > 0 && b.texture && b.texture->get_create_info().levels > 1)
                            flags |= F_MIP;
                        if (b.fromTarget)
                            flags |= F_REPEAT_IN_SHADER;
                        p.texNorm[0] = static_cast<float>(1u << std::min<uint32_t>(ctx.tex0.tw, 10u));
                        p.texNorm[1] = static_cast<float>(1u << std::min<uint32_t>(ctx.tex0.th, 10u));
                        p.texNorm[2] = 1.0f / extentW;
                        p.texNorm[3] = 1.0f / extentH;
                        p.region[0] = static_cast<float>((ctx.clamp >> 4) & 0x3FFu);
                        p.region[1] = static_cast<float>((ctx.clamp >> 14) & 0x3FFu);
                        p.region[2] = static_cast<float>((ctx.clamp >> 24) & 0x3FFu);
                        p.region[3] = static_cast<float>((ctx.clamp >> 34) & 0x3FFu);
                        p.lod[0] = static_cast<float>(mxl);
                        p.lod[1] = static_cast<float>((tex1 >> 19) & 3u);
                        int k = static_cast<int>((tex1 >> 32) & 0xFFFu);
                        if (k & 0x800)
                            k -= 0x1000;
                        p.lod[2] = k / 16.0f;
                        p.lod[3] = static_cast<float>(tex1 & 1u);
                        p.texOffset[0] = b.texOffsetX;
                        p.texOffset[1] = b.texOffsetY;
                        // Decoded textures and REPEAT: the sampler wraps; everything else clamps.
                        const bool repeatU = !b.fromTarget && wrapS == 0u, repeatV = !b.fromTarget && wrapT == 0u;
                        cmd.set_texture(0, 0, *texView,
                                        sampler(mmag, mmin == 1u || mmin >= 4u, mmin == 3u || mmin == 5u, repeatU, repeatV));
                    }
                    else
                        cmd.set_texture(0, 0, *texView, sampler(false, false, false, false, false));
                    if (s.prim.fge)
                        flags |= F_FGE;
                    p.fogColor[0] = s.fogR;
                    p.fogColor[1] = s.fogG;
                    p.fogColor[2] = s.fogB;
                    p.fogColor[3] = alphaScale;
                    p.texa[0] = s.texa.ta0;
                    p.texa[1] = s.texa.aem ? 1.0f : 0.0f;
                    p.texa[2] = s.texa.ta1;
                    p.texa[3] = ct24Target ? 1.0f : 0.0f;
                    p.mode[0] = flags;
                    p.mode[1] = (test >> 1) & 7u;
                    p.mode[2] = (test >> 4) & 0xFFu;
                    const bool ate = test & 1u;
                    const uint32_t afail = (test >> 12) & 3u;
                    p.mode[3] = ate ? 0u : 2u;
                    if (date)
                        p.mode[0] |= F_DATE | (((test >> 15) & 1u) ? F_DATM : 0u);
                    cmd.set_texture(0, 1, *destView, sampler(false, false, false, false, false));
                    if (ctx.fba & 1u)
                        unsupported("FBA");

                    auto *dst = static_cast<HwVertex *>(cmd.allocate_vertex_data(0, b.vertexCount * sizeof(HwVertex), sizeof(HwVertex)));
                    std::memcpy(dst, vertices.data() + b.firstVertex, b.vertexCount * sizeof(HwVertex));

                    cmd.set_color_write_mask(mask);
                    cmd.push_constants(&p, 0, sizeof(p));
                    cmd.draw(b.vertexCount);

                    // Pixels failing the alpha test still update what AFAIL lets them.
                    if (ate && afail != 0u && p.mode[1] != 1u)
                    {
                        p.mode[3] = 1u;
                        if (afail == 1u) // FB_ONLY
                            cmd.set_depth_test(depthImage != nullptr, false);
                        else if (afail == 2u) // ZB_ONLY
                            mask = 0u;
                        else // RGB_ONLY
                        {
                            mask &= 0x7u;
                            cmd.set_depth_test(depthImage != nullptr, false);
                        }
                        cmd.set_color_write_mask(mask);
                        cmd.push_constants(&p, 0, sizeof(p));
                        cmd.draw(b.vertexCount);
                    }
                }
                endPass(cmd, passOpen);

            }

            // ---------------------------------------------------------------- scanout
            const Vulkan::Image *scanout(Vulkan::CommandBuffer &cmd, const GSPresentationRequest &r)
            {
                const bool en1 = r.pmode & 1u, en2 = (r.pmode >> 1) & 1u;
                const uint64_t dispfb = en1 ? r.dispfb1 : (en2 ? r.dispfb2 : r.dispfb1);
                const uint64_t display = en1 ? r.display1 : (en2 ? r.display2 : r.display1);
                const uint32_t fbp = dispfb & 0x1FFu;
                const uint32_t dbx = (dispfb >> 32) & 0x7FFu, dby = (dispfb >> 43) & 0x7FFu;
                const uint32_t magh = ((display >> 23) & 0xFu) + 1u;
                const uint32_t dw = static_cast<uint32_t>((display >> 32) & 0xFFFu) + 1u;
                uint32_t dh = static_cast<uint32_t>((display >> 44) & 0x7FFu) + 1u;
                auto it = m_targets.find(fbp);
                if (it == m_targets.end() || !it->second.color)
                {
                    static int logged = 0;
                    if (logged++ < 5)
                    {
                        std::cerr << "[hwgs] scanout: no target at fbp " << fbp << " (pmode " << std::hex << r.pmode
                                  << " dispfb " << dispfb << std::dec << "); targets:";
                        for (auto &[f, t] : m_targets)
                            std::cerr << " " << f;
                        std::cerr << "\n";
                    }
                    return nullptr;
                }
                Target &t = it->second;
                const uint32_t w = std::min(dw / magh, t.width - std::min(dbx, t.width));
                if (dh > t.height)
                    dh /= 2u; // a frame-sized DISPLAY over field buffers
                const uint32_t h = std::min(dh, t.height - std::min(dby, t.height));
                if (!w || !h)
                {
                    static int logged = 0;
                    if (logged++ < 5)
                        std::cerr << "[hwgs] scanout: empty display (" << dw << "/" << magh << " x " << dh << " at " << dbx << ","
                                  << dby << " of " << t.width << "x" << t.height << ")\n";
                    return nullptr;
                }
                const uint32_t sx = t.sx, sy = t.sy;
                const uint32_t outW = w * sx, outH = h * sy;

                m_scanoutIndex = (m_scanoutIndex + 1u) % 3u;
                Vulkan::ImageHandle &out = m_scanout[m_scanoutIndex];
                if (!out || out->get_width() != outW || out->get_height() != outH)
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(outW, outH, VK_FORMAT_R8G8B8A8_UNORM);
                    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    out = m_dev->create_image(info);
                }
                cmd.image_barrier(*t.color, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                cmd.image_barrier(*out, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT,
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT);
                cmd.copy_image(*out, *t.color, {0, 0, 0}, {static_cast<int32_t>(dbx * sx), static_cast<int32_t>(dby * sy), 0},
                               {outW, outH, 1}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
                cmd.image_barrier(*out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
                cmd.image_barrier(*t.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                return out.get();
            }

            void readbackCopy(Vulkan::CommandBuffer &cmd, const Vulkan::Image &image)
            {
                const uint32_t w = image.get_width(), h = image.get_height();
                Vulkan::BufferCreateInfo info = {};
                info.size = static_cast<VkDeviceSize>(w) * h * 4u;
                info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                info.domain = Vulkan::BufferDomain::CachedHost;
                if (!m_readback || m_readback->get_create_info().size < info.size)
                    m_readback = m_dev->create_buffer(info);
                cmd.image_barrier(image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                cmd.copy_image_to_buffer(*m_readback, image, 0, {}, {w, h, 1}, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
                cmd.image_barrier(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                cmd.barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                            VK_ACCESS_2_HOST_READ_BIT);
            }

            PresentationFrame readbackFrame(uint32_t w, uint32_t h)
            {
                PresentationFrame frame{};
                const auto *src = static_cast<const uint8_t *>(m_dev->map_host_buffer(*m_readback, Vulkan::MEMORY_ACCESS_READ_BIT));
                frame.width = w;
                frame.height = h;
                frame.stride = w;
                frame.pixels.assign(src, src + static_cast<size_t>(w) * h * 4u);
                for (size_t i = 3; i < frame.pixels.size(); i += 4)
                    frame.pixels[i] = 0xFF;
                m_dev->unmap_host_buffer(*m_readback, Vulkan::MEMORY_ACCESS_READ_BIT);
                return frame;
            }

            std::unique_lock<std::mutex> lockDevice()
            {
                pgsRegisterThread();
                return std::unique_lock<std::mutex>(m_shared->mutex);
            }

            Vulkan::Context m_context; // a device of its own (no presenter) ...
            Vulkan::Device m_ownDevice;
            PgsShared m_ownShared;
            bool m_own = false;
            PgsShared *m_shared = nullptr; // ... or the presenter's
            Vulkan::Device *m_dev = nullptr;
            PgsDeviceInfo m_info;
            Vulkan::Program *m_drawProgram = nullptr;
            Vulkan::ImageHandle m_white;
            std::unordered_map<uint32_t, Vulkan::SamplerHandle> m_samplers;
            std::atomic<uint32_t> m_scale{2};
            std::atomic<bool> m_rescale{false};
            std::atomic<uint32_t> m_yScale{2};

            GSCpuBackend m_cpu; // transfers and readbacks over the shared local memory
            uint8_t *m_vram = nullptr;
            uint32_t m_vramSize = 0;
            GSMem::TexturePageCache m_pageCache;
            std::array<uint16_t, 512> m_clut{};
            std::array<uint32_t, 2> m_clutCbp{};
            std::array<uint32_t, kPages> m_pageVersion{};

            // GS thread -> Present (m_frameMutex).
            std::mutex m_frameMutex;
            std::vector<Batch> m_batches;
            std::vector<HwVertex> m_vertices;
            bool m_open = false;
            uint64_t m_epoch = 1;
            struct PageSpan
            {
                uint32_t first, second, fbw;
            };
            std::unordered_map<uint32_t, PageSpan> m_targetPages; // fbp -> its pages
            struct CachedTexture
            {
                Vulkan::ImageHandle image;
                uint64_t lastUse = 0;
            };
            std::unordered_map<uint64_t, CachedTexture> m_textures;
            std::atomic<uint64_t> m_frame{0};
            // RT_HWGS_STATS=1: per-frame counts, logged every 120 presents.
            struct Stats
            {
                uint64_t batches = 0, vertices = 0, passes = 0, snapshots = 0, decodes = 0, presents = 0, dateBatches = 0;
            } m_stats;
            std::map<size_t, uint64_t> m_breakAt;
            uint64_t m_breakEpoch = 0;
            uint64_t m_lastEviction = 0;
            // GS thread: the state (and resolved texture) of the newest batch.
            Batch m_current;
            bool m_haveCurrent = false;
            uint64_t m_currentEpoch = 0;
            mutable std::mutex m_targetPagesMutex;

            // Recording (device lock).
            std::unordered_map<uint32_t, Target> m_targets;
            std::unordered_map<uint32_t, Vulkan::ImageHandle> m_depths;
            const Vulkan::Image *m_passColor = nullptr, *m_passDepth = nullptr;
            Vulkan::ImageHandle m_scanout[3];
            uint32_t m_scanoutIndex = 0;
            Vulkan::BufferHandle m_readback;
        };
    }

    std::unique_ptr<GSRasterBackend> createHwBackend(const HwOptions &options, std::string &error, PgsControl **control)
    {
        auto backend = std::make_unique<HwBackend>();
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
    std::unique_ptr<GSRasterBackend> createHwBackend(const HwOptions &, std::string &error, PgsControl **)
    {
        error = "built without Vulkan (PS2X_ENABLE_PGS)";
        return nullptr;
    }
}

#endif
