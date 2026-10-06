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
#include "runtime/gs/gs_motion.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "../gs_pgs_shared.h"
#include "../post/post_spirv.h"
#include "hw_spirv.h"
#include "gs_hw_textures.h"

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
#include <thread>
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
            uint32_t motion; // screen motion (two halves, GS pixels; 0 for the HUD and 2D)
        };
        static_assert(sizeof(HwVertex) == 36);

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
            F_REPLACED = 32768u, // a texture-pack image (gs_hw_textures.h)
            F_RECOLOR = 65536u,  // ... recoloured for another palette
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
            // Texture packs: the replacement image, the part of the texture it stands for (texels)
            // and, for another palette, its recolour map.
            Vulkan::ImageHandle replacement;
            float rect[4] = {};
            bool recolor = false;
            float recolorMap[20] = {};
            bool map = false; // the game's map (PgsShared::wantMap)
        };

        struct Target
        {
            uint32_t fbp = 0, fbw = 0, psm = 0;
            uint32_t width = 0, height = 0; // GS pixels
            uint32_t sx = 1, sy = 1;        // device pixels per GS pixel
            Vulkan::ImageHandle color, snapshot;
            Vulkan::ImageHandle motion; // RG16F per-pixel motion, while TAA wants it
            bool motionStale = false;   // scanned out: cleared before the next frame draws into it
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
                m_shared->scanoutRing = 0;
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
                // RT_HWGS_MOTION=1: motion vectors without a presenter asking (headless tests).
                if (const char *e = std::getenv("RT_HWGS_MOTION"); e && *e == '1')
                {
                    m_shared->wantMotion = true;
                    MotionTracker::instance().setEnabled(true);
                }
                m_dev = m_shared->device;
                if (const char *s = std::getenv("RT_GS_SCALE"))
                    m_scale = std::clamp<uint32_t>(static_cast<uint32_t>(std::atoi(s)), 1u, 6u);

                const auto lock = lockDevice();
                Vulkan::ResourceLayout vert = {};
                vert.input_mask = 0x3F;
                vert.output_mask = 0xF;
                vert.push_constant_size = sizeof(Push);
                Vulkan::ResourceLayout frag = {};
                frag.input_mask = 0xF;
                frag.output_mask = 0x3; // colour, motion (unused without the motion attachment)
                frag.push_constant_size = sizeof(Push);
                frag.sets[0].sampled_image_mask = 0x3;
                frag.sets[0].fp_mask = 0x3;
                frag.sets[0].uniform_buffer_mask = 0x4; // the recolour map (texture packs)
                frag.spec_constant_mask = 0x7; // flags, ATST, alpha-test pass (draw.frag)
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
                // The recolour map draws without one bind (unused by their pipelines).
                Vulkan::BufferCreateInfo ubo = {};
                ubo.size = 20 * sizeof(float);
                ubo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                ubo.domain = Vulkan::BufferDomain::Device;
                const float zero[20] = {};
                m_noRecolor = m_dev->create_buffer(ubo, zero);

                m_shared->attached = true;
                m_shared->scanoutRing = 3; // m_scanout
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
                if (std::getenv("RT_GS_SCALE"))
                    return; // the override wins over the setting
                const uint32_t scale = samples >= 16 ? 4 : samples >= 4 ? 2 : 1;
                if (scale != m_scale.load())
                {
                    m_scale = scale;
                    m_rescale = true;
                }
            }
            uint32_t superSampling() const override { return m_scale * m_scale; }
            void setSharpTextures(bool) override {}
            void setTextures(const std::string &dumpDir, const std::string &packDir) override
            {
                m_packs.configure(dumpDir, packDir);
            }
            TexturePackStats texturePackStats() const override
            {
                return {m_packs.packSize(), m_packs.replacedCount()};
            }
            void setAnisotropy(uint32_t level) override { m_anisotropy = std::clamp<uint32_t>(level, 1u, 16u); }
            void setDecodeHook(DecodeHook hook) override
            {
                {
                    std::lock_guard<std::mutex> lock(m_hookMutex);
                    m_decodeHook = std::move(hook);
                }
                m_dropDecoded = true; // decode (and so repaint) everything again
                ++m_epoch;
            }

            // ---------------------------------------------------------------- GSRasterBackend
            bool WantsPrimitives() const override { return true; }
            // The classifier's HUD/scene verdicts (GSPrimitiveBatch::vertexClass) for the UI mask.
            bool WantsVertexSideband() const override
            {
                return m_shared && (m_shared->wantUi.load(std::memory_order_relaxed) || m_shared->wantMotion.load(std::memory_order_relaxed));
            }

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

            // Whether two primitives draw with the same state, as far as record() reads it: the
            // primitive type, shading, FST and XYOFFSET are applied to the vertices; texture state
            // counts only with TME, fog colour with FGE, ALPHA with ABE, the Z buffer when Z is
            // tested or written. Field by field: the frontend's structs carry garbage padding.
            static bool sameDraw(const GSDrawState &a, const GSDrawState &b)
            {
                const GSContext &x = a.context, &y = b.context;
                if (a.prim.tme != b.prim.tme || a.prim.abe != b.prim.abe || a.prim.fge != b.prim.fge)
                    return false;
                if (x.frame.fbp != y.frame.fbp || x.frame.fbw != y.frame.fbw || x.frame.psm != y.frame.psm ||
                    x.frame.fbmsk != y.frame.fbmsk)
                    return false;
                if (x.scissor.x0 != y.scissor.x0 || x.scissor.x1 != y.scissor.x1 || x.scissor.y0 != y.scissor.y0 ||
                    x.scissor.y1 != y.scissor.y1)
                    return false;
                if (x.test != y.test || x.fba != y.fba || x.zbuf.zmask != y.zbuf.zmask || a.pabe != b.pabe)
                    return false;
                const bool zte = (x.test >> 16) & 1u;
                const bool depth = (zte && ((x.test >> 17) & 3u) != 1u) || !x.zbuf.zmask;
                if (depth && (x.zbuf.zbp != y.zbuf.zbp || x.zbuf.psm != y.zbuf.psm))
                    return false;
                if (a.prim.abe && x.alpha != y.alpha)
                    return false;
                if (a.prim.fge && (a.fogR != b.fogR || a.fogG != b.fogG || a.fogB != b.fogB))
                    return false;
                if (a.prim.tme)
                {
                    const GSTex0Reg &t = x.tex0, &u = y.tex0;
                    if (t.tbp0 != u.tbp0 || t.tbw != u.tbw || t.psm != u.psm || t.tw != u.tw || t.th != u.th ||
                        t.tcc != u.tcc || t.tfx != u.tfx || t.cbp != u.cbp || t.cpsm != u.cpsm || t.csm != u.csm ||
                        t.csa != u.csa)
                        return false;
                    if (x.tex1 != y.tex1 || x.clamp != y.clamp || x.miptbp1 != y.miptbp1 || x.miptbp2 != y.miptbp2)
                        return false;
                    if (a.texa.ta0 != b.texa.ta0 || a.texa.aem != b.texa.aem || a.texa.ta1 != b.texa.ta1)
                        return false;
                }
                return true;
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
                m_gsThread.store(std::this_thread::get_id(), std::memory_order_relaxed);

                // A new batch when the state or the texture contents changed (decided and resolved
                // without the frame lock: decoding a texture takes the device lock; m_current is the
                // GS thread's own).
                const uint64_t epoch = m_epoch.load(std::memory_order_relaxed);
                // The frontend numbers its draw states: the same serial is the same state.
                const bool sameSerial = batch.stateSerial != 0 && batch.stateSerial == m_currentSerial;
                const bool fresh = !m_haveCurrent || m_currentEpoch != epoch || (!sameSerial && !sameDraw(m_current.state, state));
                m_currentSerial = batch.stateSerial;
                static const bool stats = std::getenv("RT_HWGS_STATS") != nullptr;
                if (stats && fresh && m_haveCurrent && m_currentEpoch != epoch)
                    ++m_breakEpoch;
                if (fresh)
                {
                    // The previous batch's staged vertices go out first, and it ends.
                    publish(true);
                    m_current = Batch{};
                    m_current.state = state;
                    m_current.map = isMap(state);
                    if (state.prim.tme)
                        resolveTexture(m_current);
                    m_haveCurrent = true;
                    m_currentEpoch = epoch;
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
                    o.motion = v.motion;
                    m_stats.motionVertices += v.motion != 0;
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
                    m_stage.insert(m_stage.end(), quad, quad + 6);
                }
                else
                {
                    // Flat shading takes the colour of the last vertex.
                    const GSVertex &flat = batch.vertices[2];
                    for (int i = 0; i < 3; ++i)
                        m_stage.push_back(convert(batch.vertices[i], state.prim.iip ? batch.vertices[i] : flat));
                }
                // The UI mask: the HUD's triangles, drawn into a mask at Present.
                if (batch.vertexClass == 1u)
                {
                    const size_t n = type == GS_PRIM_SPRITE ? 6u : 3u;
                    for (size_t i = m_stage.size() - n; i < m_stage.size(); ++i)
                        m_uiStage.push_back({m_stage[i].x, m_stage[i].y});
                }
                if (m_stage.size() >= 1536u)
                    publish(false);
            }

            // The game's map: every course-map and town-minimap draw has TEST_2 = 0x3380B (alpha test
            // GEQUAL 0x80 with RGB_ONLY, Z test ALWAYS; a template at ELF 0x2A43B0, nothing else uses
            // it), and in towns its bezel is three sprites of one palette bank ([map] in Road Trip's
            // config/game_state.toml).
            static bool isMap(const GSDrawState &s)
            {
                if (s.prim.ctxt && (s.context.test & 0x7FFFFu) == 0x3380Bu)
                    return true;
                const GSTex0Reg &t = s.context.tex0;
                return !s.prim.ctxt && s.prim.type == GS_PRIM_SPRITE && s.prim.tme && t.tbp0 == 7828u && t.psm == 0x14u &&
                       t.cbp == 7892u && t.csa == 10u;
            }

            // GS thread: hands the staged vertices of the current batch to the shared lists (one lock
            // per batch change or 1536 vertices, not per primitive); `end` closes the batch. Opening
            // a batch for another frame buffer (a buffer flip) or after a long run records what's pending.
            void publish(bool end)
            {
                bool flush = false;
                {
                    std::lock_guard<std::mutex> lock(m_frameMutex);
                    if (!m_stage.empty())
                    {
                        if (!m_open)
                        {
                            flush = !m_batches.empty() &&
                                    (m_batches.back().state.context.frame.fbp != m_current.state.context.frame.fbp ||
                                     m_vertices.size() > 200000u || m_batches.size() > 4000u);
                            Batch b = m_current;
                            b.firstVertex = static_cast<uint32_t>(m_vertices.size());
                            b.vertexCount = 0;
                            m_batches.push_back(std::move(b));
                            m_open = true;
                        }
                        m_vertices.insert(m_vertices.end(), m_stage.begin(), m_stage.end());
                        m_uiVerts.insert(m_uiVerts.end(), m_uiStage.begin(), m_uiStage.end());
                        m_uiStage.clear();
                        m_batches.back().vertexCount += static_cast<uint32_t>(m_stage.size());
                        m_stage.clear();
                    }
                    if (end && m_open)
                        closeBatch();
                }
                if (flush)
                    flushPending();
            }

            // Records and submits what the GS thread has queued (the device lock before the frame lock,
            // as in Present, so batches are recorded in order).
            void flushPending()
            {
                const auto device = lockDevice();
                // The pending lists go to the recorder; the GS thread continues in the spare ones
                // (cleared, capacity kept: no reallocation every frame).
                std::vector<Batch> &batches = m_recordBatches;
                std::vector<HwVertex> &vertices = m_recordVertices;
                {
                    std::lock_guard<std::mutex> lock(m_frameMutex);
                    batches.swap(m_batches);
                    vertices.swap(m_vertices);
                    if (m_open && !batches.empty())
                    {
                        // The open batch continues: move it and its vertices to the fresh lists.
                        Batch open = std::move(batches.back());
                        batches.pop_back();
                        m_vertices.assign(vertices.begin() + open.firstVertex, vertices.end());
                        vertices.resize(open.firstVertex);
                        open.firstVertex = 0;
                        m_batches.push_back(std::move(open));
                    }
                }
                if (batches.empty())
                    return;
                auto cmd = m_dev->request_command_buffer();
                record(*cmd, batches, vertices);
                batches.clear();
                vertices.clear();
                submitRecorded(cmd);
                // A device of its own: a frame context per flush (waits for the one a few flushes
                // back), so the GPU can't fall behind and resources are recycled.
                if (m_own)
                    m_dev->next_frame_context();
            }

            void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override
            {
                // The game reloads the CLUT with every TEX0, mostly from the same place with the same
                // colours: skip the reload when the source and its page are unchanged, and only a
                // real change starts new batches.
                const uint64_t source = mix(mix(mix(0, tex0.cbp | (static_cast<uint64_t>(tex0.cpsm) << 16) |
                                                         (static_cast<uint64_t>(tex0.csm) << 24) |
                                                         (static_cast<uint64_t>(tex0.csa) << 32) |
                                                         (static_cast<uint64_t>(tex0.psm) << 40) |
                                                         (static_cast<uint64_t>(tex0.cld) << 48)),
                                                texclut.cbw | (texclut.cou << 8) | (static_cast<uint32_t>(texclut.cov) << 16)),
                                            m_pageVersion[(tex0.cbp / 32u) % kPages]);
                if (source == m_lastClutSource && tex0.cld != 0u)
                    return;
                m_lastClutSource = source;
                const auto before = m_clut;
                loadClut(tex0, texclut);
                if (m_clut == before)
                    return;
                publish(true);
                ++m_epoch;
            }

            // Writes into local memory: the destination pages are hashed before and after, and only a
            // real change counts (new page versions, new batches, textures decoded again). The game
            // uploads its palettes and some textures again every frame, mostly with the same bytes.
            void BeginTransfer(const GSTransferCommand &command) override
            {
                finishUpload(); // an upload cut short by the next transfer
                // Texture packs: the images the game uploads, by base address (a texture is named
                // by the whole upload it is drawn from, as paraLLEl-GS does).
                if (command.direction == 0u && command.trxpos.dsax == 0u && command.trxpos.dsay == 0u)
                    m_uploads[command.bitbltbuf.dbp] = {command.bitbltbuf.dpsm, command.trxreg.rrw, command.trxreg.rrh};
                else if (command.direction == 2u)
                    m_uploads.erase(command.bitbltbuf.dbp);
                if (command.direction == 0u || command.direction == 2u)
                {
                    uint32_t first, last;
                    pageRange(command.bitbltbuf.dbp, command.bitbltbuf.dbw, command.bitbltbuf.dpsm, command.trxpos.dsax,
                              command.trxpos.dsay, command.trxreg.rrw, command.trxreg.rrh, first, last);
                    m_upload = {true, first, std::min(last, first + kPages - 1u), hashPages(first, last)};
                    if (command.direction == 2u)
                    {
                        uint32_t sfirst, slast;
                        pageRange(command.bitbltbuf.sbp, command.bitbltbuf.sbw, command.bitbltbuf.spsm, command.trxpos.ssax,
                                  command.trxpos.ssay, command.trxreg.rrw, command.trxreg.rrh, sfirst, slast);
                        if (findTargetForPages(sfirst, slast))
                            unsupported("local-to-local copy out of a render target");
                    }
                }
                else if (command.direction == 1u)
                    unsupported("local-to-host readback (render targets aren't written back)");
                m_cpu.BeginTransfer(command);
                m_pageCache.Invalidate();
                if (command.direction == 2u) // the CPU GS copies at once
                    finishUpload();
            }

            void UploadImage(const uint8_t *data, uint32_t sizeBytes) override
            {
                m_cpu.UploadImage(data, sizeBytes);
                m_pageCache.Invalidate();
                const GSTransferSnapshot t = m_cpu.GetTransferSnapshot();
                if (m_upload.active && t.copiedPixels >= t.totalPixels)
                    finishUpload();
            }

            uint64_t hashPages(uint32_t first, uint32_t last) const
            {
                if (!m_vram)
                    return 0;
                uint64_t h = 0x9E3779B97F4A7C15ull;
                for (uint32_t p = first; p <= last && p < first + kPages; ++p)
                {
                    const auto *w = reinterpret_cast<const uint64_t *>(m_vram + (p % kPages) * 8192u);
                    for (uint32_t i = 0; i < 1024u; ++i)
                        h = (h ^ w[i]) * 0x100000001B3ull + (h >> 29);
                }
                return h;
            }

            void finishUpload()
            {
                if (!m_upload.active)
                    return;
                m_upload.active = false;
                if (hashPages(m_upload.first, m_upload.last) == m_upload.hash)
                    return;
                publish(true);
                std::lock_guard<std::mutex> lock(m_frameMutex);
                for (uint32_t p = m_upload.first; p <= m_upload.last; ++p)
                    ++m_pageVersion[p % kPages];
                ++m_epoch;
            }

            // FINISH (the end of a frame's drawing, GS thread) publishes the staged vertices; the other
            // callers (presentation) are on other threads and leave them to the GS thread.
            void Flush() override
            {
                if (std::this_thread::get_id() == m_gsThread.load(std::memory_order_relaxed))
                    publish(false);
            }
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
                    std::vector<Batch> &batches = m_recordBatches;
                    std::vector<HwVertex> &vertices = m_recordVertices;
                    {
                        std::lock_guard<std::mutex> frameLock(m_frameMutex);
                        if (m_open)
                            closeBatch();
                        batches.swap(m_batches);
                        vertices.swap(m_vertices);
                        m_uiDraw.swap(m_uiVerts);
                        m_uiVerts.clear();
                    }
                    if (m_rescale.exchange(false))
                        m_targets.clear(), m_depths.clear();
                    servicePacks();
                    m_yScale = request.progressiveFields ? 2u : 1u;
                    m_motionOn = m_shared && m_shared->wantMotion.load(std::memory_order_relaxed);
                    auto cmd = m_dev->request_command_buffer();
                    // The presenter may still be sampling an older scanout image.
                    cmd->barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
                    record(*cmd, batches, vertices);
                    batches.clear();
                    vertices.clear();
                    const Vulkan::Image *scan = scanout(*cmd, request);
                    publishMap(*cmd);
                    if (scan && m_shared && m_shared->wantUi.load(std::memory_order_relaxed))
                        drawUiMask(*cmd);
                    m_uiDraw.clear();
                    if (scan)
                    {
                        outW = scan->get_width();
                        outH = scan->get_height();
                        m_shared->scanout = m_scanout[m_scanoutIndex];
                        m_shared->motion = m_motionOut;
                        ++m_shared->presentSerial;
                    }
                    if (scan && request.readback)
                        readbackCopy(*cmd, *scan);
                    {
                        Vulkan::Fence used;
                        submitRecorded(cmd, &used);
                        if (request.readback && scan)
                            fence = used;
                    }
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
                    std::fprintf(stderr, "[hwgs] vertices with motion: %.0f per present (motion %s)\n", m_stats.motionVertices.load() / n,
                                 m_motionOn ? "on" : "off");
                    m_stats.batches = m_stats.vertices = m_stats.passes = m_stats.snapshots = m_stats.decodes = m_stats.presents = 0;
                    m_stats.dateBatches = 0;
                    m_stats.motionVertices = 0;
                    std::fprintf(stderr, "[hwgs] batches broken by texture/CLUT changes: %.0f per present\n", m_breakEpoch / n);
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
                // Headless (no presents for a while): settings, results and new pack images are
                // serviced here, as paraLLEl-GS does on its GIF path.
                if (m_packs.serviceDue())
                {
                    const auto lock = lockDevice();
                    servicePacks();
                }
                b.texture = decodedTexture(b.state);
                if (m_packs.active())
                    resolvePack(b);
            }

            // ---------------------------------------------------------------- texture dumps and packs
            // Device lock held.
            void servicePacks()
            {
                if (m_packs.service(*m_dev))
                {
                    // Another pack, or evicted images: every texture goes to the tools again (the
                    // decoded textures stay), and new batches look their replacements up again.
                    m_resubmit = true;
                    ++m_epoch;
                }
            }

            // The part of a texture paraLLEl-GS decodes along one axis (gs_util.cpp
            // compute_effective_texture_extent): dumps and packs name textures by those pixels.
            static void effectiveExtent(uint32_t size, uint32_t mode, uint32_t lo, uint32_t hi, uint32_t levels,
                                        uint32_t &base, uint32_t &extent)
            {
                base = 0;
                extent = size;
                if (mode == 2u) // REGION_CLAMP
                {
                    lo = std::min(lo, hi);
                    extent = std::max(hi, lo) - lo + 1u;
                    if (levels > 1u)
                    {
                        const uint32_t mask = (1u << (levels - 1u)) - 1u;
                        extent += lo & mask;
                        lo &= ~mask;
                        extent = (extent + mask) & ~mask;
                    }
                    base = lo;
                }
                else if (mode == 3u) // REGION_REPEAT
                {
                    if (lo == 0u)
                    {
                        extent = 1u;
                        base = hi;
                    }
                    else
                    {
                        const uint32_t mskMsb = 31u - static_cast<uint32_t>(__builtin_clz(lo));
                        const uint32_t fixLsb = hi ? static_cast<uint32_t>(__builtin_ctz(hi)) : 32u;
                        if (fixLsb > mskMsb)
                        {
                            extent = std::min<uint32_t>(1u << (mskMsb + 1u), extent);
                            base = hi;
                        }
                    }
                }
            }

            struct TexRect
            {
                uint32_t x = 0, y = 0, w = 0, h = 0;
            };
            static TexRect effectiveRect(const GSContext &ctx, uint64_t clamp)
            {
                const GSTex0Reg &tex = ctx.tex0;
                const uint32_t twl = std::min<uint32_t>(tex.tw, 10u), thl = std::min<uint32_t>(tex.th, 10u);
                const uint32_t mmin = (ctx.tex1 >> 6) & 7u;
                uint32_t levels = 1u;
                if (mmin > 1u && mmin <= 5u) // a mipmapping filter: MXL counts
                    levels = std::min<uint32_t>(std::min<uint32_t>(((ctx.tex1 >> 2) & 7u) + 1u, std::min(twl, thl) + 1u), 7u);
                TexRect r;
                effectiveExtent(1u << twl, clamp & 3u, (clamp >> 4) & 0x3FFu, (clamp >> 14) & 0x3FFu, levels, r.x, r.w);
                effectiveExtent(1u << thl, (clamp >> 2) & 3u, (clamp >> 24) & 0x3FFu, (clamp >> 34) & 0x3FFu, levels, r.y, r.h);
                return r;
            }

            // A draw's texture as the texture tools see it: the part paraLLEl-GS would decode (the whole
            // upload, for a sprite cut out of an atlas with REGION_CLAMP), submitted once per content,
            // and the pack's replacement for it if there is one. GS thread.
            void resolvePack(Batch &b)
            {
                const GSContext &ctx = b.state.context;
                const GSTex0Reg &tex = ctx.tex0;
                TexRect rect = effectiveRect(ctx, ctx.clamp);
                const uint32_t wms = ctx.clamp & 3u, wmt = (ctx.clamp >> 2) & 3u;
                if (wms == 2u && wmt == 2u)
                {
                    auto up = m_uploads.find(tex.tbp0);
                    if (up != m_uploads.end() && up->second.psm == tex.psm && up->second.width && up->second.height &&
                        rect.x + rect.w <= up->second.width && rect.y + rect.h <= up->second.height &&
                        (rect.w != up->second.width || rect.h != up->second.height || rect.x || rect.y))
                    {
                        uint32_t first, last;
                        pageRange(tex.tbp0, tex.tbw, tex.psm, 0, 0, up->second.width, up->second.height, first, last);
                        if (!findTargetForPages(first, last))
                        {
                            // REGION_CLAMP over the whole upload: MINU/MINV 0, MAXU/MAXV its last texel.
                            const uint64_t whole = 2u | (2u << 2) | (static_cast<uint64_t>((up->second.width - 1u) & 0x3FFu) << 14) |
                                                   (static_cast<uint64_t>((up->second.height - 1u) & 0x3FFu) << 34);
                            rect = effectiveRect(ctx, whole);
                        }
                    }
                }
                if (!rect.w || !rect.h)
                    return;

                // Stable key: the description without the palette's colours (Road Trip reloads
                // palettes every frame, and a fade keeps it); the cache key adds contents and palette.
                uint64_t stable = mix(0x7E57AB1Eull, tex.tbp0 | (static_cast<uint64_t>(tex.tbw) << 14) |
                                                     (static_cast<uint64_t>(tex.psm) << 20) | (static_cast<uint64_t>(tex.cpsm) << 26) |
                                                     (static_cast<uint64_t>(tex.cbp) << 32) | (static_cast<uint64_t>(tex.csa) << 46) |
                                                     (static_cast<uint64_t>(tex.csm) << 51));
                stable = mix(stable, rect.x | (rect.y << 10) | (static_cast<uint64_t>(rect.w) << 20) | (static_cast<uint64_t>(rect.h) << 32));
                stable = mix(stable, b.state.texa.ta0 | (b.state.texa.aem << 8) | (b.state.texa.ta1 << 16));
                uint64_t key = stable;
                uint32_t first, last;
                pageRange(tex.tbp0, tex.tbw, tex.psm, rect.x, rect.y, rect.w, rect.h, first, last);
                for (uint32_t p = first; p <= last && p < first + kPages; ++p)
                    key = mix(key, (static_cast<uint64_t>(p) << 32) | m_pageVersion[p % kPages]);
                if (isIndexed(tex.psm))
                {
                    const uint32_t n = isFourBit(tex.psm) ? 16u : 256u;
                    for (uint32_t i = 0; i < n; ++i)
                        key = mix(key, clutColor(b.state, i));
                }

                if (m_resubmit.exchange(false))
                    m_submitted.clear();
                if (m_submitted.insert(key).second)
                {
                    std::vector<uint8_t> rgba(static_cast<size_t>(rect.w) * rect.h * 4u);
                    uint32_t *out = reinterpret_cast<uint32_t *>(rgba.data());
                    for (uint32_t y = 0; y < rect.h; ++y)
                        for (uint32_t x = 0; x < rect.w; ++x)
                            out[static_cast<size_t>(y) * rect.w + x] = texel(b.state, tex.psm, tex.tbp0, tex.tbw, rect.x + x, rect.y + y);
                    m_packs.submit(key, stable, rect.w, rect.h, tex.psm, std::move(rgba));
                }

                HwTexturePacks::Bound bound = m_packs.lookup(key, stable);
                if (!bound.image)
                    return;
                b.replacement = std::move(bound.image);
                b.rect[0] = static_cast<float>(rect.x);
                b.rect[1] = static_cast<float>(rect.y);
                b.rect[2] = static_cast<float>(rect.w);
                b.rect[3] = static_cast<float>(rect.h);
                b.recolor = bound.recolor;
                if (bound.recolor)
                    std::copy(std::begin(bound.transform), std::end(bound.transform), b.recolorMap);
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
                if (m_dropDecoded.exchange(false))
                    m_textures.clear();
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
                {
                    // The app may repaint the top level (PgsControl::setDecodeHook).
                    std::lock_guard<std::mutex> lock(m_hookMutex);
                    if (m_decodeHook)
                    {
                        DecodedTexture d;
                        d.psm = tex.psm;
                        d.width = 1u << tw;
                        d.height = 1u << th;
                        d.rgba = data[0].data();
                        const uint32_t psm = tex.psm, tbp = lv[0].tbp, tbw = lv[0].tbw;
                        d.index = [this, psm, tbp, tbw](uint32_t x, uint32_t y)
                        { return GSMem::ReadTexture(m_pageCache, m_vram, psm, tbp, tbw, x, y) & 0xFFu; };
                        m_decodeHook(d);
                    }
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

            // A target's motion image (RG16F, the colour's size), cleared to no motion.
            void createMotion(Target &t, Vulkan::CommandBuffer &cmd)
            {
                auto info = Vulkan::ImageCreateInfo::render_target(t.color->get_width(), t.color->get_height(), VK_FORMAT_R16G16_SFLOAT);
                info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                t.motion = m_dev->create_image(info);
                clearMotion(*t.motion, cmd, VK_IMAGE_LAYOUT_UNDEFINED);
            }

            void clearMotion(const Vulkan::Image &image, Vulkan::CommandBuffer &cmd, VkImageLayout from)
            {
                cmd.image_barrier(image, from, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT);
                VkClearValue zero = {};
                cmd.clear_image(image, zero);
                cmd.image_barrier(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            }

            // The map layer for this frame: an opaque image like the frame buffer's, filled with the
            // second screen's map background before the frame's first map draw (two, swapped at
            // Present, so the presenter samples a finished one).
            Target &mapLayer(const Target &frame, Vulkan::CommandBuffer &cmd, bool &passOpen)
            {
                // Drawn at 4x whatever the game's render scale, so the second screen can enlarge it
                // sharply (its few hundred triangles cost nothing).
                constexpr uint32_t kMapScale = 4;
                const uint32_t sx = std::max(kMapScale, frame.sx), sy = std::max(kMapScale * m_yScale, frame.sy);
                Target &m = m_mapLayers[m_mapIndex];
                const bool fresh = !m.color || m.width != frame.width || m.height != frame.height || m.sx != sx || m.sy != sy;
                if (fresh || !m_mapDrawn)
                {
                    endPass(cmd, passOpen);
                    if (fresh)
                    {
                        m = Target{};
                        m.fbp = ~0u;
                        m.width = frame.width;
                        m.height = frame.height;
                        m.sx = sx;
                        m.sy = sy;
                        auto info = Vulkan::ImageCreateInfo::render_target(frame.width * sx, frame.height * sy, VK_FORMAT_R8G8B8A8_UNORM);
                        info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                        info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                        m.color = m_dev->create_image(info);
                    }
                    cmd.image_barrier(*m.color, VK_IMAGE_LAYOUT_UNDEFINED, // cleared: its old contents don't matter
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                      VK_ACCESS_2_TRANSFER_WRITE_BIT);
                    VkClearValue bg = {};
                    bg.color.float32[0] = 0x0B / 255.0f; // the second screen's navy
                    bg.color.float32[1] = 0x12 / 255.0f;
                    bg.color.float32[2] = 0x40 / 255.0f;
                    bg.color.float32[3] = 1.0f;
                    cmd.clear_image(*m.color, bg);
                    cmd.image_barrier(*m.color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                    m_mapDrawn = true;
                    m_mapBox[0] = m_mapBox[1] = 1e9f;
                    m_mapBox[2] = m_mapBox[3] = -1e9f;
                }
                return m;
            }

            // The part of the frame the map covers (inside its scissor: the town map's vertices reach
            // far outside it). The town bezel is drawn into the layer but not counted: the second
            // screen shows the map itself.
            void noteMapBox(const Batch &b, const std::vector<HwVertex> &vertices)
            {
                // The map itself, not what moves on it: the track and streets are flat-shaded, the
                // cars' markers Gouraud (a marker at the track's edge would move the box).
                if (!b.state.prim.ctxt || b.state.prim.iip)
                    return;
                const GSScissorReg &sc = b.state.context.scissor;
                float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
                for (uint32_t i = b.firstVertex; i < b.firstVertex + b.vertexCount && i < vertices.size(); ++i)
                {
                    x0 = std::min(x0, vertices[i].x), x1 = std::max(x1, vertices[i].x);
                    y0 = std::min(y0, vertices[i].y), y1 = std::max(y1, vertices[i].y);
                }
                x0 = std::max(x0, static_cast<float>(sc.x0)), y0 = std::max(y0, static_cast<float>(sc.y0));
                x1 = std::min(x1, static_cast<float>(sc.x1 + 1)), y1 = std::min(y1, static_cast<float>(sc.y1 + 1));
                if (x1 <= x0 || y1 <= y0)
                    return;
                m_mapBox[0] = std::min(m_mapBox[0], x0), m_mapBox[1] = std::min(m_mapBox[1], y0);
                m_mapBox[2] = std::max(m_mapBox[2], x1), m_mapBox[3] = std::max(m_mapBox[3], y1);
            }

            // At Present: hands this frame's map (if one was drawn) to the second screen.
            void publishMap(Vulkan::CommandBuffer &cmd)
            {
                if (!m_shared)
                    return;
                const Target &m = m_mapLayers[m_mapIndex];
                if (m_mapDrawn && m.color && m_mapBox[2] > m_mapBox[0])
                {
                    // Steady while the same map is shown: the box only grows (frames where part of the
                    // map is clipped or drawn later don't shrink it), and starts again once the map
                    // has been gone a while.
                    if (m_mapStable[2] > m_mapStable[0])
                        for (int i = 0; i < 2; ++i)
                        {
                            m_mapBox[i] = std::min(m_mapBox[i], m_mapStable[i]);
                            m_mapBox[i + 2] = std::max(m_mapBox[i + 2], m_mapStable[i + 2]);
                        }
                    std::memcpy(m_mapStable, m_mapBox, sizeof(m_mapBox));
                    // Sampled by the presenter from here on (back to an attachment when reused).
                    cmd.image_barrier(*m.color, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                    m_shared->map = m.color;
                    m_shared->mapUv[0] = m_mapBox[0] / m.width;
                    m_shared->mapUv[1] = m_mapBox[1] / m.height;
                    m_shared->mapUv[2] = m_mapBox[2] / m.width;
                    m_shared->mapUv[3] = m_mapBox[3] / m.height;
                    // GS pixels on a 4:3 TV: a 640x224 field fills it, so a pixel is 4/3 * 224/640 as
                    // wide as it is tall... per field line.
                    const float pixelAspect = (4.0f / 3.0f) * 224.0f / 640.0f;
                    m_shared->mapAspect = (m_mapBox[2] - m_mapBox[0]) * pixelAspect / (m_mapBox[3] - m_mapBox[1]);
                    m_mapIndex ^= 1u;
                    m_mapMissed = 0;
                }
                else if (++m_mapMissed > 10)
                {
                    m_shared->map.reset(); // no map for a while: none to show
                    m_mapStable[0] = m_mapStable[2] = 0; // the next map (another course or town) starts afresh
                }
                m_mapDrawn = false;
            }

            void endPass(Vulkan::CommandBuffer &cmd, bool &passOpen)
            {
                if (passOpen)
                {
                    cmd.end_render_pass();
                    passOpen = false;
                    m_passColor = nullptr;
                    m_passDepth = nullptr;
                    m_passMotion = nullptr;
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

            // Texture-pack images: trilinear with the anisotropy setting, each axis clamped or repeated.
            const Vulkan::Sampler &anisoSampler(bool clampU, bool clampV)
            {
                const uint32_t level = m_anisotropy.load();
                if (level != m_samplerAnisotropy)
                {
                    for (auto &s : m_anisoSamplers)
                        s.reset();
                    m_samplerAnisotropy = level;
                }
                auto &s = m_anisoSamplers[clampU | (clampV << 1)];
                if (!s)
                {
                    Vulkan::SamplerCreateInfo info = {};
                    info.mag_filter = VK_FILTER_LINEAR;
                    info.min_filter = VK_FILTER_LINEAR;
                    info.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                    info.address_mode_u = clampU ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
                    info.address_mode_v = clampV ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
                    info.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                    const float maxAniso = m_dev->get_gpu_properties().limits.maxSamplerAnisotropy;
                    info.anisotropy_enable = level > 1u && m_dev->get_device_features().enabled_features.samplerAnisotropy;
                    info.max_anisotropy = std::min(static_cast<float>(level), maxAniso);
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

            // The recording's vertices, uploaded once into a slot of a ring of persistent buffers
            // (Granite's vertex pool has 4 KB blocks: most batches got a buffer of their own, created
            // and destroyed every frame). A slot is reused once its last submit has finished.
            const Vulkan::Buffer *uploadVertices(const std::vector<HwVertex> &vertices)
            {
                if (vertices.empty())
                    return nullptr;
                m_vboSlot = (m_vboSlot + 1u) % kVboRing;
                VboSlot &slot = m_vbo[m_vboSlot];
                if (slot.fence)
                {
                    slot.fence->wait();
                    slot.fence.reset();
                }
                const size_t bytes = vertices.size() * sizeof(HwVertex);
                if (!slot.buffer || slot.buffer->get_create_info().size < bytes)
                {
                    Vulkan::BufferCreateInfo info = {};
                    info.size = std::max<size_t>(bytes + bytes / 2, 1u << 20);
                    info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
                    info.domain = Vulkan::BufferDomain::Host;
                    slot.buffer = m_dev->create_buffer(info);
                }
                void *dst = m_dev->map_host_buffer(*slot.buffer, Vulkan::MEMORY_ACCESS_WRITE_BIT);
                std::memcpy(dst, vertices.data(), bytes);
                m_dev->unmap_host_buffer(*slot.buffer, Vulkan::MEMORY_ACCESS_WRITE_BIT);
                return slot.buffer.get();
            }

            // The fence of the submit that used the current vertex slot.
            void submitRecorded(Vulkan::CommandBufferHandle &cmd, Vulkan::Fence *extra = nullptr)
            {
                Vulkan::Fence fence;
                m_dev->submit(cmd, &fence);
                if (m_vboUsed)
                    m_vbo[m_vboSlot].fence = fence;
                m_vboUsed = false;
                if (extra)
                    *extra = fence;
            }

            void record(Vulkan::CommandBuffer &cmd, const std::vector<Batch> &batches, const std::vector<HwVertex> &vertices)
            {
                ++m_frame;
                bool passOpen = false;
                const Vulkan::Buffer *vbo = uploadVertices(vertices);
                m_vboUsed = vbo != nullptr;
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
                    Target &frameTarget =
                        target(ctx.frame.fbp, ctx.frame.fbw, fpsm, static_cast<uint32_t>(ctx.scissor.y1) + 1u, cmd, passOpen);
                    const bool toMap = b.map && m_shared && m_shared->wantMap.load(std::memory_order_relaxed);
                    Target &t = toMap ? mapLayer(frameTarget, cmd, passOpen) : frameTarget;
                    if (toMap)
                        noteMapBox(b, vertices);

                    const bool zte = (ctx.test >> 16) & 1u;
                    const uint32_t ztst = (ctx.test >> 17) & 3u;
                    const bool zwrite = !ctx.zbuf.zmask && !toMap;
                    const bool useDepth = !toMap && ((zte && ztst != 1u) || zwrite);
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
                        else if (b.replacement)
                        {
                            // A pack image stands for the texture's rect at any size.
                            texView = &b.replacement->get_view();
                            extentW = b.rect[2];
                            extentH = b.rect[3];
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

                    // Motion for TAA: a second attachment on the frame's targets while it is wanted.
                    Vulkan::Image *motionImage = nullptr;
                    if (m_motionOn && !toMap)
                    {
                        if (!t.motion || t.motion->get_width() != t.color->get_width() || t.motion->get_height() != t.color->get_height())
                        {
                            endPass(cmd, passOpen);
                            createMotion(t, cmd);
                        }
                        if (t.motionStale)
                        {
                            endPass(cmd, passOpen);
                            clearMotion(*t.motion, cmd, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
                            t.motionStale = false;
                        }
                        motionImage = t.motion.get();
                    }
                    if (passOpen && (m_passColor != t.color.get() || m_passDepth != depthImage || m_passMotion != motionImage))
                        endPass(cmd, passOpen);
                    if (!passOpen)
                    {
                        Vulkan::RenderPassInfo rp = {};
                        rp.num_color_attachments = motionImage ? 2 : 1;
                        rp.color_attachments[0] = &t.color->get_view();
                        if (motionImage)
                            rp.color_attachments[1] = &motionImage->get_view();
                        rp.load_attachments = motionImage ? 3 : 1;
                        rp.store_attachments = motionImage ? 3 : 1;
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
                        m_passMotion = motionImage;
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
                    cmd.set_vertex_attrib(5, 0, VK_FORMAT_R16G16_SFLOAT, offsetof(HwVertex, motion));

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
                    if (fpsm == GS_PSM_CT24 || toMap)
                        mask &= 0x7u; // (the map layer stays opaque)

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
                        if (mxl > 0 && !b.replacement && b.texture && b.texture->get_create_info().levels > 1)
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
                        if (b.replacement)
                        {
                            // Trilinear/anisotropic over the pack image's mips, clamped axes to the edge.
                            flags |= F_REPLACED | (b.recolor ? F_RECOLOR : 0u);
                            p.texOffset[0] = -b.rect[0];
                            p.texOffset[1] = -b.rect[1];
                            cmd.set_texture(0, 0, *texView, anisoSampler(wrapS == 1u || wrapS == 2u, wrapT == 1u || wrapT == 2u));
                            if (b.recolor)
                                std::memcpy(cmd.allocate_constant_data(0, 2, sizeof(b.recolorMap)), b.recolorMap, sizeof(b.recolorMap));
                        }
                        else
                        {
                            // Decoded textures and REPEAT: the sampler wraps; everything else clamps.
                            const bool repeatU = !b.fromTarget && wrapS == 0u, repeatV = !b.fromTarget && wrapT == 0u;
                            cmd.set_texture(0, 0, *texView,
                                            sampler(mmag, mmin == 1u || mmin >= 4u, mmin == 3u || mmin == 5u, repeatU, repeatV));
                        }
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
                    if (!(flags & F_RECOLOR))
                        cmd.set_uniform_buffer(0, 2, *m_noRecolor);
                    if (ctx.fba & 1u)
                        unsupported("FBA");

                    cmd.set_vertex_binding(0, *vbo, 0, sizeof(HwVertex));
                    // Motion from opaque draws only (blending would blend it); the HUD writes its zero.
                    cmd.set_color_write_mask(mask | (motionImage && !s.prim.abe ? 0x30u : 0u));
                    cmd.set_specialization_constant_mask(0x7);
                    cmd.set_specialization_constant(0, p.mode[0]);
                    cmd.set_specialization_constant(1, p.mode[1]);
                    cmd.set_specialization_constant(2, p.mode[3]);
                    cmd.push_constants(&p, 0, sizeof(p));
                    cmd.draw(b.vertexCount, 1, b.firstVertex);

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
                        cmd.set_specialization_constant(2, p.mode[3]);
                        cmd.push_constants(&p, 0, sizeof(p));
                        cmd.draw(b.vertexCount, 1, b.firstVertex);
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
                m_scanGeom = {dbx, dby, sx, sy, outW, outH};

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
                // Motion: the same part, for TAA (the scanout's size, GS pixels), then cleared for the
                // next frame.
                m_motionOut.reset();
                if (m_motionOn && t.motion)
                {
                    Vulkan::ImageHandle &mo = m_motionScan[m_scanoutIndex];
                    if (!mo || mo->get_width() != outW || mo->get_height() != outH)
                    {
                        auto info = Vulkan::ImageCreateInfo::render_target(outW, outH, VK_FORMAT_R16G16_SFLOAT);
                        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                        info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                        mo = m_dev->create_image(info);
                    }
                    cmd.image_barrier(*t.motion, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                    cmd.image_barrier(*mo, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                      0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                    cmd.copy_image(*mo, *t.motion, {0, 0, 0}, {static_cast<int32_t>(dbx * sx), static_cast<int32_t>(dby * sy), 0},
                                   {outW, outH, 1}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
                    cmd.image_barrier(*mo, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                    cmd.image_barrier(*t.motion, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                      VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                    t.motionStale = true; // (a repeated present shows this frame's motion again)
                    m_motionOut = mo;
                }
                return out.get();
            }

            // The UI mask (PgsShared::ui, the scanout's size): 1 where the frame's HUD and 2D screens
            // drew (the classifier's UI class), so post-processing leaves them as drawn.
            void drawUiMask(Vulkan::CommandBuffer &cmd)
            {
                const ScanGeom &g = m_scanGeom;
                Vulkan::ImageHandle &mask = m_uiMask[m_scanoutIndex];
                if (!mask || mask->get_width() != g.outW || mask->get_height() != g.outH)
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(g.outW, g.outH, VK_FORMAT_R8G8B8A8_UNORM);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    mask = m_dev->create_image(info);
                }
                // Positions relative to the scanned-out part of the frame; white, no texture, no test.
                std::vector<HwVertex> verts(m_uiDraw.size());
                for (size_t i = 0; i < m_uiDraw.size(); ++i)
                {
                    verts[i] = HwVertex{};
                    verts[i].x = m_uiDraw[i][0] - static_cast<float>(g.dbx);
                    verts[i].y = m_uiDraw[i][1] - static_cast<float>(g.dby);
                    verts[i].rgba = 0xFFFFFFFFu;
                    verts[i].q = 1.0f;
                }
                cmd.image_barrier(*mask, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                  0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                Vulkan::RenderPassInfo rp = {};
                rp.num_color_attachments = 1;
                rp.color_attachments[0] = &mask->get_view();
                rp.clear_attachments = 1;
                rp.store_attachments = 1;
                cmd.begin_render_pass(rp);
                if (!verts.empty())
                {
                    cmd.set_program(m_drawProgram);
                    cmd.set_opaque_state();
                    cmd.set_cull_mode(VK_CULL_MODE_NONE);
                    cmd.set_depth_test(false, false);
                    cmd.set_blend_enable(false);
                    cmd.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
                    cmd.set_vertex_attrib(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(HwVertex, x));
                    cmd.set_vertex_attrib(1, 0, VK_FORMAT_R32_SFLOAT, offsetof(HwVertex, z));
                    cmd.set_vertex_attrib(2, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(HwVertex, rgba));
                    cmd.set_vertex_attrib(3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(HwVertex, s));
                    cmd.set_vertex_attrib(4, 0, VK_FORMAT_R32_SFLOAT, offsetof(HwVertex, fog));
                    cmd.set_vertex_attrib(5, 0, VK_FORMAT_R16G16_SFLOAT, offsetof(HwVertex, motion));
                    auto *dst = static_cast<HwVertex *>(cmd.allocate_vertex_data(0, verts.size() * sizeof(HwVertex), sizeof(HwVertex)));
                    std::memcpy(dst, verts.data(), verts.size() * sizeof(HwVertex));
                    cmd.set_texture(0, 0, m_white->get_view(), sampler(false, false, false, false, false));
                    cmd.set_texture(0, 1, m_white->get_view(), sampler(false, false, false, false, false));
                    cmd.set_uniform_buffer(0, 2, *m_noRecolor);
                    Push p = {};
                    const float w = static_cast<float>(g.outW), h = static_cast<float>(g.outH);
                    p.posScale[0] = 2.0f * g.sx / w;
                    p.posScale[1] = 2.0f * g.sy / h;
                    p.posScale[2] = 1.0f / w - 1.0f;
                    p.posScale[3] = 1.0f / h - 1.0f;
                    p.fogColor[3] = 255.0f;
                    p.mode[3] = 2u; // no alpha test
                    cmd.set_color_write_mask(0xF);
                    cmd.set_specialization_constant_mask(0x7);
                    cmd.set_specialization_constant(0, 0u);
                    cmd.set_specialization_constant(1, 0u);
                    cmd.set_specialization_constant(2, 2u);
                    cmd.push_constants(&p, 0, sizeof(p));
                    cmd.draw(static_cast<uint32_t>(verts.size()));
                }
                cmd.end_render_pass();
                cmd.image_barrier(*mask, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                m_shared->ui = mask;
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
            // Progressive fields (the default) until the first Present says otherwise; RT_PROGRESSIVE_FIELDS=0
            // (the regression harness) says so before any target is made.
            std::atomic<uint32_t> m_yScale{[] {
                const char *pf = std::getenv("RT_PROGRESSIVE_FIELDS");
                return pf && std::strcmp(pf, "0") == 0 ? 1u : 2u;
            }()};

            GSCpuBackend m_cpu; // transfers and readbacks over the shared local memory
            uint8_t *m_vram = nullptr;
            uint32_t m_vramSize = 0;
            GSMem::TexturePageCache m_pageCache;
            std::array<uint16_t, 512> m_clut{};
            std::array<uint32_t, 2> m_clutCbp{};
            uint64_t m_lastClutSource = 0;
            struct PendingUpload
            {
                bool active = false;
                uint32_t first = 0, last = 0;
                uint64_t hash = 0;
            } m_upload; // GS thread
            std::array<uint32_t, kPages> m_pageVersion{};

            // GS thread -> Present (m_frameMutex).
            std::mutex m_frameMutex;
            std::vector<Batch> m_batches;
            std::vector<HwVertex> m_vertices;
            std::vector<HwVertex> m_stage; // GS thread: the current batch's vertices not yet published
            std::atomic<std::thread::id> m_gsThread{};
            // Recording side (device lock): swapped with the two above, capacity kept.
            std::vector<Batch> m_recordBatches;
            std::vector<HwVertex> m_recordVertices;
            bool m_open = false;
            std::atomic<uint64_t> m_epoch{1};
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
            // Texture dumps and packs.
            HwTexturePacks m_packs;
            struct UploadExtent
            {
                uint32_t psm = 0, width = 0, height = 0;
            };
            std::unordered_map<uint32_t, UploadExtent> m_uploads; // GS thread: uploads by base address
            std::unordered_set<uint64_t> m_submitted;             // GS thread: textures handed to the tools
            std::atomic<bool> m_resubmit{false};
            // The app's repaint of decoded textures (setDecodeHook); m_dropDecoded: decode all again.
            std::mutex m_hookMutex;
            DecodeHook m_decodeHook;
            std::atomic<bool> m_dropDecoded{false};
            std::atomic<uint32_t> m_anisotropy{16};
            uint32_t m_samplerAnisotropy = 0; // recording: the level m_anisoSamplers were made for
            Vulkan::SamplerHandle m_anisoSamplers[4];
            Vulkan::BufferHandle m_noRecolor;
            std::atomic<uint64_t> m_frame{0};
            // RT_HWGS_STATS=1: per-frame counts, logged every 120 presents.
            struct Stats
            {
                uint64_t batches = 0, vertices = 0, passes = 0, snapshots = 0, decodes = 0, presents = 0, dateBatches = 0;
                std::atomic<uint64_t> motionVertices{0};
            } m_stats;
            uint64_t m_breakEpoch = 0;
            uint64_t m_lastEviction = 0;
            // GS thread: the state (and resolved texture) of the newest batch.
            Batch m_current;
            bool m_haveCurrent = false;
            uint64_t m_currentSerial = 0;
            uint64_t m_currentEpoch = 0;
            mutable std::mutex m_targetPagesMutex;

            // Recording (device lock).
            std::unordered_map<uint32_t, Target> m_targets;
            std::unordered_map<uint32_t, Vulkan::ImageHandle> m_depths;
            const Vulkan::Image *m_passColor = nullptr, *m_passDepth = nullptr;
            Vulkan::ImageHandle m_scanout[3];
            // The UI mask: the HUD's triangles (GS frame coordinates), staged by the GS thread,
            // handed over with the batches, drawn at Present; the scanout's geometry they map to.
            std::vector<std::array<float, 2>> m_uiStage, m_uiVerts, m_uiDraw;
            Vulkan::ImageHandle m_uiMask[3];
            struct ScanGeom
            {
                uint32_t dbx = 0, dby = 0, sx = 1, sy = 1, outW = 0, outH = 0;
            } m_scanGeom;
            // Motion vectors (PgsShared::wantMotion): per-target attachments, the scanned-out part.
            bool m_motionOn = false;
            const Vulkan::Image *m_passMotion = nullptr;
            Vulkan::ImageHandle m_motionScan[3], m_motionOut;
            Target m_mapLayers[2];
            uint32_t m_mapIndex = 0, m_mapMissed = 0;
            bool m_mapDrawn = false;
            float m_mapBox[4] = {};
            float m_mapStable[4] = {}; // the box so far for the map on show (grow-only)
            uint32_t m_scanoutIndex = 0;
            Vulkan::BufferHandle m_readback;
            static constexpr uint32_t kVboRing = 8;
            struct VboSlot
            {
                Vulkan::BufferHandle buffer;
                Vulkan::Fence fence;
            } m_vbo[kVboRing];
            uint32_t m_vboSlot = 0;
            bool m_vboUsed = false;
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
