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
#include "runtime/gs/gs_display_phase.h"
#include "runtime/gs/gs_motion.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "../gs_pgs_shared.h"
#include "../post/post_spirv.h"
#include "hw_spirv.h"
#include "gs_hw_textures.h"
#include "ps2x/state_archive.h"

#include "context.hpp"
#include "command_buffer.hpp"
#include "device.hpp"

#include <algorithm>
#include <chrono>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if defined(__linux__)
#include <sys/resource.h>
#endif

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

        // A draw pipeline's state: everything of it that varies between the game's draws (the
        // render pass's attachments, depth, blending, write mask and the shader's specialisation
        // constants). Every state drawn is noted in the cache directory, and the known states
        // (those and the list built in, hw_pipeline_states.inc) are compiled on a worker at start,
        // so the GS thread finds them ready instead of compiling for 8-25 ms in the middle of a
        // frame (Adreno; ~100 of them in a town at first).
        struct PipeKey
        {
            uint8_t depth = 0, motion = 0;  // attachments: D32F depth, RG16F motion
            uint8_t ztest = 0, zwrite = 0;
            uint8_t zop = VK_COMPARE_OP_GREATER_OR_EQUAL; // (set_opaque_state's, when untouched)
            uint8_t blend = 0, src = 0, dst = 0, op = 0;
            uint8_t mask = 0xF;
            uint32_t spec[3] = {};

            uint64_t hash() const
            {
                uint64_t h = 1469598103934665603ull;
                const uint32_t words[] = {uint32_t(depth) | uint32_t(motion) << 8 | uint32_t(ztest) << 16 | uint32_t(zwrite) << 24,
                                          uint32_t(zop) | uint32_t(blend) << 8 | uint32_t(src) << 16 | uint32_t(dst) << 24,
                                          uint32_t(op) | uint32_t(mask) << 8, spec[0], spec[1], spec[2]};
                for (uint32_t w : words)
                    h = (h ^ w) * 1099511628211ull;
                return h;
            }
            std::string text() const
            {
                char line[96];
                std::snprintf(line, sizeof(line), "%u %u %u %u %u %u %u %u %u %u %u %u %u", depth, motion, ztest, zwrite, zop,
                              blend, src, dst, op, mask, spec[0], spec[1], spec[2]);
                return line;
            }
            static bool parse(const char *line, PipeKey &k)
            {
                unsigned v[13];
                if (std::sscanf(line, "%u %u %u %u %u %u %u %u %u %u %u %u %u", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6],
                                &v[7], &v[8], &v[9], &v[10], &v[11], &v[12]) != 13)
                    return false;
                k.depth = uint8_t(v[0] != 0), k.motion = uint8_t(v[1] != 0), k.ztest = uint8_t(v[2] != 0), k.zwrite = uint8_t(v[3] != 0);
                k.zop = uint8_t(std::min(v[4], 7u)), k.blend = uint8_t(v[5] != 0);
                k.src = uint8_t(std::min(v[6], unsigned(VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)));
                k.dst = uint8_t(std::min(v[7], unsigned(VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)));
                k.op = uint8_t(std::min(v[8], unsigned(VK_BLEND_OP_MAX)));
                k.mask = uint8_t(v[9] & (v[1] ? 0x3Fu : 0xFu));
                k.spec[0] = v[10], k.spec[1] = v[11], k.spec[2] = v[12];
                return true;
            }
        };

        // States the game draws with (RT_GS_BACKEND=hw), recorded on the AYN Thor across towns, the
        // open world, races and menus: compiled at the first start, before any is in the cache.
        const char *const kBuiltinPipeStates[] = {
#include "hw_pipeline_states.inc"
        };

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
            // Every triangle flat (one Z, no perspective): 2D, drawn as paraLLEl-GS supersamples
            // UI (attributes at the GS pixel's own sample, Snap::Attributes).
            bool flat = true;
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
            // Draws that may change the alpha (DATE reads only its MSB): a snapshot taken since the
            // last of them still serves DATE, however much RGB was drawn after it.
            uint64_t alphaSerial = 1, snapshotAlphaSerial = 0;
            // DATE through the stencil: the Z buffer whose stencil mirrors this target's alpha MSB
            // (whole target), valid while alphaSerial and the Z buffers' generation are unchanged.
            const Vulkan::Image *stencilDepth = nullptr;
            uint64_t stencilSerial = 0, stencilGeneration = 0, stencilPrepass = 0;
        };

        // The render targets and Z buffers frames draw into: the real ones, and one set per shadow
        // frame (re-rendered frame generation).
        struct TargetSet
        {
            std::unordered_map<uint32_t, Target> targets;
            std::unordered_map<uint32_t, Vulkan::ImageHandle> depths;
        };

        float halfToFloat(uint16_t h)
        {
            const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
            float f;
            if (exp == 0)
                f = std::ldexp(static_cast<float>(man), -24);
            else if (exp == 31)
                f = man ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
            else
                f = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
            return sign ? -f : f;
        }

        // A shadow frame's vertices: the 3D moved on by t frames along each vertex's motion (the HUD
        // and 2D carry none, or the UI marks, and stay). A triangle with a vertex moving more than
        // 200 GS pixels (a camera cut, a mismatched object) isn't moved. Z and texture
        // coordinates stay, as paraLLEl-GS's shadow frames keep them.
        void moveVertices(const std::vector<HwVertex> &in, float t, std::vector<HwVertex> &out)
        {
            out = in;
            for (size_t i = 0; i + 3 <= out.size(); i += 3)
            {
                float dx[3], dy[3];
                bool move = false, keep = false;
                for (int k = 0; k < 3; ++k)
                {
                    const uint32_t m = out[i + k].motion;
                    dx[k] = halfToFloat(static_cast<uint16_t>(m & 0xFFFFu));
                    dy[k] = halfToFloat(static_cast<uint16_t>(m >> 16));
                    if (!std::isfinite(dx[k]) || !std::isfinite(dy[k]))
                        dx[k] = dy[k] = 0.0f;
                    if (std::fabs(dx[k]) > 200.0f || std::fabs(dy[k]) > 200.0f)
                        keep = true;
                    move |= m != 0u;
                }
                if (!move || keep)
                    continue;
                for (int k = 0; k < 3; ++k)
                {
                    out[i + k].x += t * dx[k];
                    out[i + k].y += t * dy[k];
                }
            }
        }

        class HwBackend final : public GSRasterBackend, public PgsControl
        {
        public:
            ~HwBackend() override
            {
                stopPipelineWorker();
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
                m_shared->submitsUnderLock = false;
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
                frag.spec_constant_mask = 0xF; // flags, ATST, alpha-test pass, DATE split (draw.frag)
                m_drawProgram = m_dev->request_program(hw_spirv::draw_vert, sizeof(hw_spirv::draw_vert), hw_spirv::draw_frag,
                                                       sizeof(hw_spirv::draw_frag), &vert, &frag);
                {
                    // The depth snapshot: the presenter's full-screen triangle and depth_copy.frag.
                    Vulkan::ResourceLayout fsVert = {};
                    fsVert.output_mask = 0x1;
                    Vulkan::ResourceLayout copyFrag = {};
                    copyFrag.input_mask = 0x1;
                    copyFrag.output_mask = 0x1;
                    copyFrag.push_constant_size = 16;
                    copyFrag.sets[0].sampled_image_mask = 0x1;
                    copyFrag.sets[0].fp_mask = 0x1;
                    m_depthCopyProgram = m_dev->request_program(post_spirv::fullscreen_vert, sizeof(post_spirv::fullscreen_vert),
                                                                hw_spirv::depth_copy_frag, sizeof(hw_spirv::depth_copy_frag), &fsVert,
                                                                &copyFrag);
                    // DATE through the stencil buffer (date_stencil.frag): Z buffers get a stencil
                    // aspect, set from the target's alpha MSB once per run of DATE draws, which keep
                    // it in step (record()). The snapshot test copied the target and split the pass
                    // for each DATE draw: Q's Factory's 39 shadow draws a frame took the GPU to 82%
                    // at 120 Hz (55% this way; passes 57 -> 6 a present). Setting the stencil per
                    // draw, as this first did, maxed it in towns. RT_HWGS_STENCIL_DATE=0: snapshots.
                    Vulkan::ResourceLayout stencilFrag = {};
                    stencilFrag.input_mask = 0x1;
                    stencilFrag.sets[0].sampled_image_mask = 0x1;
                    stencilFrag.sets[0].fp_mask = 0x1;
                    m_dateStencilProgram = m_dev->request_program(post_spirv::fullscreen_vert, sizeof(post_spirv::fullscreen_vert),
                                                                  hw_spirv::date_stencil_frag, sizeof(hw_spirv::date_stencil_frag),
                                                                  &fsVert, &stencilFrag);
                    const char *e = std::getenv("RT_HWGS_STENCIL_DATE");
                    m_stencilDate = m_dateStencilProgram && !(e && *e == '0') &&
                                    m_dev->image_format_is_supported(VK_FORMAT_D32_SFLOAT_S8_UINT,
                                                                     VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT |
                                                                         VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT);
                    m_depthFormat = m_stencilDate ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
                }
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
                loadPipelineCache(options.pipelineCacheDir);
                warmUpPipelines();

                m_shared->attached = true;
                m_shared->scanoutRing = 3; // m_scanout
                m_shared->flushLocked = [] {};
                m_shared->submitsUnderLock = true;
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
                m_resubmit = true;    // and see again which textures it repaints (no pack image for those)
                ++m_epoch;
            }

            // ---------------------------------------------------------------- GSRasterBackend
            bool WantsPrimitives() const override { return true; }
            // Depth for temporal upscalers: taken where the 3D ends (the HUD and the post-pass
            // overwrite Z), as raw GS Z the scanout's size (PgsShared::depth).
            bool WantsDepthSnapshot() const override { return m_shared && m_shared->wantDepth.load(std::memory_order_relaxed); }
            void SnapshotDepth(uint32_t zbp, uint32_t, uint32_t fbp) override
            {
                publish(true);
                m_depthRequestZbp = zbp;
                m_depthRequestFbp = fbp;
                m_depthRequested = true;
                flushPending();
            }
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

            // A sprite covers the GS pixels whose sample point (the pixel's corner) it contains, at
            // any render scale, as paraLLEl-GS snaps sprite rasterisation: its edges move to whole
            // GS pixels (ceil: the GS's top-left rule), its attributes along with them.
            static void snapSprite(HwVertex &a, HwVertex &b, bool snapX, bool snapY)
            {
                auto snap = [](float &p0, float &p1, float *a0, float *a1, int n) {
                    const float q0 = std::ceil(p0), q1 = std::ceil(p1);
                    if (p1 != p0)
                    {
                        const float k0 = (q0 - p0) / (p1 - p0), k1 = (q1 - p0) / (p1 - p0);
                        for (int i = 0; i < n; ++i)
                        {
                            const float d = a1[i] - a0[i], base = a0[i];
                            a0[i] = base + d * k0;
                            a1[i] = base + d * k1;
                        }
                    }
                    p0 = q0, p1 = q1;
                };
                // S (or U) runs along x, T (or V) along y.
                if (snapX)
                    snap(a.x, b.x, &a.s, &b.s, 1);
                if (snapY)
                    snap(a.y, b.y, &a.t, &b.t, 1);
            }

            // How 2D (sprites, flat triangles) is drawn above 1x, as paraLLEl-GS supersamples it:
            // coverage snapped to GS pixels and attributes (texture coordinates, colour, fog) taken
            // at the GS pixel's sample point, across always, down except for interlaced fields shown
            // as fields (field-aware there: the supersampled lines are the frame's, and alternate
            // fields land between each other); the mode is the last Present's (before the first,
            // m_yScale's guess). Measured against paraLLEl-GS on the regression
            // suite's pictures (menus, cards, HUD text: crisp, no half-texel blur or shift).
            // RT_HWGS_SNAP=<letters> overrides: X, Y snap coverage, x, y attributes on that axis.
            struct Snap
            {
                bool coverX, coverY, attrX, attrY;
            };
            Snap snapRules() const
            {
                static const char *env = std::getenv("RT_HWGS_SNAP");
                if (env)
                    return {std::strchr(env, 'X') != nullptr, std::strchr(env, 'Y') != nullptr, std::strchr(env, 'x') != nullptr,
                            std::strchr(env, 'y') != nullptr};
                const bool down = !m_fieldAware.load(std::memory_order_relaxed);
                return {true, down, true, down};
            }

            // GS thread: the batch this draw goes into, a new one when the state or the texture
            // contents changed (decided and resolved without the frame lock: decoding a texture
            // takes the device lock; m_current is the GS thread's own).
            void openFor(const GSPrimitiveBatch &batch)
            {
                const GSDrawState &state = batch.state;
                m_gsThread.store(std::this_thread::get_id(), std::memory_order_relaxed);
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
            }

            bool WantsRuns() const override
            {
                static const bool on = [] { const char *e = std::getenv("RT_GS_RUNS"); return !(e && *e == '0'); }();
                return on;
            }

            // Submit for a whole strip or list: each vertex converted once (Submit converts a
            // strip's vertices three times, once per triangle), then the triangles as Submit
            // stages them.
            void SubmitRun(const GSPrimitiveBatch &batch, const GSVertex *verts, uint32_t count, bool strip) override
            {
                const GSDrawState &state = batch.state;
                if (count < 3u)
                    return;
                openFor(batch);
                const GSContext &ctx = state.context;
                const float ofx = ctx.xyoffset.ofx / 16.0f, ofy = ctx.xyoffset.ofy / 16.0f;
                const double zScale = zUnit(ctx.zbuf.psm);
                const float tw = static_cast<float>(1u << std::min<uint32_t>(ctx.tex0.tw, 10u));
                const float th = static_cast<float>(1u << std::min<uint32_t>(ctx.tex0.th, 10u));
                m_runConverted.resize(count);
                for (uint32_t i = 0; i < count; ++i)
                {
                    const GSVertex &v = verts[i];
                    HwVertex &o = m_runConverted[i];
                    o.x = v.x - ofx;
                    o.y = v.y - ofy;
                    o.z = static_cast<float>(std::min(v.z * zScale, 1.0));
                    o.rgba = v.r | (v.g << 8) | (v.b << 16) | (static_cast<uint32_t>(v.a) << 24);
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
                }
                const bool checkQ = state.prim.tme && !state.prim.fst;
                const uint32_t triangles = strip ? count - 2u : count / 3u;
                const size_t first = m_stage.size();
                m_stage.resize(first + static_cast<size_t>(triangles) * 3u);
                HwVertex *out = m_stage.data() + first;
                for (uint32_t t = 0; t < triangles; ++t)
                {
                    const uint32_t a = strip ? t : 3u * t;
                    out[0] = m_runConverted[a];
                    out[1] = m_runConverted[a + 1u];
                    out[2] = m_runConverted[a + 2u];
                    if (!state.prim.iip) // flat shading takes the colour of the last vertex
                        out[0].rgba = out[1].rgba = out[2].rgba;
                    // 2D as paraLLEl-GS tells it: one Z and, with STQ, one Q.
                    if (m_current.flat && (verts[a].z != verts[a + 1u].z || verts[a + 1u].z != verts[a + 2u].z ||
                                           (checkQ && (verts[a].q != verts[a + 1u].q || verts[a + 1u].q != verts[a + 2u].q))))
                        m_current.flat = false;
                    out += 3;
                }
                if (batch.vertexClass == 1u)
                    for (size_t i = first; i < m_stage.size(); ++i)
                        m_uiStage.push_back({m_stage[i].x, m_stage[i].y});
                if (m_stage.size() >= 1536u)
                    publish(false);
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
                openFor(batch);
                static const bool stats = std::getenv("RT_HWGS_STATS") != nullptr;

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
                    if (stats && v.motion) // (a sequentially consistent add per vertex: ~3% of the GS thread)
                        m_stats.motionVertices.fetch_add(1, std::memory_order_relaxed);
                    return o;
                };

                if (type == GS_PRIM_SPRITE)
                {
                    const GSVertex &a = batch.vertices[0], &b = batch.vertices[1];
                    HwVertex v0 = convert(a, b), v1 = convert(b, b);
                    v0.z = v1.z; // a sprite takes Z (and colour, fog) from its second vertex
                    v0.fog = v1.fog;
                    {
                        const Snap rules = snapRules();
                        snapSprite(v0, v1, rules.coverX, rules.coverY);
                    }
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
                    // 2D as paraLLEl-GS tells it: one Z and, with STQ, one Q.
                    const auto &v = batch.vertices;
                    if (m_current.flat && (v[0].z != v[1].z || v[1].z != v[2].z ||
                                           (state.prim.tme && !state.prim.fst && (v[0].q != v[1].q || v[1].q != v[2].q))))
                        m_current.flat = false;
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
                if (batches.empty() && !m_depthRequested)
                    return;
                auto cmd = m_dev->request_command_buffer();
                record(*cmd, batches, vertices);
                keepForShadows(batches, vertices);
                batches.clear();
                vertices.clear();
                if (m_depthRequested)
                {
                    m_depthRequested = false;
                    snapshotDepthLocked(*cmd, m_depthRequestZbp, m_depthRequestFbp);
                }
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

            // Save states (EE thread at a savable loop top, the frontend's state lock held: the GS
            // thread is between packets). The shared part is what the CPU GS saves: local memory as
            // the game wrote it (uploads and local copies: the render targets live on the GPU and
            // never reach it, so it is the same bytes on every GPU and in every run), this backend's
            // CLUT and the transfer state. The extra part (this backend only) is the texture tools'
            // upload extents and the render targets and Z buffers as drawn (raw, at the render
            // scale): Road Trip blends some of its picture with what the buffer held, so a load that
            // started from empty targets would stay a level off here and there for good. A digest
            // (state hashes) leaves the targets out. A state from another GS loads with empty
            // targets (the game's next frames redraw them); one taken at another render scale is
            // scaled to this one (its Z buffers start empty).
            bool SerializeState(ps2x::StateArchive &ar) override
            {
                if (ar.saving())
                    m_cpu.SetClutState(m_clut, m_clutCbp);
                if (!m_cpu.SerializeState(ar))
                    return false;
                if (ar.loading())
                    m_cpu.GetClutState(m_clut, m_clutCbp);
                return ar.ok();
            }

            uint32_t StateExtrasKind() const override { return ps2x::fourcc("HWG1"); }

            struct SavedImage
            {
                uint32_t key = 0, fbw = 0, psm = 0, width = 0, height = 0; // FBP/ZBP, GS size
                uint32_t sx = 1, sy = 1, w = 0, h = 0;                       // render scale, image size
                std::vector<uint8_t> bytes;                                 // RGBA8 or D32F texels
            };

            void SerializeStateExtras(ps2x::StateArchive &ar) override
            {
                ar.keyOrderedMap(m_uploads, [](ps2x::StateArchive &a, uint32_t &k) { a & k; },
                                 [](ps2x::StateArchive &a, UploadExtent &e) { a & e.psm & e.width & e.height; });
                std::vector<SavedImage> targets, depths;
                if (ar.saving() && !ar.digest())
                    readTargets(targets, depths);
                const auto image = [](ps2x::StateArchive &a, SavedImage &i)
                {
                    a & i.key & i.fbw & i.psm & i.width & i.height & i.sx & i.sy & i.w & i.h;
                    a.podVector(i.bytes);
                    if (a.loading() && i.bytes.size() != static_cast<size_t>(i.w) * i.h * 4u)
                        a.fail("hardware GS: a render target's size doesn't match its pixels");
                };
                ar.sequence(targets, image);
                ar.sequence(depths, image);
                if (ar.loading())
                {
                    m_loadedTargets = std::move(targets);
                    m_loadedDepths = std::move(depths);
                }
            }

            // Saving: the real frame's render targets and Z buffers with everything drawn so far
            // (what the GS thread has queued is recorded first).
            void readTargets(std::vector<SavedImage> &targets, std::vector<SavedImage> &depths)
            {
                publish(true);
                flushPending();
                std::vector<std::pair<SavedImage *, Vulkan::BufferHandle>> jobs;
                Vulkan::Fence fence;
                {
                    const auto device = lockDevice();
                    std::vector<uint32_t> keys, zkeys;
                    for (const auto &[fbp, t] : m_main.targets)
                        if (t.color)
                            keys.push_back(fbp);
                    for (const auto &[zbp, d] : m_main.depths)
                        if (d)
                            zkeys.push_back(zbp);
                    std::sort(keys.begin(), keys.end());
                    std::sort(zkeys.begin(), zkeys.end());
                    targets.resize(keys.size());
                    depths.resize(zkeys.size());
                    auto cmd = m_dev->request_command_buffer();
                    const auto copy = [&](SavedImage &out, const Vulkan::Image &img, bool depth)
                    {
                        out.w = img.get_width();
                        out.h = img.get_height();
                        Vulkan::BufferCreateInfo info = {};
                        info.size = static_cast<VkDeviceSize>(out.w) * out.h * 4u;
                        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                        info.domain = Vulkan::BufferDomain::CachedHost;
                        auto buffer = m_dev->create_buffer(info);
                        const VkPipelineStageFlags2 stages =
                            depth ? (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
                                  : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                        const VkAccessFlags2 access =
                            depth ? (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
                                  : (VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                        cmd->image_barrier(img, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stages, access,
                                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                        cmd->copy_image_to_buffer(*buffer, img, 0, {}, {out.w, out.h, 1}, 0, 0,
                                                  {static_cast<VkImageAspectFlags>(depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1});
                        cmd->image_barrier(img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                           VK_PIPELINE_STAGE_2_COPY_BIT, 0, stages, access);
                        jobs.emplace_back(&out, std::move(buffer));
                    };
                    for (size_t i = 0; i < keys.size(); ++i)
                    {
                        const Target &t = m_main.targets[keys[i]];
                        SavedImage &out = targets[i];
                        out.key = keys[i];
                        out.fbw = t.fbw, out.psm = t.psm, out.width = t.width, out.height = t.height, out.sx = t.sx, out.sy = t.sy;
                        copy(out, *t.color, false);
                    }
                    for (size_t i = 0; i < zkeys.size(); ++i)
                    {
                        const Vulkan::Image &d = *m_main.depths[zkeys[i]];
                        SavedImage &out = depths[i];
                        out.key = zkeys[i];
                        // The scale it was drawn at: that of a target its size.
                        for (const SavedImage &t : targets)
                            if (t.w == d.get_width() && t.h == d.get_height())
                                out.sx = t.sx, out.sy = t.sy, out.width = t.width, out.height = t.height;
                        copy(out, d, true);
                    }
                    cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                                 VK_ACCESS_2_HOST_READ_BIT);
                    m_dev->submit(cmd, &fence);
                }
                fence->wait();
                const auto device = lockDevice();
                for (auto &[out, buffer] : jobs)
                {
                    const auto *src = static_cast<const uint8_t *>(m_dev->map_host_buffer(*buffer, Vulkan::MEMORY_ACCESS_READ_BIT));
                    out->bytes.assign(src, src + static_cast<size_t>(out->w) * out->h * 4u);
                    m_dev->unmap_host_buffer(*buffer, Vulkan::MEMORY_ACCESS_READ_BIT);
                }
            }

            // Loading (device lock held, the targets just dropped): the saved targets back, at this
            // render scale; Z buffers only when the scale is the same.
            void restoreTargets()
            {
                if (m_loadedTargets.empty() && m_loadedDepths.empty())
                    return;
                const uint32_t sx = m_scale, sy = m_scale * m_yScale;
                auto cmd = m_dev->request_command_buffer();
                const auto staging = [&](const SavedImage &s)
                {
                    Vulkan::BufferCreateInfo info = {};
                    info.size = s.bytes.size();
                    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                    info.domain = Vulkan::BufferDomain::Host;
                    return m_dev->create_buffer(info, s.bytes.data());
                };
                for (const SavedImage &s : m_loadedTargets)
                {
                    if (!s.w || !s.h || !s.width || !s.height)
                        continue;
                    const uint32_t w = s.width * sx, h = s.height * sy;
                    auto info = Vulkan::ImageCreateInfo::render_target(w, h, VK_FORMAT_R8G8B8A8_UNORM);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    auto image = m_dev->create_image(info);
                    auto buffer = staging(s);
                    cmd->image_barrier(*image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                                       VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                    if (s.w == w && s.h == h)
                        cmd->copy_buffer_to_image(*image, *buffer, 0, {}, {w, h, 1}, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
                    else
                    {
                        // Saved at another render scale: scaled to this one.
                        auto tmpInfo = Vulkan::ImageCreateInfo::render_target(s.w, s.h, VK_FORMAT_R8G8B8A8_UNORM);
                        tmpInfo.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                        auto tmp = m_dev->create_image(tmpInfo);
                        cmd->image_barrier(*tmp, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                        cmd->copy_buffer_to_image(*tmp, *buffer, 0, {}, {s.w, s.h, 1}, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
                        cmd->image_barrier(*tmp, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                                           VK_ACCESS_2_TRANSFER_READ_BIT);
                        cmd->blit_image(*image, *tmp, {0, 0, 0}, {static_cast<int32_t>(w), static_cast<int32_t>(h), 1}, {0, 0, 0},
                                        {static_cast<int32_t>(s.w), static_cast<int32_t>(s.h), 1}, 0, 0, 0, 0, 1, VK_FILTER_LINEAR);
                    }
                    cmd->image_barrier(*image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                       VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                    Target &t = m_main.targets[s.key];
                    t.color = image;
                    t.fbp = s.key;
                    t.fbw = s.fbw;
                    t.psm = s.psm;
                    t.width = s.width;
                    t.height = s.height;
                    t.sx = sx;
                    t.sy = sy;
                    ++t.drawSerial;
                    ++t.alphaSerial;
                    std::lock_guard<std::mutex> lock(m_targetPagesMutex);
                    uint32_t first, last;
                    pageRange(s.key * 32u, s.fbw, GS_PSM_CT32, 0, 0, s.width, s.height, first, last);
                    m_targetPages[s.key] = {first, last, s.fbw};
                }
                for (const SavedImage &s : m_loadedDepths)
                {
                    if (!s.w || !s.h || s.sx != sx || s.sy != sy)
                        continue;
                    auto info = Vulkan::ImageCreateInfo::render_target(s.w, s.h, m_depthFormat);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    auto image = m_dev->create_image(info);
                    auto buffer = staging(s);
                    cmd->image_barrier(*image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                                       VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                    cmd->copy_buffer_to_image(*image, *buffer, 0, {}, {s.w, s.h, 1}, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1});
                    cmd->image_barrier(*image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                       VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
                    m_main.depths[s.key] = image;
                    ++m_depthGeneration;
                }
                m_dev->submit(cmd);
                m_loadedTargets.clear();
                m_loadedDepths.clear();
            }

            void StateLoaded(bool extras) override
            {
                if (!extras)
                {
                    m_uploads.clear();
                    m_loadedTargets.clear();
                    m_loadedDepths.clear();
                }
                m_upload = {};
                m_lastClutSource = 0;
                m_pageCache.Invalidate();
                m_haveCurrent = false;
                {
                    const auto device = lockDevice();
                    {
                        // What the GS thread had queued belongs to the timeline left behind.
                        std::lock_guard<std::mutex> lock(m_frameMutex);
                        m_batches.clear();
                        m_vertices.clear();
                        m_stage.clear();
                        m_uiStage.clear();
                        m_uiVerts.clear();
                        m_open = false;
                        for (auto &v : m_pageVersion) // every decoded texture is decoded again
                            ++v;
                    }
                    m_recordBatches.clear();
                    m_recordVertices.clear();
                    m_shadowBatches.clear();
                    m_shadowBatchVerts.clear();
                    m_shadowWarm = 0;
                    m_depthRequested = false;
                    // The render targets and Z buffers: the state's (restoreTargets), else empty.
                    m_main = TargetSet{};
                    for (TargetSet &s : m_shadowSets)
                        s = TargetSet{};
                    {
                        std::lock_guard<std::mutex> lock(m_targetPagesMutex);
                        m_targetPages.clear();
                    }
                    restoreTargets();
                }
                ++m_epoch;
            }

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
                    {
                        m_main = {};
                        for (auto &set : m_shadowSets)
                            set = {};
                    }
                    servicePacks();
                    m_yScale = request.progressiveFields ? 2u : 1u;
                    m_motionOn = m_shared && m_shared->wantMotion.load(std::memory_order_relaxed);
                    // Shadow frames start from empty buffers: published once the game has drawn
                    // every buffer again.
                    // Frame skip: while the game is behind, no shadows (they start over afterwards).
                    // A pause (frame skip, GPU headroom) keeps the shadows' targets: throwing them
                    // away and making them again at every pause cost a spike, and targets the game
                    // draws only now and then (the sky) were missing afterwards (flashes).
                    const uint32_t wanted = m_shared ? std::min(m_shared->wantShadows.load(std::memory_order_relaxed), kMaxShadows) : 0u;
                    if (wanted != m_shadowsWanted)
                    {
                        m_shadowsWanted = wanted;
                        for (auto &set : m_shadowSets)
                            set = {};
                        m_shadowWarm = 0;
                    }
                    const uint32_t shadows = request.pauseShadows ? 0u : wanted;
                    if (shadows != m_shadows)
                    {
                        m_shadows = shadows;
                        m_shadowBatches.clear();
                        m_shadowBatchVerts.clear();
                        m_shadowWarm = 0;
                    }
                    auto cmd = m_dev->request_command_buffer();
                    // The presenter may still be sampling an older scanout image.
                    cmd->barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
                    record(*cmd, batches, vertices);
                    // A frame whose drawing was skipped: the last picture stays (no new scanout).
                    const Vulkan::Image *scan = request.frameSkipped ? nullptr : scanout(*cmd, request);
                    noteMapPresent();
                    // Drawing from here on (the next frame) snaps as this scanout shows (snapRules).
                    m_fieldAware = (request.smode2 & 3u) == 3u && !request.progressiveFields;
                    if (scan && m_shared && m_shared->wantUi.load(std::memory_order_relaxed))
                        drawUiMask(*cmd);
                    m_uiDraw.clear();
                    if (scan)
                    {
                        outW = scan->get_width();
                        outH = scan->get_height();
                        m_shared->scanout = m_scanout[m_scanoutIndex];
                        m_shared->motion = m_motionOut;
                        if (m_shared->wantDepth.load(std::memory_order_relaxed))
                        {
                            // The snapshot of the 3D in the buffer on display; else the newest (2D
                            // frames keep the last 3D's).
                            const auto it = m_depthByFbp.find(m_scanFbp);
                            m_shared->depth = it != m_depthByFbp.end() ? it->second : m_depthReady;
                        }
                        ++m_shared->presentSerial;
                        m_shared->pictureSerial = request.frame3D;
                    }
                    if (scan && request.readback)
                        readbackCopy(*cmd, *scan);
                    {
                        Vulkan::Fence used;
                        submitRecorded(cmd, &used);
                        if (request.readback && scan)
                            fence = used;
                    }
                    // The shadows after the real frame (submitted first, so they don't delay it).
                    if (m_shadows)
                        keepForShadows(batches, vertices);
                    // A still frame (no vertex moves: menus, Q's Factory, pauses) needs no shadow:
                    // re-rendering it cost as much GPU as the real frame (Thor, Q's Factory at
                    // 120 Hz: 90% busy, and the display fell behind and showed strips of garbage).
                    // Nothing is published, so the presenter repeats the real frame; the shadows
                    // warm up again (three presents) once things move, as their targets are stale.
                    bool still = m_shadows != 0;
                    for (size_t v = 0; still && v < m_shadowBatchVerts.size(); ++v)
                        still = m_shadowBatchVerts[v].motion == 0u;
                    if (still && m_shadows && !m_shadowBatchVerts.empty())
                    {
                        m_shadowBatches.clear();
                        m_shadowBatchVerts.clear();
                        m_shadowWarm = 0;
                    }
                    else if (m_shadows)
                    {
                        auto shadowCmd = m_dev->request_command_buffer();
                        recordShadows(*shadowCmd, m_shadowBatches, m_shadowBatchVerts);
                        m_shadowBatches.clear();
                        m_shadowBatchVerts.clear();
                        m_shadowIndex = (m_shadowIndex + 1u) % 3u;
                        m_shadowWarm = std::min(m_shadowWarm + 1u, 3u);
                        for (uint32_t i = 0; i < m_shadows; ++i)
                        {
                            m_set = &m_shadowSets[i];
                            m_inShadow = true;
                            m_shadowSlot = i;
                            const Vulkan::Image *shadowScan = scan ? scanout(*shadowCmd, request) : nullptr;
                            m_inShadow = false;
                            m_set = &m_main;
                            if (shadowScan && m_shadowWarm >= 3u)
                            {
                                m_shared->shadowScanout[i] = m_shadowScan[i][m_shadowIndex];
                                m_shared->shadowSerial[i] = m_shared->presentSerial;
                            }
                        }
                        submitRecorded(shadowCmd);
                    }
                    batches.clear();
                    vertices.clear();
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
                    std::fprintf(stderr, "[hwgs] shadows: %u, recording %.2f ms per present; vertex-ring waits %.1f per present, %.2f ms\n",
                                 m_shadows, m_stats.shadowUs / n / 1000.0, m_stats.vboWaits / n, m_stats.vboWaitUs / n / 1000.0);
                    m_stats.vboWaits = m_stats.vboWaitUs = m_stats.shadowUs = 0;
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
                const TexRect own = rect; // the part this draw samples
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
                {
                    m_submitted.clear();
                    m_repainted.clear();
                }
                if (m_submitted.insert(key).second)
                {
                    std::vector<uint8_t> rgba(static_cast<size_t>(rect.w) * rect.h * 4u);
                    uint32_t *out = reinterpret_cast<uint32_t *>(rgba.data());
                    for (uint32_t y = 0; y < rect.h; ++y)
                        for (uint32_t x = 0; x < rect.w; ++x)
                            out[static_cast<size_t>(y) * rect.w + x] = texel(b.state, tex.psm, tex.tbp0, tex.tbw, rect.x + x, rect.y + y);
                    {
                        // A texture the app repaints (the button glyphs in the font atlas, for the pad
                        // in use): draws that sample the repainted part keep the repainted decode (the
                        // pack's image shows the original glyphs); the rest still use the pack. The
                        // tools get the original (dumps stay the game's).
                        std::lock_guard<std::mutex> lock(m_hookMutex);
                        if (m_decodeHook)
                        {
                            std::vector<uint8_t> copy(rgba);
                            DecodedTexture d;
                            d.psm = tex.psm;
                            d.width = rect.w;
                            d.height = rect.h;
                            d.rgba = reinterpret_cast<uint32_t *>(copy.data());
                            const uint32_t psm = tex.psm, tbp = tex.tbp0, tbw = tex.tbw, rx = rect.x, ry = rect.y;
                            d.index = [this, psm, tbp, tbw, rx, ry](uint32_t x, uint32_t y)
                            { return GSMem::ReadTexture(m_pageCache, m_vram, psm, tbp, tbw, rx + x, ry + y) & 0xFFu; };
                            m_decodeHook(d);
                            const uint32_t *a = reinterpret_cast<const uint32_t *>(rgba.data());
                            const uint32_t *c = reinterpret_cast<const uint32_t *>(copy.data());
                            uint32_t x0 = rect.w, y0 = rect.h, x1 = 0, y1 = 0;
                            for (uint32_t y = 0; y < rect.h; ++y)
                                for (uint32_t x = 0; x < rect.w; ++x)
                                    if (a[static_cast<size_t>(y) * rect.w + x] != c[static_cast<size_t>(y) * rect.w + x])
                                    {
                                        x0 = std::min(x0, x), y0 = std::min(y0, y);
                                        x1 = std::max(x1, x + 1), y1 = std::max(y1, y + 1);
                                    }
                            if (x1 > x0)
                                m_repainted[key] = {rect.x + x0, rect.y + y0, x1 - x0, y1 - y0};
                        }
                    }
                    m_packs.submit(key, stable, rect.w, rect.h, tex.psm, std::move(rgba));
                }
                if (auto r = m_repainted.find(key); r != m_repainted.end())
                {
                    const TexRect &p = r->second;
                    if (own.x < p.x + p.w && p.x < own.x + own.w && own.y < p.y + p.h && p.y < own.y + own.h)
                        return;
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
                    if (isIndexed(tex.psm))
                    {
                        // The palette once (TEXA applied), then an index and a lookup per texel: per
                        // texel it was a palette walk too (the GS thread's busiest decode).
                        uint32_t palette[256];
                        const uint32_t entries = isFourBit(tex.psm) ? 16u : 256u;
                        for (uint32_t i = 0; i < entries; ++i)
                            palette[i] = clutColor(s, i);
                        const uint32_t mask = entries - 1u;
                        for (uint32_t y = 0; y < h; ++y)
                        {
                            uint32_t *row = data[l].data() + static_cast<size_t>(y) * w;
                            for (uint32_t x = 0; x < w; ++x)
                                row[x] = palette[GSMem::ReadTexture(m_pageCache, m_vram, tex.psm, lv[l].tbp, lv[l].tbw, x, y) & mask];
                        }
                    }
                    else
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
                Target &t = m_set->targets[fbp];
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
                    ++t.alphaSerial;
                    std::lock_guard<std::mutex> lock(m_targetPagesMutex);
                    uint32_t first, last;
                    pageRange(fbp * 32u, fbw, GS_PSM_CT32, 0, 0, width, height, first, last);
                    m_targetPages[fbp] = {first, last, fbw};
                }
                return t;
            }

            Vulkan::ImageHandle &depth(uint32_t zbp, const Target &t, Vulkan::CommandBuffer &cmd, bool &passOpen)
            {
                Vulkan::ImageHandle &d = m_set->depths[zbp];
                const uint32_t w = t.color->get_width(), h = t.color->get_height();
                if (!d || d->get_width() != w || d->get_height() != h)
                {
                    endPass(cmd, passOpen);
                    auto info = Vulkan::ImageCreateInfo::render_target(w, h, m_depthFormat);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT; // the depth snapshot (temporal upscalers)
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    d = m_dev->create_image(info);
                    ++m_depthGeneration;
                    cmd.image_barrier(*d, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                                      0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                    VkClearValue zero = {};
                    cmd.clear_image(*d, zero, m_stencilDate ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT);
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
                // One size for both layers whatever frame buffer the map is drawn with: the game's
                // two buffers differ in height, and layers of two sizes made the second screen
                // pair one layer's box with the other's image (the map jumped every frame). The
                // map lies in the field's 224 lines.
                constexpr uint32_t kMapLines = 256;
                const uint32_t sx = std::max(kMapScale, frame.sx), sy = std::max(kMapScale * m_yScale, frame.sy);
                Target &m = m_mapLayers[m_mapIndex];
                const bool fresh = !m.color || m.width != frame.width || m.height != kMapLines || m.sx != sx || m.sy != sy;
                if (fresh || !m_mapDrawn)
                {
                    endPass(cmd, passOpen);
                    if (fresh)
                    {
                        m = Target{};
                        m.fbp = ~0u;
                        m.width = frame.width;
                        m.height = kMapLines;
                        m.sx = sx;
                        m.sy = sy;
                        auto info = Vulkan::ImageCreateInfo::render_target(frame.width * sx, kMapLines * sy, VK_FORMAT_R8G8B8A8_UNORM);
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
                // Edges the scissor cut (the town map's window) are exact; the others get a margin.
                m_mapClipped[0] |= x0 < sc.x0, m_mapClipped[1] |= y0 < sc.y0;
                m_mapClipped[2] |= x1 > sc.x1 + 1, m_mapClipped[3] |= y1 > sc.y1 + 1;
                x0 = std::max(x0, static_cast<float>(sc.x0)), y0 = std::max(y0, static_cast<float>(sc.y0));
                x1 = std::min(x1, static_cast<float>(sc.x1 + 1)), y1 = std::min(y1, static_cast<float>(sc.y1 + 1));
                if (x1 <= x0 || y1 <= y0)
                    return;
                m_mapBox[0] = std::min(m_mapBox[0], x0), m_mapBox[1] = std::min(m_mapBox[1], y0);
                m_mapBox[2] = std::max(m_mapBox[2], x1), m_mapBox[3] = std::max(m_mapBox[3], y1);
            }

            // At the end of a map's run of batches: hands it to the second screen.
            void publishMap(Vulkan::CommandBuffer &cmd)
            {
                if (!m_shared)
                    return;
                const Target &m = m_mapLayers[m_mapIndex];
                if (m_mapDrawn && m.color && m_mapBox[2] > m_mapBox[0])
                {
                    // A fixed size while the same map is shown: the first frame's box with a margin
                    // where the scissor didn't cut it (a race map's outline wobbles a pixel or two as
                    // it is drawn; the town map fills its window exactly), grown only when the
                    // drawing leaves it, kept until the map has been gone for two seconds (another
                    // course or town). A box that changed with the drawing made the second screen's
                    // map change size.
                    constexpr float kMargin = 4.0f; // GS pixels
                    const bool inside = m_mapStable[2] > m_mapStable[0] && m_mapBox[0] >= m_mapStable[0] && m_mapBox[1] >= m_mapStable[1] &&
                                        m_mapBox[2] <= m_mapStable[2] && m_mapBox[3] <= m_mapStable[3];
                    if (!inside)
                    {
                        const bool had = m_mapStable[2] > m_mapStable[0];
                        for (int i = 0; i < 2; ++i)
                        {
                            const float lo = m_mapBox[i] - (m_mapClipped[i] ? 0.0f : kMargin);
                            const float hi = m_mapBox[i + 2] + (m_mapClipped[i + 2] ? 0.0f : kMargin);
                            m_mapStable[i] = had ? std::min(m_mapStable[i], lo) : lo;
                            m_mapStable[i + 2] = had ? std::max(m_mapStable[i + 2], hi) : hi;
                        }
                        m_mapStable[0] = std::max(m_mapStable[0], 0.0f), m_mapStable[1] = std::max(m_mapStable[1], 0.0f);
                        m_mapStable[2] = std::min(m_mapStable[2], static_cast<float>(m.width));
                        m_mapStable[3] = std::min(m_mapStable[3], static_cast<float>(m.height));
                    }
                    std::memcpy(m_mapBox, m_mapStable, sizeof(m_mapBox));
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
                    m_mapPublished = true;
                }
                m_mapDrawn = false;
                m_mapClipped[0] = m_mapClipped[1] = m_mapClipped[2] = m_mapClipped[3] = false;
            }

            // At Present: presents without a map (the map's run ends publish it, publishMap).
            void noteMapPresent()
            {
                if (!m_shared)
                    return;
                if (m_mapPublished || m_mapDrawn)
                    m_mapMissed = 0;
                else
                {
                    ++m_mapMissed;
                    if (m_mapMissed > 10)
                        m_shared->map.reset(); // no map for a while: none to show
                    if (m_mapMissed > 120)
                        m_mapStable[0] = m_mapStable[2] = 0; // the next map (another course or town) gets its own box
                }
                m_mapPublished = false;
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
            // alphaOnly: for DATE, which reads the alpha MSB only (Q's Factory: 39 DATE draws a frame,
            // each copying the whole target and splitting the render pass, GPU 84% at 120 Hz).
            const Vulkan::Image &snapshot(Target &t, Vulkan::CommandBuffer &cmd, bool &passOpen, bool alphaOnly = false)
            {
                if (t.snapshot && (t.snapshotSerial == t.drawSerial || (alphaOnly && t.snapshotAlphaSerial == t.alphaSerial)))
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
                t.snapshotAlphaSerial = t.alphaSerial;
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
                    if (!slot.fence->wait_timeout(0))
                    {
                        // The GPU still reads this slot: the ring is too small for the frame.
                        const auto t0 = std::chrono::steady_clock::now();
                        slot.fence->wait();
                        ++m_stats.vboWaits;
                        m_stats.vboWaitUs += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
                    }
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
                m_vboPending.push_back(m_vboSlot);
                return slot.buffer.get();
            }

            // The fence of the submit that used the current vertex slot.
            void submitRecorded(Vulkan::CommandBufferHandle &cmd, Vulkan::Fence *extra = nullptr)
            {
                Vulkan::Fence fence;
                m_dev->submit(cmd, &fence);
                for (uint32_t slot : m_vboPending) // every vertex slot this submit reads
                    m_vbo[slot].fence = fence;
                m_vboPending.clear();
                if (extra)
                    *extra = fence;
            }

            void record(Vulkan::CommandBuffer &cmd, const std::vector<Batch> &batches, const std::vector<HwVertex> &vertices)
            {
                ++m_frame;
                bool passOpen = false;
                static const bool gpuTimes = [] { const char *e = std::getenv("RT_GPU_TIMES"); return e && *e == '1'; }();
                Vulkan::QueryPoolHandle tsStart = gpuTimes ? cmd.write_timestamp(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) : Vulkan::QueryPoolHandle{};
                const Vulkan::Buffer *vbo = uploadVertices(vertices);
                if (!m_inShadow)
                {
                    m_stats.batches += batches.size();
                    m_stats.vertices += vertices.size();
                }
                for (const Batch &b : batches)
                {
                    const GSDrawState &s = b.state;
                    if (!m_inShadow && ((s.context.test >> 14) & 1u))
                        ++m_stats.dateBatches;
                    if (!m_inShadow && debugSkip(b, vertices))
                        continue;
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
                    if (toMap && m_inShadow)
                        continue; // (the real frame draws it into the map layer, not the frame)
                    // A frame's map goes to the second screen when the game moves on to its other
                    // frame buffer (the next frame), not at Present: the GS thread runs up to a frame
                    // behind the EE, so a Present could come in the middle of a map (half a map
                    // shown) or after two (drawn over each other): the map flashed. (Not at the end
                    // of a run of map batches either: other drawing comes between the map's parts.)
                    if (!m_inShadow)
                    {
                        if (m_mapDrawn && ctx.frame.fbp != m_mapFbp)
                        {
                            endPass(cmd, passOpen);
                            publishMap(cmd);
                        }
                        if (toMap)
                            m_mapFbp = ctx.frame.fbp;
                    }
                    Target &t = toMap ? mapLayer(frameTarget, cmd, passOpen) : frameTarget;
                    if (toMap)
                        noteMapBox(b, vertices);

                    const bool zte = (ctx.test >> 16) & 1u;
                    const uint32_t ztst = (ctx.test >> 17) & 3u;
                    const bool zwrite = !ctx.zbuf.zmask && !toMap;
                    const bool useDepth = !toMap && ((zte && ztst != 1u) || zwrite);
                    // DATE through the stencil aspect of the Z buffer's image (Z itself only tested
                    // and written as the draw says). Not in the map layer (no Z buffer of its size:
                    // the shader tests the snapshot there).
                    const bool dateStencil = ((ctx.test >> 14) & 1u) && m_stencilDate && !toMap;
                    const Vulkan::Image *depthImage = nullptr;
                    if (useDepth || dateStencil)
                        depthImage = depth(ctx.zbuf.zbp, t, cmd, passOpen).get();

                    // The texture: decoded, or a snapshot of a render target.
                    const Vulkan::ImageView *texView = &m_white->get_view();
                    float extentW = 1, extentH = 1, alphaScale = 255.0f;
                    bool ct24Target = false;
                    if (s.prim.tme)
                    {
                        if (b.fromTarget)
                        {
                            auto it = m_set->targets.find(b.targetFbp);
                            if (it != m_set->targets.end() && it->second.color)
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
                    // Through the stencil, it is set from the alpha once per run of DATE draws (the
                    // DATE draws keep it in step); a snapshot only to set it.
                    const bool stencilValid = dateStencil && t.stencilDepth == depthImage && t.stencilSerial == t.alphaSerial &&
                                              t.stencilGeneration == m_depthGeneration && t.stencilPrepass == m_stencilPrepasses;
                    if (date && !stencilValid)
                        destView = &snapshot(t, cmd, passOpen, true).get_view();

                    // Motion for TAA: a second attachment on the frame's targets while it is wanted.
                    Vulkan::Image *motionImage = nullptr;
                    if (m_motionOn && !toMap && !m_inShadow)
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
                    if (((ctx.frame.fbmsk >> 24) & 0xFFu) != 0xFFu && fpsm != GS_PSM_CT24 && !toMap)
                        ++t.alphaSerial;

                    const float devW = static_cast<float>(t.color->get_width()), devH = static_cast<float>(t.color->get_height());
                    const float sx = static_cast<float>(t.sx), sy = static_cast<float>(t.sy);
                    if (dateStencil)
                    {
                        if (!stencilValid)
                        {
                            dateStencilPrepass(cmd, *destView, devW, devH);
                            ++m_stencilPrepasses; // (another target sharing the Z buffer must set it again)
                        }
                        t.stencilPrepass = m_stencilPrepasses;
                        t.stencilDepth = depthImage;
                        t.stencilSerial = t.alphaSerial; // (this draw's alpha writes: the stencil ops follow them)
                        t.stencilGeneration = m_depthGeneration;
                    }

                    // The pipeline's state (applyPipe; noted for the next start's warm-up).
                    PipeKey key;
                    key.depth = depthImage != nullptr;
                    key.motion = motionImage != nullptr;

                    // Depth.
                    if (depthImage && useDepth)
                    {
                        static const VkCompareOp ops[4] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_ALWAYS,
                                                           VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_GREATER};
                        key.ztest = 1;
                        key.zwrite = zwrite;
                        key.zop = static_cast<uint8_t>(zte ? ops[ztst] : VK_COMPARE_OP_ALWAYS);
                    }

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
                        key.blend = 1;
                        key.src = static_cast<uint8_t>(src);
                        key.dst = static_cast<uint8_t>(dst);
                        key.op = static_cast<uint8_t>(op);
                        const float fix = static_cast<float>((ctx.alpha >> 32) & 0xFFu) / 128.0f;
                        const float constants[4] = {0, 0, 0, std::min(fix, 1.0f)};
                        cmd.set_blend_constants(constants);
                        if (s.pabe)
                            unsupported("PABE");
                    }

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
                    // Above 1x, 2D's attributes as paraLLEl-GS interpolates them when it supersamples
                    // (snapRules). Not for render-target textures (post-passes copy device pixels
                    // 1:1) or texture-pack images (filtered at full resolution).
                    if ((sx > 1.0f || sy > 1.0f) && !b.replacement && !b.fromTarget)
                    {
                        const Snap rules = snapRules();
                        if (s.prim.type == GS_PRIM_SPRITE || b.flat)
                        {
                            // The snap scale per axis (1 = none), 4 bits each.
                            p.texOffset[2] = 1.0f;
                            p.texOffset[3] = static_cast<float>((rules.attrX ? static_cast<uint32_t>(sx) : 1u) |
                                                                ((rules.attrY ? static_cast<uint32_t>(sy) : 1u) << 4));
                        }
                    }
                    p.mode[0] = flags;
                    p.mode[1] = (test >> 1) & 7u;
                    p.mode[2] = (test >> 4) & 0xFFu;
                    const bool ate = test & 1u;
                    const uint32_t afail = (test >> 12) & 3u;
                    p.mode[3] = ate ? 0u : 2u;
                    if (date)
                        p.mode[0] |= (dateStencil ? 0u : F_DATE) | (((test >> 15) & 1u) ? F_DATM : 0u);
                    cmd.set_texture(0, 1, *destView, sampler(false, false, false, false, false));
                    if (!(flags & F_RECOLOR))
                        cmd.set_uniform_buffer(0, 2, *m_noRecolor);
                    if (ctx.fba & 1u)
                        unsupported("FBA");

                    cmd.set_vertex_binding(0, *vbo, 0, sizeof(HwVertex));
                    // Motion from unblended draws and the usual blend, Cs * As + Cd * (1 - As) (ALPHA 0x44,
                    // the motion output carries the colour's alpha); other blends (additive effects)
                    // leave it. The HUD writes its zero.
                    const bool motionWrite = motionImage && (!s.prim.abe || (ctx.alpha & 0xFFu) == 0x44u);
                    key.mask = static_cast<uint8_t>(mask | (motionWrite ? 0x30u : 0u));
                    key.spec[0] = p.mode[0];
                    key.spec[1] = p.mode[1];
                    key.spec[2] = p.mode[3];
                    // One draw; with DATE through the stencil, the pixels whose written alpha MSB
                    // leaves DATM (they flip the stencil: later primitives fail there) and those that
                    // keep it are two draws (exact while a batch's pixels do one or the other, as the
                    // game's shadows do). Packed in spec[2] above the alpha-test pass (applyPipe).
                    const uint32_t datm = ((test >> 15) & 1u) << 12;
                    auto drawBatch = [&](PipeKey k) {
                        if (!dateStencil)
                        {
                            applyPipe(cmd, k);
                            cmd.push_constants(&p, 0, sizeof(p));
                            cmd.draw(b.vertexCount, 1, b.firstVertex);
                            return;
                        }
                        const uint32_t amode = k.spec[2] & 0xFFu;
                        const bool writesAlpha = k.mask & 8u;
                        k.spec[2] = amode | (writesAlpha ? (1u << 8) | (2u << 10) : (1u << 10)) | datm;
                        applyPipe(cmd, k);
                        cmd.push_constants(&p, 0, sizeof(p));
                        cmd.draw(b.vertexCount, 1, b.firstVertex);
                        if (!writesAlpha)
                            return;
                        k.spec[2] = amode | (2u << 8) | (1u << 10) | datm;
                        applyPipe(cmd, k);
                        cmd.draw(b.vertexCount, 1, b.firstVertex);
                    };
                    // Pixels failing the alpha test still update what AFAIL lets them. Where Z can't
                    // change a colour (no Z test, or Z not written), in the GS's primitive order: the
                    // part every pixel writes, then (in order again) what only passing pixels add.
                    // Passing pixels then failing ones (below) put a batch's failing pixels over its
                    // later primitives: the race map's track over the car markers.
                    // FB_ONLY: every pixel's colour and alpha, then the passing ones' Z.
                    // ZB_ONLY: every pixel's Z, then the passing ones' colour.
                    // RGB_ONLY: every pixel's colour, then the passing ones' alpha and Z.
                    const bool failWrites = ate && afail != 0u && p.mode[1] != 1u;
                    const bool zw = key.zwrite != 0;
                    if (failWrites && (!zw || !key.ztest || key.zop == VK_COMPARE_OP_ALWAYS))
                    {
                        const uint32_t motionBits = key.mask & 0x30u;
                        const uint32_t common = afail == 1u ? mask : afail == 2u ? 0u : (mask & 0x7u);
                        const uint32_t passing = afail == 1u ? 0u : afail == 2u ? mask : (mask & 0x8u);
                        if (common || (zw && afail == 2u))
                        {
                            PipeKey all = key;
                            all.mask = static_cast<uint8_t>(common | (afail == 1u ? motionBits : 0u));
                            all.zwrite = zw && afail == 2u;
                            all.spec[2] = 2u; // no alpha test
                            p.mode[3] = 2u;
                            drawBatch(all);
                        }
                        if (passing || (zw && afail != 2u) || (afail != 1u && motionBits))
                        {
                            PipeKey pass = key;
                            pass.mask = static_cast<uint8_t>(passing | (afail != 1u ? motionBits : 0u));
                            pass.zwrite = zw && afail != 2u;
                            pass.spec[2] = 0u; // the alpha test
                            p.mode[3] = 0u;
                            drawBatch(pass);
                        }
                        continue;
                    }
                    drawBatch(key);

                    if (failWrites)
                    {
                        p.mode[3] = 1u;
                        if (afail == 1u) // FB_ONLY
                            key.zwrite = 0;
                        else if (afail == 2u) // ZB_ONLY
                            mask = 0u;
                        else // RGB_ONLY
                        {
                            mask &= 0x7u;
                            key.zwrite = 0;
                        }
                        key.mask = static_cast<uint8_t>(mask);
                        key.spec[2] = p.mode[3];
                        drawBatch(key);
                    }
                }
                endPass(cmd, passOpen);
                if (gpuTimes)
                    m_dev->register_time_interval("GPU", std::move(tsStart), cmd.write_timestamp(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT),
                                                  m_inShadow ? "gs: shadow recording" : "gs: real recording");
            }

            // Batches recorded before Present (buffer flips on the GS thread) wait for the shadows,
            // which are all recorded at Present on the presenter's thread (the GS thread is the
            // busiest: recording them there cost the game frames at 120 Hz).
            void keepForShadows(const std::vector<Batch> &batches, const std::vector<HwVertex> &vertices)
            {
                if (!m_shadows || batches.empty())
                    return;
                const uint32_t base = static_cast<uint32_t>(m_shadowBatchVerts.size());
                m_shadowBatchVerts.insert(m_shadowBatchVerts.end(), vertices.begin(), vertices.end());
                for (const Batch &b : batches)
                {
                    m_shadowBatches.push_back(b);
                    m_shadowBatches.back().firstVertex += base;
                }
            }

            // Re-rendered frame generation: the same batches into each shadow's own targets, with
            // the 3D moved on by (i + 1) / (shadows + 1) of a frame, so each shadow's frame buffers
            // hold the game's frames as they would look that much later.
            void recordShadows(Vulkan::CommandBuffer &cmd, const std::vector<Batch> &batches, const std::vector<HwVertex> &vertices)
            {
                if (batches.empty() || !m_shadows)
                    return;
                const auto t0 = std::chrono::steady_clock::now();
                for (uint32_t i = 0; i < m_shadows; ++i)
                {
                    moveVertices(vertices, static_cast<float>(i + 1) / static_cast<float>(m_shadows + 1), m_shadowVerts);
                    m_set = &m_shadowSets[i];
                    m_inShadow = true;
                    record(cmd, batches, m_shadowVerts);
                    m_inShadow = false;
                    m_set = &m_main;
                }
                m_stats.shadowUs += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            }

            // ---------------------------------------------------------------- scanout
            // The buffer on display into the next scanout image: the real frame's (shared with the
            // presenter, with its motion), or (m_inShadow) shadow m_shadowSlot's.
            const Vulkan::Image *scanout(Vulkan::CommandBuffer &cmd, const GSPresentationRequest &r)
            {
                const bool en1 = r.pmode & 1u, en2 = (r.pmode >> 1) & 1u;
                const uint64_t dispfb = en1 ? r.dispfb1 : (en2 ? r.dispfb2 : r.dispfb1);
                const uint64_t display = en1 ? r.display1 : (en2 ? r.display2 : r.display1);
                const uint32_t fbp = dispfb & 0x1FFu;
                if (!m_inShadow)
                    m_scanFbp = fbp;
                const uint32_t dbx = (dispfb >> 32) & 0x7FFu, dby = (dispfb >> 43) & 0x7FFu;
                const uint32_t magh = ((display >> 23) & 0xFu) + 1u;
                const uint32_t dw = static_cast<uint32_t>((display >> 32) & 0xFFFu) + 1u;
                uint32_t dh = static_cast<uint32_t>((display >> 44) & 0x7FFu) + 1u;
                auto it = m_set->targets.find(fbp);
                if (it == m_set->targets.end() || !it->second.color)
                {
                    static int logged = 0;
                    if (!m_inShadow && logged++ < 5)
                    {
                        std::cerr << "[hwgs] scanout: no target at fbp " << fbp << " (pmode " << std::hex << r.pmode
                                  << " dispfb " << dispfb << std::dec << "); targets:";
                        for (auto &[f, t] : m_set->targets)
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
                // Interlaced fields at 2x and up (not progressive fields): scanned out as
                // paraLLEl-GS does (field-aware high-resolution scanout), half a field line higher in
                // phase 0 than in phase 1, one half line shorter, so the game's half-line offset of
                // alternate fields (sceGsSetHalfOffset) lines the two fields' pictures up.
                const bool interlaced = (r.smode2 & 3u) == 3u && !r.progressiveFields && sy >= 2u;
                const uint32_t yOff = interlaced && g_displayFieldPhase.load(std::memory_order_relaxed) == 0u ? sy / 2u : 0u;
                const uint32_t outW = w * sx, outH = h * sy - (interlaced ? sy / 2u : 0u);
                if (!m_inShadow)
                {
                    m_scanGeom = {dbx, dby, sx, sy, outW, outH, yOff};
                    m_scanoutIndex = (m_scanoutIndex + 1u) % 3u;
                }
                Vulkan::ImageHandle &out = m_inShadow ? m_shadowScan[m_shadowSlot][m_shadowIndex] : m_scanout[m_scanoutIndex];
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
                cmd.copy_image(*out, *t.color, {0, 0, 0}, {static_cast<int32_t>(dbx * sx), static_cast<int32_t>(dby * sy + yOff), 0},
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
                if (m_inShadow)
                    return out.get();
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
                    cmd.copy_image(*mo, *t.motion, {0, 0, 0}, {static_cast<int32_t>(dbx * sx), static_cast<int32_t>(dby * sy + yOff), 0},
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

            void snapshotDepthLocked(Vulkan::CommandBuffer &cmd, uint32_t zbp, uint32_t fbp)
            {
                const ScanGeom &g = m_scanGeom;
                auto it = m_main.depths.find(zbp);
                if (it == m_main.depths.end() || !it->second || !g.outW || !g.outH || !m_depthCopyProgram)
                    return;
                const Vulkan::Image &depth = *it->second;
                m_depthIndex = (m_depthIndex + 1u) % 3u;
                Vulkan::ImageHandle &out = m_depthOut[m_depthIndex];
                if (!out || out->get_width() != g.outW || out->get_height() != g.outH)
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(g.outW, g.outH, VK_FORMAT_R32_SFLOAT);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    out = m_dev->create_image(info);
                }
                cmd.image_barrier(depth, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                cmd.image_barrier(*out, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                  0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                Vulkan::RenderPassInfo rp = {};
                rp.num_color_attachments = 1;
                rp.color_attachments[0] = &out->get_view();
                rp.store_attachments = 1;
                cmd.begin_render_pass(rp);
                cmd.set_program(m_depthCopyProgram);
                cmd.set_opaque_state();
                cmd.set_depth_test(false, false);
                cmd.set_cull_mode(VK_CULL_MODE_NONE);
                cmd.set_texture(0, 0, depth.get_view(), Vulkan::StockSampler::NearestClamp);
                const float dw = static_cast<float>(depth.get_width()), dh = static_cast<float>(depth.get_height());
                const float rect[4] = {g.dbx * g.sx / dw, (g.dby * g.sy + g.yOff) / dh, g.outW / dw, g.outH / dh};
                cmd.push_constants(rect, 0, sizeof(rect));
                cmd.draw(3);
                cmd.end_render_pass();
                cmd.image_barrier(*out, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                cmd.image_barrier(depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
                m_depthReady = out;
                m_depthByFbp[fbp] = out;
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
                    verts[i].y = m_uiDraw[i][1] - static_cast<float>(g.dby) - static_cast<float>(g.yOff) / static_cast<float>(g.sy);
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

            // ---------------------------------------------------------------- pipelines
            // The draw pipeline's state, set on `cmd` the same way for a draw and for the warm-up.
            // DATE through the stencil: before the batch, stencil bit 0 = the target's alpha MSB (from
            // its snapshot) where the batch can draw (its bounding box inside the scissor; shadows
            // cover little of the frame).
            // The whole target's stencil from its alpha MSB (the snapshot `dest`), before a run of
            // DATE draws.
            void dateStencilPrepass(Vulkan::CommandBuffer &cmd, const Vulkan::ImageView &dest, float devW, float devH)
            {
                VkClearRect rect = {};
                rect.rect.extent.width = static_cast<uint32_t>(devW);
                rect.rect.extent.height = static_cast<uint32_t>(devH);
                rect.layerCount = 1;
                VkClearValue zero = {};
                cmd.clear_quad(0, rect, zero, VK_IMAGE_ASPECT_STENCIL_BIT);
                cmd.set_program(m_dateStencilProgram);
                cmd.set_opaque_state();
                cmd.set_cull_mode(VK_CULL_MODE_NONE);
                cmd.set_depth_test(false, false);
                cmd.set_color_write_mask(0u);
                cmd.set_scissor(rect.rect);
                cmd.set_stencil_test(true);
                cmd.set_stencil_ops(VK_COMPARE_OP_ALWAYS, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP);
                cmd.set_stencil_reference(1u, 1u, 1u);
                cmd.set_specialization_constant_mask(0);
                cmd.set_texture(0, 0, dest, sampler(false, false, false, false, false));
                cmd.draw(3);
            }

            // Finding which draw a difference against paraLLEl-GS comes from (scripts/hwgs_ab.py):
            // RT_HWGS_BATCH_LOG=<file> logs every batch recorded (state, bounding box; "record N"
            // before each recording), RT_HWGS_DROP=<TEST hex>,... leaves out the batches with those
            // TEST values (t<hex>), ALPHA values (a<hex>) or TEX0.TBP0 (b<dec>).
            bool debugSkip(const Batch &b, const std::vector<HwVertex> &vertices)
            {
                static FILE *log = [] {
                    const char *p = std::getenv("RT_HWGS_BATCH_LOG");
                    return p && *p ? std::fopen(p, "w") : nullptr;
                }();
                static const std::string drop = [] {
                    const char *e = std::getenv("RT_HWGS_DROP");
                    return e ? "," + std::string(e) + "," : std::string();
                }();
                if (!log && drop.empty())
                    return false;
                const GSContext &c = b.state.context;
                const unsigned long long test = c.test & 0x7FFFFu, alpha = c.alpha & 0xFF000000FFull;
                if (log)
                {
                    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f, z0 = 1e9f, z1 = -1e9f;
                    for (uint32_t i = b.firstVertex; i < b.firstVertex + b.vertexCount && i < vertices.size(); ++i)
                    {
                        x0 = std::min(x0, vertices[i].x), x1 = std::max(x1, vertices[i].x);
                        y0 = std::min(y0, vertices[i].y), y1 = std::max(y1, vertices[i].y);
                        z0 = std::min(z0, vertices[i].z), z1 = std::max(z1, vertices[i].z);
                    }
                    std::fprintf(log,
                                 "frame=%llu fbp=%u zbp=%u prim=%u tme=%u abe=%u fge=%u test=%05llx alpha=%llx zmsk=%u tbp=%u tw=%u th=%u "
                                 "psm=%u tfx=%u tcc=%u tex1=%llx target=%d flat=%d verts=%u box=%.1f,%.1f,%.1f,%.1f z=%.7f..%.7f\n",
                                 static_cast<unsigned long long>(m_frame.load()), c.frame.fbp, c.zbuf.zbp, b.state.prim.type, b.state.prim.tme,
                                 b.state.prim.abe, b.state.prim.fge, test, alpha, c.zbuf.zmask, c.tex0.tbp0, 1u << c.tex0.tw,
                                 1u << c.tex0.th, c.tex0.psm, c.tex0.tfx, c.tex0.tcc,
                                 static_cast<unsigned long long>(c.tex1 & 0xFFFFFFFFFull), b.fromTarget ? 1 : 0, b.flat ? 1 : 0,
                                 b.vertexCount, x0, y0, x1, y1, z0, z1);
                }
                char t[32], a[32], tb[32];
                std::snprintf(t, sizeof(t), ",t%llx,", test);
                std::snprintf(a, sizeof(a), ",a%llx,", alpha);
                std::snprintf(tb, sizeof(tb), ",b%u,", c.tex0.tbp0);
                return !drop.empty() && (drop.find(t) != std::string::npos || drop.find(a) != std::string::npos ||
                                         (b.state.prim.tme && drop.find(tb) != std::string::npos));
            }

            void applyPipe(Vulkan::CommandBuffer &cmd, const PipeKey &k, bool note = true)
            {
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
                cmd.set_depth_test(k.ztest, k.zwrite);
                cmd.set_depth_compare(static_cast<VkCompareOp>(k.zop));
                if (k.blend)
                {
                    cmd.set_blend_enable(true);
                    cmd.set_blend_factors(static_cast<VkBlendFactor>(k.src), VK_BLEND_FACTOR_ONE,
                                          static_cast<VkBlendFactor>(k.dst), VK_BLEND_FACTOR_ZERO);
                    cmd.set_blend_op(static_cast<VkBlendOp>(k.op), VK_BLEND_OP_ADD);
                }
                else
                    cmd.set_blend_enable(false);
                cmd.set_color_write_mask(k.mask);
                // spec[2]: the alpha-test pass (bits 0-7), DATE through the stencil (record()): the
                // shader's DATE split (8-9), the stencil op (10-11: 1 test, 2 test and invert on
                // pass) and DATM, the stencil value that passes (12).
                if (const uint32_t st = (k.spec[2] >> 10) & 3u)
                {
                    cmd.set_stencil_test(true);
                    cmd.set_stencil_ops(VK_COMPARE_OP_EQUAL, st == 2u ? VK_STENCIL_OP_INVERT : VK_STENCIL_OP_KEEP,
                                        VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP);
                    cmd.set_stencil_reference(1u, 1u, static_cast<uint8_t>((k.spec[2] >> 12) & 1u));
                }
                cmd.set_specialization_constant_mask(0xF);
                cmd.set_specialization_constant(0, k.spec[0]);
                cmd.set_specialization_constant(1, k.spec[1]);
                cmd.set_specialization_constant(2, k.spec[2] & 0xFFu);
                cmd.set_specialization_constant(3, (k.spec[2] >> 8) & 3u);
                if (note && !m_pipeStatesPath.empty())
                {
                    const uint64_t h = k.hash();
                    std::lock_guard<std::mutex> lock(m_pipeMutex);
                    if (m_pipeSeen.insert(h).second)
                        m_pipeNew.push_back(k);
                }
            }

            // Compiled pipelines persist across runs in <cache>/hw_pipelines.bin (the presenter's
            // too: it is the presenter's device), saved as the cache grows, since apps are often
            // killed. Driver caches don't move between GPUs or drivers; the states do
            // (<cache>/hw_pipeline_states.txt, PipeKey::text() per line).
            void loadPipelineCache(const std::string &dir)
            {
                if (dir.empty())
                    return;
                std::error_code ec;
                std::filesystem::create_directories(dir, ec);
                m_pipeCachePath = (std::filesystem::path(dir) / "hw_pipelines.bin").string();
                m_pipeStatesPath = (std::filesystem::path(dir) / "hw_pipeline_states.txt").string();
                std::ifstream in(m_pipeCachePath, std::ios::binary);
                std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                m_dev->init_pipeline_cache(data.empty() ? nullptr : data.data(), data.size());
                m_pipeCacheSaved = data.size();
                std::fprintf(stderr, "[hwgs] pipeline cache: %zu KB\n", data.size() >> 10);
            }

            void savePipelineCache()
            {
                if (m_pipeCachePath.empty())
                    return;
                const size_t size = m_dev->get_pipeline_cache_size();
                if (size && size != m_pipeCacheSaved)
                {
                    std::vector<uint8_t> data(size);
                    if (m_dev->get_pipeline_cache_data(data.data(), size))
                    {
                        const std::string tmp = m_pipeCachePath + ".tmp";
                        bool ok;
                        {
                            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
                            out.write(reinterpret_cast<const char *>(data.data()), std::streamsize(size));
                            ok = static_cast<bool>(out);
                        }
                        std::error_code ec;
                        if (ok)
                            std::filesystem::rename(tmp, m_pipeCachePath, ec);
                        if (ok && !ec)
                            m_pipeCacheSaved = size;
                    }
                }
                std::vector<PipeKey> fresh;
                {
                    std::lock_guard<std::mutex> lock(m_pipeMutex);
                    fresh.swap(m_pipeNew);
                }
                if (!fresh.empty())
                {
                    std::ofstream out(m_pipeStatesPath, std::ios::app);
                    for (const PipeKey &k : fresh)
                        out << k.text() << '\n';
                }
            }

            // Every known state compiled on a worker (RT_HWGS_WARMUP=0: none). The state is set on a
            // command buffer in a render pass of 1x1 images with the real targets' formats, taken
            // out (extract_pipeline_state: the same hash a draw looks up) and the command buffer
            // dropped; the worker builds each pipeline unless a draw got there first.
            void warmUpPipelines()
            {
                std::vector<PipeKey> keys;
                std::unordered_set<uint64_t> seen;
                auto add = [&](const char *line) {
                    PipeKey k;
                    if (PipeKey::parse(line, k) && seen.insert(k.hash()).second)
                        keys.push_back(k);
                };
                for (const char *line : kBuiltinPipeStates)
                    add(line);
                if (!m_pipeStatesPath.empty())
                {
                    std::ifstream in(m_pipeStatesPath);
                    for (std::string line; std::getline(in, line);)
                        add(line.c_str());
                }
                {
                    std::lock_guard<std::mutex> lock(m_pipeMutex);
                    m_pipeSeen = seen; // only new states are written to the file
                }
                const char *e = std::getenv("RT_HWGS_WARMUP");
                std::vector<Vulkan::DeferredPipelineCompile> compiles;
                if (!keys.empty() && !(e && *e == '0'))
                {
                    auto target = [&](VkFormat format) {
                        return m_dev->create_image(Vulkan::ImageCreateInfo::render_target(1, 1, format));
                    };
                    auto color = target(VK_FORMAT_R8G8B8A8_UNORM), motion = target(VK_FORMAT_R16G16_SFLOAT),
                         depth = target(m_depthFormat);
                    Vulkan::BufferCreateInfo vb = {};
                    vb.size = 3 * sizeof(HwVertex);
                    vb.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
                    vb.domain = Vulkan::BufferDomain::Device;
                    auto vbo = m_dev->create_buffer(vb);
                    auto cmd = m_dev->request_command_buffer();
                    for (uint32_t pass = 0; pass < 4; ++pass)
                    {
                        const bool withDepth = pass & 1u, withMotion = pass & 2u;
                        Vulkan::RenderPassInfo rp = {};
                        rp.num_color_attachments = withMotion ? 2 : 1;
                        rp.color_attachments[0] = &color->get_view();
                        if (withMotion)
                            rp.color_attachments[1] = &motion->get_view();
                        if (withDepth)
                            rp.depth_stencil = &depth->get_view();
                        bool open = false;
                        for (const PipeKey &k : keys)
                        {
                            if (k.depth != withDepth || k.motion != withMotion)
                                continue;
                            if (!open)
                                cmd->begin_render_pass(rp), open = true;
                            applyPipe(*cmd, k, false);
                            cmd->set_vertex_binding(0, *vbo, 0, sizeof(HwVertex));
                            compiles.emplace_back();
                            cmd->extract_pipeline_state(compiles.back());
                        }
                        if (open)
                            cmd->end_render_pass();
                    }
                    m_dev->submit_discard(cmd);
                }
                m_pipeTotal = static_cast<uint32_t>(compiles.size());
                m_pipeThread = std::thread([this, compiles = std::move(compiles)] {
#if defined(__linux__)
                    setpriority(PRIO_PROCESS, 0, 5); // (this thread) behind the game's
#endif
                    const auto start = std::chrono::steady_clock::now();
                    uint32_t built = 0;
                    for (const auto &c : compiles)
                    {
                        if (m_pipeStop)
                            break;
                        if (c.program->get_pipeline(c.hash).pipeline == VK_NULL_HANDLE &&
                            Vulkan::CommandBuffer::build_graphics_pipeline(m_dev, c, Vulkan::CommandBuffer::CompileMode::AsyncThread).pipeline)
                            ++built;
                        m_pipeDone.fetch_add(1, std::memory_order_release);
                    }
                    m_pipeDone.store(m_pipeTotal, std::memory_order_release); // (stopped early: nothing to wait for)
                    if (!compiles.empty())
                        savePipelineCache(); // at once: the app may be killed before the next save
                    if (!compiles.empty())
                        std::fprintf(stderr, "[hwgs] pipelines: %zu known states, %u compiled ahead in %.0f ms\n", compiles.size(), built,
                                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                    while (!m_pipeStop)
                    {
                        std::unique_lock<std::mutex> lock(m_pipeMutex);
                        m_pipeCv.wait_for(lock, std::chrono::seconds(10), [this] { return m_pipeStop.load(); });
                        lock.unlock();
                        savePipelineCache();
                    }
                });
            }

            void stopPipelineWorker()
            {
                if (!m_pipeThread.joinable())
                    return;
                {
                    std::lock_guard<std::mutex> lock(m_pipeMutex);
                    m_pipeStop = true;
                }
                m_pipeCv.notify_all();
                m_pipeThread.join();
                savePipelineCache();
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
            // Pipelines: the persistent cache, the states drawn (noted for the next start), the
            // worker that compiles the known ones ahead and saves both.
            std::string m_pipeCachePath, m_pipeStatesPath;
            size_t m_pipeCacheSaved = 0;
            std::mutex m_pipeMutex;
            std::condition_variable m_pipeCv;
            std::unordered_set<uint64_t> m_pipeSeen;
            std::vector<PipeKey> m_pipeNew;
            std::atomic<bool> m_pipeStop{false};
            uint32_t m_pipeTotal = 0;                // states the worker compiles at start ...
            std::atomic<uint32_t> m_pipeDone{0};     // ... and how many it got through

        public:
            HwPipelinePrep pipelinePrep() const
            {
                return {m_pipeTotal, std::min(m_pipeTotal, m_pipeDone.load(std::memory_order_acquire))};
            }

        private:
            std::thread m_pipeThread;
            Vulkan::ImageHandle m_white;
            std::unordered_map<uint32_t, Vulkan::SamplerHandle> m_samplers;
            std::atomic<uint32_t> m_scale{2};
            std::atomic<bool> m_rescale{false};
            // Progressive fields (the default) until the first Present says otherwise; RT_PROGRESSIVE_FIELDS=0
            // (the regression harness) says so before any target is made (drawing doesn't depend on
            // when the host first presents: save states replay the same pictures).
            static bool interlacedFromEnv()
            {
                const char *pf = std::getenv("RT_PROGRESSIVE_FIELDS");
                return pf && std::strcmp(pf, "0") == 0;
            }
            std::atomic<bool> m_fieldAware{interlacedFromEnv()}; // interlaced fields shown as fields (Snap)
            std::atomic<uint32_t> m_yScale{interlacedFromEnv() ? 1u : 2u};

            GSCpuBackend m_cpu; // transfers and readbacks over the shared local memory
            std::vector<SavedImage> m_loadedTargets, m_loadedDepths; // a state's targets, until StateLoaded
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
            std::unordered_map<uint64_t, TexRect> m_repainted; // GS thread: what the decode hook changes in submitted textures
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
                uint64_t vboWaits = 0, vboWaitUs = 0, shadowUs = 0; // frame generation's costs
                std::atomic<uint64_t> motionVertices{0};
            } m_stats;
            uint64_t m_breakEpoch = 0;
            uint64_t m_lastEviction = 0;
            // GS thread: the state (and resolved texture) of the newest batch.
            Batch m_current;
            std::vector<HwVertex> m_runConverted; // SubmitRun's vertices, converted once
            bool m_haveCurrent = false;
            uint64_t m_currentSerial = 0;
            uint64_t m_currentEpoch = 0;
            mutable std::mutex m_targetPagesMutex;

            // Recording (device lock). m_set: the set being recorded (the real one, or a shadow's).
            static constexpr uint32_t kMaxShadows = 3;
            TargetSet m_main, m_shadowSets[kMaxShadows];
            TargetSet *m_set = &m_main;
            uint32_t m_shadows = 0; // shadow frames rendered (PgsShared::wantShadows)
            bool m_inShadow = false;
            std::vector<HwVertex> m_shadowVerts;
            std::vector<Batch> m_shadowBatches; // this frame's batches so far, for the shadows
            std::vector<HwVertex> m_shadowBatchVerts;
            Vulkan::ImageHandle m_shadowScan[kMaxShadows][3];
            uint32_t m_shadowIndex = 0, m_shadowSlot = 0;
            uint32_t m_shadowsWanted = 0; // shadows asked for (m_shadows: 0 while paused)
            uint32_t m_shadowWarm = 0; // presents since the shadows (re)started: their buffers fill first
            const Vulkan::Image *m_passColor = nullptr, *m_passDepth = nullptr;
            Vulkan::ImageHandle m_scanout[3];
            // The UI mask: the HUD's triangles (GS frame coordinates), staged by the GS thread,
            // handed over with the batches, drawn at Present; the scanout's geometry they map to.
            std::vector<std::array<float, 2>> m_uiStage, m_uiVerts, m_uiDraw;
            Vulkan::ImageHandle m_uiMask[3];
            struct ScanGeom
            {
                uint32_t dbx = 0, dby = 0, sx = 1, sy = 1, outW = 0, outH = 0;
                uint32_t yOff = 0; // device rows skipped at the top (interlaced phase 0)
            } m_scanGeom;
            // Motion vectors (PgsShared::wantMotion): per-target attachments, the scanned-out part.
            bool m_motionOn = false;
            const Vulkan::Image *m_passMotion = nullptr;
            Vulkan::ImageHandle m_motionScan[3], m_motionOut;
            // Depth for temporal upscalers (WantsDepthSnapshot): requested by the GS thread where the
            // HUD starts, copied after the frame's 3D is recorded.
            Vulkan::Program *m_depthCopyProgram = nullptr;
            Vulkan::Program *m_dateStencilProgram = nullptr; // DATE: the stencil from the alpha MSB
            bool m_stencilDate = false;                      // DATE through the stencil (D32S8 Z buffers)
            uint64_t m_depthGeneration = 1;                  // bumped when a Z buffer is made or loaded
            uint64_t m_stencilPrepasses = 0;
            VkFormat m_depthFormat = VK_FORMAT_D32_SFLOAT;
            bool m_depthRequested = false;
            uint32_t m_depthRequestZbp = 0, m_depthRequestFbp = 0, m_depthIndex = 0;
            std::map<uint32_t, Vulkan::ImageHandle> m_depthByFbp; // the latest snapshot per 3D frame buffer
            Vulkan::ImageHandle m_depthOut[3], m_depthReady;
            uint32_t m_scanFbp = 0; // the frame buffer the last scanout showed
            Target m_mapLayers[2];
            uint32_t m_mapIndex = 0, m_mapMissed = 0;
            bool m_mapDrawn = false;
            uint32_t m_mapFbp = 0;       // the frame buffer the map being drawn belongs to
            bool m_mapPublished = false; // a map went to the second screen since the last Present
            float m_mapBox[4] = {};
            float m_mapStable[4] = {}; // the fixed box of the map on show
            bool m_mapClipped[4] = {};  // this frame's map edges the scissor cut
            uint32_t m_scanoutIndex = 0;
            Vulkan::BufferHandle m_readback;
            static constexpr uint32_t kVboRing = 32; // shadow frames upload their own vertices too
            struct VboSlot
            {
                Vulkan::BufferHandle buffer;
                Vulkan::Fence fence;
            } m_vbo[kVboRing];
            uint32_t m_vboSlot = 0;
            std::vector<uint32_t> m_vboPending; // slots written since the last submit
        };
    }

    bool hwPipelinePrep(GSRasterBackend *backend, HwPipelinePrep &out)
    {
        const auto *hw = dynamic_cast<const HwBackend *>(backend);
        if (!hw)
            return false;
        out = hw->pipelinePrep();
        return true;
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

    bool hwPipelinePrep(GSRasterBackend *, HwPipelinePrep &)
    {
        return false;
    }
}

#endif
