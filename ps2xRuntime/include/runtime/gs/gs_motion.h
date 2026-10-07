#pragma once

// Per-vertex screen motion for temporal upscalers (Road Trip recomp).
//
// Road Trip's VU1 microprogram draws every 3D vertex as ftoi4(C * v / w) with one 4x4 matrix C per
// object (camera x world), set up by a "setup" MSCAL and used by the vertex MSCALs after it. At
// each MSCAL the tracker snapshots VU1 data memory and works out C, and the C the same object had
// last frame (objects are matched by draw order and world position). The GS frontend then turns
// each XGKICKed vertex back into object space with C^-1 and re-projects it with last frame's C:
// the difference is the vertex's motion on screen (cars included, not just the camera's).
//
// Microprogram facts (see the plan's Phase 5): MSCAL 2 = setup B (C = VU4..7 x VU0..3), E = setup A
// (C = VU12..15 x VU8..11 x VU0..3), 0 = init (C = VU12..15 x VU8..11); vertex loops 4 and A use C,
// 8 and 6 add the batch offset VU26 to positions (C x T(VU26)), C is the sky. Matrices are stored as
// four column vectors (sceVu0 layout).

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace ps2x::gs
{
    struct Mat4
    {
        double m[16]; // column-major: m[col * 4 + row]
    };

    struct MotionContext
    {
        Mat4 cur;      // this frame's C
        Mat4 curInv;   // its inverse
        Mat4 prev;     // last frame's C for the same object (== cur when unmatched: no motion)
        float jitter[2] = {}; // the camera jitter (GS pixels) of the frame this run belongs to
    };

    class MotionTracker
    {
    public:
        static MotionTracker &instance();

        void setEnabled(bool enabled);
        bool enabled() const { return m_enabled.load(std::memory_order_relaxed); }

        // GIF/VIF1 worker, before each VU1 MSCAL run: the context id to mark that run's XGKICKs
        // with (0 = none: setup runs, or not enabled).
        uint32_t onMscal(uint32_t startPC, const uint8_t *vuData);
        // EE, at the game's clear (the VIF queue is drained): this frame's objects become last
        // frame's.
        void frameStart();
        // EE, as the player camera is built: the jitter it gets (GS pixels). The VU1 runs that
        // follow carry it, so the GS thread, which lags the EE by up to a frame, knows the jitter
        // of the 3D it is drawing.
        void noteCameraJitter(float x, float y);
        // GS thread: a context by id (false if unknown or overwritten).
        bool context(uint32_t id, MotionContext &out) const;

        // Reserved per-vertex values (NaN half pairs, never real motion): the vertex belongs to the
        // UI (shown as drawn, not post-processed), or leaves the UI mask as it is (full-screen
        // fades and post passes). See WideLayout::VertexClass.
        static constexpr uint32_t kVertexUi = 0x7E017E01u;
        static constexpr uint32_t kVertexNeutral = 0x7E027E02u;

        // GIF packet marker: an A+D write to this unused GS register carries the context id.
        static constexpr uint8_t kMarkerRegister = 0x7E;
        static void makeMarker(uint32_t id, uint8_t out[32]);
        // If `data` is a marker packet, its id (else false).
        static bool readMarker(const uint8_t *data, uint32_t size, uint32_t &id);

        // Motion of each vertex (XYZ2/XYZF2/XYZ3/XYZF3, in kick order) of a PATH1 GIF packet, as
        // packed half2 (dx, dy) in GS pixels: current minus previous position.
        static void packetMotion(const uint8_t *data, uint32_t size, const MotionContext &ctx, std::vector<uint32_t> &out);
        // Re-rendered frame generation: a copy of a PATH1 GIF packet with every vertex re-projected
        // with the object's matrix moved on by t frames (C + t (C - C_prev): t = 0.5 is half a
        // frame ahead) on screen (Z kept, so coplanar decals stay equal), and perspective texture
        // coordinates (PACKED ST/Q) rescaled for the new w.
        // `keep` (optional, one per vertex in kick order, like packetMotion's output): vertices
        // that stay as they are (the HUD).
        static void packetReproject(const uint8_t *data, uint32_t size, const MotionContext &ctx, double t,
                                    std::vector<uint8_t> &out, const std::vector<uint8_t> *keep = nullptr);

        struct Stats
        {
            uint32_t objects = 0, matched = 0;
        };
        Stats lastFrameStats() const;

    private:
        enum class Kind : uint8_t { SetupB, SetupA, Init };
        struct Object
        {
            Kind kind;
            double tx, ty, tz; // world translation (W's last column)
            double r[9];       // world rotation (W's 3x3, columns normalised)
            Mat4 base;         // C before any batch offset
        };

        Mat4 matchPrevious(const Object &obj, uint32_t ordinal) const;

        std::atomic<bool> m_enabled{false};
        mutable std::mutex m_mutex;
        std::vector<Object> m_cur[3], m_prev[3];
        // The object whose vertex MSCALs follow.
        bool m_haveActive = false;
        Kind m_activeKind = Kind::Init;
        Mat4 m_activeBase{}, m_activePrevBase{};
        // Contexts by id (a ring; ids start at 1).
        static constexpr uint32_t kRing = 1u << 14;
        std::vector<MotionContext> m_ring = std::vector<MotionContext>(kRing);
        // Written by the VU1 thread under m_mutex; context() reads the ring without the lock (the
        // GS thread asks once per PATH1 packet, and waiting behind onMscal's matching cost it
        // ~40% of its time on the Thor): the entry is published by the release store of m_nextId.
        std::atomic<uint32_t> m_nextId{1};
        float m_cameraJitter[2] = {};
        Stats m_stats{}, m_lastStats{};
    };
}
