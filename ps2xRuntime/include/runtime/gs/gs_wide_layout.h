#pragma once

// Widescreen without stretching the game's 2D. The GS frame stays 640 wide (PS2 units) and the
// presenter stretches it to the wide window, so with the 3D camera widened (a game hook) the
// world fills the screen. Everything 2D must then be drawn narrower by k = (4/3) / aspect so it
// shows at its true shape after the stretch:
//
//  - Driving frames (the first draw after the game's clear, frameStart, is 3D, PATH1): the HUD phase starts at the
//    first textured PATH2/3 draw in context 1; from there to the end of the frame every primitive
//    is HUD. Each HUD primitive is narrowed either around the screen centre ("4:3 centred") or
//    towards its own edge ("screen edges": left third -> left edge, right third -> right edge,
//    middle third -> centred). With edges, the primitives of one packet that touch or overlap
//    form one element and move together, by the element's position (a message window is a
//    frame of tiles with its text on top: it stays whole, centred). A wide 3D-drawn HUD element
//    (the town minimap, PATH1, scissored to its frame) takes the anchor of the HUD element drawn
//    just before it. Full-screen primitives (fades, the final post-pass) are left to stretch, as
//    are full-screen scissors.
//  - 2D-backed frames (title, menus, shops): nothing is changed; the presenter shows them 4:3
//    (pillarboxed) instead (lastFrameWas2D). A frame whose first draw is a full-screen 2D picture
//    (a texture outside the frame buffers) is 2D-backed from that draw on, so a new screen is
//    never shown stretched; a frame whose first draw is 3D is a driving one at once. A frame
//    whose first draw is a blended full-screen fill (the fade to "Now loading" over the last
//    picture) keeps the verdict it has. Other 2D-first frames vote, with hysteresis.
//    RT_WIDE_DEBUG=1 logs verdict changes, =2 also each frame's first draw.
//
// See the HUD spike in the recomp's plan for the evidence behind these rules. Not thread-safe: it
// runs inside the GS frontend's packet processing (under its state lock).

#include <cstdint>
#include <vector>

namespace ps2x::gs
{
    enum class HudPlacement : uint8_t
    {
        Centred, // the HUD keeps its 4:3 layout in the middle of the wide screen
        Edges    // HUD elements move out to the screen edges on their side
    };

    class WideLayout
    {
    public:
        // aspect: the window's (e.g. 16/9). At 4:3 (or narrower) the layout is off.
        void configure(float aspect, HudPlacement placement);
        bool active() const { return m_k < 0.999f; }
        float horizontalScale() const { return m_k; } // (4/3) / aspect

        struct PrimState
        {
            uint32_t type = 0; // GS PRIM.PRIM (0 point .. 6 sprite)
            bool tme = false;
            bool abe = false; // alpha blending (a fill that fades what is there)
            bool ctxt = false;
            uint32_t ofx[2] = {0, 0}; // XYOFFSET_1/2 OFX and OFY (12.4 fixed point)
            uint32_t ofy[2] = {0, 0};
            uint32_t tbp[2] = {0, 0}; // TEX0_1/2 TBP0 (blocks)
        };

        // Rewrites the X of 2D vertices (and HUD scissors) in a GIF packet in place. pathIndex as
        // GS::processGIFPacket: 0 = PATH1 (VU1), 1 = PATH2, 2 = PATH3. `state` is the GS state at
        // the start of the packet.
        void transformPacket(uint32_t pathIndex, uint8_t *data, uint32_t sizeBytes, const PrimState &state);

        // What each vertex of the last transformed packet belongs to (in kick order), when
        // requested: the scene (processed by AA/upscalers), the UI (HUD, 2D screens: shown as
        // drawn), or neither (full-screen fades and post passes: leave the UI mask as it is).
        enum class VertexClass : uint8_t { Scene, Ui, Neutral };
        void setRecordClasses(bool on) { m_recordClasses = on; }
        const std::vector<VertexClass> &vertexClasses() const { return m_classes; }
        // A PATH1 packet (the 3D) needs no pass once the frame's kind is known and its HUD hasn't
        // begun: every vertex is Scene in a driving frame, Ui on a 2D screen, and nothing else
        // changes. skipPacket() records that: no classes, defaultClass() for every vertex.
        bool plainPath1() const { return m_mode != Mode::Unknown && (m_mode == Mode::Screen2D || !m_hud); }
        void skipPacket()
        {
            m_classes.clear();
            m_defaultClass = m_mode == Mode::Driving ? VertexClass::Scene : VertexClass::Ui;
        }
        VertexClass defaultClass() const { return m_defaultClass; }

        // The game cleared the screen to start a frame (GS::markFrameStart, from its clear routine).
        void frameStart();
        // The guest showed a new frame (the presenter latched it). Frames drawn without the clear
        // are 2D-backed: the title draws a full-screen picture instead.
        void framePresented();

        // True once per driving frame, after the packet in which its HUD phase began (the 3D is
        // complete before that packet: the time to keep its depth).
        bool consumeHudStart()
        {
            const bool started = m_hudStarted;
            m_hudStarted = false;
            return started;
        }

        // The last finished frame was 2D-backed (show it 4:3), or a driving frame (show it wide).
        bool lastFrameWas2D() const { return m_lastFrame2D; }
        // A driving frame is being drawn or was the last one (the camera hook widens only then).
        bool driving() const { return !m_lastFrame2D; }

    private:
        enum class Mode : uint8_t { Unknown, Driving, Screen2D };
        enum class Anchor : uint8_t { Centre, Left, Right };

        struct Vertex
        {
            uint8_t *x;    // where X (16 bits, 12.4 fixed point) is stored
            bool ctx2;     // which XYOFFSET applies
            float y;       // screen Y
        };
        // HUD primitives of the packet being transformed, placed at its end (resolvePacket).
        struct PendingVertex
        {
            uint8_t *x;
            uint32_t ofx;
        };
        struct PendingUnit
        {
            uint32_t first, count; // in m_pendVerts
            float minX, maxX, minY, maxY;
            bool follow;           // a wide 3D-drawn element: takes the anchor drawn before it
            uint32_t group;        // the element it belongs to
        };
        struct PendingScissor
        {
            uint8_t *value;
            uint32_t unitsBefore;  // pending units before it (it takes the last one's anchor)
        };

        void flushUnit(uint32_t pathIndex, const PrimState &st);
        void resolvePacket();
        void parsePacket(uint32_t pathIndex, uint8_t *data, uint32_t sizeBytes, PrimState &st);
        void setVerdict(bool is2D);
        float toScreen(uint16_t gsX, bool ctx2, const PrimState &st) const;
        float place(float x, Anchor a) const;
        void transformScissor(uint8_t *value, Anchor anchor) const;
        void vote(bool looks2D);

        float m_k = 1.0f;
        HudPlacement m_placement = HudPlacement::Centred;
        Mode m_mode = Mode::Unknown;
        bool m_hud = false;
        Anchor m_lastAnchor = Anchor::Centre;
        bool m_lastFrame2D = false;
        uint32_t m_clears = 0;
        uint32_t m_2DVotes = 0, m_driveVotes = 0;
        bool m_hudStarted = false;
        bool m_recordClasses = false;
        VertexClass m_defaultClass = VertexClass::Scene; // past the end of m_classes
        std::vector<VertexClass> m_classes;
        Vertex m_unit[2048];
        uint32_t m_unitCount = 0;
        uint32_t m_unitOverflow = 0; // vertices beyond m_unit's room (still counted for the classes)
        std::vector<PendingVertex> m_pendVerts;
        std::vector<PendingUnit> m_pendUnits;
        std::vector<PendingScissor> m_pendScissors;
        std::vector<Anchor> m_groupAnchor;
    };
}
