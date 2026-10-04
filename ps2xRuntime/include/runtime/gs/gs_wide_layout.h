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
//    middle third -> centred). A wide 3D-drawn HUD element (the town minimap, PATH1, scissored to
//    its frame) takes the anchor of the HUD element drawn just before it. Full-screen primitives (fades, the final post-pass) are left
//    to stretch, as are full-screen scissors.
//  - 2D-backed frames (title, menus, shops): nothing is changed; the presenter shows them 4:3
//    (pillarboxed) instead (frameIs2D).
//
// See the HUD spike in the recomp's plan for the evidence behind these rules. Not thread-safe: it
// runs inside the GS frontend's packet processing (under its state lock).

#include <cstdint>

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
            bool ctxt = false;
            uint32_t ofx[2] = {0, 0}; // XYOFFSET_1/2 OFX and OFY (12.4 fixed point)
            uint32_t ofy[2] = {0, 0};
        };

        // Rewrites the X of 2D vertices (and HUD scissors) in a GIF packet in place. pathIndex as
        // GS::processGIFPacket: 0 = PATH1 (VU1), 1 = PATH2, 2 = PATH3. `state` is the GS state at
        // the start of the packet.
        void transformPacket(uint32_t pathIndex, uint8_t *data, uint32_t sizeBytes, const PrimState &state);

        // The game cleared the screen to start a frame (GS::markFrameStart, from its clear routine).
        void frameStart();
        // The guest showed a new frame (the presenter latched it). Frames drawn without the clear
        // are 2D-backed: the title draws a full-screen picture instead.
        void framePresented();

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
        };

        void flushUnit(uint32_t pathIndex, const PrimState &st);
        float toScreen(uint16_t gsX, bool ctx2, const PrimState &st) const;
        uint16_t fromScreen(float x, bool ctx2, const PrimState &st) const;
        float place(float x, Anchor a) const;
        void transformScissor(uint8_t *value);
        void vote(bool looks2D);

        float m_k = 1.0f;
        HudPlacement m_placement = HudPlacement::Centred;
        Mode m_mode = Mode::Unknown;
        bool m_hud = false;
        Anchor m_lastAnchor = Anchor::Centre;
        bool m_lastFrame2D = false;
        uint32_t m_clears = 0;
        uint32_t m_2DVotes = 0, m_driveVotes = 0;
        float m_dbgMinY = 0;   // clears since the last presented frame
        Vertex m_unit[2048];
        uint32_t m_unitCount = 0;
    };
}
