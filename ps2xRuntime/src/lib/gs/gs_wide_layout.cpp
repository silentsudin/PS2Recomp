#include "runtime/gs/gs_wide_layout.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2x::gs
{
    namespace
    {
        constexpr float kWidth = 640.0f;
        constexpr float kFullScreen = 600.0f; // wider than this: a fade or post-pass, left to stretch
        constexpr float kFullHeight = 200.0f; // of the 224-line field: a full-screen picture
        constexpr uint32_t kTextureArea = 0x1A40; // blocks below: frame buffers and Z (post-passes sample them)
        constexpr float kTouch = 1.0f;        // primitives this close belong to one HUD element

        // GIF PACKED register descriptors and GS register addresses (A+D).
        enum : uint8_t
        {
            kPrim = 0x00, kXyzf2 = 0x04, kXyz2 = 0x05, kTex0_1 = 0x06, kTex0_2 = 0x07, kXyzf3 = 0x0C, kXyz3 = 0x0D, kAd = 0x0E,
        };
        enum : uint8_t
        {
            kRegPrim = 0x00, kRegXyzf2 = 0x04, kRegXyz2 = 0x05, kRegTex0_1 = 0x06, kRegTex0_2 = 0x07, kRegXyzf3 = 0x0C, kRegXyz3 = 0x0D,
            kRegXyoffset1 = 0x18, kRegXyoffset2 = 0x19, kRegScissor1 = 0x40, kRegScissor2 = 0x41,
        };
        constexpr uint32_t kSprite = 6;

        uint64_t load64(const uint8_t *p)
        {
            uint64_t v;
            std::memcpy(&v, p, 8);
            return v;
        }

        uint16_t load16(const uint8_t *p)
        {
            uint16_t v;
            std::memcpy(&v, p, 2);
            return v;
        }

        void store16(uint8_t *p, uint16_t v) { std::memcpy(p, &v, 2); }

        void setPrim(WideLayout::PrimState &st, uint64_t prim)
        {
            st.type = static_cast<uint32_t>(prim & 7u);
            st.tme = ((prim >> 4) & 1u) != 0;
            st.abe = ((prim >> 6) & 1u) != 0;
            st.ctxt = ((prim >> 9) & 1u) != 0;
        }
    }

    void WideLayout::setVerdict(bool is2D)
    {
        static const bool debug = [] { const char *e = std::getenv("RT_WIDE_DEBUG"); return e && *e >= '1'; }();
        if (debug && is2D != m_lastFrame2D)
            std::fprintf(stderr, "[wide] %s\n", is2D ? "2D-backed: shown 4:3" : "driving: shown wide");
        m_lastFrame2D = is2D;
    }

    void WideLayout::vote(bool looks2D)
    {
        // Hysteresis: a few odd frames (a town frame that starts with a 2D draw, a frame drawn
        // without the clear) must not flip the picture between wide and 4:3.
        if (looks2D)
        {
            m_driveVotes = 0;
            if (++m_2DVotes >= 10)
                setVerdict(true);
        }
        else
        {
            m_2DVotes = 0;
            if (++m_driveVotes >= 2)
                setVerdict(false);
        }
    }

    void WideLayout::frameStart()
    {
        static const bool debug = [] { const char *e = std::getenv("RT_WIDE_DEBUG"); return e && *e >= '1'; }();
        if (debug && m_mode != Mode::Unknown && (m_mode == Mode::Screen2D) != m_lastFrame2D)
            std::fprintf(stderr, "[wide] a %s frame shown %s\n", m_mode == Mode::Screen2D ? "2D-first" : "3D-first",
                         m_lastFrame2D ? "4:3" : "wide");
        if (m_mode != Mode::Unknown)
            vote(m_mode == Mode::Screen2D);
        ++m_clears;
        m_mode = Mode::Unknown;
        m_hud = false;
        m_lastAnchor = Anchor::Centre;
    }

    void WideLayout::framePresented()
    {
        if (m_clears == 0)
            vote(true);
        m_clears = 0;
    }

    void WideLayout::configure(float aspect, HudPlacement placement)
    {
        m_k = aspect > 4.0f / 3.0f + 0.01f ? (4.0f / 3.0f) / aspect : 1.0f;
        m_placement = placement;
    }

    float WideLayout::toScreen(uint16_t gsX, bool ctx2, const PrimState &st) const
    {
        return (static_cast<float>(gsX) - static_cast<float>(st.ofx[ctx2 ? 1 : 0])) / 16.0f;
    }

    float WideLayout::place(float x, Anchor a) const
    {
        switch (a)
        {
        case Anchor::Left: return x * m_k;
        case Anchor::Right: return kWidth - (kWidth - x) * m_k;
        default: return kWidth * 0.5f + (x - kWidth * 0.5f) * m_k;
        }
    }

    void WideLayout::flushUnit(uint32_t pathIndex, const PrimState &st)
    {
        if (m_unitCount == 0)
            return;
        const uint32_t n = m_unitCount;
        const uint32_t total = n + m_unitOverflow;
        m_unitCount = 0;
        m_unitOverflow = 0;
        // The class of this unit's vertices (setRecordClasses), decided below.
        VertexClass cls = VertexClass::Scene;
        struct Record
        {
            WideLayout *self;
            uint32_t n;
            VertexClass *cls;
            ~Record()
            {
                if (self->m_recordClasses)
                    self->m_classes.insert(self->m_classes.end(), n, *cls);
            }
        } record{this, total, &cls};

        float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
        for (uint32_t i = 0; i < n; ++i)
        {
            const float x = toScreen(load16(m_unit[i].x), m_unit[i].ctx2, st);
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
            minY = std::min(minY, m_unit[i].y);
            maxY = std::max(maxY, m_unit[i].y);
        }
        const float width = maxX - minX;

        if (m_mode == Mode::Unknown)
        {
            m_mode = pathIndex == 0 ? Mode::Driving : Mode::Screen2D;
            // Decided at the frame's first draw, so the frame is presented by its own verdict
            // (votes come at the next clear, after it was shown): 3D first is a driving frame; a
            // full-screen 2D picture first (title, menus, shop interiors) is a 2D-backed one. A
            // texture in the frame buffers' area is a post-pass (a late one of the frame before),
            // which proves nothing. Other 2D-first frames only vote (votes are counted at the
            // next clear, so m_2DVotes >= 1 means the frame before was 2D-first too).
            const uint32_t tbp = st.tbp[st.ctxt ? 1 : 0];
            static const bool firstDrawDebug = [] { const char *e = std::getenv("RT_WIDE_DEBUG"); return e && *e == '2'; }();
            if (firstDrawDebug)
                std::fprintf(stderr, "[wide] first draw: path %u prim %u tme %d abe %d ctxt %d tbp %#x box %.0f..%.0f x %.0f..%.0f\n",
                             pathIndex, st.type, st.tme ? 1 : 0, st.abe ? 1 : 0, st.ctxt ? 1 : 0, tbp, minX, maxX, minY, maxY);
            const bool fullScreen = width >= kFullScreen && maxY - minY >= kFullHeight;
            if (m_mode == Mode::Driving)
            {
                m_2DVotes = 0;
                setVerdict(false);
            }
            else if (fullScreen && st.abe && !st.tme)
            {
                // A blended full-screen fill first (the fade to "Now loading" over the last
                // picture): it shows the picture before it, so the frame keeps that picture's
                // verdict (it was squeezed to 4:3 for the fade). Kept wide, it is drawn as a
                // driving frame, so the text on it is HUD and keeps its shape. (A copy of the
                // frame buffer first is no such case: a late post-pass, before the 3D.)
                m_mode = m_lastFrame2D ? Mode::Screen2D : Mode::Driving;
            }
            else if (fullScreen &&
                     ((st.tme && tbp >= kTextureArea) || (!st.tme && m_2DVotes >= 1)))
            {
                // A full-screen fill (the black behind "Now loading") proves it from the second
                // such frame on: a single 2D-first frame in a driving scene keeps its verdict.
                m_driveVotes = 0;
                m_2DVotes = std::max<uint32_t>(m_2DVotes, 10);
                setVerdict(true);
            }
        }
        if (m_mode != Mode::Driving)
        {
            cls = VertexClass::Ui; // a 2D-backed screen: all of it is shown as drawn
            return;
        }
        if (!m_hud && pathIndex >= 1 && !st.ctxt && st.tme)
        {
            m_hud = true;
            m_hudStarted = true;
        }
        // Full-screen sprites (the fade, the final post-pass) stretch with the 3D.
        if (m_hud)
            cls = (st.type == kSprite && width >= kFullScreen) ? VertexClass::Neutral : VertexClass::Ui;
        if (!m_hud || (st.type == kSprite && width >= kFullScreen) || !active())
            return;

        // Placed at the end of the packet, once its elements are known (resolvePacket).
        PendingUnit u;
        u.first = static_cast<uint32_t>(m_pendVerts.size());
        u.count = n;
        u.minX = minX;
        u.maxX = maxX;
        u.minY = minY;
        u.maxY = maxY;
        u.follow = pathIndex == 0 && width > kWidth / 2.0f;
        u.group = 0;
        for (uint32_t i = 0; i < n; ++i)
            m_pendVerts.push_back({m_unit[i].x, st.ofx[m_unit[i].ctx2 ? 1 : 0]});
        m_pendUnits.push_back(u);
    }

    void WideLayout::resolvePacket()
    {
        const size_t units = m_pendUnits.size();
        // Elements: primitives that touch or overlap (a message window's tiles and the text on
        // them) are one element. Groups are merged by their bounding boxes.
        struct Box
        {
            float minX, maxX, minY, maxY;
            bool live;
        };
        std::vector<Box> groups;
        std::vector<uint32_t> parent, live;
        auto find = [&](uint32_t g) {
            while (parent[g] != g)
                g = parent[g] = parent[parent[g]];
            return g;
        };
        if (m_placement == HudPlacement::Edges)
        {
            groups.reserve(units);
            parent.reserve(units);
            for (size_t i = 0; i < units; ++i)
            {
                PendingUnit &u = m_pendUnits[i];
                if (u.follow)
                    continue;
                Box box{u.minX, u.maxX, u.minY, u.maxY, true};
                const uint32_t id = static_cast<uint32_t>(groups.size());
                // Absorb every element this one touches (again while the grown box reaches more).
                for (bool merged = true; merged;)
                {
                    merged = false;
                    for (size_t k = 0; k < live.size();)
                    {
                        Box &o = groups[live[k]];
                        if (box.minX > o.maxX + kTouch || o.minX > box.maxX + kTouch ||
                            box.minY > o.maxY + kTouch || o.minY > box.maxY + kTouch)
                        {
                            ++k;
                            continue;
                        }
                        box.minX = std::min(box.minX, o.minX);
                        box.maxX = std::max(box.maxX, o.maxX);
                        box.minY = std::min(box.minY, o.minY);
                        box.maxY = std::max(box.maxY, o.maxY);
                        o.live = false;
                        parent[live[k]] = id;
                        live[k] = live.back();
                        live.pop_back();
                        merged = true;
                    }
                }
                groups.push_back(box);
                parent.push_back(id);
                live.push_back(id);
                u.group = id;
            }
        }
        m_groupAnchor.assign(groups.size(), Anchor::Centre);
        for (uint32_t g = 0; g < groups.size(); ++g)
            if (groups[g].live)
            {
                const float centre = (groups[g].minX + groups[g].maxX) * 0.5f;
                m_groupAnchor[g] = centre < kWidth / 3.0f ? Anchor::Left
                                   : centre > kWidth * 2.0f / 3.0f ? Anchor::Right
                                                                   : Anchor::Centre;
            }

        // Units in drawing order: each one's anchor, and the anchor before each scissor.
        size_t scissor = 0;
        auto placeScissors = [&](uint32_t unitsBefore) {
            for (; scissor < m_pendScissors.size() && m_pendScissors[scissor].unitsBefore <= unitsBefore; ++scissor)
                transformScissor(m_pendScissors[scissor].value, m_lastAnchor);
        };
        for (size_t i = 0; i < units; ++i)
        {
            placeScissors(static_cast<uint32_t>(i));
            const PendingUnit &u = m_pendUnits[i];
            Anchor anchor = Anchor::Centre;
            if (m_placement == HudPlacement::Edges)
                anchor = u.follow ? m_lastAnchor : m_groupAnchor[find(u.group)];
            m_lastAnchor = anchor;
            for (uint32_t v = u.first; v < u.first + u.count; ++v)
            {
                const PendingVertex &pv = m_pendVerts[v];
                const float x = (static_cast<float>(load16(pv.x)) - static_cast<float>(pv.ofx)) / 16.0f;
                const float placed = std::round(place(x, anchor) * 16.0f + static_cast<float>(pv.ofx));
                store16(pv.x, static_cast<uint16_t>(std::clamp(placed, 0.0f, 65535.0f)));
            }
        }
        placeScissors(UINT32_MAX);
        m_pendUnits.clear();
        m_pendVerts.clear();
        m_pendScissors.clear();
    }

    void WideLayout::transformScissor(uint8_t *value, Anchor anchor) const
    {
        // SCISSOR: SCAX0 bits 0-10, SCAX1 bits 16-26 (pixels, no offset).
        uint64_t v = load64(value);
        const float x0 = static_cast<float>(v & 0x7FFu), x1 = static_cast<float>((v >> 16) & 0x7FFu);
        const float n0 = std::floor(place(x0, anchor)), n1 = std::ceil(place(x1 + 1.0f, anchor)) - 1.0f;
        v &= ~((0x7FFull) | (0x7FFull << 16));
        v |= static_cast<uint64_t>(std::clamp(n0, 0.0f, 2047.0f)) | (static_cast<uint64_t>(std::clamp(n1, 0.0f, 2047.0f)) << 16);
        std::memcpy(value, &v, 8);
    }

    void WideLayout::transformPacket(uint32_t pathIndex, uint8_t *data, uint32_t sizeBytes, const PrimState &start)
    {
        PrimState st = start;
        m_defaultClass = VertexClass::Scene;
        m_unitCount = 0;
        m_unitOverflow = 0;
        m_classes.clear();
        m_pendUnits.clear();
        m_pendVerts.clear();
        m_pendScissors.clear();
        parsePacket(pathIndex, data, sizeBytes, st);
        flushUnit(pathIndex, st);
        resolvePacket();
    }

    void WideLayout::parsePacket(uint32_t pathIndex, uint8_t *data, uint32_t sizeBytes, PrimState &st)
    {
        // A unit is one sprite or point, or the run of vertices of one primitive type in a GIF tag.
        auto flush = [&] { flushUnit(pathIndex, st); };
        auto addVertex = [&](uint8_t *x, uint16_t y) {
            if (m_unitCount < sizeof(m_unit) / sizeof(m_unit[0]))
                m_unit[m_unitCount++] = {x, st.ctxt, (static_cast<float>(y) - static_cast<float>(st.ofy[st.ctxt ? 1 : 0])) / 16.0f};
            else
                ++m_unitOverflow;
            if ((st.type == kSprite && m_unitCount == 2) || st.type == 0)
                flush();
        };
        auto scissor = [&](uint8_t *q) {
            flush(); // it takes the anchor of the element drawn before it
            // Full-width scissors stay; the others follow the HUD (placed with the packet's units).
            const uint64_t v = load64(q);
            const float x0 = static_cast<float>(v & 0x7FFu), x1 = static_cast<float>((v >> 16) & 0x7FFu);
            if (m_hud && m_mode == Mode::Driving && active() && !(x0 <= 0.5f && x1 >= kWidth - 1.5f))
                m_pendScissors.push_back({q, static_cast<uint32_t>(m_pendUnits.size())});
        };

        uint32_t offset = 0;
        while (offset + 16 <= sizeBytes)
        {
            const uint64_t tagLo = load64(data + offset), tagHi = load64(data + offset + 8);
            offset += 16;
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFF);
            const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 3);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xF);
            if (nreg == 0)
                nreg = 16;
            flush();
            if ((tagLo >> 46) & 1)
                setPrim(st, (tagLo >> 47) & 0x7FF);

            if (flg == 0) // PACKED
            {
                for (uint32_t loop = 0; loop < nloop; ++loop)
                    for (uint32_t r = 0; r < nreg; ++r)
                    {
                        if (offset + 16 > sizeBytes)
                            return;
                        uint8_t *q = data + offset;
                        offset += 16;
                        const uint8_t desc = static_cast<uint8_t>((tagHi >> (r * 4)) & 0xF);
                        if (desc == kPrim)
                        {
                            flush();
                            setPrim(st, load64(q) & 0x7FF);
                        }
                        else if (desc == kXyz2 || desc == kXyzf2 || desc == kXyz3 || desc == kXyzf3)
                            addVertex(q, load16(q + 4));
                        else if (desc == kTex0_1 || desc == kTex0_2)
                        {
                            flush();
                            st.tbp[desc - kTex0_1] = static_cast<uint32_t>(load64(q) & 0x3FFF);
                        }
                        else if (desc == kAd)
                        {
                            const uint8_t reg = static_cast<uint8_t>(load64(q + 8) & 0xFF);
                            const uint64_t v = load64(q);
                            if (reg == kRegPrim)
                            {
                                flush();
                                setPrim(st, v & 0x7FF);
                            }
                            else if (reg == kRegXyz2 || reg == kRegXyzf2 || reg == kRegXyz3 || reg == kRegXyzf3)
                                addVertex(q, static_cast<uint16_t>(v >> 16));
                            else if (reg == kRegTex0_1 || reg == kRegTex0_2)
                            {
                                flush();
                                st.tbp[reg - kRegTex0_1] = static_cast<uint32_t>(v & 0x3FFF);
                            }
                            else if (reg == kRegXyoffset1 || reg == kRegXyoffset2)
                            {
                                flush(); // queued vertices use the old offset
                                st.ofx[reg - kRegXyoffset1] = static_cast<uint32_t>(v & 0xFFFF);
                                st.ofy[reg - kRegXyoffset1] = static_cast<uint32_t>((v >> 32) & 0xFFFF);
                            }
                            else if (reg == kRegScissor1 || reg == kRegScissor2)
                                scissor(q);
                        }
                    }
            }
            else if (flg == 1) // REGLIST: 64-bit register values, same descriptors
            {
                for (uint32_t loop = 0; loop < nloop; ++loop)
                    for (uint32_t r = 0; r < nreg; ++r)
                    {
                        if (offset + 8 > sizeBytes)
                            return;
                        uint8_t *q = data + offset;
                        offset += 8;
                        const uint8_t desc = static_cast<uint8_t>((tagHi >> (r * 4)) & 0xF);
                        if (desc == kPrim)
                        {
                            flush();
                            setPrim(st, load64(q) & 0x7FF);
                        }
                        else if (desc == kXyz2 || desc == kXyzf2 || desc == kXyz3 || desc == kXyzf3)
                            addVertex(q, static_cast<uint16_t>(load64(q) >> 16));
                        else if (desc == kTex0_1 || desc == kTex0_2)
                        {
                            flush();
                            st.tbp[desc - kTex0_1] = static_cast<uint32_t>(load64(q) & 0x3FFF);
                        }
                    }
                if ((nloop * nreg) & 1)
                    offset += 8;
            }
            else // IMAGE (or disabled): data only
                offset += nloop * 16;
        }
    }
}
