#include "runtime/gs/gs_wide_layout.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ps2x::gs
{
    namespace
    {
        constexpr float kWidth = 640.0f;
        constexpr float kFullScreen = 600.0f; // wider than this: a fade or post-pass, left to stretch

        // GIF PACKED register descriptors and GS register addresses (A+D).
        enum : uint8_t
        {
            kPrim = 0x00, kXyzf2 = 0x04, kXyz2 = 0x05, kXyzf3 = 0x0C, kXyz3 = 0x0D, kAd = 0x0E,
        };
        enum : uint8_t
        {
            kRegPrim = 0x00, kRegXyzf2 = 0x04, kRegXyz2 = 0x05, kRegXyzf3 = 0x0C, kRegXyz3 = 0x0D,
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
            st.ctxt = ((prim >> 9) & 1u) != 0;
        }
    }

    void WideLayout::vote(bool looks2D)
    {
        // Hysteresis: a few odd frames (a town frame that starts with a 2D draw, a frame drawn
        // without the clear) must not flip the picture between wide and 4:3.
        if (looks2D)
        {
            m_driveVotes = 0;
            if (++m_2DVotes >= 10)
                m_lastFrame2D = true;
        }
        else
        {
            m_2DVotes = 0;
            if (++m_driveVotes >= 2)
                m_lastFrame2D = false;
        }
    }

    void WideLayout::frameStart()
    {
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

    uint16_t WideLayout::fromScreen(float x, bool ctx2, const PrimState &st) const
    {
        const float v = std::round(x * 16.0f + static_cast<float>(st.ofx[ctx2 ? 1 : 0]));
        return static_cast<uint16_t>(std::clamp(v, 0.0f, 65535.0f));
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
        m_unitCount = 0;

        float minX = 1e9f, maxX = -1e9f;
        for (uint32_t i = 0; i < n; ++i)
        {
            const float x = toScreen(load16(m_unit[i].x), m_unit[i].ctx2, st);
            minX = std::min(minX, x);
            maxX = std::max(maxX, x);
        }
        const float width = maxX - minX;

        if (m_mode == Mode::Unknown)
            m_mode = pathIndex == 0 ? Mode::Driving : Mode::Screen2D;
        if (m_mode != Mode::Driving)
            return;
        if (!m_hud && pathIndex >= 1 && !st.ctxt && st.tme)
        {
            m_hud = true;
            m_hudStarted = true;
        }
        // Full-screen sprites (the fade, the final post-pass) stretch with the 3D.
        if (!m_hud || (st.type == kSprite && width >= kFullScreen) || !active())
            return;

        Anchor anchor = Anchor::Centre;
        if (m_placement == HudPlacement::Edges)
        {
            // By its own position (thirds of the screen), except a wide 3D-drawn element (the town
            // minimap, scissored to its frame) goes with the element drawn just before it.
            if (pathIndex == 0 && width > kWidth / 2.0f)
                anchor = m_lastAnchor;
            else
            {
                const float centre = (minX + maxX) * 0.5f;
                anchor = centre < kWidth / 3.0f ? Anchor::Left : centre > kWidth * 2.0f / 3.0f ? Anchor::Right : Anchor::Centre;
            }
        }
        m_lastAnchor = anchor;
        for (uint32_t i = 0; i < n; ++i)
            store16(m_unit[i].x, fromScreen(place(toScreen(load16(m_unit[i].x), m_unit[i].ctx2, st), anchor), m_unit[i].ctx2, st));
    }

    void WideLayout::transformScissor(uint8_t *value)
    {
        // SCISSOR: SCAX0 bits 0-10, SCAX1 bits 16-26 (pixels, no offset). Full-width ones stay.
        uint64_t v = load64(value);
        const float x0 = static_cast<float>(v & 0x7FFu), x1 = static_cast<float>((v >> 16) & 0x7FFu);
        if (!m_hud || m_mode != Mode::Driving || !active() || (x0 <= 0.5f && x1 >= kWidth - 1.5f))
            return;
        const float n0 = std::floor(place(x0, m_lastAnchor)), n1 = std::ceil(place(x1 + 1.0f, m_lastAnchor)) - 1.0f;
        v &= ~((0x7FFull) | (0x7FFull << 16));
        v |= static_cast<uint64_t>(std::clamp(n0, 0.0f, 2047.0f)) | (static_cast<uint64_t>(std::clamp(n1, 0.0f, 2047.0f)) << 16);
        std::memcpy(value, &v, 8);
    }

    void WideLayout::transformPacket(uint32_t pathIndex, uint8_t *data, uint32_t sizeBytes, const PrimState &start)
    {
        PrimState st = start;
        m_unitCount = 0;
        // A unit is one sprite or point, or the run of vertices of one primitive type in a GIF tag.
        auto flush = [&] { flushUnit(pathIndex, st); };
        auto addVertex = [&](uint8_t *x, uint16_t) {
            if (m_unitCount < sizeof(m_unit) / sizeof(m_unit[0]))
                m_unit[m_unitCount++] = {x, st.ctxt};
            if ((st.type == kSprite && m_unitCount == 2) || st.type == 0)
                flush();
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
                        {
                            flush();
                            return;
                        }
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
                            else if (reg == kRegXyoffset1 || reg == kRegXyoffset2)
                            {
                                flush(); // queued vertices use the old offset
                                st.ofx[reg - kRegXyoffset1] = static_cast<uint32_t>(v & 0xFFFF);
                                st.ofy[reg - kRegXyoffset1] = static_cast<uint32_t>((v >> 32) & 0xFFFF);
                            }
                            else if (reg == kRegScissor1 || reg == kRegScissor2)
                            {
                                flush(); // the anchor of the element drawn before it
                                transformScissor(q);
                            }
                        }
                    }
            }
            else if (flg == 1) // REGLIST: 64-bit register values, same descriptors
            {
                for (uint32_t loop = 0; loop < nloop; ++loop)
                    for (uint32_t r = 0; r < nreg; ++r)
                    {
                        if (offset + 8 > sizeBytes)
                        {
                            flush();
                            return;
                        }
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
                    }
                if ((nloop * nreg) & 1)
                    offset += 8;
            }
            else // IMAGE (or disabled): data only
                offset += nloop * 16;
        }
        flush();
    }
}
