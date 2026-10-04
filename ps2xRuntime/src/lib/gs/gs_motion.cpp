#include "runtime/gs/gs_motion.h"

#include <algorithm>

#include <cmath>
#include <cstring>

namespace ps2x::gs
{
    namespace
    {
        Mat4 identity()
        {
            Mat4 r{};
            r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0;
            return r;
        }

        // Four VU1 data qwords from `qw` as the columns of a matrix.
        Mat4 load(const uint8_t *vuData, uint32_t qw)
        {
            Mat4 r{};
            for (int i = 0; i < 16; ++i)
            {
                float f;
                std::memcpy(&f, vuData + qw * 16 + i * 4, 4);
                r.m[i] = f;
            }
            return r;
        }

        Mat4 mul(const Mat4 &a, const Mat4 &b)
        {
            Mat4 r{};
            for (int c = 0; c < 4; ++c)
                for (int row = 0; row < 4; ++row)
                {
                    double s = 0.0;
                    for (int k = 0; k < 4; ++k)
                        s += a.m[k * 4 + row] * b.m[c * 4 + k];
                    r.m[c * 4 + row] = s;
                }
            return r;
        }

        Mat4 translate(const uint8_t *vuData, uint32_t qw)
        {
            Mat4 r = identity();
            for (int i = 0; i < 3; ++i)
            {
                float f;
                std::memcpy(&f, vuData + qw * 16 + i * 4, 4);
                r.m[12 + i] = f;
            }
            return r;
        }

        bool invert(const Mat4 &mat, Mat4 &out)
        {
            const double *m = mat.m;
            double inv[16];
            inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
            inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
            inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
            inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
            inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
            inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
            inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
            inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
            inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
            inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
            inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
            inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
            inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
            inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
            inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
            inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
            const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
            if (!std::isfinite(det) || std::fabs(det) < 1e-30)
                return false;
            for (int i = 0; i < 16; ++i)
                out.m[i] = inv[i] / det;
            return true;
        }

        uint16_t toHalf(float f)
        {
            uint32_t x;
            std::memcpy(&x, &f, 4);
            const uint32_t sign = (x >> 16) & 0x8000u;
            int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = x & 0x7FFFFFu;
            if (exp <= 0)
                return static_cast<uint16_t>(sign); // tiny: zero
            if (exp >= 31)
                return static_cast<uint16_t>(sign | 0x7BFFu); // clamp to the largest finite
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
        }

        uint64_t load64(const uint8_t *p)
        {
            uint64_t v;
            std::memcpy(&v, p, 8);
            return v;
        }
    }

    MotionTracker &MotionTracker::instance()
    {
        static MotionTracker tracker;
        return tracker;
    }

    void MotionTracker::setEnabled(bool enabled) { m_enabled.store(enabled, std::memory_order_relaxed); }

    Mat4 MotionTracker::matchPrevious(const Object &obj, uint32_t ordinal) const
    {
        const auto &prev = m_prev[static_cast<int>(obj.kind)];
        auto dist2 = [&](const Object &o) {
            const double dx = o.tx - obj.tx, dy = o.ty - obj.ty, dz = o.tz - obj.tz;
            return dx * dx + dy * dy + dz * dz;
        };
        // Same draw order and close by: the same object (the common case). Otherwise the nearest.
        constexpr double kNear = 40.0 * 40.0;
        if (ordinal < prev.size() && dist2(prev[ordinal]) < kNear)
            return prev[ordinal].base;
        const Object *best = nullptr;
        double bestD = kNear;
        for (const Object &o : prev)
        {
            const double d = dist2(o);
            if (d < bestD)
            {
                bestD = d;
                best = &o;
            }
        }
        return best ? best->base : obj.base; // unmatched: no motion
    }

    uint32_t MotionTracker::onMscal(uint32_t startPC, const uint8_t *vuData)
    {
        if (!enabled() || !vuData)
            return 0;
        std::lock_guard<std::mutex> lock(m_mutex);
        // Setup runs: work out this object's C and last frame's.
        if (startPC == 0x10 || startPC == 0x70 || startPC == 0x00)
        {
            Object obj{};
            Mat4 world = identity();
            if (startPC == 0x10)
            {
                obj.kind = Kind::SetupB;
                world = load(vuData, 0);
                obj.base = mul(load(vuData, 4), world);
            }
            else
            {
                obj.kind = startPC == 0x70 ? Kind::SetupA : Kind::Init;
                if (startPC == 0x70)
                    world = load(vuData, 0);
                obj.base = mul(mul(load(vuData, 12), load(vuData, 8)), world);
            }
            obj.tx = world.m[12];
            obj.ty = world.m[13];
            obj.tz = world.m[14];
            auto &list = m_cur[static_cast<int>(obj.kind)];
            m_activePrevBase = matchPrevious(obj, static_cast<uint32_t>(list.size()));
            ++m_stats.objects;
            if (std::memcmp(&m_activePrevBase, &obj.base, sizeof(Mat4)) != 0)
                ++m_stats.matched; // found last frame's (a different C; identical ones also mean "still")
            m_activeBase = obj.base;
            m_activeKind = obj.kind;
            m_haveActive = true;
            list.push_back(obj);
            return 0;
        }
        if (!m_haveActive)
            return 0;
        MotionContext ctx;
        ctx.cur = m_activeBase;
        ctx.prev = m_activePrevBase;
        // Loops 8 and 6 add the batch offset VU26 to positions.
        if (startPC == 0x40 || startPC == 0x30)
        {
            const Mat4 t = translate(vuData, 26);
            ctx.cur = mul(ctx.cur, t);
            ctx.prev = mul(ctx.prev, t);
        }
        if (!invert(ctx.cur, ctx.curInv))
            return 0;
        const uint32_t id = m_nextId++;
        if (m_nextId == 0)
            m_nextId = 1;
        m_ring[id % kRing] = ctx;
        return id;
    }

    void MotionTracker::frameStart()
    {
        if (!enabled())
            return;
        std::lock_guard<std::mutex> lock(m_mutex);
        for (int k = 0; k < 3; ++k)
        {
            m_prev[k].swap(m_cur[k]);
            m_cur[k].clear();
        }
        m_haveActive = false;
        m_lastStats = m_stats;
        m_stats = {};
    }

    bool MotionTracker::context(uint32_t id, MotionContext &out) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (id == 0 || m_nextId - id >= kRing)
            return false;
        out = m_ring[id % kRing];
        return true;
    }

    MotionTracker::Stats MotionTracker::lastFrameStats() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_lastStats;
    }

    void MotionTracker::makeMarker(uint32_t id, uint8_t out[32])
    {
        // GIF tag: NLOOP 1, EOP 0, PACKED, NREG 1, REGS = A+D; then A+D data = id, register 0x7E.
        const uint64_t tagLo = 1ull | (1ull << 60);
        const uint64_t tagHi = 0xEull;
        const uint64_t lo = id, hi = kMarkerRegister;
        std::memcpy(out, &tagLo, 8);
        std::memcpy(out + 8, &tagHi, 8);
        std::memcpy(out + 16, &lo, 8);
        std::memcpy(out + 24, &hi, 8);
    }

    bool MotionTracker::readMarker(const uint8_t *data, uint32_t size, uint32_t &id)
    {
        if (size != 32)
            return false;
        if (load64(data) != (1ull | (1ull << 60)) || load64(data + 8) != 0xEull || (load64(data + 24) & 0xFF) != kMarkerRegister)
            return false;
        id = static_cast<uint32_t>(load64(data + 16));
        return true;
    }

    void MotionTracker::packetMotion(const uint8_t *data, uint32_t size, const MotionContext &ctx, std::vector<uint32_t> &out)
    {
        out.clear();
        auto vertex = [&](uint16_t gx, uint16_t gy, uint32_t gz) {
            // Back to object space with C^-1 (w from the row that must give 1), forward with last
            // frame's C.
            const double X = gx / 16.0, Y = gy / 16.0, Z = static_cast<double>(gz);
            // Far outside the screen (the guard band, culled triangles): no motion needed.
            if (std::fabs(X - 2048.0) > 1024.0 || std::fabs(Y - 2048.0) > 512.0)
            {
                out.push_back(0);
                return;
            }
            const double *inv = ctx.curInv.m;
            const double ww = inv[3] * X + inv[7] * Y + inv[11] * Z + inv[15];
            uint32_t packed = 0;
            if (std::fabs(ww) > 1e-20)
            {
                const double w = 1.0 / ww;
                double obj[4];
                for (int r = 0; r < 4; ++r)
                    obj[r] = (inv[r] * X + inv[4 + r] * Y + inv[8 + r] * Z + inv[12 + r]) * w;
                const double *p = ctx.prev.m;
                const double px = p[0] * obj[0] + p[4] * obj[1] + p[8] * obj[2] + p[12] * obj[3];
                const double py = p[1] * obj[0] + p[5] * obj[1] + p[9] * obj[2] + p[13] * obj[3];
                const double pw = p[3] * obj[0] + p[7] * obj[1] + p[11] * obj[2] + p[15] * obj[3];
                if (std::fabs(pw) > 1e-20)
                {
                    const float dx = static_cast<float>(X - px / pw), dy = static_cast<float>(Y - py / pw);
                    if (std::isfinite(dx) && std::isfinite(dy))
                        packed = toHalf(dx) | (static_cast<uint32_t>(toHalf(dy)) << 16);
                }
            }
            out.push_back(packed);
        };

        uint32_t offset = 0;
        while (offset + 16 <= size)
        {
            const uint64_t tagLo = load64(data + offset), tagHi = load64(data + offset + 8);
            offset += 16;
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFF);
            const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 3);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xF);
            if (nreg == 0)
                nreg = 16;
            if (flg == 0)
            {
                for (uint32_t loop = 0; loop < nloop; ++loop)
                    for (uint32_t r = 0; r < nreg; ++r)
                    {
                        if (offset + 16 > size)
                            return;
                        const uint8_t *q = data + offset;
                        offset += 16;
                        const uint8_t desc = static_cast<uint8_t>((tagHi >> (r * 4)) & 0xF);
                        if (desc == 0x4 || desc == 0xC) // XYZF2/3 packed: Z in hi bits 4..27
                            vertex(static_cast<uint16_t>(load64(q)), static_cast<uint16_t>(load64(q) >> 32),
                                   static_cast<uint32_t>((load64(q + 8) >> 4) & 0xFFFFFF));
                        else if (desc == 0x5 || desc == 0xD) // XYZ2/3 packed: Z in hi bits 0..31
                            vertex(static_cast<uint16_t>(load64(q)), static_cast<uint16_t>(load64(q) >> 32),
                                   static_cast<uint32_t>(load64(q + 8)));
                        else if (desc == 0xE)
                        {
                            const uint8_t reg = static_cast<uint8_t>(load64(q + 8));
                            const uint64_t v = load64(q);
                            if (reg == 0x04 || reg == 0x0C)
                                vertex(static_cast<uint16_t>(v), static_cast<uint16_t>(v >> 16), static_cast<uint32_t>((v >> 32) & 0xFFFFFF));
                            else if (reg == 0x05 || reg == 0x0D)
                                vertex(static_cast<uint16_t>(v), static_cast<uint16_t>(v >> 16), static_cast<uint32_t>(v >> 32));
                        }
                    }
            }
            else if (flg == 1)
            {
                for (uint32_t loop = 0; loop < nloop; ++loop)
                    for (uint32_t r = 0; r < nreg; ++r)
                    {
                        if (offset + 8 > size)
                            return;
                        const uint64_t v = load64(data + offset);
                        offset += 8;
                        const uint8_t desc = static_cast<uint8_t>((tagHi >> (r * 4)) & 0xF);
                        if (desc == 0x4 || desc == 0xC)
                            vertex(static_cast<uint16_t>(v), static_cast<uint16_t>(v >> 16), static_cast<uint32_t>((v >> 32) & 0xFFFFFF));
                        else if (desc == 0x5 || desc == 0xD)
                            vertex(static_cast<uint16_t>(v), static_cast<uint16_t>(v >> 16), static_cast<uint32_t>(v >> 32));
                    }
                if ((nloop * nreg) & 1)
                    offset += 8;
            }
            else
                offset += nloop * 16;
        }
    }

    void MotionTracker::packetReproject(const uint8_t *data, uint32_t size, const MotionContext &ctx, double t,
                                        std::vector<uint8_t> &out)
    {
        out.assign(data, data + size);
        uint8_t *buf = out.data();
        auto store64 = [](uint8_t *p, uint64_t v) { std::memcpy(p, &v, 8); };
        // The vertex at (gx, gy, gz), moved on by t: false if it is left as it is. `ratio` is
        // w / w' for the texture coordinates.
        auto move = [&](uint16_t &gx, uint16_t &gy, uint32_t &gz, uint32_t zMax, double &ratio) {
            const double X = gx / 16.0, Y = gy / 16.0, Z = static_cast<double>(gz);
            if (std::fabs(X - 2048.0) > 1024.0 || std::fabs(Y - 2048.0) > 512.0)
                return false;
            const double *inv = ctx.curInv.m;
            const double ww = inv[3] * X + inv[7] * Y + inv[11] * Z + inv[15];
            if (std::fabs(ww) < 1e-20)
                return false;
            const double w = 1.0 / ww;
            double obj[4];
            for (int r = 0; r < 4; ++r)
                obj[r] = (inv[r] * X + inv[4 + r] * Y + inv[8 + r] * Z + inv[12 + r]) * w;
            const double *p = ctx.prev.m;
            double prev[4];
            for (int r = 0; r < 4; ++r)
                prev[r] = p[r] * obj[0] + p[4 + r] * obj[1] + p[8 + r] * obj[2] + p[12 + r] * obj[3];
            const double cur[4] = {X * w, Y * w, Z * w, w};
            double moved[4];
            for (int r = 0; r < 4; ++r)
                moved[r] = cur[r] + t * (cur[r] - prev[r]);
            if (!(std::fabs(moved[3]) > 1e-20) || (moved[3] > 0) != (w > 0))
                return false;
            const double nx = moved[0] / moved[3] * 16.0, ny = moved[1] / moved[3] * 16.0, nz = moved[2] / moved[3];
            if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz))
                return false;
            gx = static_cast<uint16_t>(std::clamp(std::lround(nx), 0l, 65535l));
            gy = static_cast<uint16_t>(std::clamp(std::lround(ny), 0l, 65535l));
            gz = static_cast<uint32_t>(std::clamp(nz, 0.0, static_cast<double>(zMax)));
            ratio = w / moved[3];
            return true;
        };
        auto scaleSt = [&](uint8_t *st, double ratio) {
            // PACKED ST: S (bits 0..31), T (32..63), Q (64..95), floats.
            for (int k = 0; k < 3; ++k)
            {
                float f;
                std::memcpy(&f, st + 4 * k, 4);
                f = static_cast<float>(f * ratio);
                std::memcpy(st + 4 * k, &f, 4);
            }
        };

        uint32_t offset = 0;
        while (offset + 16 <= size)
        {
            const uint64_t tagLo = load64(buf + offset), tagHi = load64(buf + offset + 8);
            offset += 16;
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFF);
            const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 3);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xF);
            if (nreg == 0)
                nreg = 16;
            if (flg == 0)
            {
                for (uint32_t loop = 0; loop < nloop; ++loop)
                {
                    uint8_t *st = nullptr; // this vertex's ST (it comes before its XYZ)
                    for (uint32_t r = 0; r < nreg; ++r)
                    {
                        if (offset + 16 > size)
                            return;
                        uint8_t *q = buf + offset;
                        offset += 16;
                        const uint8_t desc = static_cast<uint8_t>((tagHi >> (r * 4)) & 0xF);
                        if (desc == 0x2)
                            st = q;
                        else if (desc == 0x4 || desc == 0xC || desc == 0x5 || desc == 0xD)
                        {
                            const bool fog = desc == 0x4 || desc == 0xC;
                            const uint64_t lo = load64(q), hi = load64(q + 8);
                            uint16_t gx = static_cast<uint16_t>(lo), gy = static_cast<uint16_t>(lo >> 32);
                            uint32_t gz = fog ? static_cast<uint32_t>((hi >> 4) & 0xFFFFFF) : static_cast<uint32_t>(hi);
                            double ratio = 1.0;
                            if (move(gx, gy, gz, fog ? 0xFFFFFFu : 0xFFFFFFFFu, ratio))
                            {
                                store64(q, (lo & ~0x0000FFFF0000FFFFull) | gx | (static_cast<uint64_t>(gy) << 32));
                                store64(q + 8, fog ? ((hi & ~(0xFFFFFFull << 4)) | (static_cast<uint64_t>(gz) << 4))
                                                   : ((hi & ~0xFFFFFFFFull) | gz));
                                if (st)
                                    scaleSt(st, ratio);
                            }
                            st = nullptr;
                        }
                        else if (desc == 0xE)
                        {
                            const uint8_t reg = static_cast<uint8_t>(load64(q + 8));
                            const uint64_t v = load64(q);
                            if (reg == 0x04 || reg == 0x0C || reg == 0x05 || reg == 0x0D)
                            {
                                const bool fog = reg == 0x04 || reg == 0x0C;
                                uint16_t gx = static_cast<uint16_t>(v), gy = static_cast<uint16_t>(v >> 16);
                                uint32_t gz = fog ? static_cast<uint32_t>((v >> 32) & 0xFFFFFF) : static_cast<uint32_t>(v >> 32);
                                double ratio = 1.0;
                                if (move(gx, gy, gz, fog ? 0xFFFFFFu : 0xFFFFFFFFu, ratio))
                                    store64(q, fog ? ((v & 0xFF00000000000000ull) | gx | (static_cast<uint64_t>(gy) << 16) |
                                                      (static_cast<uint64_t>(gz) << 32))
                                                   : (gx | (static_cast<uint64_t>(gy) << 16) | (static_cast<uint64_t>(gz) << 32)));
                            }
                        }
                    }
                }
            }
            else if (flg == 1)
            {
                // REGLIST: positions only (ST and Q are separate registers here).
                for (uint32_t loop = 0; loop < nloop; ++loop)
                    for (uint32_t r = 0; r < nreg; ++r)
                    {
                        if (offset + 8 > size)
                            return;
                        uint8_t *q = buf + offset;
                        offset += 8;
                        const uint8_t desc = static_cast<uint8_t>((tagHi >> (r * 4)) & 0xF);
                        if (desc == 0x4 || desc == 0xC || desc == 0x5 || desc == 0xD)
                        {
                            const bool fog = desc == 0x4 || desc == 0xC;
                            const uint64_t v = load64(q);
                            uint16_t gx = static_cast<uint16_t>(v), gy = static_cast<uint16_t>(v >> 16);
                            uint32_t gz = fog ? static_cast<uint32_t>((v >> 32) & 0xFFFFFF) : static_cast<uint32_t>(v >> 32);
                            double ratio = 1.0;
                            if (move(gx, gy, gz, fog ? 0xFFFFFFu : 0xFFFFFFFFu, ratio))
                                store64(q, fog ? ((v & 0xFF00000000000000ull) | gx | (static_cast<uint64_t>(gy) << 16) |
                                                  (static_cast<uint64_t>(gz) << 32))
                                               : (gx | (static_cast<uint64_t>(gy) << 16) | (static_cast<uint64_t>(gz) << 32)));
                        }
                    }
                if ((nloop * nreg) & 1)
                    offset += 8;
            }
            else
                offset += nloop * 16;
        }
    }
}
