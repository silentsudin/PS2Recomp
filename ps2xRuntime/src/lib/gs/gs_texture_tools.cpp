#include "gs_texture_tools.h"

#include "raylib.h" // PNG write/read (stb)

#include <algorithm>
#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace ps2x::gs
{
    namespace
    {
        uint64_t contentHash(uint32_t w, uint32_t h, const std::vector<uint8_t> &rgba)
        {
            // FNV-1a 64 over size and pixels.
            uint64_t hash = 1469598103934665603ull;
            auto mix = [&](const uint8_t *p, size_t n) {
                for (size_t i = 0; i < n; ++i)
                {
                    hash ^= p[i];
                    hash *= 1099511628211ull;
                }
            };
            mix(reinterpret_cast<const uint8_t *>(&w), 4);
            mix(reinterpret_cast<const uint8_t *>(&h), 4);
            mix(rgba.data(), rgba.size());
            return hash;
        }

        // RT_TEXTURE_RECOLOR=0: replace exact matches only (tests).
        bool recolorEnabled()
        {
            static const bool on = [] { const char *e = std::getenv("RT_TEXTURE_RECOLOR"); return !e || *e != '0'; }();
            return on;
        }

        // RT_TEXTURE_RECOLOR_TEST=swap: recolour every match with red and blue swapped (tests the
        // recolour path end to end).
        bool recolorSwapTest()
        {
            static const bool on = [] { const char *e = std::getenv("RT_TEXTURE_RECOLOR_TEST"); return e && !std::strcmp(e, "swap"); }();
            return on;
        }

        bool paletted(uint32_t psm) { return psm == 0x13 || psm == 0x14 || psm == 0x1B || psm == 0x24 || psm == 0x2C; }

        // A paletted texture's index pattern, palette aside: each distinct colour is a class,
        // numbered by first appearance; the key hashes the class image. colours[i] is class i's
        // colour. Two palettes of one texture give the same pattern unless one of them merges
        // colours the other keeps apart.
        bool indexPattern(uint32_t w, uint32_t h, const std::vector<uint8_t> &rgba, uint64_t &key, std::vector<uint32_t> &colours)
        {
            std::unordered_map<uint32_t, uint8_t> classOf;
            colours.clear();
            uint64_t hash = 1469598103934665603ull ^ 0x5348415045ull; // distinct from content keys
            auto mix = [&](uint32_t v, int bytes) {
                for (int i = 0; i < bytes; ++i)
                {
                    hash ^= (v >> (8 * i)) & 0xFF;
                    hash *= 1099511628211ull;
                }
            };
            mix(w, 4);
            mix(h, 4);
            for (size_t i = 0; i + 4 <= rgba.size(); i += 4)
            {
                uint32_t c;
                std::memcpy(&c, &rgba[i], 4);
                auto it = classOf.find(c);
                if (it == classOf.end())
                {
                    if (colours.size() == 256)
                        return false;
                    it = classOf.emplace(c, uint8_t(colours.size())).first;
                    colours.push_back(c);
                }
                mix(it->second, 1);
            }
            key = hash;
            return true;
        }

        // Affine map (4x4 + offset, RGBA in 0..255) taking the reference palette's colours to
        // the variant's, least squares, pulled towards identity where the colours don't pin it
        // down. False if it doesn't fit (a palette swap no affine map explains).
        bool fitRecolor(const std::vector<uint32_t> &from, const std::vector<uint32_t> &to, float out[20])
        {
            if (from.size() != to.size() || from.empty())
                return false;
            double ata[5][5] = {}, atb[5][4] = {};
            auto chan = [](uint32_t c, int k) { return double((c >> (8 * k)) & 0xFF); };
            for (size_t n = 0; n < from.size(); ++n)
            {
                double a[5] = {chan(from[n], 0), chan(from[n], 1), chan(from[n], 2), chan(from[n], 3), 1.0};
                for (int i = 0; i < 5; ++i)
                {
                    for (int j = 0; j < 5; ++j)
                        ata[i][j] += a[i] * a[j];
                    for (int k = 0; k < 4; ++k)
                        atb[i][k] += a[i] * chan(to[n], k);
                }
            }
            const double lambda = 1.0;
            for (int i = 0; i < 5; ++i)
            {
                ata[i][i] += lambda;
                if (i < 4)
                    atb[i][i] += lambda; // towards identity
            }
            // Gauss-Jordan on [ata | atb].
            double m[5][9];
            for (int i = 0; i < 5; ++i)
            {
                for (int j = 0; j < 5; ++j)
                    m[i][j] = ata[i][j];
                for (int k = 0; k < 4; ++k)
                    m[i][5 + k] = atb[i][k];
            }
            for (int col = 0; col < 5; ++col)
            {
                int piv = col;
                for (int r = col + 1; r < 5; ++r)
                    if (std::fabs(m[r][col]) > std::fabs(m[piv][col]))
                        piv = r;
                if (std::fabs(m[piv][col]) < 1e-9)
                    return false;
                std::swap(m[col], m[piv]);
                const double d = m[col][col];
                for (int j = 0; j < 9; ++j)
                    m[col][j] /= d;
                for (int r = 0; r < 5; ++r)
                    if (r != col && m[r][col] != 0.0)
                    {
                        const double f = m[r][col];
                        for (int j = 0; j < 9; ++j)
                            m[r][j] -= f * m[col][j];
                    }
            }
            // x[i][k] = m[i][5 + k]: output channel k = sum_i x[i][k] * in_i (in_4 = 1).
            for (size_t n = 0; n < from.size(); ++n)
            {
                double a[5] = {chan(from[n], 0), chan(from[n], 1), chan(from[n], 2), chan(from[n], 3), 1.0};
                for (int k = 0; k < 4; ++k)
                {
                    double v = 0.0;
                    for (int i = 0; i < 5; ++i)
                        v += a[i] * m[i][5 + k];
                    if (std::fabs(v - chan(to[n], k)) > 3.0)
                        return false;
                }
            }
            for (int k = 0; k < 4; ++k)
            {
                for (int i = 0; i < 4; ++i)
                    out[k * 4 + i] = float(m[i][5 + k]);
                out[16 + k] = float(m[4][5 + k]);
            }
            return true;
        }

        bool parseKey(const std::string &name, uint64_t &key)
        {
            if (name.size() < 16)
                return false;
            for (int i = 0; i < 16; ++i)
                if (!std::isxdigit(static_cast<unsigned char>(name[i])))
                    return false;
            key = std::strtoull(name.substr(0, 16).c_str(), nullptr, 16);
            return true;
        }
    }

    TextureTools::TextureTools() : m_thread([this] { worker(); }) {}

    TextureTools::~TextureTools()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        m_thread.join();
    }

    void TextureTools::configure(const std::string &dumpDir, const std::string &packDir)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (dumpDir != m_dumpDir)
        {
            m_dumpDir = dumpDir;
            m_dumped.clear();
            if (!dumpDir.empty())
            {
                std::error_code ec;
                std::filesystem::create_directories(dumpDir, ec);
                for (const auto &e : std::filesystem::directory_iterator(dumpDir, ec))
                {
                    uint64_t key;
                    if (parseKey(e.path().filename().string(), key))
                        m_dumped.insert(key);
                }
            }
        }
        if (packDir != m_packDir)
        {
            m_packDir = packDir;
            m_pack.clear();
            m_handedOut.clear();
            m_shapes.clear();
            m_ready.clear(); // outcomes for the previous pack
            if (!packDir.empty())
                indexPack(packDir);
        }
    }

    void TextureTools::indexPack(const std::string &dir)
    {
        std::error_code ec;
        for (const auto &e : std::filesystem::recursive_directory_iterator(dir, ec))
        {
            if (!e.is_regular_file())
                continue;
            const std::string name = e.path().filename().string();
            uint64_t key;
            if (e.path().extension() == ".png" && parseKey(name, key))
                m_pack[key] = e.path().string();
        }
        std::fprintf(stderr, "[textures] pack %s: %zu replacements\n", dir.c_str(), m_pack.size());
    }

    void TextureTools::forget(uint64_t contentKey)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_handedOut.erase(contentKey);
    }

    TextureTools::Stats TextureTools::stats() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        Stats s = m_stats;
        s.patterns = m_shapes.size();
        return s;
    }

    size_t TextureTools::packSize() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_pack.size();
    }

    bool TextureTools::active() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return !m_dumpDir.empty() || !m_pack.empty();
    }

    void TextureTools::submit(uint64_t cacheKey, uint64_t stableKey, uint32_t width, uint32_t height, uint32_t psm,
                              std::vector<uint8_t> rgba)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_jobs.size() > 512)
                return; // a burst at a scene load: it comes back when it is decoded again
            m_jobs.push_back({cacheKey, stableKey, width, height, psm, std::move(rgba)});
        }
        m_cv.notify_one();
    }

    void TextureTools::collect(std::vector<Replacement> &out)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto &r : m_ready)
            out.push_back(std::move(r));
        m_ready.clear();
    }

    void TextureTools::worker()
    {
        for (;;)
        {
            Job job;
            std::string dumpDir, packPath;
            bool dump = false, handedOut = false;
            uint64_t key = 0;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [&] { return m_stop || !m_jobs.empty(); });
                if (m_stop)
                    return;
                job = std::move(m_jobs.front());
                m_jobs.pop_front();
            }
            key = contentHash(job.width, job.height, job.rgba);
            uint64_t shapeKey = 0;
            std::vector<uint32_t> colours;
            const bool hasShape = paletted(job.psm) && indexPattern(job.width, job.height, job.rgba, shapeKey, colours);
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                dumpDir = m_dumpDir;
                dump = !dumpDir.empty() && m_dumped.insert(key).second;
                auto it = m_pack.find(key);
                if (it != m_pack.end())
                {
                    packPath = it->second;
                    handedOut = m_handedOut.count(key) != 0;
                }
            }
            if (dump)
            {
                // PS2 alpha (0x80 = opaque) to PNG alpha.
                std::vector<uint8_t> png = job.rgba;
                for (size_t i = 3; i < png.size(); i += 4)
                    png[i] = static_cast<uint8_t>(std::min(255u, png[i] * 255u / 128u));
                char name[96];
                std::snprintf(name, sizeof(name), "%016llx_%ux%u_psm%02x.png", static_cast<unsigned long long>(key), job.width,
                              job.height, job.psm);
                Image image = {png.data(), static_cast<int>(job.width), static_cast<int>(job.height), 1,
                               PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
                // Under another name, then renamed: several games may dump into one folder.
                const std::string path = (std::filesystem::path(dumpDir) / name).string();
                const std::string part = path + "." + std::to_string(getpid()) + ".part.png";
                if (ExportImage(image, part.c_str()))
                    std::rename(part.c_str(), path.c_str());
            }
            bool packActive;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                packActive = !m_pack.empty();
            }
            if (!packActive)
                continue;
            Replacement r;
            r.cacheKey = job.cacheKey;
            r.stableKey = job.stableKey;
            r.contentKey = key;
            r.hit = !packPath.empty();
            if (hasShape)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (r.hit && recolorSwapTest())
                {
                    // Test hook: recolour exact matches too, red and blue swapped.
                    r.recolor = true;
                    float t[20] = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
                    std::memcpy(r.transform, t, sizeof(t));
                }
                auto s = r.hit ? m_shapes.end() : m_shapes.find(shapeKey);
                if (r.hit)
                    m_shapes[shapeKey] = {key, colours};
                else if (s != m_shapes.end() && recolorEnabled() && (++m_stats.fits, fitRecolor(s->second.colours, colours, r.transform)))
                {
                    ++m_stats.fitted;
                    // Another palette of a replaced texture (a fade, a colour flash): its image,
                    // recoloured.
                    auto p = m_pack.find(s->second.contentKey);
                    if (p != m_pack.end())
                    {
                        r.hit = r.recolor = true;
                        r.contentKey = s->second.contentKey;
                        packPath = p->second;
                        handedOut = m_handedOut.count(r.contentKey) != 0;
                    }
                }
            }
            if (r.hit && !handedOut)
            {
                Image image = LoadImage(packPath.c_str());
                if (!image.data)
                    continue;
                ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
                r.width = static_cast<uint32_t>(image.width);
                r.height = static_cast<uint32_t>(image.height);
                const uint8_t *src = static_cast<const uint8_t *>(image.data);
                r.rgba.assign(src, src + size_t(r.width) * r.height * 4);
                UnloadImage(image);
                // PNG alpha back to PS2 alpha.
                for (size_t i = 3; i < r.rgba.size(); i += 4)
                    r.rgba[i] = static_cast<uint8_t>((r.rgba[i] * 128u + 127u) / 255u);
            }
            std::lock_guard<std::mutex> lock(m_mutex);
            if (r.hit)
                m_handedOut.insert(r.contentKey);
            m_ready.push_back(std::move(r));
        }
    }
}
