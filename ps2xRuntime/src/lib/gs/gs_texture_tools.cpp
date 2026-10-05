#include "gs_texture_tools.h"

#include "raylib.h" // PNG write/read (stb)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
                ExportImage(image, (std::filesystem::path(dumpDir) / name).string().c_str());
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
                m_handedOut.insert(key);
            m_ready.push_back(std::move(r));
        }
    }
}
