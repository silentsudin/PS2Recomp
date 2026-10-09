#include "runtime/gs/gs_texture_pack_cache.h"

#include "raylib.h" // PNG read (stb)

#if defined(PS2X_HAVE_ASTCENC)
#include "astcenc.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unordered_map>
#include <unistd.h>

namespace ps2x::gs::packcache
{
    namespace
    {
        namespace fs = std::filesystem;

        constexpr char kMagic[4] = {'R', 'T', 'A', '1'};
        struct Header
        {
            char magic[4];
            uint32_t version, width, height, levels, block;
            uint64_t srcSize;
            int64_t srcTime;
        };

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

        // <...>/textures/packs/<name> caches in <...>/textures/cache/<name>; a pack elsewhere
        // (RT_TEXTURE_PACK) beside itself, in <name>.rtcache.
        fs::path cacheDir(const std::string &packDir)
        {
            fs::path p = fs::path(packDir).lexically_normal();
            if (p.filename().empty())
                p = p.parent_path();
            if (p.parent_path().filename() == "packs")
                return p.parent_path().parent_path() / "cache" / p.filename();
            return p.parent_path() / (p.filename().string() + ".rtcache");
        }

        fs::path cacheFile(const std::string &packDir, uint64_t key)
        {
            char name[32];
            std::snprintf(name, sizeof(name), "%016llx.astc", static_cast<unsigned long long>(key));
            return cacheDir(packDir) / name;
        }

        bool sourceStamp(const std::string &png, uint64_t &size, int64_t &time)
        {
            std::error_code ec;
            size = fs::file_size(png, ec);
            if (ec)
                return false;
            time = static_cast<int64_t>(fs::last_write_time(png, ec).time_since_epoch().count());
            return !ec;
        }

        bool readHeader(std::ifstream &in, Header &h)
        {
            in.read(reinterpret_cast<char *>(&h), sizeof(h));
            return in && std::memcmp(h.magic, kMagic, 4) == 0 && h.version == 1 && h.block == 4 && h.levels >= 1 &&
                   h.levels <= 16 && h.width && h.height;
        }

        bool current(const std::string &packDir, const std::string &png, uint64_t key)
        {
            uint64_t size;
            int64_t time;
            if (!sourceStamp(png, size, time))
                return false;
            std::ifstream in(cacheFile(packDir, key), std::ios::binary);
            Header h;
            return in && readHeader(in, h) && h.srcSize == size && h.srcTime == time;
        }

        struct Entry
        {
            std::string png;
            uint64_t key;
        };

        std::vector<Entry> packImages(const std::string &packDir)
        {
            // One image per key, the last the walk meets: the same choice TextureTools::indexPack
            // makes (m_pack[key] = path), so the file that is cached is the file that is drawn. Two
            // PNGs of one key (a subfolder of originals, a pack made twice) shared one cache file,
            // each encoding over the other's stamp: one was always "missing", and every start and
            // every menu opening converted it again.
            std::vector<Entry> out;
            std::unordered_map<uint64_t, size_t> at;
            std::error_code ec;
            for (const auto &e : fs::recursive_directory_iterator(packDir, ec))
            {
                uint64_t key;
                if (!e.is_regular_file() || e.path().extension() != ".png" || !parseKey(e.path().filename().string(), key))
                    continue;
                const auto [it, added] = at.try_emplace(key, out.size());
                if (added)
                    out.push_back({e.path().string(), key});
                else
                    out[it->second].png = e.path().string();
            }
            return out;
        }

#if defined(PS2X_HAVE_ASTCENC)
        // One worker's encoder (a context per thread: astcenc's are single-threaded here).
        struct Encoder
        {
            astcenc_context *ctx = nullptr;
            Encoder()
            {
                astcenc_config config;
                if (astcenc_config_init(ASTCENC_PRF_LDR, 4, 4, 1, ASTCENC_PRE_FAST, 0, &config) == ASTCENC_SUCCESS)
                    if (astcenc_context_alloc(&config, 1, &ctx, nullptr) != ASTCENC_SUCCESS)
                        ctx = nullptr;
            }
            ~Encoder()
            {
                if (ctx)
                    astcenc_context_free(ctx);
            }

            bool level(std::vector<uint8_t> &rgba, uint32_t w, uint32_t h, std::vector<uint8_t> &out)
            {
                astcenc_image image{};
                image.dim_x = w;
                image.dim_y = h;
                image.dim_z = 1;
                image.data_type = ASTCENC_TYPE_U8;
                void *slices[1] = {rgba.data()};
                image.data = slices;
                const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
                const size_t bytes = static_cast<size_t>((w + 3u) / 4u) * ((h + 3u) / 4u) * 16u;
                const size_t at = out.size();
                out.resize(at + bytes);
                const bool ok = astcenc_compress_image(ctx, &image, &swizzle, out.data() + at, bytes, 0) == ASTCENC_SUCCESS;
                astcenc_compress_reset(ctx);
                return ok;
            }

            // The PNG, PS2 alpha (0x80 opaque, as the GS draws pack images), every mip level.
            bool encode(const std::string &packDir, const Entry &e)
            {
                if (!ctx)
                    return false;
                uint64_t srcSize;
                int64_t srcTime;
                if (!sourceStamp(e.png, srcSize, srcTime))
                    return false;
                Image img = LoadImage(e.png.c_str());
                if (!img.data)
                    return false;
                ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
                uint32_t w = static_cast<uint32_t>(img.width), h = static_cast<uint32_t>(img.height);
                std::vector<uint8_t> rgba(static_cast<const uint8_t *>(img.data),
                                          static_cast<const uint8_t *>(img.data) + static_cast<size_t>(w) * h * 4u);
                UnloadImage(img);
                for (size_t i = 3; i < rgba.size(); i += 4)
                    rgba[i] = static_cast<uint8_t>((rgba[i] * 128u + 127u) / 255u);

                Header hdr{};
                std::memcpy(hdr.magic, kMagic, 4);
                hdr.version = 1;
                hdr.width = w;
                hdr.height = h;
                hdr.block = 4;
                hdr.srcSize = srcSize;
                hdr.srcTime = srcTime;
                std::vector<uint8_t> data;
                std::vector<uint32_t> sizes;
                for (;;)
                {
                    const size_t before = data.size();
                    if (!level(rgba, w, h, data))
                        return false;
                    sizes.push_back(static_cast<uint32_t>(data.size() - before));
                    ++hdr.levels;
                    if (w == 1u && h == 1u)
                        break;
                    // The next level: a 2x2 box filter (edges clamped).
                    const uint32_t nw = std::max(1u, w / 2u), nh = std::max(1u, h / 2u);
                    std::vector<uint8_t> next(static_cast<size_t>(nw) * nh * 4u);
                    for (uint32_t y = 0; y < nh; ++y)
                        for (uint32_t x = 0; x < nw; ++x)
                        {
                            const uint32_t x0 = std::min(2u * x, w - 1u), x1 = std::min(2u * x + 1u, w - 1u);
                            const uint32_t y0 = std::min(2u * y, h - 1u), y1 = std::min(2u * y + 1u, h - 1u);
                            for (uint32_t c = 0; c < 4u; ++c)
                            {
                                const uint32_t sum = rgba[(static_cast<size_t>(y0) * w + x0) * 4u + c] +
                                                     rgba[(static_cast<size_t>(y0) * w + x1) * 4u + c] +
                                                     rgba[(static_cast<size_t>(y1) * w + x0) * 4u + c] +
                                                     rgba[(static_cast<size_t>(y1) * w + x1) * 4u + c];
                                next[(static_cast<size_t>(y) * nw + x) * 4u + c] = static_cast<uint8_t>((sum + 2u) / 4u);
                            }
                        }
                    rgba.swap(next);
                    w = nw;
                    h = nh;
                }

                std::error_code ec;
                const fs::path file = cacheFile(packDir, e.key);
                fs::create_directories(file.parent_path(), ec);
                const fs::path part = file.string() + "." + std::to_string(getpid()) + ".part";
                {
                    std::ofstream out(part, std::ios::binary | std::ios::trunc);
                    out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
                    out.write(reinterpret_cast<const char *>(sizes.data()), static_cast<std::streamsize>(sizes.size() * 4u));
                    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
                    if (!out)
                    {
                        fs::remove(part, ec);
                        return false;
                    }
                }
                fs::rename(part, file, ec);
                return !ec;
            }
        };
#endif
    }

    bool available()
    {
#if defined(PS2X_HAVE_ASTCENC)
        return true;
#else
        return false;
#endif
    }

    bool load(const std::string &packDir, const std::string &png, uint64_t key, Compressed &out)
    {
        uint64_t size;
        int64_t time;
        if (!sourceStamp(png, size, time))
            return false;
        std::ifstream in(cacheFile(packDir, key), std::ios::binary);
        Header h;
        if (!in || !readHeader(in, h) || h.srcSize != size || h.srcTime != time)
            return false;
        std::vector<uint32_t> sizes(h.levels);
        in.read(reinterpret_cast<char *>(sizes.data()), static_cast<std::streamsize>(sizes.size() * 4u));
        size_t total = 0;
        out.offsets.clear();
        for (uint32_t s : sizes)
        {
            out.offsets.push_back(total);
            total += s;
        }
        out.data.resize(total);
        in.read(reinterpret_cast<char *>(out.data.data()), static_cast<std::streamsize>(total));
        if (!in)
            return false;
        out.width = h.width;
        out.height = h.height;
        out.levels = h.levels;
        return true;
    }

    size_t missing(const std::string &packDir)
    {
        size_t n = 0;
        for (const Entry &e : packImages(packDir))
            n += current(packDir, e.png, e.key) ? 0u : 1u;
        return n;
    }

    void prepare(const std::string &packDir, Progress &progress, unsigned threads)
    {
#if defined(PS2X_HAVE_ASTCENC)
        std::vector<Entry> todo;
        for (Entry &e : packImages(packDir))
            if (!current(packDir, e.png, e.key))
                todo.push_back(std::move(e));
        progress.total = static_cast<uint32_t>(todo.size());
        progress.done = 0;
        if (todo.empty())
            return;
        if (threads == 0)
            threads = std::max(1u, std::thread::hardware_concurrency());
        std::atomic<size_t> next{0};
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < threads; ++t)
            workers.emplace_back([&] {
                Encoder enc;
                for (size_t i; !progress.cancel && (i = next.fetch_add(1)) < todo.size();)
                {
                    if (!enc.encode(packDir, todo[i]))
                        progress.failed.fetch_add(1);
                    progress.done.fetch_add(1);
                }
            });
        for (auto &w : workers)
            w.join();
#else
        (void)packDir;
        progress.total = 0;
#endif
    }
}
