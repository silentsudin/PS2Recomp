#include "runtime/gs/gs_texture_pack_cache.h"

#include "raylib.h" // PNG read (stb)

#if defined(PS2X_HAVE_ASTCENC)
#include "astcenc.h"
#endif
#if defined(PS2X_HAVE_BC7ENC)
#include "bc7enc.h"
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unistd.h>

namespace ps2x::gs::packcache
{
    namespace
    {
        namespace fs = std::filesystem;

        // One magic per format (ASTC's is the one files had before BC7 existed).
        constexpr char kMagicAstc[4] = {'R', 'T', 'A', '1'};
        constexpr char kMagicBc7[4] = {'R', 'T', 'B', '1'};
        struct Header
        {
            char magic[4];
            uint32_t version, width, height, levels, block;
            uint64_t srcSize;
            int64_t srcTime;
        };

        const char *magicOf(Format f)
        {
            return f == Format::Bc7 ? kMagicBc7 : kMagicAstc;
        }

        std::atomic<int> g_deviceAstc{-1}, g_deviceBc7{-1}; // -1: no hardware GS has asked yet

        bool encoderBuilt(Format f)
        {
            switch (f)
            {
#if defined(PS2X_HAVE_ASTCENC)
            case Format::Astc:
                return true;
#endif
#if defined(PS2X_HAVE_BC7ENC)
            case Format::Bc7:
                return true;
#endif
            default:
                return false;
            }
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

        std::mutex g_dirMutex;
        std::string g_privateRoot;                                // setPrivateCacheRoot
        std::unordered_map<std::string, fs::path> g_resolvedDirs; // pack folder -> its cache folder

        // Whether files can be created in `dir` (made if missing): a file is written and removed.
        bool writable(const fs::path &dir)
        {
            std::error_code ec;
            fs::create_directories(dir, ec);
            const fs::path probe = dir / (".probe." + std::to_string(getpid()));
            {
                std::ofstream out(probe, std::ios::binary | std::ios::trunc);
                if (!(out << 'x'))
                    return false;
            }
            fs::remove(probe, ec);
            return true;
        }

        // <...>/textures/packs/<name> caches in <...>/textures/cache/<name>; a pack elsewhere
        // (RT_TEXTURE_PACK) beside itself, in <name>.rtcache. Where that can't be written (a
        // textures folder copied in by another user, e.g. a root file manager on Android: every
        // image failed to save and the pack converted again at every start and menu opening, #17),
        // in the private root's <name> instead. Decided once per pack and run.
        fs::path cacheDir(const std::string &packDir)
        {
            std::lock_guard<std::mutex> lock(g_dirMutex);
            if (auto it = g_resolvedDirs.find(packDir); it != g_resolvedDirs.end())
                return it->second;
            fs::path p = fs::path(packDir).lexically_normal();
            if (p.filename().empty())
                p = p.parent_path();
            const fs::path beside = p.parent_path().filename() == "packs" ? p.parent_path().parent_path() / "cache" / p.filename()
                                                                          : p.parent_path() / (p.filename().string() + ".rtcache");
            fs::path dir = beside;
            if (!writable(beside) && !g_privateRoot.empty())
            {
                const fs::path fallback = fs::path(g_privateRoot) / p.filename();
                if (writable(fallback))
                {
                    std::fprintf(stderr, "[textures] can't write %s: pack cache in %s\n", beside.string().c_str(), fallback.string().c_str());
                    dir = fallback;
                }
                else
                    std::fprintf(stderr, "[textures] can't write %s or %s: the pack loads from PNG\n", beside.string().c_str(),
                                 fallback.string().c_str());
            }
            g_resolvedDirs.emplace(packDir, dir);
            return dir;
        }

        // Each format has its own file, so switching formats never reads the other's.
        fs::path cacheFile(const std::string &packDir, uint64_t key, Format f)
        {
            char name[32];
            std::snprintf(name, sizeof(name), "%016llx.%s", static_cast<unsigned long long>(key), f == Format::Bc7 ? "bc7" : "astc");
            return cacheDir(packDir) / name;
        }

        // An image this format's encoder couldn't take (the PNG doesn't load, the encoder refuses
        // it): its stamp, so it isn't tried again at every start while the PNG stays the same.
        fs::path failMarker(const std::string &packDir, uint64_t key, Format f)
        {
            fs::path p = cacheFile(packDir, key, f);
            p += ".fail";
            return p;
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

        bool readHeader(std::ifstream &in, Header &h, Format f)
        {
            in.read(reinterpret_cast<char *>(&h), sizeof(h));
            return in && std::memcmp(h.magic, magicOf(f), 4) == 0 && h.version == 1 && h.block == 4 && h.levels >= 1 &&
                   h.levels <= 16 && h.width && h.height;
        }

        // Why an image has no current cached copy (missing() counts each reason).
        enum class Stale : uint8_t
        {
            Current,
            Failed,    // no copy, but the encoder failed on this very PNG before: not tried again
            NoSource,  // the PNG's size/time can't be read
            NoFile,    // no cache file
            BadHeader, // not a cache file of this format and version
            Size,      // made from a PNG of another size
            Time,      // made from a PNG with another modification time
        };

        Stale staleness(const std::string &packDir, const std::string &png, uint64_t key, Format f, Header *stored = nullptr,
                        uint64_t *nowSize = nullptr, int64_t *nowTime = nullptr)
        {
            uint64_t size;
            int64_t time;
            if (!sourceStamp(png, size, time))
                return Stale::NoSource;
            if (nowSize)
                *nowSize = size;
            if (nowTime)
                *nowTime = time;
            std::ifstream in(cacheFile(packDir, key, f), std::ios::binary);
            if (!in)
            {
                std::ifstream marker(failMarker(packDir, key, f), std::ios::binary);
                uint64_t markedSize = 0;
                int64_t markedTime = 0;
                if (marker.read(reinterpret_cast<char *>(&markedSize), sizeof(markedSize)) &&
                    marker.read(reinterpret_cast<char *>(&markedTime), sizeof(markedTime)) && markedSize == size && markedTime == time)
                    return Stale::Failed;
                return Stale::NoFile;
            }
            Header h;
            if (!readHeader(in, h, f))
                return Stale::BadHeader;
            if (stored)
                *stored = h;
            if (h.srcSize != size)
                return Stale::Size;
            return h.srcTime == time ? Stale::Current : Stale::Time;
        }

        bool current(const std::string &packDir, const std::string &png, uint64_t key, Format f)
        {
            const Stale why = staleness(packDir, png, key, f);
            return why == Stale::Current || why == Stale::Failed;
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

        // One worker's encoder for one level (RGBA8, PS2 alpha) of the format: appends its blocks.
        struct LevelEncoder
        {
            virtual ~LevelEncoder() = default;
            virtual bool ok() const = 0;
            virtual bool level(std::vector<uint8_t> &rgba, uint32_t w, uint32_t h, std::vector<uint8_t> &out) = 0;
        };

#if defined(PS2X_HAVE_ASTCENC)
        // A context per thread (astcenc's are single-threaded here). Fast preset.
        struct AstcEncoder final : LevelEncoder
        {
            astcenc_context *ctx = nullptr;
            AstcEncoder()
            {
                astcenc_config config;
                if (astcenc_config_init(ASTCENC_PRF_LDR, 4, 4, 1, ASTCENC_PRE_FAST, 0, &config) == ASTCENC_SUCCESS)
                    if (astcenc_context_alloc(&config, 1, &ctx, nullptr) != ASTCENC_SUCCESS)
                        ctx = nullptr;
            }
            ~AstcEncoder() override
            {
                if (ctx)
                    astcenc_context_free(ctx);
            }
            bool ok() const override { return ctx != nullptr; }

            bool level(std::vector<uint8_t> &rgba, uint32_t w, uint32_t h, std::vector<uint8_t> &out) override
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
        };
#endif

#if defined(PS2X_HAVE_BC7ENC)
        // bc7enc (plain BC7, no RDO): uber level 0 (its fastest), every partition pattern (estimated
        // through its filter bank), RGBA errors weighted alike as astcenc's LDR profile does. PS2
        // alpha is never 255, so every block takes the alpha modes (5, 6, 7). On the M3 Max about
        // 1.8x the ASTC fast preset's time per image, within 0.5 dB of its error on a 4x pack;
        // perceptual (YCbCr) weights were slower and further from the PNG.
        struct Bc7Encoder final : LevelEncoder
        {
            bc7enc_compress_block_params params;
            Bc7Encoder()
            {
                static std::once_flag once;
                std::call_once(once, [] { bc7enc_compress_block_init(); });
                bc7enc_compress_block_params_init(&params);
                params.m_uber_level = 0;
                params.m_max_partitions = BC7ENC_MAX_PARTITIONS;
                bc7enc_compress_block_params_init_linear_weights(&params);
            }
            bool ok() const override { return true; }

            bool level(std::vector<uint8_t> &rgba, uint32_t w, uint32_t h, std::vector<uint8_t> &out) override
            {
                const uint32_t bw = (w + 3u) / 4u, bh = (h + 3u) / 4u;
                const size_t at = out.size();
                out.resize(at + static_cast<size_t>(bw) * bh * 16u);
                uint8_t *dst = out.data() + at;
                uint8_t block[64];
                for (uint32_t by = 0; by < bh; ++by)
                    for (uint32_t bx = 0; bx < bw; ++bx, dst += 16)
                    {
                        // Edge blocks repeat the last row/column (as the ASTC encoder does).
                        for (uint32_t y = 0; y < 4u; ++y)
                        {
                            const uint32_t sy = std::min(by * 4u + y, h - 1u);
                            for (uint32_t x = 0; x < 4u; ++x)
                            {
                                const uint32_t sx = std::min(bx * 4u + x, w - 1u);
                                std::memcpy(block + (y * 4u + x) * 4u, rgba.data() + (static_cast<size_t>(sy) * w + sx) * 4u, 4);
                            }
                        }
                        bc7enc_compress_block(dst, block, &params);
                    }
                return true;
            }
        };
#endif

        std::unique_ptr<LevelEncoder> makeEncoder(Format f)
        {
            switch (f)
            {
#if defined(PS2X_HAVE_ASTCENC)
            case Format::Astc:
                return std::make_unique<AstcEncoder>();
#endif
#if defined(PS2X_HAVE_BC7ENC)
            case Format::Bc7:
                return std::make_unique<Bc7Encoder>();
#endif
            default:
                return nullptr;
            }
        }

        // The PNG, PS2 alpha (0x80 opaque, as the GS draws pack images), every mip level.
        enum class Fail : uint8_t
        {
            None,
            Load,
            Encode,
            Write,
            Rename,
        };

        // The first failure of each kind is logged with its cause (a pack that never finishes
        // converting says why).
        void noteFailure(Fail kind, const std::string &png, const std::string &why)
        {
            static std::atomic<bool> logged[5];
            if (!logged[static_cast<size_t>(kind)].exchange(true))
                std::fprintf(stderr, "[textures] pack image not prepared (%s): %s: %s\n",
                             kind == Fail::Load ? "load" : kind == Fail::Encode ? "encode" : kind == Fail::Write ? "write" : "rename",
                             png.c_str(), why.c_str());
        }

        Fail encode(LevelEncoder &enc, Format f, const std::string &packDir, const Entry &e)
        {
            if (!enc.ok())
            {
                noteFailure(Fail::Encode, e.png, "no encoder context");
                return Fail::Encode;
            }
            uint64_t srcSize;
            int64_t srcTime;
            if (!sourceStamp(e.png, srcSize, srcTime))
            {
                noteFailure(Fail::Load, e.png, "can't read its size and time");
                return Fail::Load;
            }
            Image img = LoadImage(e.png.c_str());
            if (!img.data)
            {
                noteFailure(Fail::Load, e.png, "not a PNG raylib can read");
                return Fail::Load;
            }
            ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
            uint32_t w = static_cast<uint32_t>(img.width), h = static_cast<uint32_t>(img.height);
            std::vector<uint8_t> rgba(static_cast<const uint8_t *>(img.data),
                                      static_cast<const uint8_t *>(img.data) + static_cast<size_t>(w) * h * 4u);
            UnloadImage(img);
            for (size_t i = 3; i < rgba.size(); i += 4)
                rgba[i] = static_cast<uint8_t>((rgba[i] * 128u + 127u) / 255u);

            Header hdr{};
            std::memcpy(hdr.magic, magicOf(f), 4);
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
                if (!enc.level(rgba, w, h, data))
                {
                    noteFailure(Fail::Encode, e.png, "level " + std::to_string(hdr.levels) + " (" + std::to_string(w) + "x" + std::to_string(h) + ")");
                    return Fail::Encode;
                }
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
            const fs::path file = cacheFile(packDir, e.key, f);
            fs::create_directories(file.parent_path(), ec);
            const fs::path part = file.string() + "." + std::to_string(getpid()) + ".part";
            {
                std::ofstream out(part, std::ios::binary | std::ios::trunc);
                out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
                out.write(reinterpret_cast<const char *>(sizes.data()), static_cast<std::streamsize>(sizes.size() * 4u));
                out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
                if (!out)
                {
                    const int err = errno;
                    fs::remove(part, ec);
                    noteFailure(Fail::Write, e.png, part.string() + ": " + std::strerror(err));
                    return Fail::Write;
                }
            }
            fs::rename(part, file, ec);
            if (ec)
            {
                noteFailure(Fail::Rename, e.png, part.string() + " -> " + file.string() + ": " + ec.message());
                fs::remove(part, ec);
                return Fail::Rename;
            }
            return Fail::None;
        }
    }

    namespace
    {
        void markFailed(const std::string &packDir, const Entry &e, Format f)
        {
            uint64_t size;
            int64_t time;
            if (!sourceStamp(e.png, size, time))
                return;
            std::ofstream out(failMarker(packDir, e.key, f), std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char *>(&size), sizeof(size));
            out.write(reinterpret_cast<const char *>(&time), sizeof(time));
        }
    }

    const char *formatName(Format f)
    {
        switch (f)
        {
        case Format::Astc:
            return "ASTC";
        case Format::Bc7:
            return "BC7";
        default:
            return "none";
        }
    }

    Format wanted()
    {
        static const Format f = [] {
#if defined(__APPLE__) || defined(__ANDROID__)
            Format choice = Format::Astc;
#else
            Format choice = Format::Bc7;
#endif
            if (const char *e = std::getenv("RT_TEXTURE_CACHE_FORMAT"); e && *e)
            {
                const std::string v(e);
                if (v == "astc")
                    choice = Format::Astc;
                else if (v == "bc7")
                    choice = Format::Bc7;
                else if (v == "none" || v == "0")
                    choice = Format::None;
                else
                    std::fprintf(stderr, "[textures] RT_TEXTURE_CACHE_FORMAT=%s: not astc, bc7 or none; ignored\n", e);
            }
            if (const char *e = std::getenv("RT_TEXTURE_ASTC"); e && *e == '0')
                choice = Format::None;
            if (choice != Format::None && !encoderBuilt(choice))
            {
                std::fprintf(stderr, "[textures] this build has no %s encoder: texture packs load from PNG\n", formatName(choice));
                choice = Format::None;
            }
            return choice;
        }();
        return f;
    }

    void setPrivateCacheRoot(const std::string &dir)
    {
        std::lock_guard<std::mutex> lock(g_dirMutex);
        g_privateRoot = dir;
        g_resolvedDirs.clear();
    }

    void setDeviceSamples(bool astc, bool bc7)
    {
        const bool first = g_deviceAstc.exchange(astc ? 1 : 0) < 0;
        g_deviceBc7 = bc7 ? 1 : 0;
        if (first)
        {
            const Format w = wanted();
            std::fprintf(stderr, "[textures] pack cache: %s (device samples ASTC %s, BC7 %s)\n",
                         formatName(active()), astc ? "yes" : "no", bc7 ? "yes" : "no");
            if (w != Format::None && active() == Format::None)
                std::fprintf(stderr, "[textures] the GPU can't sample %s: texture packs load from PNG\n", formatName(w));
        }
    }

    Format active()
    {
        switch (wanted())
        {
        case Format::Astc:
            return g_deviceAstc.load() == 1 ? Format::Astc : Format::None;
        case Format::Bc7:
            return g_deviceBc7.load() == 1 ? Format::Bc7 : Format::None;
        default:
            return Format::None;
        }
    }

    bool available()
    {
        return active() != Format::None;
    }

    bool load(const std::string &packDir, const std::string &png, uint64_t key, Compressed &out)
    {
        const Format f = active();
        if (f == Format::None)
            return false;
        uint64_t size;
        int64_t time;
        if (!sourceStamp(png, size, time))
            return false;
        std::ifstream in(cacheFile(packDir, key, f), std::ios::binary);
        Header h;
        if (!in || !readHeader(in, h, f) || h.srcSize != size || h.srcTime != time)
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
        out.format = f;
        out.width = h.width;
        out.height = h.height;
        out.levels = h.levels;
        return true;
    }

    size_t missing(const std::string &packDir)
    {
        const Format f = active();
        if (f == Format::None)
            return 0;
        // Counted by reason, the first of each logged with what the cache file says against the
        // PNG (an image converted on every start shows here why).
        size_t n = 0, by[7] = {};
        std::string example[7];
        for (const Entry &e : packImages(packDir))
        {
            Header stored{};
            uint64_t size = 0;
            int64_t time = 0;
            const Stale why = staleness(packDir, e.png, e.key, f, &stored, &size, &time);
            if (why == Stale::Current || why == Stale::Failed)
                continue;
            ++n;
            const size_t i = static_cast<size_t>(why);
            if (by[i]++ == 0)
                example[i] = cacheFile(packDir, e.key, f).string() + " (stored size " + std::to_string(stored.srcSize) + " time " +
                             std::to_string(stored.srcTime) + ", PNG size " + std::to_string(size) + " time " + std::to_string(time) + ")";
        }
        if (n)
        {
            static const char *const names[7] = {"current", "failed before", "no PNG stamp", "no file", "bad header", "size differs", "time differs"};
            std::fprintf(stderr, "[textures] pack %s: %zu image(s) without a current %s copy:", packDir.c_str(), n, formatName(f));
            for (size_t i = 2; i < 7; ++i)
                if (by[i])
                    std::fprintf(stderr, " %s %zu", names[i], by[i]);
            std::fprintf(stderr, "\n");
            for (size_t i = 2; i < 7; ++i)
                if (by[i])
                    std::fprintf(stderr, "[textures]   %s, e.g. %s\n", names[i], example[i].c_str());
        }
        return n;
    }

    void prepare(const std::string &packDir, Progress &progress, unsigned threads)
    {
        const Format f = active();
        progress.total = 0;
        progress.done = 0;
        if (f == Format::None)
            return;
        std::vector<Entry> todo;
        for (Entry &e : packImages(packDir))
            if (!current(packDir, e.png, e.key, f))
                todo.push_back(std::move(e));
        progress.total = static_cast<uint32_t>(todo.size());
        if (todo.empty())
            return;
        if (threads == 0)
            threads = std::max(1u, std::thread::hardware_concurrency());
        std::atomic<size_t> next{0};
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < threads; ++t)
            workers.emplace_back([&] {
                const auto enc = makeEncoder(f);
                for (size_t i; !progress.cancel && (i = next.fetch_add(1)) < todo.size();)
                {
                    const Fail fail = enc ? encode(*enc, f, packDir, todo[i]) : Fail::Encode;
                    if (enc && (fail == Fail::Load || fail == Fail::Encode))
                        markFailed(packDir, todo[i], f);
                    else if (fail == Fail::None)
                    {
                        std::error_code ec;
                        fs::remove(failMarker(packDir, todo[i].key, f), ec); // (one from an earlier version of the PNG)
                    }
                    if (fail != Fail::None)
                    {
                        progress.failed.fetch_add(1);
                        (fail == Fail::Load     ? progress.failedLoad
                         : fail == Fail::Encode ? progress.failedEncode
                         : fail == Fail::Write  ? progress.failedWrite
                                                : progress.failedRename)
                            .fetch_add(1);
                    }
                    progress.done.fetch_add(1);
                }
            });
        for (auto &w : workers)
            w.join();
        std::fprintf(stderr, "[textures] pack prepare %s: %u of %u done%s, %u failed (load %u, encode %u, write %u, rename %u)\n",
                     packDir.c_str(), progress.done.load(), progress.total.load(), progress.cancel ? " (stopped)" : "",
                     progress.failed.load(), progress.failedLoad.load(), progress.failedEncode.load(), progress.failedWrite.load(),
                     progress.failedRename.load());
    }
}
