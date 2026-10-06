// Save state files (see runtime/ps2_save_state.h for the layout): a header that can be read on its
// own (menus list slots from it), then the state's chunks compressed with zstd.

#include "runtime/ps2_save_state.h"

#include "ps2x/state_archive.h"

#include <zstd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace ps2_save_state
{
    namespace
    {
        constexpr char kMagic[8] = {'R', 'T', 'S', 'T', 'A', 'T', 'E', '\x1A'};
        constexpr uint32_t kCompressionNone = 0;
        constexpr uint32_t kCompressionZstd = 1;
        constexpr int kZstdLevel = 3;
        // Fixed part: magic, format version, header size, game, compression, vblank, saved time,
        // raw size, payload size.
        constexpr size_t kFixedHeader = 8 + 4 + 4 + 4 + 4 + 8 + 8 + 8 + 8;
        constexpr size_t kMaxHeader = 1u << 20;
        constexpr size_t kMaxRaw = 256u << 20;

        class Out
        {
        public:
            std::vector<uint8_t> bytes;
            template <class T>
            void put(T v)
            {
                const auto *p = reinterpret_cast<const uint8_t *>(&v);
                bytes.insert(bytes.end(), p, p + sizeof(T));
            }
            void str(const std::string &s)
            {
                put<uint32_t>(static_cast<uint32_t>(s.size()));
                bytes.insert(bytes.end(), s.begin(), s.end());
            }
        };

        class In
        {
        public:
            In(const uint8_t *data, size_t size) : m_data(data), m_size(size) {}
            template <class T>
            T get()
            {
                T v{};
                if (m_pos + sizeof(T) > m_size)
                {
                    m_bad = true;
                    return v;
                }
                std::memcpy(&v, m_data + m_pos, sizeof(T));
                m_pos += sizeof(T);
                return v;
            }
            std::string str()
            {
                const uint32_t n = get<uint32_t>();
                if (m_bad || n > m_size - m_pos)
                {
                    m_bad = true;
                    return {};
                }
                std::string s(reinterpret_cast<const char *>(m_data + m_pos), n);
                m_pos += n;
                return s;
            }
            [[nodiscard]] bool bad() const { return m_bad; }
            [[nodiscard]] size_t pos() const { return m_pos; }

        private:
            const uint8_t *m_data;
            size_t m_size;
            size_t m_pos = 0;
            bool m_bad = false;
        };

        std::vector<uint8_t> encodeHeader(const FileInfo &info)
        {
            Out o;
            o.bytes.insert(o.bytes.end(), kMagic, kMagic + 8);
            o.put<uint32_t>(kFileFormatVersion);
            o.put<uint32_t>(0); // header size, patched below
            o.put<uint32_t>(info.gameCrc);
            o.put<uint32_t>(info.compression);
            o.put<uint64_t>(info.vblank);
            o.put<int64_t>(info.savedUnixTime);
            o.put<uint64_t>(info.rawSize);
            o.put<uint64_t>(info.payloadSize);
            o.put<uint32_t>(static_cast<uint32_t>(info.metadata.size()));
            for (const auto &[k, v] : info.metadata)
            {
                o.str(k);
                o.str(v);
            }
            o.put<uint32_t>(static_cast<uint32_t>(info.chunks.size()));
            for (const FileChunk &c : info.chunks)
            {
                o.put<uint32_t>(c.id);
                o.put<uint16_t>(c.version);
                o.put<uint64_t>(c.size);
                o.put<uint64_t>(c.hash);
                o.put<uint64_t>(c.digest);
            }
            const uint32_t size = static_cast<uint32_t>(o.bytes.size() + 8);
            std::memcpy(o.bytes.data() + 12, &size, 4);
            o.put<uint64_t>(ps2x::stateHash(o.bytes.data(), o.bytes.size()));
            return o.bytes;
        }

        // Header from the first bytes of a file; `headerSize` is set once the size is known.
        bool decodeHeader(const std::vector<uint8_t> &bytes, FileInfo &info, size_t &headerSize, std::string &error)
        {
            if (bytes.size() < kFixedHeader || std::memcmp(bytes.data(), kMagic, 8) != 0)
            {
                error = "not a save state";
                return false;
            }
            In in(bytes.data() + 8, bytes.size() - 8);
            info.formatVersion = in.get<uint32_t>();
            headerSize = in.get<uint32_t>();
            if (info.formatVersion == 0 || info.formatVersion > kFileFormatVersion)
            {
                error = "made by a newer version of the app (file format " + std::to_string(info.formatVersion) + ")";
                return false;
            }
            if (headerSize < kFixedHeader + 8 || headerSize > kMaxHeader)
            {
                error = "the file is damaged (header size)";
                return false;
            }
            if (bytes.size() < headerSize)
                return true; // the caller reads more
            uint64_t stored = 0;
            std::memcpy(&stored, bytes.data() + headerSize - 8, 8);
            if (ps2x::stateHash(bytes.data(), headerSize - 8) != stored)
            {
                error = "the file is damaged (header)";
                return false;
            }
            info.gameCrc = in.get<uint32_t>();
            info.compression = in.get<uint32_t>();
            info.vblank = in.get<uint64_t>();
            info.savedUnixTime = in.get<int64_t>();
            info.rawSize = in.get<uint64_t>();
            info.payloadSize = in.get<uint64_t>();
            const uint32_t metaCount = in.get<uint32_t>();
            info.metadata.clear();
            for (uint32_t i = 0; i < metaCount && !in.bad(); ++i)
            {
                std::string k = in.str();
                std::string v = in.str();
                info.metadata.emplace_back(std::move(k), std::move(v));
            }
            const uint32_t chunkCount = in.get<uint32_t>();
            info.chunks.clear();
            for (uint32_t i = 0; i < chunkCount && !in.bad(); ++i)
            {
                FileChunk c;
                c.id = in.get<uint32_t>();
                c.version = in.get<uint16_t>();
                c.size = in.get<uint64_t>();
                c.hash = in.get<uint64_t>();
                c.digest = in.get<uint64_t>();
                info.chunks.push_back(c);
            }
            if (in.bad() || 8 + in.pos() + 8 != headerSize)
            {
                error = "the file is damaged (header layout)";
                return false;
            }
            if (info.compression != kCompressionZstd && info.compression != kCompressionNone)
            {
                error = "unknown compression";
                return false;
            }
            if (info.rawSize > kMaxRaw)
            {
                error = "the file is damaged (implausible size)";
                return false;
            }
            return true;
        }

        bool readAll(const std::string &path, std::vector<uint8_t> &out, size_t limit, std::string &error)
        {
            std::ifstream f(path, std::ios::binary);
            if (!f)
            {
                error = "can't open the file";
                return false;
            }
            f.seekg(0, std::ios::end);
            const auto size = static_cast<size_t>(std::max<std::streamoff>(0, f.tellg()));
            f.seekg(0);
            out.resize(std::min(size, limit));
            f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size()));
            if (static_cast<size_t>(f.gcount()) != out.size())
            {
                error = "can't read the file";
                return false;
            }
            return true;
        }
    }

    std::string FileInfo::meta(const std::string &key) const
    {
        for (const auto &[k, v] : metadata)
            if (k == key)
                return v;
        return {};
    }

    bool writeStateFile(const std::string &path, FileInfo &info, const std::vector<uint8_t> &raw, std::string &error, bool compress)
    {
        info.formatVersion = kFileFormatVersion;
        info.rawSize = raw.size();
        info.compression = compress ? kCompressionZstd : kCompressionNone;
        std::vector<uint8_t> payload;
        if (compress)
        {
            payload.resize(ZSTD_compressBound(raw.size()));
            ZSTD_CCtx *cctx = ZSTD_createCCtx();
            if (!cctx)
            {
                error = "out of memory";
                return false;
            }
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, kZstdLevel);
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1);
            const size_t n = ZSTD_compress2(cctx, payload.data(), payload.size(), raw.data(), raw.size());
            ZSTD_freeCCtx(cctx);
            if (ZSTD_isError(n))
            {
                error = std::string("compression failed: ") + ZSTD_getErrorName(n);
                return false;
            }
            payload.resize(n);
        }
        info.payloadSize = compress ? payload.size() : raw.size();
        const std::vector<uint8_t> header = encodeHeader(info);
        info.fileSize = header.size() + info.payloadSize;

        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path target(path);
        if (target.has_parent_path())
            fs::create_directories(target.parent_path(), ec);
        // Its own temp name: two saves to one slot in quick succession write side by side, and
        // the last rename wins.
        static std::atomic<uint32_t> s_writes{0};
        const fs::path tmp = target.string() + ".tmp" + std::to_string(s_writes.fetch_add(1) + 1);
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f)
            {
                error = "can't write " + tmp.string();
                return false;
            }
            f.write(reinterpret_cast<const char *>(header.data()), static_cast<std::streamsize>(header.size()));
            const std::vector<uint8_t> &body = compress ? payload : raw;
            f.write(reinterpret_cast<const char *>(body.data()), static_cast<std::streamsize>(body.size()));
            f.flush();
            if (!f)
            {
                f.close();
                fs::remove(tmp, ec);
                error = "can't write " + tmp.string() + " (disk full?)";
                return false;
            }
        }
        fs::rename(tmp, target, ec);
        if (ec)
        {
            fs::remove(tmp, ec);
            error = "can't rename the file into place";
            return false;
        }
        return true;
    }

    bool readStateFileInfo(const std::string &path, FileInfo &info, std::string &error)
    {
        std::vector<uint8_t> head;
        if (!readAll(path, head, kMaxHeader, error))
            return false;
        size_t headerSize = 0;
        if (!decodeHeader(head, info, headerSize, error))
            return false;
        if (head.size() < headerSize)
        {
            error = "the file is damaged (cut short)";
            return false;
        }
        std::error_code ec;
        info.fileSize = static_cast<uint64_t>(std::filesystem::file_size(path, ec));
        return true;
    }

    bool readStateFile(const std::string &path, FileInfo &info, std::vector<uint8_t> &raw, std::string &error)
    {
        std::vector<uint8_t> bytes;
        if (!readAll(path, bytes, ~size_t(0), error))
            return false;
        size_t headerSize = 0;
        if (!decodeHeader(bytes, info, headerSize, error))
            return false;
        if (bytes.size() < headerSize || bytes.size() - headerSize != info.payloadSize)
        {
            error = "the file is damaged (cut short)";
            return false;
        }
        info.fileSize = bytes.size();
        if (info.compression == kCompressionNone)
        {
            if (info.payloadSize != info.rawSize)
            {
                error = "the file is damaged (sizes)";
                return false;
            }
            raw.assign(bytes.begin() + static_cast<std::ptrdiff_t>(headerSize), bytes.end());
            return true;
        }
        const unsigned long long content = ZSTD_getFrameContentSize(bytes.data() + headerSize, info.payloadSize);
        if (content != info.rawSize)
        {
            error = "the file is damaged (compressed data)";
            return false;
        }
        raw.resize(info.rawSize);
        // The frame's checksum covers the data; each chunk's hash is checked after.
        const size_t n = ZSTD_decompress(raw.data(), raw.size(), bytes.data() + headerSize, info.payloadSize);
        if (ZSTD_isError(n) || n != info.rawSize)
        {
            error = "the file is damaged (compressed data)";
            raw.clear();
            return false;
        }
        return true;
    }
}
