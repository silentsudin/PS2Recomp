#pragma once

// Save states: one archive type that writes, reads or hashes, so each subsystem has a single
// serializeState(StateArchive &) that lists its fields once. Fields are stored as raw
// little-endian bytes (every host we build for is little-endian), in chunks named by a fourcc
// with their size, so a reader can say which part is short or out of step.
//
// The bytes are a std::vector, and chunks() gives each chunk's offset, size and a 64-bit hash (for
// comparing two states part by part). A chunk is {u32 fourcc, u16 version, u64 size, data}; a
// reader finds chunks by id (ps2_save_state.h has the file around it).

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ps2x
{
    // A fast 64-bit hash over bytes (four interleaved multiply-xor lanes), for comparing states.
    inline uint64_t stateHash(const uint8_t *data, size_t size, uint64_t seed = 0x9E3779B97F4A7C15ull)
    {
        constexpr uint64_t kMul = 0x9FB21C651E98DF25ull;
        uint64_t lanes[4] = {seed, seed ^ 0xC2B2AE3D27D4EB4Full, seed ^ 0x165667B19E3779F9ull, seed ^ 0x85EBCA77C2B2AE63ull};
        size_t i = 0;
        for (; i + 32 <= size; i += 32)
        {
            for (int l = 0; l < 4; ++l)
            {
                uint64_t w;
                std::memcpy(&w, data + i + 8 * l, 8);
                lanes[l] = (lanes[l] ^ w) * kMul;
                lanes[l] ^= lanes[l] >> 29;
            }
        }
        uint64_t h = lanes[0] ^ (lanes[1] * 3u) ^ (lanes[2] * 5u) ^ (lanes[3] * 7u) ^ size;
        for (; i < size; ++i)
            h = (h ^ data[i]) * 0x100000001B3ull;
        h ^= h >> 31;
        h *= kMul;
        h ^= h >> 29;
        return h;
    }

    constexpr uint32_t fourcc(const char (&s)[5])
    {
        return uint32_t(uint8_t(s[0])) | uint32_t(uint8_t(s[1])) << 8 | uint32_t(uint8_t(s[2])) << 16 |
               uint32_t(uint8_t(s[3])) << 24;
    }

    inline std::string fourccName(uint32_t id)
    {
        std::string s(4, ' ');
        for (int i = 0; i < 4; ++i)
            s[i] = char((id >> (8 * i)) & 0xFFu);
        return s;
    }

    template <class T>
    struct IsStdArray : std::false_type
    {
    };
    template <class T, size_t N>
    struct IsStdArray<std::array<T, N>> : std::true_type
    {
    };

    class StateArchive
    {
    public:
        struct Chunk
        {
            uint32_t id = 0;
            size_t offset = 0; // of the chunk's data (after its header)
            size_t size = 0;
            uint64_t hash = 0;
        };

        static StateArchive writer(std::vector<uint8_t> &out)
        {
            out.clear();
            return StateArchive(&out, nullptr);
        }
        // A reader indexes the chunks first: begin() then finds a chunk by its id wherever it is,
        // so a state can be read whatever order (or extra chunks) the writer used.
        static StateArchive reader(const std::vector<uint8_t> &in)
        {
            StateArchive ar(nullptr, &in);
            ar.indexChunks();
            return ar;
        }

        // The chunks found by a reader (id, version, data offset and size), in file order.
        struct IndexEntry
        {
            uint32_t id = 0;
            uint16_t version = 0;
            size_t offset = 0;
            size_t size = 0;
        };
        [[nodiscard]] const std::vector<IndexEntry> &index() const noexcept { return m_index; }
        [[nodiscard]] const IndexEntry *find(uint32_t id) const noexcept
        {
            for (const IndexEntry &e : m_index)
                if (e.id == id)
                    return &e;
            return nullptr;
        }

        [[nodiscard]] bool saving() const noexcept { return m_out != nullptr; }
        [[nodiscard]] bool loading() const noexcept { return m_in != nullptr; }
        [[nodiscard]] bool ok() const noexcept { return m_error.empty(); }
        [[nodiscard]] const std::string &error() const noexcept { return m_error; }
        [[nodiscard]] const std::vector<Chunk> &chunks() const noexcept { return m_chunks; }

        // Marks the archive bad (the first reason sticks); reads then yield zeros.
        void fail(std::string why)
        {
            if (m_error.empty())
                m_error = std::move(why) + (m_chunkOpen ? " (in " + fourccName(m_chunkId) + ")" : "");
        }

        void bytes(void *data, size_t size)
        {
            if (size == 0)
                return;
            if (m_out)
            {
                const auto *p = static_cast<const uint8_t *>(data);
                m_out->insert(m_out->end(), p, p + size);
                return;
            }
            const size_t limit = m_chunkOpen ? m_chunkEnd : m_in->size();
            if (!ok() || m_pos + size > limit)
            {
                fail(m_chunkOpen ? "chunk shorter than this build expects" : "archive too short");
                std::memset(data, 0, size);
                return;
            }
            std::memcpy(data, m_in->data() + m_pos, size);
            m_pos += size;
        }

        // Scalars (and arrays of them) only: structs are written field by field, so a state does
        // not depend on how a build lays out host structs (padding, member order, sizes).
        template <class T>
        void value(T &v)
        {
            if constexpr (std::is_array_v<T>)
            {
                for (auto &x : v)
                    value(x);
            }
            else if constexpr (IsStdArray<T>::value)
            {
                for (auto &x : v)
                    value(x);
            }
            else if constexpr (std::is_same_v<T, bool>)
            {
                uint8_t b = v ? 1u : 0u;
                bytes(&b, 1);
                v = b != 0u;
            }
            else
            {
                static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>, "value() takes scalars: write structs field by field");
                bytes(&v, sizeof(T));
            }
        }

        template <class T>
        StateArchive &operator&(T &v)
        {
            value(v);
            return *this;
        }

        // A value only read or written through accessors (atomics, private fields).
        template <class T, class Get, class Set>
        void property(Get &&get, Set &&set)
        {
            T v{};
            if (saving())
                v = get();
            value(v);
            if (loading())
                set(v);
        }

        void size(size_t &n)
        {
            uint64_t v = n;
            value(v);
            if (loading())
            {
                if (v > (uint64_t(1) << 32))
                {
                    fail("implausible element count");
                    v = 0;
                }
                n = static_cast<size_t>(v);
            }
        }

        void string(std::string &s)
        {
            size_t n = s.size();
            size(n);
            if (loading())
                s.resize(n);
            bytes(s.data(), n);
        }

        // A vector of scalars (guest memory images and the like).
        template <class T>
        void podVector(std::vector<T> &v)
        {
            static_assert(std::is_arithmetic_v<T>, "podVector() takes scalars; use sequence()");
            size_t n = v.size();
            size(n);
            if (loading())
                v.resize(n);
            bytes(v.data(), n * sizeof(T));
        }

        template <class T>
        void podDeque(std::deque<T> &d)
        {
            static_assert(std::is_arithmetic_v<T>, "podDeque() takes scalars; use sequence()");
            size_t n = d.size();
            size(n);
            if (loading())
                d.assign(n, T{});
            for (T &x : d)
                value(x);
        }

        // Any sequence container: the count, then each element through `each(ar, element)`.
        template <class C, class F>
        void sequence(C &c, F &&each)
        {
            size_t n = c.size();
            size(n);
            if (loading())
            {
                c.clear();
                c.resize(n);
            }
            for (auto &x : c)
                each(*this, x);
        }

        // An ordered map (std::map / multimap): key and value through the callbacks.
        template <class M, class K, class V>
        void orderedMap(M &m, K &&key, V &&val)
        {
            size_t n = m.size();
            size(n);
            if (saving())
            {
                for (auto &[k, v] : m)
                {
                    auto kc = k;
                    key(*this, kc);
                    val(*this, v);
                }
                return;
            }
            m.clear();
            for (size_t i = 0; i < n && ok(); ++i)
            {
                typename M::key_type k{};
                typename M::mapped_type v{};
                key(*this, k);
                val(*this, v);
                m.emplace(std::move(k), std::move(v));
            }
        }

        // An unordered map, rebuilt so it iterates in the same order (code that walks one can pick
        // differently otherwise): elements are written in iteration order and inserted back in
        // reverse into a table with the same bucket count, which reproduces the order with the
        // standard library's node insertion (checked; a mismatch is reported, not fatal).
        template <class M, class K, class V>
        void unorderedMap(M &m, K &&key, V &&val)
        {
            size_t n = m.size();
            size(n);
            size_t buckets = m.bucket_count();
            size(buckets);
            if (saving())
            {
                for (auto &[k, v] : m)
                {
                    auto kc = k;
                    key(*this, kc);
                    val(*this, v);
                }
                return;
            }
            std::vector<std::pair<typename M::key_type, typename M::mapped_type>> items(n);
            std::vector<typename M::key_type> keys;
            keys.reserve(n);
            for (size_t i = 0; i < n && ok(); ++i)
            {
                key(*this, items[i].first);
                val(*this, items[i].second);
                keys.push_back(items[i].first);
            }
            m.clear();
            m.rehash(buckets);
            for (size_t i = n; i-- > 0;)
                m.emplace(std::move(items[i].first), std::move(items[i].second));
            orderCheck(m, keys);
        }

        // An unordered map whose iteration order the code using it never depends on: written in
        // key order, so the state doesn't depend on the standard library's hashing or history.
        template <class M, class K, class V>
        void keyOrderedMap(M &m, K &&key, V &&val)
        {
            size_t n = m.size();
            size(n);
            if (saving())
            {
                std::vector<typename M::key_type> keys;
                keys.reserve(n);
                for (const auto &kv : m)
                    keys.push_back(kv.first);
                std::sort(keys.begin(), keys.end());
                for (const auto &k : keys)
                {
                    auto kc = k;
                    key(*this, kc);
                    val(*this, m.find(k)->second);
                }
                return;
            }
            m.clear();
            for (size_t i = 0; i < n && ok(); ++i)
            {
                typename M::key_type k{};
                typename M::mapped_type v{};
                key(*this, k);
                val(*this, v);
                m.emplace(std::move(k), std::move(v));
            }
        }

        template <class S, class K>
        void unorderedSet(S &s, K &&key)
        {
            size_t n = s.size();
            size(n);
            size_t buckets = s.bucket_count();
            size(buckets);
            if (saving())
            {
                for (const auto &k : s)
                {
                    auto kc = k;
                    key(*this, kc);
                }
                return;
            }
            std::vector<typename S::key_type> items(n);
            for (size_t i = 0; i < n && ok(); ++i)
                key(*this, items[i]);
            s.clear();
            s.rehash(buckets);
            for (size_t i = n; i-- > 0;)
                s.emplace(std::move(items[i]));
        }

        // Chunks: begin(id, version) ... end(). Each chunk carries its own version, so a loader
        // can keep reading older layouts of it (version() while reading); a reader checks the id
        // and that the chunk was read to its end.
        void begin(uint32_t id, uint16_t version = 1)
        {
            if (m_chunkOpen)
                end();
            m_chunkOpen = true;
            m_chunkId = id;
            uint32_t tag = id;
            uint16_t ver = version;
            uint64_t size = 0;
            if (m_out)
            {
                value(tag);
                value(ver);
                m_sizeAt = m_out->size();
                value(size);
                m_chunkStart = m_out->size();
                m_version = version;
                return;
            }
            const IndexEntry *e = ok() ? find(id) : nullptr;
            if (!e)
            {
                if (ok())
                    fail("the state has no chunk " + fourccName(id));
                m_version = version;
                m_chunkStart = m_chunkEnd = m_pos;
                return;
            }
            m_version = e->version;
            if (e->version > version)
                fail("chunk " + fourccName(id) + " version " + std::to_string(e->version) + " is newer than this build reads (" +
                     std::to_string(version) + ")");
            m_pos = e->offset;
            m_chunkStart = e->offset;
            m_chunkEnd = e->offset + e->size;
        }

        // The version of the chunk being read (or written).
        [[nodiscard]] uint16_t version() const noexcept { return m_version; }

        // Reader: passes over the rest of the open chunk.
        void skipRest()
        {
            if (m_in && m_chunkOpen && ok())
                m_pos = m_chunkEnd;
        }

        void end()
        {
            if (!m_chunkOpen)
                return;
            m_chunkOpen = false;
            Chunk c{m_chunkId, m_chunkStart, 0, 0};
            if (m_out)
            {
                c.size = m_out->size() - m_chunkStart;
                const uint64_t size = c.size;
                std::memcpy(m_out->data() + m_sizeAt, &size, 8);
                c.hash = stateHash(m_out->data() + c.offset, c.size);
            }
            else
            {
                c.size = m_chunkEnd - m_chunkStart;
                if (ok() && m_pos != m_chunkEnd)
                    fail("chunk " + fourccName(m_chunkId) + " read " + std::to_string(m_pos - m_chunkStart) + " of " +
                         std::to_string(c.size) + " bytes");
                if (ok())
                    c.hash = stateHash(m_in->data() + c.offset, c.size);
            }
            m_chunks.push_back(c);
        }

        // Order checks of rebuilt unordered maps that did not come out the same.
        [[nodiscard]] uint32_t orderMismatches() const noexcept { return m_orderMismatches; }

    private:
        StateArchive(std::vector<uint8_t> *out, const std::vector<uint8_t> *in) : m_out(out), m_in(in) {}

        void indexChunks()
        {
            size_t pos = 0;
            const size_t n = m_in->size();
            while (pos < n)
            {
                if (pos + 14 > n)
                {
                    fail("archive damaged (chunk header cut short)");
                    return;
                }
                IndexEntry e;
                uint64_t size = 0;
                std::memcpy(&e.id, m_in->data() + pos, 4);
                std::memcpy(&e.version, m_in->data() + pos + 4, 2);
                std::memcpy(&size, m_in->data() + pos + 6, 8);
                e.offset = pos + 14;
                if (size > n - e.offset)
                {
                    fail("archive damaged (chunk " + fourccName(e.id) + " runs past the end)");
                    return;
                }
                e.size = static_cast<size_t>(size);
                m_index.push_back(e);
                pos = e.offset + e.size;
            }
        }

        template <class M, class Keys>
        void orderCheck(const M &m, const Keys &keys)
        {
            size_t i = 0;
            for (const auto &[k, v] : m)
            {
                (void)v;
                if (i >= keys.size() || !(keys[i] == k))
                {
                    ++m_orderMismatches;
                    return;
                }
                ++i;
            }
        }

        std::vector<uint8_t> *m_out = nullptr;
        const std::vector<uint8_t> *m_in = nullptr;
        size_t m_pos = 0;
        std::string m_error;
        std::vector<Chunk> m_chunks;
        std::vector<IndexEntry> m_index;
        bool m_chunkOpen = false;
        uint32_t m_chunkId = 0;
        size_t m_chunkStart = 0, m_chunkEnd = 0, m_sizeAt = 0;
        uint32_t m_orderMismatches = 0;
        uint16_t m_version = 0;
    };
}
