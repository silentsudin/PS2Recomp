#pragma once

// Texture dumps and HD texture packs (Road Trip recomp). paraLLEl-GS decodes every texture it
// samples into an RGBA8 image (palette and TEXA applied); with texture tools on, each newly
// decoded texture is read back and named by a hash of its decoded pixels (so the same texture has
// the same name across runs, wherever the game puts it in GS memory, and a palette swap is a
// different texture). Dumps are written as <hash>_<W>x<H>_<psm>.png; a pack is a folder of PNGs
// whose names start with the 16-digit hash. Hashing, PNG writing and decoding run on a worker.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ps2x::gs
{
    class TextureTools
    {
    public:
        TextureTools();
        ~TextureTools();

        // Empty strings switch dumping / the pack off. Any thread.
        void configure(const std::string &dumpDir, const std::string &packDir);
        bool active() const;

        // A decoded texture read back by the GS (RGBA8, PS2 alpha: 0x80 = opaque). GS thread.
        void submit(uint64_t cacheKey, uint64_t stableKey, uint32_t width, uint32_t height, uint32_t psm,
                    std::vector<uint8_t> rgba);

        // The outcome for each submitted texture (when a pack is active).
        struct Replacement
        {
            uint64_t cacheKey;   // the GS texture cache key
            uint64_t stableKey;  // the cache key without the palette instance (prediction key)
            uint64_t contentKey; // its content hash (one image per content)
            bool hit = false;    // the pack replaces it
            uint32_t width = 0, height = 0;
            std::vector<uint8_t> rgba; // empty if the image for contentKey was handed out before
        };
        // Outcomes since the last call. GS thread.
        void collect(std::vector<Replacement> &out);

    private:
        struct Job
        {
            uint64_t cacheKey, stableKey;
            uint32_t width, height, psm;
            std::vector<uint8_t> rgba;
        };
        void worker();
        void indexPack(const std::string &dir);

        mutable std::mutex m_mutex;
        std::condition_variable m_cv;
        std::deque<Job> m_jobs;
        std::vector<Replacement> m_ready;
        std::string m_dumpDir, m_packDir;
        std::unordered_set<uint64_t> m_dumped;                // content keys already on disk
        std::unordered_map<uint64_t, std::string> m_pack;     // content key -> replacement PNG
        std::unordered_set<uint64_t> m_handedOut;             // content keys whose pixels were returned
        bool m_stop = false;
        std::thread m_thread;
    };
}
