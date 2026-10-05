#pragma once

// Texture dumps and HD packs for the hardware GS (gs_hw_backend.cpp). The backend decodes every
// texture on the CPU, so each new decode goes to TextureTools as it is (named by its pixels, as
// paraLLEl-GS's dumps are); pack hits come back as images created here and bound in place of the
// decoded texture. Same rules as the paraLLEl-GS backend: images are created at most 24 MB per
// frame, kept within a memory budget (least recently drawn go first), and another palette of a
// replaced texture is drawn from its pack image recoloured.

#if defined(PS2X_HAVE_PGS)

#include "../gs_texture_tools.h"

#include "device.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ps2x::gs
{
    class HwTexturePacks
    {
    public:
        // Empty strings switch dumping / the pack off (any thread; takes effect at the next service()).
        void configure(const std::string &dumpDir, const std::string &packDir);
        // Dumping or a pack is on (as of the last service; cheap: checked for every new batch).
        bool active() const { return m_active.load(std::memory_order_relaxed); }
        size_t packSize() const { return m_tools.packSize(); }
        size_t replacedCount() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_replacedContent.size();
        }

        // A newly decoded texture (RGBA8, PS2 alpha). stableKey: its description without the palette.
        // GS thread.
        void submit(uint64_t cacheKey, uint64_t stableKey, uint32_t width, uint32_t height, uint32_t psm,
                    std::vector<uint8_t> rgba);

        // What a texture is drawn with: a pack image (and, for another palette, its recolour map:
        // rows r, g, b, a then the offset), or no image. GS thread.
        struct Bound
        {
            Vulkan::ImageHandle image;
            bool recolor = false;
            float transform[20] = {};
        };
        Bound lookup(uint64_t cacheKey, uint64_t stableKey);

        // Collects TextureTools' results, creates pending images within the frame's budget and evicts
        // over the memory budget. Returns true when every texture must be submitted again and
        // looked up again (switching packs, evictions). GS thread, with the device lock held.
        bool service(Vulkan::Device &device);
        // No service() for 50 ms while a configuration change is pending or tools are active
        // (headless runs present only now and then). Any thread.
        bool serviceDue() const
        {
            if (!m_reconfigurePending.load(std::memory_order_relaxed) && !active())
                return false;
            const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch()).count();
            return now - m_lastService.load(std::memory_order_relaxed) > 50;
        }

    private:
        struct Image
        {
            Vulkan::ImageHandle image;
            size_t bytes = 0;
            uint64_t lastUsed = 0;
        };
        struct Binding
        {
            uint64_t contentKey = 0;
            bool recolor = false;
            float transform[20] = {};
        };
        struct Pending
        {
            uint32_t width = 0, height = 0;
            std::vector<uint8_t> rgba;
            bool queued = false;
        };
        static constexpr size_t kCreateBytesPerFrame = 24u << 20;

        TextureTools m_tools;
        mutable std::mutex m_mutex; // lookup (GS thread) against service (presentation)
        std::mutex m_configMutex;
        std::string m_dumpDir, m_packDir;
        bool m_reconfigure = false;
        std::atomic<bool> m_reconfigurePending{false};
        std::atomic<bool> m_active{false};
        std::atomic<int64_t> m_lastService{0};
        std::unordered_map<uint64_t, Image> m_images;       // content key -> image
        std::unordered_map<uint64_t, Binding> m_byCache;    // decoded texture -> its replacement
        std::unordered_map<uint64_t, Binding> m_byStable;   // prediction: the description's last replacement
        std::unordered_map<uint64_t, Pending> m_pending;    // content key -> pixels waiting for an image
        std::deque<uint64_t> m_pendingOrder;
        std::unordered_map<uint64_t, bool> m_replacedContent;
        std::vector<TextureTools::Replacement> m_results;
        uint64_t m_frame = 0; // the last service, in ms
        int64_t m_lastLog = 0;
        size_t m_bytes = 0, m_evicted = 0;
        size_t m_budget = [] {
            const char *mb = std::getenv("RT_TEXTURE_PACK_BUDGET_MB");
            return size_t(mb ? std::strtoull(mb, nullptr, 10) : 2048) << 20;
        }();
    };
}

#endif
