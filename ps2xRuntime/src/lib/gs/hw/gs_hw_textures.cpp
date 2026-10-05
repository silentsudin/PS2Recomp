// Texture dumps and HD packs for the hardware GS (see gs_hw_textures.h).

#include "gs_hw_textures.h"

#if defined(PS2X_HAVE_PGS)

#include <algorithm>
#include <cstdio>

namespace ps2x::gs
{
    void HwTexturePacks::configure(const std::string &dumpDir, const std::string &packDir)
    {
        std::lock_guard<std::mutex> lock(m_configMutex);
        m_dumpDir = dumpDir;
        m_packDir = packDir;
        m_reconfigure = true;
        m_reconfigurePending = true;
    }

    void HwTexturePacks::submit(uint64_t cacheKey, uint64_t stableKey, uint32_t width, uint32_t height, uint32_t psm,
                                std::vector<uint8_t> rgba)
    {
        m_tools.submit(cacheKey, stableKey, width, height, psm, std::move(rgba));
    }

    HwTexturePacks::Bound HwTexturePacks::lookup(uint64_t cacheKey, uint64_t stableKey)
    {
        // This decode's own result, else the description's last replacement (Road Trip reloads
        // palettes every frame: each reload is a new decode, which the prediction covers until its
        // own result arrives).
        std::lock_guard<std::mutex> lock(m_mutex);
        const Binding *b = nullptr;
        if (auto it = m_byCache.find(cacheKey); it != m_byCache.end())
            b = &it->second;
        else if (auto st = m_byStable.find(stableKey); st != m_byStable.end())
            b = &st->second;
        if (!b)
            return {};
        auto img = m_images.find(b->contentKey);
        if (img == m_images.end())
            return {};
        img->second.lastUsed = m_frame;
        Bound out;
        out.image = img->second.image;
        out.recolor = b->recolor;
        if (b->recolor)
            std::copy(std::begin(b->transform), std::end(b->transform), out.transform);
        return out;
    }

    bool HwTexturePacks::service(Vulkan::Device &device)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
        m_lastService = now;
        m_frame = static_cast<uint64_t>(now); // lookups stamp images with it
        bool resubmit = false;
        {
            std::lock_guard<std::mutex> config(m_configMutex);
            if (m_reconfigure)
            {
                // Another pack (or none): forget every replacement; the textures go to TextureTools afresh.
                m_reconfigure = false;
                m_reconfigurePending = false;
                m_tools.configure(m_dumpDir, m_packDir);
                m_images.clear();
                m_byCache.clear();
                m_byStable.clear();
                m_pending.clear();
                m_pendingOrder.clear();
                m_replacedContent.clear();
                m_bytes = 0;
                resubmit = true;
            }
        }
        m_active = m_tools.active();
        if (!m_active)
            return resubmit;

        m_results.clear();
        m_tools.collect(m_results);
        for (auto &r : m_results)
        {
            if (!r.hit)
            {
                // This description decodes to something the pack doesn't have: stop predicting it.
                m_byStable.erase(r.stableKey);
                m_byCache.erase(r.cacheKey);
                continue;
            }
            Binding b;
            b.contentKey = r.contentKey;
            b.recolor = r.recolor;
            if (r.recolor)
                std::copy(std::begin(r.transform), std::end(r.transform), b.transform);
            m_byCache[r.cacheKey] = b;
            m_byStable[r.stableKey] = b;
            m_replacedContent[r.contentKey] = true;
            if (!r.rgba.empty() && !m_images.count(r.contentKey))
            {
                Pending &p = m_pending[r.contentKey];
                p.width = r.width;
                p.height = r.height;
                p.rgba = std::move(r.rgba);
                if (!p.queued)
                {
                    p.queued = true;
                    m_pendingOrder.push_back(r.contentKey);
                }
            }
        }

        // New images within the frame's budget (a 4x pack has images of 50 MB with mips).
        size_t created = 0;
        while (!m_pendingOrder.empty() && created < kCreateBytesPerFrame)
        {
            const uint64_t key = m_pendingOrder.front();
            m_pendingOrder.pop_front();
            auto it = m_pending.find(key);
            if (it == m_pending.end() || it->second.rgba.empty())
                continue;
            Pending p = std::move(it->second);
            m_pending.erase(it);
            auto info = Vulkan::ImageCreateInfo::immutable_2d_image(p.width, p.height, VK_FORMAT_R8G8B8A8_UNORM, true);
            Vulkan::ImageInitialData init = {p.rgba.data(), 0, 0};
            Image image;
            image.image = device.create_image(info, &init);
            if (!image.image)
                continue;
            image.bytes = p.rgba.size() * 4 / 3;
            image.lastUsed = m_frame;
            created += p.rgba.size();
            m_bytes += image.bytes;
            m_images[key] = std::move(image);
        }

        // Over the budget: images not drawn for two seconds go, oldest first, down to 80%.
        if (m_bytes > m_budget)
        {
            std::vector<std::pair<uint64_t, uint64_t>> byAge;
            for (auto &[key, img] : m_images)
                if (m_frame - img.lastUsed > 2000)
                    byAge.emplace_back(img.lastUsed, key);
            std::sort(byAge.begin(), byAge.end());
            for (auto [age, key] : byAge)
            {
                if (m_bytes <= m_budget / 5 * 4)
                    break;
                auto it = m_images.find(key);
                m_bytes -= it->second.bytes;
                m_images.erase(it);
                m_tools.forget(key); // its pixels are loaded again when it is needed
                ++m_evicted;
                resubmit = true; // submitted again, so the evicted image's pixels come back
            }
        }

        if (now - m_lastLog >= 5000 && !m_images.empty())
        {
            m_lastLog = now;
            const auto ts = m_tools.stats();
            std::fprintf(stderr, "[textures] pack: %zu images, %zu MB (budget %zu), %zu pending, %zu evicted; palettes: %zu "
                                 "patterns, %zu/%zu fitted\n",
                         m_images.size(), m_bytes >> 20, m_budget >> 20, m_pendingOrder.size(), m_evicted, ts.patterns,
                         ts.fitted, ts.fits);
        }
        return resubmit;
    }
}

#endif
