#include "runtime/gs/ps2_gif_arbiter.h"
#include <algorithm>
#include <cstring>
#include <iterator>

GifArbiter::GifArbiter(ProcessPacketFn processFn)
    : m_processFn(std::move(processFn))
{
}

bool GifArbiter::isImagePacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes < 16u)
        return false;

    uint64_t tagLo = 0;
    std::memcpy(&tagLo, data, sizeof(tagLo));
    const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
    return flg == 2u;
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl)
{
    if (!data || sizeBytes < 16 || (!m_processFn && !m_processPathFn))
        return;

    GifArbiterPacket pkt;
    pkt.pathId = pathId;
    pkt.path2DirectHl = (pathId == GifPathId::Path2) && path2DirectHl;
    pkt.path3Image = (pathId == GifPathId::Path3) && isImagePacket(data, sizeBytes);
    if (m_free.empty())
    {
        std::lock_guard<std::mutex> lock(m_poolMutex);
        m_free.swap(m_returned);
    }
    if (!m_free.empty())
    {
        pkt.data = std::move(m_free.back());
        m_free.pop_back();
    }
    pkt.data.resize(sizeBytes);
    std::memcpy(pkt.data.data(), data, sizeBytes);
    m_queue.push_back(std::move(pkt));
}

void GifArbiter::recycle(std::vector<GifArbiterPacket> &&packets)
{
    std::lock_guard<std::mutex> lock(m_poolMutex);
    for (auto &pkt : packets)
        if (m_returned.size() < 4096u && pkt.data.capacity() != 0u)
            m_returned.push_back(std::move(pkt.data));
    packets.clear();
}

void GifArbiter::sortQueue()
{
    std::stable_sort(m_queue.begin(), m_queue.end(),
                     [](const GifArbiterPacket &a, const GifArbiterPacket &b)
                     {
                         // DIRECTHL cannot preempt PATH3 IMAGE transfers.
                         if (a.path2DirectHl != b.path2DirectHl || a.path3Image != b.path3Image)
                         {
                             if (a.path3Image && b.path2DirectHl)
                                 return true;
                             if (a.path2DirectHl && b.path3Image)
                                 return false;
                         }
                         return pathPriority(a.pathId) < pathPriority(b.pathId);
                     });
}

void GifArbiter::process(const std::vector<GifArbiterPacket> &packets) const
{
    for (const auto &pkt : packets)
    {
        if (pkt.data.empty())
            continue;
        if (m_processPathFn)
            m_processPathFn(pkt.data.data(), static_cast<uint32_t>(pkt.data.size()), pkt.pathId);
        else if (m_processFn)
            m_processFn(pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
    }
}

void GifArbiter::drain()
{
    if (!m_processFn && !m_processPathFn)
        return;
    sortQueue();
    process(m_queue);
    m_queue.clear();
}

void GifArbiter::drainInto(std::vector<GifArbiterPacket> &out)
{
    if (m_queue.empty())
        return;
    sortQueue();
    if (out.empty())
        out.swap(m_queue);
    else
    {
        std::move(m_queue.begin(), m_queue.end(), std::back_inserter(out));
        m_queue.clear();
    }
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}
