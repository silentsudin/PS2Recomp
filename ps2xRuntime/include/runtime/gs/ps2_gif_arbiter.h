#ifndef PS2_GIF_ARBITER_H
#define PS2_GIF_ARBITER_H

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

enum class GifPathId : uint8_t
{
    Path1 = 1,
    Path2 = 2,
    Path3 = 3,
};

struct GifArbiterPacket
{
    GifPathId pathId;
    bool path2DirectHl = false;
    bool path3Image = false;
    std::vector<uint8_t> data;
};

class GifArbiter
{
public:
    using ProcessPacketFn = std::function<void(const uint8_t *, uint32_t)>;
    // Same, plus the GIF path the packet arrived on (takes precedence when set).
    using ProcessPathPacketFn = std::function<void(const uint8_t *, uint32_t, GifPathId)>;

    GifArbiter() = default;
    explicit GifArbiter(ProcessPacketFn processFn);

    void setProcessPacketFn(ProcessPacketFn fn) { m_processFn = std::move(fn); }
    void setProcessPathPacketFn(ProcessPathPacketFn fn) { m_processPathFn = std::move(fn); }

    void submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl = false);
    // A pooled packet buffer to fill, and a filled one queued as it is (no copy): VU1 kicks are
    // copied out of VU memory once, into the buffer the GS thread will read.
    std::vector<uint8_t> takeBuffer();
    void submitOwned(GifPathId pathId, std::vector<uint8_t> &&data);

    void drain();
    // drain() in two halves: order the queued packets and move them (appended) to `out`, then hand
    // them to the GS later, possibly from another thread.
    void drainInto(std::vector<GifArbiterPacket> &out);
    void process(const std::vector<GifArbiterPacket> &packets) const;
    // Processed packets' buffers back for reuse (any thread): submit() takes from them instead of
    // allocating a buffer per packet.
    void recycle(std::vector<GifArbiterPacket> &&packets);
    // A recycled (empty) batch list for `out` when it has no room: a list handed to the GS thread
    // left its owner empty, and growing it again each batch (1, 2, 4 .. 32, moving every packet)
    // cost the VU1 thread ~9% on the Thor.
    void takeBatch(std::vector<GifArbiterPacket> &out);
    bool empty() const { return m_queue.empty(); }

private:
    ProcessPacketFn m_processFn;
    ProcessPathPacketFn m_processPathFn;
    std::vector<GifArbiterPacket> m_queue;
    std::vector<std::vector<uint8_t>> m_free;     // submit()'s own
    std::mutex m_poolMutex;
    std::vector<std::vector<uint8_t>> m_returned; // from recycle(), under m_poolMutex
    std::vector<std::vector<GifArbiterPacket>> m_spareBatches; // emptied lists, under m_poolMutex

    void sortQueue();
    static bool isImagePacket(const uint8_t *data, uint32_t sizeBytes);
    static uint8_t pathPriority(GifPathId id);
};

#endif
