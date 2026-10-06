// Save states, spike A: how often could a snapshot be taken at the top of EeScheduler::run()?
//
// canSnapshot() says what stands in the way at a loop top: live host continuations (by the kind
// their creation site gave them), a half-done GIF/GS path, an open memory card file or a playing
// movie. RT_SNAPSHOT_PROBE=1 checks it at every vblank's first loop top (with the host state) and
// whether any later loop top before the next vblank would do, and prints a summary every 1800
// vblanks and when the scheduler stops. RT_SNAPSHOT_PROBE=<file> (%p = pid) also writes one CSV
// row per vblank there.

#include "runtime/ee_scheduler.h"

#include "Stubs/MPEG.h"
#include "Stubs/MemoryCard.h"
#include "ps2_runtime.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gif_arbiter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

const char *eeContinuationKindName(EeContinuationKind kind) noexcept
{
    switch (kind)
    {
    case EeContinuationKind::None: return "none";
    case EeContinuationKind::VSyncResumePc: return "vsync_resume_pc";
    case EeContinuationKind::CdStreamRead: return "cd_stream_read";
    case EeContinuationKind::MpegPictureWait: return "mpeg_picture_wait";
    case EeContinuationKind::MpegPictureVSync: return "mpeg_picture_vsync";
    case EeContinuationKind::IpuInitDone: return "ipu_init_done";
    case EeContinuationKind::SifCommandFree: return "sif_command_free";
    case EeContinuationKind::RpcServerReturn: return "rpc_server_return";
    case EeContinuationKind::RpcEndCallback: return "rpc_end_callback";
    case EeContinuationKind::SyscallOverrideResult: return "syscall_override_result";
    case EeContinuationKind::ExitHandlerChain: return "exit_handler_chain";
    case EeContinuationKind::MpegCallbackFree: return "mpeg_callback_free";
    case EeContinuationKind::Count: break;
    }
    return "?";
}

namespace
{
    std::array<std::atomic<uint64_t>, static_cast<size_t>(EeContinuationKind::Count)> g_continuationsCreated{};
}

void eeNoteContinuationCreated(EeContinuationKind kind) noexcept
{
    if (kind < EeContinuationKind::Count)
        g_continuationsCreated[static_cast<size_t>(kind)].fetch_add(1u, std::memory_order_relaxed);
}

uint64_t eeContinuationsCreated(EeContinuationKind kind) noexcept
{
    return kind < EeContinuationKind::Count ? g_continuationsCreated[static_cast<size_t>(kind)].load(std::memory_order_relaxed) : 0u;
}

namespace
{
    constexpr std::array<const char *, 8> kBlockerNames = {
        "continuation", "path3_fifo", "dma_pending", "vif1_path2_image",
        "gif_arbiter", "gs_transfer", "memory_card", "mpeg"};
    constexpr size_t kKindCount = static_cast<size_t>(EeContinuationKind::Count);
    // Full checks (they wait for the GIF/VIF1 worker) per vblank interval after the vblank's own.
    constexpr uint32_t kMaxExtraFullChecks = 64u;
    constexpr uint64_t kSummaryEvery = 1800u;

    template <class Continuation>
    void noteContinuation(const Continuation &continuation, EeSnapshotCheck &check, uint16_t &counter)
    {
        if (!continuation)
            return;
        ++counter;
        check.continuationKinds |= 1u << static_cast<uint32_t>(continuation.kind());
        check.blockers |= kSnapshotContinuation;
    }

    std::string describe(uint32_t blockers, uint32_t kinds)
    {
        std::string out;
        for (size_t bit = 0; bit < kBlockerNames.size(); ++bit)
        {
            if ((blockers & (1u << bit)) == 0u)
                continue;
            if (bit == 0u)
            {
                for (size_t kind = 0; kind < kKindCount; ++kind)
                    if ((kinds & (1u << kind)) != 0u)
                        out += std::string(out.empty() ? "" : "|") + "cont:" +
                               eeContinuationKindName(static_cast<EeContinuationKind>(kind));
                continue;
            }
            out += std::string(out.empty() ? "" : "|") + kBlockerNames[bit];
        }
        return out;
    }
}

EeSnapshotCheck EeScheduler::canSnapshot(bool hostState)
{
    assertExecutor();
    EeSnapshotCheck check{};
    for (const auto &[id, item] : m_threads)
    {
        (void)id;
        noteContinuation(item.wait.completion, check, check.waitContinuations);
        noteContinuation(item.resumeCompletion, check, check.resumeContinuations);
        for (const GuestInvocation &invocation : item.invocations)
            noteContinuation(invocation.onComplete, check, check.invocationContinuations);
    }
    for (const GuestInvocation &invocation : m_pendingInvocations)
        noteContinuation(invocation.onComplete, check, check.pendingInvocationContinuations);

    if (!hostState)
        return check;

    PS2Memory &memory = m_runtime.memory();
    memory.syncGifVif1();
    const PS2Memory::GifPathPending gif = memory.gifPathPending();
    if (gif.path3MaskedFifo)
        check.blockers |= kSnapshotPath3Fifo;
    if (gif.dmaTransfers)
        check.blockers |= kSnapshotDmaPending;
    if (gif.vif1Path2Image)
        check.blockers |= kSnapshotVif1Path2Image;
    if (!m_runtime.gifArbiter().empty())
        check.blockers |= kSnapshotGifArbiter;
    if (m_runtime.gsUnsynced().transferPending())
        check.blockers |= kSnapshotGsTransfer;
    if (!ps2_stubs::getMemoryCardDebugSnapshot().openFiles.empty())
        check.blockers |= kSnapshotMemoryCard;
    if (ps2_stubs::mpegPlaybackActive())
        check.blockers |= kSnapshotMpeg;
    return check;
}

struct EeScheduler::SnapshotProbe
{
    FILE *csv = nullptr;
    bool started = false;
    uint64_t tick = 0;           // the vblank whose interval is open
    EeSnapshotCheck atVblank{};  // its first loop top
    bool anyOk = false;          // some loop top in the interval
    uint32_t loopTops = 0;
    uint32_t extraFullChecks = 0;

    uint64_t vblanks = 0;
    uint64_t okAtVblank = 0;
    uint64_t okAny = 0;
    std::array<uint64_t, kBlockerNames.size()> blockerVblanks{};
    std::array<uint64_t, kKindCount> kindVblanks{};
    uint64_t streakAt = 0, longestAt = 0, longestAtEnd = 0; // consecutive vblanks not savable at the vblank
    uint64_t streakAny = 0, longestAny = 0, longestAnyEnd = 0;
    std::array<uint64_t, 6> waitHistogram{}; // vblanks until savable at a vblank: 0, 1, 2-3, 4-15, 16-59, 60+
    // Every loop top (continuations only): how many, how many held one, and of which kind.
    uint64_t allLoopTops = 0;
    uint64_t loopTopsWithContinuation = 0;
    std::array<uint64_t, kKindCount> kindLoopTops{};
    // RT_SNAPSHOT_PROBE_ALL=1: the full check at every loop top, blockers counted per loop top.
    bool fullEveryLoopTop = false;
    uint64_t loopTopsBlocked = 0;
    std::array<uint64_t, kBlockerNames.size()> blockerLoopTops{};

    void noteLoopTop(const EeSnapshotCheck &check)
    {
        ++allLoopTops;
        if (!check.ok())
            ++loopTopsBlocked;
        for (size_t bit = 0; bit < kBlockerNames.size(); ++bit)
            if ((check.blockers & (1u << bit)) != 0u)
                ++blockerLoopTops[bit];
        if (check.continuationKinds == 0u)
            return;
        ++loopTopsWithContinuation;
        for (size_t kind = 0; kind < kKindCount; ++kind)
            if ((check.continuationKinds & (1u << kind)) != 0u)
                ++kindLoopTops[kind];
    }
};

void EeScheduler::SnapshotProbeDeleter::operator()(SnapshotProbe *probe) const noexcept
{
    if (probe && probe->csv)
        std::fclose(probe->csv);
    delete probe;
}

void EeScheduler::snapshotProbeStart()
{
    const char *env = std::getenv("RT_SNAPSHOT_PROBE");
    if (!env || !*env || std::strcmp(env, "0") == 0)
    {
        m_snapshotProbe.reset();
        return;
    }
    m_snapshotProbe.reset(new SnapshotProbe());
    if (const char *all = std::getenv("RT_SNAPSHOT_PROBE_ALL"); all && *all == '1')
        m_snapshotProbe->fullEveryLoopTop = true;
    if (std::strcmp(env, "1") != 0)
    {
        std::string path(env);
        if (const size_t at = path.find("%p"); at != std::string::npos)
            path.replace(at, 2, std::to_string(static_cast<long>(getpid())));
        m_snapshotProbe->csv = std::fopen(path.c_str(), "w");
        if (m_snapshotProbe->csv)
            std::fprintf(m_snapshotProbe->csv,
                         "vblank,ok_at_vblank,blockers,kinds,wait,resume,invocation,pending,ok_any,loop_tops,why\n");
        else
            std::fprintf(stderr, "[snapshot-probe] can't write %s\n", path.c_str());
    }
    std::fprintf(stderr, "[snapshot-probe] on\n");
}

namespace
{
    void bucketWait(std::array<uint64_t, 6> &histogram, uint64_t streak)
    {
        const size_t bucket = streak == 0u ? 0u : streak == 1u ? 1u : streak <= 3u ? 2u : streak <= 15u ? 3u : streak <= 59u ? 4u : 5u;
        ++histogram[bucket];
    }
}

static void printSnapshotProbeSummary(const char *when, uint64_t tick, uint64_t vblanks, uint64_t okAtVblank, uint64_t okAny,
                                      const uint64_t *blockerVblanks, const uint64_t *kindVblanks,
                                      uint64_t longestAt, uint64_t longestAtEnd, uint64_t longestAny, uint64_t longestAnyEnd,
                                      const std::array<uint64_t, 6> &waits)
{
    if (vblanks == 0u)
        return;
    std::string reasons;
    for (size_t bit = 1; bit < kBlockerNames.size(); ++bit)
        if (blockerVblanks[bit] != 0u)
            reasons += " " + std::string(kBlockerNames[bit]) + "=" + std::to_string(blockerVblanks[bit]);
    for (size_t kind = 0; kind < kKindCount; ++kind)
        if (kindVblanks[kind] != 0u)
            reasons += " cont:" + std::string(eeContinuationKindName(static_cast<EeContinuationKind>(kind))) + "=" +
                       std::to_string(kindVblanks[kind]);
    std::fprintf(stderr,
                 "[snapshot-probe] %s vblank=%llu: %llu vblanks, savable at the vblank %.1f%%, within it %.1f%%; "
                 "longest unsavable run %llu (ending %llu), within %llu (ending %llu); "
                 "waits 0:%llu 1:%llu 2-3:%llu 4-15:%llu 16-59:%llu 60+:%llu; blocked by%s\n",
                 when, static_cast<unsigned long long>(tick), static_cast<unsigned long long>(vblanks),
                 100.0 * static_cast<double>(okAtVblank) / static_cast<double>(vblanks),
                 100.0 * static_cast<double>(okAny) / static_cast<double>(vblanks),
                 static_cast<unsigned long long>(longestAt), static_cast<unsigned long long>(longestAtEnd),
                 static_cast<unsigned long long>(longestAny), static_cast<unsigned long long>(longestAnyEnd),
                 static_cast<unsigned long long>(waits[0]), static_cast<unsigned long long>(waits[1]),
                 static_cast<unsigned long long>(waits[2]), static_cast<unsigned long long>(waits[3]),
                 static_cast<unsigned long long>(waits[4]), static_cast<unsigned long long>(waits[5]),
                 reasons.empty() ? " nothing" : reasons.c_str());
}

namespace
{
    // Closes the open vblank interval into the totals.
    template <class Probe>
    void closeInterval(Probe &p)
    {
        const EeSnapshotCheck &c = p.atVblank;
        ++p.vblanks;
        if (c.ok())
        {
            ++p.okAtVblank;
            bucketWait(p.waitHistogram, p.streakAt);
            p.streakAt = 0;
        }
        else
        {
            ++p.streakAt;
            if (p.streakAt > p.longestAt)
            {
                p.longestAt = p.streakAt;
                p.longestAtEnd = p.tick;
            }
        }
        if (p.anyOk)
        {
            ++p.okAny;
            p.streakAny = 0;
        }
        else if (++p.streakAny > p.longestAny)
        {
            p.longestAny = p.streakAny;
            p.longestAnyEnd = p.tick;
        }
        for (size_t bit = 0; bit < kBlockerNames.size(); ++bit)
            if ((c.blockers & (1u << bit)) != 0u)
                ++p.blockerVblanks[bit];
        for (size_t kind = 0; kind < kKindCount; ++kind)
            if ((c.continuationKinds & (1u << kind)) != 0u)
                ++p.kindVblanks[kind];
        if (p.csv)
            std::fprintf(p.csv, "%llu,%d,0x%x,0x%x,%u,%u,%u,%u,%d,%u,%s\n",
                         static_cast<unsigned long long>(p.tick), c.ok() ? 1 : 0, c.blockers, c.continuationKinds,
                         c.waitContinuations, c.resumeContinuations, c.invocationContinuations,
                         c.pendingInvocationContinuations, p.anyOk ? 1 : 0, p.loopTops,
                         describe(c.blockers, c.continuationKinds).c_str());
    }
}

namespace
{
    template <class Probe>
    void printLoopTops(const Probe &p)
    {
        std::string kinds;
        for (size_t kind = 0; kind < kKindCount; ++kind)
            if (p.kindLoopTops[kind] != 0u)
                kinds += " " + std::string(eeContinuationKindName(static_cast<EeContinuationKind>(kind))) + "=" +
                         std::to_string(p.kindLoopTops[kind]);
        std::string created;
        for (size_t kind = 1; kind < kKindCount; ++kind)
            if (const uint64_t n = eeContinuationsCreated(static_cast<EeContinuationKind>(kind)))
                created += " " + std::string(eeContinuationKindName(static_cast<EeContinuationKind>(kind))) + "=" +
                           std::to_string(n);
        std::string blocked;
        for (size_t bit = 0; bit < kBlockerNames.size(); ++bit)
            if (p.blockerLoopTops[bit] != 0u)
                blocked += " " + std::string(kBlockerNames[bit]) + "=" + std::to_string(p.blockerLoopTops[bit]);
        std::fprintf(stderr,
                     "[snapshot-probe] loop tops %llu (%s), blocked %llu:%s; with a continuation %llu:%s; "
                     "continuations made:%s\n",
                     static_cast<unsigned long long>(p.allLoopTops),
                     p.fullEveryLoopTop ? "all checked in full" : "continuations only after the vblank's",
                     static_cast<unsigned long long>(p.loopTopsBlocked), blocked.empty() ? " none" : blocked.c_str(),
                     static_cast<unsigned long long>(p.loopTopsWithContinuation),
                     kinds.empty() ? " none" : kinds.c_str(), created.empty() ? " none" : created.c_str());
    }
}

void EeScheduler::snapshotProbeAtLoopTop()
{
    SnapshotProbe &p = *m_snapshotProbe;
    if (!p.started || m_vsyncTick != p.tick)
    {
        if (p.started)
        {
            closeInterval(p);
            if (p.vblanks % kSummaryEvery == 0u)
            {
                printSnapshotProbeSummary("so far", p.tick, p.vblanks, p.okAtVblank, p.okAny, p.blockerVblanks.data(),
                                          p.kindVblanks.data(), p.longestAt, p.longestAtEnd, p.longestAny,
                                          p.longestAnyEnd, p.waitHistogram);
                printLoopTops(p);
                if (p.csv)
                    std::fflush(p.csv);
            }
        }
        p.started = true;
        p.tick = m_vsyncTick;
        p.atVblank = canSnapshot(true);
        p.noteLoopTop(p.atVblank);
        p.anyOk = p.atVblank.ok();
        p.loopTops = 1;
        p.extraFullChecks = 0;
        return;
    }
    ++p.loopTops;
    const EeSnapshotCheck quick = canSnapshot(p.fullEveryLoopTop);
    p.noteLoopTop(quick);
    if (p.anyOk || p.extraFullChecks >= kMaxExtraFullChecks || !quick.ok())
        return;
    ++p.extraFullChecks;
    p.anyOk = canSnapshot(true).ok();
}

void EeScheduler::snapshotProbeFinish()
{
    SnapshotProbe &p = *m_snapshotProbe;
    if (p.started)
    {
        closeInterval(p);
        p.started = false;
    }
    printSnapshotProbeSummary("final", p.tick, p.vblanks, p.okAtVblank, p.okAny, p.blockerVblanks.data(),
                              p.kindVblanks.data(), p.longestAt, p.longestAtEnd, p.longestAny, p.longestAnyEnd,
                              p.waitHistogram);
    printLoopTops(p);
    if (p.csv)
        std::fflush(p.csv);
}
