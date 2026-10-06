// Save states: the whole emulated machine copied to and from an in-memory slot or a file at a
// scheduler loop top. See runtime/ps2_save_state.h (SaveStateFile.cpp has the file format).

#include "runtime/ps2_save_state.h"

#include "runtime/ee_scheduler.h"
#include "runtime/ps2_guest_clock.h"
#include "runtime/ps2_test_harness.h"
#include "runtime/gs/gs_frontend.h"
#include "ps2_runtime.h"
#include "ps2x/iop/iop_subsystem.h"
#include "ps2x/state_archive.h"
#include "Stubs/StubState.h"
#include "../gs/gs_state_io.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

using ps2x::StateArchive;
using ps2x::fourcc;

namespace
{
    constexpr auto kHostDeadlineUnit = std::chrono::nanoseconds(1);

    // The EE register file, field by field (128-bit GPRs and VU0 vectors as their 16 guest bytes).
    void serializeContext(StateArchive &ar, R5900Context &c)
    {
        for (auto &r : c.r)
            ar.bytes(&r, 16);
        ar & c.pc & c.insn_count & c.hi & c.lo & c.hi1 & c.lo1 & c.sa;
        for (auto &v : c.vu0_vf)
            ar.bytes(&v, 16);
        ar & c.vi & c.vu0_q & c.vu0_p & c.vu0_i;
        ar.bytes(&c.vu0_r, 16);
        ar.bytes(&c.vu0_acc, 16);
        ar & c.vu0_status & c.vu0_mac_flags & c.vu0_clip_flags & c.vu0_clip_flags2;
        ar & c.vu0_cmsar0 & c.vu0_cmsar1 & c.vu0_cmsar2 & c.vu0_cmsar3;
        ar & c.vu0_vpu_stat & c.vu0_vpu_stat2 & c.vu0_vpu_stat3 & c.vu0_vpu_stat4 & c.vu0_tpc & c.vu0_tpc2;
        ar & c.vu0_fbrst & c.vu0_fbrst2 & c.vu0_fbrst3 & c.vu0_fbrst4 & c.vu0_itop & c.vu0_top & c.vu0_info & c.vu0_xitop & c.vu0_pc;
        ar & c.vu0_cf;
        ar & c.cop0_index & c.cop0_random & c.cop0_entrylo0 & c.cop0_entrylo1 & c.cop0_context & c.cop0_pagemask;
        ar & c.cop0_wired & c.cop0_badvaddr & c.cop0_count & c.cop0_entryhi & c.cop0_compare & c.cop0_status;
        ar & c.cop0_cause & c.cop0_epc & c.cop0_prid & c.cop0_config & c.cop0_badpaddr & c.cop0_debug & c.cop0_perf;
        ar & c.cop0_taglo & c.cop0_taghi & c.cop0_errorepc;
        ar & c.llbit & c.lladdr & c.in_delay_slot & c.branch_pc & c.cop2_ccr & c.f & c.f_acc & c.fcr31;
    }

    void serializeVu(StateArchive &ar, VU1Interpreter &vu)
    {
        VU1State &v = vu.state();
        ar & v.vf & v.vi & v.acc & v.q & v.p & v.i & v.r & v.pc & v.mac & v.clip & v.status;
        ar & v.ebit & v.haltAfterDelaySlot & v.dBitEnabled & v.tBitEnabled & v.stoppedByD & v.stoppedByT;
        ar & v.top & v.itop & v.branchPending & v.branchTarget & v.branchDelay;
        // The interpreter's free-running cycle counter (between programs; restoring it empties
        // the pipelines, which are empty between programs anyway).
        uint64_t cycle = vu.cycle();
        ar & cycle;
        if (ar.loading())
            vu.restoreCycle(cycle);
    }

    template <class F>
    void requireEmpty(StateArchive &ar, bool hashOnly, const F &continuation, const char *what)
    {
        bool live = static_cast<bool>(continuation);
        if (ar.saving())
        {
            if (live && !hashOnly)
                ar.fail(std::string("live host continuation: ") + what + " (" +
                        eeContinuationKindName(continuation.kind()) + ")");
        }
        uint8_t flag = live ? 1u : 0u;
        ar & flag;
        if (ar.loading() && flag)
            ar.fail(std::string("state holds a host continuation: ") + what);
    }

    void serializeWait(StateArchive &ar, EeWaitState &wait, bool hashOnly)
    {
        ar & wait.reason;
        uint8_t index = static_cast<uint8_t>(wait.payload.index());
        ar & index;
        switch (index)
        {
        case 0:
            if (ar.loading())
                wait.payload = std::monostate{};
            break;
        case 1:
        {
            EeSemaphoreWait w = ar.saving() ? std::get<EeSemaphoreWait>(wait.payload) : EeSemaphoreWait{};
            ar & w.id;
            if (ar.loading())
                wait.payload = w;
            break;
        }
        case 2:
        {
            EeEventFlagWait w = ar.saving() ? std::get<EeEventFlagWait>(wait.payload) : EeEventFlagWait{};
            ar & w.id & w.bits & w.mode & w.resultAddress;
            if (ar.loading())
                wait.payload = w;
            break;
        }
        case 3:
        {
            EeVSyncWait w = ar.saving() ? std::get<EeVSyncWait>(wait.payload) : EeVSyncWait{};
            ar & w.afterTick & w.fixedResult;
            if (ar.loading())
                wait.payload = w;
            break;
        }
        case 4:
        {
            EeExternalWait w = ar.saving() ? std::get<EeExternalWait>(wait.payload) : EeExternalWait{};
            ar & w.type & w.token;
            if (ar.loading())
                wait.payload = w;
            break;
        }
        default:
            ar.fail("bad wait payload");
        }
        requireEmpty(ar, hashOnly, wait.completion, "wait completion");
        if (ar.loading())
            wait.completion = {};
    }

    void serializeInvocation(StateArchive &ar, GuestInvocation &inv, bool hashOnly)
    {
        ar & inv.kind;
        ar & inv.sequence;
        ar & inv.tag;
        serializeContext(ar, inv.context);
        requireEmpty(ar, hashOnly, inv.onComplete, "invocation completion");
        if (ar.loading())
            inv.onComplete = {};
    }

    void serializeThread(StateArchive &ar, GuestThread &t, bool hashOnly)
    {
        ar & t.id;
        serializeContext(ar, t.context);
        ar & t.entry & t.stack & t.stackSize & t.gp & t.attr & t.option & t.arg;
        ar & t.initialPriority & t.currentPriority & t.status & t.suspendCount & t.wakeupCount;
        ar & t.ownsStack & t.tlsBase;
        serializeWait(ar, t.wait, hashOnly);
        requireEmpty(ar, hashOnly, t.resumeCompletion, "resume completion");
        if (ar.loading())
            t.resumeCompletion = {};
        ar.sequence(t.invocations, [hashOnly](StateArchive &a, GuestInvocation &inv) { serializeInvocation(a, inv, hashOnly); });
    }

    // Field by field: events are built on the stack, so their padding is noise.
    void serializeEvent(StateArchive &ar, EeEvent &e) { ar & e.type & e.id & e.value; }

    std::chrono::nanoseconds eeCyclesToNs(uint64_t cycles)
    {
        return std::chrono::nanoseconds(static_cast<int64_t>(static_cast<double>(cycles) * 1e9 / EeScheduler::kEeClockHz));
    }
}

void EeScheduler::serializeState(StateArchive &ar, bool hashOnly)
{
    const auto intKey = [](StateArchive &a, int &k) { a & k; };
    for (auto &queue : m_readyQueues)
        ar.podDeque(queue);
    // Key order: nothing in the scheduler depends on its maps' iteration order.
    ar.keyOrderedMap(m_threads, intKey, [hashOnly](StateArchive &a, GuestThread &t) { serializeThread(a, t, hashOnly); });
    ar.keyOrderedMap(m_semaphores, intKey, [](StateArchive &a, EeSemaphore &s)
                    {
                        a & s.id & s.count & s.maxCount & s.initCount & s.attr & s.option;
                        a.podDeque(s.waiters);
                    });
    ar.keyOrderedMap(m_eventFlags, intKey, [](StateArchive &a, EeEventFlag &f)
                    {
                        a & f.id & f.attr & f.option & f.initBits & f.bits;
                        a.podDeque(f.waiters);
                    });
    ar.keyOrderedMap(m_alarms, intKey, [](StateArchive &a, EeAlarm &v) { a & v.id & v.ticks & v.handler & v.argument & v.gp & v.sp; });
    const auto handler = [](StateArchive &a, EeIrqHandler &v)
    { a & v.id & v.cause & v.handler & v.argument & v.gp & v.sp & v.enabled & v.order; };
    ar.keyOrderedMap(m_intcHandlers, intKey, handler);
    ar.keyOrderedMap(m_dmacHandlers, intKey, handler);
    ar & m_nextThreadId & m_nextInvocationThreadId & m_nextSemaphoreId & m_nextEventFlagId & m_nextAlarmId;
    ar & m_nextIntcHandlerId & m_nextDmacHandlerId & m_intcHeadOrder & m_intcTailOrder & m_dmacHeadOrder & m_dmacTailOrder;
    ar & m_enabledIntcMask & m_enabledDmacMask;
    ar & m_currentThreadId & m_rescheduleRequested & m_timeSliceExpired & m_insideInterrupt;
    ar & m_pendingEeTimerInterrupts & m_eeCycle & m_sliceEndCycle;
    ar.property<bool>([this] { return m_checkpointPending.load(); }, [this](bool v) { m_checkpointPending.store(v); });
    ar & m_debugPublishCountdown;
    {
        std::lock_guard lock(m_eventMutex);
        ar.sequence(m_events, [](StateArchive &a, EeEvent &e) { serializeEvent(a, e); });
        // (Their host deadlines are in EETM.)
        const auto now = std::chrono::steady_clock::now();
        ar.sequence(m_deadlines, [&](StateArchive &a, ScheduledEvent &e)
                    {
                        a & e.deadlineCycle;
                        serializeEvent(a, e.event);
                        a & e.sequence;
                        if (a.loading())
                            e.hostDeadline = now;
                    });
        ar.sequence(m_pendingInvocations, [hashOnly](StateArchive &a, GuestInvocation &inv) { serializeInvocation(a, inv, hashOnly); });
        if (ar.loading())
            updateNextDeadline();
    }
    ar & m_eventSequence & m_invocationSequence & m_vsyncTick & m_vsyncFlagAddress & m_vsyncTickAddress;
    ar & m_gsVSyncCallback & m_gsVSyncCallbackGp & m_gsVSyncCallbackSp;
    ar.keyOrderedMap(m_invocationStackTops, [](StateArchive &a, uint64_t &k) { a & k; }, [](StateArchive &a, uint32_t &v) { a & v; });
    if (ar.loading())
    {
        // Paced virtual time: the guest clock continues from here.
        if (m_virtualTime && m_virtualSpeed > 0.0)
            m_virtualEpoch = std::chrono::steady_clock::now() -
                             std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                 std::chrono::duration<double, std::nano>(static_cast<double>(eeCyclesToNs(m_eeCycle).count()) / m_virtualSpeed));
        ps2_guest_clock::g_vblanks.store(m_vsyncTick, std::memory_order_relaxed);
        m_runtime.memory().gs().vsyncTick.store(m_vsyncTick, std::memory_order_release);
        copyMainContextToRuntime();
        publishSnapshot();
    }
}

void EeScheduler::serializeHostTiming(StateArchive &ar, bool hashOnly)
{
    std::lock_guard lock(m_eventMutex);
    size_t n = m_deadlines.size();
    ar.size(n);
    if (ar.loading() && n != m_deadlines.size())
    {
        ar.fail("scheduled events and their deadlines differ");
        return;
    }
    if (hashOnly)
        return; // wall-clock: not in a digest
    const auto now = std::chrono::steady_clock::now();
    for (ScheduledEvent &e : m_deadlines)
    {
        int64_t rel = ar.saving() ? std::chrono::duration_cast<std::chrono::nanoseconds>(e.hostDeadline - now).count() : 0;
        ar & rel;
        if (ar.loading())
            e.hostDeadline = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(rel * kHostDeadlineUnit);
    }
    if (ar.loading())
        updateNextDeadline();
}

std::vector<std::string> EeScheduler::invalidResumePcs(const std::function<bool(uint32_t)> &valid) const
{
    std::vector<std::string> bad;
    const auto check = [&](const std::string &what, const R5900Context &c)
    {
        if (!valid(c.pc))
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "%s pc 0x%X", what.c_str(), c.pc);
            bad.emplace_back(buf);
        }
    };
    std::vector<int> ids;
    for (const auto &[id, t] : m_threads)
        ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    for (const int id : ids)
    {
        const GuestThread &t = m_threads.at(id);
        if (t.status == EeThreadStatus::Dormant && t.invocations.empty())
            continue;
        const std::string name = "thread " + std::to_string(id);
        if (t.status != EeThreadStatus::Dormant)
            check(name, t.context);
        for (size_t i = 0; i < t.invocations.size(); ++i)
            check(name + " invocation " + std::to_string(i), t.invocations[i].context);
    }
    for (const GuestInvocation &inv : m_pendingInvocations)
        check("pending invocation", inv.context);
    return bad;
}

void PS2Memory::serializeState(StateArchive &ar)
{
    if (ar.saving())
    {
        // Not expected at a loop top after syncGifVif1() (canSnapshot refuses otherwise).
        const GifPathPending pending = gifPathPending();
        if (pending.path3MaskedFifo || pending.dmaTransfers || pending.vif1Path2Image)
            ar.fail("GIF/VIF work pending");
    }
    ar.keyOrderedMap(m_ioRegisters, [](StateArchive &a, uint32_t &k) { a & k; }, [](StateArchive &a, uint32_t &v) { a & v; });
    ar & gs_regs.pmode & gs_regs.smode1 & gs_regs.smode2 & gs_regs.srfsh & gs_regs.synch1 & gs_regs.synch2;
    ar & gs_regs.syncv & gs_regs.dispfb1 & gs_regs.display1 & gs_regs.dispfb2 & gs_regs.display2;
    ar & gs_regs.extbuf & gs_regs.extdata & gs_regs.extwrite & gs_regs.bgcolor;
    ar.property<uint64_t>([this] { return gs_regs.csr.load(); }, [this](uint64_t v) { gs_regs.csr.store(v); });
    ar.property<uint64_t>([this] { return gs_regs.vsyncTick.load(); }, [this](uint64_t v) { gs_regs.vsyncTick.store(v); });
    ar & gs_regs.imr & gs_regs.busdir & gs_regs.siglblid;
    for (VIFRegisters *v : {&vif0_regs, &vif1_regs})
        ar & v->stat & v->fbrst & v->err & v->mark & v->cycle & v->mode & v->num & v->mask & v->code & v->itops & v->base
           & v->ofst & v->tops & v->itop & v->top & v->row & v->col;
    for (DMARegisters &d : dma_regs)
        ar & d.chcr & d.madr & d.qwc & d.tadr & d.asr0 & d.asr1 & d.sadr;
    ar.sequence(m_tlbEntries, [](StateArchive &a, TLBEntry &e) { a & e.vpn & e.pfn & e.mask & e.valid; });
    ar & m_path3Masked & m_vif1PendingPath2ImageQwc & m_vif1PendingPath2DirectHl;
    {
        std::lock_guard<std::mutex> lock(m_completedDmacMutex);
        ar.podVector(m_completedDmacCauses);
    }
    for (EeTimer &t : m_eeTimers)
        ar & t.count & t.mode & t.compare & t.hold & t.clockRemainder;
    ar & m_seenGifCopy & m_lastVblankDispfb;
    if (ar.loading())
    {
        markVU0CodeModified();
        markVU1CodeModified();
    }
}

bool GS::serializeState(StateArchive &ar)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    using ps2x::gs_state::io;
    for (GSContext &c : m_ctx)
        io(ar, c);
    io(ar, m_prim);
    io(ar, m_primRegister);
    io(ar, m_prmodeRegister);
    ar & m_curR & m_curG & m_curB & m_curA & m_curQ & m_curS & m_curT & m_curU & m_curV & m_curFog;
    ar & m_fogR & m_fogG & m_fogB & m_prmodecont & m_pabe & m_scanmsk & m_dimx & m_dthe & m_colclamp;
    io(ar, m_texa);
    io(ar, m_texclut);
    io(ar, m_bitbltbuf);
    io(ar, m_trxpos);
    io(ar, m_trxreg);
    ar & m_trxdir;
    ar.property<uint64_t>([this] { return m_hostTransferBitsLeft.load(); }, [this](uint64_t v) { m_hostTransferBitsLeft.store(v); });
    for (GSVertex &v : m_vtxQueue)
        io(ar, v);
    ar & m_vtxCount & m_vtxIndex & m_packetKicks & m_curPath & m_motionId;
    ar & m_lastDisplayBaseBytes;
    io(ar, m_preferredDisplaySourceFrame);
    ar & m_preferredDisplayDestFbp & m_hasPreferredDisplaySource;
    if (ar.loading())
        ++m_drawStateSerial; // the next draw rebuilds its batch state
    std::lock_guard<std::mutex> backendLock(m_backendLifetimeMutex);
    if (!m_backend || !m_backend->SerializeState(ar))
    {
        ar.fail("the GS backend can't save its state");
        return false;
    }
    // The backend's own part, apart: another backend passes over it (and rebuilds its own).
    uint32_t kind = m_backend->StateExtrasKind();
    ar & kind;
    std::vector<uint8_t> extras;
    if (ar.saving() && kind != 0u)
    {
        StateArchive x = StateArchive::writer(extras);
        x.begin(kind);
        m_backend->SerializeStateExtras(x);
        x.end();
    }
    ar.podVector(extras);
    if (ar.loading())
    {
        bool restored = false;
        if (ar.ok() && kind != 0u && kind == m_backend->StateExtrasKind())
        {
            StateArchive x = StateArchive::reader(extras);
            x.begin(kind);
            m_backend->SerializeStateExtras(x);
            x.end();
            if (!x.ok())
                ar.fail("GS backend state: " + x.error());
            restored = x.ok();
        }
        if (ar.ok())
            m_backend->StateLoaded(restored);
    }
    return ar.ok();
}

uint32_t GS::stateBackendKind()
{
    std::lock_guard<std::mutex> backendLock(m_backendLifetimeMutex);
    return m_backend ? m_backend->StateExtrasKind() : 0u;
}

void PS2Runtime::serializeState(StateArchive &ar)
{
    serializeContext(ar, m_cpuContext);
    {
        std::lock_guard<std::mutex> lock(m_eeKernelStateMutex);
        ar.keyOrderedMap(m_eeExitHandlers, [](StateArchive &a, int &k) { a & k; },
                        [](StateArchive &a, std::vector<EeExitHandlerRegistration> &v)
                        { a.sequence(v, [](StateArchive &b, EeExitHandlerRegistration &r) { b & r.function & r.argument; }); });
        const auto u32 = [](StateArchive &a, uint32_t &v) { a & v; };
        ar.keyOrderedMap(m_eeSyscallOverrides, u32, u32);
        ar.unorderedSet(m_eeSyscallMirrorAddresses, u32);
    }
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        ar.sequence(m_guestHeapBlocks, [](StateArchive &a, GuestHeapBlock &b) { a & b.addr & b.size & b.free; });
        ar & m_guestHeapBase & m_guestHeapEnd & m_guestHeapLimit & m_guestHeapSuggestedBase & m_guestHeapConfigured;
    }
    {
        std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
        ar & m_asyncCallbackStackFloor & m_asyncCallbackStackTop;
    }
    // Open guest files (VFS): reopened by path at their positions on load.
    {
        const IoPaths &paths = getIoPaths();
        m_vfs.serializeState(ar, PS2VfsMounts{paths.hostRoot, paths.cdRoot, paths.mcRoot}, m_romDevice);
    }
    if (ar.loading())
        m_vu1NativeCheckedGeneration = ~0ull;
}

namespace ps2_save_state
{
    namespace
    {
        // Every chunk this build reads and writes, with its version (bump a chunk's version when
        // its fields change; its reader can keep reading older versions when the change is
        // additive). Loading refuses states with chunks missing from here or newer than here.
        struct ChunkDef
        {
            uint32_t id;
            uint16_t version;
        };
        constexpr ChunkDef kChunks[] = {
            {fourcc("HEAD"), 1}, {fourcc("EERM"), 1}, {fourcc("SPAD"), 1}, {fourcc("EEHW"), 1},
            {fourcc("VU0M"), 1}, {fourcc("VU1M"), 1}, {fourcc("VU0S"), 1}, {fourcc("VU1S"), 1},
            {fourcc("EESC"), 1}, {fourcc("EETM"), 1}, {fourcc("EERT"), 1}, {fourcc("GSFE"), 1},
            {fourcc("HPAD"), 1}, {fourcc("HMCD"), 1}, {fourcc("HCDV"), 1}, {fourcc("HSIF"), 1},
            {fourcc("HRPC"), 1}, {fourcc("HAUD"), 1}, {fourcc("HDMA"), 1}, {fourcc("HGSS"), 1},
            {fourcc("HFIO"), 1}, {fourcc("HLIB"), 1}, {fourcc("IOPS"), 1}, {fourcc("SPU2"), 1},
        };

        struct SectionDef
        {
            uint32_t id;
            uint16_t version;
            Section fn;
        };

        // A load waiting for the EE thread.
        struct PendingLoad
        {
            bool file = false;
            std::string path;
            std::string resavePath;
            std::vector<uint8_t> bytes;
            std::vector<std::pair<std::string, uint64_t>> digest; // at the save
        };

        std::mutex g_mutex;
        Result g_result;
        uint64_t g_loopTops = 0;
        std::set<std::string> g_blockers;
        std::vector<uint8_t> g_slot;
        std::vector<StateArchive::Chunk> g_slotChunks;
        std::vector<std::pair<std::string, uint64_t>> g_slotDigest; // hashOnly chunk hashes at save
        std::vector<SectionDef> g_sections;
        std::vector<uint8_t> g_scratch;
        std::atomic<uint32_t> g_gameCrc{0};
        std::string g_savePath;   // file save requested (empty: the slot)
        bool g_saveCompressed = true;
        std::atomic<bool> g_failNextLoad{false};
        PendingLoad g_pendingLoad; // file load requested
        MetadataProvider g_metadata;

        std::mutex g_writeMutex;
        std::condition_variable g_writeCv;
        int g_writesInFlight = 0;
        WriteResult g_write;

        // At exit: let a file being written finish (its temp file would be left otherwise).
        struct WriteDrain
        {
            ~WriteDrain() { waitForWrites(15000); }
        } g_writeDrain;

        uint16_t knownVersion(uint32_t id)
        {
            for (const ChunkDef &c : kChunks)
                if (c.id == id)
                    return c.version;
            for (const SectionDef &s : g_sections)
                if (s.id == id)
                    return s.version;
            return 0;
        }

        std::set<uint32_t> skippedChunks()
        {
            std::set<uint32_t> out;
            if (const char *e = std::getenv("RT_SAVE_STATE_SKIP"))
            {
                std::stringstream in(e);
                for (std::string item; std::getline(in, item, ',');)
                    if (item.size() == 4)
                        out.insert(uint32_t(uint8_t(item[0])) | uint32_t(uint8_t(item[1])) << 8 |
                                   uint32_t(uint8_t(item[2])) << 16 | uint32_t(uint8_t(item[3])) << 24);
            }
            return out;
        }

        // The whole machine, chunk by chunk. hashOnly: a digest (no host-relative values, no
        // refusals for pending work).
        void serializeAll(PS2Runtime &rt, StateArchive &ar, bool hashOnly, const std::set<uint32_t> &skip, std::string *skipped)
        {
            PS2Memory &mem = rt.memory();
            const auto chunk = [&](uint32_t id, auto &&fn)
            {
                if (!ar.ok())
                    return;
                if (ar.loading() && skip.count(id))
                {
                    if (skipped)
                        *skipped += (skipped->empty() ? "" : ",") + ps2x::fourccName(id);
                    return;
                }
                ar.begin(id, knownVersion(id));
                fn();
                ar.end();
            };
            // States belong to the game (its ELF), not to a build of the app.
            chunk(fourcc("HEAD"), [&]
                  {
                      uint32_t magic = fourcc("RTSS"), crc = g_gameCrc.load();
                      ar & magic & crc;
                      if (ar.loading() && (magic != fourcc("RTSS") || crc != g_gameCrc.load()))
                          ar.fail("the state is for another game (ELF CRC differs)");
                  });
            chunk(fourcc("EERM"), [&] { ar.bytes(mem.getRDRAM(), PS2_RAM_SIZE); });
            chunk(fourcc("SPAD"), [&] { ar.bytes(mem.getScratchpad(), PS2_SCRATCHPAD_SIZE); });
            // (PS2Memory's legacy IOP RAM block is unused: the IOP has its own memory, chunk IOPS.)
            chunk(fourcc("EEHW"), [&] { mem.serializeState(ar); });
            chunk(fourcc("VU0M"), [&] { ar.bytes(mem.getVU0Code(), PS2_VU0_CODE_SIZE); ar.bytes(mem.getVU0Data(), PS2_VU0_DATA_SIZE); });
            chunk(fourcc("VU1M"), [&] { ar.bytes(mem.getVU1Code(), PS2_VU1_CODE_SIZE); ar.bytes(mem.getVU1Data(), PS2_VU1_DATA_SIZE); });
            chunk(fourcc("VU0S"), [&] { serializeVu(ar, rt.vu0()); });
            chunk(fourcc("VU1S"), [&]
                  {
                      if (ar.saving() && !hashOnly && !rt.vu1().lastRunEnded())
                          ar.fail("VU1 microprogram stopped mid-way");
                      serializeVu(ar, rt.vu1());
                  });
            chunk(fourcc("EESC"), [&] { rt.eeScheduler().serializeState(ar, hashOnly); });
            chunk(fourcc("EETM"), [&] { rt.eeScheduler().serializeHostTiming(ar, hashOnly); });
            chunk(fourcc("EERT"), [&] { rt.serializeState(ar); });
            chunk(fourcc("GSFE"), [&] { rt.gsUnsynced().serializeState(ar); });
            chunk(fourcc("HPAD"), [&] { ps2_stubs::serializePadState(ar); });
            chunk(fourcc("HMCD"), [&] { ps2_stubs::serializeMemoryCardState(ar); });
            chunk(fourcc("HCDV"), [&] { ps2_stubs::serializeCdState(ar); });
            chunk(fourcc("HSIF"), [&] { ps2_stubs::serializeSifState(ar); });
            chunk(fourcc("HRPC"), [&] { ps2_syscalls::serializeRpcState(ar); });
            chunk(fourcc("HAUD"), [&] { ps2_stubs::serializeAudioStubState(ar); });
            chunk(fourcc("HDMA"), [&] { ps2_stubs::serializeDmaStubState(ar); });
            chunk(fourcc("HGSS"), [&] { ps2_stubs::serializeGsStubState(ar); });
            chunk(fourcc("HFIO"), [&] { ps2_stubs::serializeFileIoStubState(ar); });
            chunk(fourcc("HLIB"), [&] { ps2_stubs::serializeLibcFileState(ar); });
            chunk(fourcc("IOPS"), [&] { rt.iopSubsystem()->serializeState(ar); });
            chunk(fourcc("SPU2"), [&] { rt.iopSubsystem()->serializeSpu2State(ar); });
            for (SectionDef &s : g_sections)
                chunk(s.id, [&] { s.fn(ar); });
        }

        std::vector<std::pair<std::string, uint64_t>> digestOf(const StateArchive &ar)
        {
            std::vector<std::pair<std::string, uint64_t>> out;
            for (const auto &c : ar.chunks())
                out.emplace_back(ps2x::fourccName(c.id), c.hash);
            return out;
        }

        std::vector<std::pair<std::string, uint64_t>> digest(PS2Runtime &rt)
        {
            StateArchive ar = StateArchive::writer(g_scratch);
            serializeAll(rt, ar, true, {}, nullptr);
            return digestOf(ar);
        }

        std::string describeBlockers(const EeSnapshotCheck &check)
        {
            static const char *names[] = {"continuation", "path3_fifo", "dma_pending", "vif1_path2_image",
                                          "gif_arbiter", "gs_transfer", "memory_card", "mpeg"};
            std::string out;
            for (int bit = 0; bit < 8; ++bit)
                if (check.blockers & (1u << bit))
                    out += std::string(out.empty() ? "" : "|") + names[bit];
            return out;
        }

        void finish(Result r)
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            r.sequence = g_result.sequence + 1;
            r.loopTops = g_loopTops;
            for (const auto &b : g_blockers)
                r.blockers += (r.blockers.empty() ? "" : ",") + b;
            g_result = std::move(r);
            g_savePath.clear();
            g_pendingLoad = PendingLoad{};
            detail::g_request.store(0, std::memory_order_relaxed);
        }

        void startRequest(Op op)
        {
            g_loopTops = 0;
            g_blockers.clear();
            detail::g_request.store(static_cast<int>(op), std::memory_order_relaxed);
        }

        const char *backendName(uint32_t kind)
        {
            if (kind == 0u)
                return "cpu";
            if (kind == fourcc("PGS1"))
                return "pgs";
            return "other";
        }

        // A worker compresses and writes a taken state.
        void startWrite(std::string path, FileInfo info, std::vector<uint8_t> bytes, bool compress)
        {
            {
                std::lock_guard<std::mutex> lock(g_writeMutex);
                ++g_writesInFlight;
            }
            std::thread([path = std::move(path), info = std::move(info), bytes = std::move(bytes), compress]() mutable
                        {
                            const auto start = std::chrono::steady_clock::now();
                            WriteResult w;
                            w.path = path;
                            w.rawBytes = bytes.size();
                            w.ok = writeStateFile(path, info, bytes, w.error, compress);
                            w.fileBytes = w.ok ? info.fileSize : 0u;
                            w.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                            std::fprintf(stderr, "[save-state] wrote %s: %s, %llu -> %llu bytes, %.1f ms\n", path.c_str(),
                                         w.ok ? "ok" : w.error.c_str(), static_cast<unsigned long long>(w.rawBytes),
                                         static_cast<unsigned long long>(w.fileBytes), w.ms);
                            std::lock_guard<std::mutex> lock(g_writeMutex);
                            w.sequence = g_write.sequence + 1;
                            g_write = std::move(w);
                            --g_writesInFlight;
                            g_writeCv.notify_all();
                        })
                .detach();
        }

        FileInfo fileInfoFor(PS2Runtime &runtime, uint64_t vblank, const StateArchive &ar,
                             const std::vector<std::pair<std::string, uint64_t>> &dig, const MetadataProvider &metadata)
        {
            FileInfo info;
            info.gameCrc = g_gameCrc.load();
            info.vblank = vblank;
            info.savedUnixTime = static_cast<int64_t>(std::time(nullptr));
            info.metadata.emplace_back("gs", backendName(runtime.gsUnsynced().stateBackendKind()));
            if (metadata)
                for (auto &kv : metadata())
                    info.metadata.push_back(std::move(kv));
            for (const auto &c : ar.chunks())
            {
                FileChunk fc;
                fc.id = c.id;
                fc.version = knownVersion(c.id);
                fc.size = c.size;
                fc.hash = c.hash;
                for (const auto &[name, h] : dig)
                    if (name == ps2x::fourccName(c.id))
                        fc.digest = h;
                info.chunks.push_back(fc);
            }
            return info;
        }

        // Checks a decompressed state against its header and this build: every chunk intact,
        // known, not newer than this build reads, none missing.
        bool validate(const FileInfo &info, const std::vector<uint8_t> &raw, std::string &error)
        {
            StateArchive ar = StateArchive::reader(raw);
            if (!ar.ok())
            {
                error = "the file is damaged (" + ar.error() + ")";
                return false;
            }
            const auto &index = ar.index();
            if (index.size() != info.chunks.size())
            {
                error = "the file is damaged (chunk table)";
                return false;
            }
            for (size_t i = 0; i < index.size(); ++i)
            {
                const auto &e = index[i];
                const FileChunk &c = info.chunks[i];
                if (e.id != c.id || e.version != c.version || e.size != c.size ||
                    ps2x::stateHash(raw.data() + e.offset, e.size) != c.hash)
                {
                    error = "the file is damaged (chunk " + ps2x::fourccName(c.id) + ")";
                    return false;
                }
            }
            std::lock_guard<std::mutex> lock(g_mutex);
            for (const auto &e : index)
            {
                const uint16_t known = knownVersion(e.id);
                if (known == 0u || e.version > known)
                {
                    error = "made by a newer version of the app (" + ps2x::fourccName(e.id) + " v" + std::to_string(e.version) + ")";
                    return false;
                }
            }
            const auto need = [&](uint32_t id) -> bool
            {
                if (ar.find(id))
                    return true;
                error = "made by an older version of the app (it has no " + ps2x::fourccName(id) + ")";
                return false;
            };
            for (const ChunkDef &c : kChunks)
                if (!need(c.id))
                    return false;
            for (const SectionDef &s : g_sections)
                if (!need(s.id))
                    return false;
            return true;
        }
    }

    void request(Op op)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_savePath.clear();
        g_pendingLoad = PendingLoad{};
        startRequest(op);
    }

    void requestSaveFile(const std::string &path, bool compress)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_pendingLoad = PendingLoad{};
        g_savePath = path;
        g_saveCompressed = compress;
        startRequest(Op::Save);
    }

    void failNextLoadForTest() { g_failNextLoad.store(true); }

    bool requestLoadFile(const std::string &path, std::string &error, const std::string &resavePath)
    {
        const auto start = std::chrono::steady_clock::now();
        FileInfo info;
        std::vector<uint8_t> raw;
        if (!readStateFile(path, info, raw, error))
        {
            std::fprintf(stderr, "[save-state] refused %s: %s\n", path.c_str(), error.c_str());
            return false;
        }
        if (info.gameCrc != g_gameCrc.load())
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "made for another game (ELF CRC %08X, this is %08X)", info.gameCrc, g_gameCrc.load());
            error = buf;
            std::fprintf(stderr, "[save-state] refused %s: %s\n", path.c_str(), error.c_str());
            return false;
        }
        if (!validate(info, raw, error))
        {
            std::fprintf(stderr, "[save-state] refused %s: %s\n", path.c_str(), error.c_str());
            return false;
        }
        std::fprintf(stderr, "[save-state] read %s: %llu -> %zu bytes, checked in %.1f ms\n", path.c_str(),
                     static_cast<unsigned long long>(info.fileSize), raw.size(),
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        PendingLoad pending;
        pending.file = true;
        pending.path = path;
        pending.resavePath = resavePath;
        pending.bytes = std::move(raw);
        for (const FileChunk &c : info.chunks)
            pending.digest.emplace_back(ps2x::fourccName(c.id), c.digest);
        std::lock_guard<std::mutex> lock(g_mutex);
        g_savePath.clear();
        g_pendingLoad = std::move(pending);
        startRequest(Op::Load);
        return true;
    }

    bool waitForWrites(int timeoutMs)
    {
        std::unique_lock<std::mutex> lock(g_writeMutex);
        return g_writeCv.wait_for(lock, std::chrono::milliseconds(timeoutMs), [] { return g_writesInFlight == 0; });
    }

    WriteResult lastWrite()
    {
        std::lock_guard<std::mutex> lock(g_writeMutex);
        return g_write;
    }

    void setGameIdentity(uint32_t elfCrc32) { g_gameCrc.store(elfCrc32); }
    uint32_t gameIdentity() { return g_gameCrc.load(); }

    void setMetadataProvider(MetadataProvider provider)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_metadata = std::move(provider);
    }

    void cancel()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_savePath.clear();
        g_pendingLoad = PendingLoad{};
        detail::g_request.store(0, std::memory_order_relaxed);
    }

    Result lastResult()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_result;
    }

    void registerSection(uint32_t id, Section section, uint16_t version)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (SectionDef &s : g_sections)
            if (s.id == id)
            {
                s = SectionDef{id, version, std::move(section)};
                return;
            }
        g_sections.push_back(SectionDef{id, version, std::move(section)});
    }

    std::vector<std::pair<std::string, uint64_t>> chunkHashes(PS2Runtime &runtime)
    {
        runtime.memory().syncGifVif1();
        return digest(runtime);
    }

    std::vector<uint8_t> chunkBytes(PS2Runtime &runtime, const std::string &name)
    {
        runtime.memory().syncGifVif1();
        StateArchive ar = StateArchive::writer(g_scratch);
        serializeAll(runtime, ar, true, {}, nullptr);
        for (const auto &c : ar.chunks())
            if (ps2x::fourccName(c.id) == name)
                return std::vector<uint8_t>(g_scratch.begin() + c.offset, g_scratch.begin() + c.offset + c.size);
        return {};
    }

    std::vector<std::pair<std::string, std::pair<size_t, uint64_t>>> slotChunks()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        std::vector<std::pair<std::string, std::pair<size_t, uint64_t>>> out;
        for (const auto &c : g_slotChunks)
            out.push_back({ps2x::fourccName(c.id), {c.size, c.hash}});
        return out;
    }

    void service(PS2Runtime &runtime)
    {
        const Op op = static_cast<Op>(detail::g_request.load(std::memory_order_relaxed));
        if (op == Op::None)
            return;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            ++g_loopTops;
        }
        EeScheduler &scheduler = runtime.eeScheduler();
        const EeSnapshotCheck check = scheduler.canSnapshot(true); // also waits for the GIF/VIF1 worker
        if (!check.ok())
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_blockers.insert(describeBlockers(check));
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        const auto ms = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); };
        Result r;
        r.op = op;
        r.vblank = scheduler.currentVSyncTick();
        if (op == Op::Save)
        {
            std::string path;
            MetadataProvider metadata;
            bool compressed = true;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                path = g_savePath;
                metadata = g_metadata;
                compressed = g_saveCompressed;
            }
            r.path = path;
            std::vector<uint8_t> bytes;
            bytes.reserve(48u << 20);
            StateArchive ar = StateArchive::writer(bytes);
            serializeAll(runtime, ar, false, {}, nullptr);
            r.ok = ar.ok();
            r.error = ar.error();
            r.bytes = bytes.size();
            if (r.ok)
            {
                auto dig = digest(runtime);
                if (!path.empty())
                    startWrite(path, fileInfoFor(runtime, r.vblank, ar, dig, metadata), std::move(bytes), compressed);
                else
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    g_slotChunks = ar.chunks();
                    g_slot = std::move(bytes);
                    g_slotDigest = std::move(dig);
                }
            }
            r.ms = ms();
            std::fprintf(stderr, "[save-state] save at vblank %llu: %s, %zu bytes, %.1f ms%s%s\n",
                         static_cast<unsigned long long>(r.vblank), r.ok ? "ok" : r.error.c_str(), r.bytes, r.ms,
                         path.empty() ? "" : ", writing ", path.c_str());
            finish(std::move(r));
            return;
        }

        PendingLoad load;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_pendingLoad.file)
                load = std::move(g_pendingLoad);
            else
            {
                load.bytes = g_slot;
                load.digest = g_slotDigest;
            }
        }
        r.path = load.path;
        if (load.bytes.empty())
        {
            r.error = "no state saved";
            finish(std::move(r));
            return;
        }

        // A copy of the machine as it is, put back if the load fails half-way.
        std::vector<uint8_t> backup;
        backup.reserve(48u << 20);
        bool haveBackup = false;
        {
            StateArchive b = StateArchive::writer(backup);
            serializeAll(runtime, b, false, {}, nullptr);
            haveBackup = b.ok();
        }

        const std::set<uint32_t> skip = skippedChunks();
        StateArchive ar = StateArchive::reader(load.bytes);
        serializeAll(runtime, ar, false, skip, &r.skipped);
        r.ok = ar.ok();
        r.error = ar.error();
        r.bytes = load.bytes.size();
        r.orderMismatches = ar.orderMismatches();
        if (r.ok)
        {
            // Every guest context must resume where this build has code (a recompiled function
            // or one of its resume points).
            const bool failForTest = g_failNextLoad.exchange(false);
            const auto bad = scheduler.invalidResumePcs([&runtime, failForTest](uint32_t pc)
                                                        { return !failForTest && runtime.hasFunction(pc); });
            if (!bad.empty())
            {
                r.ok = false;
                r.error = "this build can't resume the game there (" + bad.front() +
                          (bad.size() > 1 ? " and " + std::to_string(bad.size() - 1) + " more" : std::string()) + ")";
            }
        }
        if (!r.ok)
        {
            if (haveBackup)
            {
                StateArchive back = StateArchive::reader(backup);
                serializeAll(runtime, back, false, {}, nullptr);
                r.rolledBack = back.ok();
            }
            if (!r.rolledBack)
            {
                // Half a machine: nothing sensible can run on.
                std::fprintf(stderr, "[save-state] load failed: %s; can't put the machine back, stopping\n", r.error.c_str());
                runtime.requestStop();
            }
            else
            {
                std::fprintf(stderr, "[save-state] load refused: %s; the game runs on as it was\n", r.error.c_str());
            }
            r.vblank = scheduler.currentVSyncTick();
            r.ms = ms();
            finish(std::move(r));
            return;
        }
        r.vblank = scheduler.currentVSyncTick();
        ps2_test::onStateLoaded(r.vblank);
        // The digest right after loading matches the one taken at the save, chunk by chunk
        // (except chunks left out).
        const auto now = digest(runtime);
        r.reserializedEqual = now.size() == load.digest.size();
        for (size_t i = 0; r.reserializedEqual && i < now.size(); ++i)
        {
            if (now[i].first != load.digest[i].first)
                r.reserializedEqual = false;
            else if (now[i].second != load.digest[i].second && r.skipped.find(now[i].first) == std::string::npos)
            {
                r.reserializedEqual = false;
                r.error = "chunk " + now[i].first + " differs after loading";
            }
        }
        if (!load.resavePath.empty())
        {
            // Saved again at once: the same bytes as loaded (but EETM's wall-clock deadlines).
            std::vector<uint8_t> again;
            again.reserve(load.bytes.size());
            StateArchive w = StateArchive::writer(again);
            serializeAll(runtime, w, false, {}, nullptr);
            if (w.ok())
            {
                MetadataProvider metadata;
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    metadata = g_metadata;
                }
                startWrite(load.resavePath, fileInfoFor(runtime, r.vblank, w, now, metadata), std::move(again), true);
            }
        }
        r.ms = ms();
        std::fprintf(stderr, "[save-state] load to vblank %llu%s%s: ok, %.1f ms, digest %s%s%s, order mismatches %u\n",
                     static_cast<unsigned long long>(r.vblank), load.path.empty() ? "" : " from ", load.path.c_str(), r.ms,
                     r.reserializedEqual ? "identical" : "DIFFERS", r.skipped.empty() ? "" : ", skipped ", r.skipped.c_str(),
                     r.orderMismatches);
        finish(std::move(r));
    }
}
