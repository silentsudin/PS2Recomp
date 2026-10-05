#pragma once

// Bridge between VU1Interpreter and VU1 microcode recompiled to C++ by ps2_vu1_recomp.
//
// Lean code (the default; see "lean code" below): straight-line C++ per explored node, with the
// timing resolved at build time and only the instruction semantics (the interpreter's own
// execUpper/execLower/value helpers on constant words) run per pair.
//
// --interp-steps code: the interpreter's run loop with decode and hazard-stall computation done
// at build time: for each instruction pair it advances the cycle counter by the precomputed
// stall (committing pipelines / progressing XGKICK exactly as the interpreter would) and then
// runs VU1Interpreter::executePair() with a constexpr decoded pair, dispatching on the PC/branch
// state that executePair() produces.
//
// Both deopt to the interpreter for anything outside the statically explored model: they hand
// over the interpreter's exact state at that point and it finishes the run.

#include "runtime/vu/ps2_vu1_exec.inl"

#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

struct Vu1Native
{
    using Pair = VU1Interpreter::DecodedInstructionPair;
    using Usage = VU1Interpreter::InstructionUsage;
    using Access = VU1Interpreter::VfAccess;
    using Pipeline = VU1Interpreter::Pipeline;

    enum Result : int
    {
        NotHandled = 0, // no native code for this entry; nothing was touched
        Handled = 1,    // ran to completion natively
        Deopt = 2,      // bailed out mid-run; caller must restore its snapshot and interpret
    };

    struct Frame
    {
        int previousRoundingMode = -1;
        bool useVuRounding = false;
        uint64_t budgetEnd = 0;
        uint8_t *vuData = nullptr;
        uint32_t dataSize = 0;
        GS *gs = nullptr;
        PS2Memory *memory = nullptr;
        uint32_t codeSize = 0;
        uint64_t previousFpcr = 0;
        bool restoreFpcr = false;
    };

    // VU1Interpreter::execute() + the prologue of run().
    static inline Frame beginExecute(VU1Interpreter &vu, uint32_t codeSize, uint8_t *vuData, uint32_t dataSize,
                                     GS &gs, PS2Memory *memory, uint32_t startPC, uint32_t top, uint32_t itop,
                                     uint32_t maxCycles, bool fast = false)
    {
        if (fast)
            vu.resetSchedulerLight();
        else
            vu.resetScheduler();
        auto &s = vu.m_state;
        s.pc = startPC & vu.microAddressMask();
        s.ebit = false;
        s.haltAfterDelaySlot = false;
        s.stoppedByD = false;
        s.stoppedByT = false;
        s.top = top;
        s.itop = itop;
        s.branchPending = false;
        s.branchTarget = 0;
        s.branchDelay = 0;
        s.vf[0][0] = 0.0f;
        s.vf[0][1] = 0.0f;
        s.vf[0][2] = 0.0f;
        s.vf[0][3] = 1.0f;

        vu.m_activeVuData = vuData;
        vu.m_activeVuDataSize = dataSize;
        vu.m_activeGs = &gs;
        vu.m_activeMemory = memory;
        vu.m_lazyFlags = true;

        Frame f;
#if defined(__aarch64__)
        if (fast)
        {
            // Fast mode: round toward zero and flush denormals to zero (FPCR.RMode = RZ, FPCR.FZ),
            // like the VU, set directly (fesetround() is a library call per run).
            uint64_t fpcr;
            __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
            f.previousFpcr = fpcr;
            const uint64_t want = (fpcr & ~(3ull << 22)) | (3ull << 22) | (1ull << 24);
            if (want != fpcr)
            {
                __asm__ volatile("msr fpcr, %0" : : "r"(want));
                f.restoreFpcr = true;
            }
        }
        else
#endif
        {
            f.previousRoundingMode = std::fegetround();
            f.useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
        }
        f.budgetEnd = vu.m_cycle + maxCycles;
        f.vuData = vuData;
        f.dataSize = dataSize;
        f.gs = &gs;
        f.memory = memory;
        f.codeSize = codeSize;
        return f;
    }

    // The epilogue of run(); `ended` must be true (native code only finishes whole programs).
    static inline void leaveLazyFlags(VU1Interpreter &vu)
    {
        vu.m_lazyFlags = false;
        vu.commitDueFlags();
        vu.m_nextCommitCycle = 0; // re-evaluate everything on the next commit
    }

    static inline void endRun(VU1Interpreter &vu, const Frame &f)
    {
        leaveLazyFlags(vu);
        vu.flushPipelines();
        vu.m_state.ebit = false;
        vu.m_state.haltAfterDelaySlot = false;
        vu.m_pendingHaltD = false;
        vu.m_pendingHaltT = false;
        vu.m_lastRunEnded = true;
        vu.m_state.cycles = vu.m_cycle;
        restoreRounding(f);
    }

    static inline void restoreRounding(const Frame &f)
    {
#if defined(__aarch64__)
        if (f.restoreFpcr)
            __asm__ volatile("msr fpcr, %0" : : "r"(f.previousFpcr));
#endif
        if (f.useVuRounding && f.previousRoundingMode != -1)
            std::fesetround(f.previousRoundingMode);
    }

    enum StepResult : int
    {
        Continue = 0,
        Ended = 1,
        Bail = 2,
    };

    // One loop iteration of run(): stall `stall` cycles, then execute the pair.
    // Flags = false when no MAC/status reader is reachable from this pair (see ps2_vu1_recomp).
    template <bool Flags = true, bool Fast = false, bool Plain = false>
    __attribute__((always_inline)) static inline StepResult step(VU1Interpreter &vu, const Frame &f, const Pair &pair, uint32_t stall,
                                                                 uint32_t pairPc = 0)
    {
        if constexpr (Fast)
            vu.m_state.pc = pairPc; // plain pairs don't advance it, so set it for every fast pair
        if (stall != 0u)
            vu.advanceTo(vu.m_cycle + stall);
        if constexpr (!Fast)
        {
            if (vu.m_cycle >= f.budgetEnd || vu.m_stopRequested)
                return Bail;
        }
        bool ended;
        [[clang::always_inline]] ended = vu.template executePair<true, Flags, Fast, Plain>(pair, f.vuData, f.dataSize, *f.gs, f.memory, f.codeSize);
        if (vu.m_stopRequested)
        {
            if constexpr (Plain)
                vu.m_state.pc = (pairPc + 8u) & (f.codeSize - 1u);
            return Bail;
        }
        return ended ? Ended : Continue;
    }

    // Fast mode checks the cycle budget only at loop heads (see ps2_vu1_recomp).
    static inline bool overBudget(const VU1Interpreter &vu, const Frame &f)
    {
        return vu.m_cycle >= f.budgetEnd || vu.m_stopRequested;
    }

    // FNV-1a over VU1 code memory: identifies the image a native module was compiled from.
    static inline uint64_t imageHash(const uint8_t *code, uint32_t size)
    {
        uint64_t h = 0xcbf29ce484222325ull;
        for (uint32_t i = 0; i < size; ++i)
            h = (h ^ code[i]) * 0x100000001b3ull;
        return h;
    }

    // Why the last native run bailed out (diagnostics only).
    enum DeoptReason : int
    {
        DeoptNone = 0,
        DeoptXgkickBusy = 1, // XGKICK while the previous transfer is still running
        DeoptHaltBit = 2,    // D/T bit with halts enabled
        DeoptStep = 3,       // budget exhausted / reserved instruction
        DeoptDispatch = 4,   // control went somewhere the explorer didn't see (computed JR, ...)
        DeoptExit = 5,       // program ended/continued unexpectedly
        DeoptKickWrite = 6,  // lean: a store into the part of a kicked packet PATH1 hasn't read yet
        DeoptKickPacket = 7, // lean: a kicked packet the static copy can't handle (bad tag, too big)
    };
    static inline int &lastDeoptReason() { static thread_local int r = DeoptNone; return r; }
    static inline uint32_t &lastDeoptPc() { static thread_local uint32_t p = 0; return p; }
    static inline void noteDeopt(const VU1Interpreter &vu, int reason)
    {
        lastDeoptReason() = reason;
        lastDeoptPc() = vu.m_state.pc;
#ifdef PS2X_VU1_NATIVE_DEBUG
        std::fprintf(stderr, "[vu1-native] deopt %d at pc 0x%x pending: %s\n", reason, vu.m_state.pc, pendingSummary(vu).c_str());
#endif
    }

    // Guards for things the static model assumes.
    static inline bool xgkickActive(const VU1Interpreter &vu) { return vu.m_xgkick.active; }
    static inline bool dBitEnabled(const VU1Interpreter &vu) { return vu.m_state.dBitEnabled; }
    static inline bool tBitEnabled(const VU1Interpreter &vu) { return vu.m_state.tBitEnabled; }

    // XGKICK issued while the previous transfer still runs: stall exactly like run() does.
    static inline void waitXgkick(VU1Interpreter &vu)
    {
        while (vu.m_xgkick.active)
            vu.advanceOneCycle();
    }

    // True if no timing-relevant pipeline state is pending (what the explorer's "drained"
    // variant of a node assumes).
    static inline bool timingDrained(const VU1Interpreter &vu)
    {
        const uint64_t c = vu.m_cycle;
        if (vu.m_fdiv.valid || vu.m_efuResourceReady > c)
            return false;
        for (const auto &e : vu.m_efu)
            if (e.valid)
                return false;
        for (const auto &reg : vu.m_vfReady)
            for (uint64_t r : reg)
                if (r > c)
                    return false;
        for (uint64_t r : vu.m_viReady)
            if (r > c)
                return false;
        for (uint64_t r : vu.m_accReady)
            if (r > c)
                return false;
        return true;
    }

    static std::string pendingSummary(const VU1Interpreter &vu)
    {
        std::string out;
        const uint64_t c = vu.m_cycle;
        if (vu.m_fdiv.valid) out += "fdiv(" + std::to_string(vu.m_fdiv.readyCycle - c) + ") ";
        if (vu.m_efuResourceReady > c) out += "efuRes ";
        for (const auto &e : vu.m_efu) if (e.valid) out += "efu(" + std::to_string(e.readyCycle - c) + ") ";
        for (int r = 0; r < 32; ++r) for (int k = 0; k < 4; ++k) if (vu.m_vfReady[r][k] > c) out += "vf" + std::to_string(r) + " ";
        for (int r = 0; r < 16; ++r) if (vu.m_viReady[r] > c) out += "vi" + std::to_string(r) + " ";
        for (int k = 0; k < 4; ++k) if (vu.m_accReady[k] > c) out += "acc ";
        return out;
    }

    // Explorer: advance until timingDrained().
    static void drain(VU1Interpreter &vu)
    {
        for (int guard = 0; guard < 4096 && !timingDrained(vu); ++guard)
            vu.advanceOneCycle();
    }

    // Bail out: the interpreter's state is exactly what run() would have, so just keep going there.
    static inline int continueInInterpreter(VU1Interpreter &vu, const Frame &f, uint8_t *vuCode)
    {
        leaveLazyFlags(vu);
        restoreRounding(f);
        const uint64_t used = vu.m_cycle >= f.budgetEnd ? 0 : f.budgetEnd - vu.m_cycle;
        vu.run(vuCode, f.codeSize, f.vuData, f.dataSize, *f.gs, f.memory, static_cast<uint32_t>(used));
        return Deopt;
    }

    // Control state that selects the next statically known node.
    static inline uint32_t pc(const VU1Interpreter &vu) { return vu.m_state.pc; }
    static inline void setPc(VU1Interpreter &vu, uint32_t pc) { vu.m_state.pc = pc; }
    static inline bool branchPending(const VU1Interpreter &vu) { return vu.m_state.branchPending; }
    static inline uint32_t branchTarget(const VU1Interpreter &vu) { return vu.m_state.branchTarget; }

    // ------------------------------------------------------------------ lean code
    // (ps2_vu1_recomp's default mode). Every node's timing state is known at build time, so the
    // generated code doesn't run the cycle model: it adds each pair's precomputed stall to a cycle
    // counter, executes the pair's operations directly (the interpreter's execUpper/execLower and
    // value helpers on constant words), and commits Q/P results at the statically known pairs.
    // What stays dynamic: MAC/status flags (the interpreter's ring, only from pairs whose flags a
    // reader can see), the clip flag (a small ring here) and XGKICK, whose packet is copied whole at
    // the kick and handed to the GIF when PATH1 would have finished it. A store into a part PATH1
    // hasn't read yet deopts before it lands. A deopt rebuilds the interpreter's pipeline state
    // from the node's static timing (LeanNode) and continues there.

    struct LeanCtx
    {
        float qPend = 0.0f; // FDIV result in flight
        uint32_t qDi = 0u;
        float p[2]{};       // EFU results in flight, per interpreter slot
        float pNew = 0.0f;  // the current pair's EFU result
        int32_t viBackup = 0;
        // Clip flag: the working value and the last pushes (cycle, value); a push becomes visible
        // 4 cycles after its issue cycle. At most one push per cycle, so of 8 at least 4 are visible.
        uint32_t wclip = 0u, clip0 = 0u, clipN = 0u;
        uint64_t clipCyc[8]; // only [0, clipN) are read
        uint32_t clipVal[8];
        // XGKICK in flight (its bytes are in vu.m_xgkick.packet).
        bool kickPending = false;
        uint32_t kickSrc = 0u, kickBytes = 0u;
        uint64_t kickIssue = 0u, kickEnd = 0u, kickLast = 0u;
    };

    // Static timing of a node at its entry (relative to the cycle there), for deopts.
    struct LeanNode
    {
        uint16_t pc;
        uint16_t branchTarget;
        uint8_t flags; // 1 branchPending, 2 ebit, 4 haltAfterDelaySlot
        uint8_t branchDelay;
        uint8_t backupReg; // VI whose pre-write value the next branch reads, 0xFF none
        uint8_t fdivRel;   // 0 = no FDIV result in flight
        uint8_t efuRel[2];
        uint8_t efuResRel;
        uint8_t timingCount;
        uint32_t timingOffset; // (code, rel) pairs: code < 128 VF reg*4+lane, < 144 VI, < 148 ACC lane
    };

    // Generated lean code reaches the interpreter's operations through these.
    template <uint32_t U, bool Flags, bool Fast>
    __attribute__((always_inline)) static inline void leanUpper(VU1Interpreter &vu)
    {
        if constexpr (Fast && !Flags)
        {
            bool done;
            [[clang::always_inline]] done = vu.template execUpperFast<false>(U);
            if (done)
                return;
        }
        [[clang::always_inline]] vu.template execUpper<Flags>(U);
    }
    template <uint32_t L, uint32_t U>
    __attribute__((always_inline)) static inline void leanLower(VU1Interpreter &vu, uint8_t *mem)
    {
        [[clang::always_inline]] vu.execLower(L, mem, 0x4000u, *vu.m_activeGs, nullptr, U);
    }
    template <uint32_t L>
    __attribute__((always_inline)) static inline void leanFdiv(VU1Interpreter &vu, LeanCtx &c)
    {
        uint32_t latency;
        vu.fdivResult(L, c.qPend, c.qDi, latency);
    }
    template <uint32_t L>
    __attribute__((always_inline)) static inline void leanEfu(VU1Interpreter &vu, LeanCtx &c)
    {
        uint32_t latency;
        vu.efuResult(L, c.pNew, latency);
    }
    static inline void leanSync(VU1Interpreter &vu, uint64_t cyc) { vu.m_cycle = cyc; }
    static inline void leanCommitFlags(VU1Interpreter &vu, uint64_t cyc)
    {
        vu.m_cycle = cyc;
        vu.commitDueFlags();
    }
    static inline bool stopRequested(const VU1Interpreter &vu) { return vu.m_stopRequested; }
    static inline uint64_t cycle(const VU1Interpreter &vu) { return vu.m_cycle; }

    // PS2X_VU1_LEAN_STRESS (tests): take every deopt the lean code could take (any store while a
    // packet is in flight, any XGKICK while the previous one runs).
    static constexpr bool leanStress()
    {
#ifdef PS2X_VU1_LEAN_STRESS
        return true;
#else
        return false;
#endif
    }

    static inline void leanBegin(VU1Interpreter &vu, LeanCtx &c)
    {
        c.wclip = c.clip0 = vu.m_state.clip;
    }

    static inline void leanClipPush(LeanCtx &c, uint64_t cyc)
    {
        const uint32_t i = c.clipN++ & 7u;
        c.clipCyc[i] = cyc;
        c.clipVal[i] = c.wclip;
    }

    __attribute__((always_inline)) static inline void leanClip(VU1Interpreter &vu, LeanCtx &c, uint32_t fs, uint32_t ft, uint64_t cyc)
    {
        const uint32_t flags = VU1Interpreter::clipFlags(vu.m_state.vf[fs], vu.m_state.vf[ft]);
        c.wclip = ((c.wclip << 6) | (flags & 0x3Fu)) & 0xFFFFFFu;
        leanClipPush(c, cyc);
    }

    // FCSET: replaces a clip push of the same cycle (the pair's own CLIP), like queueFcset.
    static inline void leanFcset(LeanCtx &c, uint32_t value, uint64_t cyc)
    {
        if (c.clipN != 0u && c.clipCyc[(c.clipN - 1u) & 7u] == cyc)
            --c.clipN;
        c.wclip = value & 0xFFFFFFu;
        leanClipPush(c, cyc);
    }

    // The clip flag register as an instruction issued at `cyc` reads it.
    static inline uint32_t leanClipAt(const LeanCtx &c, uint64_t cyc)
    {
        const uint32_t n = c.clipN;
        const uint32_t lim = n > 8u ? n - 8u : 0u;
        for (uint32_t k = n; k > lim; --k)
        {
            const uint32_t i = (k - 1u) & 7u;
            if (c.clipCyc[i] + 4u <= cyc)
                return c.clipVal[i];
        }
        return c.clip0;
    }

    // Q lands: due MAC/status flags first (FDIV rewrites status), like commitReadyPipelines.
    static inline void leanCommitQ(VU1Interpreter &vu, LeanCtx &c, uint64_t at)
    {
        if (vu.m_liveFlag != 0u)
        {
            vu.m_cycle = at;
            vu.commitDueFlags();
        }
        vu.m_state.q = c.qPend;
        vu.m_state.status = (vu.m_state.status & 0xFCFu) | c.qDi | (c.qDi << 6);
    }

    // Walks the GIF tags of a packet at VU address `src` like progressXgkick does; returns its size
    // or 0 if PATH1 would stop on it (bad format, over the buffer).
    static inline uint32_t leanPacketBytes(const uint8_t *mem, uint32_t dataSize, uint32_t src)
    {
        uint32_t off = 0u;
        for (;;)
        {
            if (off > VU1Interpreter::XgkickPipeline::kBufferSize - 16u)
                return 0u;
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, mem + ((src + off) % dataSize), sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else if (format == 2u)
                tagBytes += static_cast<uint64_t>(nloop) * 16u;
            else
                return 0u;
            if (tagBytes > VU1Interpreter::XgkickPipeline::kBufferSize - off)
                return 0u;
            off += static_cast<uint32_t>(tagBytes);
            if ((tagLo >> 15) & 1u)
                return off;
        }
    }

    // XGKICK issued at cycle `issue`: copy the whole packet now. PATH1 reads qword k at cycle
    // issue + 1 + 2k and finishes (hands it to the GIF) at issue + 2n - 1.
    static inline bool leanKick(VU1Interpreter &vu, LeanCtx &c, const uint8_t *mem, uint32_t dataSize, int32_t vi, uint64_t issue)
    {
        const uint32_t src = (static_cast<uint32_t>(static_cast<uint16_t>(vi)) * 16u) % dataSize;
        const uint32_t bytes = leanPacketBytes(mem, dataSize, src);
        if (bytes == 0u)
            return false;
        uint8_t *dst = vu.m_xgkick.packet.data();
        if (src + bytes <= dataSize)
            std::memcpy(dst, mem + src, bytes);
        else
            for (uint32_t done = 0; done < bytes;)
            {
                const uint32_t at = (src + done) % dataSize;
                const uint32_t n = std::min(bytes - done, dataSize - at);
                std::memcpy(dst + done, mem + at, n);
                done += n;
            }
        const uint32_t qwords = bytes / 16u;
        c.kickPending = true;
        c.kickSrc = src;
        c.kickBytes = bytes;
        c.kickIssue = issue;
        c.kickEnd = issue + 2u * qwords - 1u;
        c.kickLast = issue + 2u * (qwords - 1u);
        return true;
    }

    static inline void leanSubmitKick(VU1Interpreter &vu, LeanCtx &c)
    {
        vu.m_xgkick.totalBytes = c.kickBytes;
        vu.m_xgkick.active = true;
        vu.finishXgkick();
        c.kickPending = false;
        c.kickLast = 0u;
    }

    // A store issued at `cyc` to `addr` would land in a qword PATH1 reads later (so the packet
    // differs from the copy taken at the kick).
    static inline bool leanKickConflict(const LeanCtx &c, uint32_t addr, uint64_t cyc, uint32_t dataSize)
    {
        if (!c.kickPending || cyc > c.kickLast)
            return false;
        if (leanStress())
            return true;
        if (c.kickBytes > dataSize)
            return true;
        const uint32_t k = ((addr - c.kickSrc) & (dataSize - 1u)) >> 4;
        return k < (c.kickBytes >> 4) && c.kickIssue + 2ull * k >= cyc;
    }

    // Rebuilds the interpreter's state at the entry of node `n` (cycle `cyc`) and finishes there.
    __attribute__((noinline)) static int leanDeopt(VU1Interpreter &vu, const Frame &f, LeanCtx &c, uint64_t cyc,
                                                   const LeanNode &n, const uint8_t *timing, int reason, uint8_t *vuCode)
    {
        auto &s = vu.m_state;
        vu.m_cycle = cyc;
        s.cycles = cyc;
        for (auto &reg : vu.m_vfReady)
            reg = {};
        vu.m_viReady = {};
        vu.m_accReady = {};
        for (uint32_t i = 0; i < n.timingCount; ++i)
        {
            const uint8_t code = timing[n.timingOffset + 2u * i], rel = timing[n.timingOffset + 2u * i + 1u];
            if (code < 128u)
                vu.m_vfReady[code >> 2][code & 3u] = cyc + rel;
            else if (code < 144u)
                vu.m_viReady[code - 128u] = cyc + rel;
            else if (code < 148u)
                vu.m_accReady[code - 144u] = cyc + rel;
        }
        vu.m_efuResourceReady = n.efuResRel ? cyc + n.efuResRel : 0u;
        vu.m_fdiv = {};
        if (n.fdivRel)
        {
            vu.m_fdiv.valid = true;
            vu.m_fdiv.readyCycle = cyc + n.fdivRel;
            vu.m_fdiv.value = c.qPend;
            vu.m_fdiv.statusDi = c.qDi;
        }
        vu.m_liveEfu = 0u;
        for (int i = 0; i < 2; ++i)
        {
            vu.m_efu[i] = {};
            if (n.efuRel[i])
            {
                vu.m_efu[i].valid = true;
                vu.m_efu[i].readyCycle = cyc + n.efuRel[i];
                vu.m_efu[i].value = c.p[i];
                ++vu.m_liveEfu;
            }
        }
        s.pc = n.pc;
        s.branchPending = (n.flags & 1u) != 0u;
        s.branchTarget = n.branchTarget;
        s.branchDelay = n.branchDelay;
        s.ebit = (n.flags & 2u) != 0u;
        s.haltAfterDelaySlot = (n.flags & 4u) != 0u;
        vu.m_viBranchBackupValid = n.backupReg != 0xFFu;
        vu.m_viBranchBackupReg = n.backupReg != 0xFFu ? n.backupReg : 0u;
        vu.m_viBranchBackupValue = c.viBackup;

        // Clip: visible pushes are committed; the rest join the flag ring in issue order.
        s.clip = leanClipAt(c, cyc);
        vu.m_workingClip = c.wclip;
        {
            VU1Interpreter::FlagPipelineEntry merged[2 * VU1Interpreter::kMaxFlagEntries];
            uint32_t count = 0;
            const uint32_t clipFirst = c.clipN > 8u ? c.clipN - 8u : 0u;
            uint32_t ci = clipFirst;
            auto nextClip = [&]() -> bool
            {
                while (ci < c.clipN && c.clipCyc[ci & 7u] + 4u <= cyc)
                    ++ci;
                return ci < c.clipN;
            };
            for (uint32_t r = 0; r < vu.m_liveFlag; ++r)
            {
                const auto &e = vu.m_flagPipeline[(vu.m_flagHead + r) % VU1Interpreter::kMaxFlagEntries];
                while (nextClip() && c.clipCyc[ci & 7u] <= e.issueCycle && count < 2 * VU1Interpreter::kMaxFlagEntries)
                {
                    auto &m = merged[count++];
                    m = {};
                    m.issueCycle = c.clipCyc[ci & 7u];
                    m.readyCycle = m.issueCycle + VU1Interpreter::kFmacLatency;
                    m.clip = c.clipVal[ci & 7u];
                    m.writesClip = m.valid = true;
                    ++ci;
                }
                if (count < 2 * VU1Interpreter::kMaxFlagEntries)
                    merged[count++] = e;
            }
            while (nextClip() && count < 2 * VU1Interpreter::kMaxFlagEntries)
            {
                auto &m = merged[count++];
                m = {};
                m.issueCycle = c.clipCyc[ci & 7u];
                m.readyCycle = m.issueCycle + VU1Interpreter::kFmacLatency;
                m.clip = c.clipVal[ci & 7u];
                m.writesClip = m.valid = true;
                ++ci;
            }
            if (count > VU1Interpreter::kMaxFlagEntries)
                count = VU1Interpreter::kMaxFlagEntries; // cannot happen: at most 2 entries per cycle for 4 cycles
            vu.m_flagPipeline = {};
            vu.m_flagHead = 0u;
            for (uint32_t i = 0; i < count; ++i)
                vu.m_flagPipeline[i] = merged[i];
            vu.m_liveFlag = count;
        }

        // XGKICK in flight: PATH1's progress by now, from the copy (its bytes so far are what
        // PATH1 read: no store hit them, or this would have deopted earlier).
        if (c.kickPending)
        {
            const uint64_t credits = 1u + (cyc - c.kickIssue);
            const uint32_t qwords = c.kickBytes / 16u;
            if (credits / 2u >= qwords)
                leanSubmitKick(vu, c);
            else
            {
                const uint32_t copied = static_cast<uint32_t>(credits / 2u) * 16u;
                auto &x = vu.m_xgkick;
                x.clear();
                x.active = true;
                x.sourceAddress = c.kickSrc;
                x.copiedBytes = copied;
                x.cycleCredit = static_cast<uint32_t>(credits - 2u * (credits / 2u));
                x.issueCycle = c.kickIssue;
                uint32_t off = 0u;
                while (off < copied)
                {
                    uint64_t tagLo = 0;
                    std::memcpy(&tagLo, x.packet.data() + off, sizeof(tagLo));
                    const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
                    const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
                    uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
                    if (nreg == 0u)
                        nreg = 16u;
                    uint32_t tagBytes = 16u;
                    if (format == 0u)
                        tagBytes += nloop * nreg * 16u;
                    else if (format == 1u)
                        tagBytes += ((nloop * nreg + 1u) & ~1u) * 8u;
                    else
                        tagBytes += nloop * 16u;
                    const bool eop = ((tagLo >> 15) & 1u) != 0u;
                    const uint32_t end = off + tagBytes;
                    if (eop)
                        x.totalBytes = end;
                    if (end > copied)
                    {
                        x.currentTagEnd = end;
                        x.currentTagEop = eop;
                        break;
                    }
                    off = end;
                }
                c.kickPending = false;
            }
        }
        vu.m_nextCommitCycle = 0u;
        noteDeopt(vu, reason);
        return continueInInterpreter(vu, f, vuCode);
    }

    // End of program: like endRun(), with the FDIV/EFU results still in flight after the last pair
    // (rel 0 = none) and a kicked packet handed over.
    static inline void leanEnd(VU1Interpreter &vu, const Frame &f, LeanCtx &c, uint64_t cyc, uint8_t fdivRel,
                               uint8_t efuRel0, uint8_t efuRel1)
    {
        uint64_t end = cyc;
        if (c.kickPending)
        {
            end = std::max(end, c.kickEnd);
            leanSubmitKick(vu, c);
        }
        if (c.clipN != 0u)
            vu.m_state.clip = c.clipVal[(c.clipN - 1u) & 7u];
        vu.m_workingClip = vu.m_state.clip;
        vu.m_cycle = cyc;
        if (vu.m_liveFlag == 0u)
        {
            // Nothing else is queued: Q and P land in ready order (equal cycles: EFU slot order).
            if (fdivRel)
            {
                vu.m_state.q = c.qPend;
                vu.m_state.status = (vu.m_state.status & 0xFCFu) | c.qDi | (c.qDi << 6);
                end = std::max<uint64_t>(end, cyc + fdivRel);
            }
            if (efuRel0 || efuRel1)
            {
                const int last = efuRel1 >= efuRel0 && efuRel1 ? 1 : 0;
                vu.m_state.p = c.p[last];
                end = std::max<uint64_t>(end, cyc + std::max(efuRel0, efuRel1));
            }
            vu.m_lazyFlags = false;
            vu.m_nextCommitCycle = 0;
        }
        else
        {
            if (fdivRel)
            {
                vu.m_fdiv.valid = true;
                vu.m_fdiv.readyCycle = cyc + fdivRel;
                vu.m_fdiv.value = c.qPend;
                vu.m_fdiv.statusDi = c.qDi;
                vu.notePipelineReady(vu.m_fdiv.readyCycle);
            }
            const uint8_t rel[2] = {efuRel0, efuRel1};
            for (int i = 0; i < 2; ++i)
                if (rel[i])
                {
                    vu.m_efu[i].valid = true;
                    vu.m_efu[i].readyCycle = cyc + rel[i];
                    vu.m_efu[i].value = c.p[i];
                    ++vu.m_liveEfu;
                    vu.notePipelineReady(vu.m_efu[i].readyCycle);
                }
            leaveLazyFlags(vu);
            vu.flushPipelines();
        }
        if (vu.m_cycle < end)
            vu.m_cycle = end;
        vu.m_state.ebit = false;
        vu.m_state.haltAfterDelaySlot = false;
        vu.m_pendingHaltD = false;
        vu.m_pendingHaltT = false;
        vu.m_lastRunEnded = true;
        vu.m_state.cycles = vu.m_cycle;
        restoreRounding(f);
    }

    // LQ..ISWR store addresses (for leanKickConflict), exactly as execLower computes them.
    static inline uint32_t leanStoreAddr(const VU1State &s, uint32_t lower, uint32_t dataSize)
    {
        const uint32_t opHi = (lower >> 25) & 0x7Fu;
        const uint8_t viT = VIT(lower), viS = VIS(lower);
        uint32_t addr = 0u;
        if (opHi == 0x01u) // SQ
            addr = static_cast<uint32_t>(static_cast<int32_t>(s.vi[viT] + IMM11(lower))) * 16u;
        else if (opHi == 0x05u) // ISW
            addr = static_cast<uint32_t>(static_cast<int32_t>(s.vi[viS] + IMM11(lower))) * 16u;
        else
        {
            const uint8_t funct2 = static_cast<uint8_t>((lower & 0x3u) | ((lower >> 4) & 0x7Cu));
            if (funct2 == 0x35u) // SQI
                addr = static_cast<uint32_t>(static_cast<uint16_t>(s.vi[viT])) * 16u;
            else if (funct2 == 0x37u) // SQD
                addr = static_cast<uint32_t>(static_cast<uint16_t>(viT != 0u ? static_cast<int16_t>(s.vi[viT] - 1) : s.vi[viT])) * 16u;
            else // ISWR
                addr = static_cast<uint32_t>(static_cast<uint16_t>(s.vi[viS])) * 16u;
        }
        return addr & (dataSize - 1u);
    }

    // ------------------------------------------------------------------ recompiler tooling
    // (used by ps2_vu1_recomp at build time; not by generated code)

    static Pair decode(const VU1Interpreter &vu, const uint8_t *code, uint32_t pc)
    {
        return vu.decodeInstructionPair(code, pc);
    }

    // Same stall loop as run(); returns the number of cycles stalled.
    static uint32_t resolveStall(VU1Interpreter &vu, const Pair &pair)
    {
        const uint64_t start = vu.m_cycle;
        uint64_t ready = vu.calculatePairReadyCycle(pair);
        while (ready > vu.m_cycle)
        {
            vu.advanceTo(ready);
            ready = vu.calculatePairReadyCycle(pair);
        }
        return static_cast<uint32_t>(vu.m_cycle - start);
    }

    static bool executeForExploration(VU1Interpreter &vu, const Pair &pair, uint8_t *data, uint32_t dataSize,
                                      GS &gs, uint32_t codeSize)
    {
        vu.m_activeVuData = data;
        vu.m_activeVuDataSize = dataSize;
        vu.m_activeGs = &gs;
        vu.m_activeMemory = nullptr;
        const bool ended = vu.executePair(pair, data, dataSize, gs, nullptr, codeSize);
        // Model: an XGKICK transfer completes before the next XGKICK (guarded at runtime).
        vu.m_xgkick.active = false;
        vu.m_stopRequested = false;
        return ended;
    }

    // Overrides a conditional branch's outcome after executePair() ran it (its delay countdown
    // has already been consumed, so a taken branch is pending with delay 0).
    static void forceBranch(VU1Interpreter &vu, bool taken, uint32_t target)
    {
        vu.m_state.branchPending = taken;
        if (taken)
            vu.m_state.branchTarget = target;
        vu.m_state.branchDelay = 0u;
    }


    // Lean exploration helpers.
    struct LeanProbe
    {
        LeanNode node{};
        std::vector<std::pair<uint8_t, uint8_t>> timing;
        bool fdivValid = false;
        bool efuValid[2]{};
    };
    static uint8_t rel8(uint64_t ready, uint64_t now) { return ready > now ? static_cast<uint8_t>(std::min<uint64_t>(ready - now, 255)) : 0u; }
    static LeanProbe leanSnapshot(const VU1Interpreter &vu)
    {
        LeanProbe p;
        const uint64_t c = vu.m_cycle;
        const auto &s = vu.m_state;
        for (int r = 0; r < 32; ++r)
            for (int k = 0; k < 4; ++k)
                if (uint8_t v = rel8(vu.m_vfReady[r][k], c))
                    p.timing.push_back({static_cast<uint8_t>(r * 4 + k), v});
        for (int r = 0; r < 16; ++r)
            if (uint8_t v = rel8(vu.m_viReady[r], c))
                p.timing.push_back({static_cast<uint8_t>(128 + r), v});
        for (int k = 0; k < 4; ++k)
            if (uint8_t v = rel8(vu.m_accReady[k], c))
                p.timing.push_back({static_cast<uint8_t>(144 + k), v});
        p.node.pc = static_cast<uint16_t>(s.pc);
        p.node.branchTarget = static_cast<uint16_t>(s.branchTarget);
        p.node.flags = static_cast<uint8_t>((s.branchPending ? 1u : 0u) | (s.ebit ? 2u : 0u) | (s.haltAfterDelaySlot ? 4u : 0u));
        p.node.branchDelay = static_cast<uint8_t>(s.branchDelay);
        p.node.backupReg = vu.m_viBranchBackupValid ? vu.m_viBranchBackupReg : 0xFFu;
        p.fdivValid = vu.m_fdiv.valid;
        p.node.fdivRel = vu.m_fdiv.valid ? rel8(vu.m_fdiv.readyCycle, c) : 0u;
        for (int i = 0; i < 2; ++i)
        {
            p.efuValid[i] = vu.m_efu[i].valid;
            p.node.efuRel[i] = vu.m_efu[i].valid ? rel8(vu.m_efu[i].readyCycle, c) : 0u;
        }
        p.node.efuResRel = rel8(vu.m_efuResourceReady, c);
        return p;
    }
    // Tags in-flight Q/P values (NaNs: real results are normalized and never NaN) to see which land when.
    static constexpr uint32_t kTagQ = 0x7FC0A001u, kTagP0 = 0x7FC0A002u, kTagP1 = 0x7FC0A003u, kTagNone = 0x7FC0A00Fu;
    static float tagF(uint32_t bits) { float f; std::memcpy(&f, &bits, 4); return f; }
    static uint32_t bitsF(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
    static void leanTag(VU1Interpreter &vu)
    {
        if (vu.m_fdiv.valid)
            vu.m_fdiv.value = tagF(kTagQ);
        if (vu.m_efu[0].valid)
            vu.m_efu[0].value = tagF(kTagP0);
        if (vu.m_efu[1].valid)
            vu.m_efu[1].value = tagF(kTagP1);
    }
    // After a phase: did Q land (from the tagged value)? Which P slot landed last (-1 none)?
    static bool leanQLanded(const VU1Interpreter &vu) { return bitsF(vu.m_state.q) == kTagQ; }
    static int leanPLanded(const VU1Interpreter &vu)
    {
        const uint32_t b = bitsF(vu.m_state.p);
        return b == kTagP0 ? 0 : b == kTagP1 ? 1 : -1;
    }
    static void leanUntag(VU1Interpreter &vu)
    {
        vu.m_state.q = tagF(kTagNone);
        vu.m_state.p = tagF(kTagNone);
    }
    static bool efuValid(const VU1Interpreter &vu, int i) { return vu.m_efu[i].valid; }
    static uint32_t drainCycles(const VU1Interpreter &vu)
    {
        VU1Interpreter copy = vu;
        const uint64_t start = copy.m_cycle;
        drain(copy);
        return static_cast<uint32_t>(copy.m_cycle - start);
    }
    static VU1State &state(VU1Interpreter &vu) { return vu.m_state; }
    static void prepareExploration(VU1Interpreter &vu)
    {
        vu.resetScheduler();
        vu.m_state = VU1State{};
        vu.m_state.vf[0][3] = 1.0f;
    }

    // Everything that influences future stalls or control flow, relative to the current cycle.
    // Two states with equal keys execute the same instruction stream with identical timing.
    static std::string timingKey(const VU1Interpreter &vu, uint16_t linkViMask)
    {
        std::string k;
        const uint64_t c = vu.m_cycle;
        auto rel = [&](uint64_t ready) -> uint8_t { return ready > c ? static_cast<uint8_t>(std::min<uint64_t>(ready - c, 255)) : 0; };
        for (const auto &reg : vu.m_vfReady)
            for (uint64_t r : reg)
                k.push_back(static_cast<char>(rel(r)));
        for (uint64_t r : vu.m_viReady)
            k.push_back(static_cast<char>(rel(r)));
        for (uint64_t r : vu.m_accReady)
            k.push_back(static_cast<char>(rel(r)));
        k.push_back(static_cast<char>(vu.m_fdiv.valid ? 1 + rel(vu.m_fdiv.readyCycle) : 0));
        for (const auto &e : vu.m_efu)
            k.push_back(static_cast<char>(e.valid ? 1 + rel(e.readyCycle) : 0));
        k.push_back(static_cast<char>(rel(vu.m_efuResourceReady)));
        const auto &s = vu.m_state;
        k.push_back(static_cast<char>(s.branchPending));
        if (s.branchPending)
        {
            k.push_back(static_cast<char>(s.branchDelay));
            k.append(reinterpret_cast<const char *>(&s.branchTarget), sizeof(s.branchTarget));
        }
        k.push_back(static_cast<char>(s.ebit));
        k.push_back(static_cast<char>(s.haltAfterDelaySlot));
        k.push_back(static_cast<char>(vu.m_viBranchBackupValid ? vu.m_viBranchBackupReg : 0xFF));
        for (int r = 1; r < 16; ++r)
            if (linkViMask & (1u << r))
                k.append(reinterpret_cast<const char *>(&s.vi[r]), sizeof(s.vi[r]));
        return k;
    }
};
