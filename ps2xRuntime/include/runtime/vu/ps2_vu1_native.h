#pragma once

// Bridge between VU1Interpreter and VU1 microcode recompiled to C++ by ps2_vu1_recomp.
//
// Recompiled code is the interpreter's run loop with decode and hazard-stall computation done
// at build time: for each instruction pair it advances the cycle counter by the precomputed
// stall (committing pipelines / progressing XGKICK exactly as the interpreter would) and then
// runs VU1Interpreter::executePair() with a constexpr decoded pair, so every instruction's
// effect is literally the interpreter's code. Control flow dispatches on the PC/branch state
// that executePair() produces; anything outside the statically explored model deopts, and the
// caller re-runs the whole program in the interpreter from a snapshot.

#include "runtime/vu/ps2_vu1_exec.inl"

#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <string>

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
    };

    // VU1Interpreter::execute() + the prologue of run().
    static inline Frame beginExecute(VU1Interpreter &vu, uint32_t codeSize, uint8_t *vuData, uint32_t dataSize,
                                     GS &gs, PS2Memory *memory, uint32_t startPC, uint32_t top, uint32_t itop,
                                     uint32_t maxCycles)
    {
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
        f.previousRoundingMode = std::fegetround();
        f.useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
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
    __attribute__((always_inline)) static inline StepResult step(VU1Interpreter &vu, const Frame &f, const Pair &pair, uint32_t stall)
    {
        if (stall != 0u)
            vu.advanceTo(vu.m_cycle + stall);
        if (vu.m_cycle >= f.budgetEnd || vu.m_stopRequested)
            return Bail;
        bool ended;
        [[clang::always_inline]] ended = vu.executePair<true>(pair, f.vuData, f.dataSize, *f.gs, f.memory, f.codeSize);
        if (vu.m_stopRequested)
            return Bail;
        return ended ? Ended : Continue;
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
    static inline bool branchPending(const VU1Interpreter &vu) { return vu.m_state.branchPending; }
    static inline uint32_t branchTarget(const VU1Interpreter &vu) { return vu.m_state.branchTarget; }

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
