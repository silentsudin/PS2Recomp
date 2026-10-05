// ps2_vu1_recomp: statically recompiles VU1 microcode to C++.
//
//   ps2_vu1_recomp --elf GAME.ELF --out vu1_native.cpp [--entries 0x0,0x10,...] [--exact] [--interp-steps]
//   ps2_vu1_recomp --image code.bin --out vu1_native.cpp [--entries ...]
//
// The VU1 code image is assembled from the ELF's .DVP.ovlytab overlays (or read raw). Starting
// from each MSCAL entry with the clean pipeline state execute() produces, the CFG is explored
// with the interpreter's own timing model (forcing both outcomes of conditional branches) until
// every reachable (pc, timing state) node is known.
//
// Default (lean) output: each node is straight-line code. Its stall is a constant added to a
// cycle counter, the pair's operations run directly on constant words, the cycle where each
// DIV/EFU result lands in Q/P is resolved at build time, branches and delay slots are plain gotos,
// and an XGKICK copies the whole packet at the kick (see "lean code" in runtime/vu/ps2_vu1_native.h).
// --interp-steps: the previous output, one VU1Interpreter::executePair() per node on a constexpr
// decoded pair after the node's precomputed stall (the interpreter's cycle model at run time).
// --exact: the interpreter's exact float model everywhere instead of the 4-lane fast path (FTZ, RZ,
// clamping) for pairs whose MAC/status flags nothing reads. See runtime/vu/ps2_vu1_native.h.

#include "runtime/vu/ps2_vu1_native.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <memory>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kCodeSize = 0x4000;
    constexpr size_t kMaxNodes = 200000;

    bool readFile(const std::string &path, std::vector<uint8_t> &out)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            return false;
        out.assign(std::istreambuf_iterator<char>(in), {});
        return true;
    }

    // Lays the .DVP.ovlytab overlays (EE lma -> VU vma) into a VU1 code image.
    bool imageFromElf(const std::vector<uint8_t> &elf, std::vector<uint8_t> &image, std::string &error)
    {
        auto u32 = [&](size_t off) { uint32_t v; std::memcpy(&v, elf.data() + off, 4); return v; };
        auto u16 = [&](size_t off) { uint16_t v; std::memcpy(&v, elf.data() + off, 2); return v; };
        if (elf.size() < 52 || std::memcmp(elf.data(), "\x7f" "ELF", 4) != 0)
            return error = "not an ELF", false;
        const uint32_t shoff = u32(32), shnum = u16(48), shstr = u16(50);
        struct Sec { std::string name; uint32_t addr, off, size; };
        std::vector<Sec> secs;
        const uint32_t strOff = u32(shoff + shstr * 40 + 16);
        for (uint32_t i = 0; i < shnum; ++i)
        {
            const size_t h = shoff + i * 40;
            secs.push_back({std::string(reinterpret_cast<const char *>(elf.data() + strOff + u32(h))),
                            u32(h + 12), u32(h + 16), u32(h + 20)});
        }
        const Sec *tab = nullptr;
        for (auto &s : secs)
            if (s.name == ".DVP.ovlytab")
                tab = &s;
        if (!tab)
            return error = "no .DVP.ovlytab in ELF", false;

        // Overlay sections appear in table order; each entry is (name offset, lma, vma).
        std::vector<const Sec *> overlays;
        for (auto &s : secs)
            if (s.name.rfind(".DVP.overlay", 0) == 0)
                overlays.push_back(&s);
        const uint32_t entries = tab->size / 12;
        if (entries != overlays.size())
            return error = "overlay table/section count mismatch", false;

        image.assign(kCodeSize, 0);
        for (uint32_t i = 0; i < entries; ++i)
        {
            // The overlay sections are placeholders; the code lives at the load address (lma),
            // inside an allocated section such as .vutext.
            const uint32_t lma = u32(tab->off + i * 12 + 4);
            const uint32_t vma = u32(tab->off + i * 12 + 8);
            const Sec &ov = *overlays[i];
            if (vma + ov.size > kCodeSize)
                return error = "overlay exceeds VU1 code memory", false;
            const Sec *home = nullptr;
            for (auto &s : secs)
                if (s.addr != 0 && lma >= s.addr && lma + ov.size <= s.addr + s.size)
                    home = &s;
            if (!home)
                return error = "overlay load address not inside any section", false;
            std::memcpy(image.data() + vma, elf.data() + home->off + (lma - home->addr), ov.size);
            std::cerr << "overlay " << i << ": " << ov.size << " bytes -> VU 0x" << std::hex << vma << std::dec << "\n";
        }
        return true;
    }

    struct Edge
    {
        uint32_t pc;
        bool pending;
        uint32_t target;
        size_t node;
    };

    struct Node
    {
        uint32_t pc = 0;
        uint32_t stall = 0;
        bool ended = false;
        bool unsupported = false;
        size_t drainedVariant = SIZE_MAX; // XGKICK pairs: node to resume at after a dynamic stall
        std::vector<Edge> succ;
        // Lean mode: the static timing at entry and where in-flight Q/P results land.
        Vu1Native::LeanProbe entry;
        bool qPre = false, qPost = false; // FDIV result lands during the stall / after the pair
        int pPre = -1, pPost = -1;        // EFU slot whose result lands last during the stall / after
        int efuSlot = -1;                 // slot the pair's own EFU op takes
        uint32_t drainCycles = 0;         // XGKICK pairs: cycles until timingDrained()
        Vu1Native::LeanProbe exit;        // ended nodes: state after the last pair
    };

    std::string hex(uint64_t v)
    {
        std::ostringstream s;
        s << "0x" << std::hex << v;
        return s.str();
    }

    std::string emitUsage(const Vu1Native::Usage &u)
    {
        std::ostringstream s;
        s << "{{{{" << int(u.vfRead[0].reg) << "," << int(u.vfRead[0].lanes) << "},{" << int(u.vfRead[1].reg) << ","
          << int(u.vfRead[1].lanes) << "}}},{" << int(u.vfWrite.reg) << "," << int(u.vfWrite.lanes) << "},"
          << int(u.vfReadCount) << "," << u.viRead << "," << u.viWrite << "," << int(u.accRead) << ","
          << int(u.accWrite) << "," << int(u.latency) << "," << int(u.vfLatency) << "," << int(u.viLatency)
          << ",static_cast<Vu1Native::Pipeline>(" << int(u.pipeline) << ")," << u.waitQ << "," << u.waitP << ","
          << u.readsClip << "," << u.writesClip << "," << u.delaysNextBranchRead << "," << u.reserved << "}";
        return s.str();
    }

    std::string emitPair(const Vu1Native::Pair &p)
    {
        std::ostringstream s;
        s << "{" << hex(p.lower) << "u," << hex(p.upper) << "u," << emitUsage(p.lowerUsage) << ","
          << emitUsage(p.upperUsage) << "," << p.iBit << "," << p.eBit << "," << p.mBit << "," << p.dBit << ","
          << p.tBit << "," << int(p.upperVfShadowReg) << "," << int(p.suppressedLowerVf) << "}";
        return s.str();
    }

    bool isConditionalBranch(const Vu1Native::Pair &p)
    {
        if (p.iBit || p.lowerUsage.pipeline != Vu1Native::Pipeline(6 /* PipelineBranch */))
            return false;
        const uint32_t op = p.lower >> 25;
        return op >= 0x28 && op <= 0x2F;
    }
    // ------------------------------------------------------------------ lean code generation
    // See "lean code" in runtime/vu/ps2_vu1_native.h.

    enum class LowerKind
    {
        None,      // NOP or I-bit immediate
        Plain,     // execLower as is
        Branch,    // B BAL JR JALR IBxx: control flow is generated per node
        Xgkick,    // copied at the node (leanKick)
        Fdiv,      // DIV SQRT RSQRT -> c.qPend
        Efu,       // EFU ops -> c.pNew
        Wait,      // WAITQ WAITP (stall only)
        Fcset,     // clip ring
        ClipRead,  // FCEQ FCAND FCOR FCGET: clip ring
        FlagRead,  // FSxx FMxx FSSET: the interpreter's flag ring, committed up to now
        Store,     // SQ ISW SQI SQD ISWR: execLower, but checked against an in-flight XGKICK
    };

    uint8_t special2(uint32_t w) { return static_cast<uint8_t>((w & 0x3u) | ((w >> 4) & 0x7Cu)); }

    LowerKind lowerKind(const Vu1Native::Pair &p)
    {
        if (p.iBit || p.lower == 0u || p.lower == 0x8000033Cu)
            return LowerKind::None;
        const uint32_t opHi = (p.lower >> 25) & 0x7Fu;
        switch (opHi)
        {
        case 0x01: case 0x05: return LowerKind::Store;
        case 0x10: case 0x12: case 0x13: case 0x1C: return LowerKind::ClipRead;
        case 0x11: return LowerKind::Fcset;
        case 0x14: case 0x15: case 0x16: case 0x17: case 0x18: case 0x1A: case 0x1B: return LowerKind::FlagRead;
        case 0x20: case 0x21: case 0x24: case 0x25: case 0x28: case 0x29: case 0x2C: case 0x2D: case 0x2E: case 0x2F:
            return LowerKind::Branch;
        case 0x40:
        {
            if ((p.lower & 0x3Fu) < 0x3Cu)
                return LowerKind::Plain;
            const uint8_t f = special2(p.lower);
            if (f == 0x35u || f == 0x37u || f == 0x3Fu)
                return LowerKind::Store;
            if (f >= 0x38u && f <= 0x3Au)
                return LowerKind::Fdiv;
            if (f == 0x3Bu || f == 0x7Bu)
                return LowerKind::Wait;
            if ((f >= 0x70u && f <= 0x7Au) || f == 0x7Cu || f == 0x7Du)
                return LowerKind::Efu;
            if (f == 0x6Cu)
                return LowerKind::Xgkick;
            return LowerKind::Plain;
        }
        default:
            return LowerKind::Plain;
        }
    }

    bool isClipUpper(uint32_t upper)
    {
        return (upper & 0x3Fu) >= 0x3Cu && special2(upper) == 0x1Fu;
    }

    // Might the pair leave something in VF0/VI0 (which run() resets after every pair)? Conservative.
    bool mayWriteZeroRegs(const Vu1Native::Pair &p)
    {
        const uint32_t u = p.upper;
        if ((u & 0x3Fu) < 0x3Cu)
        {
            if (FD(u) == 0u)
                return true;
        }
        else
        {
            const uint8_t sop = special2(u);
            if (((sop >= 0x10u && sop <= 0x17u) || sop == 0x1Du) && FT(u) == 0u)
                return true;
        }
        if (p.iBit || p.lower == 0u || p.lower == 0x8000033Cu)
            return false;
        const uint32_t opHi = (p.lower >> 25) & 0x7Fu;
        if (opHi == 0x00u && FT(p.lower) == 0u)
            return true;
        if (opHi == 0x40u && (p.lower & 0x3Fu) >= 0x3Cu)
        {
            const uint8_t f = special2(p.lower);
            if ((f == 0x30u || f == 0x31u || f == 0x34u || f == 0x36u || f == 0x3Du || f == 0x40u || f == 0x41u || f == 0x64u) &&
                FT(p.lower) == 0u)
                return true;
        }
        return false;
    }

    uint32_t normalizedImmediate(uint32_t bits)
    {
        const uint32_t exponent = bits & 0x7F800000u, sign = bits & 0x80000000u;
        if (exponent == 0u)
            return sign;
        if (exponent == 0x7F800000u)
            return sign | 0x7F7FFFFFu;
        return bits;
    }

    std::string hx(uint64_t v)
    {
        std::ostringstream s;
        s << "0x" << std::hex << v << "u";
        return s.str();
    }

    // `out` has the include and an open anonymous namespace. With `split`, each entry (microprogram)
    // becomes its own function in its own file next to `outPath` (<stem>.part<i>.cpp): one function
    // holding every program compiled superlinearly slowly (minutes on a phone), and parts compile
    // in parallel. Nodes reachable from several entries are emitted in each.
    bool emitLean(std::ofstream &out, const std::vector<Node> &nodes, const std::vector<uint32_t> &entries,
                  const std::vector<size_t> &entryNodes, const std::vector<uint8_t> &image, const VU1Interpreter &base,
                  const std::set<uint32_t> &flagsLive, const std::map<uint32_t, bool> &pcsUsed,
                  const std::vector<bool> &loopHead, bool exact, bool split, const std::string &outPath)
    {
        const std::string fast = exact ? "false" : "true";
        // One function per instruction pair: its upper and lower operations on constant words
        // (timing, Q/P landing, branches and XGKICK are per node).
        auto pairFn = [&](std::ostream &out, uint32_t pc)
        {
            const auto p = Vu1Native::decode(base, image.data(), pc);
            const LowerKind lk = lowerKind(p);
            const bool flags = flagsLive.count(pc) != 0u;
            out << "    __attribute__((always_inline)) inline void X" << std::hex << pc << std::dec
                << "(VU1Interpreter &vu, VU1State &s, uint8_t *mem, Vu1Native::LeanCtx &c, uint64_t cyc)\n    {\n"
                << "        (void)vu; (void)s; (void)mem; (void)c; (void)cyc;\n";
            if (!p.iBit && p.lowerUsage.delaysNextBranchRead && p.lowerUsage.viWrite != 0u)
            {
                for (uint32_t r = 1; r < 16u; ++r)
                    if (p.lowerUsage.viWrite & (1u << r))
                    {
                        out << "        c.viBackup = s.vi[" << r << "];\n";
                        break;
                    }
            }
            std::string upper;
            if (isClipUpper(p.upper))
                upper = "Vu1Native::leanClip(vu, c, " + std::to_string(FS(p.upper)) + "u, " + std::to_string(FT(p.upper)) + "u, cyc);";
            else if (flags)
                upper = "Vu1Native::leanSync(vu, cyc); Vu1Native::leanUpper<" + hx(p.upper) + ", true, false>(vu);";
            else
                upper = "Vu1Native::leanUpper<" + hx(p.upper) + ", false, " +
                        ((p.iBit || p.upperVfShadowReg != 0u) ? std::string("false") : fast) + ">(vu);";
            std::string lower;
            const std::string L = hx(p.lower), U = hx(p.upper);
            switch (lk)
            {
            case LowerKind::None:
            case LowerKind::Branch:
            case LowerKind::Xgkick:
            case LowerKind::Wait:
                break;
            case LowerKind::Plain:
            case LowerKind::Store:
                lower = "Vu1Native::leanLower<" + L + ", " + U + ">(vu, mem);";
                break;
            case LowerKind::Fdiv:
                lower = "Vu1Native::leanFdiv<" + L + ">(vu, c);";
                break;
            case LowerKind::Efu:
                lower = "Vu1Native::leanEfu<" + L + ">(vu, c);";
                break;
            case LowerKind::Fcset:
                lower = "Vu1Native::leanFcset(c, " + hx(p.lower & 0xFFFFFFu) + ", cyc);";
                break;
            case LowerKind::FlagRead:
                lower = "Vu1Native::leanCommitFlags(vu, cyc); Vu1Native::leanLower<" + L + ", " + U + ">(vu, mem);";
                break;
            case LowerKind::ClipRead:
            {
                const uint32_t opHi = (p.lower >> 25) & 0x7Fu, imm = p.lower & 0xFFFFFFu;
                const std::string clip = "Vu1Native::leanClipAt(c, cyc)";
                if (opHi == 0x10u)
                    lower = "s.vi[1] = ((" + clip + " & 0xFFFFFFu) == " + hx(imm) + ") ? 1 : 0;";
                else if (opHi == 0x12u)
                    lower = "s.vi[1] = ((" + clip + " & " + hx(imm) + ") != 0u) ? 1 : 0;";
                else if (opHi == 0x13u)
                    lower = "s.vi[1] = ((" + clip + " | " + hx(imm) + ") == 0xFFFFFFu) ? 1 : 0;";
                else if (VIT(p.lower) != 0u)
                    lower = "s.vi[" + std::to_string(VIT(p.lower)) + "] = static_cast<int32_t>(" + clip + " & 0x0FFFu);";
                break;
            }
            }
            if (p.upperVfShadowReg != 0u)
            {
                const std::string R = std::to_string(p.upperVfShadowReg);
                out << "        float oldVf[4], upperVf[4];\n"
                    << "        std::memcpy(oldVf, s.vf[" << R << "], 16);\n"
                    << "        " << upper << "\n"
                    << "        std::memcpy(upperVf, s.vf[" << R << "], 16);\n"
                    << "        std::memcpy(s.vf[" << R << "], oldVf, 16);\n"
                    << "        " << lower << "\n"
                    << "        std::memcpy(s.vf[" << R << "], upperVf, 16);\n";
            }
            else
            {
                out << "        " << upper << "\n";
                if (!lower.empty())
                    out << "        " << lower << "\n";
            }
            if (p.iBit)
                out << "        { const uint32_t b = " << hx(normalizedImmediate(p.lower)) << "; std::memcpy(&s.i, &b, 4); }\n";
            if (mayWriteZeroRegs(p))
                out << "        s.vf[0][0] = 0.0f; s.vf[0][1] = 0.0f; s.vf[0][2] = 0.0f; s.vf[0][3] = 1.0f; s.vi[0] = 0;\n";
            out << "    }\n";
        };

        // Static timing per node, for deopts.
        std::map<std::vector<std::pair<uint8_t, uint8_t>>, uint32_t> timingOffsets;
        std::vector<uint8_t> timingBytes;
        std::ostringstream tables;
        tables << "namespace vu1gen\n{\n    extern const Vu1Native::LeanNode kLeanNodes[] = {\n";
        for (const Node &n : nodes)
        {
            auto it = timingOffsets.find(n.entry.timing);
            if (it == timingOffsets.end())
            {
                it = timingOffsets.emplace(n.entry.timing, static_cast<uint32_t>(timingBytes.size())).first;
                for (auto [code, rel] : n.entry.timing)
                {
                    timingBytes.push_back(code);
                    timingBytes.push_back(rel);
                }
            }
            const auto &e = n.entry.node;
            tables << "        {" << hx(e.pc) << ", " << hx(e.branchTarget) << ", " << int(e.flags) << ", " << int(e.branchDelay) << ", "
                << int(e.backupReg) << ", " << int(e.fdivRel) << ", {" << int(e.efuRel[0]) << ", " << int(e.efuRel[1]) << "}, "
                << int(e.efuResRel) << ", " << n.entry.timing.size() << ", " << it->second << "},\n";
        }
        tables << "    };\n    extern const uint8_t kLeanTiming[] = {";
        for (size_t i = 0; i < timingBytes.size(); ++i)
            tables << (i % 32 == 0 ? "\n        " : "") << int(timingBytes[i]) << ",";
        tables << "0};\n}\n\n";
        const char *tableDecls = "namespace vu1gen\n{\n    extern const Vu1Native::LeanNode kLeanNodes[];\n"
                                 "    extern const uint8_t kLeanTiming[];\n}\n\n";
        const std::string params = "VU1Interpreter &vu, uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData, uint32_t dataSize, GS &gs, "
                                   "PS2Memory *memory,\n    uint32_t startPC, uint32_t top, uint32_t itop, uint32_t maxCycles";
        const std::string prologue =
            "    Vu1Native::Frame f = Vu1Native::beginExecute(vu, codeSize, vuData, dataSize, gs, memory, startPC, top, itop, "
            "maxCycles, " + fast + ");\n"
            "    Vu1Native::LeanCtx c;\n    Vu1Native::leanBegin(vu, c);\n"
            "    VU1State &s = Vu1Native::state(vu);\n    uint8_t *const mem = vuData;\n"
            "    uint64_t cyc = Vu1Native::cycle(vu);\n    uint32_t dn = 0;\n    int dr = 0;\n";

        auto deopt = [&](size_t id, const char *reason, const std::string &indent)
        {
            std::ostringstream s;
            s << indent << "{\n" << indent << "    dn = " << id << "u;\n" << indent << "    dr = Vu1Native::" << reason << ";\n"
              << indent << "    goto deopt;\n" << indent << "}\n";
            return s.str();
        };
        // The nodes in `ids` (closed under successors), as labelled code.
        auto emitNodes = [&](std::ostream &out, const std::vector<size_t> &ids) -> bool
        {
        for (size_t id : ids)
        {
            const Node &n = nodes[id];
            out << "N" << id << ":\n";
            if (n.unsupported)
            {
                out << deopt(id, "DeoptStep", "    ");
                continue;
            }
            const auto p = Vu1Native::decode(base, image.data(), n.pc);
            const LowerKind lk = lowerKind(p);
            const auto &e = n.entry.node;
            const uint8_t backup = e.backupReg;
            auto rv = [&](uint32_t r) -> std::string
            {
                if (r == 0u)
                    return "0";
                if (r == backup)
                    return "c.viBackup";
                return "s.vi[" + std::to_string(r) + "]";
            };
            if (loopHead[id])
                out << "    if (cyc >= f.budgetEnd || Vu1Native::stopRequested(vu))\n" << deopt(id, "DeoptStep", "    ");
            if (p.dBit)
                out << "    if (Vu1Native::dBitEnabled(vu))\n" << deopt(id, "DeoptHaltBit", "    ");
            if (p.tBit)
                out << "    if (Vu1Native::tBitEnabled(vu))\n" << deopt(id, "DeoptHaltBit", "    ");
            if (lk == LowerKind::Xgkick)
            {
                // The previous packet: done by now, or wait for it like run() (then everything has
                // drained: continue at the drained variant), or let the interpreter wait.
                out << "    if (c.kickPending)\n    {\n        if (cyc < c.kickEnd)\n        {\n";
                if (n.drainedVariant == id)
                    out << "            if (Vu1Native::leanStress())\n" << deopt(id, "DeoptXgkickBusy", "            ")
                        << "            cyc = c.kickEnd;\n            Vu1Native::leanSubmitKick(vu, c);\n";
                else if (n.drainedVariant != SIZE_MAX)
                {
                    out << "            if (!Vu1Native::leanStress() && c.kickEnd - cyc >= " << n.drainCycles << "u)\n            {\n";
                    if (e.fdivRel)
                        out << "                Vu1Native::leanCommitQ(vu, c, cyc + " << int(e.fdivRel) << "u);\n";
                    std::vector<std::pair<int, int>> efu;
                    for (int i = 0; i < 2; ++i)
                        if (e.efuRel[i])
                            efu.push_back({e.efuRel[i], i});
                    std::sort(efu.begin(), efu.end());
                    for (auto [rel, i] : efu)
                        out << "                s.p = c.p[" << i << "];\n";
                    out << "                cyc = c.kickEnd;\n                Vu1Native::leanSubmitKick(vu, c);\n"
                        << "                goto N" << n.drainedVariant << ";\n            }\n";
                    out << deopt(id, "DeoptXgkickBusy", "            ");
                }
                else
                    out << deopt(id, "DeoptXgkickBusy", "            ");
                out << "        }\n        else\n            Vu1Native::leanSubmitKick(vu, c);\n    }\n";
                out << "    if (!Vu1Native::leanKick(vu, c, mem, 0x4000u, s.vi[" << int(VIS(p.lower)) << "], cyc + " << n.stall << "u))\n"
                    << deopt(id, "DeoptKickPacket", "    ");
            }
            const uint32_t opHi = (p.lower >> 25) & 0x7Fu;
            if (lk == LowerKind::Branch && (opHi == 0x24u || opHi == 0x25u))
            {
                if (n.succ.size() != 1u || !n.succ[0].pending)
                    { std::cerr << "ps2_vu1_recomp: JR node without a single pending successor\n"; return false; }
                out << "    if (((static_cast<uint32_t>(static_cast<uint16_t>(" << rv(VIS(p.lower)) << ")) * 8u) & 0x3FFFu) != "
                    << hx(n.succ[0].target) << ")\n" << deopt(id, "DeoptDispatch", "    ");
            }
            if (lk == LowerKind::Store)
                out << "    if (c.kickPending && Vu1Native::leanKickConflict(c, Vu1Native::leanStoreAddr(s, " << hx(p.lower)
                    << ", 0x4000u), cyc + " << n.stall << "u, 0x4000u))\n" << deopt(id, "DeoptKickWrite", "    ");
            if (n.qPre)
                out << "    Vu1Native::leanCommitQ(vu, c, cyc + " << int(e.fdivRel) << "u);\n";
            if (n.pPre >= 0)
                out << "    s.p = c.p[" << n.pPre << "];\n";
            if (n.stall)
                out << "    cyc += " << n.stall << "u;\n";
            out << "    X" << std::hex << n.pc << std::dec << "(vu, s, mem, c, cyc);\n";
            if (n.efuSlot >= 0)
                out << "    c.p[" << n.efuSlot << "] = c.pNew;\n";
            std::string taken;
            if (lk == LowerKind::Branch)
            {
                const uint32_t it = VIT(p.lower), is = VIS(p.lower);
                if ((opHi == 0x21u || opHi == 0x25u) && it != 0u)
                    out << "    s.vi[" << it << "] = " << ((n.pc + 16u) / 8u) << ";\n";
                const std::string a = "static_cast<int16_t>(" + rv(is) + ")", b = "static_cast<int16_t>(" + rv(it) + ")";
                switch (opHi)
                {
                case 0x28: taken = a + " == " + b; break;
                case 0x29: taken = a + " != " + b; break;
                case 0x2C: taken = a + " < 0"; break;
                case 0x2D: taken = a + " > 0"; break;
                case 0x2E: taken = a + " <= 0"; break;
                case 0x2F: taken = a + " >= 0"; break;
                default: break;
                }
                if (!taken.empty() && rv(it) == "c.viBackup" && (opHi == 0x21u || opHi == 0x25u))
                    { std::cerr << "ps2_vu1_recomp: unexpected link/backup overlap\n"; return false; }
            }
            if (!taken.empty())
                out << "    {\n        const bool taken = " << taken << ";\n";
            const std::string in = taken.empty() ? "    " : "        ";
            if (n.qPost)
                out << in << "Vu1Native::leanCommitQ(vu, c, cyc + 1u);\n";
            if (n.pPost >= 0)
                out << in << "s.p = c.p[" << n.pPost << "];\n";
            out << in << "cyc += 1u;\n";
            if (n.ended)
            {
                const auto &x = n.exit.node;
                out << in << "s.pc = " << hx(x.pc) << ";\n" << in << "s.branchPending = " << ((x.flags & 1u) ? "true" : "false")
                    << ";\n" << in << "s.branchTarget = " << hx(x.branchTarget) << ";\n" << in << "s.branchDelay = "
                    << int(x.branchDelay) << ";\n" << in << "Vu1Native::leanEnd(vu, f, c, cyc, " << int(x.fdivRel) << ", "
                    << int(x.efuRel[0]) << ", " << int(x.efuRel[1]) << ");\n" << in << "return Vu1Native::Handled;\n";
                if (!taken.empty())
                    out << "    }\n";
                continue;
            }
            if (!taken.empty())
            {
                if (n.succ.size() != 2u)
                    { std::cerr << "ps2_vu1_recomp: conditional branch without two successors\n"; return false; }
                out << "        if (taken)\n            goto N" << n.succ[0].node << ";\n        goto N" << n.succ[1].node << ";\n    }\n";
            }
            else
            {
                if (n.succ.size() != 1u)
                    { std::cerr << "ps2_vu1_recomp: node without a single successor\n"; return false; }
                out << "    goto N" << n.succ[0].node << ";\n";
            }
        }
        out << "deopt:\n    return Vu1Native::leanDeopt(vu, f, c, cyc, vu1gen::kLeanNodes[dn], vu1gen::kLeanTiming, dr, vuCode);\n}\n\n";
        return true;
        };

        const char *checks = "    if (dataSize != 0x4000u || codeSize != 0x4000u)\n        return Vu1Native::NotHandled;\n";
        if (!split)
        {
            for (const auto &[pc, used] : pcsUsed)
                pairFn(out, pc);
            out << "}\n\n" << tables.str();
            out << "extern \"C\" __attribute__((visibility(\"default\"))) int rt_vu1_native_execute(\n    " << params << ")\n{\n"
                << "    switch (startPC)\n    {\n";
            for (uint32_t e : entries)
                out << "    case " << hx(e) << ":\n        break;\n";
            out << "    default:\n        return Vu1Native::NotHandled;\n    }\n" << checks << prologue << "    switch (startPC)\n    {\n";
            for (size_t i = 0; i < entries.size(); ++i)
                out << "    case " << hx(entries[i]) << ":\n        goto N" << entryNodes[i] << ";\n";
            out << "    default:\n        return Vu1Native::NotHandled;\n    }\n\n";
            std::vector<size_t> all(nodes.size());
            for (size_t i = 0; i < all.size(); ++i)
                all[i] = i;
            if (!emitNodes(out, all))
                return false;
        }
        else
        {
            // Each entry's nodes: everything reachable from it (successors, drained variants).
            const std::filesystem::path base(outPath);
            for (size_t i = 0; i < entries.size(); ++i)
            {
                std::vector<bool> seen(nodes.size(), false);
                std::vector<size_t> work{entryNodes[i]}, ids;
                seen[entryNodes[i]] = true;
                while (!work.empty())
                {
                    const size_t id = work.back();
                    work.pop_back();
                    ids.push_back(id);
                    auto visit = [&](size_t next)
                    {
                        if (next < nodes.size() && !seen[next])
                        {
                            seen[next] = true;
                            work.push_back(next);
                        }
                    };
                    for (const Edge &e : nodes[id].succ)
                        visit(e.node);
                    if (nodes[id].drainedVariant != SIZE_MAX)
                        visit(nodes[id].drainedVariant);
                }
                std::sort(ids.begin(), ids.end());
                std::set<uint32_t> pcs;
                for (size_t id : ids)
                    if (!nodes[id].unsupported)
                        pcs.insert(nodes[id].pc);
                const std::filesystem::path partPath =
                    base.parent_path() / (base.stem().string() + ".part" + std::to_string(i) + base.extension().string());
                std::ofstream part(partPath);
                part << "// Generated by ps2_vu1_recomp from the game's VU1 microcode. Do not edit; do not distribute.\n"
                     << "#include \"runtime/vu/ps2_vu1_native.h\"\n\nnamespace\n{\n";
                for (uint32_t pc : pcs)
                    pairFn(part, pc);
                part << "}\n\n" << tableDecls << "int rt_vu1_native_entry" << i << "(" << params << ")\n{\n" << prologue
                     << "    goto N" << entryNodes[i] << ";\n";
                if (!emitNodes(part, ids) || !part.good())
                    return false;
                std::cerr << "wrote " << partPath.string() << " (" << ids.size() << " nodes)\n";
            }
            out << "}\n\n" << tables.str();
            for (size_t i = 0; i < entries.size(); ++i)
                out << "int rt_vu1_native_entry" << i << "(" << params << ");\n";
            out << "\nextern \"C\" __attribute__((visibility(\"default\"))) int rt_vu1_native_execute(\n    " << params << ")\n{\n"
                << checks << "    switch (startPC)\n    {\n";
            for (size_t i = 0; i < entries.size(); ++i)
                out << "    case " << hx(entries[i]) << ":\n        return rt_vu1_native_entry" << i
                    << "(vu, vuCode, codeSize, vuData, dataSize, gs, memory, startPC, top, itop, maxCycles);\n";
            out << "    default:\n        return Vu1Native::NotHandled;\n    }\n}\n\n";
        }
        out << "// Final MAC/status may differ from the interpreter where no instruction can read them.\n"
            << "extern \"C\" __attribute__((visibility(\"default\"))) int rt_vu1_native_flags_exact()\n{\n    return "
            << (flagsLive.size() == pcsUsed.size() ? 1 : 0) << ";\n}\n\n"
            << "extern \"C\" __attribute__((visibility(\"default\"))) uint64_t rt_vu1_native_image_hash()\n{\n    return "
            << hex(Vu1Native::imageHash(image.data(), static_cast<uint32_t>(image.size()))) << "ull;\n}\n";
        return true;
    }
}

// Callable in-process too (ps2_vu1_recomp_lib, e.g. an app recompiling on a device that can't spawn it).
int ps2x_vu1_recomp_main(int argc, char **argv)
{
    std::string elfPath, imagePath, outPath;
    std::vector<uint32_t> entries;
    bool exact = false; // --exact: bit-exact float model (for vu1_replay); default is the fast SIMD path
    bool interpSteps = false; // --interp-steps: the previous generator (one VU1Interpreter::executePair per pair)
    bool split = false;       // --split: one function and file per entry (lean code; see emitLean)
    // PS2X_VU1_RECOMP_FLAGS (space-separated, e.g. "--interp-steps") adds options, for A/B builds
    // through the app's game builder.
    std::vector<std::string> args(argv + 1, argv + argc);
    if (const char *extra = std::getenv("PS2X_VU1_RECOMP_FLAGS"))
    {
        std::istringstream words(extra);
        for (std::string w; words >> w;)
            args.push_back(w);
    }
    for (size_t i = 0; i < args.size(); ++i)
    {
        const std::string a = args[i];
        auto next = [&]() { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--elf") elfPath = next();
        else if (a == "--image") imagePath = next();
        else if (a == "--out") outPath = next();
        else if (a == "--exact") exact = true;
        else if (a == "--interp-steps") interpSteps = true;
        else if (a == "--split") split = true;
        else if (a == "--entries")
        {
            std::stringstream list(next());
            for (std::string e; std::getline(list, e, ',');)
                entries.push_back(static_cast<uint32_t>(std::stoul(e, nullptr, 0)));
        }
        else
        {
            std::cerr << "unknown argument " << a << "\n";
            return 2;
        }
    }
    if (outPath.empty() || (elfPath.empty() == imagePath.empty()))
    {
        std::cerr << "usage: ps2_vu1_recomp (--elf ELF | --image BIN) --out FILE.cpp [--entries a,b,...] [--exact] [--interp-steps] [--split]\n";
        return 2;
    }

    std::vector<uint8_t> image, raw, data(0x4000, 0);
    std::string error;
    if (!elfPath.empty())
    {
        if (!readFile(elfPath, raw) || !imageFromElf(raw, image, error))
        {
            std::cerr << "ps2_vu1_recomp: " << (error.empty() ? "cannot read ELF" : error) << "\n";
            return 1;
        }
    }
    else if (!readFile(imagePath, image) || image.size() < kCodeSize)
    {
        std::cerr << "ps2_vu1_recomp: bad image\n";
        return 1;
    }
    image.resize(kCodeSize);

    VU1Interpreter base;
    Vu1Native::prepareExploration(base);

    // Registers JR/JALR jump through: their values steer control flow, so they are part of the key.
    uint16_t linkMask = 0;
    for (uint32_t pc = 0; pc < kCodeSize; pc += 8)
    {
        const auto p = Vu1Native::decode(base, image.data(), pc);
        const uint32_t op = p.lower >> 25;
        if (!p.iBit && (op == 0x24 || op == 0x25))
            linkMask |= static_cast<uint16_t>(1u << ((p.lower >> 11) & 0xF));
    }

    if (entries.empty())
    {
        // Default: the B-instruction jump table at the start of the image (one entry per 16 bytes).
        for (uint32_t pc = 0; pc < 0x400; pc += 16)
        {
            const auto p = Vu1Native::decode(base, image.data(), pc);
            if (p.iBit || (p.lower >> 25) != 0x20)
                break;
            entries.push_back(pc);
        }
    }
    std::cerr << entries.size() << " entries, link VI mask 0x" << std::hex << linkMask << std::dec << "\n";

    GS gs;
    std::vector<Node> nodes;
    std::map<std::pair<uint32_t, std::string>, size_t> index;
    std::deque<std::pair<size_t, std::unique_ptr<VU1Interpreter>>> work;

    auto intern = [&](const VU1Interpreter &vu) -> size_t
    {
        const uint32_t pc = Vu1Native::state(const_cast<VU1Interpreter &>(vu)).pc;
        auto key = std::make_pair(pc, Vu1Native::timingKey(vu, linkMask));
        auto it = index.find(key);
        if (it != index.end())
            return it->second;
        const size_t id = nodes.size();
        nodes.push_back(Node{pc});
        index.emplace(std::move(key), id);
        work.emplace_back(id, std::make_unique<VU1Interpreter>(vu));
        return id;
    };

    std::vector<size_t> entryNodes;
    for (uint32_t e : entries)
    {
        VU1Interpreter vu = base;
        Vu1Native::state(vu).pc = e;
        entryNodes.push_back(intern(vu));
    }

    while (!work.empty())
    {
        if (nodes.size() > kMaxNodes)
        {
            std::cerr << "ps2_vu1_recomp: state space too large (" << nodes.size() << " nodes)\n";
            return 1;
        }
        auto [id, vu] = std::move(work.front());
        work.pop_front();

        const uint32_t pc = nodes[id].pc;
        const auto pair = Vu1Native::decode(*vu, image.data(), pc);
        nodes[id].entry = Vu1Native::leanSnapshot(*vu);
        if (nodes[id].entry.fdivValid && nodes[id].entry.node.fdivRel == 0u)
            std::cerr << "warning: node " << id << " has an FDIV result due at entry\n";
        if (pair.upperUsage.reserved || pair.lowerUsage.reserved || pc + 8 > kCodeSize)
        {
            nodes[id].unsupported = true;
            continue;
        }
        if (!pair.iBit && pair.lowerUsage.pipeline == Vu1Native::Pipeline(7 /* PipelineXgkick */))
        {
            VU1Interpreter drained = *vu;
            Vu1Native::drain(drained);
            const size_t variant = intern(drained);
            nodes[id].drainedVariant = variant;
            nodes[id].drainCycles = Vu1Native::drainCycles(*vu);
        }
        Vu1Native::leanTag(*vu);
        nodes[id].stall = Vu1Native::resolveStall(*vu, pair);
        nodes[id].qPre = Vu1Native::leanQLanded(*vu);
        nodes[id].pPre = Vu1Native::leanPLanded(*vu);
        const bool efuBefore[2] = {Vu1Native::efuValid(*vu, 0), Vu1Native::efuValid(*vu, 1)};
        Vu1Native::leanUntag(*vu);
        const bool endedHere =
            Vu1Native::executeForExploration(*vu, pair, data.data(), static_cast<uint32_t>(data.size()), gs, kCodeSize);
        nodes[id].qPost = Vu1Native::leanQLanded(*vu);
        nodes[id].pPost = Vu1Native::leanPLanded(*vu);
        for (int i = 0; i < 2; ++i)
            if (!efuBefore[i] && Vu1Native::efuValid(*vu, i))
                nodes[id].efuSlot = i;
        if (endedHere)
        {
            nodes[id].ended = true;
            nodes[id].exit = Vu1Native::leanSnapshot(*vu);
            continue;
        }

        auto addSucc = [&](VU1Interpreter &next)
        {
            const auto &s = Vu1Native::state(next);
            const Edge edge{s.pc, s.branchPending, s.branchTarget, 0};
            const size_t to = intern(next);
            Edge e = edge;
            e.node = to;
            nodes[id].succ.push_back(e);
        };

        if (isConditionalBranch(pair))
        {
            const int32_t imm = static_cast<int32_t>(static_cast<int16_t>(static_cast<int32_t>(pair.lower << 21) >> 21));
            const uint32_t target = (pc + 8 + imm * 8) & (kCodeSize - 1);
            for (bool taken : {true, false})
            {
                VU1Interpreter copy = *vu;
                Vu1Native::forceBranch(copy, taken, target);
                addSucc(copy);
            }
        }
        else
            addSucc(*vu);
    }

    size_t unsupported = 0, ended = 0;
    for (auto &n : nodes)
    {
        unsupported += n.unsupported;
        ended += n.ended;
    }
    std::cerr << nodes.size() << " nodes (" << ended << " exits, " << unsupported << " unsupported)\n";

    // ---------------------------------------------------------------- emit C++
    std::ofstream out(outPath);
    out << "// Generated by ps2_vu1_recomp from the game's VU1 microcode. Do not edit; do not distribute.\n"
        << "#include \"runtime/vu/ps2_vu1_native.h\"\n\nnamespace\n{\n";
    std::map<uint32_t, bool> pcsUsed;
    for (auto &n : nodes)
        if (!n.unsupported)
            pcsUsed[n.pc] = true;
    for (auto &[pc, used] : pcsUsed)
    {
        (void)used;
        if (!interpSteps)
            break; // lean code doesn't use the decoded pairs
        out << "    constexpr Vu1Native::Pair kP" << std::hex << pc << std::dec << " = "
            << emitPair(Vu1Native::decode(base, image.data(), pc)) << ";\n";
    }
    // Flag liveness. MAC/status results matter only where a reader (FSxx, FMxx) can see them.
    // Flags become visible 4 cycles after issue, so a reader sees the newest FMAC issued at least
    // 4 cycles earlier: a pair's flags are dead if, on every path to a reader, another FMAC issues
    // 4 or more pairs (each pair >= 1 cycle) before that reader. Backward pass over (pc, distance
    // to reader capped at 4). Unsupported nodes continue in the interpreter, so they count as
    // readers. Clip flags are always kept.
    std::map<uint32_t, std::set<uint32_t>> pcSucc;
    std::set<uint32_t> readers, fmac;
    for (auto &n : nodes)
    {
        for (const Edge &e : n.succ)
            pcSucc[n.pc].insert(nodes[e.node].pc);
        if (n.drainedVariant != SIZE_MAX)
            pcSucc[n.pc].insert(nodes[n.drainedVariant].pc);
        if (n.unsupported)
        {
            readers.insert(n.pc);
            continue;
        }
        const auto pair = Vu1Native::decode(base, image.data(), n.pc);
        const uint32_t op = (pair.lower >> 25) & 0x7Fu;
        if (!pair.iBit && (op == 0x14u || op == 0x15u || op == 0x16u || op == 0x17u || op == 0x18u || op == 0x1Au || op == 0x1Bu))
            readers.insert(n.pc);
        // Upper ops that write MAC/status (the ones execUpper routes through applyFmacDest*).
        const uint32_t upperOp = pair.upper & 0x3Fu;
        const uint32_t flagOp = upperOp < 0x3Cu ? upperOp : ((pair.upper & 0x3u) | ((pair.upper >> 4) & 0x7Cu));
        if (flagOp <= 0x0Fu || (flagOp >= 0x18u && flagOp <= 0x1Cu) || flagOp == 0x1Eu ||
            (flagOp >= 0x20u && flagOp <= 0x2Au) || (flagOp >= 0x2Cu && flagOp <= 0x2Eu))
            fmac.insert(n.pc);
    }
    std::map<uint32_t, std::set<uint32_t>> pcPred;
    for (auto &[from, tos] : pcSucc)
        for (uint32_t to : tos)
            pcPred[to].insert(from);
    // open[pc] bit t: an unblocked path from pc reaches a reader in t pairs (t capped at 4).
    std::map<uint32_t, uint32_t> open;
    std::set<uint32_t> flagsLive;
    std::vector<std::pair<uint32_t, uint32_t>> liveWork;
    for (uint32_t r : readers)
    {
        open[r] |= 1u;
        liveWork.push_back({r, 0u});
        flagsLive.insert(r);
    }
    while (!liveWork.empty())
    {
        const auto [pc, t] = liveWork.back();
        liveWork.pop_back();
        const uint32_t t2 = std::min(t + 1u, 4u);
        for (uint32_t p : pcPred[pc])
        {
            flagsLive.insert(p); // p's own flags can reach the reader
            if (fmac.count(p) && t2 >= 4u)
                continue; // p's flags shadow everything issued before it on this path
            if ((open[p] & (1u << t2)) == 0u)
            {
                open[p] |= 1u << t2;
                liveWork.push_back({p, t2});
            }
        }
    }
    std::cerr << flagsLive.size() << " of " << pcsUsed.size() << " pairs need exact MAC/status flags\n";

    // Fast mode checks the cycle budget only where a loop can come back to (a node reached
    // from an equal or higher pc), so runaway programs still hand over to the interpreter.
    std::vector<bool> loopHead(nodes.size(), false);
    std::vector<bool> delaySlot(nodes.size(), false); // entered with a branch pending
    for (const Node &n : nodes)
        for (const Edge &e : n.succ)
        {
            if (nodes[e.node].pc <= n.pc)
                loopHead[e.node] = true;
            if (e.pending)
                delaySlot[e.node] = true;
        }

    if (!interpSteps)
    {
        if (!emitLean(out, nodes, entries, entryNodes, image, base, flagsLive, pcsUsed, loopHead, exact, split, outPath))
            return 1;
        std::cerr << "wrote " << outPath << " (lean)\n";
        return out.good() ? 0 : 1;
    }

    auto isPlain = [&](size_t id)
    {
        const Node &n = nodes[id];
        if (exact || n.unsupported || n.ended || delaySlot[id] || n.succ.size() != 1u || n.succ[0].pending)
            return false;
        const auto pair = Vu1Native::decode(base, image.data(), n.pc);
        return !pair.iBit && !pair.eBit && !pair.dBit && !pair.tBit &&
               pair.lowerUsage.pipeline != Vu1Native::Pipeline(6 /* PipelineBranch */) &&
               n.succ[0].pc == ((n.pc + 8u) & (kCodeSize - 1u));
    };
    std::set<uint32_t> plainPcs;
    for (size_t id = 0; id < nodes.size(); ++id)
        if (isPlain(id))
            plainPcs.insert(nodes[id].pc);

    // One specialized step per instruction pair: flatten inlines executePair/execUpper/execLower
    // with the constant pair, folding the decode away; nodes call these with their stall.
    for (auto &[pc, used] : pcsUsed)
    {
        (void)used;
        out << "    __attribute__((flatten, noinline)) Vu1Native::StepResult S" << std::hex << pc << std::dec
            << "(VU1Interpreter &vu, const Vu1Native::Frame &f, uint32_t stall)\n    {\n        return Vu1Native::step<"
            << (flagsLive.count(pc) ? "true" : "false") << ", " << (exact ? "false" : "true") << ", false>(vu, f, kP" << std::hex << pc
            << ", stall, 0x" << pc << std::dec << "u);\n    }\n";
        if (plainPcs.count(pc))
            out << "    __attribute__((flatten, noinline)) Vu1Native::StepResult P" << std::hex << pc << std::dec
                << "(VU1Interpreter &vu, const Vu1Native::Frame &f, uint32_t stall)\n    {\n        return Vu1Native::step<"
                << (flagsLive.count(pc) ? "true" : "false") << ", true, true>(vu, f, kP" << std::hex << pc << ", stall, 0x" << pc
                << std::dec << "u);\n    }\n";
    }
    out << "}\n\n"
        << "extern \"C\" __attribute__((visibility(\"default\"))) int rt_vu1_native_execute(\n"
        << "    VU1Interpreter &vu, uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory,\n"
        << "    uint32_t startPC, uint32_t top, uint32_t itop, uint32_t maxCycles)\n{\n"
        << "    switch (startPC)\n    {\n";
    for (size_t i = 0; i < entries.size(); ++i)
        out << "    case " << hex(entries[i]) << "u:\n        break;\n";
    out << "    default:\n        return Vu1Native::NotHandled;\n    }\n"
        << "    Vu1Native::Frame f = Vu1Native::beginExecute(vu, codeSize, vuData, dataSize, gs, memory, startPC, top, itop, maxCycles, "
        << (exact ? "false" : "true") << ");\n"
        << "    switch (startPC)\n    {\n";
    for (size_t i = 0; i < entries.size(); ++i)
        out << "    case " << hex(entries[i]) << "u:\n        goto N" << entryNodes[i] << ";\n";
    out << "    default:\n        goto deopt;\n    }\n\n";

    for (size_t id = 0; id < nodes.size(); ++id)
    {
        const Node &n = nodes[id];
        out << "N" << id << ":\n";
        const bool plain = isPlain(id);
        // Fast mode: a preceding plain pair leaves m_state.pc at its own pc, so deopts before this
        // pair set the right one first.
        const std::string setPc = exact ? std::string() : "        Vu1Native::setPc(vu, " + hex(n.pc) + "u);\n";
        if (!exact && loopHead[id] && !n.unsupported)
            out << "    if (Vu1Native::overBudget(vu, f))\n    {\n" << setPc << "        Vu1Native::noteDeopt(vu, Vu1Native::DeoptStep);\n        goto deopt;\n    }\n";
        if (n.unsupported)
        {
            out << "    goto deopt;\n";
            continue;
        }
        const auto pair = Vu1Native::decode(base, image.data(), n.pc);
        if (!pair.iBit && pair.lowerUsage.pipeline == Vu1Native::Pipeline(7 /* PipelineXgkick */))
        {
            out << "    if (Vu1Native::xgkickActive(vu))\n    {\n        Vu1Native::waitXgkick(vu);\n"
                << "        if (!Vu1Native::timingDrained(vu))\n        {\n" << setPc
                << "            Vu1Native::noteDeopt(vu, Vu1Native::DeoptXgkickBusy);\n            goto deopt;\n        }\n";
            if (n.drainedVariant != SIZE_MAX && n.drainedVariant != id)
                out << "        goto N" << n.drainedVariant << ";\n";
            out << "    }\n";
        }
        if (pair.dBit)
            out << "    if (Vu1Native::dBitEnabled(vu))\n    {\n" << setPc << "        Vu1Native::noteDeopt(vu, Vu1Native::DeoptHaltBit);\n        goto deopt;\n    }\n";
        if (pair.tBit)
            out << "    if (Vu1Native::tBitEnabled(vu))\n    {\n" << setPc << "        Vu1Native::noteDeopt(vu, Vu1Native::DeoptHaltBit);\n        goto deopt;\n    }\n";
        out << "    switch (" << (plain ? "P" : "S") << std::hex << n.pc << std::dec << "(vu, f, " << n.stall << "u))\n    {\n"
            << "    case Vu1Native::Continue:\n        break;\n"
            << "    case Vu1Native::Ended:\n        " << (n.ended ? "goto done;" : "Vu1Native::noteDeopt(vu, Vu1Native::DeoptExit);\n        goto deopt;") << "\n"
            << "    default:\n        Vu1Native::noteDeopt(vu, Vu1Native::DeoptStep);\n        goto deopt;\n    }\n";
        if (n.ended)
        {
            out << "    Vu1Native::noteDeopt(vu, Vu1Native::DeoptExit);\n    goto deopt;\n";
            continue;
        }
        // A plain pair (no branch, not in a delay slot, one successor) always continues at pc+8.
        if (!exact && n.succ.size() == 1u && !delaySlot[id] && !pair.iBit &&
            pair.lowerUsage.pipeline != Vu1Native::Pipeline(6 /* PipelineBranch */) && !n.succ[0].pending &&
            n.succ[0].pc == ((n.pc + 8u) & (kCodeSize - 1u)))
        {
            out << "    goto N" << n.succ[0].node << ";\n";
            continue;
        }
        out << "    {\n        const uint32_t pc = Vu1Native::pc(vu);\n        const bool pending = Vu1Native::branchPending(vu);\n";
        for (const Edge &e : n.succ)
        {
            out << "        if (pc == " << hex(e.pc) << "u && " << (e.pending ? "pending && Vu1Native::branchTarget(vu) == " + hex(e.target) + "u" : "!pending")
                << ")\n            goto N" << e.node << ";\n";
        }
        out << "        Vu1Native::noteDeopt(vu, Vu1Native::DeoptDispatch);\n        goto deopt;\n    }\n";
    }
    out << "done:\n    Vu1Native::endRun(vu, f);\n    return Vu1Native::Handled;\n"
        << "deopt:\n    return Vu1Native::continueInInterpreter(vu, f, vuCode);\n}\n\n"
        << "// Final MAC/status may differ from the interpreter where no instruction can read them.\n"
        << "extern \"C\" __attribute__((visibility(\"default\"))) int rt_vu1_native_flags_exact()\n{\n    return "
        << (flagsLive.size() == pcsUsed.size() ? 1 : 0) << ";\n}\n\n"
        << "extern \"C\" __attribute__((visibility(\"default\"))) uint64_t rt_vu1_native_image_hash()\n{\n    return "
        << hex(Vu1Native::imageHash(image.data(), kCodeSize)) << "ull;\n}\n";
    std::cerr << "wrote " << outPath << "\n";
    return 0;
}

#ifndef PS2X_VU1_RECOMP_NO_MAIN
int main(int argc, char **argv)
{
    return ps2x_vu1_recomp_main(argc, argv);
}
#endif
