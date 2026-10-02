// ps2_vu1_recomp: statically recompiles VU1 microcode to C++.
//
//   ps2_vu1_recomp --elf GAME.ELF --out vu1_native.cpp [--entries 0x0,0x10,...]
//   ps2_vu1_recomp --image code.bin --out vu1_native.cpp [--entries ...]
//
// The VU1 code image is assembled from the ELF's .DVP.ovlytab overlays (or read raw). Starting
// from each MSCAL entry with the clean pipeline state execute() produces, the CFG is explored
// with the interpreter's own timing model (forcing both outcomes of conditional branches) until
// every reachable (pc, timing state) node is known. Each node becomes one step of generated code:
// its precomputed hazard stall plus VU1Interpreter::executePair() on a constexpr decoded pair,
// followed by a dispatch to the statically known successor. See runtime/vu/ps2_vu1_native.h.

#include "runtime/vu/ps2_vu1_native.h"

#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <memory>
#include <sstream>
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
}

int main(int argc, char **argv)
{
    std::string elfPath, imagePath, outPath;
    std::vector<uint32_t> entries;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--elf") elfPath = next();
        else if (a == "--image") imagePath = next();
        else if (a == "--out") outPath = next();
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
        std::cerr << "usage: ps2_vu1_recomp (--elf ELF | --image BIN) --out FILE.cpp [--entries a,b,...]\n";
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
        }
        nodes[id].stall = Vu1Native::resolveStall(*vu, pair);
        if (Vu1Native::executeForExploration(*vu, pair, data.data(), static_cast<uint32_t>(data.size()), gs, kCodeSize))
        {
            nodes[id].ended = true;
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

    // One specialized step per instruction pair: flatten inlines executePair/execUpper/execLower
    // with the constant pair, folding the decode away; nodes call these with their stall.
    for (auto &[pc, used] : pcsUsed)
    {
        (void)used;
        out << "    __attribute__((flatten, noinline)) Vu1Native::StepResult S" << std::hex << pc << std::dec
            << "(VU1Interpreter &vu, const Vu1Native::Frame &f, uint32_t stall)\n    {\n        return Vu1Native::step<"
            << (flagsLive.count(pc) ? "true" : "false") << ">(vu, f, kP" << std::hex << pc << std::dec << ", stall);\n    }\n";
    }
    out << "}\n\n"
        << "extern \"C\" __attribute__((visibility(\"default\"))) int rt_vu1_native_execute(\n"
        << "    VU1Interpreter &vu, uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory,\n"
        << "    uint32_t startPC, uint32_t top, uint32_t itop, uint32_t maxCycles)\n{\n"
        << "    switch (startPC)\n    {\n";
    for (size_t i = 0; i < entries.size(); ++i)
        out << "    case " << hex(entries[i]) << "u:\n        break;\n";
    out << "    default:\n        return Vu1Native::NotHandled;\n    }\n"
        << "    Vu1Native::Frame f = Vu1Native::beginExecute(vu, codeSize, vuData, dataSize, gs, memory, startPC, top, itop, maxCycles);\n"
        << "    switch (startPC)\n    {\n";
    for (size_t i = 0; i < entries.size(); ++i)
        out << "    case " << hex(entries[i]) << "u:\n        goto N" << entryNodes[i] << ";\n";
    out << "    default:\n        goto deopt;\n    }\n\n";

    for (size_t id = 0; id < nodes.size(); ++id)
    {
        const Node &n = nodes[id];
        out << "N" << id << ":\n";
        if (n.unsupported)
        {
            out << "    goto deopt;\n";
            continue;
        }
        const auto pair = Vu1Native::decode(base, image.data(), n.pc);
        if (!pair.iBit && pair.lowerUsage.pipeline == Vu1Native::Pipeline(7 /* PipelineXgkick */))
        {
            out << "    if (Vu1Native::xgkickActive(vu))\n    {\n        Vu1Native::waitXgkick(vu);\n"
                << "        if (!Vu1Native::timingDrained(vu))\n        {\n"
                << "            Vu1Native::noteDeopt(vu, Vu1Native::DeoptXgkickBusy);\n            goto deopt;\n        }\n";
            if (n.drainedVariant != SIZE_MAX && n.drainedVariant != id)
                out << "        goto N" << n.drainedVariant << ";\n";
            out << "    }\n";
        }
        if (pair.dBit)
            out << "    if (Vu1Native::dBitEnabled(vu))\n    {\n        Vu1Native::noteDeopt(vu, Vu1Native::DeoptHaltBit);\n        goto deopt;\n    }\n";
        if (pair.tBit)
            out << "    if (Vu1Native::tBitEnabled(vu))\n    {\n        Vu1Native::noteDeopt(vu, Vu1Native::DeoptHaltBit);\n        goto deopt;\n    }\n";
        out << "    switch (S" << std::hex << n.pc << std::dec << "(vu, f, " << n.stall << "u))\n    {\n"
            << "    case Vu1Native::Continue:\n        break;\n"
            << "    case Vu1Native::Ended:\n        " << (n.ended ? "goto done;" : "Vu1Native::noteDeopt(vu, Vu1Native::DeoptExit);\n        goto deopt;") << "\n"
            << "    default:\n        Vu1Native::noteDeopt(vu, Vu1Native::DeoptStep);\n        goto deopt;\n    }\n";
        if (n.ended)
        {
            out << "    Vu1Native::noteDeopt(vu, Vu1Native::DeoptExit);\n    goto deopt;\n";
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
