// Replays an RT_VU1_CAPTURE file and checks a VU1 implementation against it bit-for-bit.
//
//   vu1_replay <capture.vu1cap> [--verbose] [--bench N] [--split B,B,...]
//
// --reference-interp ignores the recorded results and checks against the interpreter run on the
// same inputs (for hand-made captures).
// --split runs each record again with a cycle budget of B (so a native module bails out
// mid-program and the interpreter takes over its state), resumes it in the interpreter until it
// ends, and checks that the result is still the recorded one.
//
// Each record's inputs (VU1 state, data memory, code image) are fed to the implementation
// under test; its final state, data memory and XGKICK packets must equal the recorded ones.

#include "runtime/gs/gs_frontend.h"
#include "runtime/vu/ps2_vu1_native.h"

#include <chrono>
#include <map>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Provided when a recompiled VU1 module is linked in (vu1_replay_native).
extern "C" int rt_vu1_native_execute(VU1Interpreter &vu, uint8_t *vuCode, uint32_t codeSize, uint8_t *vuData, uint32_t dataSize,
                                     GS &gs, PS2Memory *memory, uint32_t startPC, uint32_t top, uint32_t itop,
                                     uint32_t maxCycles) __attribute__((weak));
// 0 when the module skips MAC/status flags nothing can read; those final values aren't compared.
extern "C" int rt_vu1_native_flags_exact() __attribute__((weak));

namespace
{
    struct Reader
    {
        std::vector<uint8_t> buf;
        size_t pos = 0;
        bool ok = true;

        bool read(void *dst, size_t n)
        {
            if (pos + n > buf.size())
                return ok = false;
            std::memcpy(dst, buf.data() + pos, n);
            pos += n;
            return true;
        }
        uint32_t u32() { uint32_t v = 0; read(&v, 4); return v; }
        uint8_t u8() { uint8_t v = 0; read(&v, 1); return v; }
        bool atEnd() const { return pos >= buf.size(); }
    };

    struct Record
    {
        uint8_t kind = 0;
        uint32_t startPC = 0, top = 0, itop = 0;
        bool dBit = false, tBit = false;
        VU1State in{}, out{};
        std::vector<uint8_t> dataIn, dataOut;
        bool ended = false;
        std::vector<std::vector<uint8_t>> packets;
    };

    bool sameBits(const void *a, const void *b, size_t n) { return std::memcmp(a, b, n) == 0; }

    // Field-wise compare (the raw struct has padding and an absolute cycle counter).
    bool g_compareFlags = true;

    std::string diffState(const VU1State &want, const VU1State &got)
    {
        std::string out;
        auto note = [&](const std::string &s) { if (out.size() < 400) out += s + " "; };
        for (int r = 0; r < 32; ++r)
            for (int c = 0; c < 4; ++c)
                if (!sameBits(&want.vf[r][c], &got.vf[r][c], 4))
                    note("vf" + std::to_string(r) + "xyzw"[c]);
        for (int r = 0; r < 16; ++r)
            if (want.vi[r] != got.vi[r])
                note("vi" + std::to_string(r));
        for (int c = 0; c < 4; ++c)
            if (!sameBits(&want.acc[c], &got.acc[c], 4))
                note(std::string("acc") + "xyzw"[c]);
        if (!sameBits(&want.q, &got.q, 4)) note("q");
        if (!sameBits(&want.p, &got.p, 4)) note("p");
        if (!sameBits(&want.i, &got.i, 4)) note("i");
        if (want.r != got.r) note("r");
        if (want.pc != got.pc) note("pc");
        if (g_compareFlags && want.mac != got.mac) note("mac");
        if (want.clip != got.clip) note("clip");
        if (g_compareFlags && want.status != got.status) note("status");
        if (want.stoppedByD != got.stoppedByD || want.stoppedByT != got.stoppedByT) note("stop");
        return out;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: vu1_replay <capture.vu1cap> [--verbose]\n";
        return 2;
    }
    const bool verbose = argc > 2 && std::string(argv[2]) == "--verbose";

    Reader r;
    {
        std::ifstream in(argv[1], std::ios::binary);
        r.buf.assign(std::istreambuf_iterator<char>(in), {});
    }
    char magic[8];
    if (!r.read(magic, 8) || std::memcmp(magic, "VU1CAP01", 8) != 0)
    {
        std::cerr << "not a VU1 capture\n";
        return 2;
    }
    std::vector<uint8_t> code(r.u32());
    r.read(code.data(), code.size());

    g_compareFlags = !(rt_vu1_native_flags_exact && rt_vu1_native_flags_exact() == 0);
    if (!g_compareFlags)
        std::cout << "(final MAC/status not compared: module skips unobservable flags)\n";
    GS gs; // never touched: XGKICK packets go to the sink
    VU1Interpreter vu;
    uint32_t total = 0, failed = 0, skipped = 0, handled = 0, deopts = 0;
    std::map<std::pair<int, uint32_t>, uint32_t> deoptWhy;
    int bench = 0;
    std::vector<uint32_t> splits;
    for (int a = 2; a < argc; ++a)
    {
        if (std::string(argv[a]) == "--bench" && a + 1 < argc)
            bench = std::atoi(argv[a + 1]);
        if (std::string(argv[a]) == "--split" && a + 1 < argc)
            for (const char *p = argv[a + 1]; *p;)
            {
                char *end = nullptr;
                splits.push_back(static_cast<uint32_t>(std::strtoul(p, &end, 0)));
                p = *end ? end + 1 : end;
            }
    }
    uint32_t splitRuns = 0, splitFailed = 0, splitDeopts = 0;
    bool referenceInterp = false;
    for (int a = 2; a < argc; ++a)
        if (std::string(argv[a]) == "--reference-interp")
            referenceInterp = true;
    double benchSeconds = 0.0;
    uint64_t vuCycles = 0;

    while (!r.atEnd() && r.ok)
    {
        char tag[4];
        if (!r.read(tag, 4) || std::memcmp(tag, "REC1", 4) != 0)
            break;
        Record rec;
        rec.kind = r.u8();
        rec.startPC = r.u32();
        rec.top = r.u32();
        rec.itop = r.u32();
        rec.dBit = r.u8() != 0;
        rec.tBit = r.u8() != 0;
        const uint32_t stateSize = r.u32();
        if (stateSize != sizeof(VU1State))
        {
            std::cerr << "capture VU1State size " << stateSize << " != " << sizeof(VU1State) << "\n";
            return 2;
        }
        r.read(&rec.in, sizeof(VU1State));
        rec.dataIn.resize(r.u32());
        r.read(rec.dataIn.data(), rec.dataIn.size());
        rec.ended = r.u8() != 0;
        r.read(&rec.out, sizeof(VU1State));
        rec.dataOut.resize(rec.dataIn.size());
        r.read(rec.dataOut.data(), rec.dataOut.size());
        const uint32_t packetCount = r.u32();
        for (uint32_t p = 0; p < packetCount && r.ok; ++p)
        {
            std::vector<uint8_t> pk(r.u32());
            r.read(pk.data(), pk.size());
            rec.packets.push_back(std::move(pk));
        }
        if (!r.ok)
            break;
        ++total;

        // A resume continues interpreter-internal pipeline state we don't capture; only runs
        // that start from a clean MSCAL are exactly reproducible.
        if (rec.kind != 0)
        {
            ++skipped;
            continue;
        }

        if (referenceInterp)
        {
            VU1Interpreter ref;
            ref.state() = rec.in;
            ref.state().dBitEnabled = rec.dBit;
            ref.state().tBitEnabled = rec.tBit;
            rec.dataOut = rec.dataIn;
            rec.packets.clear();
            ref.setXgkickSink([&](const uint8_t *p, uint32_t n) { rec.packets.emplace_back(p, p + n); });
            ref.execute(code.data(), static_cast<uint32_t>(code.size()), rec.dataOut.data(),
                        static_cast<uint32_t>(rec.dataOut.size()), gs, nullptr, rec.startPC, rec.top, rec.itop, 65536);
            rec.out = ref.state();
            rec.ended = ref.lastRunEnded();
        }
        std::vector<uint8_t> data = rec.dataIn;
        std::vector<std::vector<uint8_t>> packets;
        vu.reset();
        vu.state() = rec.in;
        vu.state().dBitEnabled = rec.dBit;
        vu.state().tBitEnabled = rec.tBit;
        vu.setXgkickSink([&](const uint8_t *p, uint32_t n) { packets.emplace_back(p, p + n); });
        int native = 0; // 0 = interpreter only
        if (rt_vu1_native_execute)
        {
            native = rt_vu1_native_execute(vu, code.data(), static_cast<uint32_t>(code.size()), data.data(),
                                           static_cast<uint32_t>(data.size()), gs, nullptr, rec.startPC, rec.top,
                                           rec.itop, 65536);
            if (native == 2) // finished in the interpreter after a bail-out
            {
                ++deopts;
                ++deoptWhy[{Vu1Native::lastDeoptReason(), Vu1Native::lastDeoptPc()}];
            }
            else if (native == 1)
                ++handled;
        }
        if (native == 0)
            vu.execute(code.data(), static_cast<uint32_t>(code.size()), data.data(), static_cast<uint32_t>(data.size()),
                       gs, nullptr, rec.startPC, rec.top, rec.itop, 65536);

        vuCycles += vu.state().cycles; // reset() zeroed the cycle counter
        std::string problems = diffState(rec.out, vu.state());
        if (data != rec.dataOut)
        {
            size_t first = 0;
            while (first < data.size() && data[first] == rec.dataOut[first])
                ++first;
            problems += "data@" + std::to_string(first) + " ";
        }
        if (packets != rec.packets)
            problems += "xgkick(" + std::to_string(packets.size()) + " vs " + std::to_string(rec.packets.size()) + ") ";
        if (vu.lastRunEnded() != rec.ended)
            problems += "ended ";

        for (uint32_t budget : splits)
        {
            std::vector<uint8_t> d2 = rec.dataIn;
            std::vector<std::vector<uint8_t>> pk2;
            vu.reset();
            vu.state() = rec.in;
            vu.state().dBitEnabled = rec.dBit;
            vu.state().tBitEnabled = rec.tBit;
            vu.setXgkickSink([&](const uint8_t *p, uint32_t n) { pk2.emplace_back(p, p + n); });
            int r = rt_vu1_native_execute
                        ? rt_vu1_native_execute(vu, code.data(), static_cast<uint32_t>(code.size()), d2.data(),
                                                static_cast<uint32_t>(d2.size()), gs, nullptr, rec.startPC, rec.top,
                                                rec.itop, budget)
                        : 0;
            if (r == 0)
                vu.execute(code.data(), static_cast<uint32_t>(code.size()), d2.data(), static_cast<uint32_t>(d2.size()),
                           gs, nullptr, rec.startPC, rec.top, rec.itop, budget);
            splitDeopts += r == 2;
            for (int guard = 0; guard < 1000 && !vu.lastRunEnded(); ++guard)
                vu.resume(code.data(), static_cast<uint32_t>(code.size()), d2.data(), static_cast<uint32_t>(d2.size()), gs,
                          nullptr, rec.top, rec.itop, 65536);
            std::string sp = diffState(rec.out, vu.state());
            if (d2 != rec.dataOut)
                sp += "data ";
            if (pk2 != rec.packets)
                sp += "xgkick(" + std::to_string(pk2.size()) + " vs " + std::to_string(rec.packets.size()) + ") ";
            if (!vu.lastRunEnded())
                sp += "never ended ";
            ++splitRuns;
            if (!sp.empty())
            {
                ++splitFailed;
                if (splitFailed <= 10)
                    std::cout << "record " << (total - 1) << " pc=0x" << std::hex << rec.startPC << std::dec << " split at "
                              << budget << ": " << sp << "\n";
            }
        }

        if (bench > 0)
        {
            // Time only the VU1 call; resetting the inputs (a 16 KiB copy) is not part of it.
            std::vector<uint8_t> d2(rec.dataIn.size());
            vu.setXgkickSink([](const uint8_t *, uint32_t) {});
            for (int b = 0; b < bench; ++b)
            {
                std::memcpy(d2.data(), rec.dataIn.data(), d2.size());
                vu.state() = rec.in; // execute() resets the pipeline scheduler itself
                vu.state().dBitEnabled = rec.dBit;
                vu.state().tBitEnabled = rec.tBit;
                const auto t0 = std::chrono::steady_clock::now();
                int r = rt_vu1_native_execute
                            ? rt_vu1_native_execute(vu, code.data(), static_cast<uint32_t>(code.size()), d2.data(),
                                                    static_cast<uint32_t>(d2.size()), gs, nullptr, rec.startPC,
                                                    rec.top, rec.itop, 65536)
                            : 0;
                if (r == 0)
                    vu.execute(code.data(), static_cast<uint32_t>(code.size()), d2.data(),
                               static_cast<uint32_t>(d2.size()), gs, nullptr, rec.startPC, rec.top, rec.itop, 65536);
                benchSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }
        }

        if (!problems.empty())
        {
            ++failed;
            if (verbose || failed <= 10)
                std::cout << "record " << (total - 1) << " pc=0x" << std::hex << rec.startPC << std::dec
                          << ": " << problems << "\n";
        }
    }

    std::cout << "replayed " << (total - skipped) << " of " << total << " records (" << skipped
              << " MSCNT resumes skipped), " << failed << " mismatched";
    if (rt_vu1_native_execute)
        std::cout << "; native handled " << handled << ", deopted " << deopts;
    std::cout << "\n";
    for (auto &[why, n] : deoptWhy)
        std::cout << "  deopt reason " << why.first << " at pc 0x" << std::hex << why.second << std::dec << ": " << n << "\n";
    if (!splits.empty())
        std::cout << "split runs: " << splitRuns << ", " << splitFailed << " mismatched, " << splitDeopts << " bailed out natively\n";
    std::cout << "average " << (vuCycles / std::max<uint32_t>(1u, total - skipped)) << " VU cycles per run\n";
    if (bench > 0)
        std::cout << "bench: " << (benchSeconds * 1e6 / (double(total - skipped) * bench)) << " us per run\n";
    return failed == 0 && splitFailed == 0 && r.ok ? 0 : 1;
}
