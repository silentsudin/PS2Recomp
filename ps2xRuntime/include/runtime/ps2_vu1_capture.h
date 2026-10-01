#pragma once

// Records VU1 microprogram runs (inputs + outputs) so a VU1 implementation can be checked
// bit-for-bit against the interpreter offline (see ps2xRuntime/tools/vu1_replay).
//
// Enabled with RT_VU1_CAPTURE=<file>. Optional: RT_VU1_CAPTURE_SKIP (runs to skip first,
// default 0), RT_VU1_CAPTURE_COUNT (runs to record, default 2000), RT_VU1_CAPTURE_STRIDE
// (record every Nth run, default 1).
//
// File: "VU1CAP01", u32 codeSize, code bytes, then records:
//   "REC1", u8 kind (0 = MSCAL/execute, 1 = MSCNT/resume), u32 startPC, top, itop,
//   u8 dBitEnabled, tBitEnabled, u32 stateSize, VU1State (in), u32 dataSize, data (in),
//   u8 ended, VU1State (out), data (out), u32 packetCount, { u32 size, bytes }...

#include "runtime/ps2_vu1.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

class Vu1Capture
{
public:
    // Returns nullptr unless RT_VU1_CAPTURE is set.
    static std::unique_ptr<Vu1Capture> fromEnvironment();
    ~Vu1Capture();

    enum class Kind : uint8_t
    {
        Execute = 0,
        Resume = 1,
    };

    // Call before the run. Returns true if this run is being recorded; the caller must then
    // call end() after it. Installs an XGKICK sink that records packets and forwards them.
    bool begin(Kind kind, VU1Interpreter &vu, uint32_t startPC, uint32_t top, uint32_t itop,
               const uint8_t *code, uint32_t codeSize, const uint8_t *data, uint32_t dataSize,
               VU1Interpreter::XgkickSink forward);
    void end(VU1Interpreter &vu, const uint8_t *data, uint32_t dataSize);

private:
    std::FILE *m_file = nullptr;
    uint64_t m_seen = 0;
    uint64_t m_recorded = 0;
    uint64_t m_skip = 0;
    uint64_t m_count = 2000;
    uint64_t m_stride = 1;
    bool m_wroteCode = false;
    std::vector<std::vector<uint8_t>> m_packets;
};
