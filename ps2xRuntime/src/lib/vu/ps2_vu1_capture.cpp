#include "runtime/ps2_vu1_capture.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace
{
    uint64_t envU64(const char *name, uint64_t fallback)
    {
        const char *v = std::getenv(name);
        return (v && *v) ? std::strtoull(v, nullptr, 10) : fallback;
    }

    void put(std::FILE *f, const void *p, size_t n) { std::fwrite(p, 1, n, f); }
    void put32(std::FILE *f, uint32_t v) { put(f, &v, sizeof(v)); }
    void put8(std::FILE *f, uint8_t v) { put(f, &v, sizeof(v)); }
}

std::unique_ptr<Vu1Capture> Vu1Capture::fromEnvironment()
{
    const char *path = std::getenv("RT_VU1_CAPTURE");
    if (!path || !*path)
        return nullptr;
    auto capture = std::unique_ptr<Vu1Capture>(new Vu1Capture());
    capture->m_file = std::fopen(path, "wb");
    if (!capture->m_file)
    {
        std::cerr << "[vu1-capture] cannot open " << path << std::endl;
        return nullptr;
    }
    capture->m_skip = envU64("RT_VU1_CAPTURE_SKIP", 0);
    capture->m_count = envU64("RT_VU1_CAPTURE_COUNT", 2000);
    capture->m_stride = std::max<uint64_t>(1, envU64("RT_VU1_CAPTURE_STRIDE", 1));
    std::cerr << "[vu1-capture] recording to " << path << " (skip " << capture->m_skip << ", count "
              << capture->m_count << ", stride " << capture->m_stride << ")" << std::endl;
    return capture;
}

Vu1Capture::~Vu1Capture()
{
    if (m_file)
        std::fclose(m_file);
}

bool Vu1Capture::begin(Kind kind, VU1Interpreter &vu, uint32_t startPC, uint32_t top, uint32_t itop,
                       const uint8_t *code, uint32_t codeSize, const uint8_t *data, uint32_t dataSize,
                       VU1Interpreter::XgkickSink forward)
{
    const uint64_t index = m_seen++;
    if (!m_file || m_recorded >= m_count || index < m_skip || ((index - m_skip) % m_stride) != 0)
        return false;

    if (!m_wroteCode)
    {
        put(m_file, "VU1CAP01", 8);
        put32(m_file, codeSize);
        put(m_file, code, codeSize);
        m_wroteCode = true;
    }

    put(m_file, "REC1", 4);
    put8(m_file, static_cast<uint8_t>(kind));
    put32(m_file, startPC);
    put32(m_file, top);
    put32(m_file, itop);
    put8(m_file, vu.state().dBitEnabled ? 1 : 0);
    put8(m_file, vu.state().tBitEnabled ? 1 : 0);
    put32(m_file, static_cast<uint32_t>(sizeof(VU1State)));
    put(m_file, &vu.state(), sizeof(VU1State));
    put32(m_file, dataSize);
    put(m_file, data, dataSize);

    m_packets.clear();
    vu.setXgkickSink([this, forward](const uint8_t *p, uint32_t n)
                     {
                         m_packets.emplace_back(p, p + n);
                         if (forward)
                             forward(p, n); });
    return true;
}

void Vu1Capture::end(VU1Interpreter &vu, const uint8_t *data, uint32_t dataSize)
{
    vu.setXgkickSink(nullptr);
    put8(m_file, vu.lastRunEnded() ? 1 : 0);
    put(m_file, &vu.state(), sizeof(VU1State));
    put(m_file, data, dataSize);
    put32(m_file, static_cast<uint32_t>(m_packets.size()));
    for (const auto &packet : m_packets)
    {
        put32(m_file, static_cast<uint32_t>(packet.size()));
        put(m_file, packet.data(), packet.size());
    }
    std::fflush(m_file); // the app exits with _Exit, so nothing may stay buffered
    if (++m_recorded == m_count)
    {
        std::cerr << "[vu1-capture] recorded " << m_recorded << " runs" << std::endl;
    }
}
