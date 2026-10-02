#include "iop_spu2.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr int32_t kFilter[5][2] = {{0, 0}, {60, 0}, {115, -52}, {98, -55}, {122, -60}};

        int16_t clamp16(int32_t v)
        {
            return static_cast<int16_t>(std::clamp(v, -32768, 32767));
        }

        // Core register offsets (relative to the core's 0x400 window).
        constexpr uint32_t kPmon = 0x180, kNon = 0x184, kVmixl = 0x188, kVmixel = 0x18C, kVmixr = 0x190,
                           kVmixer = 0x194, kMmix = 0x198, kAttr = 0x19A, kIrqaHi = 0x19C, kIrqaLo = 0x19E,
                           kKonLo = 0x1A0, kKonHi = 0x1A2, kKoffLo = 0x1A4, kKoffHi = 0x1A6, kTsaHi = 0x1A8,
                           kTsaLo = 0x1AA, kData = 0x1AC, kAdmas = 0x1B0, kVoiceAddr = 0x1C0, kEndxLo = 0x340,
                           kEndxHi = 0x342, kStatx = 0x344;
        constexpr uint32_t kCommonVolumes = 0x760; // MVOL/EVOL/AVOL/BVOL per core, 0x28 apart
        constexpr uint32_t kIrqInfo = 0x7C2;
    }

    Spu2::Spu2() : m_ram(kRamHalfwords)
    {
        const char *env = std::getenv("RT_SPU2_TRACE");
        m_traceEnabled = env && (*env == '1' || *env == '2');
        m_traceVerbose = env && *env == '2';
        reset();
    }

    void Spu2::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint16_t{0});
        std::memset(m_regs, 0, sizeof(m_regs));
        for (auto &core : m_cores)
            core = Core{};
        m_irqInfo = 0;
        m_interruptPending = false;
        m_cycleCarry = 0;
        m_out.clear();
    }

    int32_t Spu2::fixedVolume(uint16_t reg)
    {
        if (reg & 0x8000u)
            return 0x7FFF; // sweep mode: not emulated, play at full volume
        return static_cast<int16_t>(static_cast<uint16_t>(reg << 1));
    }

    uint16_t Spu2::read16(uint32_t physicalAddress)
    {
        const uint32_t offset = (physicalAddress - kBase) & 0x7FFu;
        if (m_traceVerbose)
        {
            static int logged = 0;
            static uint32_t lastOffset = ~0u;
            const uint32_t l = offset & 0x3FFu;
            const bool interesting = offset >= 0x7C0u || (offset < 0x760u && (l == 0x19A || l == 0x344 || l == 0x1B0 || l == 0x1A8 || l == 0x1AA));
            if (interesting && offset != lastOffset && logged++ < 3000)
                std::fprintf(stderr, "[spu2] r %03x\n", offset);
            lastOffset = offset;
        }
        if (offset < 0x760u)
        {
            const int c = offset >= 0x400u ? 1 : 0;
            Core &core = m_cores[c];
            const uint32_t local = offset - static_cast<uint32_t>(c) * 0x400u;
            if (local < 0x180u)
            {
                Voice &v = core.voices[local >> 4];
                switch (local & 0xFu)
                {
                case 0xA: return static_cast<uint16_t>(v.level);      // ENVX
                case 0xC: return static_cast<uint16_t>(v.level);      // VOLXL (approximation)
                case 0xE: return static_cast<uint16_t>(v.level);      // VOLXR
                default: break;
                }
            }
            else if (local >= kVoiceAddr && local < kVoiceAddr + 24u * 0xCu)
            {
                const uint32_t rel = local - kVoiceAddr;
                Voice &v = core.voices[rel / 0xCu];
                switch (rel % 0xCu)
                {
                case 0x0: return static_cast<uint16_t>(v.ssa >> 16);
                case 0x2: return static_cast<uint16_t>(v.ssa);
                case 0x4: return static_cast<uint16_t>(v.lsa >> 16);
                case 0x6: return static_cast<uint16_t>(v.lsa);
                case 0x8: return static_cast<uint16_t>(v.nax >> 16);
                case 0xA: return static_cast<uint16_t>(v.nax);
                default: break;
                }
            }
            switch (local)
            {
            case kAttr: return core.attr;
            case kIrqaHi: return static_cast<uint16_t>(core.irqa >> 16);
            case kIrqaLo: return static_cast<uint16_t>(core.irqa);
            case kTsaHi: return static_cast<uint16_t>(core.tsa >> 16);
            case kTsaLo: return static_cast<uint16_t>(core.tsa);
            case kAdmas: return core.admas;
            case kEndxLo: return static_cast<uint16_t>(core.endx);
            case kEndxHi: return static_cast<uint16_t>(core.endx >> 16);
            case kStatx: return core.statx;
            case kData:
            {
                const uint16_t value = m_ram[core.tsa & (kRamHalfwords - 1u)];
                core.tsa = (core.tsa + 1u) & (kRamHalfwords - 1u);
                return value;
            }
            default: break;
            }
        }
        else if (offset == kIrqInfo)
            return m_irqInfo;
        return raw(offset);
    }

    void Spu2::write16(uint32_t physicalAddress, uint16_t value)
    {
        const uint32_t offset = (physicalAddress - kBase) & 0x7FFu;
        if (m_traceVerbose)
        {
            static int logged = 0;
            const uint32_t l = offset & 0x3FFu;
            const bool interesting = offset >= 0x7C0u || (offset < 0x760u && (l == 0x19A || l == 0x344 || l == 0x1B0 || l == 0x1A8 || l == 0x1AA || l == 0x19C || l == 0x19E));
            if (interesting && logged++ < 3000)
                std::fprintf(stderr, "[spu2] w %03x = %04x\n", offset, value);
        }
        raw(offset) = value;
        if (offset >= 0x760u)
        {
            if (offset == kIrqInfo)
                m_irqInfo = value;
            else if (offset < 0x760u + 2u * 0x28u)
            {
                const int c = offset >= 0x760u + 0x28u ? 1 : 0;
                Core &core = m_cores[c];
                switch ((offset - 0x760u) % 0x28u)
                {
                case 0x0: core.mvolL = value; break;
                case 0x2: core.mvolR = value; break;
                case 0x8: core.avolL = value; break;
                case 0xA: core.avolR = value; break;
                case 0xC: core.bvolL = value; break;
                case 0xE: core.bvolR = value; break;
                default: break;
                }
            }
            return;
        }

        const int c = offset >= 0x400u ? 1 : 0;
        Core &core = m_cores[c];
        const uint32_t local = offset - static_cast<uint32_t>(c) * 0x400u;
        if (local < 0x180u)
        {
            Voice &v = core.voices[local >> 4];
            switch (local & 0xFu)
            {
            case 0x0: v.volL = value; break;
            case 0x2: v.volR = value; break;
            case 0x4: v.pitch = value; break;
            case 0x6: v.adsr1 = value; break;
            case 0x8: v.adsr2 = value; break;
            case 0xA: v.level = static_cast<int16_t>(value); break;
            default: break;
            }
            return;
        }
        if (local >= kVoiceAddr && local < kVoiceAddr + 24u * 0xCu)
        {
            const uint32_t rel = local - kVoiceAddr;
            Voice &v = core.voices[rel / 0xCu];
            auto setHi = [&](uint32_t &reg) { reg = ((static_cast<uint32_t>(value) & 0xFu) << 16) | (reg & 0xFFFFu); };
            auto setLo = [&](uint32_t &reg) { reg = (reg & 0xF0000u) | value; };
            switch (rel % 0xCu)
            {
            case 0x0: setHi(v.ssa); break;
            case 0x2: setLo(v.ssa); break;
            case 0x4: setHi(v.lsa); v.lsaWritten = true; break;
            case 0x6: setLo(v.lsa); v.lsaWritten = true; break;
            case 0x8: setHi(v.nax); break;
            case 0xA: setLo(v.nax); break;
            default: break;
            }
            return;
        }
        auto lo16 = [&](uint32_t &reg) { reg = (reg & 0xFF0000u) | value; };
        auto hi8 = [&](uint32_t &reg) { reg = (reg & 0xFFFFu) | ((static_cast<uint32_t>(value) & 0xFFu) << 16); };
        switch (local)
        {
        case kPmon: lo16(core.pmon); break;
        case kPmon + 2: hi8(core.pmon); break;
        case kNon: lo16(core.non); break;
        case kNon + 2: hi8(core.non); break;
        case kVmixl: lo16(core.vmixl); break;
        case kVmixl + 2: hi8(core.vmixl); break;
        case kVmixel: lo16(core.vmixel); break;
        case kVmixel + 2: hi8(core.vmixel); break;
        case kVmixr: lo16(core.vmixr); break;
        case kVmixr + 2: hi8(core.vmixr); break;
        case kVmixer: lo16(core.vmixer); break;
        case kVmixer + 2: hi8(core.vmixer); break;
        case kMmix: core.mmix = value; break;
        case kAttr:
            // Rewriting the IRQ enable bit re-arms the interrupt.
            if ((value & 0x40u) && !(core.attr & 0x40u))
                core.irqFired = false;
            if (!(value & 0x40u))
                core.irqFired = false;
            core.attr = value;
            break;
        case kIrqaHi: core.irqa = ((static_cast<uint32_t>(value) & 0xFu) << 16) | (core.irqa & 0xFFFFu); break;
        case kIrqaLo: core.irqa = (core.irqa & 0xF0000u) | value; break;
        case kKonLo: keyOn(core, value); break;
        case kKonHi: keyOn(core, static_cast<uint32_t>(value & 0xFFu) << 16); break;
        case kKoffLo: keyOff(core, value); break;
        case kKoffHi: keyOff(core, static_cast<uint32_t>(value & 0xFFu) << 16); break;
        case kTsaHi: core.tsa = ((static_cast<uint32_t>(value) & 0xFu) << 16) | (core.tsa & 0xFFFFu); break;
        case kTsaLo: core.tsa = (core.tsa & 0xF0000u) | value; break;
        case kData:
            ++m_trace.pioWrites;
            checkIrq(c, core.tsa, 1u);
            m_ram[core.tsa & (kRamHalfwords - 1u)] = value;
            core.tsa = (core.tsa + 1u) & (kRamHalfwords - 1u);
            break;
        case kAdmas:
            core.admas = value;
            if (!value)
            {
                core.admaL.clear();
                core.admaR.clear();
            }
            break;
        case kEndxLo: core.endx &= ~static_cast<uint32_t>(value); break;
        case kEndxHi: core.endx &= ~(static_cast<uint32_t>(value & 0xFFu) << 16); break;
        case kStatx: core.statx = value; break;
        default: break;
        }
    }

    void Spu2::traceTick()
    {
        if (!m_traceEnabled || ++m_trace.samples % 48000u != 0u)
            return;
        int active[2] = {0, 0};
        for (int c = 0; c < 2; ++c)
            for (const Voice &v : m_cores[c].voices)
                active[c] += v.phase != Phase::Off;
        std::fprintf(stderr, "[spu2] kon=%u koff=%u dma=%u adma=%u pio=%u irq=%u active=%d/%d admas=%x/%x queued=%zu/%zu attr=%04x/%04x mvol=%04x/%04x\n",
                     m_trace.keyOns, m_trace.keyOffs, m_trace.dmas, m_trace.admaDmas, m_trace.pioWrites, m_trace.irqs,
                     active[0], active[1], m_cores[0].admas, m_cores[1].admas, m_cores[0].admaL.size(), m_cores[1].admaL.size(),
                     m_cores[0].attr, m_cores[1].attr, m_cores[1].mvolL, m_cores[1].mvolR);
        const uint64_t samples = m_trace.samples;
        m_trace = Trace{};
        m_trace.samples = samples;
    }

    void Spu2::keyOn(Core &core, uint32_t mask)
    {
        m_trace.keyOns += static_cast<uint32_t>(__builtin_popcount(mask));
        for (int i = 0; i < 24; ++i)
        {
            if (!(mask & (1u << i)))
                continue;
            Voice &v = core.voices[i];
            if (m_traceEnabled)
            {
                static int logged = 0;
                if (logged++ < 300)
                    std::fprintf(stderr, "[spu2] kon core=%d v=%d ssa=%05x hdr=%04x %04x %04x pitch=%04x adsr=%04x/%04x vol=%04x/%04x mix=%d%d\n",
                                 static_cast<int>(&core - m_cores), i, v.ssa, m_ram[v.ssa & (kRamHalfwords - 1u)],
                                 m_ram[(v.ssa + 1u) & (kRamHalfwords - 1u)], m_ram[(v.ssa + 2u) & (kRamHalfwords - 1u)],
                                 v.pitch, v.adsr1, v.adsr2, v.volL, v.volR, (core.vmixl >> i) & 1u, (core.vmixr >> i) & 1u);
            }
            v.nax = v.ssa;
            if (!v.lsaWritten)
                v.lsa = v.ssa;
            v.lsaWritten = false;
            v.phase = Phase::Attack;
            v.level = 0;
            v.envCounter = 0;
            v.counter = 0;
            v.decodedPos = 28;
            v.hist1 = v.hist2 = 0;
            v.prev = v.cur = 0;
            core.endx &= ~(1u << i);
        }
    }

    void Spu2::keyOff(Core &core, uint32_t mask)
    {
        m_trace.keyOffs += static_cast<uint32_t>(__builtin_popcount(mask));
        for (int i = 0; i < 24; ++i)
            if ((mask & (1u << i)) && core.voices[i].phase != Phase::Off)
            {
                core.voices[i].phase = Phase::Release;
                core.voices[i].envCounter = 0;
            }
    }

    void Spu2::checkIrq(int coreIndex, uint32_t address, uint32_t halfwords)
    {
        for (int c = 0; c < 2; ++c)
        {
            Core &core = m_cores[c];
            if (!(core.attr & 0x40u) || core.irqFired)
                continue;
            if (core.irqa >= address && core.irqa < address + halfwords)
            {
                core.irqFired = true;
                m_irqInfo |= static_cast<uint16_t>(4u << c);
                m_interruptPending = true;
                ++m_trace.irqs;
            }
        }
        (void)coreIndex;
    }

    void Spu2::decodeBlock(int coreIndex, Voice &v, int voiceIndex)
    {
        Core &core = m_cores[coreIndex];
        const uint32_t base = v.nax & (kRamHalfwords - 1u) & ~7u;
        checkIrq(coreIndex, base, 8u);
        const uint16_t header = m_ram[base];
        uint32_t shift = header & 0xFu;
        if (shift > 12u)
            shift = 9u; // hardware treats 13..15 like 9
        const uint32_t filter = std::min<uint32_t>((header >> 4) & 0x7u, 4u);
        const uint32_t flags = header >> 8;
        if (flags & 4u)
        {
            if (!v.lsaWritten)
                v.lsa = base;
        }
        for (int i = 0; i < 28; ++i)
        {
            const uint16_t word = m_ram[(base + 1u + static_cast<uint32_t>(i / 4)) & (kRamHalfwords - 1u)];
            const int32_t nibble = static_cast<int16_t>(static_cast<uint16_t>(((word >> ((i & 3) * 4)) & 0xFu) << 12));
            int32_t sample = nibble >> shift;
            sample += (v.hist1 * kFilter[filter][0] + v.hist2 * kFilter[filter][1]) >> 6;
            sample = clamp16(sample);
            v.decoded[i] = static_cast<int16_t>(sample);
            v.hist2 = v.hist1;
            v.hist1 = sample;
        }
        v.decodedPos = 0;
        v.nax = (base + 8u) & (kRamHalfwords - 1u);
        if (flags & 1u)
        {
            core.endx |= 1u << voiceIndex;
            v.nax = v.lsa;
            if (!(flags & 2u))
            {
                v.phase = Phase::Release; // end without repeat: silence
                v.level = 0;
            }
        }
    }

    void Spu2::stepEnvelope(Voice &v)
    {
        uint32_t rate = 0;
        bool decrease = false, exponential = false;
        switch (v.phase)
        {
        case Phase::Off:
            return;
        case Phase::Attack:
            rate = (v.adsr1 >> 8) & 0x7Fu;
            exponential = (v.adsr1 & 0x8000u) != 0;
            break;
        case Phase::Decay:
            rate = ((v.adsr1 >> 4) & 0xFu) << 2;
            decrease = exponential = true;
            break;
        case Phase::Sustain:
            rate = (v.adsr2 >> 6) & 0x7Fu;
            exponential = (v.adsr2 & 0x8000u) != 0;
            decrease = (v.adsr2 & 0x4000u) != 0;
            break;
        case Phase::Release:
            rate = (v.adsr2 & 0x1Fu) << 2;
            exponential = (v.adsr2 & 0x20u) != 0;
            decrease = true;
            break;
        }
        const int32_t shift = static_cast<int32_t>(rate >> 2);
        const int32_t stepIndex = static_cast<int32_t>(rate & 3u);
        int32_t step = decrease ? (-8 + stepIndex) : (7 - stepIndex);
        uint32_t cycles = 1u << std::max(0, shift - 11);
        step = step * (1 << std::max(0, 11 - shift));
        if (exponential && !decrease && v.level > 0x6000)
            cycles *= 4u;
        if (exponential && decrease)
            step = (step * v.level) >> 15;
        if (++v.envCounter >= cycles)
        {
            v.envCounter = 0;
            v.level = std::clamp(v.level + step, 0, 0x7FFF);
        }

        switch (v.phase)
        {
        case Phase::Attack:
            if (v.level >= 0x7FFF)
            {
                v.level = 0x7FFF;
                v.phase = Phase::Decay;
            }
            break;
        case Phase::Decay:
            if (v.level <= static_cast<int32_t>(((v.adsr1 & 0xFu) + 1u) * 0x800u))
                v.phase = Phase::Sustain;
            break;
        case Phase::Release:
            if (v.level <= 0)
            {
                v.level = 0;
                v.phase = Phase::Off;
            }
            break;
        default:
            break;
        }
    }

    int32_t Spu2::voiceSample(int coreIndex, Voice &v, int voiceIndex)
    {
        v.counter += std::min<uint32_t>(v.pitch, 0x3FFFu);
        while (v.counter >= 0x1000u)
        {
            v.counter -= 0x1000u;
            if (v.decodedPos >= 28u)
                decodeBlock(coreIndex, v, voiceIndex);
            v.prev = v.cur;
            v.cur = v.decoded[v.decodedPos++];
            if (v.phase == Phase::Off)
                return 0;
        }
        const int32_t frac = static_cast<int32_t>(v.counter);
        const int32_t sample = v.prev + (((v.cur - v.prev) * frac) >> 12);
        stepEnvelope(v);
        return (sample * v.level) >> 15;
    }

    void Spu2::mixOneSample(int16_t &outL, int16_t &outR)
    {
        int32_t coreOut[2][2] = {};
        for (int c = 0; c < 2; ++c)
        {
            Core &core = m_cores[c];
            int32_t l = 0, r = 0;
            for (int i = 0; i < 24; ++i)
            {
                Voice &v = core.voices[i];
                if (v.phase == Phase::Off)
                    continue;
                const int32_t s = voiceSample(c, v, i);
                if (core.vmixl & (1u << i))
                    l += (s * fixedVolume(v.volL)) >> 15;
                if (core.vmixr & (1u << i))
                    r += (s * fixedVolume(v.volR)) >> 15;
            }
            // AutoDMA input (streamed PCM), played at 48 kHz.
            if (core.admas && !core.admaL.empty() && !core.admaR.empty())
            {
                l += (core.admaL.front() * fixedVolume(core.bvolL)) >> 15;
                r += (core.admaR.front() * fixedVolume(core.bvolR)) >> 15;
                core.admaL.pop_front();
                core.admaR.pop_front();
            }
            coreOut[c][0] = l;
            coreOut[c][1] = r;
        }
        // Core 0 feeds core 1's external input; core 1's master volume drives the outputs.
        const int32_t l = coreOut[1][0] + ((clamp16(coreOut[0][0]) * fixedVolume(m_cores[0].mvolL)) >> 15);
        const int32_t r = coreOut[1][1] + ((clamp16(coreOut[0][1]) * fixedVolume(m_cores[0].mvolR)) >> 15);
        outL = clamp16((clamp16(l) * fixedVolume(m_cores[1].mvolL)) >> 15);
        outR = clamp16((clamp16(r) * fixedVolume(m_cores[1].mvolR)) >> 15);
    }

    void Spu2::advance(uint64_t iopCycles)
    {
        m_cycleCarry += iopCycles;
        while (m_cycleCarry >= kCyclesPerSample)
        {
            m_cycleCarry -= kCyclesPerSample;
            int16_t l, r;
            mixOneSample(l, r);
            traceTick();
            m_out.push_back(l);
            m_out.push_back(r);
            if (m_out.size() >= 512u)
            {
                if (m_sink)
                    m_sink(m_out.data(), m_out.size() / 2u);
                m_out.clear();
            }
        }
    }

    uint64_t Spu2::dma(int core, uint8_t *iopRam, uint32_t bytes, bool toSpu)
    {
        Core &c = m_cores[core & 1];
        const uint32_t halfwords = bytes / 2u;
        if (m_traceVerbose)
            std::fprintf(stderr, "[spu2] dma core=%d bytes=%u toSpu=%d tsa=%05x admas=%x\n", core, bytes, toSpu ? 1 : 0, c.tsa, c.admas);
        c.statx &= ~0x80u; // set again by dmaComplete(); LIBSD's DMA interrupt handler waits for it
        ++m_trace.dmas;
        if (toSpu && c.admas)
        {
            ++m_trace.admaDmas;
            // AutoDMA: blocks of 512 bytes left + 512 bytes right (256 samples each).
            for (uint32_t block = 0; block + 0x400u <= bytes; block += 0x400u)
            {
                const auto *left = reinterpret_cast<const int16_t *>(iopRam + block);
                const auto *right = reinterpret_cast<const int16_t *>(iopRam + block + 0x200u);
                c.admaL.insert(c.admaL.end(), left, left + 256);
                c.admaR.insert(c.admaR.end(), right, right + 256);
            }
            // Complete when the SPU is about to run dry, so the driver queues the next block in time.
            const uint64_t queued = c.admaL.size();
            return queued > 512u ? (queued - 512u) * kCyclesPerSample : 64u;
        }
        if (toSpu)
        {
            checkIrq(core, c.tsa, halfwords);
            for (uint32_t i = 0; i < halfwords; ++i)
            {
                uint16_t value;
                std::memcpy(&value, iopRam + i * 2u, 2);
                m_ram[(c.tsa + i) & (kRamHalfwords - 1u)] = value;
            }
        }
        else
        {
            for (uint32_t i = 0; i < halfwords; ++i)
            {
                const uint16_t value = m_ram[(c.tsa + i) & (kRamHalfwords - 1u)];
                std::memcpy(iopRam + i * 2u, &value, 2);
            }
        }
        c.tsa = (c.tsa + halfwords) & (kRamHalfwords - 1u);
        return std::max<uint64_t>(static_cast<uint64_t>(bytes / 4u) * 2u, 64u);
    }

    void Spu2::dmaComplete(int core)
    {
        // STATX bit 7 = transfer finished. LIBSD's DMA interrupt handler polls for it, then clears
        // ATTR's DMA mode bits itself.
        m_cores[core & 1].statx |= 0x80u;
    }

    bool Spu2::takeInterrupt()
    {
        const bool pending = m_interruptPending;
        m_interruptPending = false;
        return pending;
    }
}
