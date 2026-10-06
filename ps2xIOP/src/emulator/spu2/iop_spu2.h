#pragma once

// Register-level SPU2 (the PS2's sound processor) for the IOP emulator: two cores with 24 ADPCM
// voices each, 2 MiB of sound RAM, PIO and DMA transfers (IOP DMA channels 4 and 7), AutoDMA
// streaming input, the IRQA interrupt (IOP IRQ 9), MMIX dry/wet routing, the reverb (effect) unit,
// and 48 kHz stereo output. Noise and pitch modulation are not emulated yet.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <vector>

namespace ps2x
{
    class StateArchive;
}

namespace ps2x::iop::detail
{
    class Spu2
    {
    public:
        static constexpr uint32_t kBase = 0x1F900000u;
        static constexpr uint32_t kRamHalfwords = 1u << 20; // 2 MiB
        static constexpr uint32_t kCyclesPerSample = 768u;  // 36.864 MHz IOP / 48 kHz

        using AudioSink = std::function<void(const int16_t *interleavedStereo, size_t frames)>;

        Spu2();
        void reset();
        // Save states: the whole state, written or read through `ar`.
        void serializeState(ps2x::StateArchive &ar);

        [[nodiscard]] uint16_t read16(uint32_t physicalAddress);
        void write16(uint32_t physicalAddress, uint16_t value);

        // A DMA on channel 4 (core 0) or 7 (core 1) started: `toSpu` copies `bytes` from IOP RAM
        // into sound RAM (or the AutoDMA stream), otherwise the other way. Returns the IOP cycles
        // until the transfer counts as complete.
        uint64_t dma(int core, uint8_t *iopRam, uint32_t bytes, bool toSpu);

        // The DMA on `core` finished (sets STATX bit 7).
        void dmaComplete(int core);

        // Produces output for `iopCycles` of elapsed IOP time.
        void advance(uint64_t iopCycles);

        // True once if the IRQA interrupt fired since the last call (IOP interrupt 9).
        [[nodiscard]] bool takeInterrupt();

        void setAudioSink(AudioSink sink) { m_sink = std::move(sink); }

    private:
        enum class Phase : uint8_t { Off, Attack, Decay, Sustain, Release };

        struct Voice
        {
            uint16_t volL = 0, volR = 0, pitch = 0, adsr1 = 0, adsr2 = 0;
            uint32_t ssa = 0, lsa = 0, nax = 0;
            bool lsaWritten = false;
            Phase phase = Phase::Off;
            int32_t level = 0;
            uint32_t envCounter = 0;
            uint32_t counter = 0;      // 12-bit pitch fraction
            int16_t decoded[28]{};
            uint32_t decodedPos = 28;  // 28 = need a new block
            int32_t hist1 = 0, hist2 = 0;
            int32_t prev = 0, cur = 0; // samples around the playback position (linear interpolation)
        };

        struct Core
        {
            Voice voices[24];
            uint32_t vmixl = 0, vmixr = 0, vmixel = 0, vmixer = 0, pmon = 0, non = 0;
            uint32_t endx = 0;
            uint32_t irqa = 0, tsa = 0;
            uint16_t attr = 0, mmix = 0, admas = 0, statx = 0;
            uint16_t mvolL = 0, mvolR = 0, avolL = 0, avolR = 0, bvolL = 0, bvolR = 0;
            std::deque<int16_t> admaL, admaR;
            bool irqFired = false;
            // Reverb: runs at 24 kHz on the wet bus, in sound RAM between ESA and EEA.
            uint32_t reverbX = 0;      // position in the effect area
            bool reverbPhase = false;  // second 48 kHz sample of the pair
            int32_t wetInL = 0, wetInR = 0;
            int32_t revPrevL = 0, revPrevR = 0, revCurL = 0, revCurR = 0;
        };

        static int32_t fixedVolume(uint16_t reg);
        uint16_t &raw(uint32_t offset) { return m_regs[(offset & 0x7FFu) >> 1]; }
        void keyOn(Core &core, uint32_t mask);
        void keyOff(Core &core, uint32_t mask);
        void decodeBlock(int coreIndex, Voice &voice, int voiceIndex);
        int32_t voiceSample(int coreIndex, Voice &voice, int voiceIndex);
        void stepEnvelope(Voice &voice);
        void checkIrq(int coreIndex, uint32_t address, uint32_t halfwords);
        void mixOneSample(int16_t &outL, int16_t &outR);
        // Mixes one core: dry and wet sends gated by MMIX, plus the reverb return (EVOL).
        void mixCore(int c, int32_t voiceDry[2], int32_t voiceWet[2], int32_t input[2], int32_t ext[2], int32_t out[2]);
        void reverb(int c, int32_t wetL, int32_t wetR, int32_t &outL, int32_t &outR);
        void reverbStep(int c, int32_t inL, int32_t inR, int32_t &outL, int32_t &outR);

        std::vector<uint16_t> m_ram;
        uint16_t m_regs[0x400]{};
        Core m_cores[2];
        uint16_t m_irqInfo = 0;
        bool m_interruptPending = false;
        uint64_t m_cycleCarry = 0;
        std::vector<int16_t> m_out;
        AudioSink m_sink;

        // RT_SPU2_TRACE=1: once per emulated second, log what the sound driver did.
        struct Trace
        {
            uint32_t keyOns = 0, keyOffs = 0, dmas = 0, admaDmas = 0, pioWrites = 0, irqs = 0;
            uint64_t samples = 0;
        } m_trace;
        bool m_traceEnabled = false;
        bool m_traceVerbose = false;
        void traceTick();
    };
}
