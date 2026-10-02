// SPU2 mixer and reverb checks: an impulse on the AutoDMA input, sent dry and wet through MMIX.
#include "emulator/spu2/iop_spu2.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
    using ps2x::iop::detail::Spu2;

    std::vector<int16_t> runImpulse(bool fxEnable, bool roomPreset)
    {
        Spu2 spu;
        std::vector<int16_t> out;
        spu.setAudioSink([&](const int16_t *s, size_t frames) { out.insert(out.end(), s, s + frames * 2); });
        auto w = [&](uint32_t off, uint16_t v) { spu.write16(Spu2::kBase + off, v); };
        const uint32_t core1 = 0x400u, common = 0x760u + 0x28u;
        w(core1 + 0x19A, static_cast<uint16_t>(0x8000u | (fxEnable ? 0x80u : 0u))); // ATTR
        w(common + 0x0, 0x3FFF); // MVOL
        w(common + 0x2, 0x3FFF);
        w(common + 0x4, 0x3FFF); // EVOL
        w(common + 0x6, 0x3FFF);
        w(common + 0xC, 0x3FFF); // BVOL (input volume)
        w(common + 0xE, 0x3FFF);
        w(core1 + 0x198, 0x0F0); // MMIX: input to dry and wet
        w(core1 + 0x2E0, 0x8);   // ESA = 0x80000
        w(core1 + 0x2E2, 0x0);
        w(core1 + 0x33C, 0x8);   // EEA = 0x8FFFF
        if (roomPreset)
        {
            // The PS1 "Room" preset; its addresses are in 8-byte units, SPU2's in halfwords.
            const uint16_t addresses[22] = {0x007D, 0x005B, 0x04D6, 0x0333, 0x03F0, 0x0227, 0x0374, 0x01EF,
                                            0x0334, 0x01B5, 0, 0, 0, 0, 0, 0, 0, 0, 0x01B4, 0x0136, 0x00B8, 0x005C};
            for (uint32_t i = 0; i < 22; ++i)
            {
                const uint32_t v = addresses[i] * 4u;
                w(core1 + 0x2E4 + i * 4, static_cast<uint16_t>(v >> 16));
                w(core1 + 0x2E6 + i * 4, static_cast<uint16_t>(v));
            }
            const uint16_t coefficients[10] = {0x6D80, 0x54B8, 0xBED0, 0, 0, 0xBA80, 0x5800, 0x5300, 0x7FFF, 0x7FFF};
            for (uint32_t i = 0; i < 10; ++i)
                w(common + 0x14 + i * 2, coefficients[i]);
        }
        w(core1 + 0x1B0, 2); // ADMAS: core 1 AutoDMA
        std::vector<uint8_t> blocks(0x400u * 48u, 0);
        const int16_t impulse = 20000;
        std::memcpy(blocks.data(), &impulse, 2);          // left
        std::memcpy(blocks.data() + 0x200u, &impulse, 2); // right
        spu.dma(1, blocks.data(), static_cast<uint32_t>(blocks.size()), true);
        spu.advance(uint64_t{256} * 48u * Spu2::kCyclesPerSample);
        return out;
    }

    double rms(const std::vector<int16_t> &s, size_t from, size_t to)
    {
        double e = 0;
        for (size_t i = from; i < to && i < s.size(); ++i)
            e += double(s[i]) * s[i];
        return std::sqrt(e / double(to - from));
    }

    bool check(bool ok, const char *what)
    {
        if (!ok)
            std::cerr << "FAIL: " << what << "\n";
        return ok;
    }
}

int main()
{
    const auto room = runImpulse(true, true);
    const auto fxOff = runImpulse(false, true);
    const auto noCoefficients = runImpulse(true, false);
    bool ok = true;
    ok &= check(room.size() == 2u * 256u * 48u, "48 blocks of AutoDMA input produce as many output samples");
    ok &= check(room[0] > 19000, "the dry impulse passes through");
    ok &= check(rms(room, 400, 8592) > 50.0, "the room preset leaves a reverb tail");
    ok &= check(rms(room, 16784, 24576) < rms(room, 4496, 12688), "the tail decays");
    ok &= check(rms(fxOff, 400, fxOff.size()) == 0.0, "no tail with the effect unit disabled");
    ok &= check(noCoefficients == fxOff, "zero coefficients give exactly the dry output");
    if (!ok)
        return 1;
    std::cout << "ps2xIOP SPU2 tests passed\n";
    return 0;
}
