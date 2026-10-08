#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ps2x::iop
{
    // Native stand-ins for a module's own functions: a game's sound driver can spend most of the
    // IOP's interpreted instructions in a few routines. A function is registered before its module
    // loads, and bound when a module whose name contains `module` is loaded (or a state is loaded)
    // and the code at its base + `offset` hashes to `hash` (nativeCodeHash over `words` words,
    // leaving out those in `relocated`: they hold the load address). A call to it then runs `run`
    // instead, which reads its arguments and writes its results through the context and returns
    // how many guest instructions the routine would have executed. The calling thread then owes
    // them (IopCpuState::nativeDebt): it spends that many instruction slots, across time slices and
    // with interrupts taken between them, as the routine would have, so the IOP's timing, its
    // interrupts, DMA and sound come out as interpreted. RT_IOP_NATIVE=0 binds none.
    class NativeMemory
    {
    public:
        virtual ~NativeMemory() = default;
        virtual uint8_t read8(uint32_t address) const = 0;
        virtual uint16_t read16(uint32_t address) const = 0;
        virtual uint32_t read32(uint32_t address) const = 0;
        virtual void write8(uint32_t address, uint8_t value) = 0;
        virtual void write16(uint32_t address, uint16_t value) = 0;
        virtual void write32(uint32_t address, uint32_t value) = 0;
    };

    struct NativeContext
    {
        uint32_t *gpr = nullptr; // the 32 registers; the call returns to $ra afterwards
        uint32_t function = 0;   // the hooked function's address
        NativeMemory *memory = nullptr;
    };

    struct NativeFunction
    {
        std::string module; // part of the module's name, e.g. "SNDMOD"
        uint32_t offset = 0;
        uint32_t words = 0;
        uint64_t hash = 0;
        std::vector<uint32_t> relocated; // word indices left out of the hash
        std::function<uint64_t(NativeContext &)> run;
    };

    void registerNativeFunction(NativeFunction function);

    // FNV-1a (64-bit) over the words' little-endian bytes, skipping the relocated ones.
    uint64_t nativeCodeHash(const uint32_t *words, uint32_t count, const std::vector<uint32_t> &relocated);
}
