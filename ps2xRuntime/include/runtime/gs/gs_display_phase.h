#pragma once

// Field phase of the buffer on display, tracked on the guest's own vblank timeline.
//
// Road Trip draws each buffer for one field (with that field's half-line offset) and flips it in
// during the previous field, so the buffer is meant for the field opposite to the vblank at which it
// first appears. PS2Memory::sampleDisplayAtVblank() records that at every guest vblank; the GPU GS
// scanout reads it, so the picture does not depend on when the host happens to present.

#include <atomic>
#include <cstdint>

namespace ps2x::gs
{
    // Defined once in the runtime (ps2_memory.cpp): an inline variable would be a second copy in the
    // recompiled game module on Windows.
    extern std::atomic<uint32_t> g_displayFieldPhase;
}
