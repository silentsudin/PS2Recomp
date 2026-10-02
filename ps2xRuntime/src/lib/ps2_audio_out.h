#pragma once

#include <cstddef>
#include <cstdint>

// Host audio output for the emulated SPU2: 48 kHz stereo from the IOP is queued in a ring
// buffer that a raylib audio stream callback drains on the audio thread.
void ps2AudioOutStart();
void ps2AudioOutStop();
void ps2AudioOutSubmit(const int16_t *interleavedStereo, size_t frames);
