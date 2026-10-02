#pragma once

// Test and replay support, driven from guest vblanks so that runs are reproducible (use with
// RT_TIME=virtual):
//  - Pad input reaches the game only at vblank boundaries. Each vblank takes the next state from a
//    movie (RT_MOVIE_PLAY), a script (RT_INPUT_SCRIPT, times in seconds of guest time), the test
//    server, or the host's live input, in that order.
//  - RT_MOVIE_RECORD=<file> records the pad state the game saw, one line per change.
//  - RT_STATE_HASH=<file> logs hashes of guest memory every RT_STATE_HASH_INTERVAL vblanks
//    (default 60), to compare runs for determinism.
//  - RT_EXIT_AT_VBLANK=<n> stops the game after n vblanks.
//
// Movie format (text): "# roadtrip-movie 1" header, then "<vblank> <buttons hex> <lx> <ly> <rx> <ry>"
// whenever the state changes, and "# marker <vblank> <kind> <text>" lines. Buttons are active-low
// DualShock bits (0xFFFF = nothing pressed); sticks are 0..255 with 128 centred.

#include <cstdint>
#include <string>

class PS2Runtime;

namespace ps2_test
{
    struct PadState
    {
        uint16_t buttons = 0xFFFFu;
        uint8_t lx = 128, ly = 128, rx = 128, ry = 128;
        bool operator==(const PadState &) const = default;
    };

    // Host input from the keyboard/gamepad (any thread). Used when nothing else drives the pad.
    void setLiveInput(const PadState &state);
    // Adds a marker to the movie being recorded (any thread).
    void addMarker(const std::string &kind, const std::string &text = {});
    // True while a movie or script drives the pad.
    bool inputScripted();

    // Called on the EE thread at the start of every guest vblank.
    void onVblank(PS2Runtime &runtime, uint64_t vblank);
}
