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
// whenever the state changes, "# marker <vblank> <kind> <text>" lines, and "# end <vblank>". Buttons are active-low
// DualShock bits (0xFFFF = nothing pressed); sticks are 0..255 with 128 centred.

#include <cstdint>
#include <string>
#include <vector>

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

    // Ends the movie being recorded ("# end <vblank>": after it, playback returns to live input;
    // without it the last state holds). Called on exit.
    void finishRecording();

    // Sound output (48 kHz stereo), for the test server's "audio" command.
    void onAudio(const int16_t *interleavedStereo, size_t frames);

    // Called on the EE thread at the start of every guest vblank.
    void onVblank(PS2Runtime &runtime, uint64_t vblank);

    // RT_TEST_SOCKET=<path>: a JSON-lines control server on a Unix socket for test drivers (one
    // client at a time). While a client is attached the game runs in lockstep: it parks at a vblank
    // until the client asks for more, so reads, writes, pad changes and frame grabs happen between
    // two exact vblanks. Commands (one JSON object per line, reply likewise):
    //   {"cmd":"run","vblanks":N}            run N vblanks, reply {"vblank":v} when parked again
    //   {"cmd":"pad","buttons":B,"lx":..}    pad state from the next vblank on (B active-low)
    //   {"cmd":"release_pad"}                back to movie/script/live input
    //   {"cmd":"read","space":"ee|spr|iop|vu1","addr":A,"len":L}  -> {"data":"<hex>"}
    //   {"cmd":"write","space":...,"addr":A,"data":"<hex>"}
    //   {"cmd":"frame","path":P}             the presented picture as raw RGBA -> {"width":W,"height":H}
    //   {"cmd":"stats"}                      vblank, presented frames, thread load
    //   {"cmd":"audio"}                      sound since the last query: frames, rms, peak, hash
    //   {"cmd":"quit"}
    void startServerIfRequested(PS2Runtime &runtime);

    // Render thread: a frame grab is waiting (latch the picture even if the vblank is unchanged),
    // and hands the latched picture (tightly packed RGBA) to it.
    bool frameCaptureRequested();
    void deliverFrameCapture(const std::vector<uint8_t> &rgba, uint32_t width, uint32_t height);
}
