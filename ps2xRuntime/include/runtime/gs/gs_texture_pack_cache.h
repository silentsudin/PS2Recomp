#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Texture packs cached as ASTC 4x4 (a full mip chain per image, PS2 alpha), so a pack image loads
// with no PNG decoding (decoding a 4x pack's images when entering a building took a core and a
// half on the Thor and held frames up to 36 ms) at a quarter of RGBA8's memory. One file per
// image: <data>/textures/cache/<pack folder>/<16-digit hash>.astc, which also records the PNG's
// size and modification time (an edited image is encoded again). Built ahead of play by
// prepare() (the app's "Preparing texture pack" step), never during it.
namespace ps2x::gs::packcache
{
    struct Compressed
    {
        uint32_t width = 0, height = 0, levels = 0;
        std::vector<uint8_t> data;   // the levels, largest first, 16 bytes a 4x4 block
        std::vector<size_t> offsets; // where each level starts in `data`
    };

    // Whether this build can encode (the ASTC encoder is linked in).
    bool available();

    // The cached copy of `png` (its key as in the pack's file names), if it is there and current.
    bool load(const std::string &packDir, const std::string &png, uint64_t key, Compressed &out);

    struct Progress
    {
        std::atomic<uint32_t> done{0}, total{0}, failed{0};
        std::atomic<bool> cancel{false};
    };
    // The pack's images without a current cached copy.
    size_t missing(const std::string &packDir);
    // Encodes them on `threads` workers (0: all cores); returns when done or cancelled.
    void prepare(const std::string &packDir, Progress &progress, unsigned threads = 0);
}
