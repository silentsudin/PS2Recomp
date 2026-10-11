#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Texture packs cached as a GPU block format (a full mip chain per image, PS2 alpha), so a pack
// image loads with no PNG decoding (decoding a 4x pack's images when entering a building took a
// core and a half on the Thor and held frames up to 36 ms) at a quarter of RGBA8's memory. ASTC 4x4
// on Apple and Android, BC7 elsewhere (desktop GPUs don't sample ASTC); both 16 bytes a 4x4 block.
// One file per image and format: <data>/textures/cache/<pack folder>/<16-digit hash>.astc|.bc7,
// which also records the PNG's size and modification time (an edited image is encoded again).
// Built ahead of play by prepare() (the app's "Getting <pack> ready" step), never during it.
namespace ps2x::gs::packcache
{
    enum class Format : uint8_t
    {
        None = 0,
        Astc = 1, // ASTC 4x4, UNORM
        Bc7 = 2,  // BC7, UNORM
    };
    const char *formatName(Format f); // "ASTC", "BC7", "none"

    // The format this platform caches in: ASTC on Apple and Android, BC7 elsewhere.
    // RT_TEXTURE_CACHE_FORMAT=astc|bc7|none overrides it (RT_TEXTURE_ASTC=0 is none). None when this
    // build has no encoder for it.
    Format wanted();
    // Set by the hardware GS once its device exists: which formats the device samples.
    void setDeviceSamples(bool astc, bool bc7);
    // The format in use: wanted(), if the device samples it; else None (and None until a hardware
    // GS has asked its device). With None packs load from their PNGs and nothing is converted.
    Format active();

    struct Compressed
    {
        Format format = Format::None;
        uint32_t width = 0, height = 0, levels = 0;
        std::vector<uint8_t> data;   // the levels, largest first, 16 bytes a 4x4 block
        std::vector<size_t> offsets; // where each level starts in `data`
    };

    // Whether packs are cached here (active() is a format).
    bool available();

    // A folder the app can always write (its private data): a pack whose usual cache folder can't
    // be written caches in <dir>/<pack folder name> instead (set before the first pack is used).
    void setPrivateCacheRoot(const std::string &dir);

    // The cached copy of `png` (its key as in the pack's file names) in the active format, if it is
    // there and current.
    bool load(const std::string &packDir, const std::string &png, uint64_t key, Compressed &out);

    struct Progress
    {
        std::atomic<uint32_t> done{0}, total{0}, failed{0};
        // Why images failed (each also counts in `failed`): the PNG didn't load, the encoder
        // refused it, the cache file couldn't be written, or not moved into place.
        std::atomic<uint32_t> failedLoad{0}, failedEncode{0}, failedWrite{0}, failedRename{0};
        std::atomic<bool> cancel{false};
    };
    // The pack's images without a current cached copy in the active format.
    size_t missing(const std::string &packDir);
    // Encodes them on `threads` workers (0: all cores); returns when done or cancelled.
    void prepare(const std::string &packDir, Progress &progress, unsigned threads = 0);
}
