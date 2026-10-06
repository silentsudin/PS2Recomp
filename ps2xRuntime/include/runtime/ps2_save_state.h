#pragma once

// Save states.
//
// A snapshot is taken only at the top of EeScheduler::run() (after the vblank's events), when
// EeScheduler::canSnapshot(true) says nothing host-side is in flight: then every bit of guest
// state is in RDRAM, the scheduler and the host-side emulation objects, and each of those has a
// serializeState(ps2x::StateArchive &) (ps2x/state_archive.h). A request waits at loop tops until
// one is savable. Nothing here runs unless asked for (test socket, the app's menu).
//
// States belong to the game (the boot ELF's CRC-32), not to a build of the app: they hold guest
// state (memory, registers, guest PCs: recompiled code resumes at guest addresses) and host
// emulation state written field by field. Every chunk has its own version; a loader keeps reading
// older versions where a change is additive.
//
// File (`<data>/states/slot<N>.rtstate`, written on a worker thread to `<path>.tmp<n>`, then
// renamed; a crash leaves at most a stray temp file, never a half-written slot):
//   "RTSTATE\x1A", u32 format version, u32 header size, u32 game CRC, u32 compression (1 = zstd,
//   0 = stored), u64 vblank, i64 saved (unix seconds), u64 raw size, u64 payload size,
//   u32 n + n x {string key, string value}           metadata (place, GS backend, app version...)
//   u32 n + n x {u32 fourcc, u16 version, u64 size, u64 hash, u64 digest}   the chunks
//   u64 hash of the header so far
//   payload: the chunks ({u32 fourcc, u16 version, u64 size, data} each, covering it whole), one
//   zstd frame with its checksum. Strings are u32 length + bytes; everything little-endian.
// Hashes are ps2x::stateHash (state_archive.h).
// A chunk's `digest` is its hash as RT_STATE_HASH_FULL / state_hash computes it (no host-relative
// values): after a load, the machine's digest must match it chunk by chunk.
//
// Loading reads, decompresses and checks the whole file first (magic, format version, header
// hash, game, sizes, zstd checksum, every chunk's hash, that this build knows every chunk and its
// version) and refuses with a reason without touching the machine. The EE thread then keeps a
// copy of the running machine, loads the state, checks that every guest thread resumes at an
// address this build has code for, and puts the copy back if anything failed.
//
// RT_SAVE_STATE_SKIP=<fourcc>,... leaves those chunks out of a load (e.g. IOPS,SPU2: keep the
// IOP and the sound chip running on from where they are, to measure what restoring them takes).
// RT_STATE_HASH_FULL=1 makes RT_STATE_HASH log one hash per chunk instead of the memory hashes.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

class PS2Runtime;
namespace ps2x
{
    class StateArchive;
}

namespace ps2_save_state
{
    constexpr uint32_t kFileFormatVersion = 1;

    enum class Op : int
    {
        None = 0,
        Save = 1,
        Load = 2,
    };

    namespace detail
    {
        inline std::atomic<int> g_request{0};
    }

    // Cheap check for the scheduler's loop top.
    inline bool requested() noexcept { return detail::g_request.load(std::memory_order_relaxed) != 0; }

    // Any thread: save into / load from the in-memory slot at the next savable loop top.
    void request(Op op);
    void cancel();

    // Any thread: save to `path` at the next savable loop top. The EE thread copies the state
    // (milliseconds); a worker compresses and writes it (waitForWrites, lastWrite).
    // `compress` false stores the chunks as they are (diagnostics, tests).
    void requestSaveFile(const std::string &path, bool compress = true);
    // Any thread: reads and checks `path` here; false (with the reason, nothing changed) if it is
    // refused, else the EE thread loads it at the next savable loop top (lastResult).
    // `resavePath`: right after a successful load, the machine is saved there again (tests: a
    // state loaded and saved again is the same bytes, except EETM's wall-clock deadlines).
    bool requestLoadFile(const std::string &path, std::string &error, const std::string &resavePath = {});
    // Tests: the next load fails its last check (after the machine was overwritten), so the
    // machine is put back as it was.
    void failNextLoadForTest();

    // Waits until no state file is being written; false on timeout.
    bool waitForWrites(int timeoutMs = 30000);

    struct Result
    {
        uint64_t sequence = 0; // bumps when a request finishes
        Op op = Op::None;
        bool ok = false;
        uint64_t vblank = 0;      // the guest vblank the state belongs to
        uint64_t loopTops = 0;    // loop tops tried (blocked ones + the one that worked)
        size_t bytes = 0;
        double ms = 0.0;
        uint32_t orderMismatches = 0; // unordered maps that came back in another order
        bool reserializedEqual = false; // load: the digest right after matches the one at the save
        bool rolledBack = false;        // load failed and the machine was put back as it was
        std::string error;
        std::string blockers; // why loop tops were refused
        std::string skipped;  // chunks left out of a load
        std::string path;     // file saves/loads
    };
    Result lastResult();

    struct WriteResult
    {
        uint64_t sequence = 0;
        bool ok = false;
        std::string path;
        std::string error;
        uint64_t rawBytes = 0;
        uint64_t fileBytes = 0;
        double ms = 0.0; // compress + write
    };
    WriteResult lastWrite();

    // A state file's header.
    struct FileChunk
    {
        uint32_t id = 0;
        uint16_t version = 0;
        uint64_t size = 0;
        uint64_t hash = 0;   // of the chunk's bytes as stored
        uint64_t digest = 0; // of the chunk as the digest serialization writes it
    };
    struct FileInfo
    {
        uint32_t formatVersion = 0;
        uint32_t gameCrc = 0;
        uint64_t vblank = 0;
        int64_t savedUnixTime = 0;
        uint32_t compression = 1;
        uint64_t rawSize = 0, payloadSize = 0, fileSize = 0;
        std::vector<std::pair<std::string, std::string>> metadata;
        std::vector<FileChunk> chunks;
        [[nodiscard]] std::string meta(const std::string &key) const;
    };
    // The header only (cheap: for listing slots). False with a reason if it isn't a usable state
    // file (the game and chunk checks happen when loading).
    bool readStateFileInfo(const std::string &path, FileInfo &info, std::string &error);
    // Low level (SaveStateFile.cpp).
    bool writeStateFile(const std::string &path, FileInfo &info, const std::vector<uint8_t> &raw, std::string &error,
                        bool compress = true);
    bool readStateFile(const std::string &path, FileInfo &info, std::vector<uint8_t> &raw, std::string &error);

    // The game a state belongs to (the boot ELF's CRC-32): loading checks it.
    void setGameIdentity(uint32_t elfCrc32);
    uint32_t gameIdentity();

    // Key/value pairs the app adds to a saved file's header (place, time played...). Called on
    // the EE thread when a state is taken.
    using MetadataProvider = std::function<std::vector<std::pair<std::string, std::string>>()>;
    void setMetadataProvider(MetadataProvider provider);

    // EE thread, at the scheduler's loop top.
    void service(PS2Runtime &runtime);

    // Per-chunk hashes of the state as it is now (EE thread; at a vblank or loop top). For
    // comparing runs: needs no savable point (pending host work is hashed as it stands).
    std::vector<std::pair<std::string, uint64_t>> chunkHashes(PS2Runtime &runtime);

    // One chunk of the digest serialization as it is now (for diffing two runs), empty if unknown.
    std::vector<uint8_t> chunkBytes(PS2Runtime &runtime, const std::string &name);

    // Chunks of the in-memory slot (name, size, hash), empty without one.
    std::vector<std::pair<std::string, std::pair<size_t, uint64_t>>> slotChunks();

    // Extra state from the app (game hooks), saved after the runtime's in registration order.
    // `version` is the section's own version (bump it when its fields change).
    using Section = std::function<void(ps2x::StateArchive &)>;
    void registerSection(uint32_t fourcc, Section section, uint16_t version = 1);
}
