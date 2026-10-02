#include "runtime/ps2_test_harness.h"

#include "ps2_runtime.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <vector>

namespace ps2_stubs
{
    void setPadOverrideState(uint16_t buttons, uint8_t lx, uint8_t ly, uint8_t rx, uint8_t ry);
}

namespace ps2_test
{
    namespace
    {
        struct ButtonName
        {
            const char *name;
            uint16_t mask;
        };
        constexpr ButtonName kButtons[] = {
            {"select", 0x0001}, {"l3", 0x0002}, {"r3", 0x0004}, {"start", 0x0008}, {"up", 0x0010},
            {"right", 0x0020}, {"down", 0x0040}, {"left", 0x0080}, {"l2", 0x0100}, {"r2", 0x0200},
            {"l1", 0x0400}, {"r1", 0x0800}, {"triangle", 0x1000}, {"circle", 0x2000}, {"cross", 0x4000},
            {"square", 0x8000},
        };
        constexpr double kVblanksPerSecond = 60000.0 / 1001.0;

        struct MovieEntry
        {
            uint64_t vblank;
            PadState state;
        };

        struct ScriptPress
        {
            uint64_t from, until;
            uint16_t mask;
        };

        std::mutex g_mutex;
        PadState g_live;
        bool g_configured = false;
        std::vector<MovieEntry> g_movie;
        size_t g_movieIndex = 0;
        uint64_t g_movieEnd = 0;
        std::vector<ScriptPress> g_script;
        FILE *g_record = nullptr;
        PadState g_lastRecorded;
        bool g_recordedAny = false;
        uint64_t g_currentVblank = 0;
        FILE *g_hashLog = nullptr;
        uint64_t g_hashInterval = 60;

        uint64_t hashBytes(const uint8_t *data, size_t size, uint64_t h = 0xcbf29ce484222325ull)
        {
            // 64-bit FNV-1a over 8-byte words: fast enough for 32 MB once a second.
            size_t i = 0;
            for (; i + 8 <= size; i += 8)
            {
                uint64_t w;
                std::memcpy(&w, data + i, 8);
                h = (h ^ w) * 0x100000001b3ull;
            }
            for (; i < size; ++i)
                h = (h ^ data[i]) * 0x100000001b3ull;
            return h;
        }

        bool loadMovie(const char *path)
        {
            std::ifstream in(path);
            if (!in)
                return false;
            for (std::string line; std::getline(in, line);)
            {
                if (line.empty() || line[0] == '#')
                    continue;
                std::istringstream fields(line);
                MovieEntry e{};
                unsigned buttons = 0, lx = 128, ly = 128, rx = 128, ry = 128;
                fields >> e.vblank >> std::hex >> buttons >> std::dec >> lx >> ly >> rx >> ry;
                if (!fields)
                    continue;
                e.state = PadState{static_cast<uint16_t>(buttons), static_cast<uint8_t>(lx), static_cast<uint8_t>(ly),
                                   static_cast<uint8_t>(rx), static_cast<uint8_t>(ry)};
                g_movie.push_back(e);
                g_movieEnd = e.vblank;
            }
            return true;
        }

        void loadScript(const char *spec)
        {
            // "<seconds>:<button>[:<hold seconds>],..." in guest time.
            std::istringstream items(spec);
            for (std::string item; std::getline(items, item, ',');)
            {
                std::istringstream parts(item);
                std::string at, name, hold;
                std::getline(parts, at, ':');
                std::getline(parts, name, ':');
                std::getline(parts, hold, ':');
                for (const auto &b : kButtons)
                {
                    if (name != b.name)
                        continue;
                    const double start = std::atof(at.c_str());
                    const double length = hold.empty() ? 0.25 : std::atof(hold.c_str());
                    const auto from = static_cast<uint64_t>(start * kVblanksPerSecond);
                    g_script.push_back({from, from + std::max<uint64_t>(1, static_cast<uint64_t>(length * kVblanksPerSecond)), b.mask});
                }
            }
            std::fprintf(stderr, "[input] script: %zu presses\n", g_script.size());
        }

        void configureLocked()
        {
            if (g_configured)
                return;
            g_configured = true;
            if (const char *play = std::getenv("RT_MOVIE_PLAY"))
            {
                if (loadMovie(play))
                    std::fprintf(stderr, "[movie] playing %s (%zu changes, ends at vblank %llu)\n", play, g_movie.size(),
                                 static_cast<unsigned long long>(g_movieEnd));
                else
                    std::fprintf(stderr, "[movie] cannot read %s\n", play);
            }
            if (const char *script = std::getenv("RT_INPUT_SCRIPT"))
                loadScript(script);
            if (const char *record = std::getenv("RT_MOVIE_RECORD"))
            {
                g_record = std::fopen(record, "w");
                if (g_record)
                    std::fprintf(g_record, "# roadtrip-movie 1\n");
            }
            if (const char *hash = std::getenv("RT_STATE_HASH"))
            {
                g_hashLog = std::fopen(hash, "w");
                if (const char *interval = std::getenv("RT_STATE_HASH_INTERVAL"))
                    g_hashInterval = std::max<uint64_t>(1, std::strtoull(interval, nullptr, 10));
            }
        }

        PadState nextStateLocked(uint64_t vblank, bool &scripted)
        {
            scripted = false;
            if (!g_movie.empty() && vblank <= g_movieEnd + 1u)
            {
                while (g_movieIndex + 1 < g_movie.size() && g_movie[g_movieIndex + 1].vblank <= vblank)
                    ++g_movieIndex;
                scripted = true;
                return g_movie[g_movieIndex].vblank <= vblank ? g_movie[g_movieIndex].state : PadState{};
            }
            if (!g_script.empty())
            {
                scripted = true;
                PadState s{};
                for (const auto &p : g_script)
                    if (vblank >= p.from && vblank < p.until)
                        s.buttons &= static_cast<uint16_t>(~p.mask);
                // Live input still works alongside a script (e.g. to take over afterwards).
                s.buttons &= g_live.buttons;
                return s;
            }
            return g_live;
        }

        void logStateHash(PS2Runtime &runtime, uint64_t vblank)
        {
            PS2Memory &memory = runtime.memory();
            memory.syncGifVif1();
            const uint64_t ee = hashBytes(memory.getRDRAM(), PS2_RAM_SIZE);
            const uint64_t spr = hashBytes(memory.getScratchpad(), PS2_SCRATCHPAD_SIZE);
            const uint64_t vu1 = hashBytes(memory.getVU1Data(), PS2_VU1_DATA_SIZE);
            static std::vector<uint8_t> iop(2u * 1024u * 1024u);
            const uint64_t iopHash = runtime.readIopMemory(0, iop.data(), iop.size()) ? hashBytes(iop.data(), iop.size()) : 0u;
            std::fprintf(g_hashLog, "%llu ee=%016llx spr=%016llx vu1=%016llx iop=%016llx\n",
                         static_cast<unsigned long long>(vblank), static_cast<unsigned long long>(ee),
                         static_cast<unsigned long long>(spr), static_cast<unsigned long long>(vu1),
                         static_cast<unsigned long long>(iopHash));
            std::fflush(g_hashLog);
        }
    }

    void setLiveInput(const PadState &state)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_live = state;
    }

    void addMarker(const std::string &kind, const std::string &text)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_record)
        {
            std::fprintf(g_record, "# marker %llu %s %s\n", static_cast<unsigned long long>(g_currentVblank), kind.c_str(),
                         text.c_str());
            std::fflush(g_record);
        }
    }

    bool inputScripted()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        return !g_movie.empty() || !g_script.empty();
    }

    void onVblank(PS2Runtime &runtime, uint64_t vblank)
    {
        PadState state;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            configureLocked();
            g_currentVblank = vblank;
            bool scripted = false;
            state = nextStateLocked(vblank, scripted);
            if (g_record && (!g_recordedAny || !(state == g_lastRecorded)))
            {
                std::fprintf(g_record, "%llu %04x %u %u %u %u\n", static_cast<unsigned long long>(vblank), state.buttons,
                             state.lx, state.ly, state.rx, state.ry);
                std::fflush(g_record);
                g_lastRecorded = state;
                g_recordedAny = true;
            }
        }
        ps2_stubs::setPadOverrideState(state.buttons, state.lx, state.ly, state.rx, state.ry);
        if (g_hashLog && vblank % g_hashInterval == 0u)
            logStateHash(runtime, vblank);
        // RT_EXIT_AT_VBLANK=<n>: stop the game after n guest vblanks (unattended runs).
        static const uint64_t exitAt = []
        { const char *e = std::getenv("RT_EXIT_AT_VBLANK"); return e ? std::strtoull(e, nullptr, 10) : 0ull; }();
        if (exitAt != 0u && vblank >= exitAt)
        {
            if (g_record)
                std::fflush(g_record);
            std::fprintf(stderr, "[test] reached vblank %llu, stopping\n", static_cast<unsigned long long>(vblank));
            runtime.requestStop();
        }
    }
}
