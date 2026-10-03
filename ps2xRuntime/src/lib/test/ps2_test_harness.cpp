#include "runtime/ps2_test_harness.h"

#include "ps2_runtime.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>
#include <unistd.h>
#include <unistd.h>
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

        // Test server (lockstep).
        std::condition_variable g_parkCv;   // game thread waits here while parked
        std::condition_variable g_serverCv; // server waits for the game to park / a capture
        bool g_serverEnabled = false;
        std::atomic<bool> g_rendering{[] { const char *e = std::getenv("RT_RENDER"); return !(e && *e == '0'); }()};
        bool g_attached = false;
        std::atomic<bool> g_quitRequested{false}; // an orderly quit: shut down normally
        uint64_t g_runTarget = 0;           // park when reaching this vblank
        uint64_t g_parkedAt = UINT64_MAX;   // vblank the game is parked at, or UINT64_MAX
        bool g_serverPadActive = false;
        PadState g_serverPad;
        // Sound since the last "audio" query.
        uint64_t g_audioFrames = 0;
        double g_audioSquares = 0.0;
        int g_audioPeak = 0;
        uint64_t g_audioHash = 0xcbf29ce484222325ull;
        double g_audioLrDiff = 0.0; // sum of |left - right|: 0 for mono sound
        // Frame capture hand-off with the render thread.
        bool g_captureRequested = false;
        bool g_captureDone = false;
        std::vector<uint8_t> g_capture;
        uint32_t g_captureWidth = 0, g_captureHeight = 0;
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
            g_movieEnd = UINT64_MAX; // without an end line the last state holds forever
            for (std::string line; std::getline(in, line);)
            {
                if (line.rfind("# end ", 0) == 0)
                {
                    g_movieEnd = std::strtoull(line.c_str() + 6, nullptr, 10);
                    continue;
                }
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
                    std::fprintf(stderr, "[movie] playing %s (%zu changes, ends at vblank %lld)\n", play, g_movie.size(),
                                 g_movieEnd == UINT64_MAX ? -1ll : static_cast<long long>(g_movieEnd));
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
            if (!g_movie.empty() && vblank < g_movieEnd)
            {
                while (g_movieIndex + 1 < g_movie.size() && g_movie[g_movieIndex + 1].vblank <= vblank)
                    ++g_movieIndex;
                scripted = true;
                return g_movie[g_movieIndex].vblank <= vblank ? g_movie[g_movieIndex].state : PadState{};
            }
            if (g_serverPadActive)
            {
                scripted = true;
                return g_serverPad;
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

    void finishRecording()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_record)
            return;
        std::fprintf(g_record, "# end %llu\n", static_cast<unsigned long long>(g_currentVblank));
        std::fclose(g_record);
        g_record = nullptr;
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
            std::unique_lock<std::mutex> lock(g_mutex);
            configureLocked();
            g_currentVblank = vblank;
            // Lockstep: with a test server, park at the target vblank until the client asks for
            // more (or detaches). The game starts parked at vblank 1 so runs begin identically.
            if (g_serverEnabled && (!g_attached || vblank >= g_runTarget))
            {
                g_parkedAt = vblank;
                g_serverCv.notify_all();
                g_parkCv.wait(lock, [&]
                              { return (g_attached && vblank < g_runTarget) || !g_serverEnabled || runtime.isStopRequested(); });
                g_parkedAt = UINT64_MAX;
            }
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
            finishRecording();
            std::fprintf(stderr, "[test] reached vblank %llu, stopping\n", static_cast<unsigned long long>(vblank));
            runtime.requestStop();
        }
    }

    namespace
    {
        std::string jsonValue(const std::string &line, const char *key)
        {
            const std::string needle = std::string("\"") + key + "\"";
            size_t at = line.find(needle);
            if (at == std::string::npos)
                return {};
            at = line.find(':', at + needle.size());
            if (at == std::string::npos)
                return {};
            ++at;
            while (at < line.size() && line[at] == ' ')
                ++at;
            if (at < line.size() && line[at] == '"')
            {
                const size_t end = line.find('"', at + 1);
                return line.substr(at + 1, end == std::string::npos ? std::string::npos : end - at - 1);
            }
            size_t end = at;
            while (end < line.size() && line[end] != ',' && line[end] != '}' && line[end] != ' ')
                ++end;
            return line.substr(at, end - at);
        }

        uint64_t jsonNumber(const std::string &line, const char *key, uint64_t fallback = 0)
        {
            const std::string v = jsonValue(line, key);
            return v.empty() ? fallback : std::strtoull(v.c_str(), nullptr, 0);
        }

        std::string toHex(const uint8_t *data, size_t size)
        {
            static const char *digits = "0123456789abcdef";
            std::string out(size * 2, '0');
            for (size_t i = 0; i < size; ++i)
            {
                out[2 * i] = digits[data[i] >> 4];
                out[2 * i + 1] = digits[data[i] & 15];
            }
            return out;
        }

        std::vector<uint8_t> fromHex(const std::string &hex)
        {
            std::vector<uint8_t> out(hex.size() / 2);
            for (size_t i = 0; i < out.size(); ++i)
                out[i] = static_cast<uint8_t>(std::strtoul(hex.substr(2 * i, 2).c_str(), nullptr, 16));
            return out;
        }

        // Guest memory for read/write: returns a host pointer for [addr, addr+len) or nullptr.
        uint8_t *memorySpace(PS2Runtime &runtime, const std::string &space, uint32_t addr, size_t len)
        {
            PS2Memory &memory = runtime.memory();
            if (space == "ee" || space.empty())
            {
                const uint32_t phys = addr & 0x1FFFFFFFu;
                return phys + len <= PS2_RAM_SIZE ? memory.getRDRAM() + phys : nullptr;
            }
            if (space == "spr")
            {
                const uint32_t off = addr & (PS2_SCRATCHPAD_SIZE - 1u);
                return off + len <= PS2_SCRATCHPAD_SIZE ? memory.getScratchpad() + off : nullptr;
            }
            if (space == "vu1")
            {
                memory.syncGifVif1();
                const uint32_t off = addr & (PS2_VU1_DATA_SIZE - 1u);
                return off + len <= PS2_VU1_DATA_SIZE ? memory.getVU1Data() + off : nullptr;
            }
            return nullptr;
        }

        std::string handle(PS2Runtime &runtime, const std::string &line)
        {
            const std::string cmd = jsonValue(line, "cmd");
            if (cmd == "run")
            {
                const uint64_t count = jsonNumber(line, "vblanks", 1);
                std::unique_lock<std::mutex> lock(g_mutex);
                if (g_parkedAt == UINT64_MAX)
                    g_serverCv.wait(lock, [&] { return g_parkedAt != UINT64_MAX || runtime.isStopRequested(); });
                const uint64_t from = g_parkedAt;
                g_runTarget = from + count;
                g_parkCv.notify_all();
                g_serverCv.wait(lock, [&]
                                { return (g_parkedAt != UINT64_MAX && g_parkedAt >= g_runTarget) || runtime.isStopRequested(); });
                if (runtime.isStopRequested())
                    return "{\"ok\":false,\"error\":\"stopped\"}";
                return "{\"ok\":true,\"vblank\":" + std::to_string(g_parkedAt) + "}";
            }
            if (cmd == "pad")
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_serverPad = PadState{static_cast<uint16_t>(jsonNumber(line, "buttons", 0xFFFF)),
                                       static_cast<uint8_t>(jsonNumber(line, "lx", 128)),
                                       static_cast<uint8_t>(jsonNumber(line, "ly", 128)),
                                       static_cast<uint8_t>(jsonNumber(line, "rx", 128)),
                                       static_cast<uint8_t>(jsonNumber(line, "ry", 128))};
                g_serverPadActive = true;
                return "{\"ok\":true}";
            }
            if (cmd == "release_pad")
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_serverPadActive = false;
                return "{\"ok\":true}";
            }
            if (cmd == "read" || cmd == "write")
            {
                const std::string space = jsonValue(line, "space");
                const auto addr = static_cast<uint32_t>(jsonNumber(line, "addr"));
                if (cmd == "read")
                {
                    const size_t len = jsonNumber(line, "len", 4);
                    std::vector<uint8_t> buffer(len);
                    bool ok = false;
                    if (space == "iop")
                        ok = runtime.readIopMemory(addr, buffer.data(), len);
                    else if (const uint8_t *src = memorySpace(runtime, space, addr, len))
                    {
                        std::memcpy(buffer.data(), src, len);
                        ok = true;
                    }
                    return ok ? "{\"ok\":true,\"data\":\"" + toHex(buffer.data(), len) + "\"}"
                              : "{\"ok\":false,\"error\":\"bad address\"}";
                }
                const std::vector<uint8_t> data = fromHex(jsonValue(line, "data"));
                bool ok = false;
                if (space == "iop")
                    ok = runtime.writeIopMemory(addr, data.data(), data.size());
                else if (uint8_t *dst = memorySpace(runtime, space, addr, data.size()))
                {
                    std::memcpy(dst, data.data(), data.size());
                    ok = true;
                }
                return ok ? "{\"ok\":true}" : "{\"ok\":false,\"error\":\"bad address\"}";
            }
            if (cmd == "frame")
            {
                const std::string path = jsonValue(line, "path");
                runtime.memory().syncGifVif1();
                std::unique_lock<std::mutex> lock(g_mutex);
                g_captureDone = false;
                g_captureRequested = true;
                if (!g_serverCv.wait_for(lock, std::chrono::seconds(10), [] { return g_captureDone; }))
                {
                    g_captureRequested = false;
                    return "{\"ok\":false,\"error\":\"no frame\"}";
                }
                if (FILE *f = std::fopen(path.c_str(), "wb"))
                {
                    std::fwrite(g_capture.data(), 1, g_capture.size(), f);
                    std::fclose(f);
                }
                else
                    return "{\"ok\":false,\"error\":\"cannot write\"}";
                return "{\"ok\":true,\"width\":" + std::to_string(g_captureWidth) + ",\"height\":" +
                       std::to_string(g_captureHeight) + "}";
            }
            if (cmd == "stats")
            {
                PS2Memory &memory = runtime.memory();
                std::lock_guard<std::mutex> lock(g_mutex);
                return "{\"ok\":true,\"vblank\":" + std::to_string(g_currentVblank) +
                       ",\"flips\":" + std::to_string(memory.displayFlips()) +
                       ",\"vif1_busy_ns\":" + std::to_string(memory.gifVif1BusyNanos()) +
                       ",\"gs_busy_ns\":" + std::to_string(memory.gsThreadBusyNanos()) + "}";
            }
            if (cmd == "render")
            {
                setRenderingEnabled(jsonNumber(line, "on", 1) != 0);
                return "{\"ok\":true}";
            }
            if (cmd == "step")
            {
                // Pad (optional) + run + reads, one round trip for driving bots.
                if (!jsonValue(line, "buttons").empty())
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    g_serverPad = PadState{static_cast<uint16_t>(jsonNumber(line, "buttons", 0xFFFF)),
                                           static_cast<uint8_t>(jsonNumber(line, "lx", 128)),
                                           static_cast<uint8_t>(jsonNumber(line, "ly", 128)),
                                           static_cast<uint8_t>(jsonNumber(line, "rx", 128)),
                                           static_cast<uint8_t>(jsonNumber(line, "ry", 128))};
                    g_serverPadActive = true;
                }
                const std::string ran = handle(runtime, "{\"cmd\":\"run\",\"vblanks\":" +
                                                            std::to_string(jsonNumber(line, "vblanks", 1)) + "}");
                if (ran.find("\"ok\":true") == std::string::npos)
                    return ran;
                std::string reply = "{\"ok\":true,\"vblank\":" + std::to_string(g_parkedAt) + ",\"data\":[";
                std::istringstream reads(jsonValue(line, "reads"));
                bool first = true;
                for (std::string item; std::getline(reads, item, ',');)
                {
                    std::istringstream parts(item);
                    std::string space, addrText, lenText;
                    std::getline(parts, space, ':');
                    std::getline(parts, addrText, ':');
                    std::getline(parts, lenText, ':');
                    const auto addr = static_cast<uint32_t>(std::strtoull(addrText.c_str(), nullptr, 0));
                    const size_t len = std::strtoull(lenText.c_str(), nullptr, 0);
                    std::vector<uint8_t> buffer(len);
                    if (space == "iop")
                        runtime.readIopMemory(addr, buffer.data(), len);
                    else if (const uint8_t *src = memorySpace(runtime, space, addr, len))
                        std::memcpy(buffer.data(), src, len);
                    reply += std::string(first ? "" : ",") + "\"" + toHex(buffer.data(), len) + "\"";
                    first = false;
                }
                return reply + "]}";
            }
            if (cmd == "marker")
            {
                addMarker(jsonValue(line, "kind"), jsonValue(line, "text"));
                return "{\"ok\":true}";
            }
            if (cmd == "audio")
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                const double rms = g_audioFrames ? std::sqrt(g_audioSquares / (2.0 * static_cast<double>(g_audioFrames))) : 0.0;
                char reply[256];
                std::snprintf(reply, sizeof(reply),
                              "{\"ok\":true,\"frames\":%llu,\"rms\":%.2f,\"peak\":%d,\"hash\":\"%016llx\",\"lr_diff\":%.3f}",
                              static_cast<unsigned long long>(g_audioFrames), rms, g_audioPeak,
                              static_cast<unsigned long long>(g_audioHash),
                              g_audioFrames ? g_audioLrDiff / static_cast<double>(g_audioFrames) : 0.0);
                g_audioFrames = 0;
                g_audioSquares = 0.0;
                g_audioPeak = 0;
                g_audioHash = 0xcbf29ce484222325ull;
                g_audioLrDiff = 0.0;
                return reply;
            }
            if (cmd == "quit")
            {
                g_quitRequested.store(true);
                runtime.requestStop();
                std::lock_guard<std::mutex> lock(g_mutex);
                g_parkCv.notify_all();
                return "{\"ok\":true}";
            }
            return "{\"ok\":false,\"error\":\"unknown command\"}";
        }

        void serveClient(PS2Runtime &runtime, int fd)
        {
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_attached = true;
            }
            std::string buffer;
            char chunk[65536];
            for (;;)
            {
                const ssize_t n = ::read(fd, chunk, sizeof(chunk));
                if (n <= 0)
                    break;
                buffer.append(chunk, static_cast<size_t>(n));
                size_t nl;
                while ((nl = buffer.find('\n')) != std::string::npos)
                {
                    const std::string line = buffer.substr(0, nl);
                    buffer.erase(0, nl + 1);
                    const std::string reply = handle(runtime, line) + "\n";
                    size_t sent = 0;
                    while (sent < reply.size())
                    {
                        const ssize_t w = ::write(fd, reply.data() + sent, reply.size() - sent);
                        if (w <= 0)
                            break;
                        sent += static_cast<size_t>(w);
                    }
                }
            }
            ::close(fd);
            if (g_quitRequested.load())
                return; // the normal shutdown is under way (finishes a recorded movie, etc.)
            // The test driver is gone (it crashed or was killed): a test instance has no
            // reason to keep running, so it exits rather than linger in the background.
            std::fprintf(stderr, "[test] client disconnected; exiting\n");
            std::fflush(nullptr);
            std::_Exit(0);
        }

        // Test instances also exit if their parent process dies or no client ever attaches, so
        // a crashed or killed test run never leaves games behind.
        void startOrphanWatch()
        {
            const pid_t parent = ::getppid();
            std::thread([parent]
                        {
                            for (;;)
                            {
                                std::this_thread::sleep_for(std::chrono::seconds(1));
                                if (::getppid() != parent)
                                {
                                    std::fprintf(stderr, "[test] parent process gone; exiting\n");
                                    std::fflush(nullptr);
                                    std::_Exit(0);
                                }
                            } })
                .detach();
        }
    }

    void startServerIfRequested(PS2Runtime &runtime)
    {
        const char *path = std::getenv("RT_TEST_SOCKET");
        if (!path)
            return;
        const int listener = ::socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);
        ::unlink(path);
        if (listener < 0 || ::bind(listener, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
            ::listen(listener, 1) != 0)
        {
            std::fprintf(stderr, "[test] cannot listen on %s\n", path);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_serverEnabled = true;
        }
        std::fprintf(stderr, "[test] control socket %s\n", path);
        startOrphanWatch();
        std::thread([&runtime, listener]
                    {
                        // One client for the whole run; the game waits at vblank 1 until it attaches.
                        pollfd waiting{listener, POLLIN, 0};
                        if (::poll(&waiting, 1, 300 * 1000) <= 0)
                        {
                            std::fprintf(stderr, "[test] no client attached within 300 s; exiting\n");
                            std::fflush(nullptr);
                            std::_Exit(0);
                        }
                        const int fd = ::accept(listener, nullptr, nullptr);
                        ::close(listener);
                        if (fd >= 0)
                            serveClient(runtime, fd); })
            .detach();
    }

    bool renderingEnabled() { return g_rendering.load(std::memory_order_relaxed); }
    void setRenderingEnabled(bool on) { g_rendering.store(on, std::memory_order_relaxed); }

    void onAudio(const int16_t *interleavedStereo, size_t frames)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_serverEnabled && !g_attached)
            return;
        for (size_t i = 0; i < frames; ++i)
            g_audioLrDiff += std::abs(static_cast<int>(interleavedStereo[2 * i]) - interleavedStereo[2 * i + 1]);
        for (size_t i = 0; i < frames * 2; ++i)
        {
            const int v = interleavedStereo[i];
            g_audioSquares += static_cast<double>(v) * v;
            g_audioPeak = std::max(g_audioPeak, v < 0 ? -v : v);
            g_audioHash = (g_audioHash ^ static_cast<uint16_t>(v)) * 0x100000001b3ull;
        }
        g_audioFrames += frames;
    }

    bool frameCaptureRequested()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_captureRequested;
    }

    void deliverFrameCapture(const std::vector<uint8_t> &rgba, uint32_t width, uint32_t height)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_captureRequested)
            return;
        g_capture = rgba;
        g_captureWidth = width;
        g_captureHeight = height;
        g_captureRequested = false;
        g_captureDone = true;
        g_serverCv.notify_all();
    }
}
