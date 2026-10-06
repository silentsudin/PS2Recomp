#pragma once

#include "ps2_runtime.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ps2x
{
    class StateArchive;
}

// This exception is the EE equivalent of a longjmp to the dispatcher.  It is
// not an error and must only be caught at EeScheduler::run().
struct EeDispatcherTransfer final
{
};

enum class EeThreadStatus : uint8_t
{
    Running,
    Ready,
    Waiting,
    WaitingSuspended,
    Suspended,
    Dormant,
};

enum class EeWaitReason : uint8_t
{
    None,
    Sleep,
    Semaphore,
    EventFlag,
    VSync,
    External,
    Mpeg,
};

struct EeSemaphoreWait
{
    int id = 0;
};

struct EeEventFlagWait
{
    int id = 0;
    uint32_t bits = 0;
    uint32_t mode = 0;
    uint32_t resultAddress = 0;
};

struct EeVSyncWait
{
    uint64_t afterTick = 0;
    int fixedResult = -1;
};

struct EeExternalWait
{
    uint32_t type = 0;
    uint64_t token = 0;
};

using EeWaitPayload = std::variant<std::monostate,
                                   EeSemaphoreWait,
                                   EeEventFlagWait,
                                   EeVSyncWait,
                                   EeExternalWait>;

// What a host-side continuation does, named at its creation site. A save state can only be taken
// while every live continuation is one it knows how to write down and rebuild ({kind, args}); for
// now none is (EeScheduler::canSnapshot reports them by kind).
enum class EeContinuationKind : uint8_t
{
    None,
    VSyncResumePc,         // PS2Runtime::eeWaitVSyncTicks: resume at a guest pc
    CdStreamRead,          // sceCdStRead waiting for buffered sectors: read the rest
    MpegPictureWait,       // sceMpegGetPicture waiting for a decoded frame: retry
    MpegPictureVSync,      // sceMpegGetPicture waiting for the frame's tick: retry
    IpuInitDone,           // sceIpuInit after the guest's SETD4_CHCR call
    SifCommandFree,        // after a guest SIF command handler: free its packet copy
    RpcServerReturn,       // SifCallRpc after the guest server function: copy back, end function
    RpcEndCallback,        // SifCallRpc after the guest end function: complete the client
    SyscallOverrideResult, // after a guest syscall override: copy its v0 to the caller
    ExitHandlerChain,      // after a thread's exit handlers: exit (and delete) the thread
    MpegCallbackFree,      // after a guest MPEG callback: free its data block
    Count,
};

[[nodiscard]] const char *eeContinuationKindName(EeContinuationKind kind) noexcept;
// Counts continuations made, per kind (diagnostics: RT_SNAPSHOT_PROBE reports them).
void eeNoteContinuationCreated(EeContinuationKind kind) noexcept;
[[nodiscard]] uint64_t eeContinuationsCreated(EeContinuationKind kind) noexcept;

// A host continuation (std::function) that carries its kind. Built only with a kind, so every
// creation site says what it is.
template <class Signature>
class EeContinuation;

template <class R, class... Args>
class EeContinuation<R(Args...)>
{
public:
    EeContinuation() = default;
    template <class F>
    EeContinuation(EeContinuationKind kind, F &&fn) : m_kind(kind), m_fn(std::forward<F>(fn))
    {
        eeNoteContinuationCreated(kind);
    }

    explicit operator bool() const noexcept { return static_cast<bool>(m_fn); }
    R operator()(Args... args) const { return m_fn(std::forward<Args>(args)...); }
    [[nodiscard]] EeContinuationKind kind() const noexcept { return m_fn ? m_kind : EeContinuationKind::None; }

private:
    EeContinuationKind m_kind = EeContinuationKind::None;
    std::function<R(Args...)> m_fn;
};

// Runs when a waiting thread is resumed (before its guest code).
using EeResumeContinuation = EeContinuation<void(R5900Context &)>;
// Runs when a guest invocation returns: (the invocation's context, the context it returns to).
using EeInvocationContinuation = EeContinuation<void(const R5900Context &, R5900Context &)>;

struct EeWaitState
{
    EeWaitReason reason = EeWaitReason::None;
    EeWaitPayload payload{};
    EeResumeContinuation completion;
};

enum class GuestInvocationKind : uint8_t
{
    Interrupt,
    Alarm,
    GsCallback,
    RpcCallback,
    SyscallOverride,
    ExitHandler,
    HleCall,
    SifCommand,
};

struct GuestInvocation
{
    GuestInvocationKind kind = GuestInvocationKind::Interrupt;
    uint64_t sequence = 0;
    uint64_t tag = 0;
    R5900Context context{};
    EeInvocationContinuation onComplete;
};

struct GuestThread
{
    int id = 0;
    R5900Context context{};
    uint32_t entry = 0;
    uint32_t stack = 0;
    uint32_t stackSize = 0;
    uint32_t gp = 0;
    uint32_t attr = 0;
    uint32_t option = 0;
    uint32_t arg = 0;
    int initialPriority = 0;
    int currentPriority = 0;
    EeThreadStatus status = EeThreadStatus::Dormant;
    int suspendCount = 0;
    uint32_t wakeupCount = 0;
    bool ownsStack = false;
    uint32_t tlsBase = 0;
    EeWaitState wait{};
    EeResumeContinuation resumeCompletion;
    std::vector<GuestInvocation> invocations;

    [[nodiscard]] R5900Context &activeContext()
    {
        return invocations.empty() ? context : invocations.back().context;
    }

    [[nodiscard]] const R5900Context &activeContext() const
    {
        return invocations.empty() ? context : invocations.back().context;
    }
};

struct EeSemaphore
{
    int id = 0;
    int count = 0;
    int maxCount = 0;
    int initCount = 0;
    uint32_t attr = 0;
    uint32_t option = 0;
    std::deque<int> waiters;
};

struct EeEventFlag
{
    int id = 0;
    uint32_t attr = 0;
    uint32_t option = 0;
    uint32_t initBits = 0;
    uint32_t bits = 0;
    std::deque<int> waiters;
};

struct EeAlarm
{
    int id = 0;
    uint16_t ticks = 0;
    uint32_t handler = 0;
    uint32_t argument = 0;
    uint32_t gp = 0;
    uint32_t sp = 0;
};

struct EeIrqHandler
{
    int id = 0;
    uint32_t cause = 0;
    uint32_t handler = 0;
    uint32_t argument = 0;
    uint32_t gp = 0;
    uint32_t sp = 0;
    bool enabled = true;
    int order = 0;
};

struct EeThreadSnapshot
{
    int id = 0;
    uint32_t pc = 0;
    uint32_t ra = 0;
    uint32_t sp = 0;
    uint32_t contextGp = 0;
    uint32_t entry = 0;
    uint32_t stack = 0;
    uint32_t stackSize = 0;
    uint32_t gp = 0;
    int initialPriority = 0;
    int currentPriority = 0;
    EeThreadStatus status = EeThreadStatus::Dormant;
    EeWaitReason waitReason = EeWaitReason::None;
    int waitId = 0;
    int suspendCount = 0;
    uint32_t wakeupCount = 0;
    uint32_t invocationDepth = 0;
};

struct EeSemaphoreSnapshot
{
    int id = 0;
    int count = 0;
    int maxCount = 0;
    uint32_t waiters = 0;
};

struct EeEventFlagSnapshot
{
    int id = 0;
    uint32_t bits = 0;
    uint32_t initBits = 0;
    uint32_t attr = 0;
    uint32_t waiters = 0;
};

struct EeKernelSnapshot
{
    uint64_t sequence = 0;
    uint64_t eeCycle = 0;
    uint64_t sliceEndCycle = 0;
    uint64_t nextEventCycle = 0;
    int runningThreadId = 0;
    std::vector<EeThreadSnapshot> threads;
    std::vector<EeSemaphoreSnapshot> semaphores;
    std::vector<EeEventFlagSnapshot> eventFlags;
};

enum class EeEventType : uint8_t
{
    Stop,
    VBlankStart,
    VBlankEnd,
    Dmac,
    ExternalWake,
    Alarm,
};

struct EeEvent
{
    EeEventType type = EeEventType::ExternalWake;
    uint32_t id = 0;
    uint64_t value = 0;
};

// Why a save state can't be taken at this scheduler loop top (EeScheduler::canSnapshot).
enum EeSnapshotBlocker : uint32_t
{
    kSnapshotContinuation = 1u << 0,   // a live host continuation (see kinds)
    kSnapshotPath3Fifo = 1u << 1,      // PATH3 packets held while PATH3 is masked
    kSnapshotDmaPending = 1u << 2,     // GIF/VIF DMA transfers not processed yet
    kSnapshotVif1Path2Image = 1u << 3, // VIF1 DIRECT image data split across DMA transfers
    kSnapshotGifArbiter = 1u << 4,     // GIF arbiter packets not handed to the GS
    kSnapshotGsTransfer = 1u << 5,     // a GS host<->local transfer half done
    kSnapshotMemoryCard = 1u << 6,     // a memory card file open
    kSnapshotMpeg = 1u << 7,           // an MPEG movie playing
};

struct EeSnapshotCheck
{
    uint32_t blockers = 0;                 // EeSnapshotBlocker bits
    uint32_t continuationKinds = 0;        // bit (1 << EeContinuationKind) per live kind
    uint16_t waitContinuations = 0;        // in waiting threads' wait states
    uint16_t resumeContinuations = 0;      // ready threads about to run one
    uint16_t invocationContinuations = 0;  // on threads' invocation stacks
    uint16_t pendingInvocationContinuations = 0;
    [[nodiscard]] bool ok() const noexcept { return blockers == 0u; }
};

struct EeThreadCreateParams
{
    uint32_t attr = 0;
    uint32_t entry = 0;
    uint32_t stack = 0;
    uint32_t stackSize = 0;
    uint32_t gp = 0;
    int priority = 0;
    uint32_t option = 0;
};

class EeScheduler
{
public:
    static constexpr int kMainThreadId = 1;
    static constexpr int kFirstThreadId = 2;
    static constexpr int kLastThreadId = 255;
    static constexpr int kPriorityCount = 128;
    static constexpr uint64_t kEeClockHz = 294912000ull;
    static constexpr uint32_t kGeneratedCheckpointCycles = 32u;
    static constexpr uint32_t kGuestDispatchCycles = 8u;
    static constexpr uint64_t kDefaultTimeSliceCycles = 65536ull;

    explicit EeScheduler(PS2Runtime &runtime);
    ~EeScheduler();

    EeScheduler(const EeScheduler &) = delete;
    EeScheduler &operator=(const EeScheduler &) = delete;

    void reset(uint8_t *rdram, const R5900Context &mainContext);
    void run();
    void requestStop();
    void postEvent(EeEvent event);
    [[nodiscard]] bool checkpointDue(uint32_t cycles = kGeneratedCheckpointCycles) noexcept;
    void accountCycles(uint32_t cycles) noexcept;
    [[nodiscard]] bool isExecutingGuest() const noexcept;

    // Kernel object API. All calls except postEvent/requestStop execute on the
    // EE executor and therefore need no host synchronization.
    void setupCurrentThread(uint32_t stack, uint32_t stackSize, uint32_t gp);
    int createThread(const EeThreadCreateParams &params);
    int deleteThread(int id, uint32_t &ownedStack);
    int startThread(int id, uint32_t arg, const R5900Context &caller, bool interruptSafe);
    [[noreturn]] void exitCurrent(bool deleteThread);
    int terminateThread(int id, uint32_t &ownedStack, bool interruptSafe);
    int suspendThread(int id, bool interruptSafe);
    int resumeThread(int id, bool interruptSafe);
    void sleepCurrent();
    int wakeupThread(int id, bool interruptSafe);
    int cancelWakeup(int id);
    int changePriority(int id, int priority, bool interruptSafe, int &oldPriority);
    int rotateReadyQueue(int priority, bool interruptSafe);
    int releaseWait(int id, bool interruptSafe);
    void transferIfRequested(bool interruptSafe);

    int createSemaphore(int initCount, int maxCount, uint32_t attr, uint32_t option);
    int deleteSemaphore(int id, bool interruptSafe);
    int signalSemaphore(int id, bool interruptSafe);
    int pollSemaphore(int id);
    void waitSemaphore(int id);

    int createEventFlag(uint32_t initialBits, uint32_t attr, uint32_t option);
    int deleteEventFlag(int id, bool interruptSafe);
    int setEventFlag(int id, uint32_t bits, bool interruptSafe);
    int clearEventFlag(int id, uint32_t mask);
    int pollEventFlag(int id, uint32_t bits, uint32_t mode, uint32_t &observedBits);
    void waitEventFlag(int id, uint32_t bits, uint32_t mode, uint32_t resultAddress);

    int setAlarm(uint16_t ticks, uint32_t handler, uint32_t argument, uint32_t gp, uint32_t sp);
    int cancelAlarm(int id);
    void queueInvocation(GuestInvocation invocation);
    [[noreturn]] void invokeCurrent(GuestInvocation invocation);
    [[noreturn]] void invokeCurrentSequence(std::vector<GuestInvocation> invocations);
    [[nodiscard]] bool hasInvocation(GuestInvocationKind kind, uint64_t tag) const;
    [[nodiscard]] uint32_t invocationStackTop();

    int addIrqHandler(bool dmac,
                      uint32_t cause,
                      uint32_t handler,
                      bool append,
                      uint32_t argument,
                      uint32_t gp,
                      uint32_t sp);
    int removeIrqHandler(bool dmac, uint32_t cause, int id);
    int setIrqHandlerEnabled(bool dmac, int id, bool enabled);
    int setIrqCauseEnabled(bool dmac, uint32_t cause, bool enabled);
    void dispatchIrq(bool dmac, uint32_t cause);
    void setVSyncFlag(uint32_t flagAddress, uint32_t tickAddress);
    [[nodiscard]] uint64_t currentVSyncTick() const noexcept;
    // Moves the next vblank's host time by `ns` (clamped to +-200 us), to keep the guest's frames
    // in phase with the display (see PS2Runtime::run). No effect in virtual time.
    void nudgeVBlank(int64_t ns) noexcept { m_vblankNudgeNs.fetch_add(ns, std::memory_order_relaxed); }
    // The host time the last vblank was scheduled for (steady_clock ns), 0 before the first.
    [[nodiscard]] int64_t lastVBlankHostNs() const noexcept { return m_lastVblankHostNs.load(std::memory_order_relaxed); }
    uint32_t setGsVSyncCallback(uint32_t callback, uint32_t gp, uint32_t sp);

    [[noreturn]] void waitVSync(uint64_t afterTick, int fixedResult = -1, EeResumeContinuation completion = {});
    void completeVSync(uint64_t tick);
    void completeExternalWait(uint32_t type, uint64_t token, int result);
    [[noreturn]] void waitExternal(EeWaitReason reason, uint32_t type, uint64_t token, EeResumeContinuation completion = {});

    [[nodiscard]] GuestThread *thread(int id);
    [[nodiscard]] const GuestThread *thread(int id) const;
    [[nodiscard]] EeSemaphore *semaphore(int id);
    [[nodiscard]] const EeSemaphore *semaphore(int id) const;
    [[nodiscard]] EeEventFlag *eventFlag(int id);
    [[nodiscard]] const EeEventFlag *eventFlag(int id) const;
    [[nodiscard]] GuestThread *currentThread();
    [[nodiscard]] const GuestThread *currentThread() const;
    [[nodiscard]] int currentThreadId() const noexcept;
    [[nodiscard]] R5900Context *currentContext();
    [[nodiscard]] uint8_t *rdram() const noexcept;

    // Direct syscall tests use the same main-thread record without starting a
    // second executor. Production execution calls reset() before run().
    void bindMainContextForSyscall(R5900Context &ctx, uint8_t *rdram);

    [[nodiscard]] EeKernelSnapshot snapshot() const;
    void publishSnapshot();

    // Whether a save state could be taken now (only meaningful at the top of run()). With
    // `hostState`, also checks the GIF/GS path (waits for the GIF/VIF1 worker first), the memory
    // card and MPEG; otherwise only the scheduler's continuations.
    [[nodiscard]] EeSnapshotCheck canSnapshot(bool hostState);
    // Save states (SaveState.cpp): the scheduler's whole state (threads, queues, kernel objects,
    // handlers, invocations, deadlines, counters). Writing needs canSnapshot() ok unless `hashOnly`
    // (a digest, any time on the executor); reading rebases host deadlines on now.
    void serializeState(ps2x::StateArchive &ar, bool hashOnly = false);
    // The host-relative part (scheduled events' wall-clock deadlines, relative to now), apart so
    // the rest of a state is the same bytes however long after a load it is saved again.
    void serializeHostTiming(ps2x::StateArchive &ar, bool hashOnly = false);
    // Save states: every place a guest context will resume (threads that aren't dormant, their
    // invocation stacks, pending invocations) whose pc `valid` rejects, as "thread 3 pc 0x1234".
    [[nodiscard]] std::vector<std::string> invalidResumePcs(const std::function<bool(uint32_t)> &valid) const;

private:
    std::atomic<int64_t> m_vblankNudgeNs{0};
    std::atomic<int64_t> m_lastVblankHostNs{0};

    struct ScheduledEvent
    {
        uint64_t deadlineCycle = 0;
        std::chrono::steady_clock::time_point hostDeadline{};
        EeEvent event{};
        uint64_t sequence = 0;
    };

    void assertExecutor() const;
public:
    [[nodiscard]] bool onExecutorThread() const noexcept { return m_executorThread == std::this_thread::get_id(); }
    // RT_TIME=virtual: guest time comes only from EE cycles, so a run is reproducible whatever the
    // host does. RT_SPEED=<x> paces it at x times real time (default 1), RT_SPEED=max runs it as
    // fast as the host allows.
    [[nodiscard]] bool virtualTime() const noexcept { return m_virtualTime; }
private:
    [[nodiscard]] int allocateThreadId();
    GuestThread &acquireInvocationThread();
    void enqueueReady(GuestThread &thread, bool front = false);
    void removeReady(GuestThread &thread);
    [[nodiscard]] GuestThread *selectReady();
    void makeRunning(GuestThread &thread);
    void makeDormant(GuestThread &thread);
    void removeFromWaitObject(GuestThread &thread);
    [[noreturn]] void blockCurrent(EeWaitState wait);
    void makeReady(GuestThread &thread, int result, bool interruptSafe);
    void requestPreemptionIfHigher(const GuestThread &readyThread, bool interruptSafe);
    void applyPendingPreemption();
    void processPendingEvents();
    void processDueDeadlines();
    void processEvent(const EeEvent &event);
    void finishEventWaiters(EeEventFlag &flag, bool interruptSafe);
    [[nodiscard]] static bool eventCondition(uint32_t current, uint32_t requested, uint32_t mode);
    static int waitObjectId(const EeWaitState &wait);
    void writeGuestU32(uint32_t address, uint32_t value);
    void waitForEvent();
    void scheduleEvent(uint64_t deadlineCycle, std::chrono::steady_clock::time_point hostDeadline, EeEvent event);
    void updateNextDeadline();
    [[nodiscard]] bool hasReadyAtOrAbovePriority(int priority) const;
    void renewTimeSlice();
    void copyMainContextToRuntime();
    void publishDebugContext(const R5900Context &context);
    void publishIdleDebugContext();

    // RT_SNAPSHOT_PROBE (EeSnapshotProbe.cpp): per-vblank canSnapshot() statistics.
    struct SnapshotProbe;
    struct SnapshotProbeDeleter
    {
        void operator()(SnapshotProbe *probe) const noexcept;
    };
    std::unique_ptr<SnapshotProbe, SnapshotProbeDeleter> m_snapshotProbe;
    void snapshotProbeStart();
    void snapshotProbeAtLoopTop();
    void snapshotProbeFinish();

    PS2Runtime &m_runtime;
    uint8_t *m_rdram = nullptr;
    std::array<std::deque<int>, kPriorityCount> m_readyQueues{};
    std::unordered_map<int, GuestThread> m_threads;
    std::unordered_map<int, EeSemaphore> m_semaphores;
    std::unordered_map<int, EeEventFlag> m_eventFlags;
    std::unordered_map<int, EeAlarm> m_alarms;
    std::unordered_map<int, EeIrqHandler> m_intcHandlers;
    std::unordered_map<int, EeIrqHandler> m_dmacHandlers;
    int m_nextThreadId = kFirstThreadId;
    int m_nextInvocationThreadId = -1;
    int m_nextSemaphoreId = 1;
    int m_nextEventFlagId = 1;
    int m_nextAlarmId = 1;
    int m_nextIntcHandlerId = 1;
    int m_nextDmacHandlerId = 1;
    int m_intcHeadOrder = 0;
    int m_intcTailOrder = 1000;
    int m_dmacHeadOrder = 0;
    int m_dmacTailOrder = 1000;
    uint32_t m_enabledIntcMask = 0xFFFFFFFFu;
    uint32_t m_enabledDmacMask = 0xFFFFFFFFu;
    int m_currentThreadId = 0;
    bool m_rescheduleRequested = false;
    bool m_timeSliceExpired = false;
    bool m_insideInterrupt = false;
    uint32_t m_pendingEeTimerInterrupts = 0;
    uint64_t m_eeCycle = 0;
    bool m_virtualTime = false;
    double m_virtualSpeed = 1.0; // 0 = unthrottled
    std::chrono::steady_clock::time_point m_virtualEpoch{};
    // Virtual time: the host time before which guest cycle `cycle` must not be reached (the pacing
    // target for RT_SPEED); time_point::min() when unthrottled.
    [[nodiscard]] std::chrono::steady_clock::time_point virtualPacingTime(uint64_t cycle) const;
    uint64_t m_sliceEndCycle = kDefaultTimeSliceCycles;
    std::thread::id m_executorThread{};
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_guestExecuting{false};
    std::atomic<bool> m_stopRequested{false};
    std::atomic<bool> m_checkpointPending{false};
    uint32_t m_debugPublishCountdown = 0u;

    mutable std::mutex m_eventMutex;
    std::condition_variable m_eventCv;
    std::deque<EeEvent> m_events;
    std::vector<ScheduledEvent> m_deadlines;
    std::deque<GuestInvocation> m_pendingInvocations;
    uint64_t m_eventSequence = 0;
    uint64_t m_invocationSequence = 0;
    uint64_t m_vsyncTick = 0;
    uint32_t m_vsyncFlagAddress = 0;
    uint32_t m_vsyncTickAddress = 0;
    uint32_t m_gsVSyncCallback = 0;
    uint32_t m_gsVSyncCallbackGp = 0;
    uint32_t m_gsVSyncCallbackSp = 0;
    std::unordered_map<uint64_t, uint32_t> m_invocationStackTops;
    std::atomic<uint64_t> m_nextDeadlineCycle{0};

    mutable std::mutex m_snapshotMutex;
    EeKernelSnapshot m_snapshot;
    uint64_t m_snapshotSequence = 0;
};
