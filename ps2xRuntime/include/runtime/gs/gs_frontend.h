#ifndef PS2_GS_FRONTEND_H
#define PS2_GS_FRONTEND_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_wide_layout.h"
#include "runtime/gs/gs_motion.h"

struct GSDebugSnapshot
{
    GSContext ctx[2]{};
    GSPrimReg prim{};
    GSTexaReg texa{};
    GSTexClutReg texclut{};
    uint64_t scanmsk = 0;
    uint64_t dimx = 0;
    uint64_t dthe = 0;
    uint64_t colclamp = 0;
    GSBitBltBuf bitbltbuf{};
    GSTrxPos trxpos{};
    GSTrxReg trxreg{};
    uint32_t trxdir = 0;
    uint32_t transferX = 0;
    uint32_t transferY = 0;
    uint32_t transferTotalPixels = 0;
    uint32_t transferCopiedPixels = 0;
    uint32_t lastDisplayBaseBytes = 0;
    GSFrameReg preferredDisplaySourceFrame{};
    uint32_t preferredDisplayDestFbp = 0;
    bool hasPreferredDisplaySource = false;
    uint32_t hostPresentationWidth = 0;
    uint32_t hostPresentationHeight = 0;
    uint32_t hostPresentationDisplayFbp = 0;
    uint32_t hostPresentationSourceFbp = 0;
    bool hostPresentationUsedPreferred = false;
    bool hasHostPresentationFrame = false;
    size_t localToHostPendingBytes = 0;
};

enum class GSDebugEventKind : uint8_t
{
    GifTag = 0,
    Register = 1,
    Draw = 2,
    Transfer = 3,
    Present = 4,
};

struct GSDebugHistoryEntry
{
    uint64_t seq = 0;
    uint64_t vsyncTick = 0;
    uint32_t frameIndex = 0;
    GSDebugEventKind kind = GSDebugEventKind::Register;

    uint8_t reg = 0;
    uint64_t regValue = 0;

    uint32_t gifSizeBytes = 0;
    uint32_t gifNloop = 0;
    uint8_t gifFlg = 0;
    uint8_t gifNreg = 0;

    GSPrimReg prim{};
    GSFrameReg frame{};
    GSZbufReg zbuf{};
    GSTex0Reg tex0{};
    GSScissorReg scissor{};
    uint64_t test = 0;
    uint64_t alpha = 0;

    uint32_t vertexCount = 0;
    float xMin = 0.0f;
    float xMax = 0.0f;
    float yMin = 0.0f;
    float yMax = 0.0f;
    double zMin = 0.0;
    double zMax = 0.0;
    uint8_t aMin = 0;
    uint8_t aMax = 0;

    GSBitBltBuf bitbltbuf{};
    GSTrxPos trxpos{};
    GSTrxReg trxreg{};
    uint32_t trxdir = 0;
    uint32_t transferPixels = 0;

    uint32_t displayFbp = 0;
    uint32_t sourceFbp = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    bool usedPreferred = false;
};

class GS
{
public:
    GS();
    ~GS() = default;

    void init(uint8_t *vram, uint32_t vramSize, struct GSRegisters *privRegs = nullptr);
    void reset();
    void setRasterBackend(std::unique_ptr<GSRasterBackend> backend);

    void processGIFPacket(const uint8_t *data, uint32_t sizeBytes); // PATH3
    // pathIndex: 0 = PATH1, 1 = PATH2, 2 = PATH3 (forwarded to a GSPacketMirror backend).
    void processGIFPacket(uint32_t pathIndex, const uint8_t *data, uint32_t sizeBytes);
    bool processNativePackedGIFPacket(const uint8_t *data, uint32_t sizeBytes);

    // Widescreen: the window's aspect and where the HUD goes (gs_wide_layout.h). 4:3 = off.
    void setWideLayout(float aspect, ps2x::gs::HudPlacement placement);
    // The game started drawing a frame (its clear, sceGsClear-style, from the GS stubs).
    void markFrameStart();

    // Progressive fields: the game draws 224-line fields, nudged half a line down on every other
    // one (sceGsSetHalfOffset) for an interlaced TV. With this on, the game hook drops the nudge
    // and the scanout shows each field as a whole progressive picture instead of deinterlacing.
    void setProgressiveFields(bool on) { m_progressiveFields.store(on, std::memory_order_relaxed); }
    bool progressiveFields() const { return m_progressiveFields.load(std::memory_order_relaxed); }

    // Temporal AA / upscaling: a sub-pixel camera jitter per frame (Halton 2,3), in frame-buffer
    // pixels, sized so it spans one pixel of the picture (fbPerPixel: frame-buffer pixels per
    // picture pixel). The game hook adds it to the player cameras (cameraJitter).
    void setTemporalJitter(bool on, float fbPerPixelX, float fbPerPixelY);
    bool cameraJitter(float &x, float &y) const;
    // The jitter of the frame whose depth/motion were last kept, and of the one before.
    void snapshotJitter(float &curX, float &curY, float &prevX, float &prevY) const;
    // The last frame was a 2D-backed screen (title, menus): show it 4:3 even when widescreen is on.
    bool lastFrameWas2D() const;
    // Widescreen is on and the game is driving (the 3D camera should be widened).
    bool wideDriving() const;
    float wideHorizontalScale() const;
    void uploadImageNative(uint64_t bitbltbuf,
                           uint64_t trxpos,
                           uint64_t trxreg,
                           uint64_t trxdir,
                           const uint8_t *data,
                           uint32_t sizeBytes);
    void writeRegister(uint8_t regAddr, uint64_t value);

    const uint8_t *lockDisplaySnapshot(uint32_t &outSize);
    void unlockDisplaySnapshot();
    uint32_t getLastDisplayBaseBytes() const;
    const GSFrameReg &getContextFrame(int index) const
    {
        return m_ctx[(index != 0) ? 1 : 0].frame;
    }
    GSDebugSnapshot getDebugSnapshot() const;
    std::vector<GSDebugHistoryEntry> getDebugHistory() const;
    void clearDebugHistory();
    bool isDebugHistoryPaused() const;
    void setDebugHistoryPaused(bool paused);
    bool getPreferredDisplaySource(GSFrameReg &outSource, uint32_t &outDestFbp) const;
    // keepOnGpu: a GPU presenter shows the picture (the backend keeps it as an image); readback
    // additionally copies it to the CPU like the default path (test captures).
    void latchHostPresentationFrame(bool keepOnGpu = false, bool readback = true);
    bool copyLatchedHostPresentationFrame(std::vector<uint8_t> &outPixels,
                                          uint32_t &outWidth,
                                          uint32_t &outHeight,
                                          uint32_t *outDisplayFbp = nullptr,
                                          uint32_t *outSourceFbp = nullptr,
                                          bool *outUsedPreferred = nullptr) const;
    bool clearFramebufferContext(uint32_t contextIndex, uint32_t rgba);
    bool clearActiveFramebuffer(uint32_t rgba);
    uint64_t nativeImageUploadCount() const { return m_nativeImageUploadCount; }
    uint64_t nativePackedGIFPacketCount() const { return m_nativePackedGIFPacketCount; }

    uint32_t consumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes);

    void refreshDisplaySnapshot();

    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);
    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;

private:
    void snapshotVRAM();
    void writeRegisterUnlocked(uint8_t regAddr, uint64_t value);
    void writeRegisterPacked(uint8_t regDesc, uint64_t lo, uint64_t hi);
    void uploadImageNativeUnlocked(uint64_t bitbltbuf,
                                   uint64_t trxpos,
                                   uint64_t trxreg,
                                   uint64_t trxdir,
                                   const uint8_t *data,
                                   uint32_t sizeBytes);
    void vertexKick(bool drawing);

    void recordDebugEventUnlocked(GSDebugHistoryEntry entry);
    GSDebugHistoryEntry makeDebugEventUnlocked(GSDebugEventKind kind) const;
    void recordGifTagDebugEventUnlocked(uint32_t sizeBytes, uint32_t nloop, uint8_t flg, uint32_t nreg);
    void recordRegisterDebugEventUnlocked(uint8_t regAddr, uint64_t value);
    void recordDrawDebugEventUnlocked(int vertexCount);
    void recordTransferDebugEventUnlocked();
    void recordPresentDebugEventUnlocked(uint32_t displayFbp, uint32_t sourceFbp, uint32_t width, uint32_t height, bool usedPreferred);

    void processImageData(const uint8_t *data, uint32_t sizeBytes);
    bool tryProcessNativeImageUploadPacket(const uint8_t *data, uint32_t sizeBytes);
    GSPrimitiveBatch buildDrawBatch(int vertexCount) const;
    void updatePreferredDisplaySourceForDraw(const GSPrimitiveBatch &batch);
    GSPresentationRequest buildPresentationRequestUnlocked() const;


    GSContext &activeContext();

    uint8_t *m_localMemoryStorage = nullptr;
    uint32_t m_localMemorySize = 0u;
    struct GSRegisters *m_privRegs = nullptr;
    mutable std::recursive_mutex m_stateMutex;
    mutable std::mutex m_backendLifetimeMutex;
    mutable std::mutex m_presentationMutex;

    GSContext m_ctx[2];
    GSPrimReg m_prim{};
    ps2x::gs::WideLayout m_wide;           // widescreen 2D placement (gs_wide_layout.h)
    uint32_t m_motionId = 0;               // the motion context of the PATH1 packets that follow
    std::vector<uint32_t> m_motionScratch;
    void accumulateMotionStats();
    uint64_t m_motionVerts = 0;
    uint64_t m_motionHist[6] = {};
    GSZbufReg m_lastZbuf3D{};              // the Z buffer of the last 3D (PATH1) draw
    bool m_haveZbuf3D = false;
    // m_wide's answers for other threads without the state lock (the game's camera hook asks on
    // every camera build): published after each change, under the lock.
    std::atomic<bool> m_wideDriving{false}, m_wide2D{false};
    std::atomic<bool> m_jitterOn{false};
    std::atomic<bool> m_progressiveFields{false};
    std::vector<uint8_t> m_shadowScratch[3]; // re-projected PATH1 packets for shadow frames
    std::vector<uint8_t> m_shadowKeep;        // per vertex: UI, not moved in shadow frames
    std::atomic<float> m_jitterScaleX{0.5f}, m_jitterScaleY{0.25f};
    std::atomic<uint32_t> m_frameIndex{0};
    float m_snapJitter[4] = {}; // cur x, y, prev x, y (under m_stateMutex)
    std::atomic<float> m_wideK{1.0f};
    void publishWideUnlocked();
    std::vector<uint8_t> m_wideScratch;    // the packet being transformed
    GSPrimReg m_primRegister{};
    GSPrimReg m_prmodeRegister{};

    uint8_t m_curR = 0x80, m_curG = 0x80, m_curB = 0x80, m_curA = 0x80;
    float m_curQ = 1.0f;
    float m_curS = 0.0f, m_curT = 0.0f;
    uint16_t m_curU = 0, m_curV = 0;
    uint8_t m_curFog = 0;
    uint8_t m_fogR = 0, m_fogG = 0, m_fogB = 0;

    bool m_prmodecont = true;
    bool m_pabe = false;
    uint64_t m_scanmsk = 0;
    uint64_t m_dimx = 0;
    uint64_t m_dthe = 0;
    uint64_t m_colclamp = 0;
    GSTexaReg m_texa{0u, false, 0u};
    GSTexClutReg m_texclut{0u, 0u, 0u};

    GSBitBltBuf m_bitbltbuf{};
    GSTrxPos m_trxpos{};
    GSTrxReg m_trxreg{};
    uint32_t m_trxdir = 3;


    // RT_GS_BATCH_LOG=<file>: one line per run of similar draws (path, target, primitive,
    // texture, screen bounding box) per frame, for telling HUD/2D from 3D.
    uint32_t m_curPath = 2u;
    void logBatchVertex(const GSVertex &vtx);
    void surveyDrawUnlocked();
    void surveyTransferUnlocked();

    static constexpr int kMaxVerts = 6;
    GSVertex m_vtxQueue[kMaxVerts];
    int m_vtxCount = 0;
    int m_vtxIndex = 0;

    std::vector<uint8_t> m_displaySnapshot;
    std::mutex m_snapshotMutex;
    uint32_t m_lastDisplayBaseBytes = 0;
    GSFrameReg m_preferredDisplaySourceFrame{};
    uint32_t m_preferredDisplayDestFbp = 0;
    bool m_hasPreferredDisplaySource = false;
    std::vector<uint8_t> m_hostPresentationFrame;
    uint32_t m_hostPresentationWidth = 0;
    uint32_t m_hostPresentationStride = 0;
    uint32_t m_hostPresentationHeight = 0;
    uint32_t m_hostPresentationDisplayFbp = 0;
    uint32_t m_hostPresentationSourceFbp = 0;
    bool m_hostPresentationUsedPreferred = false;
    bool m_hasHostPresentationFrame = false;
    uint64_t m_nativeImageUploadCount = 0;
    uint64_t m_nativePackedGIFPacketCount = 0;

    static constexpr size_t kDebugHistoryCapacity = 512;
    std::array<GSDebugHistoryEntry, kDebugHistoryCapacity> m_debugHistory{};
    size_t m_debugHistoryWrite = 0;
    size_t m_debugHistoryCount = 0;
    uint64_t m_debugNextSeq = 1;
    uint32_t m_debugFrameIndex = 0;
    uint64_t m_debugLastVsyncTick = UINT64_MAX;
    bool m_debugHistoryPaused = true;

    std::unique_ptr<GSRasterBackend> m_backend;
    GSPacketMirror *m_packetMirror = nullptr; // m_backend, if it renders from the raw stream
    bool m_backendWantsPrimitives = true;
};

#endif
