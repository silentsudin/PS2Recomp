#pragma once

#include "runtime/gs/gs_types.h"

#include <cstdint>
#include <vector>

// Implemented by backends that consume the raw GS command stream (e.g. a GPU GS that parses
// GIF packets itself) instead of the frontend's decoded primitives.
class GSPacketMirror
{
public:
    virtual ~GSPacketMirror() = default;
    // pathIndex: 0 = PATH1, 1 = PATH2, 2 = PATH3.
    virtual void MirrorGifPacket(uint32_t pathIndex, const uint8_t *data, uint32_t sizeBytes) = 0;
    // The same, with each vertex's side data in kick order: screen motion (packed half2 dx, dy in
    // GS pixels) or a UI class (gs_motion.h kVertexUi / kVertexNeutral). Backends that can't use
    // it drop it.
    virtual void MirrorGifPacketWithMotion(uint32_t pathIndex, const uint8_t *data, uint32_t sizeBytes,
                                           const uint32_t *motion, uint32_t motionCount)
    {
        (void)motion;
        (void)motionCount;
        MirrorGifPacket(pathIndex, data, sizeBytes);
    }
    virtual void MirrorRegisterWrite(uint8_t regAddr, uint64_t value) = 0;
    // Re-rendered frame generation: how many shadow frames are rendered per guest frame, and,
    // right before a MirrorGifPacket* call, that packet's version for shadow frame `index` (the
    // 3D re-projected). Packets without a version go to the shadows as they are.
    virtual uint32_t ShadowFrames() const { return 0; }
    virtual void SetShadowVariant(uint32_t index, const uint8_t *data, uint32_t sizeBytes)
    {
        (void)index;
        (void)data;
        (void)sizeBytes;
    }
};

class GSRasterBackend
{
public:
    virtual ~GSRasterBackend() = default;

    // False if the backend renders from a GSPacketMirror stream and does not need Submit().
    virtual bool WantsPrimitives() const { return true; }

    virtual void Initialize(uint8_t *vram, uint32_t vramSize) = 0;
    virtual void Reset() = 0;

    virtual void Submit(const GSPrimitiveBatch &batch) = 0;
    virtual void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) = 0;

    virtual void BeginTransfer(const GSTransferCommand &command) = 0;
    virtual void UploadImage(const uint8_t *data, uint32_t sizeBytes) = 0;

    virtual void Flush() = 0;
    // Depth for temporal upscalers: whether to keep each frame's 3D depth, and keeping it (called
    // when the frame's 3D is complete, before its HUD and post-processing overwrite Z).
    virtual bool WantsDepthSnapshot() const { return false; }
    // Per-vertex side data (motion, UI classes) wanted at all (MirrorGifPacketWithMotion).
    virtual bool WantsVertexSideband() const { return false; }
    // fbp: the frame buffer the 3D drew into (the snapshot belongs to that picture).
    virtual void SnapshotDepth(uint32_t zbp, uint32_t fbw, uint32_t fbp)
    {
        (void)zbp;
        (void)fbw;
        (void)fbp;
    }
    virtual void TextureFlush() = 0;
    virtual void Sync(GSSyncReason reason) = 0;
    virtual PresentationFrame Present(const GSPresentationRequest &request) = 0;

    virtual bool ClearFramebuffer(const GSContext &context, uint32_t rgba) = 0;
    virtual uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) = 0;

    virtual uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const = 0;
    virtual void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) = 0;
    virtual void SnapshotVram(std::vector<uint8_t> &out) const = 0;
    virtual GSTransferSnapshot GetTransferSnapshot() const = 0;
};
