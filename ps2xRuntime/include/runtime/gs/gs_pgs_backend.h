#pragma once

// GPU GS backend built on paraLLEl-GS (Vulkan compute; MoltenVK on macOS).
//
// The GS frontend keeps parsing the command stream for its bookkeeping (CSR, FINISH/SIGNAL,
// transfer state), but rendering happens on the GPU: every raw GIF packet and register write
// is mirrored into paraLLEl-GS. VRAM is copied back into the frontend's local memory only
// when the CPU needs it (local->host transfers, VRAM reads, debug snapshots).

#include "runtime/gs/gs_backend.h"

#include <cstdint>
#include <memory>
#include <string>

namespace ps2x::gs
{
    struct PgsOptions
    {
        // Vulkan library to load (e.g. a bundled libMoltenVK.dylib). Empty = system default.
        std::string vulkanLibrary;
        // Directory for the Vulkan pipeline cache. Empty = no persistent cache.
        std::string pipelineCacheDir;
    };

    struct PgsDeviceInfo
    {
        std::string name;      // e.g. "Apple M3 Max"
        uint32_t vendorId = 0; // PCI vendor (0x106B Apple, 0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel, 0x5143 Qualcomm, 0x13B5 Arm)
        uint32_t deviceId = 0;
        uint32_t apiVersion = 0;
    };

    // Settings that can change while the game runs (any thread).
    class PgsControl
    {
    public:
        virtual ~PgsControl() = default;
        virtual PgsDeviceInfo deviceInfo() const = 0;
        // 1, 2, 4, 8 or 16 samples per pixel. 4 and up also double the scanout resolution.
        virtual void setSuperSampling(uint32_t samples) = 0;
        virtual uint32_t superSampling() const = 0;
        // Always sample the top texture level (sharper textures when supersampling).
        virtual void setSharpTextures(bool on) = 0;
    };

    // Returns nullptr (and fills `error`) if paraLLEl-GS isn't compiled in or Vulkan init fails.
    // `control` (optional) receives an interface for runtime settings, valid while the backend lives.
    std::unique_ptr<GSRasterBackend> createPgsBackend(const PgsOptions &options, std::string &error,
                                                      PgsControl **control = nullptr);

    bool pgsAvailable();
}
