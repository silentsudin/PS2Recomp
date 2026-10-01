#pragma once

// GPU GS backend built on paraLLEl-GS (Vulkan compute; MoltenVK on macOS).
//
// The GS frontend keeps parsing the command stream for its bookkeeping (CSR, FINISH/SIGNAL,
// transfer state), but rendering happens on the GPU: every raw GIF packet and register write
// is mirrored into paraLLEl-GS. VRAM is copied back into the frontend's local memory only
// when the CPU needs it (local->host transfers, VRAM reads, debug snapshots).

#include "runtime/gs/gs_backend.h"

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

    // Returns nullptr (and fills `error`) if paraLLEl-GS isn't compiled in or Vulkan init fails.
    std::unique_ptr<GSRasterBackend> createPgsBackend(const PgsOptions &options, std::string &error);

    bool pgsAvailable();
}
