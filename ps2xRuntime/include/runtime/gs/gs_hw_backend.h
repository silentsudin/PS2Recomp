#pragma once

// Hardware GS: draws the GS's primitives with the Vulkan graphics pipeline (as emulators do on
// phones), on the Vulkan presenter's device. It covers what Road Trip does, not the whole GS:
// render targets stay on the GPU at a chosen scale, paletted textures are decoded on the CPU
// (cached by content and palette), blending maps to fixed-function blend. Anything the game
// never does is logged once as unsupported. RT_GS_BACKEND=hw.

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_pgs_backend.h"

#include <memory>
#include <string>

namespace ps2x::gs
{
    struct HwOptions
    {
        // The Vulkan presenter (createPgsPresenter) whose device this renders on; nullptr = a
        // device of its own (headless, tests), pictures read back to the CPU.
        HostPresenter *presenter = nullptr;
        // Vulkan library to load for a device of its own (e.g. a bundled libMoltenVK.dylib).
        std::string vulkanLibrary;
        // Directory for the persistent Vulkan pipeline cache and the pipeline states drawn (compiled
        // ahead at the next start). Empty = neither.
        std::string pipelineCacheDir;
    };

    // nullptr (and `error`) if Vulkan is unavailable. `control` gets the runtime settings
    // interface: supersampling 1/4/16 select a render scale of 1x/2x/4x.
    std::unique_ptr<GSRasterBackend> createHwBackend(const HwOptions &options, std::string &error,
                                                     PgsControl **control = nullptr);

    // The draw pipeline states the hardware GS knows (built in, and those drawn before on this
    // device) are compiled on a worker from the moment it starts: from the persistent pipeline
    // cache in milliseconds, or, on a first run or after a driver or app update, in seconds (the
    // app shows "Preparing graphics" meanwhile). False if `backend` isn't the hardware GS.
    struct HwPipelinePrep
    {
        uint32_t total = 0; // states to compile
        uint32_t done = 0;  // of those, compiled (or already there)
    };
    bool hwPipelinePrep(GSRasterBackend *backend, HwPipelinePrep &out);
}
