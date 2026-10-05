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
    };

    // nullptr (and `error`) if Vulkan is unavailable. `control` gets the runtime settings
    // interface: supersampling 1/4/16 select a render scale of 1x/2x/4x.
    std::unique_ptr<GSRasterBackend> createHwBackend(const HwOptions &options, std::string &error,
                                                     PgsControl **control = nullptr);
}
