#pragma once

// Apple MetalFX spatial upscaling for the Vulkan presenter (macOS 13+). MoltenVK runs Vulkan on
// Metal, so the presenter hands over the MTLTexture behind each VkImage and the MTLCommandQueue
// behind its queue (vkGetMTL*MVK); the scaler is encoded on that same queue between the
// presenter's Vulkan submissions. All pointers are Objective-C objects passed as void *.

#include <cstdint>
#include <memory>

namespace ps2x::gs
{
    class MetalFxSpatial
    {
    public:
        // nullptr if MetalFX is unavailable (older macOS, or no Metal device).
        static std::unique_ptr<MetalFxSpatial> create(void *mtlDevice);
        virtual ~MetalFxSpatial() = default;

        // Upscales `input` (inW x inH) into `output` (outW x outH), both RGBA8 unorm, on `queue`, and
        // waits for it to finish. False if the scaler could not be made for these sizes.
        virtual bool upscale(void *queue, void *input, uint32_t inW, uint32_t inH, void *output, uint32_t outW,
                             uint32_t outH) = 0;
    };
}
