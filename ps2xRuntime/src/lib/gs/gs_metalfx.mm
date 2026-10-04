#include "gs_metalfx.h"

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

namespace ps2x::gs
{
    namespace
    {
        class MetalFxSpatialImpl final : public MetalFxSpatial
        {
        public:
            explicit MetalFxSpatialImpl(id<MTLDevice> device) : m_device(device) {}

            bool upscale(void *queue, void *input, uint32_t inW, uint32_t inH, void *output, uint32_t outW,
                         uint32_t outH) override
            {
                @autoreleasepool
                {
                    if (!m_scaler || inW != m_inW || inH != m_inH || outW != m_outW || outH != m_outH)
                    {
                        MTLFXSpatialScalerDescriptor *desc = [[MTLFXSpatialScalerDescriptor alloc] init];
                        desc.inputWidth = inW;
                        desc.inputHeight = inH;
                        desc.outputWidth = outW;
                        desc.outputHeight = outH;
                        desc.colorTextureFormat = MTLPixelFormatRGBA8Unorm;
                        desc.outputTextureFormat = MTLPixelFormatRGBA8Unorm;
                        // The game's colours are display-referred (gamma encoded).
                        desc.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
                        m_scaler = [desc newSpatialScalerWithDevice:m_device];
                        if (!m_scaler)
                            return false;
                        m_inW = inW;
                        m_inH = inH;
                        m_outW = outW;
                        m_outH = outH;
                    }
                    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    m_scaler.colorTexture = (__bridge id<MTLTexture>)input;
                    m_scaler.outputTexture = (__bridge id<MTLTexture>)output;
                    m_scaler.inputContentWidth = inW;
                    m_scaler.inputContentHeight = inH;
                    [m_scaler encodeToCommandBuffer:cb];
                    [cb commit];
                    [cb waitUntilCompleted];
                    return cb.status == MTLCommandBufferStatusCompleted;
                }
            }

        private:
            id<MTLDevice> m_device;
            id<MTLFXSpatialScaler> m_scaler = nil;
            uint32_t m_inW = 0, m_inH = 0, m_outW = 0, m_outH = 0;
        };
    }

    namespace
    {
        class MetalFxTemporalImpl final : public MetalFxTemporal
        {
        public:
            explicit MetalFxTemporalImpl(id<MTLDevice> device) : m_device(device) {}

            bool upscale(void *queue, const Frame &f) override
            {
                @autoreleasepool
                {
                    if (!m_scaler || f.inW != m_inW || f.inH != m_inH || f.outW != m_outW || f.outH != m_outH)
                    {
                        MTLFXTemporalScalerDescriptor *desc = [[MTLFXTemporalScalerDescriptor alloc] init];
                        desc.inputWidth = f.inW;
                        desc.inputHeight = f.inH;
                        desc.outputWidth = f.outW;
                        desc.outputHeight = f.outH;
                        desc.colorTextureFormat = MTLPixelFormatRGBA8Unorm;
                        desc.depthTextureFormat = MTLPixelFormatR32Float;
                        desc.motionTextureFormat = MTLPixelFormatRG16Float;
                        desc.outputTextureFormat = MTLPixelFormatRGBA8Unorm;
                        m_scaler = [desc newTemporalScalerWithDevice:m_device];
                        if (!m_scaler)
                            return false;
                        m_inW = f.inW;
                        m_inH = f.inH;
                        m_outW = f.outW;
                        m_outH = f.outH;
                    }
                    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    m_scaler.colorTexture = (__bridge id<MTLTexture>)f.color;
                    m_scaler.depthTexture = (__bridge id<MTLTexture>)f.depth;
                    m_scaler.motionTexture = (__bridge id<MTLTexture>)f.motion;
                    m_scaler.outputTexture = (__bridge id<MTLTexture>)f.output;
                    m_scaler.inputContentWidth = f.inW;
                    m_scaler.inputContentHeight = f.inH;
                    m_scaler.jitterOffsetX = f.jitterX;
                    m_scaler.jitterOffsetY = f.jitterY;
                    m_scaler.motionVectorScaleX = f.motionScaleX;
                    m_scaler.motionVectorScaleY = f.motionScaleY;
                    m_scaler.depthReversed = YES;
                    m_scaler.reset = f.reset;
                    [m_scaler encodeToCommandBuffer:cb];
                    [cb commit];
                    [cb waitUntilCompleted];
                    return cb.status == MTLCommandBufferStatusCompleted;
                }
            }

        private:
            id<MTLDevice> m_device;
            id<MTLFXTemporalScaler> m_scaler = nil;
            uint32_t m_inW = 0, m_inH = 0, m_outW = 0, m_outH = 0;
        };
    }

    std::unique_ptr<MetalFxTemporal> MetalFxTemporal::create(void *mtlDevice)
    {
        if (@available(macOS 13.0, *))
        {
            id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
            if (device && [MTLFXTemporalScalerDescriptor supportsDevice:device])
                return std::make_unique<MetalFxTemporalImpl>(device);
        }
        return nullptr;
    }

    std::unique_ptr<MetalFxSpatial> MetalFxSpatial::create(void *mtlDevice)
    {
        if (@available(macOS 13.0, *))
        {
            id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
            if (device && [MTLFXSpatialScalerDescriptor supportsDevice:device])
                return std::make_unique<MetalFxSpatialImpl>(device);
        }
        return nullptr;
    }
}
