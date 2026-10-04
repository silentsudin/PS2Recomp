#include "gs_metal_present.h"

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

namespace ps2x::gs
{
    namespace
    {
        class MetalPresentImpl final : public MetalPresent
        {
        public:
            MetalPresentImpl(id<MTLDevice> device, CAMetalLayer *layer, bool vsync) : m_layer(layer)
            {
                m_layer.device = device;
                m_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
                m_layer.framebufferOnly = NO; // the picture is copied in
                m_layer.maximumDrawableCount = 3;
                m_layer.displaySyncEnabled = vsync;
            }

            bool present(void *queue, void *texture, uint32_t width, uint32_t height, double atSeconds,
                         std::function<void(double)> onPresented) override
            {
                @autoreleasepool
                {
                    id<MTLTexture> src = (__bridge id<MTLTexture>)texture;
                    if (!src || !queue)
                        return false;
                    const CGSize size = CGSizeMake(width, height);
                    if (!CGSizeEqualToSize(m_layer.drawableSize, size))
                        m_layer.drawableSize = size;
                    id<CAMetalDrawable> drawable = [m_layer nextDrawable];
                    if (!drawable)
                        return false;
                    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    const NSUInteger w = std::min<NSUInteger>(src.width, drawable.texture.width);
                    const NSUInteger h = std::min<NSUInteger>(src.height, drawable.texture.height);
                    [blit copyFromTexture:src
                              sourceSlice:0
                              sourceLevel:0
                             sourceOrigin:MTLOriginMake(0, 0, 0)
                               sourceSize:MTLSizeMake(w, h, 1)
                                toTexture:drawable.texture
                         destinationSlice:0
                         destinationLevel:0
                        destinationOrigin:MTLOriginMake(0, 0, 0)];
                    [blit endEncoding];
                    if (onPresented)
                    {
                        auto callback = std::move(onPresented);
                        [drawable addPresentedHandler:^(id<MTLDrawable> d) {
                            callback(d.presentedTime);
                        }];
                    }
                    if (atSeconds > 0.0)
                        [cb presentDrawable:drawable atTime:atSeconds];
                    else
                        [cb presentDrawable:drawable];
                    [cb commit];
                    return true;
                }
            }

            void setVsync(bool vsync) override { m_layer.displaySyncEnabled = vsync; }

        private:
            CAMetalLayer *m_layer;
        };
    }

    std::unique_ptr<MetalPresent> MetalPresent::create(void *mtlDevice, void *layer, bool vsync)
    {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
        CAMetalLayer *metalLayer = (__bridge CAMetalLayer *)layer;
        if (!device || !metalLayer)
            return nullptr;
        return std::make_unique<MetalPresentImpl>(device, metalLayer, vsync);
    }
}
