#pragma once

// Loads the upscaler plugins (FSR 3, DLSS, XeSS; runtime/gs/rt_upscaler_api.h) that sit next to the
// executable on Windows, and fronts them for the presenter. Elsewhere there are none: every call
// answers "nothing available".

#include "runtime/gs/rt_upscaler_api.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ps2x
{
    class UpscalerPlugins
    {
    public:
        UpscalerPlugins();
        ~UpscalerPlugins();
        UpscalerPlugins(const UpscalerPlugins &) = delete;
        UpscalerPlugins &operator=(const UpscalerPlugins &) = delete;

        // Finds and loads the plugins: rt_fsr3.dll and rt_vendor_upscalers.dll beside the executable,
        // or the ';'-separated paths in RT_UPSCALER_PLUGINS (tests). Idempotent.
        void load();
        bool loaded() const { return !m_plugins.empty(); }

        // What the plugins want enabled; the caller keeps the ones the driver has.
        std::vector<std::string> instanceExtensions();
        std::vector<std::string> deviceExtensions(void *instance, void *physicalDevice, void *getInstanceProcAddr);

        // After the device exists. Returns the RTU_KIND_ bits that work here.
        uint32_t init(const RtuInit &init);
        uint32_t supported() const { return m_supported; }
        float maxScale(uint32_t kind) const;
        std::string describe(uint32_t kind) const;

        // One context per size pair. Null on failure.
        struct Context;
        using ContextPtr = std::unique_ptr<Context, void (*)(Context *)>;
        ContextPtr create(const RtuCreateInfo &info);
        bool dispatch(Context *context, const RtuDispatch &dispatch);

        void shutdown();

    private:
        struct Plugin;
        std::vector<std::unique_ptr<Plugin>> m_plugins;
        uint32_t m_supported = 0;
        bool m_loaded = false;
    };
}
