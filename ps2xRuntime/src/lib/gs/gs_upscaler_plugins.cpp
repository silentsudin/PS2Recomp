#include "gs_upscaler_plugins.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace ps2x
{
    struct UpscalerPlugins::Plugin
    {
        std::string path;
#if defined(_WIN32)
        HMODULE module = nullptr;
#endif
        const RtuApi *api = nullptr;
        uint32_t kinds = 0; // what its init() said works
    };

    struct UpscalerPlugins::Context
    {
        UpscalerPlugins::Plugin *plugin = nullptr;
        void *handle = nullptr;
    };

    UpscalerPlugins::UpscalerPlugins() = default;
    UpscalerPlugins::~UpscalerPlugins() { shutdown(); }

#if defined(_WIN32)
    namespace
    {
        std::string narrow(const std::wstring &w)
        {
            if (w.empty())
                return {};
            const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
            std::string s(static_cast<size_t>(n), char(0));
            WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
            return s;
        }

        std::wstring widen(const std::string &s)
        {
            if (s.empty())
                return {};
            const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
            std::wstring w(static_cast<size_t>(n), wchar_t(0));
            MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
            return w;
        }

        // The executable's directory, with forward slashes.
        std::string exeDir()
        {
            std::wstring buf(32768, wchar_t(0));
            const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
            buf.resize(n);
            std::replace(buf.begin(), buf.end(), wchar_t(92), wchar_t('/'));
            const size_t slash = buf.find_last_of(wchar_t('/'));
            return narrow(slash == std::wstring::npos ? std::wstring(L".") : buf.substr(0, slash));
        }
    }

    void UpscalerPlugins::load()
    {
        if (m_loaded)
            return;
        m_loaded = true;
        std::vector<std::string> paths;
        if (const char *env = std::getenv("RT_UPSCALER_PLUGINS"); env && *env)
        {
            std::stringstream ss(env);
            for (std::string item; std::getline(ss, item, ';');)
                if (!item.empty())
                    paths.push_back(item);
        }
        else
        {
            const std::string dir = exeDir();
            for (const char *name : {"rt_fsr3.dll", "rt_vendor_upscalers.dll"})
                paths.push_back(dir + "/" + name);
        }
        for (const std::string &path : paths)
        {
            if (GetFileAttributesW(widen(path).c_str()) == INVALID_FILE_ATTRIBUTES)
                continue;
            // The DLL's own directory is searched for the vendor DLLs it needs.
            HMODULE module = LoadLibraryExW(widen(path).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (!module)
            {
                std::fprintf(stderr, "[upscaler] %s: could not load (error %lu)\n", path.c_str(), GetLastError());
                continue;
            }
            auto getApi = reinterpret_cast<RtuGetApiFn>(reinterpret_cast<void *>(GetProcAddress(module, "rtu_get_api")));
            const RtuApi *api = getApi ? getApi(RTU_API_VERSION) : nullptr;
            if (!api || api->version != RTU_API_VERSION)
            {
                std::fprintf(stderr, "[upscaler] %s: not an upscaler plugin of this version\n", path.c_str());
                FreeLibrary(module);
                continue;
            }
            auto plugin = std::make_unique<Plugin>();
            plugin->path = path;
            plugin->module = module;
            plugin->api = api;
            m_plugins.push_back(std::move(plugin));
        }
    }

    void UpscalerPlugins::shutdown()
    {
        for (auto &p : m_plugins)
        {
            if (p->api && p->api->shutdown)
                p->api->shutdown();
            if (p->module)
                FreeLibrary(p->module);
        }
        m_plugins.clear();
        m_supported = 0;
    }
#else
    void UpscalerPlugins::load() { m_loaded = true; }

    void UpscalerPlugins::shutdown()
    {
        m_plugins.clear();
        m_supported = 0;
    }
#endif

    std::vector<std::string> UpscalerPlugins::instanceExtensions()
    {
        std::vector<std::string> out;
        for (auto &p : m_plugins)
        {
            const char *const *names = nullptr;
            uint32_t count = 0;
            if (p->api->instance_extensions && p->api->instance_extensions(&names, &count) == 0)
                for (uint32_t i = 0; i < count; ++i)
                    out.emplace_back(names[i]);
        }
        return out;
    }

    std::vector<std::string> UpscalerPlugins::deviceExtensions(void *instance, void *physicalDevice, void *getInstanceProcAddr)
    {
        std::vector<std::string> out;
        for (auto &p : m_plugins)
        {
            const char *const *names = nullptr;
            uint32_t count = 0;
            if (p->api->device_extensions &&
                p->api->device_extensions(instance, physicalDevice, getInstanceProcAddr, &names, &count) == 0)
                for (uint32_t i = 0; i < count; ++i)
                    out.emplace_back(names[i]);
        }
        return out;
    }

    uint32_t UpscalerPlugins::init(const RtuInit &init)
    {
        m_supported = 0;
        for (auto &p : m_plugins)
        {
            p->kinds = p->api->init ? p->api->init(&init) : 0;
            m_supported |= p->kinds;
            std::fprintf(stderr, "[upscaler] %s: %s%s%s\n", p->path.c_str(), (p->kinds & RTU_KIND_FSR3) ? "FSR 3 " : "",
                         (p->kinds & RTU_KIND_DLSS) ? "DLSS " : "", (p->kinds & RTU_KIND_XESS) ? "XeSS " : "");
        }
        return m_supported;
    }

    float UpscalerPlugins::maxScale(uint32_t kind) const
    {
        for (auto &p : m_plugins)
            if ((p->kinds & kind) && p->api->max_scale)
                return p->api->max_scale(kind);
        return 1.0f;
    }

    std::string UpscalerPlugins::describe(uint32_t kind) const
    {
        for (auto &p : m_plugins)
            if ((p->kinds & kind) && p->api->describe)
                if (const char *s = p->api->describe(kind))
                    return s;
        return {};
    }

    UpscalerPlugins::ContextPtr UpscalerPlugins::create(const RtuCreateInfo &info)
    {
        auto destroy = [](Context *c)
        {
            if (c && c->handle && c->plugin->api->destroy)
                c->plugin->api->destroy(c->handle);
            delete c;
        };
        for (auto &p : m_plugins)
        {
            if (!(p->kinds & info.kind) || !p->api->create)
                continue;
            if (void *h = p->api->create(&info))
                return ContextPtr(new Context{p.get(), h}, destroy);
        }
        return ContextPtr(nullptr, destroy);
    }

    bool UpscalerPlugins::dispatch(Context *context, const RtuDispatch &dispatch)
    {
        return context && context->plugin->api->dispatch(context->handle, &dispatch) == 0;
    }
}
