// The Vulkan presenter (see ps2_host_presenter.h): an SDL3 window with a Granite WSI swapchain on
// the device paraLLEl-GS renders with. Each host frame draws the newest scanout image letterboxed
// to 4:3 and the host UI (Dear ImGui, drawn here through Granite with ImGui's own shaders), then
// presents. Nothing is read back to the CPU except for test captures and screenshots.

#include "runtime/ps2_display_clock.h"
#include "runtime/gs/gs_pgs_backend.h"
#include "runtime/ps2_host_presenter.h"

#if defined(PS2X_HAVE_PGS) && defined(PS2X_PGS_PRESENTER)

#include "gs_pgs_shared.h"
#include "runtime/gs/gs_motion.h"
#include "imgui_spirv.h"
#if defined(__APPLE__)
#include "gs_metalfx.h"
#include "gs_metal_present.h"
#endif
#include "post/post_spirv.h"
#if defined(PS2X_HAVE_ARM_ASR)
#include "ffxm_fsr2.h"
#include "ffxm_vk.h"
#endif

// FidelityFX FSR 1 constants, computed on the CPU (post/ffx, MIT).
#define A_CPU 1
#include "post/ffx/ffx_a.h"
#include "post/ffx/ffx_fsr1.h"
// SMAA's precomputed area and search textures (post/smaa, MIT).
#include "post/smaa/AreaTex.h"
#include "post/smaa/SearchTex.h"

#include "ps2_runtime.h"
#include "runtime/ee_scheduler.h"
#include "runtime/ps2_test_harness.h"

#include "context.hpp"
#include "device.hpp"
#include "wsi.hpp"

#include <SDL3/SDL_metal.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#if defined(__APPLE__)
#include <dlfcn.h>
#endif

#if defined(PS2X_ENABLE_DEBUG_UI)
#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#define PS2X_PGS_PRESENTER_UI 1
#endif

#include "raylib.h" // ExportImage for screenshots

#if defined(__ANDROID__)
#include <android/native_window.h>
#include <vulkan/vulkan_android.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>
#include <thread>
#include <condition_variable>
#include <deque>
#include <memory>
#include <vector>

namespace ps2x::gs
{
    namespace
    {
        // Present-at-time (VK_GOOGLE_display_timing; MoltenVK turns it into Metal's presentAtTime):
        // Granite presents through its device table, so its vkQueuePresentKHR is wrapped to add the
        // desired time the presenter set for this present.
        PFN_vkQueuePresentKHR g_realQueuePresent = nullptr;
        uint64_t g_desiredPresentNs = 0; // host media time; 0 = as soon as possible (presenter thread)
        uint64_t g_targetRefreshNs = 0;  // the refresh that present is aimed at
        uint32_t g_presentTimingId = 0;
        VkSwapchainKHR g_timedSwapchain = VK_NULL_HANDLE;
        // Recent presents: id -> the refresh it was aimed at (feedback for the present lead).
        struct TimedPresent
        {
            uint32_t id = 0;
            int64_t target = 0;
        };
        TimedPresent g_timedPresents[16];

        VKAPI_ATTR VkResult VKAPI_CALL queuePresentAtTime(VkQueue queue, const VkPresentInfoKHR *info)
        {
            const uint64_t desired = g_desiredPresentNs;
            g_desiredPresentNs = 0;
            if (!desired || !info || info->swapchainCount != 1)
                return g_realQueuePresent(queue, info);
            g_timedSwapchain = info->pSwapchains[0];
            const VkPresentTimeGOOGLE time = {++g_presentTimingId, desired};
            g_timedPresents[g_presentTimingId % 16] = {g_presentTimingId, static_cast<int64_t>(g_targetRefreshNs)};
            VkPresentTimesInfoGOOGLE times = {VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE, info->pNext, 1, &time};
            VkPresentInfoKHR withTime = *info;
            withTime.pNext = &times;
            return g_realQueuePresent(queue, &withTime);
        }

        // Presenting off the render thread. vkQueuePresentKHR can block in the driver for many
        // milliseconds (Adreno sleeps in queueBuffer), and Granite presents with its queue lock held
        // (and the presenter with the device lock): the hardware GS, which needs both, then misses
        // the game's vblank (Peach Town's main roads dropped to 20-45 fps). Presents are handed to a
        // thread of their own, on a queue of their own when the device has a spare one in the
        // graphics family (made with the device, PresentDeviceFactory); the swapchain calls that need
        // its external synchronisation (acquire, past timing) take the same mutex as the presents,
        // and whatever must not overlap a present (destroying or recreating a swapchain, waiting for
        // the device) first lets the queued presents go out. A present's result reaches its caller
        // with the next present to that swapchain. Android by default; RT_PRESENT_THREAD=0|1.
        bool presentThreadWanted()
        {
            static const bool on = [] {
                const char *e = std::getenv("RT_PRESENT_THREAD");
#if defined(__ANDROID__)
                return !(e && *e == '0');
#else
                return e && *e == '1';
#endif
            }();
            return on;
        }

        // RT_PRESENT_LOG=1: one line per present ([plog]): what the render thread made (tag: R real,
        // S shadow, fallbacks w = no shadow published, s = stale serial, d = 2D/size, r = repeat),
        // its guest vblank and subframe, and when it was rendered, queued and handed to the driver.
        static bool presentLogOn()
        {
            static const bool on = [] { const char *e = std::getenv("RT_PRESENT_LOG"); return e && *e == '1'; }();
            return on;
        }
        static int64_t plNow()
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        }
        std::atomic<char> g_plTag{'?'};
        std::atomic<uint64_t> g_plTick{0};
        std::atomic<uint32_t> g_plSub{0};
        std::atomic<int64_t> g_plRender{0};
        std::atomic<uint64_t> g_plSeq{0};
        std::atomic<uint64_t> g_plPic{0}, g_plPres{0};

        class PresentQueue
        {
        public:
            void start(VkDevice device, VkQueue dedicated, VkQueue shared, Vulkan::Device *granite, PFN_vkQueuePresentKHR present)
            {
                // Only on a queue of its own: on Granite's it would need Granite's queue lock, which
                // Device::wait_idle holds while it waits for the device (and drains this thread).
                if (!dedicated)
                {
                    std::fprintf(stderr, "[present] no spare queue: presents stay on the render thread\n");
                    return;
                }
                m_device = device;
                m_queue = dedicated ? dedicated : shared;
                m_dedicated = dedicated != VK_NULL_HANDLE;
                m_granite = granite;
                m_present = present;
                m_stop = false;
                m_thread = std::thread([this] { run(); });
                std::fprintf(stderr, "[present] presents on a thread of their own, %s\n",
                             m_dedicated ? "on a queue of their own" : "on the graphics queue (no spare queue)");
            }

            void stop()
            {
                if (!m_thread.joinable())
                    return;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_stop = true;
                }
                m_cv.notify_all();
                m_thread.join();
            }

            bool running() const { return m_thread.joinable(); }

            // Queues a present (a copy of it); false if it can't be copied (then present it directly).
            bool push(const VkPresentInfoKHR *info, VkResult &result)
            {
                if (!info || info->swapchainCount != 1 || info->waitSemaphoreCount > 4)
                    return false;
                Job job;
                job.tag = g_plTag.load(), job.tick = g_plTick.load(), job.sub = g_plSub.load(), job.renderUs = g_plRender.load();
                job.pushUs = plNow(), job.seq = g_plSeq.load(), job.pic = g_plPic.load(), job.pres = g_plPres.load();
                job.swapchain = info->pSwapchains[0];
                job.index = info->pImageIndices[0];
                job.waitCount = info->waitSemaphoreCount;
                for (uint32_t i = 0; i < job.waitCount; ++i)
                    job.wait[i] = info->pWaitSemaphores[i];
                for (const auto *n = static_cast<const VkBaseInStructure *>(info->pNext); n; n = n->pNext)
                {
                    switch (n->sType)
                    {
                    case VK_STRUCTURE_TYPE_PRESENT_ID_KHR:
                    {
                        const auto *id = reinterpret_cast<const VkPresentIdKHR *>(n);
                        if (id->swapchainCount != 1 || !id->pPresentIds)
                            return false;
                        job.hasId = true, job.id = id->pPresentIds[0];
                        break;
                    }
                    case VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR:
                    {
                        const auto *id = reinterpret_cast<const VkPresentId2KHR *>(n);
                        if (id->swapchainCount != 1 || !id->pPresentIds)
                            return false;
                        job.hasId2 = true, job.id = id->pPresentIds[0];
                        break;
                    }
                    case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR:
                    {
                        const auto *f = reinterpret_cast<const VkSwapchainPresentFenceInfoKHR *>(n);
                        if (f->swapchainCount != 1 || !f->pFences)
                            return false;
                        job.hasFence = true, job.fence = f->pFences[0];
                        break;
                    }
                    case VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR:
                    {
                        const auto *m = reinterpret_cast<const VkSwapchainPresentModeInfoKHR *>(n);
                        if (m->swapchainCount != 1 || !m->pPresentModes)
                            return false;
                        job.hasMode = true, job.mode = m->pPresentModes[0];
                        break;
                    }
                    case VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE:
                    {
                        const auto *t = reinterpret_cast<const VkPresentTimesInfoGOOGLE *>(n);
                        if (t->swapchainCount != 1 || !t->pTimes)
                            return false;
                        job.hasTime = true, job.time = t->pTimes[0];
                        break;
                    }
                    default:
                        return false; // (unknown: presented directly)
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    auto it = m_results.find(job.swapchain);
                    result = it != m_results.end() ? it->second : VK_SUCCESS;
                    if (it != m_results.end())
                        m_results.erase(it);
                    m_jobs.push_back(job);
                    ++m_queued;
                }
                m_cv.notify_one();
                return true;
            }

            // Until every queued present has gone to the driver.
            void drain()
            {
                if (!running() || std::this_thread::get_id() == m_thread.get_id())
                    return;
                std::unique_lock<std::mutex> lock(m_mutex);
                m_doneCv.wait(lock, [this] { return m_issued == m_queued || m_stop; });
            }

            // Until at most `pending` presents wait for the thread (the presenter keeps no backlog).
            void waitBelow(uint64_t pending)
            {
                if (!running())
                    return;
                std::unique_lock<std::mutex> lock(m_mutex);
                m_doneCv.wait(lock, [&] { return m_queued - m_issued <= pending || m_stop; });
            }

            // Held around calls on a swapchain that need external synchronisation with its presents
            // (one per swapchain: the second screen's acquire doesn't wait for the main present).
            std::mutex &swapchainMutex(VkSwapchainKHR swapchain)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto &m = m_swapchainMutexes[swapchain];
                if (!m)
                    m = std::make_unique<std::mutex>();
                return *m;
            }
            // Held while presenting: nothing else may use a queue then (vkDeviceWaitIdle).
            std::mutex &presentMutex() { return m_presentMutex; }
            bool dedicated() const { return m_dedicated; }

        private:
            struct Job
            {
                VkSwapchainKHR swapchain = VK_NULL_HANDLE;
                uint32_t index = 0, waitCount = 0;
                VkSemaphore wait[4] = {};
                bool hasId = false, hasId2 = false, hasFence = false, hasMode = false, hasTime = false;
                uint64_t id = 0;
                VkFence fence = VK_NULL_HANDLE;
                VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
                VkPresentTimeGOOGLE time = {};
                char tag = '?';
                uint64_t tick = 0, seq = 0, pic = 0, pres = 0;
                uint32_t sub = 0;
                int64_t renderUs = 0, pushUs = 0;
            };

            void run()
            {
                for (;;)
                {
                    Job job;
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_cv.wait(lock, [this] { return m_stop || !m_jobs.empty(); });
                        if (m_jobs.empty())
                            break;
                        job = m_jobs.front();
                        m_jobs.pop_front();
                    }
                    VkPresentInfoKHR info = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
                    info.waitSemaphoreCount = job.waitCount;
                    info.pWaitSemaphores = job.wait;
                    info.swapchainCount = 1;
                    info.pSwapchains = &job.swapchain;
                    info.pImageIndices = &job.index;
                    const void *chain = nullptr;
                    VkPresentIdKHR id = {VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
                    VkPresentId2KHR id2 = {VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR};
                    VkSwapchainPresentFenceInfoKHR fence = {VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR};
                    VkSwapchainPresentModeInfoKHR mode = {VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR};
                    VkPresentTimesInfoGOOGLE time = {VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE};
                    if (job.hasId)
                        id.pNext = chain, id.swapchainCount = 1, id.pPresentIds = &job.id, chain = &id;
                    if (job.hasId2)
                        id2.pNext = chain, id2.swapchainCount = 1, id2.pPresentIds = &job.id, chain = &id2;
                    if (job.hasFence)
                        fence.pNext = chain, fence.swapchainCount = 1, fence.pFences = &job.fence, chain = &fence;
                    if (job.hasMode)
                        mode.pNext = chain, mode.swapchainCount = 1, mode.pPresentModes = &job.mode, chain = &mode;
                    if (!job.hasTime && presentLogOn())
                        job.hasTime = true, job.time = {static_cast<uint32_t>(job.seq), 0};
                    if (job.hasTime)
                        time.pNext = chain, time.swapchainCount = 1, time.pTimes = &job.time, chain = &time;
                    info.pNext = chain;
                    VkResult r;
                    {
                        std::lock_guard<std::mutex> presenting(m_presentMutex);
                        std::lock_guard<std::mutex> swapchain(swapchainMutex(job.swapchain));
                        if (m_dedicated)
                            r = m_present(m_queue, &info);
                        else
                        {
                            m_granite->external_queue_lock();
                            r = m_present(m_queue, &info);
                            m_granite->external_queue_unlock();
                        }
                    }
                    if (presentLogOn())
                    {
                        static PFN_vkGetPastPresentationTimingGOOGLE past =
                            reinterpret_cast<PFN_vkGetPastPresentationTimingGOOGLE>(vkGetDeviceProcAddr(m_device, "vkGetPastPresentationTimingGOOGLE"));
                        if (past)
                        {
                            VkPastPresentationTimingGOOGLE t[16];
                            uint32_t n = 16;
                            std::lock_guard<std::mutex> swapchain(swapchainMutex(job.swapchain));
                            if (past(m_device, job.swapchain, &n, t) >= 0)
                                for (uint32_t i = 0; i < n; ++i)
                                    std::fprintf(stderr, "[past] %u actual %llu earliest %llu margin %llu sc %llx\n", t[i].presentID,
                                                 (unsigned long long)t[i].actualPresentTime, (unsigned long long)t[i].earliestPresentTime,
                                                 (unsigned long long)t[i].presentMargin, (unsigned long long)(uintptr_t)job.swapchain & 0xFFFFull);
                        }
                    }
                    if (presentLogOn())
                        std::fprintf(stderr, "[plog] %llu %c v%llu.%u pic %llu pres %llu render %lld queued %lld done %lld sc %llx\n",
                                     (unsigned long long)job.seq, job.tag, (unsigned long long)job.tick, job.sub, (unsigned long long)job.pic, (unsigned long long)job.pres,
                                     (long long)job.renderUs, (long long)job.pushUs, (long long)plNow(),
                                     (unsigned long long)(uintptr_t)job.swapchain & 0xFFFFull);
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        if (r != VK_SUCCESS)
                            m_results[job.swapchain] = r;
                        ++m_issued;
                    }
                    m_doneCv.notify_all();
                }
                std::lock_guard<std::mutex> lock(m_mutex);
                m_issued = m_queued;
                m_doneCv.notify_all();
            }

            VkDevice m_device = VK_NULL_HANDLE;
            VkQueue m_queue = VK_NULL_HANDLE;
            bool m_dedicated = false;
            Vulkan::Device *m_granite = nullptr;
            PFN_vkQueuePresentKHR m_present = nullptr;
            std::thread m_thread;
            std::mutex m_mutex, m_presentMutex;
            std::unordered_map<VkSwapchainKHR, std::unique_ptr<std::mutex>> m_swapchainMutexes;
            std::condition_variable m_cv, m_doneCv;
            std::deque<Job> m_jobs;
            std::unordered_map<VkSwapchainKHR, VkResult> m_results;
            uint64_t m_queued = 0, m_issued = 0;
            bool m_stop = false;
        };
        PresentQueue g_presentQueue;
        // The device table's originals the wrappers below call.
        PFN_vkQueuePresentKHR g_threadRealPresent = nullptr;
        PFN_vkAcquireNextImageKHR g_realAcquire = nullptr;
        PFN_vkDestroySwapchainKHR g_realDestroySwapchain = nullptr;
        PFN_vkCreateSwapchainKHR g_realCreateSwapchain = nullptr;
        PFN_vkDeviceWaitIdle g_realDeviceWaitIdle = nullptr;
        PFN_vkWaitForPresentKHR g_realWaitForPresent = nullptr;

        VKAPI_ATTR VkResult VKAPI_CALL threadedPresent(VkQueue queue, const VkPresentInfoKHR *info)
        {
            VkResult result = VK_SUCCESS;
            if (g_presentQueue.running() && g_presentQueue.push(info, result))
            {
                if (info->pResults)
                    info->pResults[0] = result;
                return result;
            }
            // (Not copyable: in order, after the queued ones, on the caller's queue.)
            g_presentQueue.drain();
            std::lock_guard<std::mutex> presenting(g_presentQueue.presentMutex());
            return g_threadRealPresent(queue, info);
        }
        VKAPI_ATTR VkResult VKAPI_CALL threadedAcquire(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout,
                                                       VkSemaphore semaphore, VkFence fence, uint32_t *index)
        {
            // A non-blocking acquire (the second screen's, under the device lock) doesn't wait for a
            // present in the driver either: not ready this time.
            std::unique_lock<std::mutex> lock(g_presentQueue.swapchainMutex(swapchain), std::defer_lock);
            if (timeout == 0)
            {
                if (!lock.try_lock())
                    return VK_NOT_READY;
            }
            else
                lock.lock();
            return g_realAcquire(device, swapchain, timeout, semaphore, fence, index);
        }
        VKAPI_ATTR void VKAPI_CALL threadedDestroySwapchain(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks *alloc)
        {
            g_presentQueue.drain();
            std::lock_guard<std::mutex> presenting(g_presentQueue.presentMutex());
            std::lock_guard<std::mutex> lock(g_presentQueue.swapchainMutex(swapchain));
            g_realDestroySwapchain(device, swapchain, alloc);
        }
        VKAPI_ATTR VkResult VKAPI_CALL threadedCreateSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR *info,
                                                               const VkAllocationCallbacks *alloc, VkSwapchainKHR *swapchain)
        {
            g_presentQueue.drain(); // (the old swapchain's presents first)
            std::lock_guard<std::mutex> lock(g_presentQueue.presentMutex());
            return g_realCreateSwapchain(device, info, alloc, swapchain);
        }
        VKAPI_ATTR VkResult VKAPI_CALL threadedDeviceWaitIdle(VkDevice device)
        {
            g_presentQueue.drain();
            std::lock_guard<std::mutex> lock(g_presentQueue.presentMutex()); // (every queue: none presenting)
            return g_realDeviceWaitIdle(device);
        }
        VKAPI_ATTR VkResult VKAPI_CALL threadedWaitForPresent(VkDevice device, VkSwapchainKHR swapchain, uint64_t id, uint64_t timeout)
        {
            g_presentQueue.drain(); // (the present it waits for has gone out)
            return g_realWaitForPresent(device, swapchain, id, timeout);
        }

        // Makes the device with one more queue in the graphics family, when it has one to spare,
        // for the presents (PresentQueue).
        class PresentDeviceFactory final : public Vulkan::DeviceFactory
        {
        public:
            VkDevice create_device(VkPhysicalDevice gpu, const VkDeviceCreateInfo *info) override
            {
                uint32_t familyCount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(gpu, &familyCount, nullptr);
                std::vector<VkQueueFamilyProperties> families(familyCount);
                vkGetPhysicalDeviceQueueFamilyProperties(gpu, &familyCount, families.data());
                std::vector<VkDeviceQueueCreateInfo> queues(info->pQueueCreateInfos, info->pQueueCreateInfos + info->queueCreateInfoCount);
                std::vector<float> priorities;
                for (auto &q : queues)
                {
                    if (q.queueFamilyIndex >= familyCount || !(families[q.queueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) ||
                        q.queueCount >= families[q.queueFamilyIndex].queueCount)
                        continue;
                    priorities.assign(q.pQueuePriorities, q.pQueuePriorities + q.queueCount);
                    priorities.push_back(1.0f);
                    m_family = q.queueFamilyIndex;
                    m_index = q.queueCount;
                    q.pQueuePriorities = priorities.data();
                    ++q.queueCount;
                    break;
                }
                std::fprintf(stderr, "[present] graphics family: %s\n", m_index != UINT32_MAX ? "a spare queue for presents" : "no spare queue");
                VkDeviceCreateInfo copy = *info;
                copy.pQueueCreateInfos = queues.data();
                VkDevice device = VK_NULL_HANDLE;
                if (vkCreateDevice(gpu, &copy, nullptr, &device) != VK_SUCCESS)
                    return VK_NULL_HANDLE;
                if (m_index != UINT32_MAX)
                {
                    auto getQueue = reinterpret_cast<PFN_vkGetDeviceQueue>(vkGetDeviceProcAddr(device, "vkGetDeviceQueue"));
                    getQueue(device, m_family, m_index, &m_queue);
                }
                return device;
            }
            VkQueue presentQueue() const { return m_queue; }

        private:
            uint32_t m_family = UINT32_MAX, m_index = UINT32_MAX;
            VkQueue m_queue = VK_NULL_HANDLE;
        };

        bool presentAtTimeWanted()
        {
#if defined(__APPLE__)
            static const bool on = [] {
                const char *e = std::getenv("RT_PRESENT_AT_TIME");
                return !(e && *e == '0');
            }();
            return on;
#else
            return false;
#endif
        }

        class SdlPlatform final : public Vulkan::WSIPlatform
        {
        public:
            std::vector<const char *> get_device_extensions() override
            {
                if (presentAtTimeWanted() || presentLogOn())
                    return {"VK_KHR_swapchain", "VK_GOOGLE_display_timing"};
                return {"VK_KHR_swapchain"};
            }

            SdlPlatform(SDL_Window *window, bool surface) : m_window(window), m_surface(surface) {}

            VkSurfaceKHR create_surface(VkInstance instance, VkPhysicalDevice) override
            {
                VkSurfaceKHR surface = VK_NULL_HANDLE;
                if (!m_surface) // presenting through Metal: no Vulkan swapchain on this window
                    return VK_NULL_HANDLE;
                if (!SDL_Vulkan_CreateSurface(m_window, instance, nullptr, &surface))
                {
                    std::cerr << "[presenter] SDL_Vulkan_CreateSurface: " << SDL_GetError() << std::endl;
                    return VK_NULL_HANDLE;
                }
                return surface;
            }

            void destroy_surface(VkInstance instance, VkSurfaceKHR surface) override
            {
                SDL_Vulkan_DestroySurface(instance, surface, nullptr);
            }

            std::vector<const char *> get_instance_extensions() override
            {
                Uint32 count = 0;
                const char *const *names = SDL_Vulkan_GetInstanceExtensions(&count);
                return names ? std::vector<const char *>(names, names + count) : std::vector<const char *>{};
            }

            uint32_t get_surface_width() override
            {
                int w = 0, h = 0;
                SDL_GetWindowSizeInPixels(m_window, &w, &h);
                return static_cast<uint32_t>(std::max(w, 1));
            }

            uint32_t get_surface_height() override
            {
                int w = 0, h = 0;
                SDL_GetWindowSizeInPixels(m_window, &w, &h);
                return static_cast<uint32_t>(std::max(h, 1));
            }

            bool alive(Vulkan::WSI &) override { return true; }
            void poll_input() override {}
            void poll_input_async(Granite::InputTrackerHandler *) override {}

            void requestResize() { resize = true; }

        private:
            SDL_Window *m_window;
            bool m_surface;
        };

        // ImGui's vertex layout, also used for the game picture's quad.
        struct Vertex
        {
            float x, y, u, v;
            uint32_t colour;
        };

        class PgsPresenter final : public HostPresenter
        {
        public:
            explicit PgsPresenter(PgsPresenterOptions options) : m_options(std::move(options)) {}
            ~PgsPresenter() override
            {
                close();
                g_presentQueue.drain();
                g_presentQueue.stop();
            }

            const char *name() const override { return "vulkan"; }

            PgsShared *shared() { return &m_shared; }

            bool open(const char *title, int width, int height) override
            {
                if (m_window)
                    return true; // opened early (the app's setup screen), now handed to the runtime
                if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
                    return fail("SDL video", SDL_GetError());
                if (!SDL_Vulkan_LoadLibrary(m_options.vulkanLibrary.empty() ? nullptr : m_options.vulkanLibrary.c_str()))
                    return fail("SDL_Vulkan_LoadLibrary", SDL_GetError());
#if defined(__APPLE__)
                // macOS presents through Metal (gs_metal_present.h); RT_METAL_PRESENT=0 uses the
                // Vulkan swapchain (MoltenVK) as other platforms do.
                if (const char *e = std::getenv("RT_METAL_PRESENT"); !(e && *e == '0'))
                    m_metalPresent = true;
#endif
                SDL_WindowFlags flags = (m_metalPresent ? SDL_WINDOW_METAL : SDL_WINDOW_VULKAN) | SDL_WINDOW_RESIZABLE |
                                        SDL_WINDOW_HIGH_PIXEL_DENSITY;
#if defined(__ANDROID__)
                flags |= SDL_WINDOW_FULLSCREEN; // immersive: no status or navigation bar
#endif
                m_window = SDL_CreateWindow(title, width, height, flags);
                if (!m_window)
                    return fail("SDL_CreateWindow", SDL_GetError());

                // Granite and SDL use the same Vulkan library (SDL's loader).
                auto getProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
                if (!getProc || !Vulkan::Context::init_loader(getProc))
                    return fail("Vulkan loader", "vkGetInstanceProcAddr unavailable");

                pgsRegisterThread();
                m_platform = std::make_unique<SdlPlatform>(m_window, !m_metalPresent);
                m_wsi.set_platform(m_platform.get());
                m_wsi.set_present_mode(m_options.vsync ? Vulkan::PresentMode::SyncToVBlank
                                                       : Vulkan::PresentMode::UnlockedMaybeTear);
                m_wsi.set_backbuffer_format(Vulkan::BackbufferFormat::UNORM);
                // begin_frame waits (VK_KHR_present_wait) until the present this many frames back is
                // on screen, under the device lock the GS thread needs. At one present per display
                // refresh, 1 makes that wait about a refresh long and the game misses frames.
                m_wsi.set_present_wait_latency(2);
                Vulkan::Context::SystemHandles handles = {};
                // As the GS's own device: push descriptors, no descriptor buffers/heaps.
                const bool presentThread = presentThreadWanted() && !m_metalPresent;
                if (presentThread)
                {
                    // As WSI::init_context_from_platform, with a device made by PresentDeviceFactory.
                    const Vulkan::ContextCreationFlags flags =
                        (Vulkan::CONTEXT_CREATION_ENABLE_ADVANCED_WSI_BIT | Vulkan::CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT);
                    auto context = Util::make_handle<Vulkan::Context>();
                    context->set_application_info(m_platform->get_application_info());
                    context->set_num_thread_indices(1);
                    context->set_system_handles(handles);
                    context->set_device_factory(&m_deviceFactory);
                    auto instanceExt = m_platform->get_instance_extensions();
                    auto deviceExt = m_platform->get_device_extensions();
                    if (!context->init_instance(instanceExt.data(), instanceExt.size(), flags))
                        return fail("Vulkan", "instance creation failed");
                    VkSurfaceKHR probe = m_platform->create_surface(context->get_instance(), VK_NULL_HANDLE);
                    const bool made = context->init_device(VK_NULL_HANDLE, probe, deviceExt.data(), deviceExt.size(), flags);
                    if (probe)
                        m_platform->destroy_surface(context->get_instance(), probe);
                    if (!made || !m_wsi.init_from_existing_context(std::move(context)))
                        return fail("Vulkan", "device creation failed");
                }
                else if (!m_wsi.init_context_from_platform(1, handles, Vulkan::CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT,
                                                           Vulkan::CONTEXT_CREATION_ENABLE_DESCRIPTOR_BUFFER_BIT |
                                                               Vulkan::CONTEXT_CREATION_ENABLE_DESCRIPTOR_HEAP_BIT))
                    return fail("Vulkan", "instance/device creation failed");
                if (presentThread)
                {
                    // Before the swapchain: its creation goes through the wrappers too.
                    auto &table = const_cast<VolkDeviceTable &>(m_wsi.get_context().get_device_table());
                    g_threadRealPresent = table.vkQueuePresentKHR;
                    g_realAcquire = table.vkAcquireNextImageKHR;
                    g_realDestroySwapchain = table.vkDestroySwapchainKHR;
                    g_realCreateSwapchain = table.vkCreateSwapchainKHR;
                    g_realDeviceWaitIdle = table.vkDeviceWaitIdle;
                    g_realWaitForPresent = table.vkWaitForPresentKHR;
                    table.vkQueuePresentKHR = threadedPresent;
                    table.vkAcquireNextImageKHR = threadedAcquire;
                    table.vkDestroySwapchainKHR = threadedDestroySwapchain;
                    table.vkCreateSwapchainKHR = threadedCreateSwapchain;
                    table.vkDeviceWaitIdle = threadedDeviceWaitIdle;
                    if (g_realWaitForPresent)
                        table.vkWaitForPresentKHR = threadedWaitForPresent;
                }
                if (!m_wsi.init_device() || (!m_metalPresent && !m_wsi.init_surface_swapchain()))
                    return fail("Vulkan", "swapchain creation failed");
                if (presentThread)
                    g_presentQueue.start(m_wsi.get_context().get_device(), m_deviceFactory.presentQueue(),
                                         m_wsi.get_context().get_queue_info().queues[Vulkan::QUEUE_INDEX_GRAPHICS],
                                         &m_wsi.get_device(), g_threadRealPresent);
                if (presentAtTimeWanted() && m_metalPresent)
                    m_presentAtTime = true; // Metal's presentDrawable:atTime:
                else if (presentAtTimeWanted())
                {
                    auto &table = const_cast<VolkDeviceTable &>(m_wsi.get_context().get_device_table());
                    g_realQueuePresent = table.vkQueuePresentKHR;
                    table.vkQueuePresentKHR = queuePresentAtTime;
                    m_presentAtTime = true;
                }
                m_shared.device = &m_wsi.get_device();
                // paraLLEl-GS advances a frame context on every flush, and the swapchain on every
                // frame: with Granite's default of 2 the game thread waits for the GPU inside a
                // flush. 4, as the GS's own device had.
                m_shared.device->init_frame_contexts(8);

                Vulkan::ResourceLayout vert = {};
                vert.input_mask = 0x7;
                vert.output_mask = 0x3;
                vert.push_constant_size = 16;
                Vulkan::ResourceLayout frag = {};
                frag.input_mask = 0x3;
                frag.output_mask = 0x1;
                frag.sets[0].sampled_image_mask = 0x1;
                frag.sets[0].fp_mask = 0x1;
                m_program = m_shared.device->request_program(imgui_spirv::__glsl_shader_vert_spv,
                                                             sizeof(imgui_spirv::__glsl_shader_vert_spv),
                                                             imgui_spirv::__glsl_shader_frag_spv,
                                                             sizeof(imgui_spirv::__glsl_shader_frag_spv), &vert, &frag);
                if (!m_program)
                    return fail("Vulkan", "presenter shaders");
                // Post-processing: a fullscreen triangle and one fragment shader per pass.
                Vulkan::ResourceLayout fsVert = {};
                fsVert.output_mask = 0x1;
                auto post = [&](const uint32_t *code, size_t size, uint32_t push, uint32_t inputs, uint32_t textures = 0x1) {
                    Vulkan::ResourceLayout frag = {};
                    frag.input_mask = inputs;
                    frag.output_mask = 0x1;
                    frag.push_constant_size = push;
                    frag.sets[0].sampled_image_mask = textures;
                    frag.sets[0].fp_mask = textures;
                    return m_shared.device->request_program(post_spirv::fullscreen_vert, sizeof(post_spirv::fullscreen_vert),
                                                            code, size, &fsVert, &frag);
                };
                m_fxaa = post(post_spirv::fxaa_frag, sizeof(post_spirv::fxaa_frag), 8, 0x1);
                // EASU in 16-bit arithmetic where the device has it (about half the cost on
                // mobile GPUs: 1.5 ms a present at the Thor's 401 MHz); RT_FSR_FP32=1 keeps 32-bit.
                const bool fp16 = m_shared.device->get_device_features().vk12_features.shaderFloat16 && !std::getenv("RT_FSR_FP32");
                m_easu = fp16 ? post(post_spirv::fsr_easu_h_frag, sizeof(post_spirv::fsr_easu_h_frag), 80, 0)
                              : post(post_spirv::fsr_easu_frag, sizeof(post_spirv::fsr_easu_frag), 80, 0);
                m_rcas = post(post_spirv::fsr_rcas_frag, sizeof(post_spirv::fsr_rcas_frag), 32, 0);
                m_sgsr1 = post(post_spirv::sgsr1_frag, sizeof(post_spirv::sgsr1_frag), 16, 0x1);
                m_sgsr2Convert = post(post_spirv::sgsr2_convert_frag, sizeof(post_spirv::sgsr2_convert_frag), sizeof(Sgsr2Push), 0x1, 0x7);
                m_sgsr2Upscale = post(post_spirv::sgsr2_upscale_frag, sizeof(post_spirv::sgsr2_upscale_frag), sizeof(Sgsr2Push), 0x1, 0x7);
                m_smaaEdges = post(post_spirv::smaa_edges_frag, sizeof(post_spirv::smaa_edges_frag), 16, 0x1, 0x1);
                m_smaaWeights = post(post_spirv::smaa_weights_frag, sizeof(post_spirv::smaa_weights_frag), 16, 0x1, 0x7);
                m_smaaBlend = post(post_spirv::smaa_blend_frag, sizeof(post_spirv::smaa_blend_frag), 16, 0x1, 0x3);
                m_depthView = post(post_spirv::depth_view_frag, sizeof(post_spirv::depth_view_frag), 4, 0x1);
                m_motionView = post(post_spirv::motion_view_frag, sizeof(post_spirv::motion_view_frag), 4, 0x1);
                m_taa = post(post_spirv::taa_frag, sizeof(post_spirv::taa_frag), 32, 0x1, 0x7);
                m_depthNormalize = post(post_spirv::depth_normalize_frag, sizeof(post_spirv::depth_normalize_frag), 0, 0x1);
                m_uiComposite = post(post_spirv::ui_composite_frag, sizeof(post_spirv::ui_composite_frag), 0, 0x1, 0x7);
                m_rcasUi = post(post_spirv::rcas_ui_frag, sizeof(post_spirv::rcas_ui_frag), 48, 0, 0x7);
                m_frameGen = post(post_spirv::frame_gen_frag, sizeof(post_spirv::frame_gen_frag), 32, 0x1, 0xF);
                m_copy = post(post_spirv::copy_frag, sizeof(post_spirv::copy_frag), 0, 0x1);
                m_flicker = post(post_spirv::flicker_frag, sizeof(post_spirv::flicker_frag), 0, 0x1, 0x7);
                m_flickerWeight = post(post_spirv::flicker_weight_frag, sizeof(post_spirv::flicker_weight_frag), 8, 0x1, 0xF);
                m_flickerGrid = post(post_spirv::flicker_grid_frag, sizeof(post_spirv::flicker_grid_frag), 0, 0x1, 0x3);
                m_flickerCut = post(post_spirv::flicker_cut_frag, sizeof(post_spirv::flicker_cut_frag), 0, 0x0, 0x1);
                if (const char *e = std::getenv("RT_SHOW_MOTION"); e && *e == '1')
                {
                    m_showMotion = true;
                    m_shared.wantDepth = true;
                    m_shared.wantMotion = true;
                    MotionTracker::instance().setEnabled(true);
                }
                if (const char *e = std::getenv("RT_SHOW_DEPTH"); e && *e == '1')
                {
                    m_showDepth = true;
                    m_shared.wantDepth = true;
                }
                if (const char *e = std::getenv("RT_SHOW_UI"); e && *e == '1')
                    m_showUi = true; // the UI mask in place of the picture
#if defined(__APPLE__)
                // MetalFX through MoltenVK's Metal objects.
                {
                    // MoltenVK's own functions: from the library SDL loaded (they are not instance
                    // commands, so vkGetInstanceProcAddr may not know them).
                    void *lib = dlopen(m_options.vulkanLibrary.empty() ? "libMoltenVK.dylib" : m_options.vulkanLibrary.c_str(),
                                       RTLD_NOW | RTLD_NOLOAD);
                    auto sym = [&](const char *name) -> void * {
                        void *f = lib ? dlsym(lib, name) : nullptr;
                        return f ? f : reinterpret_cast<void *>(getProc(m_wsi.get_context().get_instance(), name));
                    };
                    using GetDevice = void (*)(VkPhysicalDevice, void **);
                    auto getDevice = reinterpret_cast<GetDevice>(sym("vkGetMTLDeviceMVK"));
                    m_getTexture = reinterpret_cast<GetTexture>(sym("vkGetMTLTextureMVK"));
                    m_getQueue = reinterpret_cast<GetQueue>(sym("vkGetMTLCommandQueueMVK"));
                    void *mtlDevice = nullptr;
                    if (getDevice && m_getTexture && m_getQueue)
                        getDevice(m_wsi.get_context().get_gpu(), &mtlDevice);
                    m_mtlDevice = mtlDevice;
                    if (mtlDevice)
                    {
                        m_metalfx = MetalFxSpatial::create(mtlDevice);
                        m_metalfxTemporal = MetalFxTemporal::create(mtlDevice);
                    }
                    if (m_metalPresent)
                    {
                        m_metalView = SDL_Metal_CreateView(m_window);
                        if (mtlDevice && m_metalView)
                            m_metal = MetalPresent::create(mtlDevice, SDL_Metal_GetLayer(m_metalView), m_options.vsync);
                        if (!m_metal)
                            return fail("Metal", "could not present through the window's CAMetalLayer");
                    }
                    std::cout << "[presenter] MetalFX spatial "
                              << (m_metalfx ? "available"
                                            : !getDevice ? "unavailable (no MoltenVK Metal interop)"
                                            : !mtlDevice ? "unavailable (no Metal device)"
                                                         : "unavailable (not supported on this GPU/OS)")
                              << std::endl;
                }
#endif
                if (!m_fxaa || !m_easu || !m_rcas || !m_smaaEdges || !m_smaaWeights || !m_smaaBlend)
                    return fail("Vulkan", "post-processing shaders");
                {
                    auto area = Vulkan::ImageCreateInfo::immutable_2d_image(AREATEX_WIDTH, AREATEX_HEIGHT, VK_FORMAT_R8G8_UNORM);
                    Vulkan::ImageInitialData areaData = {areaTexBytes, 0, 0};
                    m_smaaArea = m_shared.device->create_image(area, &areaData);
                    auto search = Vulkan::ImageCreateInfo::immutable_2d_image(SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, VK_FORMAT_R8_UNORM);
                    Vulkan::ImageInitialData searchData = {searchTexBytes, 0, 0};
                    m_smaaSearch = m_shared.device->create_image(search, &searchData);
                }
                std::cout << "[presenter] " << (m_metalPresent ? "Metal layer " : "Vulkan swapchain ") << m_platform->get_surface_width()
                          << "x" << m_platform->get_surface_height() << (m_options.vsync ? " (vsync)" : "") << std::endl;
                return true;
            }

            void close() override
            {
                // No GPU waits here: when the loop has stopped, the GS thread may hold the device lock
                // waiting on a frame context that only presenting advances. The device and its
                // resources go with the presenter (or the process).
                m_window = nullptr;
            }

            bool closeRequested() override { return m_closeRequested; }

            void *sdlWindow() override { return m_window; }

            void uiInit() override
            {
#if defined(PS2X_PGS_PRESENTER_UI)
                if (m_ui)
                    return; // already (the setup screen)
                if (!ImGui::GetCurrentContext())
                    ImGui::CreateContext();
                ImGui_ImplSDL3_InitForVulkan(m_window);
                ImGuiIO &io = ImGui::GetIO();
                io.BackendRendererName = "ps2x_granite";
                io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures;
                m_ui = true;
#endif
            }

            void uiShutdown() override
            {
#if defined(PS2X_PGS_PRESENTER_UI)
                if (!m_ui)
                    return;
                ImGui_ImplSDL3_Shutdown();
                ImGui::DestroyContext();
                m_ui = false;
#endif
            }

            void uiBegin() override
            {
#if defined(PS2X_PGS_PRESENTER_UI)
                ImGui_ImplSDL3_NewFrame();
                ImGui::NewFrame();
#endif
            }

            void uiEnd() override
            {
#if defined(PS2X_PGS_PRESENTER_UI)
                ImGui::Render();
                m_uiFrame = true;
#endif
            }

            bool captureWindow(const std::string &pngPath) override
            {
                m_capturePath = pngPath;
                return true;
            }

            void frame(PS2Runtime &runtime, const std::function<void()> &drawUi) override
            {
                pumpEvents();

                updateTemporal(runtime);
                const auto latchStart = std::chrono::steady_clock::now();
                latch(runtime);
                m_latchMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - latchStart).count();
                m_pictureAspect = pictureAspect(runtime, *this);
                m_frame2D = runtime.gsUnsynced().lastFrameWas2D();
                m_uiFrame = false;
                if (drawUi)
                    drawUi();
                waitForDrawable();
                const auto renderStart = std::chrono::steady_clock::now();
                render();
                notePacing(runtime, renderStart);
            }

            // RT_PRESENT_DEBUG=1: where each present's work falls against the guest's vblanks, and
            // how long it holds (and waits for) the device lock the GS thread needs, every 2 s.
            // A present that keeps landing where the game's frame needs the GS can hold the game at
            // 30 fps until the phase moves (a pause does).
            void notePacing(PS2Runtime &runtime, std::chrono::steady_clock::time_point start)
            {
                static const bool debug = std::getenv("RT_PRESENT_DEBUG") != nullptr;
                if (!debug)
                    return;
                const auto end = std::chrono::steady_clock::now();
                const double renderMs = std::chrono::duration<double, std::milli>(end - start).count();
                const int64_t vblank = runtime.eeScheduler().lastVBlankHostNs();
                const int64_t startNs = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
                double phase = vblank ? std::fmod(static_cast<double>(startNs - vblank) / 1e6, 16.667) : -1.0;
                if (vblank && phase < 0.0)
                    phase += 16.667;
                m_pacing.n++;
                m_pacing.latchSum += m_latchMs;
                m_pacing.latchMax = std::max(m_pacing.latchMax, m_latchMs);
                m_pacing.renderSum += renderMs;
                m_pacing.renderMax = std::max(m_pacing.renderMax, renderMs);
                m_pacing.waitSum += m_lockWaitMs;
                m_pacing.waitMax = std::max(m_pacing.waitMax, m_lockWaitMs);
                m_pacing.heldSum += m_lockHeldMs;
                m_pacing.heldMax = std::max(m_pacing.heldMax, m_lockHeldMs);
                if (phase >= 0.0)
                {
                    m_pacing.phases++;
                    m_pacing.phaseSum += phase;
                    m_pacing.phaseMin = std::min(m_pacing.phaseMin, phase);
                    m_pacing.phaseMax = std::max(m_pacing.phaseMax, phase);
                }
                if (end - m_pacing.since < std::chrono::seconds(2))
                    return;
                const double n = static_cast<double>(m_pacing.n);
                std::fprintf(stderr,
                             "[pacing] %u presents: latch (the GS's Present) %.2f ms (max %.2f), render %.2f ms (max %.2f), device "
                             "lock held %.2f (max %.2f), waited for %.2f (max %.2f); starts %.1f ms after a vblank (%.1f..%.1f)\n",
                             m_pacing.n, m_pacing.latchSum / n, m_pacing.latchMax, m_pacing.renderSum / n, m_pacing.renderMax, m_pacing.heldSum / n, m_pacing.heldMax,
                             m_pacing.waitSum / n, m_pacing.waitMax, m_pacing.phaseSum / std::max(1u, m_pacing.phases), m_pacing.phaseMin,
                             m_pacing.phaseMax);
                m_pacing = {};
                m_pacing.since = end;
            }

            void frameUi(const std::function<void()> &drawUi) override
            {
                pumpEvents();
                m_uiFrame = false;
                if (drawUi)
                    drawUi();
                waitForDrawable();
                render(); // no GS yet: a black picture under the UI
            }

        private:
            bool fail(const char *what, const char *why)
            {
                std::cerr << "[presenter] " << what << ": " << (why ? why : "") << std::endl;
                return false;
            }

            void pumpEvents()
            {
                SDL_Event e;
                while (SDL_PollEvent(&e))
                {
#if defined(PS2X_PGS_PRESENTER_UI)
                    if (m_ui)
                        ImGui_ImplSDL3_ProcessEvent(&e);
#endif
                    if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
                        m_closeRequested = true;
                    if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
                        m_platform->requestResize();
#if defined(__ANDROID__)
                    // Android takes the window's surface away while another activity is in front
                    // (the system file picker, the home screen): drop the Vulkan surface and
                    // swapchain with it, and make new ones when the app comes back. SDL reports it
                    // as the window being minimized and restored (its app background/foreground
                    // events don't reach this queue).
                    if ((e.type == SDL_EVENT_WINDOW_MINIMIZED || e.type == SDL_EVENT_WILL_ENTER_BACKGROUND) && !m_background)
                    {
                        std::lock_guard<std::mutex> lock(m_shared.mutex);
                        pgsRegisterThread();
                        m_shared.device->wait_idle();
                        m_wsi.deinit_surface_and_swapchain();
                        m_background = true;
                    }
                    if ((e.type == SDL_EVENT_WINDOW_RESTORED || e.type == SDL_EVENT_DID_ENTER_FOREGROUND) && m_background)
                    {
                        std::lock_guard<std::mutex> lock(m_shared.mutex);
                        pgsRegisterThread();
                        const VkSurfaceKHR surface =
                            m_platform->create_surface(m_shared.device->get_instance(), m_shared.device->get_physical_device());
                        if (surface != VK_NULL_HANDLE)
                        {
                            m_wsi.reinit_surface_and_swapchain(surface);
                            m_background = false;
                        }
                    }
#endif
                }
            }

            // The newest picture: kept on the GPU by the paraLLEl-GS backend on this device, or
            // (CPU GS) uploaded. Test captures also read it back.
            void latch(PS2Runtime &runtime)
            {
                const uint64_t tick = runtime.eeScheduler().currentVSyncTick();
                const bool capture = ps2_test::frameCaptureRequested();
                if (m_latched && tick == m_lastTick && !capture)
                    return;
                m_latched = true;
                m_lastTick = tick;
                const bool gpu = m_shared.attached;
                runtime.gsUnsynced().latchHostPresentationFrame(gpu, capture || !gpu);
                // The jitter of the picture just latched (its 3D's, and the frame's before it).
                if (m_temporalOn)
                    runtime.gsUnsynced().snapshotJitter(m_jitter[0], m_jitter[1], m_jitter[2], m_jitter[3]);
                if (!capture && gpu)
                    return;
                std::vector<uint8_t> pixels;
                uint32_t w = 0, h = 0;
                const bool ok = runtime.gsUnsynced().copyLatchedHostPresentationFrame(pixels, w, h, nullptr, nullptr, nullptr);
                if (capture)
                    ps2_test::deliverFrameCapture(ok ? pixels : std::vector<uint8_t>{}, ok ? w : 0u, ok ? h : 0u);
                if (!gpu && ok && w && h)
                {
                    std::lock_guard<std::mutex> lock(m_shared.mutex);
                    pgsRegisterThread();
                    auto info = Vulkan::ImageCreateInfo::immutable_2d_image(w, h, VK_FORMAT_R8G8B8A8_UNORM);
                    Vulkan::ImageInitialData init = {pixels.data(), 0, 0};
                    m_cpuFrame = m_shared.device->create_image(info, &init);
                }
            }

            void bindState(Vulkan::CommandBuffer &cmd, float fbWidth, float fbHeight, float originX = 0, float originY = 0)
            {
                cmd.set_program(m_program);
                cmd.set_opaque_state();
                cmd.set_depth_test(false, false);
                cmd.set_cull_mode(VK_CULL_MODE_NONE);
                cmd.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
                cmd.set_blend_enable(true);
                cmd.set_blend_factors(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);
                cmd.set_blend_op(VK_BLEND_OP_ADD);
                cmd.set_vertex_attrib(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, x));
                cmd.set_vertex_attrib(1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, u));
                cmd.set_vertex_attrib(2, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(Vertex, colour));
                const float pc[4] = {2.0f / fbWidth, 2.0f / fbHeight, -1.0f - originX * 2.0f / fbWidth,
                                     -1.0f - originY * 2.0f / fbHeight};
                cmd.push_constants(pc, 0, sizeof(pc));
            }

            // The game picture, letterboxed to 4:3 (the PS2 always drives a 4:3 TV) or, with
            // widescreen, to the display aspect.
            // Where the picture goes on the swapchain: the letterbox rectangle (in whole pixels).
            VkRect2D pictureRect(float fw, float fh) const
            {
                const float w = std::floor(std::min(fw, fh * m_pictureAspect)), h = std::floor(w / m_pictureAspect);
                return {{static_cast<int32_t>((fw - w) * 0.5f), static_cast<int32_t>((fh - h) * 0.5f)},
                        {static_cast<uint32_t>(w), static_cast<uint32_t>(h)}};
            }

            // Renders a fullscreen pass from `src` into `dst` (re-created at w x h when needed).
            struct PassInput
            {
                const Vulkan::Image *image;
                Vulkan::StockSampler sampler;
            };

            void offscreenPass(Vulkan::CommandBuffer &cmd, Vulkan::ImageHandle &dst, uint32_t w, uint32_t h,
                               Vulkan::Program *program, const Vulkan::Image &src, const void *push, uint32_t pushSize)
            {
                const PassInput input[1] = {{&src, Vulkan::StockSampler::LinearClamp}};
                offscreenPass(cmd, dst, w, h, program, input, 1, push, pushSize, false);
            }

            void offscreenPass(Vulkan::CommandBuffer &cmd, Vulkan::ImageHandle &dst, uint32_t w, uint32_t h,
                               Vulkan::Program *program, const PassInput *inputs, uint32_t inputCount, const void *push,
                               uint32_t pushSize, bool clear, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM)
            {
                if (!dst || dst->get_width() != w || dst->get_height() != h || dst->get_format() != format)
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(w, h, format);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    dst = m_shared.device->create_image(info);
                }
                cmd.image_barrier(*dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                Vulkan::RenderPassInfo rp = {};
                rp.num_color_attachments = 1;
                rp.color_attachments[0] = &dst->get_view();
                rp.store_attachments = 1;
                rp.clear_attachments = clear ? 1 : 0;
                rp.clear_color[0] = {};
                cmd.begin_render_pass(rp);
                cmd.set_program(program);
                cmd.set_opaque_state();
                cmd.set_depth_test(false, false);
                cmd.set_cull_mode(VK_CULL_MODE_NONE);
                for (uint32_t i = 0; i < inputCount; ++i)
                    cmd.set_texture(0, i, inputs[i].image->get_view(), inputs[i].sampler);
                if (pushSize)
                    cmd.push_constants(push, 0, pushSize);
                cmd.draw(3);
                cmd.end_render_pass();
                cmd.image_barrier(*dst, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }

            // Before the swapchain pass: anti-aliasing at the render resolution, then FSR 1 EASU to
            // the picture's size on screen. Leaves the image the final pass draws in m_final.
            void preparePicture(Vulkan::CommandBufferHandle &cmdHandle, float fw, float fh)
            {
                Vulkan::CommandBuffer &cmd = *cmdHandle;
                const Vulkan::Image *image = sourceImage();
                m_final = image;
                m_finalRcas = false;
                if (!image)
                    return;
                const uint32_t sw = image->get_width(), sh = image->get_height();
                if (m_showMotion && m_shared.motion)
                {
                    const float scale = 1.0f / 32.0f; // 16 px of motion: fully red/green
                    const PassInput mv[1] = {{m_shared.motion.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_aaImage, sw, sh, m_motionView, mv, 1, &scale, sizeof(scale), false);
                    m_final = m_aaImage.get();
                    return;
                }
                if (m_showDepth && m_shared.depth)
                {
                    const float rcpMax = 1.0f / 16777216.0f;
                    const PassInput depth[1] = {{m_shared.depth.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_aaImage, sw, sh, m_depthView, depth, 1, &rcpMax, sizeof(rcpMax), false);
                    m_final = m_aaImage.get();
                    return;
                }
                const bool metalfxTemporal = m_post.scaling == PostProcess::Scaling::MetalFxTemporal;
                if (m_post.aa == PostProcess::AntiAliasing::Taa && !metalfxTemporal)
                {
                    if (m_noTemporal)
                        temporalAntiAliasingShadow(cmd, sw, sh);
                    else
                        temporalAntiAliasing(cmd, sw, sh);
                }
                else if (m_post.aa == PostProcess::AntiAliasing::Smaa)
                {
                    const float metrics[4] = {1.0f / static_cast<float>(sw), 1.0f / static_cast<float>(sh), static_cast<float>(sw),
                                              static_cast<float>(sh)};
                    const PassInput edges[1] = {{m_final, Vulkan::StockSampler::LinearClamp}};
                    offscreenPass(cmd, m_smaaEdgeImage, sw, sh, m_smaaEdges, edges, 1, metrics, sizeof(metrics), true);
                    const PassInput weights[3] = {{m_smaaEdgeImage.get(), Vulkan::StockSampler::LinearClamp},
                                                  {m_smaaArea.get(), Vulkan::StockSampler::LinearClamp},
                                                  {m_smaaSearch.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_smaaWeightImage, sw, sh, m_smaaWeights, weights, 3, metrics, sizeof(metrics), true);
                    const PassInput blend[2] = {{m_final, Vulkan::StockSampler::LinearClamp},
                                                {m_smaaWeightImage.get(), Vulkan::StockSampler::LinearClamp}};
                    offscreenPass(cmd, m_aaImage, sw, sh, m_smaaBlend, blend, 2, metrics, sizeof(metrics), false);
                    m_final = m_aaImage.get();
                }
                else if (m_post.aa == PostProcess::AntiAliasing::Fxaa)
                {
                    const float rcp[2] = {1.0f / static_cast<float>(sw), 1.0f / static_cast<float>(sh)};
                    offscreenPass(cmd, m_aaImage, sw, sh, m_fxaa, *m_final, rcp, sizeof(rcp));
                    m_final = m_aaImage.get();
                }
                const VkRect2D rect = pictureRect(fw, fh);
#if defined(__APPLE__)
                // Temporal needs a progressive picture: progressive fields, or 4x supersampling and
                // up (full frames). Interlaced fields below that alternate every frame: spatial.
                const bool progressive = sh >= 600 || m_progressiveFields;
                if (metalfxTemporal && progressive && m_metalfxTemporal && rect.extent.width > sw && m_shared.motion &&
                    m_shared.depth && m_shared.motion->get_width() == sw && m_shared.depth->get_width() == sw)
                {
                    // Beyond the scaler's largest ratio (640x448 to a Retina window is about 3.5x),
                    // it upscales as far as it can and the presentation scales the rest.
                    const float k = std::min({m_metalfxTemporal->maxScale(), float(rect.extent.width) / float(sw),
                                              float(rect.extent.height) / float(sh)});
                    upscaleMetalFxTemporal(cmdHandle, sw, sh, uint32_t(float(sw) * k), uint32_t(float(sh) * k));
                    return;
                }
                if (metalfxTemporal)
                    m_temporalValid = m_temporalValidShadow = false; // a 2D screen: start the histories again afterwards
                if ((m_post.scaling == PostProcess::Scaling::MetalFxSpatial || (metalfxTemporal && !progressive)) && m_metalfx &&
                    rect.extent.width > sw)
                {
                    upscaleMetalFx(cmdHandle, sw, sh, rect.extent.width, rect.extent.height);
                    return;
                }
#endif
                // Shadow frames (frame generation) don't feed GSR 2's history (it stays a sequence of
                // real frames): GSR 2 blends the shadow with the real frame's output moved on half a
                // frame (upscaleSgsr2Shadow), so the two pictures of each guest frame look alike. On
                // FSR 1 alone the shadows were sharp and aliased between soft GSR 2 frames, and at
                // 120 Hz the difference flickered at every moving edge (the "trails" behind fast
                // cars). ASR (a library) and a shadow GSR 2 can't run: FSR 1, which looks close.
                // RT_SGSR2_SHADOW=0 keeps FSR 1 for GSR 2's shadows too (A/B).
                static const bool shadowSgsr2 = [] { const char *e = std::getenv("RT_SGSR2_SHADOW"); return !(e && *e == '0'); }();
                const bool temporalUpscaler = m_post.scaling == PostProcess::Scaling::SnapdragonGsr2 ||
                                              m_post.scaling == PostProcess::Scaling::ArmAsr;
                if (shadowSgsr2 && m_noTemporal && m_post.scaling == PostProcess::Scaling::SnapdragonGsr2 && rect.extent.width > sw &&
                    upscaleSgsr2Shadow(cmd, sw, sh, rect.extent.width, rect.extent.height))
                    return;
                if ((m_post.scaling == PostProcess::Scaling::Fsr1 || (m_noTemporal && temporalUpscaler)) && rect.extent.width > sw)
                {
                    struct
                    {
                        AU1 con[16];
                        uint32_t origin[4];
                    } push = {};
                    FsrEasuCon(push.con, push.con + 4, push.con + 8, push.con + 12, static_cast<AF1>(sw), static_cast<AF1>(sh),
                               static_cast<AF1>(sw), static_cast<AF1>(sh), static_cast<AF1>(rect.extent.width),
                               static_cast<AF1>(rect.extent.height));
                    static const bool gpuTimesEasu = [] { const char *e = std::getenv("RT_GPU_TIMES"); return e && *e == '1'; }();
                    Vulkan::QueryPoolHandle tsEasu = gpuTimesEasu ? cmd.write_timestamp(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) : Vulkan::QueryPoolHandle{};
                    offscreenPass(cmd, m_upImage, rect.extent.width, rect.extent.height, m_easu, *m_final, &push, sizeof(push));
                    if (gpuTimesEasu)
                        m_shared.device->register_time_interval("GPU", std::move(tsEasu), cmd.write_timestamp(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT),
                                                                "post: FSR 1 EASU");
                    m_final = m_upImage.get();
                    m_finalRcas = true;
                }
                if (m_post.scaling == PostProcess::Scaling::SnapdragonGsr2 && !m_noTemporal && rect.extent.width > sw &&
                    upscaleSgsr2(cmd, sw, sh, rect.extent.width, rect.extent.height))
                    return;
#if defined(PS2X_HAVE_ARM_ASR)
                if (m_post.scaling == PostProcess::Scaling::ArmAsr && !m_noTemporal && rect.extent.width > sw &&
                    upscaleArmAsr(cmdHandle, sw, sh, rect.extent.width, rect.extent.height))
                    return;
#endif
                if (m_post.scaling == PostProcess::Scaling::SnapdragonGsr1 && m_sgsr1 && rect.extent.width > sw)
                {
                    // Snapdragon GSR 1: upscaling and sharpening in one pass (no RCAS after it).
                    const float push[4] = {1.0f / static_cast<float>(sw), 1.0f / static_cast<float>(sh), static_cast<float>(sw),
                                           static_cast<float>(sh)};
                    offscreenPass(cmd, m_upImage, rect.extent.width, rect.extent.height, m_sgsr1, *m_final, push, sizeof(push));
                    m_final = m_upImage.get();
                }
            }

            // TAA needs the game's depth/motion and a jittered camera: switched on and off with it.
            void updateTemporal(PS2Runtime &runtime)
            {
                m_progressiveFields = runtime.gsUnsynced().progressiveFields();
                const bool on = m_post.aa == PostProcess::AntiAliasing::Taa ||
                                m_post.scaling == PostProcess::Scaling::MetalFxTemporal ||
                                m_post.scaling == PostProcess::Scaling::SnapdragonGsr2 ||
                                m_post.scaling == PostProcess::Scaling::ArmAsr;
                // Per-pixel motion and depth: the temporal passes and the picture-warping frame
                // generation. Re-rendered frame generation needs only the tracker's per-vertex
                // motion (the side data stays on for the UI mask), not the GPU's motion attachment
                // and depth snapshots (on the Thor at 120 Hz, GPU time it can't spare).
                const bool motion = on || (m_fg.factor > 1 && !m_fg.rerender);
                m_shared.wantShadows = m_fg.factor > 1 && m_fg.rerender ? m_fg.factor - 1 : 0u;
                if (!m_showMotion && !m_showDepth)
                {
                    m_shared.wantDepth = motion;
                    m_shared.wantMotion = motion;
                    MotionTracker::instance().setEnabled(motion || m_fg.factor > 1);
                }
                // Jitter across one picture pixel: the frame buffer is 640 x 224 (fields), the
                // picture 2x or more of it.
                const Vulkan::Image *pic = m_shared.scanout.get();
                const float sx = pic ? 640.0f / static_cast<float>(pic->get_width()) : 0.5f;
                const float sy = pic ? 224.0f / static_cast<float>(pic->get_height()) : 0.25f;
                runtime.gsUnsynced().setTemporalJitter(on, sx, sy);
                m_temporalOn = on;
                if (!on)
                    m_taaValid = m_sgsr2Valid = m_asrValid = false;
                // Any post-processing (and frame generation) spares the UI (the GS marks where the HUD and 2D
                // screens drew).
                m_shared.wantUi = m_post.aa != PostProcess::AntiAliasing::None || m_post.scaling != PostProcess::Scaling::Bilinear ||
                                  m_fg.factor > 1 || m_showUi; // generated frames keep the current frame's HUD
            }

            // A shadow frame (frame generation, half a frame ahead) blended with the real frame's
            // TAA result, reprojected half a frame with the real frame's motion; it does not become
            // history (the real frames' history stays a sequence of real frames).
            void temporalAntiAliasingShadow(Vulkan::CommandBuffer &cmd, uint32_t sw, uint32_t sh)
            {
                const Vulkan::ImageHandle &history = m_taaHistory[m_taaIndex ^ 1]; // the real frame's output
                if (!m_taaValid || m_taaSerial != m_shared.pictureSerial || !history || history->get_width() != sw || history->get_height() != sh || !m_shared.motion ||
                    m_shared.motion->get_width() != sw || m_shared.motion->get_height() != sh)
                    return;
                struct
                {
                    float motionToUv[2];
                    float jitterDelta[2];
                    float rcpSize[2];
                    float blend;
                    float historyValid;
                } push = {{m_shadowT / 640.0f, m_shadowT / 224.0f}, {0.0f, 0.0f},
                          {1.0f / static_cast<float>(sw), 1.0f / static_cast<float>(sh)}, 0.1f, 1.0f};
                // t of the motion: a point at x in the shadow (frame N + t) was at x - t m in frame
                // N, whose TAA output (at frame N's jitter) is the history here. The jitter step stays
                // in: the shadow GS moved every vertex by t m, the camera's jitter step included.
                const PassInput inputs[3] = {{m_final, Vulkan::StockSampler::NearestClamp},
                                             {history.get(), Vulkan::StockSampler::LinearClamp},
                                             {m_shared.motion.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(cmd, m_taaShadowImage, sw, sh, m_taa, inputs, 3, &push, sizeof(push), false);
                m_final = m_taaShadowImage.get();
            }

            // Snapdragon GSR 2 (post/sgsr2_*.frag): the GS's motion and depth turned into its motion /
            // depth-clip buffer at the picture's size, then upscaled into a history at the output's.
            // False without 3D (a 2D screen): the caller scales as usual and the history starts again.
            struct Sgsr2Push
            {
                float renderSize[2], outputSize[2], renderSizeRcp[2], outputSizeRcp[2];
                float jitterOffset[2], scaleRatio[2], motionToNdc[2];
                float cameraFovAngleHor, minLerpContribution, reset;
                float prevValid; // the convert pass's previous buffer is the last picture's (its disocclusion test)
                float jitterDelta[2]; // GS pixels, this frame's jitter minus last frame's
                float moveConfidence, moveGamma, minWeight, historyCubic; // moving pixels; history filter (sgsr2_upscale.frag)
                float speedScale; // the blend weights' motion per game frame / the motion sampled
            };

            // Game frames between the picture a temporal pass last ran on (serial) and this one; 0 to
            // start the history again (none yet, or too far back to extrapolate the motion).
            float pictureSteps(uint64_t serial) const
            {
                const uint64_t gap = m_shared.pictureSerial - serial;
                static const bool debug = [] { const char *e = std::getenv("RT_TEMPORAL_DEBUG"); return e && *e == '1'; }();
                if (debug && serial != ~0ull && gap > 1)
                    std::fprintf(stderr, "[temporal] %llu frames since the last temporal pass\n", (unsigned long long)gap);
                return serial == ~0ull || gap > 4 ? 0.0f : static_cast<float>(gap);
            }

            // GSR 2's parameters for a picture of sw x sh to w x h, with the motion of `steps` frames
            // (GS pixels of one frame in the motion buffer).
            Sgsr2Push sgsr2Push(uint32_t sw, uint32_t sh, uint32_t w, uint32_t h, float steps) const
            {
                const float fw = static_cast<float>(sw), fh = static_cast<float>(sh), ow = static_cast<float>(w), oh = static_cast<float>(h);
                Sgsr2Push push = {};
                push.renderSize[0] = fw, push.renderSize[1] = fh;
                push.outputSize[0] = ow, push.outputSize[1] = oh;
                push.renderSizeRcp[0] = 1.0f / fw, push.renderSizeRcp[1] = 1.0f / fh;
                push.outputSizeRcp[0] = 1.0f / ow, push.outputSizeRcp[1] = 1.0f / oh;
                // The camera's jitter (GS pixels) in picture pixels.
                push.jitterOffset[0] = m_jitter[0] * fw / 640.0f;
                push.jitterOffset[1] = m_jitter[1] * fh / 224.0f;
                push.scaleRatio[0] = ow / fw;
                push.scaleRatio[1] = std::min(20.0f, std::pow(ow * oh / (fw * fh), 3.0f));
                push.motionToNdc[0] = steps * 2.0f / 640.0f;
                push.motionToNdc[1] = steps * 2.0f / 224.0f;
                // The motion comes from the jittered cameras: without this frame's jitter step (as
                // GSR expects, and as TAA takes it out too).
                push.jitterDelta[0] = m_jitter[0] - m_jitter[2];
                push.jitterDelta[1] = m_jitter[1] - m_jitter[3];
                push.cameraFovAngleHor = 0.75f; // tan(FOV / 2): Road Trip's cameras are about 74 degrees across
                push.minLerpContribution = 0.0f;
                // History on moving pixels (sgsr2_upscale.frag; 0 = Qualcomm's blend). A/B:
                // RT_SGSR2_MOVE (current-frame weight per output half-pixel of motion),
                // RT_SGSR2_GAMMA (history box in standard deviations), RT_SGSR2_MINW (least weight, 0.2 by default:
                // against the 3x-rendered picture it took the attract demo from 0.82 to 0.68 blurred
                // error, FSR 1 0.65, and removed the doubled decals and dotted rows on moving cars).
                static const float move = envFloat("RT_SGSR2_MOVE", 0.0f), gamma = envFloat("RT_SGSR2_GAMMA", 0.0f),
                                   minWeight = envFloat("RT_SGSR2_MINW", 0.2f);
                push.moveConfidence = move, push.moveGamma = gamma, push.minWeight = minWeight;
                push.speedScale = 1.0f;
                return push;
            }

            // A shadow frame (frame generation, half a frame after the real one) through GSR 2: its
            // picture upscaled and blended with the real frame's GSR 2 output, moved on half a frame
            // with the real frame's motion. Not kept as history. False when the real frame's output
            // isn't this picture's (the caller scales with FSR 1).
            bool upscaleSgsr2Shadow(Vulkan::CommandBuffer &cmd, uint32_t sw, uint32_t sh, uint32_t w, uint32_t h)
            {
                const Vulkan::ImageHandle &real = m_sgsr2History[m_sgsr2Index ^ 1]; // the real frame's output
                if (!m_sgsr2Convert || !m_sgsr2Upscale || !m_sgsr2Valid || m_sgsr2Serial != m_shared.pictureSerial || !real ||
                    real->get_width() != w || real->get_height() != h || !m_shared.motion || !m_shared.depth ||
                    m_shared.motion->get_width() != sw || m_shared.motion->get_height() != sh || m_shared.depth->get_width() != sw ||
                    m_shared.depth->get_height() != sh)
                    return false;
                // t of the motion (a point at x in the shadow, t of a frame on, was at x - t m in the
                // real frame), and the shadow's jitter: the real frame's plus t of its step (the
                // shadow GS moves every vertex on by t of its motion, jitter step included).
                const float t = m_shadowT;
                Sgsr2Push push = sgsr2Push(sw, sh, w, h, t);
                push.jitterOffset[0] = (m_jitter[0] + t * (m_jitter[0] - m_jitter[2])) * static_cast<float>(sw) / 640.0f;
                push.jitterOffset[1] = (m_jitter[1] + t * (m_jitter[1] - m_jitter[3])) * static_cast<float>(sh) / 224.0f;
                push.reset = 0.0f;
                // The history here is the real frame's output moved on t of a frame: bilinear
                // sampling softened it (shadows ~11% less road detail than the real frames, a 60 Hz
                // shimmer on textured ground at 120 Hz); Catmull-Rom keeps it as sharp.
                // RT_SGSR2_SHADOW_CUBIC=0 samples it bilinearly again (A/B).
                static const bool cubic = [] { const char *e = std::getenv("RT_SGSR2_SHADOW_CUBIC"); return !(e && *e == '0'); }();
                push.historyCubic = cubic ? 1.0f : 0.0f;
                // Its motion is t of the real frame's, but the blend weights (how much of this
                // picture against the history) go by how fast a pixel moves per game frame: at t's
                // motion a shadow kept up to twice the history the real frame does, and with it the
                // history's softness. RT_SGSR2_SHADOW_SPEED=0 weighs by t's motion again (A/B).
                static const bool perFrame = [] { const char *e = std::getenv("RT_SGSR2_SHADOW_SPEED"); return !(e && *e == '0'); }();
                push.speedScale = perFrame && t > 0.0f ? 1.0f / t : 1.0f;
                // Disocclusion against the real frame's buffer (its depth: the history here is its output).
                const Vulkan::ImageHandle &realMda = m_sgsr2Mda[m_sgsr2MdaIndex ^ 1];
                const bool prevOk = sgsr2Disocclusion() && realMda && realMda->get_width() == sw && realMda->get_height() == sh;
                push.prevValid = prevOk ? 1.0f : 0.0f;
                const PassInput convertIn[3] = {{m_shared.depth.get(), Vulkan::StockSampler::NearestClamp},
                                                {m_shared.motion.get(), Vulkan::StockSampler::NearestClamp},
                                                {prevOk ? realMda.get() : m_shared.depth.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(cmd, m_sgsr2MdaShadow, sw, sh, m_sgsr2Convert, convertIn, 3, &push, sizeof(push), false,
                              VK_FORMAT_R16G16B16A16_SFLOAT);
                const PassInput upscaleIn[3] = {{real.get(), Vulkan::StockSampler::LinearClamp},
                                                {m_sgsr2MdaShadow.get(), Vulkan::StockSampler::LinearClamp},
                                                {m_final, Vulkan::StockSampler::NearestClamp}};
                offscreenPass(cmd, m_sgsr2Shadow, w, h, m_sgsr2Upscale, upscaleIn, 3, &push, sizeof(push), false);
                m_final = m_sgsr2Shadow.get();
                m_finalRcas = true;
                return true;
            }

            static float envFloat(const char *name, float fallback)
            {
                const char *e = std::getenv(name);
                return e && *e ? static_cast<float>(std::atof(e)) : fallback;
            }

            // GSR 2's disocclusion test (on unless RT_SGSR2_DISOCC=0, for A/B).
            static bool sgsr2Disocclusion()
            {
                static const bool on = [] { const char *e = std::getenv("RT_SGSR2_DISOCC"); return !(e && *e == '0'); }();
                return on;
            }

            bool upscaleSgsr2(Vulkan::CommandBuffer &cmd, uint32_t sw, uint32_t sh, uint32_t w, uint32_t h)
            {
                if (!m_sgsr2Convert || !m_sgsr2Upscale || !m_shared.motion || !m_shared.depth || m_shared.motion->get_width() != sw ||
                    m_shared.motion->get_height() != sh || m_shared.depth->get_width() != sw || m_shared.depth->get_height() != sh)
                {
                    m_sgsr2Valid = false;
                    return false;
                }
                Vulkan::ImageHandle &out = m_sgsr2History[m_sgsr2Index];
                const Vulkan::ImageHandle &history = m_sgsr2History[m_sgsr2Index ^ 1];
                const bool sized = m_sgsr2Valid && history && history->get_width() == w && history->get_height() == h;
                if (sized && m_shared.pictureSerial == m_sgsr2Serial)
                {
                    // The same 3D again: the last output (its history) is the picture.
                    m_final = history.get();
                    m_finalRcas = true;
                    return true;
                }
                const float steps = pictureSteps(m_sgsr2Serial);
                const bool valid = sized && steps > 0.0f;
                m_sgsr2Serial = m_shared.pictureSerial;
                Sgsr2Push push = sgsr2Push(sw, sh, w, h, std::max(steps, 1.0f));
                push.reset = valid ? 0.0f : 1.0f;
                // The convert pass keeps each picture's depth in its buffer (w) and tests the history
                // against the last one's: disocclusion (sgsr2_convert.frag).
                Vulkan::ImageHandle &mda = m_sgsr2Mda[m_sgsr2MdaIndex];
                const Vulkan::ImageHandle &prevMda = m_sgsr2Mda[m_sgsr2MdaIndex ^ 1];
                const bool prevOk = sgsr2Disocclusion() && valid && prevMda && prevMda->get_width() == sw && prevMda->get_height() == sh;
                push.prevValid = prevOk ? 1.0f : 0.0f;
                const PassInput convertIn[3] = {{m_shared.depth.get(), Vulkan::StockSampler::NearestClamp},
                                                {m_shared.motion.get(), Vulkan::StockSampler::NearestClamp},
                                                {prevOk ? prevMda.get() : m_shared.depth.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(cmd, mda, sw, sh, m_sgsr2Convert, convertIn, 3, &push, sizeof(push), false,
                              VK_FORMAT_R16G16B16A16_SFLOAT);
                m_sgsr2MdaIndex ^= 1;
                const PassInput upscaleIn[3] = {{valid ? history.get() : m_final, Vulkan::StockSampler::LinearClamp},
                                                {mda.get(), Vulkan::StockSampler::LinearClamp},
                                                {m_final, Vulkan::StockSampler::NearestClamp}};
                offscreenPass(cmd, out, w, h, m_sgsr2Upscale, upscaleIn, 3, &push, sizeof(push), false);
                m_final = out.get();
                m_finalRcas = true; // GSR 2 doesn't sharpen: FSR's RCAS after it (the Sharpening option)
                m_sgsr2Index ^= 1;
                m_sgsr2Valid = true;
                return true;
            }

#if defined(PS2X_HAVE_ARM_ASR)
            // Arm Accuracy Super Resolution (FSR 2-derived, Arm's library with its own Vulkan
            // backend and prebuilt shaders): the picture, the GS's depth (normalised, larger =
            // nearer: inverted, infinite) and motion (jitter included: the library cancels it).
            // It records raw Vulkan, so it gets a command buffer of its own between Granite's.
            // False when it can't run (a 2D screen, or the library failed): the caller scales as usual.
            struct ArmAsr
            {
                arm::FfxmFsr2Context context = {};
                uint32_t renderW = 0, renderH = 0, displayW = 0, displayH = 0;
                uint32_t retireFrames = 0; // dropped: destroyed when this counts down (the GPU is done)
                bool created = false;
                ~ArmAsr()
                {
                    if (created)
                        arm::ffxmFsr2ContextDestroy(&context);
                }
            };

            static void asrMessage(arm::FfxmMsgType type, const wchar_t *message)
            {
                std::fprintf(stderr, "[asr] %s: %ls\n", type == arm::FFXM_MESSAGE_TYPE_ERROR ? "error" : "warning", message);
            }

            bool upscaleArmAsr(Vulkan::CommandBufferHandle &cmd, uint32_t sw, uint32_t sh, uint32_t w, uint32_t h)
            {
                if (m_asrFailed || !m_shared.motion || !m_shared.depth || m_shared.motion->get_width() != sw ||
                    m_shared.motion->get_height() != sh || m_shared.depth->get_width() != sw || m_shared.depth->get_height() != sh)
                {
                    m_asrValid = false;
                    return false;
                }
                Vulkan::Device &dev = *m_shared.device;
                // Contexts for up to two sizes (the picture's shape changes between screens). No GPU
                // waits here (Granite's wait_idle would wait for this very command buffer): a
                // dropped context is destroyed some frames later.
                for (auto it = m_asrRetired.begin(); it != m_asrRetired.end();)
                    it = --(*it)->retireFrames == 0 ? m_asrRetired.erase(it) : it + 1;
                auto matches = [&](const std::unique_ptr<ArmAsr> &a) {
                    return a && a->renderW == sw && a->renderH == sh && a->displayW == w && a->displayH == h;
                };
                if (!matches(m_asr) && matches(m_asrOther))
                {
                    std::swap(m_asr, m_asrOther);
                    m_asrValid = false;
                }
                if (!matches(m_asr))
                {
                    if (m_asrOther)
                    {
                        m_asrOther->retireFrames = 8;
                        m_asrRetired.push_back(std::move(m_asrOther));
                    }
                    m_asrOther = std::move(m_asr);
                    auto asr = std::make_unique<ArmAsr>();
                    // The library's Vulkan backend is one per process: made once, with room for
                    // the two contexts and one being retired.
                    arm::FfxmErrorCode err = arm::FFXM_OK;
                    const char *step = "backend";
                    if (m_asrScratch.empty())
                    {
                        constexpr size_t kContexts = 3 * FFXM_FSR2_CONTEXT_COUNT;
                        const size_t size = arm::ffxmGetScratchMemorySizeVK(dev.get_physical_device(), kContexts);
                        m_asrScratch.resize(size);
                        m_asrDevice = {dev.get_device(), dev.get_physical_device(), vkGetDeviceProcAddr};
                        err = arm::ffxmGetInterfaceVK(&m_asrIface, arm::ffxmGetDeviceVK(&m_asrDevice), m_asrScratch.data(), size,
                                                      kContexts);
                    }
                    arm::FfxmFsr2ContextDescription desc = {};
                    desc.qualityMode = arm::FFXM_FSR2_SHADER_QUALITY_MODE_QUALITY;
                    desc.flags = arm::FFXM_FSR2_ENABLE_DEPTH_INVERTED | arm::FFXM_FSR2_ENABLE_DEPTH_INFINITE |
                                 arm::FFXM_FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
                    desc.maxRenderSize = {sw, sh};
                    desc.displaySize = {w, h};
                    desc.fpMessage = asrMessage;
                    if (err == arm::FFXM_OK)
                    {
                        desc.backendInterface = m_asrIface;
                        err = arm::ffxmFsr2ContextCreate(&asr->context, &desc);
                        step = "context";
                    }
                    if (err != arm::FFXM_OK)
                    {
                        std::fprintf(stderr, "[asr] the upscaler could not be created (%s: error 0x%x); scaling without it\n", step,
                                     static_cast<unsigned>(err));
                        m_asrFailed = true;
                        return false;
                    }
                    asr->created = true;
                    asr->renderW = sw, asr->renderH = sh, asr->displayW = w, asr->displayH = h;
                    m_asr = std::move(asr);
                    m_asrValid = false;
                }
                if (!m_asrOut || m_asrOut->get_width() != w || m_asrOut->get_height() != h)
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(w, h, VK_FORMAT_R8G8B8A8_UNORM);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    m_asrOut = dev.create_image(info);
                    cmd->image_barrier(*m_asrOut, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                    m_asrValid = false;
                }
                if (m_asrValid && m_shared.pictureSerial == m_asrSerial)
                {
                    m_final = m_asrOut.get(); // the same 3D again: the last output
                    return true;
                }
                const float steps = pictureSteps(m_asrSerial);
                const bool reset = !m_asrValid || steps == 0.0f;
                m_asrSerial = m_shared.pictureSerial;
                const PassInput depthIn[1] = {{m_shared.depth.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(*cmd, m_depthNorm, sw, sh, m_depthNormalize, depthIn, 1, nullptr, 0, false, VK_FORMAT_R32_SFLOAT);
                cmd->barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
                dev.submit(cmd);

                auto resource = [](const Vulkan::Image &image, arm::FfxmSurfaceFormat format, uint32_t usage) {
                    arm::FfxmResourceDescription d = {};
                    d.type = arm::FFXM_RESOURCE_TYPE_TEXTURE2D;
                    d.format = format;
                    d.width = image.get_width();
                    d.height = image.get_height();
                    d.depth = 1;
                    d.mipCount = 1;
                    d.flags = arm::FFXM_RESOURCE_FLAGS_NONE;
                    d.usage = static_cast<arm::FfxmResourceUsage>(usage);
                    return arm::ffxmGetResourceVK(reinterpret_cast<void *>(image.get_image()), d, nullptr,
                                                  arm::FFXM_RESOURCE_STATE_PIXEL_COMPUTE_READ);
                };
                auto asrCmd = dev.request_command_buffer();
                const float fw = static_cast<float>(sw), fh = static_cast<float>(sh);
                const float k = std::max(steps, 1.0f);
                arm::FfxmFsr2DispatchDescription d = {};
                d.commandList = arm::ffxmGetCommandListVK(asrCmd->get_command_buffer());
                d.color = resource(*m_final, arm::FFXM_SURFACE_FORMAT_R8G8B8A8_UNORM, arm::FFXM_RESOURCE_USAGE_READ_ONLY);
                d.depth = resource(*m_depthNorm, arm::FFXM_SURFACE_FORMAT_R32_FLOAT, arm::FFXM_RESOURCE_USAGE_READ_ONLY);
                d.motionVectors = resource(*m_shared.motion, arm::FFXM_SURFACE_FORMAT_R16G16_FLOAT, arm::FFXM_RESOURCE_USAGE_READ_ONLY);
                d.output = resource(*m_asrOut, arm::FFXM_SURFACE_FORMAT_R8G8B8A8_UNORM,
                                    arm::FFXM_RESOURCE_USAGE_RENDERTARGET | arm::FFXM_RESOURCE_USAGE_UAV);
                // The camera's jitter in picture pixels; motion from current to previous in picture
                // pixels (ours: current minus previous, GS pixels, one frame).
                d.jitterOffset = {m_jitter[0] * fw / 640.0f, m_jitter[1] * fh / 224.0f};
                d.motionVectorScale = {-k * fw / 640.0f, -k * fh / 224.0f};
                d.renderSize = {sw, sh};
                d.enableSharpening = true;
                d.sharpness = m_post.sharpness;
                d.frameTimeDelta = 1000.0f / 59.94f;
                d.preExposure = 1.0f;
                d.reset = reset;
                d.cameraNear = 1.0f;
                d.cameraFar = 100000.0f;
                d.cameraFovAngleVertical = 1.03f; // Road Trip's cameras: about 74 degrees across at 4:3
                d.viewSpaceToMetersFactor = 1.0f;
                const arm::FfxmErrorCode err = arm::ffxmFsr2ContextDispatch(&m_asr->context, &d);
                asrCmd->barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
                dev.submit(asrCmd);
                cmd = dev.request_command_buffer();
                if (err != arm::FFXM_OK)
                {
                    std::fprintf(stderr, "[asr] dispatch failed (%d); scaling without it\n", static_cast<int>(err));
                    m_asrFailed = true;
                    return false;
                }
                m_final = m_asrOut.get();
                m_finalRcas = false; // its own RCAS (the Sharpening option)
                m_asrValid = true;
                return true;
            }
#endif

            void temporalAntiAliasing(Vulkan::CommandBuffer &cmd, uint32_t sw, uint32_t sh)
            {
                if (!m_shared.motion || m_shared.motion->get_width() != sw || m_shared.motion->get_height() != sh)
                {
                    m_taaValid = false; // a 2D screen (no 3D motion): nothing to accumulate
                    return;
                }
                Vulkan::ImageHandle &out = m_taaHistory[m_taaIndex];
                const Vulkan::ImageHandle &history = m_taaHistory[m_taaIndex ^ 1];
                const bool sized = m_taaValid && history && history->get_width() == sw && history->get_height() == sh;
                if (sized && m_shared.pictureSerial == m_taaSerial)
                {
                    m_final = history.get(); // the same 3D again: the last output
                    return;
                }
                const float steps = pictureSteps(m_taaSerial);
                const bool valid = sized && steps > 0.0f;
                m_taaSerial = m_shared.pictureSerial;
                struct
                {
                    float motionToUv[2];
                    float jitterDelta[2];
                    float rcpSize[2];
                    float blend;
                    float historyValid;
                } push = {{std::max(steps, 1.0f) / 640.0f, std::max(steps, 1.0f) / 224.0f},
                          {m_jitter[0] - m_jitter[2], m_jitter[1] - m_jitter[3]},
                          {1.0f / static_cast<float>(sw), 1.0f / static_cast<float>(sh)}, 0.1f, valid ? 1.0f : 0.0f};
                const PassInput inputs[3] = {{m_final, Vulkan::StockSampler::NearestClamp},
                                             {valid ? history.get() : m_final, Vulkan::StockSampler::LinearClamp},
                                             {m_shared.motion.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(cmd, out, sw, sh, m_taa, inputs, 3, &push, sizeof(push), false);
                m_final = out.get();
                m_taaIndex ^= 1;
                m_taaValid = true;
            }

#if defined(__APPLE__)
            // The passes so far go to the GPU and finish; MetalFX then scales m_final into the
            // window-sized m_upImage on the same Metal queue; the frame continues in a new command
            // buffer. (MoltenVK image layouts are no-ops in Metal, so the transition below only
            // keeps Vulkan's bookkeeping consistent.)
            void upscaleMetalFx(Vulkan::CommandBufferHandle &cmd, uint32_t sw, uint32_t sh, uint32_t w, uint32_t h)
            {
                Vulkan::Device &dev = *m_shared.device;
                if (!m_upImage || m_upImage->get_width() != w || m_upImage->get_height() != h ||
                    !(m_upImage->get_create_info().usage & VK_IMAGE_USAGE_STORAGE_BIT))
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(w, h, VK_FORMAT_R8G8B8A8_UNORM);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    m_upImage = dev.create_image(info);
                }
                cmd->image_barrier(*m_upImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                // Submitted (MoltenVK commits it to the Metal queue); MetalFX encodes after it on
                // that queue, so no CPU wait under the device lock.
                dev.submit(cmd);
                void *in = nullptr, *out = nullptr, *queue = nullptr;
                m_getTexture(m_final->get_image(), &in);
                m_getTexture(m_upImage->get_image(), &out);
                m_getQueue(m_wsi.get_context().get_queue_info().queues[Vulkan::QUEUE_INDEX_GRAPHICS], &queue);
                if (in && out && queue && m_metalfx->upscale(queue, in, sw, sh, out, w, h))
                    m_final = m_upImage.get();
                cmd = dev.request_command_buffer();
            }
#endif

#if defined(__APPLE__)
            // MetalFX temporal: depth normalised for it, then like the spatial scaler, with motion
            // (pointing back to last frame, in input pixels) and this frame's jitter.
            void upscaleMetalFxTemporal(Vulkan::CommandBufferHandle &cmd, uint32_t sw, uint32_t sh, uint32_t w, uint32_t h)
            {
                Vulkan::Device &dev = *m_shared.device;
                // Shadow frames (frame generation) have their own scaler and history: they arrive
                // one guest frame apart, like the real ones, half a frame later.
                if (m_noTemporal && !m_metalfxTemporalShadow && m_mtlDevice)
                    m_metalfxTemporalShadow = MetalFxTemporal::create(m_mtlDevice);
                MetalFxTemporal *scaler = m_noTemporal ? m_metalfxTemporalShadow.get() : m_metalfxTemporal.get();
                Vulkan::ImageHandle &upImage = m_noTemporal ? m_upImageShadow : m_upImage;
                bool &valid = m_noTemporal ? m_temporalValidShadow : m_temporalValid;
                if (!scaler)
                    return;
                // The shadow scaler's frames are (picture, t) apart: one guest frame each at 2x, but
                // t changes between them at 3x and 4x, and a shadow that wasn't ready leaves a gap.
                const float shadowAt = static_cast<float>(m_shared.pictureSerial % 1000000u) + m_shadowT;
                const bool same = m_noTemporal ? m_mfxShadowAt == shadowAt : m_shared.pictureSerial == m_mfxSerial;
                if (valid && same && upImage && upImage->get_width() == w && upImage->get_height() == h)
                {
                    m_final = upImage.get(); // the same 3D again: the last output
                    return;
                }
                float steps = 0.0f;
                if (!m_noTemporal)
                {
                    steps = pictureSteps(m_mfxSerial);
                    m_mfxSerial = m_shared.pictureSerial;
                }
                else
                {
                    const float dt = shadowAt - m_mfxShadowAt;
                    steps = m_mfxShadowAt >= 0.0f && dt > 0.0f && dt <= 4.0f ? dt : 0.0f;
                    m_mfxShadowAt = shadowAt;
                }
                if (steps == 0.0f)
                    valid = false;
                const PassInput depthIn[1] = {{m_shared.depth.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(*cmd, m_depthNorm, sw, sh, m_depthNormalize, depthIn, 1, nullptr, 0, false, VK_FORMAT_R32_SFLOAT);
                if (!upImage || upImage->get_width() != w || upImage->get_height() != h ||
                    !(upImage->get_create_info().usage & VK_IMAGE_USAGE_STORAGE_BIT))
                {
                    auto info = Vulkan::ImageCreateInfo::render_target(w, h, VK_FORMAT_R8G8B8A8_UNORM);
                    info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
                    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                    upImage = dev.create_image(info);
                }
                cmd->image_barrier(*upImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                // Submitted (MoltenVK commits it to the Metal queue); MetalFX encodes after it on
                // that queue, so no CPU wait under the device lock.
                dev.submit(cmd);
                MetalFxTemporal::Frame f = {};
                m_getTexture(m_final->get_image(), &f.color);
                m_getTexture(m_depthNorm->get_image(), &f.depth);
                m_getTexture(m_shared.motion->get_image(), &f.motion);
                m_getTexture(upImage->get_image(), &f.output);
                void *queue = nullptr;
                m_getQueue(m_wsi.get_context().get_queue_info().queues[Vulkan::QUEUE_INDEX_GRAPHICS], &queue);
                const float px = static_cast<float>(sw) / 640.0f, py = static_cast<float>(sh) / 224.0f;
                f.inW = sw;
                f.inH = sh;
                f.outW = w;
                f.outH = h;
                // A shadow's jitter: the real frame's plus t of its step (its GS moved every vertex on).
                const float jt = m_noTemporal ? m_shadowT : 0.0f;
                f.jitterX = (m_jitter[0] + jt * (m_jitter[0] - m_jitter[2])) * px;
                f.jitterY = (m_jitter[1] + jt * (m_jitter[1] - m_jitter[3])) * py;
                f.motionScaleX = -px * std::max(steps, 1.0f); // ours: current minus previous, GS pixels, one frame
                f.motionScaleY = -py * std::max(steps, 1.0f);
                f.reset = !valid;
                if (f.color && f.depth && f.motion && f.output && queue && scaler->upscale(queue, f))
                {
                    m_final = upImage.get();
                    valid = true;
                }
                cmd = dev.request_command_buffer();
            }
#endif

            // The guest's picture: the scanout (or the CPU GS's upload), or with progressive fields
            // the scanout with alternating pixels blended (blendFlicker).
            const Vulkan::Image *sourceImage() const
            {
                if (m_sourceOverride)
                    return m_sourceOverride;
                if (m_flickerOut)
                    return m_flickerOut.get();
                const Vulkan::ImageHandle &raw = m_shared.attached ? m_shared.scanout : m_cpuFrame;
                return raw.get();
            }

            // Progressive fields show every field whole, so effects that alternate two pictures
            // and count on an interlaced TV to blend them would strobe (flicker.frag). Once per
            // guest frame: the raw picture against the two before it.
            void blendFlicker(Vulkan::CommandBuffer &cmd)
            {
                m_flickerOut.reset();
                const Vulkan::ImageHandle &raw = m_shared.attached ? m_shared.scanout : m_cpuFrame;
                // RT_FLICKER=0: no flicker blending (A/B measurements).
                static const bool flickerOff = [] {
                    const char *e = std::getenv("RT_FLICKER");
                    return e && *e == '0';
                }();
                if (!raw || !m_progressiveFields || m_showMotion || m_showDepth || flickerOff)
                {
                    m_flickerValid = 0;
                    return;
                }
                const uint32_t w = raw->get_width(), h = raw->get_height();
                Vulkan::ImageHandle &prev = m_flickerHist[m_flickerIndex], &before = m_flickerHist[m_flickerIndex ^ 1u];
                const bool valid = m_flickerValid >= 1 && prev && prev->get_width() == w && prev->get_height() == h;
                const bool haveBefore = valid && m_flickerValid >= 2 && before && before->get_width() == w &&
                                        before->get_height() == h;
                if (valid)
                {
                    // The cut: a 32x32 grid of drastic changes, then its share (no 2048 scattered reads
                    // in one pixel).
                    const PassInput gridIn[2] = {{raw.get(), Vulkan::StockSampler::NearestClamp},
                                                 {prev.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_flickerGridImage, 32, 32, m_flickerGrid, gridIn, 2, nullptr, 0, false);
                    const PassInput cutIn[1] = {{m_flickerGridImage.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_flickerCutImage, 1, 1, m_flickerCut, cutIn, 1, nullptr, 0, false);
                    // The weight at the game's resolution (a supersampled picture is 2x or 4x it), then
                    // the blend at full resolution.
                    const uint32_t ds = std::max(1u, w / 640u);
                    const uint32_t lw = std::max(1u, w / ds), lh = std::max(1u, h / ds);
                    const float rcp[2] = {1.0f / static_cast<float>(lw), 1.0f / static_cast<float>(lh)};
                    const PassInput weightIn[4] = {{raw.get(), Vulkan::StockSampler::NearestClamp},
                                                   {prev.get(), Vulkan::StockSampler::NearestClamp},
                                                   {haveBefore ? before.get() : prev.get(), Vulkan::StockSampler::NearestClamp},
                                                   {m_flickerCutImage.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_flickerWeightImage, lw, lh, m_flickerWeight, weightIn, 4, rcp, sizeof(rcp), false,
                                  VK_FORMAT_R8_UNORM);
                    const PassInput in[3] = {{raw.get(), Vulkan::StockSampler::NearestClamp},
                                             {prev.get(), Vulkan::StockSampler::NearestClamp},
                                             {m_flickerWeightImage.get(), Vulkan::StockSampler::LinearClamp}};
                    offscreenPass(cmd, m_flickerImage, w, h, m_flicker, in, 3, nullptr, 0, false);
                }
                // The raw picture becomes "previous"; the old previous becomes "before last".
                m_flickerIndex ^= 1u;
                if (m_shared.scanoutRing >= 3u && m_shared.attached)
                    m_flickerHist[m_flickerIndex] = raw; // a ring of 3+: kept as it is until overwritten
                else
                {
                    const PassInput copyIn[1] = {{raw.get(), Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_flickerHist[m_flickerIndex], w, h, m_copy, copyIn, 1, nullptr, 0, false);
                }
                m_flickerValid = std::min(m_flickerValid + 1u, 2u);
                if (valid)
                    m_flickerOut = m_flickerImage;
            }

            // Frame generation: the guest frame's picture is prepared once and kept (with the one
            // before it); each display refresh shows it or a frame made between/after them.
            void generatedPicture(Vulkan::CommandBufferHandle &cmd, float fw, float fh, bool fresh)
            {
                if (m_fg.rerender)
                {
                    // Re-rendered: the real frame, then the shadow GS's frames of the same guest
                    // frame with the 3D moved on (gs_pgs_backend.cpp). 2D screens repeat the real one.
                    if (fresh || !m_lastPicture)
                    {
                        // The same 3D as the last vblank's (the game's frame missed the vblank): the
                        // last picture shown stays. Going back to the real frame after its
                        // generated one stepped everything that moves back half a frame (flicker).
                        if (fresh && m_lastPicture && m_lastShown && !m_frame2D && m_shared.pictureSerial == m_lastRealPicture &&
                            m_shared.pictureSerial < (1ull << 62))
                        {
                            m_final = m_lastShown;
                            m_finalRcas = m_lastShownRcas;
                            m_realSerial = m_shared.presentSerial;
                            m_plTag = 'h';
                            return;
                        }
                        preparePicture(cmd, fw, fh);
                        m_lastPicture = m_final;
                        m_lastRcas = m_finalRcas;
                        m_lastShown = m_final;
                        m_lastShownRcas = m_finalRcas;
                        m_lastRealPicture = m_shared.pictureSerial;
                        m_realSerial = m_shared.presentSerial;
                        m_plTag = 'R';
                        return;
                    }
                    const uint32_t step = m_subframe * m_fg.factor / std::max(m_presents, m_fg.factor);
                    const Vulkan::Image *raw = (m_shared.attached ? m_shared.scanout : m_cpuFrame).get();
                    const Vulkan::Image *shadow = step >= 1 && step <= 3 ? m_shared.shadowScanout[step - 1].get() : nullptr;
                    m_plTag = shadow ? 'S' : 'w';
                    // Only this real frame's own shadow (the worker may still be on it).
                    if (shadow && m_shared.shadowSerial[step - 1] != m_realSerial)
                        shadow = nullptr, m_plTag = 's';
                    if (shadow && !(raw && !m_frame2D && shadow->get_width() == raw->get_width() &&
                                    shadow->get_height() == raw->get_height()))
                        m_plTag = 'd';
                    if (shadow && raw && !m_frame2D && shadow->get_width() == raw->get_width() &&
                        shadow->get_height() == raw->get_height())
                    {
                        m_sourceOverride = shadow; // until the next render: spareUi composites from it too
                        m_shadowT = static_cast<float>(step) / static_cast<float>(m_fg.factor);
                        m_noTemporal = true;
                        preparePicture(cmd, fw, fh);
                        m_noTemporal = false;
                        m_lastShown = m_final;
                        m_lastShownRcas = m_finalRcas;
                    }
                    else
                    {
                        // (after a held vblank: the picture held, not the real frame before it)
                        m_final = m_lastShown ? m_lastShown : m_lastPicture;
                        m_finalRcas = m_lastShown ? m_lastShownRcas : m_lastRcas;
                    }
                    return;
                }
                if (fresh || !m_genHistory[m_genIndex])
                {
                    preparePicture(cmd, fw, fh);
                    if (!m_final)
                        return;
                    m_genIndex ^= 1u;
                    m_genPrevValid = static_cast<bool>(m_genHistory[m_genIndex ^ 1u]);
                    const PassInput in[1] = {{m_final, Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(*cmd, m_genHistory[m_genIndex], m_final->get_width(), m_final->get_height(), m_copy, in, 1,
                                  nullptr, 0, false);
                    m_genRcas = m_finalRcas;
                    m_genJitterDelta[0] = m_jitter[0] - m_jitter[2];
                    m_genJitterDelta[1] = m_jitter[1] - m_jitter[3];
                }
                const Vulkan::Image *cur = m_genHistory[m_genIndex].get();
                const Vulkan::Image *prev = m_genHistory[m_genIndex ^ 1u].get();
                m_final = cur;
                m_finalRcas = m_genRcas;
                // The generated frame this present shows (presents beyond the factor repeat it).
                const uint32_t step = m_subframe * m_fg.factor / std::max(m_presents, m_fg.factor);
                const float k = static_cast<float>(m_fg.factor);
                const float t = m_fg.extrapolate ? static_cast<float>(step) / k
                                                 : static_cast<float>(std::min(step + 1u, m_fg.factor)) / k;
                const bool haveMotion = m_shared.motion && m_shared.depth && m_shared.motion->get_width() == m_shared.depth->get_width();
                const bool samePrev = prev && m_genPrevValid && prev->get_width() == cur->get_width() &&
                                      prev->get_height() == cur->get_height();
                const bool generate = haveMotion && (m_fg.extrapolate ? t > 0.0f : (t < 1.0f && samePrev));
                if (!generate)
                    return;
                struct
                {
                    float motionToUv[2];
                    float jitterDelta[2];
                    float rcpMotionSize[2];
                    float t;
                    float mode;
                } push = {{1.0f / 640.0f, 1.0f / 224.0f}, {m_genJitterDelta[0], m_genJitterDelta[1]},
                          {1.0f / static_cast<float>(m_shared.motion->get_width()), 1.0f / static_cast<float>(m_shared.motion->get_height())},
                          t, m_fg.extrapolate ? 1.0f : 0.0f};
                const PassInput in[4] = {{samePrev ? prev : cur, Vulkan::StockSampler::LinearClamp},
                                         {cur, Vulkan::StockSampler::LinearClamp},
                                         {m_shared.motion.get(), Vulkan::StockSampler::NearestClamp},
                                         {m_shared.depth.get(), Vulkan::StockSampler::NearestClamp}};
                offscreenPass(*cmd, m_genImage, cur->get_width(), cur->get_height(), m_frameGen, in, 4, &push, sizeof(push), false);
                m_final = m_genImage.get();
                (void)fw;
                (void)fh;
            }

            // Puts the UI back as the game drew it over the post-processed picture: the processed
            // image (finished at the picture's size on screen, RCAS included) is mixed with the
            // original by the GS's UI mask.
            void spareUi(Vulkan::CommandBuffer &cmd, float fw, float fh)
            {
                m_uiOriginal = nullptr;
                // A shadow frame takes its HUD from the real frame (same mask, same HUD).
                const Vulkan::Image *original = m_sourceOverride ? (m_flickerOut ? m_flickerOut.get()
                                                                                 : (m_shared.attached ? m_shared.scanout : m_cpuFrame).get())
                                                                 : sourceImage();
                if (!m_final || !original || m_final == original || !m_shared.ui ||
                    m_shared.ui->get_width() != original->get_width() || m_shared.ui->get_height() != original->get_height())
                    return;
                const VkRect2D rect = pictureRect(fw, fh);
                // In drawGame's pass to the screen (RCAS too, when due): two full-screen passes
                // fewer per present. RT_FUSED_UI=0: into images first, as before.
                static const bool fused = [] { const char *e = std::getenv("RT_FUSED_UI"); return !(e && *e == '0'); }();
                if (fused && (m_rcasUi || !m_finalRcas))
                {
                    m_uiOriginal = original;
                    return;
                }
                if (m_finalRcas)
                {
                    struct
                    {
                        AU1 con[4];
                        uint32_t origin[4];
                    } push = {};
                    FsrRcasCon(push.con, (1.0f - std::clamp(m_post.sharpness, 0.0f, 1.0f)) * 2.0f);
                    const PassInput in[1] = {{m_final, Vulkan::StockSampler::NearestClamp}};
                    offscreenPass(cmd, m_rcasImage, rect.extent.width, rect.extent.height, m_rcas, in, 1, &push, sizeof(push), false);
                    m_final = m_rcasImage.get();
                    m_finalRcas = false;
                }
                const PassInput in[3] = {{m_final, Vulkan::StockSampler::LinearClamp},
                                         {original, Vulkan::StockSampler::LinearClamp},
                                         {m_shared.ui.get(), Vulkan::StockSampler::LinearClamp}};
                offscreenPass(cmd, m_uiImage, rect.extent.width, rect.extent.height, m_uiComposite, in, 3, nullptr, 0, false);
                m_final = m_uiImage.get();
            }

            void drawGame(Vulkan::CommandBuffer &cmd, float fw, float fh)
            {
                if (!m_final)
                    return;
                const VkRect2D rect = pictureRect(fw, fh);
                if (const Vulkan::Image *original = m_uiOriginal)
                {
                    // spareUi's composite (and RCAS) straight into the picture's rectangle.
                    m_uiOriginal = nullptr;
                    cmd.set_program(m_finalRcas ? m_rcasUi : m_uiComposite);
                    cmd.set_opaque_state();
                    cmd.set_depth_test(false, false);
                    cmd.set_cull_mode(VK_CULL_MODE_NONE);
                    cmd.set_viewport({static_cast<float>(rect.offset.x), static_cast<float>(rect.offset.y),
                                      static_cast<float>(rect.extent.width), static_cast<float>(rect.extent.height), 0.0f, 1.0f});
                    cmd.set_scissor(rect);
                    cmd.set_texture(0, 0, m_final->get_view(), m_finalRcas ? Vulkan::StockSampler::NearestClamp : Vulkan::StockSampler::LinearClamp);
                    cmd.set_texture(0, 1, original->get_view(), Vulkan::StockSampler::LinearClamp);
                    cmd.set_texture(0, 2, m_shared.ui->get_view(), Vulkan::StockSampler::LinearClamp);
                    if (m_finalRcas)
                    {
                        struct
                        {
                            AU1 con[4];
                            uint32_t origin[4];
                            float extent[4];
                        } push = {};
                        FsrRcasCon(push.con, (1.0f - std::clamp(m_post.sharpness, 0.0f, 1.0f)) * 2.0f);
                        push.origin[0] = static_cast<uint32_t>(rect.offset.x);
                        push.origin[1] = static_cast<uint32_t>(rect.offset.y);
                        push.extent[0] = static_cast<float>(rect.extent.width);
                        push.extent[1] = static_cast<float>(rect.extent.height);
                        cmd.push_constants(&push, 0, sizeof(push));
                    }
                    cmd.draw(3);
                    cmd.set_viewport({0.0f, 0.0f, fw, fh, 0.0f, 1.0f});
                    return;
                }
                if (m_finalRcas)
                {
                    // FSR 1 RCAS, sharpening into the picture's rectangle on the swapchain.
                    struct
                    {
                        AU1 con[4];
                        uint32_t origin[4];
                    } push = {};
                    FsrRcasCon(push.con, (1.0f - std::clamp(m_post.sharpness, 0.0f, 1.0f)) * 2.0f);
                    push.origin[0] = static_cast<uint32_t>(rect.offset.x);
                    push.origin[1] = static_cast<uint32_t>(rect.offset.y);
                    cmd.set_program(m_rcas);
                    cmd.set_opaque_state();
                    cmd.set_depth_test(false, false);
                    cmd.set_cull_mode(VK_CULL_MODE_NONE);
                    cmd.set_viewport({static_cast<float>(rect.offset.x), static_cast<float>(rect.offset.y),
                                      static_cast<float>(rect.extent.width), static_cast<float>(rect.extent.height), 0.0f, 1.0f});
                    cmd.set_scissor(rect);
                    cmd.set_texture(0, 0, m_final->get_view(), Vulkan::StockSampler::NearestClamp);
                    cmd.push_constants(&push, 0, sizeof(push));
                    cmd.draw(3);
                    cmd.set_viewport({0.0f, 0.0f, fw, fh, 0.0f, 1.0f});
                    return;
                }
                const Vulkan::Image *image = m_final;
                const float w = static_cast<float>(rect.extent.width), h = static_cast<float>(rect.extent.height);
                const float x0 = static_cast<float>(rect.offset.x), y0 = static_cast<float>(rect.offset.y);
                bindState(cmd, fw, fh);
                cmd.set_blend_enable(false);
                cmd.set_scissor({{0, 0}, {static_cast<uint32_t>(fw), static_cast<uint32_t>(fh)}});
                auto *v = static_cast<Vertex *>(cmd.allocate_vertex_data(0, 4 * sizeof(Vertex), sizeof(Vertex)));
                v[0] = {x0, y0, 0, 0, 0xFFFFFFFFu};
                v[1] = {x0 + w, y0, 1, 0, 0xFFFFFFFFu};
                v[2] = {x0 + w, y0 + h, 1, 1, 0xFFFFFFFFu};
                v[3] = {x0, y0 + h, 0, 1, 0xFFFFFFFFu};
                auto *i = static_cast<uint16_t *>(cmd.allocate_index_data(6 * sizeof(uint16_t), VK_INDEX_TYPE_UINT16));
                const uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
                std::memcpy(i, idx, sizeof(idx));
                cmd.set_texture(0, 0, image->get_view(), Vulkan::StockSampler::LinearClamp);
                cmd.draw_indexed(6);
            }

        public:
            bool supportsPostProcess() const override { return true; }
            void setPostProcess(const PostProcess &post) override { m_post = post; }
            void setFrameGeneration(const FrameGeneration &fg) override
            {
                m_fg = fg;
                m_fg.factor = std::clamp<uint32_t>(fg.factor, 1u, 4u);
                // More presents in flight per guest frame: keep begin_frame's present-wait short.
                m_wsi.set_present_wait_latency(m_fg.factor > 1 ? 3u : 2u);
            }
            uint32_t frameGenerationFactor() const override { return m_fg.factor; }
            void setPresentsPerFrame(uint32_t n) override { m_presents = std::max(n, 1u); }

        private:

#if defined(PS2X_PGS_PRESENTER_UI)
            void updateTextures(ImDrawData *dd)
            {
                if (!dd->Textures)
                    return;
                for (ImTextureData *tex : *dd->Textures)
                {
                    if (tex->Status == ImTextureStatus_OK)
                        continue;
                    if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0)
                    {
                        m_textures.erase(tex->GetTexID());
                        tex->SetTexID(ImTextureID_Invalid);
                        tex->SetStatus(ImTextureStatus_Destroyed);
                        continue;
                    }
                    if (tex->Status != ImTextureStatus_WantCreate && tex->Status != ImTextureStatus_WantUpdates)
                        continue;
                    // (Re)upload the whole texture: atlas updates are rare.
                    std::vector<uint8_t> rgba;
                    const uint8_t *pixels = static_cast<const uint8_t *>(tex->GetPixels());
                    if (tex->Format == ImTextureFormat_Alpha8)
                    {
                        rgba.resize(static_cast<size_t>(tex->Width) * tex->Height * 4u);
                        for (int p = 0; p < tex->Width * tex->Height; ++p)
                        {
                            rgba[p * 4 + 0] = rgba[p * 4 + 1] = rgba[p * 4 + 2] = 0xFF;
                            rgba[p * 4 + 3] = pixels[p];
                        }
                        pixels = rgba.data();
                    }
                    auto info = Vulkan::ImageCreateInfo::immutable_2d_image(tex->Width, tex->Height, VK_FORMAT_R8G8B8A8_UNORM);
                    Vulkan::ImageInitialData init = {pixels, 0, 0};
                    ImTextureID id = tex->GetTexID();
                    if (id == ImTextureID_Invalid)
                        id = static_cast<ImTextureID>(++m_nextTexture);
                    m_textures[id] = m_shared.device->create_image(info, &init);
                    tex->SetTexID(id);
                    tex->SetStatus(ImTextureStatus_OK);
                }
            }

            void drawUi(Vulkan::CommandBuffer &cmd, float fw, float fh)
            {
                if (m_uiFrame)
                    drawImGui(cmd, ImGui::GetDrawData(), fw, fh);
            }

            // An ImGui texture: the presenter's own (kPictureTexture) or one of the font atlas's.
            const Vulkan::ImageView *uiTexture(ImTextureID id)
            {
                if (id == static_cast<ImTextureID>(kMapTexture))
                    return m_shared.map ? &m_shared.map->get_view() : nullptr;
                if (id == static_cast<ImTextureID>(kPictureTexture))
                {
                    const Vulkan::Image *picture = m_shared.attached ? m_shared.scanout.get() : m_cpuFrame.get();
                    return picture ? &picture->get_view() : nullptr;
                }
                auto it = m_textures.find(id);
                return it == m_textures.end() || !it->second ? nullptr : &it->second->get_view();
            }

            void drawImGui(Vulkan::CommandBuffer &cmd, ImDrawData *dd, float fw, float fh)
            {
                if (!dd || dd->TotalVtxCount == 0)
                    return;
                const ImVec2 scale = dd->FramebufferScale;
                // ImGui works in window points; the swapchain is in pixels.
                bindState(cmd, dd->DisplaySize.x, dd->DisplaySize.y, dd->DisplayPos.x, dd->DisplayPos.y);
                auto *v = static_cast<ImDrawVert *>(
                    cmd.allocate_vertex_data(0, static_cast<VkDeviceSize>(dd->TotalVtxCount) * sizeof(ImDrawVert), sizeof(ImDrawVert)));
                auto *ix = static_cast<ImDrawIdx *>(cmd.allocate_index_data(
                    static_cast<VkDeviceSize>(dd->TotalIdxCount) * sizeof(ImDrawIdx),
                    sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32));
                for (const ImDrawList *list : dd->CmdLists)
                {
                    std::memcpy(v, list->VtxBuffer.Data, list->VtxBuffer.Size * sizeof(ImDrawVert));
                    std::memcpy(ix, list->IdxBuffer.Data, list->IdxBuffer.Size * sizeof(ImDrawIdx));
                    v += list->VtxBuffer.Size;
                    ix += list->IdxBuffer.Size;
                }
                uint32_t vtxBase = 0, idxBase = 0;
                for (const ImDrawList *list : dd->CmdLists)
                {
                    for (const ImDrawCmd &c : list->CmdBuffer)
                    {
                        if (c.UserCallback)
                            continue;
                        const float x0 = std::max(0.0f, (c.ClipRect.x - dd->DisplayPos.x) * scale.x);
                        const float y0 = std::max(0.0f, (c.ClipRect.y - dd->DisplayPos.y) * scale.y);
                        const float x1 = std::min(fw, (c.ClipRect.z - dd->DisplayPos.x) * scale.x);
                        const float y1 = std::min(fh, (c.ClipRect.w - dd->DisplayPos.y) * scale.y);
                        if (x1 <= x0 || y1 <= y0)
                            continue;
                        const Vulkan::ImageView *view = uiTexture(c.GetTexID());
                        if (!view)
                            continue;
                        cmd.set_scissor({{static_cast<int32_t>(x0), static_cast<int32_t>(y0)},
                                         {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}});
                        cmd.set_texture(0, 0, *view,
                                        m_nearestTextures.count(static_cast<uint64_t>(c.GetTexID())) ? Vulkan::StockSampler::NearestClamp
                                                                                                    : Vulkan::StockSampler::LinearClamp);
                        cmd.draw_indexed(c.ElemCount, 1, idxBase + c.IdxOffset, static_cast<int32_t>(vtxBase + c.VtxOffset), 0);
                    }
                    vtxBase += static_cast<uint32_t>(list->VtxBuffer.Size);
                    idxBase += static_cast<uint32_t>(list->IdxBuffer.Size);
                }
            }
#else
            void drawUi(Vulkan::CommandBuffer &, float, float) {}
#endif

            // ---- The second screen (see HostPresenter::setSecondScreen) ----
        public:
            void setSecondScreen(void *nativeWindow) override
            {
#if defined(__ANDROID__)
                if (nativeWindow)
                    ANativeWindow_acquire(static_cast<ANativeWindow *>(nativeWindow));
                std::lock_guard<std::mutex> lock(m_second.pendingMutex);
                if (m_second.pendingSet && m_second.pending)
                    ANativeWindow_release(static_cast<ANativeWindow *>(m_second.pending));
                m_second.pending = nativeWindow;
                m_second.pendingSet = true;
#else
                (void)nativeWindow;
#endif
            }

            bool secondScreenSize(int &width, int &height) const override
            {
                const uint64_t size = m_second.size.load();
                if (!size)
                    return false;
                width = static_cast<int>(size >> 32);
                height = static_cast<int>(size & 0xFFFFFFFFu);
                return true;
            }

            void submitSecondScreenUi(void *imguiContext) override { m_second.ui = imguiContext; }

            void setWantMap(bool want) override { m_shared.wantMap = want; }

            bool mapRegion(float uv[4], float &aspect) const override
            {
                std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(m_shared.mutex));
                if (!m_shared.map)
                    return false;
                std::memcpy(uv, m_shared.mapUv, sizeof(m_shared.mapUv));
                aspect = m_shared.mapAspect;
                return true;
            }

            uint64_t createUiTexture(const uint8_t *rgba, int width, int height, bool nearest) override
            {
                if (!rgba || width <= 0 || height <= 0 || !m_shared.device)
                    return 0;
                std::lock_guard<std::mutex> lock(m_shared.mutex);
                pgsRegisterThread();
                auto info = Vulkan::ImageCreateInfo::immutable_2d_image(static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                                                                        VK_FORMAT_R8G8B8A8_UNORM);
                Vulkan::ImageInitialData init = {rgba, 0, 0};
                const uint64_t id = ++m_nextTexture;
                m_textures[id] = m_shared.device->create_image(info, &init);
                if (nearest)
                    m_nearestTextures.insert(id);
                return id;
            }

            void destroyUiTexture(uint64_t id) override
            {
                std::lock_guard<std::mutex> lock(m_shared.mutex);
                pgsRegisterThread();
                // Granite defers the image's destruction until the frames using it are done.
                m_textures.erase(id);
                m_nearestTextures.erase(id);
            }

        private:
            struct SecondScreen
            {
                std::mutex pendingMutex;
                void *pending = nullptr; // a new ANativeWindow (acquired), or nullptr
                bool pendingSet = false;
                void *window = nullptr; // the ANativeWindow in use (acquired)
                VkSurfaceKHR surface = VK_NULL_HANDLE;
                VkSwapchainKHR swapchain = VK_NULL_HANDLE;
                VkExtent2D extent = {};
                bool outOfDate = false;
                VkFence acquireFence = VK_NULL_HANDLE; // the acquire's own fence (see renderSecondScreen)
                std::vector<Vulkan::ImageHandle> images;
                std::vector<Vulkan::ImageViewHandle> views;
                std::vector<Vulkan::Semaphore> release; // per image, until it is acquired again
                std::atomic<uint64_t> size{0};          // width << 32 | height, 0 = none
                void *ui = nullptr;                     // this frame's ImGui context, if submitted
            } m_second;

            void destroySecondSwapchain()
            {
                Vulkan::Device &dev = *m_shared.device;
                if (m_second.swapchain == VK_NULL_HANDLE)
                    return;
                dev.wait_idle();
                m_second.views.clear();
                m_second.images.clear();
                m_second.release.clear();
                dev.get_device_table().vkDestroySwapchainKHR(dev.get_device(), m_second.swapchain, nullptr);
                m_second.swapchain = VK_NULL_HANDLE;
                m_second.size = 0;
            }

            void destroySecondScreen()
            {
                destroySecondSwapchain();
#if defined(__ANDROID__)
                if (m_second.surface != VK_NULL_HANDLE)
                    vkDestroySurfaceKHR(m_shared.device->get_instance(), m_second.surface, nullptr);
                m_second.surface = VK_NULL_HANDLE;
                if (m_second.window)
                    ANativeWindow_release(static_cast<ANativeWindow *>(m_second.window));
                m_second.window = nullptr;
#endif
            }

            bool createSecondSwapchain()
            {
                Vulkan::Device &dev = *m_shared.device;
                const VkPhysicalDevice gpu = dev.get_physical_device();
                const auto &queues = m_wsi.get_context().get_queue_info();
                VkBool32 supported = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(gpu, queues.family_indices[Vulkan::QUEUE_INDEX_GRAPHICS], m_second.surface,
                                                     &supported);
                if (!supported)
                    return fail("second screen", "the graphics queue can't present to it");
                VkSurfaceCapabilitiesKHR caps = {};
                vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, m_second.surface, &caps);
                VkExtent2D extent = caps.currentExtent;
#if defined(__ANDROID__)
                if (extent.width == 0xFFFFFFFFu)
                    extent = {static_cast<uint32_t>(ANativeWindow_getWidth(static_cast<ANativeWindow *>(m_second.window))),
                              static_cast<uint32_t>(ANativeWindow_getHeight(static_cast<ANativeWindow *>(m_second.window)))};
#endif
                if (!extent.width || !extent.height)
                    return false;
                uint32_t count = 0;
                vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, m_second.surface, &count, nullptr);
                std::vector<VkSurfaceFormatKHR> formats(count);
                vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, m_second.surface, &count, formats.data());
                if (formats.empty())
                    return false;
                VkSurfaceFormatKHR format = formats[0];
                for (const auto &f : formats)
                    if (f.format == VK_FORMAT_R8G8B8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_UNORM)
                    {
                        format = f;
                        break;
                    }
                // Never wait for this screen: MAILBOX when there is one, and the acquire below
                // doesn't block either way.
                vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, m_second.surface, &count, nullptr);
                std::vector<VkPresentModeKHR> modes(count);
                vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, m_second.surface, &count, modes.data());
                const bool mailbox = std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end();
                VkSwapchainCreateInfoKHR info = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
                info.surface = m_second.surface;
                info.minImageCount = std::max(caps.minImageCount, 3u);
                if (caps.maxImageCount)
                    info.minImageCount = std::min(info.minImageCount, caps.maxImageCount);
                info.imageFormat = format.format;
                info.imageColorSpace = format.colorSpace;
                info.imageExtent = extent;
                info.imageArrayLayers = 1;
                info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                // The compositor turns the picture with the display; no rotated rendering here.
                info.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                                        ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                                        : caps.currentTransform;
                info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
                for (VkCompositeAlphaFlagBitsKHR a : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
                                                      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR})
                    if (caps.supportedCompositeAlpha & a)
                    {
                        info.compositeAlpha = a;
                        break;
                    }
                info.presentMode = mailbox ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
                info.clipped = VK_TRUE;
                const auto &table = dev.get_device_table();
                if (table.vkCreateSwapchainKHR(dev.get_device(), &info, nullptr, &m_second.swapchain) != VK_SUCCESS)
                {
                    m_second.swapchain = VK_NULL_HANDLE;
                    return fail("second screen", "vkCreateSwapchainKHR failed");
                }
                table.vkGetSwapchainImagesKHR(dev.get_device(), m_second.swapchain, &count, nullptr);
                std::vector<VkImage> images(count);
                table.vkGetSwapchainImagesKHR(dev.get_device(), m_second.swapchain, &count, images.data());
                auto imageInfo = Vulkan::ImageCreateInfo::render_target(extent.width, extent.height, format.format);
                imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                for (VkImage image : images)
                {
                    m_second.images.push_back(dev.wrap_image(imageInfo, image));
                    Vulkan::ImageViewCreateInfo view = {};
                    view.image = m_second.images.back().get();
                    view.format = format.format;
                    view.view_type = VK_IMAGE_VIEW_TYPE_2D;
                    view.levels = 1;
                    view.layers = 1;
                    m_second.views.push_back(dev.create_image_view(view));
                }
                m_second.release.resize(images.size());
                m_second.extent = extent;
                m_second.outOfDate = false;
                m_second.size = (static_cast<uint64_t>(extent.width) << 32) | extent.height;
                std::cout << "[presenter] second screen " << extent.width << "x" << extent.height
                          << (mailbox ? " (mailbox)" : " (fifo)") << std::endl;
                return true;
            }

            // With the device lock held, after the main screen's frame.
            void renderSecondScreen()
            {
                Vulkan::Device &dev = *m_shared.device;
#if defined(__ANDROID__)
                {
                    std::lock_guard<std::mutex> lock(m_second.pendingMutex);
                    if (m_second.pendingSet)
                    {
                        destroySecondScreen();
                        m_second.window = m_second.pending;
                        m_second.pending = nullptr;
                        m_second.pendingSet = false;
                        if (m_second.window)
                        {
                            auto create = reinterpret_cast<PFN_vkCreateAndroidSurfaceKHR>(
                                vkGetInstanceProcAddr(dev.get_instance(), "vkCreateAndroidSurfaceKHR"));
                            VkAndroidSurfaceCreateInfoKHR info = {VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
                            info.window = static_cast<ANativeWindow *>(m_second.window);
                            if (!create || create(dev.get_instance(), &info, nullptr, &m_second.surface) != VK_SUCCESS)
                            {
                                m_second.surface = VK_NULL_HANDLE;
                                fail("second screen", "vkCreateAndroidSurfaceKHR failed");
                            }
                        }
                    }
                }
#endif
                void *ui = m_second.ui;
                m_second.ui = nullptr;
                if (m_second.surface == VK_NULL_HANDLE)
                    return;
                if (m_second.outOfDate)
                    destroySecondSwapchain();
                if (m_second.swapchain == VK_NULL_HANDLE && !createSecondSwapchain())
                    return;
#if defined(PS2X_PGS_PRESENTER_UI)
                if (!ui)
                    return;
                // The image is acquired with a fence of our own and waited for before drawing:
                // a semaphore handed to add_wait_semaphore joins the device's next submission,
                // which another thread's work could take, and then this screen was drawn into an
                // image still being shown (strips of garbage on the Thor's lower screen when the
                // GPU was busy, e.g. Q's Factory).
                uint32_t index = 0;
                const auto &table = dev.get_device_table();
                if (m_second.acquireFence == VK_NULL_HANDLE)
                {
                    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                    table.vkCreateFence(dev.get_device(), &fi, nullptr, &m_second.acquireFence);
                }
                const VkResult acquired = table.vkAcquireNextImageKHR(dev.get_device(), m_second.swapchain, 0,
                                                                      VK_NULL_HANDLE, m_second.acquireFence, &index);
                if (acquired == VK_ERROR_OUT_OF_DATE_KHR || acquired == VK_ERROR_SURFACE_LOST_KHR)
                {
                    m_second.outOfDate = true;
                    return;
                }
                if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR)
                    return; // no image free yet (the screen is behind): skip it this frame
                table.vkWaitForFences(dev.get_device(), 1, &m_second.acquireFence, VK_TRUE, UINT64_MAX);
                table.vkResetFences(dev.get_device(), 1, &m_second.acquireFence);

                ImGuiContext *previous = ImGui::GetCurrentContext();
                ImGui::SetCurrentContext(static_cast<ImGuiContext *>(ui));
                ImDrawData *dd = ImGui::GetDrawData();
                if (dd)
                    updateTextures(dd);
                const Vulkan::Image &image = *m_second.images[index];
                const float fw = static_cast<float>(m_second.extent.width), fh = static_cast<float>(m_second.extent.height);
                auto cmd = dev.request_command_buffer();
                cmd->image_barrier(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                Vulkan::RenderPassInfo rp = {};
                rp.num_color_attachments = 1;
                rp.color_attachments[0] = m_second.views[index].get();
                rp.clear_attachments = 1;
                rp.store_attachments = 1;
                cmd->begin_render_pass(rp);
                drawImGui(*cmd, dd, fw, fh);
                cmd->end_render_pass();
                ImGui::SetCurrentContext(previous);
                cmd->image_barrier(image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_NONE, 0);
                Vulkan::Semaphore release;
                dev.submit(cmd, nullptr, 1, &release);

                const VkSemaphore wait = release->get_semaphore();
                VkResult result = VK_SUCCESS;
                VkPresentInfoKHR present = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
                present.waitSemaphoreCount = 1;
                present.pWaitSemaphores = &wait;
                present.swapchainCount = 1;
                present.pSwapchains = &m_second.swapchain;
                present.pImageIndices = &index;
                present.pResults = &result;
                const VkQueue queue = m_wsi.get_context().get_queue_info().queues[Vulkan::QUEUE_INDEX_GRAPHICS];
                const PFN_vkQueuePresentKHR presentFn = g_realQueuePresent ? g_realQueuePresent : table.vkQueuePresentKHR;
                VkResult overall;
                if (g_presentQueue.running())
                    overall = threadedPresent(queue, &present); // (the present thread: no lock held over it)
                else
                {
                    dev.external_queue_lock();
                    overall = presentFn(queue, &present);
                    dev.external_queue_unlock();
                }
                release->wait_external(); // consumed by the present, even when it fails
                m_second.release[index] = std::move(release);
                // (SUBOPTIMAL is expected: the compositor turns the unrotated picture.)
                if (overall == VK_ERROR_OUT_OF_DATE_KHR || overall == VK_ERROR_SURFACE_LOST_KHR)
                    m_second.outOfDate = true;
#else
                (void)ui;
#endif
            }

            void drawScene(Vulkan::CommandBuffer &cmd, float fw, float fh)
            {
                drawGame(cmd, fw, fh);
                drawUi(cmd, fw, fh);
            }

            void render()
            {
                // The last frame's presents have gone to the driver (no backlog: pacing as before,
                // but waited for here, holding nothing).
                g_presentQueue.waitBelow(0);
                const auto lockAsked = std::chrono::steady_clock::now();
                std::unique_lock<std::mutex> lock(m_shared.mutex);
                // (RT_PRESENT_DEBUG's pacing line: how long the lock was waited for and held.)
                auto lockedAt = std::chrono::steady_clock::now();
                m_lockWaitMs = std::chrono::duration<double, std::milli>(lockedAt - lockAsked).count();
                m_lockHeldMs = 0.0;
                struct Held
                {
                    std::unique_lock<std::mutex> &lock;
                    std::chrono::steady_clock::time_point &since;
                    double &total;
                    ~Held()
                    {
                        if (lock.owns_lock())
                            total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
                    }
                } held{lock, lockedAt, m_lockHeldMs};
                pgsRegisterThread();
                Vulkan::Device &dev = *m_shared.device;
#if defined(PS2X_PGS_PRESENTER_UI)
                if (m_uiFrame)
                    updateTextures(ImGui::GetDrawData());
#endif
                if (m_shared.flushLocked)
                    m_shared.flushLocked();
                const Vulkan::Image *backImage = nullptr;
                if (m_metalPresent)
                {
                    // Into an offscreen backbuffer (a ring of three: Metal may still be copying the
                    // one before), presented through Metal after the lock is released.
                    dev.next_frame_context();
                    int pw = 0, ph = 0;
                    SDL_GetWindowSizeInPixels(m_window, &pw, &ph);
                    const uint32_t bw = static_cast<uint32_t>(std::max(pw, 1)), bh = static_cast<uint32_t>(std::max(ph, 1));
                    m_backIndex = (m_backIndex + 1) % 3;
                    Vulkan::ImageHandle &bb = m_backbuffers[m_backIndex];
                    if (!bb || bb->get_width() != bw || bb->get_height() != bh)
                    {
                        auto info = Vulkan::ImageCreateInfo::render_target(bw, bh, VK_FORMAT_B8G8R8A8_UNORM);
                        info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                        info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                        bb = dev.create_image(info);
                    }
                    backImage = bb.get();
                }
                else
                {
                    // Acquiring the next image may wait a refresh or more (present wait). A
                    // backend that never leaves command buffers open outside its lock (the
                    // hardware GS) keeps drawing meanwhile: the lock is let go for it. (Holding it
                    // there cost the hardware GS's frame generation half its frames.)
                    bool begun = false;
                    if (!m_background)
                    {
                        if (m_shared.submitsUnderLock)
                        {
                            m_lockHeldMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lockedAt).count();
                            lock.unlock();
                            begun = m_wsi.begin_frame();
                            lock.lock();
                            lockedAt = std::chrono::steady_clock::now();
                        }
                        else
                            begun = m_wsi.begin_frame();
                    }
                    if (!begun)
                        return;
                    backImage = &dev.get_swapchain_view().get_image();
                }
                const Vulkan::Image &back = *backImage;
                const float fw = static_cast<float>(back.get_width()), fh = static_cast<float>(back.get_height());
                auto cmd = dev.request_command_buffer();
                // A new guest frame, or another display refresh of the same one.
                m_sourceOverride = nullptr;
                const bool fresh = m_lastTick != m_renderedTick;
                m_renderedTick = m_lastTick;
                m_subframe = fresh ? 0u : m_subframe + 1u;
                // RT_PRESENT_DUMP=<dir> (with RT_PRESENT_DUMP_FROM / _TO, guest vblanks): every guest
                // frame's presents in that range as <dir>/v<vblank>_<present>.png, the real frame
                // (0) and its generated ones (1..), as shown. With the lockstep test socket the
                // presenter shows each parked frame long enough for all of them.
                static const char *presentDump = std::getenv("RT_PRESENT_DUMP");
                if (presentDump && *presentDump && m_capturePath.empty() && m_subframe < std::max(m_fg.factor, 1u))
                {
                    static const uint64_t from = std::strtoull(std::getenv("RT_PRESENT_DUMP_FROM") ? std::getenv("RT_PRESENT_DUMP_FROM") : "0", nullptr, 10);
                    static const uint64_t to = std::strtoull(std::getenv("RT_PRESENT_DUMP_TO") ? std::getenv("RT_PRESENT_DUMP_TO") : "0", nullptr, 10);
                    // RT_PRESENT_DUMP_EVERY=k: only vblanks from + k*i (all presents) and the next one's real frame.
                    static const uint64_t every = std::max<uint64_t>(1, std::strtoull(std::getenv("RT_PRESENT_DUMP_EVERY") ? std::getenv("RT_PRESENT_DUMP_EVERY") : "1", nullptr, 10));
                    const uint64_t phase = (m_lastTick - from) % every;
                    if (m_lastTick >= from && m_lastTick <= to && (every == 1 || phase == 0 || ((phase == 1 || phase == every - 1) && m_subframe == 0)))
                    {
                        char name[64];
                        std::snprintf(name, sizeof(name), "/v%06llu_%u.png", static_cast<unsigned long long>(m_lastTick), m_subframe);
                        m_capturePath = std::string(presentDump) + name;
                    }
                }
                // RT_GPU_TIMES=1: GPU time of the presenter's passes and the GS's recordings, logged
                // every 2 s (per occurrence).
                static const bool gpuTimes = [] { const char *e = std::getenv("RT_GPU_TIMES"); return e && *e == '1'; }();
                Vulkan::QueryPoolHandle tsPost = gpuTimes ? cmd->write_timestamp(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) : Vulkan::QueryPoolHandle{};
                if (fresh)
                {
                    Vulkan::QueryPoolHandle tsFlicker = gpuTimes ? cmd->write_timestamp(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) : Vulkan::QueryPoolHandle{};
                    blendFlicker(*cmd);
                    if (gpuTimes)
                        dev.register_time_interval("GPU", std::move(tsFlicker), cmd->write_timestamp(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT),
                                                   "post: flicker blending");
                }
                m_plTag = fresh ? 'R' : 'r';
                if (m_fg.factor > 1)
                    generatedPicture(cmd, fw, fh, fresh);
                if (presentLogOn())
                {
                    g_plTag = m_plTag, g_plTick = m_lastTick, g_plSub = m_subframe, g_plRender = plNow();
                    g_plPic = m_shared.pictureSerial, g_plPres = m_shared.presentSerial;
                    g_plSeq.fetch_add(1);
                }
                else if (fresh || !m_lastPicture)
                {
                    preparePicture(cmd, fw, fh);
                    if (m_showUi && m_shared.ui)
                        m_final = m_shared.ui.get(), m_finalRcas = false;
                    m_lastPicture = m_final;
                    m_lastRcas = m_finalRcas;
                }
                else
                {
                    // Another refresh of the same guest frame: the same picture (TAA and MetalFX
                    // history must see each guest frame once).
                    m_final = m_lastPicture;
                    m_finalRcas = m_lastRcas;
                }
                spareUi(*cmd, fw, fh);
                if (gpuTimes)
                {
                    dev.register_time_interval("GPU", std::move(tsPost), cmd->write_timestamp(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT),
                                               fresh ? "post: real frame" : "post: repeated/generated frame");
                    static auto last = std::chrono::steady_clock::now();
                    if (std::chrono::steady_clock::now() - last > std::chrono::seconds(2))
                    {
                        last = std::chrono::steady_clock::now();
                        dev.timestamp_log([](const std::string &tag, const Vulkan::TimestampIntervalReport &r) {
                            std::fprintf(stderr, "[gputime] %s: %.2f ms each, %.1f per frame context\n", tag.c_str(),
                                         r.time_per_accumulation * 1000.0, r.accumulations_per_frame_context);
                        });
                        dev.timestamp_log_reset();
                    }
                }
#if defined(__APPLE__)
                if (m_metalPresent)
                {
                    cmd->image_barrier(back, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, 0,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                    Vulkan::RenderPassInfo rp = {};
                    rp.num_color_attachments = 1;
                    rp.color_attachments[0] = &back.get_view();
                    rp.clear_attachments = 1;
                    rp.store_attachments = 1;
                    cmd->begin_render_pass(rp);
                    drawScene(*cmd, fw, fh);
                    cmd->end_render_pass();
                    cmd->image_barrier(back, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                       VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                    dev.submit(cmd); // MoltenVK commits it to the Metal queue before returning
                    if (!m_capturePath.empty())
                        captureLocked(fw, fh);
                    schedulePresent();
                    void *texture = nullptr, *queue = nullptr;
                    m_getTexture(back.get_image(), &texture);
                    m_getQueue(m_wsi.get_context().get_queue_info().queues[Vulkan::QUEUE_INDEX_GRAPHICS], &queue);
                    const uint64_t desired = g_desiredPresentNs, target = g_targetRefreshNs;
                    g_desiredPresentNs = 0;
                    m_lockHeldMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lockedAt).count();
                    lock.unlock();
                    // Waits for a drawable here, holding nothing the GS thread needs.
                    m_metal->present(queue, texture, back.get_width(), back.get_height(), desired ? static_cast<double>(desired) / 1e9 : 0.0,
                                     [this, target](double presented) {
                                         if (presented > 0.0 && target)
                                             notePresented(static_cast<int64_t>(target), static_cast<int64_t>(presented * 1e9));
                                     });
                    return;
                }
#endif
                auto rp = dev.get_swapchain_render_pass(Vulkan::SwapchainRenderPass::ColorOnly);
                rp.clear_color[0] = {};
                Vulkan::QueryPoolHandle tsPass = gpuTimes ? cmd->write_timestamp(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) : Vulkan::QueryPoolHandle{};
                cmd->begin_render_pass(rp);
                drawScene(*cmd, fw, fh);
                cmd->end_render_pass();
                if (gpuTimes)
                    dev.register_time_interval("GPU", std::move(tsPass), cmd->write_timestamp(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT),
                                               "swapchain pass");
                dev.submit(cmd);
                if (!m_capturePath.empty())
                    captureLocked(fw, fh);
                schedulePresent();
                if (m_shared.submitsUnderLock)
                {
                    // The present can block in the driver for many milliseconds (Adreno sleeps in
                    // queueBuffer), and the hardware GS, which keeps no command buffer open outside
                    // the lock, would wait for it all that time and miss the game's vblank: the
                    // lock is let go for it, as for begin_frame. (RT_PRESENT_UNDER_LOCK=1: as before.)
                    static const bool underLock = [] { const char *e = std::getenv("RT_PRESENT_UNDER_LOCK"); return e && *e == '1'; }();
                    if (!underLock)
                    {
                        m_lockHeldMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lockedAt).count();
                        lock.unlock();
                        m_wsi.end_frame();
                        lock.lock();
                        lockedAt = std::chrono::steady_clock::now();
                        renderSecondScreen();
                        return;
                    }
                }
                m_wsi.end_frame();
                renderSecondScreen();
            }

            // Metal's presented times (from a Metal thread) for the adaptive present lead.
            void notePresented(int64_t target, int64_t presented)
            {
                std::lock_guard<std::mutex> lock(m_presentedMutex);
                if (m_presented.size() < 64)
                    m_presented.push_back({target, presented});
            }

            // With present-at-time, presents wait in the queue for their refresh, so the swapchain's
            // images (Metal keeps at most 3 drawables) can all be in flight at 120 presents per
            // second, and acquiring the next one would block inside render(), under the device
            // lock the GS thread needs (the game then drops to 30 fps). Wait here instead, outside
            // the lock, until the present two back is on screen and its image is free again.
            void waitForDrawable()
            {
                if (m_metalPresent)
                    return; // Metal fetches its drawable after the device lock is released
                const int64_t due = m_presentTargets[(m_presentCount + 1) % 3]; // two presents back
                if (!m_presentAtTime || !due)
                    return;
                const int64_t wait = due + 1000000 - ps2x::hostTimeNowNs();
                if (wait > 0)
                    std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<int64_t>(wait, 50000000)));
            }

            // Puts each present on a fixed grid of display refreshes: one every
            // refresh-rate / present-rate refreshes (every 2nd at 60 fps on a 120 Hz panel), so
            // every frame is on screen for the same time. Left alone, presents land anywhere in a
            // refresh and frames alternate between 1 and 3 refreshes (judder).
            void schedulePresent()
            {
                int64_t lastRefresh = 0, period = 0;
                if (!m_presentAtTime || !ps2x::displayClockSampleHost(lastRefresh, period) || ps2_test::paused())
                {
                    m_presentTarget = 0;
                    m_presentTargets[0] = m_presentTargets[1] = m_presentTargets[2] = 0;
                    return;
                }
                const double presentRate = 60.0 * static_cast<double>(std::max(m_presents, 1u));
                const int64_t step = period * std::max<int64_t>(1, std::llround(1e9 / static_cast<double>(period) / presentRate));
                // The earliest refresh this frame can make: the GPU still has the frame's work to
                // do and Core Animation needs it before that refresh. The lead adapts to what
                // this machine needs (on-screen times from VK_GOOGLE_display_timing): two misses
                // close together add 1 ms, every 30 frames on time take a quarter off (2 ms to one
                // refresh). A fixed half refresh was too little for an 8x race on an M3 Max; a
                // refresh and a half cost 8 ms of latency.
                updatePresentLead(period);
                const int64_t earliest = ps2x::hostTimeNowNs() + m_presentLead;
                auto onGrid = [&](int64_t t) { return lastRefresh + ((t - lastRefresh + period - 1) / period) * period; };
                int64_t target = m_presentTarget ? m_presentTarget + step : onGrid(earliest);
                if (target < earliest || target > earliest + 2 * step)
                    target = onGrid(earliest); // late (a frame was slow) or drifted: start the grid again
                m_presentTarget = target;
                m_presentTargets[m_presentCount++ % 3] = target;
                // Half a refresh early: Metal shows it at the first refresh at or after this time.
                g_desiredPresentNs = static_cast<uint64_t>(target - period / 2);
                g_targetRefreshNs = static_cast<uint64_t>(target);
            }

            void updatePresentLead(int64_t period)
            {
                const int64_t minLead = 2000000, maxLead = period;
                if (m_presentLead == 0)
                    m_presentLead = period / 2;
                // (target refresh, time on screen) of recent presents: from Metal's presented
                // handlers, or VK_GOOGLE_display_timing on the swapchain path.
                std::vector<std::pair<int64_t, int64_t>> shown;
                if (m_metalPresent)
                {
                    std::lock_guard<std::mutex> lock(m_presentedMutex);
                    shown.swap(m_presented);
                }
                else
                {
                    if (!m_pastTiming && g_timedSwapchain)
                        m_pastTiming = reinterpret_cast<PFN_vkGetPastPresentationTimingGOOGLE>(
                            vkGetDeviceProcAddr(m_wsi.get_context().get_device(), "vkGetPastPresentationTimingGOOGLE"));
                    if (!m_pastTiming || !g_timedSwapchain)
                        return;
                    VkPastPresentationTimingGOOGLE past[16];
                    uint32_t count = 16;
                    VkResult r;
                    {
                        std::lock_guard<std::mutex> swapchain(g_presentQueue.swapchainMutex(g_timedSwapchain)); // (externally synchronised)
                        r = m_pastTiming(m_wsi.get_context().get_device(), g_timedSwapchain, &count, past);
                    }
                    if (r != VK_SUCCESS && r != VK_INCOMPLETE)
                        return;
                    for (uint32_t i = 0; i < count; ++i)
                    {
                        const TimedPresent &sent = g_timedPresents[past[i].presentID % 16];
                        if (sent.id == past[i].presentID && sent.target && past[i].actualPresentTime)
                            shown.push_back({sent.target, static_cast<int64_t>(past[i].actualPresentTime)});
                    }
                }
                for (const auto &[sentTarget, actual] : shown)
                {
                    // Metal's presented time sits a steady few ms after the display link's refresh
                    // time; a miss shows up a whole refresh later than that offset.
                    const int64_t offset = actual - sentTarget;
                    m_offsetWindowMin = std::min(m_offsetWindowMin, offset);
                    if (++m_offsetSamples >= 240)
                    {
                        m_presentOffset = m_offsetWindowMin;
                        m_offsetWindowMin = INT64_MAX;
                        m_offsetSamples = 0;
                    }
                    if (m_presentOffset == INT64_MAX)
                        m_presentOffset = offset;
                    if (offset > m_presentOffset + period / 2)
                    {
                        // Missed its refresh. A lone miss is usually a stall elsewhere (a slow
                        // guest frame, the system); two within 60 presents mean the lead is short.
                        if (m_presentsSinceMiss < 60)
                            m_presentLead = std::min(m_presentLead + 1000000, maxLead);
                        m_presentsSinceMiss = 0;
                        m_presentsOnTime = 0;
                    }
                    else
                    {
                        m_presentsSinceMiss = std::min(m_presentsSinceMiss + 1, 1000u);
                        if (++m_presentsOnTime >= 30)
                        {
                            m_presentLead = std::max(m_presentLead - 250000, minLead);
                            m_presentsOnTime = 0;
                        }
                    }
                }
                if (std::getenv("RT_PRESENT_DEBUG"))
                {
                    static int n = 0;
                    if ((n++ % 120) == 0)
                        std::fprintf(stderr, "[present] lead %.2f ms\n", static_cast<double>(m_presentLead) / 1e6);
                }
            }

            // Draws the frame again into an image and saves it as a PNG.
            void captureLocked(float fw, float fh)
            {
                Vulkan::Device &dev = *m_shared.device;
                const uint32_t w = static_cast<uint32_t>(fw), h = static_cast<uint32_t>(fh);
                auto target = dev.create_image(Vulkan::ImageCreateInfo::render_target(w, h, VK_FORMAT_R8G8B8A8_UNORM));
                Vulkan::BufferCreateInfo binfo = {};
                binfo.size = static_cast<VkDeviceSize>(w) * h * 4u;
                binfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                binfo.domain = Vulkan::BufferDomain::CachedHost;
                auto buffer = dev.create_buffer(binfo);

                auto cmd = dev.request_command_buffer();
                Vulkan::RenderPassInfo rp = {};
                rp.num_color_attachments = 1;
                rp.color_attachments[0] = &target->get_view();
                rp.clear_attachments = 1;
                rp.store_attachments = 1;
                cmd->begin_render_pass(rp);
                drawScene(*cmd, fw, fh);
                cmd->end_render_pass();
                cmd->image_barrier(*target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                cmd->copy_image_to_buffer(*buffer, *target, 0, {}, {w, h, 1}, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
                cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                             VK_ACCESS_2_HOST_READ_BIT);
                Vulkan::Fence fence;
                dev.submit(cmd, &fence);
                fence->wait();
                std::vector<uint8_t> pixels(binfo.size);
                const void *src = dev.map_host_buffer(*buffer, Vulkan::MEMORY_ACCESS_READ_BIT);
                std::memcpy(pixels.data(), src, pixels.size());
                dev.unmap_host_buffer(*buffer, Vulkan::MEMORY_ACCESS_READ_BIT);
                // Encode the PNG off the render thread: it holds the device lock the GS thread
                // needs, and a full-window encode takes long enough to stall the game (and audio).
                std::thread([pixels = std::move(pixels), w, h, path = std::move(m_capturePath)]() mutable {
                    for (size_t p = 3; p < pixels.size(); p += 4)
                        pixels[p] = 0xFF;
                    Image image = {pixels.data(), static_cast<int>(w), static_cast<int>(h), 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
                    // Written under another name, then renamed: the file appears complete.
                    const std::string part = path + ".part.png";
                    if (ExportImage(image, part.c_str()))
                        std::rename(part.c_str(), path.c_str());
                }).detach();
                m_capturePath.clear();
            }

            PgsPresenterOptions m_options;
            SDL_Window *m_window = nullptr;
            std::unique_ptr<SdlPlatform> m_platform;
            PgsShared m_shared;
            Vulkan::WSI m_wsi; // after m_shared: destroyed first, with the device
            Vulkan::Program *m_program = nullptr;
            Vulkan::Program *m_fxaa = nullptr, *m_easu = nullptr, *m_rcas = nullptr, *m_sgsr1 = nullptr;
            Vulkan::Program *m_sgsr2Convert = nullptr, *m_sgsr2Upscale = nullptr;
            Vulkan::ImageHandle m_sgsr2Mda[2], m_sgsr2MdaShadow, m_sgsr2History[2], m_sgsr2Shadow;
            uint32_t m_sgsr2MdaIndex = 0; // the buffer the next real frame's convert pass writes
            uint32_t m_sgsr2Index = 0;
            bool m_sgsr2Valid = false;
            uint64_t m_sgsr2Serial = ~0ull, m_taaSerial = ~0ull, m_mfxSerial = ~0ull; // PgsShared::pictureSerial each last ran on
#if defined(PS2X_HAVE_ARM_ASR)
            std::vector<uint8_t> m_asrScratch; // the backend's (one per process; outlives the contexts)
            arm::FfxmInterface m_asrIface = {};
            arm::VkDeviceContext m_asrDevice = {};
            std::unique_ptr<ArmAsr> m_asr, m_asrOther;
            std::vector<std::unique_ptr<ArmAsr>> m_asrRetired;
            Vulkan::ImageHandle m_asrOut;
            uint64_t m_asrSerial = ~0ull;
            bool m_asrFailed = false;
#endif
            bool m_asrValid = false;
            PostProcess m_post;
            Vulkan::ImageHandle m_aaImage, m_upImage; // intermediate pictures
            Vulkan::Program *m_depthView = nullptr, *m_motionView = nullptr, *m_taa = nullptr;
            Vulkan::ImageHandle m_taaHistory[2];
            uint32_t m_taaIndex = 0;
            bool m_taaValid = false;
            bool m_temporalOn = false;
            float m_jitter[4] = {}; // this frame's and last frame's camera jitter (GS pixels)
            Vulkan::Program *m_depthNormalize = nullptr;
            Vulkan::ImageHandle m_depthNorm;
            Vulkan::Program *m_uiComposite = nullptr;
            Vulkan::Program *m_rcasUi = nullptr; // RCAS + the UI composite, drawn to the screen
            // spareUi's composite, done by drawGame in its pass to the screen (no images between).
            const Vulkan::Image *m_uiOriginal = nullptr;
            Vulkan::ImageHandle m_rcasImage, m_uiImage;
            bool m_temporalValid = false;
            Vulkan::ImageHandle m_taaShadowImage;
            // Interlace flicker blending.
            Vulkan::Program *m_flicker = nullptr, *m_flickerCut = nullptr, *m_flickerGrid = nullptr, *m_flickerWeight = nullptr;
            Vulkan::ImageHandle m_flickerHist[2], m_flickerImage, m_flickerOut, m_flickerCutImage, m_flickerGridImage,
                m_flickerWeightImage;
            uint32_t m_flickerIndex = 0, m_flickerValid = 0;
            // Frame generation.
            FrameGeneration m_fg;
            Vulkan::Program *m_frameGen = nullptr, *m_copy = nullptr;
            Vulkan::ImageHandle m_genHistory[2], m_genImage;
            uint32_t m_genIndex = 0, m_subframe = 0;
            bool m_genPrevValid = false, m_genRcas = false;
            float m_genJitterDelta[2] = {};
            uint64_t m_renderedTick = ~0ull;
            uint32_t m_presents = 1;
            // Presenting through Metal (macOS).
            bool m_metalPresent = false;
#if defined(__APPLE__)
            SDL_MetalView m_metalView = nullptr;
            std::unique_ptr<MetalPresent> m_metal;
#endif
            Vulkan::ImageHandle m_backbuffers[3];
            uint32_t m_backIndex = 0;
            std::mutex m_presentedMutex;
            std::vector<std::pair<int64_t, int64_t>> m_presented; // (target refresh, on screen), host ns
            bool m_presentAtTime = false;
            int64_t m_presentTarget = 0; // host ns of the refresh the last present was aimed at
            int64_t m_presentTargets[3] = {}; // the last three presents' refreshes (ring)
            int64_t m_presentLead = 0;        // how far ahead of now a present is aimed (adaptive)
            uint32_t m_presentsOnTime = 0, m_presentsSinceMiss = 1000;
            PFN_vkGetPastPresentationTimingGOOGLE m_pastTiming = nullptr;
            PresentDeviceFactory m_deviceFactory; // (the device: a spare queue for the present thread)
            int64_t m_presentOffset = INT64_MAX, m_offsetWindowMin = INT64_MAX; // presented time - target refresh
            uint32_t m_offsetSamples = 0;
            uint32_t m_presentCount = 0;
            const Vulkan::Image *m_lastPicture = nullptr; // the guest frame's picture, for repeats
            bool m_lastRcas = false;
            const Vulkan::Image *m_sourceOverride = nullptr; // a shadow frame shown instead of the scanout
            bool m_noTemporal = false;                       // processing a shadow frame: no TAA/MetalFX history
            float m_shadowT = 0.5f;
            char m_plTag = '?';
            const Vulkan::Image *m_lastShown = nullptr; // frame generation: the picture shown last (real or generated)
            bool m_lastShownRcas = false;
            uint64_t m_lastRealPicture = ~0ull;         // the 3D of the last real frame prepared                          // that shadow's time after the real frame (frames)
            double m_lockWaitMs = 0.0, m_lockHeldMs = 0.0;   // this present's device lock (RT_PRESENT_DEBUG)
            double m_latchMs = 0.0;
            struct
            {
                uint32_t n = 0, phases = 0;
                double latchSum = 0, latchMax = 0, renderSum = 0, renderMax = 0, waitSum = 0, waitMax = 0, heldSum = 0, heldMax = 0;
                double phaseSum = 0, phaseMin = 1e9, phaseMax = -1;
                std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
            } m_pacing;
            float m_mfxShadowAt = -1.0f;                     // MetalFX's shadow history: (picture + t) it last ran on
            bool m_frame2D = false;
            uint64_t m_realSerial = 0; // PgsShared::presentSerial of the real frame on show
            bool m_progressiveFields = true; // GS::progressiveFields, read each frame
            bool m_showDepth = false, m_showMotion = false, m_showUi = false;
            Vulkan::Program *m_smaaEdges = nullptr, *m_smaaWeights = nullptr, *m_smaaBlend = nullptr;
            Vulkan::ImageHandle m_smaaArea, m_smaaSearch, m_smaaEdgeImage, m_smaaWeightImage;
            const Vulkan::Image *m_final = nullptr;   // what the final pass draws this frame
            bool m_finalRcas = false;
#if defined(__APPLE__)
            using GetTexture = void (*)(VkImage, void **);
            using GetQueue = void (*)(VkQueue, void **);
            GetTexture m_getTexture = nullptr;
            GetQueue m_getQueue = nullptr;
            std::unique_ptr<MetalFxSpatial> m_metalfx;
            std::unique_ptr<MetalFxTemporal> m_metalfxTemporal;
            std::unique_ptr<MetalFxTemporal> m_metalfxTemporalShadow; // frame generation's shadow frames
            Vulkan::ImageHandle m_upImageShadow;
            bool m_temporalValidShadow = false;
            void *m_mtlDevice = nullptr;
#endif
            Vulkan::ImageHandle m_cpuFrame; // CPU GS: the uploaded picture
            std::unordered_map<uint64_t, Vulkan::ImageHandle> m_textures; // by ImTextureID
            uint64_t m_nextTexture = 0;
            std::unordered_set<uint64_t> m_nearestTextures; // createUiTexture(..., nearest)
            float m_pictureAspect = 4.0f / 3.0f;
            uint64_t m_lastTick = 0;
            bool m_latched = false;
            bool m_closeRequested = false;
            bool m_background = false; // Android: no surface while another activity is in front
            bool m_ui = false;
            bool m_uiFrame = false;
            std::string m_capturePath;
        };
    }

    PgsShared *pgsShared(HostPresenter *presenter)
    {
        auto *p = dynamic_cast<PgsPresenter *>(presenter);
        return p ? p->shared() : nullptr;
    }

    std::unique_ptr<HostPresenter> createPgsPresenter(const PgsPresenterOptions &options, std::string &error)
    {
        if (!pgsAvailable())
        {
            error = "built without paraLLEl-GS";
            return nullptr;
        }
        return std::make_unique<PgsPresenter>(options);
    }
}

#elif defined(PS2X_HAVE_PGS)

#include "gs_pgs_shared.h"

namespace ps2x::gs
{
    PgsShared *pgsShared(HostPresenter *) { return nullptr; }

    std::unique_ptr<HostPresenter> createPgsPresenter(const PgsPresenterOptions &, std::string &error)
    {
        error = "the Vulkan presenter needs the SDL3 host platform";
        return nullptr;
    }
}

#endif
