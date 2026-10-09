#pragma once

/* Upscaler plugins (Windows): FSR 3, DLSS and XeSS live in DLLs next to RoadTrip.exe, behind this C
 * interface, so the vendors' SDKs (and their licences and toolchains) stay out of the runtime. A
 * plugin exports one function, rtu_get_api(), and is loaded with LoadLibrary when it is present;
 * without it the options are simply not offered.
 *
 * The presenter hands over Vulkan handles as plain integers/pointers (this header includes no Vulkan).
 * Layout contract: every image is in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL when rtu dispatch is
 * called and must be left in it; a plugin makes its own barriers for the layouts it needs.
 * Picture conventions (the same the Arm ASR path uses):
 *   color   RGBA8 UNORM, LDR, at render size, already anti-aliased if the player chose that
 *   depth   R32 SFLOAT at render size, 0..1, larger = nearer (reverse Z), infinite far plane
 *   motion  RG16 SFLOAT at render size; raw values times mv_scale give the motion from the current
 *           pixel to where it was in the previous picture, in render-size pixels
 *   jitter  the camera's sub-pixel offset this picture, in render-size pixels
 *   output  RGBA8 UNORM at display size, with storage usage
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define RTU_API_VERSION 1

    /* Bits of RtuApi::supported() and of the `kind` arguments. */
    enum
    {
        RTU_KIND_FSR3 = 1u << 0,
        RTU_KIND_DLSS = 1u << 1,
        RTU_KIND_XESS = 1u << 2,
    };

    typedef struct RtuImage
    {
        uint64_t image;  /* VkImage */
        uint64_t view;   /* VkImageView over the whole image */
        uint32_t format; /* VkFormat */
        uint32_t width;
        uint32_t height;
    } RtuImage;

    typedef struct RtuInit
    {
        void *instance;                /* VkInstance */
        void *physical_device;         /* VkPhysicalDevice */
        void *device;                  /* VkDevice */
        void *get_instance_proc_addr;  /* PFN_vkGetInstanceProcAddr */
        void *get_device_proc_addr;    /* PFN_vkGetDeviceProcAddr */
        void *queue;                   /* a VkQueue of the graphics family (the one the commands run on) */
        uint32_t queue_family;
        uint32_t vendor_id;            /* VkPhysicalDeviceProperties */
        uint32_t device_id;
        uint32_t driver_version;
        uint32_t api_version;
        const char *app_data_dir;      /* writable, UTF-8: logs and caches */
        const char *plugin_dir;        /* where the plugin DLL (and the vendor's DLLs) are, UTF-8 */
    } RtuInit;

    typedef struct RtuCreateInfo
    {
        uint32_t kind; /* one RTU_KIND_* */
        uint32_t render_width, render_height;
        uint32_t display_width, display_height;
        uint32_t quality_hint; /* 0 = the library's quality mode for this ratio, else a vendor-neutral
                                  preference (unused so far) */
    } RtuCreateInfo;

    typedef struct RtuDispatch
    {
        void *command_buffer; /* VkCommandBuffer, recording, outside a render pass */
        RtuImage color, depth, motion, output;
        float jitter_x, jitter_y;
        float mv_scale_x, mv_scale_y;
        uint32_t render_width, render_height;
        float sharpness;      /* 0..1; applied by the library where it has its own sharpening */
        float frame_time_ms;
        float camera_near, camera_far, camera_fov_vertical; /* radians */
        uint32_t reset;       /* the history is not valid: start over */
    } RtuDispatch;

    typedef struct RtuApi
    {
        uint32_t version;

        /* Extension names the kinds this plugin offers may need. Called before the instance/device
         * exist (instance) and with them for the device; names are owned by the plugin until its next
         * call. The host drops the ones the driver doesn't have. Return 0 on success. */
        int (*instance_extensions)(const char *const **names, uint32_t *count);
        int (*device_extensions)(void *instance, void *physical_device, void *get_instance_proc_addr,
                                 const char *const **names, uint32_t *count);

        /* Called once the device exists. Returns the RTU_KIND_ bits that work on this machine. */
        uint32_t (*init)(const RtuInit *init);
        /* Largest display/render ratio per axis the kind accepts (e.g. 3.0). */
        float (*max_scale)(uint32_t kind);

        /* A context for one size pair; NULL on failure. The host keeps a few and destroys dropped ones
         * some frames later (the GPU may still be using them), never waiting for the GPU itself. */
        void *(*create)(const RtuCreateInfo *info);
        /* 0 on success. */
        int (*dispatch)(void *context, const RtuDispatch *dispatch);
        void (*destroy)(void *context);

        void (*shutdown)(void);
        /* A short human-readable line for the log, e.g. the library's version; may be NULL. */
        const char *(*describe)(uint32_t kind);
    } RtuApi;

    /* The one export. Returns NULL if `version` isn't RTU_API_VERSION. A plugin defines RTU_EXPORT as
     * its export attribute (__declspec(dllexport)) before including this header; the host leaves it
     * empty and finds the function with GetProcAddress. */
#ifndef RTU_EXPORT
#define RTU_EXPORT
#endif
    typedef const RtuApi *(*RtuGetApiFn)(uint32_t version);
    RTU_EXPORT const RtuApi *rtu_get_api(uint32_t version);

#ifdef __cplusplus
}
#endif
