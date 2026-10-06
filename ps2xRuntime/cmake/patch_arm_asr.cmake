# Arm ASR's Vulkan backend loads Vulkan through volk only on Linux, Windows and Android (an #error
# elsewhere) and includes it as <Volk/volk.h>. We build it against Granite's volk (the device's
# own loader) on every platform. Run from the Arm ASR source directory (FetchContent
# PATCH_COMMAND); idempotent.
set(_f "include/host/backends/vk/ffxm_vk.h")
file(READ "${_f}" _c)
string(FIND "${_c}" "#ifdef FFXM_VKLOADER_VOLK" _a)
string(FIND "${_c}" "#else // FFXM_VKLOADER_VOLK" _b)
if(NOT _a EQUAL -1 AND NOT _b EQUAL -1)
    string(SUBSTRING "${_c}" 0 ${_a} _head)
    string(SUBSTRING "${_c}" ${_b} -1 _tail)
    file(WRITE "${_f}" "${_head}#ifdef FFXM_VKLOADER_VOLK\n#include <volk.h> // Granite's (patched by PS2Recomp)\n${_tail}")
    message(STATUS "Arm ASR: Vulkan through Granite's volk")
endif()
