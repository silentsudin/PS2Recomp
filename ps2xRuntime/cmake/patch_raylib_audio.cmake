# raylib's CloseAudioDevice destroys the audio mutex before stopping the device, so a device
# callback still running (Android's AAudio thread) locks a destroyed mutex and aborts. Stop the
# device first. Run from the raylib source directory (FetchContent PATCH_COMMAND); idempotent.
set(_f "src/raudio.c")
file(READ "${_f}" _c)
set(_bad "ma_mutex_uninit(&AUDIO.System.lock);\n        ma_device_uninit(&AUDIO.System.device);")
set(_good "ma_device_uninit(&AUDIO.System.device);\n        ma_mutex_uninit(&AUDIO.System.lock);")
string(FIND "${_c}" "${_bad}" _at)
if(NOT _at EQUAL -1)
    string(REPLACE "${_bad}" "${_good}" _c "${_c}")
    file(WRITE "${_f}" "${_c}")
    message(STATUS "raylib: CloseAudioDevice stops the device before destroying its mutex")
endif()
