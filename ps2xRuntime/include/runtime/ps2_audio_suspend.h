#pragma once

// Host audio while the app is away (Android: another activity in front, the screen off): suspend
// closes the audio device, so nothing plays or keeps the device awake; resume reopens it and the
// SPU2's stream. Any thread; nested calls are counted.
void ps2AudioOutSuspend(bool suspend);
