#pragma once

// Host audio while the app is away (Android: another activity in front, the screen off): suspend
// closes the audio device, so nothing plays or keeps the device awake; resume reopens it and the
// SPU2's stream. Any thread; nested calls are counted.
void ps2AudioOutSuspend(bool suspend);

// The player's mix (the app's sound settings): a master gain (0 silent .. 1 as the game made it)
// and mono (left and right averaged). Applied as the sound leaves for the device, after the test
// harness's capture. Any thread.
void ps2AudioOutSetMix(float gain, bool mono);
