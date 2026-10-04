// SMAA 1x (Jimenez et al., MIT, smaa/LICENSE.txt), high preset, in GLSL. Each pass computes its
// vertex-stage offsets in the fragment shader from the fullscreen triangle's UV.
layout(push_constant) uniform Push { vec4 metrics; } pc; // 1/w, 1/h, w, h
#define SMAA_RT_METRICS pc.metrics
#define SMAA_GLSL_4 1
#define SMAA_PRESET_HIGH 1
#define SMAA_INCLUDE_VS 1
#define SMAA_INCLUDE_PS 1
#include "smaa/SMAA.hlsl"
