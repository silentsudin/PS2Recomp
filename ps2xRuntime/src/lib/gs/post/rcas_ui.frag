#version 450
// FSR 1 RCAS (MIT, ffx/LICENSE.txt) and the UI mask in the one pass to the screen (Road Trip
// recomp): sharpens the processed picture at this pixel, then shows the picture as the game drew
// it where the UI mask says the UI drew. The same result as RCAS into an image, the UI composite
// into another and a copy, without two full-screen writes and reads per present.
#extension GL_GOOGLE_include_directive : require
#define A_GPU 1
#define A_GLSL 1
#include "ffx/ffx_a.h"
layout(set = 0, binding = 0) uniform sampler2D uTex;      // processed, the picture's size on screen
layout(set = 0, binding = 1) uniform sampler2D uOriginal;
layout(set = 0, binding = 2) uniform sampler2D uUiMask;
layout(push_constant) uniform Push { uvec4 con; uvec4 origin; vec4 extent; } pc;
layout(location = 0) out vec4 outColor;
#define FSR_RCAS_F 1
AF4 FsrRcasLoadF(ASU2 p) { return texelFetch(uTex, ASU2(p), 0); }
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}
#include "ffx/ffx_fsr1.h"
void main()
{
    AF3 c;
    FsrRcasF(c.r, c.g, c.b, AU2(gl_FragCoord.xy) - pc.origin.xy, pc.con);
    const vec2 uv = (gl_FragCoord.xy - vec2(pc.origin.xy)) / pc.extent.xy;
    const float ui = texture(uUiMask, uv).r;
    outColor = vec4(mix(c, texture(uOriginal, uv).rgb, ui), 1.0);
}
