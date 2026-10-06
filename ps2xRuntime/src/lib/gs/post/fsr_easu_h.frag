#version 450
// AMD FidelityFX Super Resolution 1, EASU in 16-bit arithmetic (FsrEasuH; MIT, ffx/LICENSE.txt): the
// same filter, about twice as fast on mobile GPUs. Used where the device has shaderFloat16.
#extension GL_GOOGLE_include_directive : require
#define A_GPU 1
#define A_GLSL 1
#define A_HALF 1
#include "ffx/ffx_a.h"
layout(set = 0, binding = 0) uniform sampler2D uTex;
layout(push_constant) uniform Push { uvec4 con0, con1, con2, con3; uvec4 origin; } pc;
layout(location = 0) out vec4 outColor;
#define FSR_EASU_H 1
AH4 FsrEasuRH(AF2 p) { return AH4(textureGather(uTex, p, 0)); }
AH4 FsrEasuGH(AF2 p) { return AH4(textureGather(uTex, p, 1)); }
AH4 FsrEasuBH(AF2 p) { return AH4(textureGather(uTex, p, 2)); }
#include "ffx/ffx_fsr1.h"
void main()
{
    AH3 c;
    FsrEasuH(c, AU2(gl_FragCoord.xy) - pc.origin.xy, pc.con0, pc.con1, pc.con2, pc.con3);
    outColor = vec4(c, 1.0);
}
