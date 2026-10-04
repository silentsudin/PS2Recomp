#version 450
// AMD FidelityFX Super Resolution 1, EASU: edge-adaptive spatial upscaling (MIT, ffx/LICENSE.txt).
#extension GL_GOOGLE_include_directive : require
#define A_GPU 1
#define A_GLSL 1
#include "ffx/ffx_a.h"
layout(set = 0, binding = 0) uniform sampler2D uTex;
layout(push_constant) uniform Push { uvec4 con0, con1, con2, con3; uvec4 origin; } pc;
layout(location = 0) out vec4 outColor;
#define FSR_EASU_F 1
AF4 FsrEasuRF(AF2 p) { return textureGather(uTex, p, 0); }
AF4 FsrEasuGF(AF2 p) { return textureGather(uTex, p, 1); }
AF4 FsrEasuBF(AF2 p) { return textureGather(uTex, p, 2); }
#include "ffx/ffx_fsr1.h"
void main()
{
    AF3 c;
    FsrEasuF(c, AU2(gl_FragCoord.xy) - pc.origin.xy, pc.con0, pc.con1, pc.con2, pc.con3);
    outColor = vec4(c, 1.0);
}
