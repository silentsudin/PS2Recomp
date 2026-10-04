#version 450
// AMD FidelityFX Super Resolution 1, RCAS: robust contrast-adaptive sharpening (MIT, ffx/LICENSE.txt).
#extension GL_GOOGLE_include_directive : require
#define A_GPU 1
#define A_GLSL 1
#include "ffx/ffx_a.h"
layout(set = 0, binding = 0) uniform sampler2D uTex;
layout(push_constant) uniform Push { uvec4 con; uvec4 origin; } pc;
layout(location = 0) out vec4 outColor;
#define FSR_RCAS_F 1
AF4 FsrRcasLoadF(ASU2 p) { return texelFetch(uTex, ASU2(p), 0); }
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}
#include "ffx/ffx_fsr1.h"
void main()
{
    AF3 c;
    FsrRcasF(c.r, c.g, c.b, AU2(gl_FragCoord.xy) - pc.origin.xy, pc.con);
    outColor = vec4(c, 1.0);
}
