#version 450
#extension GL_GOOGLE_include_directive : require
layout(set = 0, binding = 0) uniform sampler2D uColor;
layout(set = 0, binding = 1) uniform sampler2D uWeights;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
#include "smaa_common.glsl"
void main()
{
    vec4 offset;
    SMAANeighborhoodBlendingVS(vUV, offset);
    outColor = SMAANeighborhoodBlendingPS(vUV, offset, uColor, uWeights);
}
