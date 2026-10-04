#version 450
#extension GL_GOOGLE_include_directive : require
layout(set = 0, binding = 0) uniform sampler2D uEdges;
layout(set = 0, binding = 1) uniform sampler2D uArea;
layout(set = 0, binding = 2) uniform sampler2D uSearch;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outWeights;
#include "smaa_common.glsl"
void main()
{
    vec2 pixcoord;
    vec4 offset[3];
    SMAABlendingWeightCalculationVS(vUV, pixcoord, offset);
    outWeights = SMAABlendingWeightCalculationPS(vUV, pixcoord, offset, uEdges, uArea, uSearch, vec4(0.0));
}
