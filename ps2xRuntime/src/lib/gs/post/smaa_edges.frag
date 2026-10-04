#version 450
#extension GL_GOOGLE_include_directive : require
layout(set = 0, binding = 0) uniform sampler2D uColor;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outEdges;
#include "smaa_common.glsl"
void main()
{
    vec4 offset[3];
    SMAAEdgeDetectionVS(vUV, offset);
    outEdges = vec4(SMAALumaEdgeDetectionPS(vUV, offset, uColor), 0.0, 0.0);
}
