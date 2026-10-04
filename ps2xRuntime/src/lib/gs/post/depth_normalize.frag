#version 450
// Raw GS Z (24-bit, larger = nearer) to 0..1 for MetalFX temporal (depthReversed).
layout(set = 0, binding = 0) uniform sampler2D uDepth;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outDepth;
void main()
{
    outDepth = vec4(texture(uDepth, vUV).r * (1.0 / 16777215.0), 0.0, 0.0, 1.0);
}
