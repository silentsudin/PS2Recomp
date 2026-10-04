#version 450
// Debug view of the depth scanout (RT_SHOW_DEPTH=1): raw GS Z on a log scale (red: log2/32, green: fractional bits).
layout(set = 0, binding = 0) uniform sampler2D uDepth;
layout(push_constant) uniform Push { float rcpMax; } pc;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main()
{
    // log2 of the raw value over 0..32 bits: works for 16, 24 and 32-bit Z alike.
    float z = texture(uDepth, vUV).r;
    float l = log2(z + 1.0) / 32.0;
    outColor = vec4(l, fract(l * 32.0), pc.rcpMax > 0.0 ? 0.0 : 1.0, 1.0);
}
