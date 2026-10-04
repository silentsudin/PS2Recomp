#version 450
// Debug view of the motion scanout (RT_SHOW_MOTION=1): x right = red, y down = green, still = grey.
layout(set = 0, binding = 0) uniform sampler2D uMotion;
layout(push_constant) uniform Push { float scale; } pc;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main()
{
    vec2 m = texture(uMotion, vUV).rg * pc.scale;
    outColor = vec4(clamp(0.5 + m, 0.0, 1.0), 0.5, 1.0);
}
