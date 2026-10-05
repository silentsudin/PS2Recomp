#version 450
// Interlace flicker blending, part 1a (Road Trip recomp): on a 32x32 grid, where the screen changed
// drastically since the last frame (flicker_cut.frag counts them).
layout(set = 0, binding = 0) uniform sampler2D uCurrent;
layout(set = 0, binding = 1) uniform sampler2D uPrevious;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main()
{
    vec3 d = abs(texture(uCurrent, vUV).rgb - texture(uPrevious, vUV).rgb);
    outColor = vec4(step(0.12, max(d.r, max(d.g, d.b))), 0.0, 0.0, 1.0);
}
