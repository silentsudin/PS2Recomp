#version 450
// Interlace flicker blending, part 1 (Road Trip recomp): into a 1x1 target, the share of the
// screen that changed drastically since the last frame (a hard cut, or the first frame of an
// alternating effect). Sampled on a 32x32 grid.
layout(set = 0, binding = 0) uniform sampler2D uCurrent;
layout(set = 0, binding = 1) uniform sampler2D uPrevious;
layout(location = 0) out vec4 outColor;
void main()
{
    float changed = 0.0;
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x)
        {
            vec2 uv = (vec2(x, y) + 0.5) / 32.0;
            vec3 d = abs(texture(uCurrent, uv).rgb - texture(uPrevious, uv).rgb);
            changed += step(0.12, max(d.r, max(d.g, d.b)));
        }
    outColor = vec4(changed / 1024.0, 0.0, 0.0, 1.0);
}
