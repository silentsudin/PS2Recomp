#version 450
// Interlace flicker blending, part 1b (Road Trip recomp): into a 1x1 target, the share of the
// screen that changed drastically since the last frame (a hard cut, or the first frame of an
// alternating effect): the 32x32 grid of flicker_grid.frag, summed.
layout(set = 0, binding = 0) uniform sampler2D uGrid;
layout(location = 0) out vec4 outColor;
void main()
{
    float changed = 0.0;
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x)
            changed += texelFetch(uGrid, ivec2(x, y), 0).r;
    outColor = vec4(changed / 1024.0, 0.0, 0.0, 1.0);
}
