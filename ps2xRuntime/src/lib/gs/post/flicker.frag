#version 450
// Interlace flicker blending (Road Trip recomp, progressive fields). Some effects alternate two
// different pictures every frame (the loading transition draws the blurred menu and black on
// alternate frames) and rely on an interlaced TV, which shows one per field, to blend them. Shown
// progressively they strobe. Where a pixel alternates (it matches the frame before last but not the
// last one), the last two frames are averaged as the TV would; moving or still content is left as
// it is. A hard cut (most of the screen changed drastically, flicker_cut.frag) is shown half and
// half for one frame, as the two fields of the cut would be: that also covers the first frame of
// an alternating effect, before the pattern can be seen.
layout(set = 0, binding = 0) uniform sampler2D uCurrent;
layout(set = 0, binding = 1) uniform sampler2D uPrevious;
layout(set = 0, binding = 2) uniform sampler2D uBeforeLast;
layout(set = 0, binding = 3) uniform sampler2D uCut; // 1x1: share of the screen that changed drastically
layout(push_constant) uniform Push
{
    vec2 rcpSize;
} pc;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

float dist(vec3 a, vec3 b)
{
    vec3 d = abs(a - b);
    return max(d.r, max(d.g, d.b));
}

void main()
{
    vec3 c = texture(uCurrent, vUV).rgb;
    vec3 p = texture(uPrevious, vUV).rgb;
    float dPrev = 0.0, dBefore = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            vec2 uv = vUV + vec2(x, y) * pc.rcpSize;
            vec3 ci = texture(uCurrent, uv).rgb;
            dPrev += dist(ci, texture(uPrevious, uv).rgb);
            dBefore += dist(ci, texture(uBeforeLast, uv).rgb);
        }
    dPrev /= 9.0;
    dBefore /= 9.0;
    float w = clamp((dPrev - 3.0 * dBefore - 0.03) * 20.0, 0.0, 1.0);
    w = max(w, clamp((texture(uCut, vec2(0.5)).r - 0.5) * 10.0, 0.0, 1.0));
    outColor = vec4(mix(c, 0.5 * (c + p), w), 1.0);
}
