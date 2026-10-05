#version 450
// Interlace flicker blending, part 3 (Road Trip recomp, progressive fields). Some effects alternate two
// different pictures every frame (the loading transition draws the blurred menu and black on
// alternate frames) and rely on an interlaced TV, which shows one per field, to blend them. Shown
// progressively they strobe. Where a pixel alternates, the last two frames are averaged as the TV
// would, by the weight flicker_weight.frag found (at the game's resolution, so this pass and that
// one stay cheap on supersampled pictures).
layout(set = 0, binding = 0) uniform sampler2D uCurrent;
layout(set = 0, binding = 1) uniform sampler2D uPrevious;
layout(set = 0, binding = 2) uniform sampler2D uWeight;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec3 c = texture(uCurrent, vUV).rgb;
    vec3 p = texture(uPrevious, vUV).rgb;
    float w = texture(uWeight, vUV).r;
    outColor = vec4(mix(c, 0.5 * (c + p), w), 1.0);
}
