#version 450
// Temporal anti-aliasing (Road Trip recomp): the jittered current frame blended with the history,
// which is reprojected with the game's per-pixel motion (gs_motion.h) and clamped to the current
// 3x3 neighbourhood in YCoCg so moved or newly visible things don't ghost.
layout(set = 0, binding = 0) uniform sampler2D uCurrent;
layout(set = 0, binding = 1) uniform sampler2D uHistory;
layout(set = 0, binding = 2) uniform sampler2D uMotion; // GS pixels, current minus previous
layout(push_constant) uniform Push
{
    vec2 motionToUv;   // 1 / frame buffer size (GS pixels to UV)
    vec2 jitterDelta;  // this frame's jitter minus last frame's, GS pixels
    vec2 rcpSize;      // 1 / picture size
    float blend;       // weight of the new frame (e.g. 0.1)
    float historyValid;
} pc;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

vec3 toYCoCg(vec3 c) { return vec3(dot(c, vec3(0.25, 0.5, 0.25)), dot(c, vec3(0.5, 0.0, -0.5)), dot(c, vec3(-0.25, 0.5, -0.25))); }
vec3 fromYCoCg(vec3 c) { return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

void main()
{
    vec3 current = texture(uCurrent, vUV).rgb;
    vec3 lo = toYCoCg(current), hi = lo;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            vec3 c = toYCoCg(texture(uCurrent, vUV + vec2(x, y) * pc.rcpSize).rgb);
            lo = min(lo, c);
            hi = max(hi, c);
        }
    vec2 motion = texture(uMotion, vUV).rg - pc.jitterDelta;
    vec2 prevUV = vUV - motion * pc.motionToUv;
    if (pc.historyValid < 0.5 || any(lessThan(prevUV, vec2(0.0))) || any(greaterThan(prevUV, vec2(1.0))))
    {
        outColor = vec4(current, 1.0);
        return;
    }
    vec3 history = fromYCoCg(clamp(toYCoCg(texture(uHistory, prevUV).rgb), lo, hi));
    outColor = vec4(mix(history, current, pc.blend), 1.0);
}
