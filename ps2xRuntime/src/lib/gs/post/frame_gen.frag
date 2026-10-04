#version 450
// Frame generation (Road Trip recomp): a frame between guest frames, made from the game's
// per-pixel motion (gs_motion.h, current minus previous position) and depth.
//  - Interpolate (mode 0): the frame at time t between the previous picture (t = 0) and the
//    current one (t = 1). The surface that is at x at time t is found in the current picture by
//    iterating s = x + (1 - t) M(s) (backward warping along its own motion); the same surface in the
//    previous picture is at s - M(s); the two are blended by t.
//  - Extrapolate (mode 1): the current picture pushed on along its motion by t of a frame:
//    s = x - t M(s).
// Where no surface of the current picture lands on x (background that a moving object uncovers),
// the motion of the farthest surface nearby is used (smallest Z: the game's depth test is GEQUAL),
// and interpolation takes the previous picture there.
layout(set = 0, binding = 0) uniform sampler2D uPrevious;
layout(set = 0, binding = 1) uniform sampler2D uCurrent;
layout(set = 0, binding = 2) uniform sampler2D uMotion; // GS pixels, current minus previous
layout(set = 0, binding = 3) uniform sampler2D uDepth;  // raw GS Z
layout(push_constant) uniform Push
{
    vec2 motionToUv;    // 1 / frame buffer size (GS pixels to UV)
    vec2 jitterDelta;   // the current frame's camera jitter minus the previous one's, GS pixels
    vec2 rcpMotionSize; // 1 / motion texture size
    float t;            // 0..1 of a guest frame
    float mode;         // 0 interpolate, 1 extrapolate
} pc;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

vec2 motionAt(vec2 uv) { return (textureLod(uMotion, uv, 0.0).rg - pc.jitterDelta) * pc.motionToUv; }

// The motion of the farthest surface in a 5x5 neighbourhood (sampled 2 texels apart).
vec2 backgroundMotion(vec2 uv)
{
    float best = 3.4e38;
    vec2 m = vec2(0.0);
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x)
        {
            vec2 p = uv + vec2(x, y) * 2.0 * pc.rcpMotionSize;
            float z = textureLod(uDepth, p, 0.0).r;
            if (z < best)
            {
                best = z;
                m = motionAt(p);
            }
        }
    return m;
}

bool inside(vec2 uv) { return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0))); }

void main()
{
    // How far (and which way) the current picture's surfaces travel to reach time t.
    float k = pc.mode > 0.5 ? -pc.t : 1.0 - pc.t;
    vec2 s = vUV + k * motionAt(vUV);
    for (int i = 0; i < 3; ++i)
        s = vUV + k * motionAt(s);
    vec2 m = motionAt(s);
    // Did it converge onto x (within a texel and a half)?
    bool found = length((s - k * m - vUV) / pc.rcpMotionSize) < 1.5 && inside(s);
    if (pc.mode > 0.5)
    {
        if (!found)
            s = vUV - pc.t * backgroundMotion(vUV);
        outColor = vec4(textureLod(uCurrent, inside(s) ? s : vUV, 0.0).rgb, 1.0);
        return;
    }
    if (found)
    {
        vec2 p = s - m;
        vec3 c = textureLod(uCurrent, s, 0.0).rgb;
        vec3 prev = textureLod(uPrevious, inside(p) ? p : s, 0.0).rgb;
        outColor = vec4(mix(prev, c, pc.t), 1.0);
        return;
    }
    // Uncovered background: only the previous picture saw it.
    vec2 mb = backgroundMotion(vUV);
    vec2 p = vUV - pc.t * mb;
    outColor = vec4(textureLod(uPrevious, inside(p) ? p : vUV, 0.0).rgb, 1.0);
}
