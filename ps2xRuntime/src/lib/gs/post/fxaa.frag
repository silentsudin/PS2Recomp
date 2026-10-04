#version 450
// Fast approximate anti-aliasing: blends along the local luma edge direction (after T. Lottes'
// FXAA). Runs at the render resolution, before scaling.
layout(set = 0, binding = 0) uniform sampler2D uTex;
layout(push_constant) uniform Push { vec2 rcpSize; } pc;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main()
{
    const float reduceMin = 1.0 / 128.0, reduceMul = 1.0 / 8.0, spanMax = 8.0;
    vec3 nw = texture(uTex, vUV + vec2(-1.0, -1.0) * pc.rcpSize).rgb;
    vec3 ne = texture(uTex, vUV + vec2(1.0, -1.0) * pc.rcpSize).rgb;
    vec3 sw = texture(uTex, vUV + vec2(-1.0, 1.0) * pc.rcpSize).rgb;
    vec3 se = texture(uTex, vUV + vec2(1.0, 1.0) * pc.rcpSize).rgb;
    vec3 m = texture(uTex, vUV).rgb;
    float lNW = luma(nw), lNE = luma(ne), lSW = luma(sw), lSE = luma(se), lM = luma(m);
    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

    vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), (lNW + lSW) - (lNE + lSE));
    float reduce = max((lNW + lNE + lSW + lSE) * 0.25 * reduceMul, reduceMin);
    float rcpMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcpMin, vec2(-spanMax), vec2(spanMax)) * pc.rcpSize;

    vec3 a = 0.5 * (texture(uTex, vUV + dir * (1.0 / 3.0 - 0.5)).rgb + texture(uTex, vUV + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 b = a * 0.5 + 0.25 * (texture(uTex, vUV + dir * -0.5).rgb + texture(uTex, vUV + dir * 0.5).rgb);
    float lB = luma(b);
    outColor = vec4((lB < lMin || lB > lMax) ? a : b, 1.0);
}
