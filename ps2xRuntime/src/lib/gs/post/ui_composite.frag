#version 450
// Only the 3D is post-processed (Road Trip recomp): where the UI mask says the UI drew (HUD, 2D
// screens), show the picture as the game drew it instead of the processed one.
layout(set = 0, binding = 0) uniform sampler2D uProcessed;
layout(set = 0, binding = 1) uniform sampler2D uOriginal;
layout(set = 0, binding = 2) uniform sampler2D uUiMask;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main()
{
    float ui = texture(uUiMask, vUV).r;
    outColor = vec4(mix(texture(uProcessed, vUV).rgb, texture(uOriginal, vUV).rgb, ui), 1.0);
}
