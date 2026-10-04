#version 450
// A plain copy (Road Trip recomp: keeps a picture for frame generation).
layout(set = 0, binding = 0) uniform sampler2D uSource;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(texture(uSource, vUV).rgb, 1.0); }
