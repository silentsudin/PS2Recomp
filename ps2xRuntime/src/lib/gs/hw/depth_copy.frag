#version 450
// Hardware GS (gs_hw_backend.cpp): the 3D's depth, where the HUD starts, as raw GS Z (24-bit,
// larger = nearer), the scanned-out part only: what paraLLEl-GS publishes (PgsShared::depth).
layout(set = 0, binding = 0) uniform sampler2D uDepth; // D32F, Z / 2^24
layout(push_constant) uniform Push
{
    vec4 rect; // xy: the part's origin, zw: its size (UV of the depth image)
} pc;
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outZ;
void main()
{
    outZ = vec4(texture(uDepth, pc.rect.xy + vUV * pc.rect.zw).r * 16777216.0, 0.0, 0.0, 1.0);
}
