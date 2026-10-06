#version 450
// Hardware GS (gs_hw_backend.cpp): GS primitives as Vulkan triangles. Positions arrive in GS
// window pixels (XYOFFSET already subtracted), Z already scaled to 0..1.
layout(location = 0) in vec2 inPos;
layout(location = 1) in float inZ;
layout(location = 2) in vec4 inColor; // RGBA8 unorm: the vertex RGBAQ colour (0x80 = 1.0 in GS terms)
layout(location = 3) in vec3 inStq;   // s, t, q (FST: u/w, v/h, 1)
layout(location = 4) in float inFog;  // 0..255
layout(location = 5) in vec2 inMotion; // screen motion, GS pixels (0 for the HUD and 2D)

layout(push_constant) uniform Push
{
    vec4 posScale; // xy: GS pixels -> NDC, zw: offset
} push;

layout(location = 0) out vec4 vColor;
// The GS interpolates S, T and Q linearly in screen space and divides per pixel.
layout(location = 1) noperspective out vec3 vStq;
layout(location = 2) noperspective out float vFog;
layout(location = 3) noperspective out vec2 vMotion;

void main()
{
    vColor = inColor;
    vStq = inStq;
    vFog = inFog;
    vMotion = inMotion;
    gl_Position = vec4(inPos * push.posScale.xy + push.posScale.zw, inZ, 1.0);
}
