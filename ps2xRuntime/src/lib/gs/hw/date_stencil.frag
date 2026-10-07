#version 450
// Hardware GS (gs_hw_backend.cpp): destination alpha test through the stencil buffer. Before a DATE
// batch, stencil bit 0 is set where the frame buffer's alpha MSB is 1 (alpha is kept as A/128),
// from the snapshot of the target; the batch then tests and updates the stencil per primitive, as
// the GS reads the alpha its own earlier primitives wrote (shadows darken once where they overlap).
layout(set = 0, binding = 0) uniform sampler2D uDest;
layout(location = 0) in vec2 vUV;
void main()
{
    if (texelFetch(uDest, ivec2(gl_FragCoord.xy), 0).a <= 0.999)
        discard;
}
