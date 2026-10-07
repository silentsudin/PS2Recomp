#version 450
// Hardware GS (gs_hw_backend.cpp): texture function, fog and alpha test of one GS draw state.
// Colours are worked in GS integer units (0..255, alpha 0x80 = 1.0). The render target keeps
// alpha as A/128 (so blend factors As and Ad are exact); textures read from a render target
// therefore scale alpha by 128, decoded textures by 255.

layout(set = 0, binding = 0) uniform sampler2D uTex;
// The render target as it was before this draw (destination alpha test).
layout(set = 0, binding = 1) uniform sampler2D uDest;
// Texture packs: another palette of a replaced texture, as an affine map of the pack image's
// colour (rows r, g, b, a, then the offset; 0..255, PS2 alpha).
layout(set = 0, binding = 2) uniform Recolor
{
    vec4 rows[5];
} uRecolor;

layout(location = 0) in vec4 vColor;
layout(location = 1) noperspective in vec3 vStq;
layout(location = 2) noperspective in float vFog;
layout(location = 3) noperspective in vec2 vMotion;

// Per pipeline (one per combination the game uses): the feature flags below, the alpha test, and
// which pass of the alpha test this is. Branches on them compile away.
layout(constant_id = 0) const uint FLAGS = 0u;
layout(constant_id = 1) const uint ATST = 1u;
layout(constant_id = 2) const uint AMODE = 2u; // 0 = alpha test, 1 = keep only failing pixels, 2 = none
// DATE through the stencil buffer (gs_hw_backend.cpp): 1 = only the pixels whose written alpha MSB
// differs from DATM (they flip the stencil), 2 = only those whose MSB equals it, 0 = all.
layout(constant_id = 3) const uint DATE_SPLIT = 0u;

layout(push_constant) uniform Push
{
    layout(offset = 16) vec4 texNorm; // xy: texture size in texels (TW, TH); zw: 1 / the image's extent in texels
    vec4 region;   // MINU, MAXU, MINV, MAXV (texels)
    vec4 fogColor; // rgb 0..255; w: texture alpha scale (255 decoded, 128 render target)
    vec4 lod;      // MXL, L, K, LCM
    vec4 texa;     // TA0, AEM, TA1, 1 = alpha from TEXA (a CT24 render target)
    uvec4 mode;    // z: AREF
    vec4 texOffset; // xy: where texel (0, 0) is in the image (a texture inside a render target);
                    // z: 1 = 2D above 1x, attributes at the GS pixel's sample point; w: the snap
                    // scale per axis (x | y << 4, 1 = none)
} push;

layout(location = 0) out vec4 outColor;
// Per-pixel motion for TAA and temporal upscalers (a second attachment while they are on).
layout(location = 1) out vec4 outMotion;

const uint F_TME = 1u, F_TCC = 8u, F_FGE = 16u, F_MIP = 1024u, F_LINEAR = 2048u, F_REPEAT_IN_SHADER = 4096u, F_DATE = 8192u,
           F_DATM = 16384u, F_REPLACED = 32768u, F_RECOLOR = 65536u;

float wrapCoord(float c, uint mode, float size, float lo, float hi, bool linear)
{
    if (mode == 0u) // REPEAT
        return ((FLAGS & F_REPEAT_IN_SHADER) != 0u) ? mod(c, size) : c;
    if (mode == 1u) // CLAMP
        return linear ? clamp(c, 0.5, size - 0.5) : clamp(c, 0.0, size - 0.001);
    if (mode == 2u) // REGION_CLAMP
        return linear ? clamp(c, lo + 0.5, hi + 0.5) : clamp(c, lo, hi + 0.999);
    // REGION_REPEAT (integer masks, point sampled)
    return float((uint(max(c, 0.0)) & uint(lo)) | uint(hi)) + 0.5;
}

bool alphaPasses(float a)
{
    float aref = float(push.mode.z);
    switch (ATST)
    {
    case 0u: return false;
    case 1u: return true;
    case 2u: return a < aref;
    case 3u: return a <= aref;
    case 4u: return a == aref;
    case 5u: return a >= aref;
    case 6u: return a > aref;
    default: return a != aref;
    }
}

void main()
{
    // Above 1x: attributes where paraLLEl-GS samples them when it supersamples (gs_hw_backend.cpp,
    // Push::texOffset). Varyings are affine in screen space, so the derivatives move them exactly.
    // Before any discard (derivatives need the whole quad).
    vec4 color = vColor;
    vec3 stq = vStq;
    float fogIn = vFog;
    if (push.texOffset.z != 0.0)
    {
        // The snap scale per axis (1 = none): the GS pixel's sample point is the centre of its
        // first device pixel.
        const uint packed = uint(push.texOffset.w);
        const vec2 scale = vec2(float(packed & 15u), float((packed >> 4u) & 15u));
        const vec2 dev = floor(gl_FragCoord.xy);
        const vec2 off = floor(dev / scale) * scale - dev;
        stq += dFdxFine(vStq) * off.x + dFdyFine(vStq) * off.y;
        color += dFdxFine(vColor) * off.x + dFdyFine(vColor) * off.y;
        fogIn += dFdxFine(vFog) * off.x + dFdyFine(vFog) * off.y;
    }

    // DATE: only where the frame buffer's alpha MSB equals DATM (alpha is kept as A/128).
    if ((FLAGS & F_DATE) != 0u)
    {
        bool msb = texelFetch(uDest, ivec2(gl_FragCoord.xy), 0).a > 0.999;
        if (msb != ((FLAGS & F_DATM) != 0u))
            discard;
    }

    vec4 f = floor(clamp(color, 0.0, 1.0) * 255.0 + 0.5);
    vec3 rgb = f.rgb;
    float a = f.a;
    const uint flags = FLAGS;

    if ((flags & F_TME) != 0u)
    {
        vec2 st = stq.xy / stq.z;
        vec2 texel = st * push.texNorm.xy;
        bool linear = (flags & F_LINEAR) != 0u;
        vec4 t;
        if ((flags & F_REPLACED) != 0u)
        {
            // A texture-pack image for the texture's rect (texOffset = -its origin, texNorm.zw =
            // 1 / its size): filtered by the hardware (mips, anisotropy).
            // Clamped axes stay inside the texels the GS could reach ([0, TW) or [MINU, MAXU + 1)),
            // inset so the filter footprint (half a pack texel when magnified, half the pixel's
            // footprint when minified) never reaches past them: the image is often a whole atlas
            // upload, and the GS point-sampled the region where we filter (a line of the
            // neighbouring sprite showed beside HUD parts).
            uint ws = (flags >> 5) & 3u, wt = (flags >> 7) & 3u;
            // The gradients are the unclamped coordinates' (a clamped edge keeps its mip level).
            vec2 dx = dFdx(texel), dy = dFdy(texel);
            vec2 half_ = max(0.5 / (vec2(textureSize(uTex, 0)) * push.texNorm.zw), 0.5 * max(abs(dx), abs(dy)));
            if (ws == 1u || ws == 2u)
            {
                float lo = ws == 1u ? 0.0 : push.region.x, hi = ws == 1u ? push.texNorm.x : push.region.y + 1.0;
                float h = min(half_.x, 0.5 * (hi - lo));
                texel.x = clamp(texel.x, lo + h, hi - h);
            }
            if (wt == 1u || wt == 2u)
            {
                float lo = wt == 1u ? 0.0 : push.region.z, hi = wt == 1u ? push.texNorm.y : push.region.w + 1.0;
                float h = min(half_.y, 0.5 * (hi - lo));
                texel.y = clamp(texel.y, lo + h, hi - h);
            }
            vec4 c = textureGrad(uTex, (texel + push.texOffset.xy) * push.texNorm.zw, dx * push.texNorm.zw,
                                 dy * push.texNorm.zw) * 255.0;
            if ((flags & F_RECOLOR) != 0u)
                c = clamp(vec4(dot(uRecolor.rows[0], c), dot(uRecolor.rows[1], c), dot(uRecolor.rows[2], c),
                               dot(uRecolor.rows[3], c)) + uRecolor.rows[4], vec4(0.0), vec4(255.0));
            t = roundEven(c) / vec4(255.0, 255.0, 255.0, push.fogColor.w);
        }
        else
        {
            texel.x = wrapCoord(texel.x, (flags >> 5) & 3u, push.texNorm.x, push.region.x, push.region.y, linear);
            texel.y = wrapCoord(texel.y, (flags >> 7) & 3u, push.texNorm.y, push.region.z, push.region.w, linear);
            float lod = 0.0;
            if ((flags & F_MIP) != 0u)
            {
                lod = push.lod.w != 0.0 ? push.lod.z : log2(1.0 / abs(stq.z)) * exp2(push.lod.y) + push.lod.z;
                lod = clamp(lod, 0.0, push.lod.x);
            }
            t = textureLod(uTex, (texel + push.texOffset.xy) * push.texNorm.zw, lod);
        }
        vec3 ct = floor(t.rgb * 255.0 + 0.5);
        float at = floor(t.a * push.fogColor.w + 0.5);
        if (push.texa.w != 0.0)
            at = (push.texa.y != 0.0 && all(equal(ct, vec3(0.0)))) ? 0.0 : push.texa.x;

        uint tfx = (flags >> 1) & 3u;
        bool tcc = (flags & F_TCC) != 0u;
        if (tfx == 0u) // MODULATE
        {
            rgb = floor(ct * f.rgb / 128.0);
            a = tcc ? floor(at * f.a / 128.0) : f.a;
        }
        else if (tfx == 1u) // DECAL
        {
            rgb = ct;
            a = tcc ? at : f.a;
        }
        else // HIGHLIGHT / HIGHLIGHT2
        {
            rgb = floor(ct * f.rgb / 128.0) + f.a;
            a = tcc ? (tfx == 2u ? at + f.a : at) : f.a;
        }
        rgb = min(rgb, vec3(255.0));
        a = min(a, 255.0);
    }

    if ((flags & F_FGE) != 0u)
    {
        float fog = floor(clamp(fogIn, 0.0, 255.0) + 0.5);
        rgb = floor((rgb * fog + push.fogColor.rgb * (255.0 - fog)) / 256.0);
    }

    if (AMODE != 2u)
    {
        bool pass = alphaPasses(a);
        if (pass == (AMODE == 1u))
            discard;
    }

    if (DATE_SPLIT != 0u)
    {
        const bool flips = (a >= 128.0) != ((flags & F_DATM) != 0u);
        if (flips != (DATE_SPLIT == 1u))
            discard;
    }

    outColor = vec4(rgb / 255.0, clamp(a / 128.0, 0.0, 1.0));
    // Alpha as the colour's: a blended draw (As, 1 - As) writes its motion where it is opaque.
    outMotion = vec4(vMotion, 0.0, outColor.a);
}
