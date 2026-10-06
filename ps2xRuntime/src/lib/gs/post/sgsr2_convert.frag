#version 450
// Snapdragon Game Super Resolution 2, 2-pass fragment version: the convert pass
// (sgsr2_convert.fs, snapdragon-gsr sgsr/v2/include/glsl_2_pass_fs, BSD-3-Clause, sgsr/LICENSE). The algorithm
// is Qualcomm's; adapted to the presenter: a push constant for the parameters, the full-screen
// triangle's UV, the GS's motion (GS pixels, current minus previous) in place of encoded velocity,
// and its depth (raw GS Z, larger = nearer: the README's reverse-Z changes).
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
precision highp float;
precision highp int;

layout(set = 0, binding = 0) uniform highp sampler2D InputDepth;    // R32F raw GS Z
layout(set = 0, binding = 1) uniform highp sampler2D InputVelocity; // RG16F, GS pixels
layout(push_constant) uniform Push
{
    vec2 renderSize;
    vec2 outputSize;
    vec2 renderSizeRcp;
    vec2 outputSizeRcp;
    vec2 jitterOffset;   // render pixels, [-0.5, 0.5]
    vec2 scaleRatio;     // output / render width; the static box scale
    vec2 motionToNdc;    // GS pixels of motion to NDC (2 / the frame's size in GS pixels)
    float cameraFovAngleHor; // tan(horizontal FOV / 2)
    float minLerpContribution;
    float reset;
    float pad;
    vec2 jitterDelta;    // GS pixels: this frame's camera jitter minus last frame's (in the motion)
} params;
layout(location = 0) in highp vec2 vUV;
layout(location = 0) out vec4 MotionDepthClipAlphaBuffer;

void main()
{
    uvec2 InputPos = uvec2(vUV * params.renderSize);
    vec2 gatherCoord = vUV - vec2(0.5) * params.renderSizeRcp;

    
    // texture gather to find nearest depth
    //      a  b  c  d
    //      e  f  g  h
    //      i  j  k  l
    //      m  n  o  p
    //btmLeft mnji
    //btmRight oplk
    //topLeft  efba
    //topRight ghdc

    vec4 btmLeft = (textureGather(InputDepth, gatherCoord, 0) * (1.0 / 16777216.0));
    vec2 v10 = vec2(params.renderSizeRcp.x * 2.0f, 0.0);
    vec4 btmRight = (textureGather(InputDepth, (gatherCoord+v10), 0) * (1.0 / 16777216.0));
    vec2 v12 = vec2(0.0, params.renderSizeRcp.y * 2.0f);
	vec4 topLeft = (textureGather(InputDepth, (gatherCoord+v12), 0) * (1.0 / 16777216.0));
	vec2 v14 = vec2(params.renderSizeRcp.x * 2.0f, params.renderSizeRcp.y * 2.0f);
	vec4 topRight = (textureGather(InputDepth, (gatherCoord+v14), 0) * (1.0 / 16777216.0));
	float maxC = max(max(max(btmLeft.z,btmRight.w),topLeft.y),topRight.x);
	float btmLeft4 = max(max(max(btmLeft.y,btmLeft.x),btmLeft.z),btmLeft.w);
	float btmLeftMax9 = max(topLeft.x,max(max(maxC,btmLeft4),btmRight.x));

    float depthclip = 0.0;
    if (maxC > 1.0e-05f)
    {
        float btmRight4 = max(max(max(btmRight.y,btmRight.x),btmRight.z),btmRight.w);
        float topLeft4 = max(max(max(topLeft.y,topLeft.x),topLeft.z),topLeft.w);
        float topRight4 = max(max(max(topRight.y,topRight.x),topRight.z),topRight.w);

        float Wdepth = 0.0;
        float Ksep = 1.37e-05f;
        float Kfov = params.cameraFovAngleHor;
        float diagonal_length = length(params.renderSize);
        float Ksep_Kfov_diagonal = Ksep * Kfov * diagonal_length;

		float Depthsep = Ksep_Kfov_diagonal * maxC;
		float EPSILON = 1.19e-07f;
		Wdepth += clamp((Depthsep / (abs(maxC - btmLeft4) + EPSILON)), 0.0, 1.0);
		Wdepth += clamp((Depthsep / (abs(maxC - btmRight4) + EPSILON)), 0.0, 1.0);
		Wdepth += clamp((Depthsep / (abs(maxC - topLeft4) + EPSILON)), 0.0, 1.0);
		Wdepth += clamp((Depthsep / (abs(maxC - topRight4) + EPSILON)), 0.0, 1.0);
        depthclip = clamp(1.0f - Wdepth * 0.25, 0.0, 1.0);
    }

    //refer to ue/fsr2 PostProcessFFX_FSR2ConvertVelocity.usf, and using nearest depth for dilated motion

    // The GS's per-pixel motion (every 3D vertex carries it, the camera's included).
    vec2 motion = (texelFetch(InputVelocity, ivec2(InputPos), 0).xy - params.jitterDelta) * params.motionToNdc;
    MotionDepthClipAlphaBuffer = vec4(motion, depthclip, 0.0);

}