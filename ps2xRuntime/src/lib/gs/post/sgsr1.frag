#version 450
// Snapdragon Game Super Resolution 1 (Qualcomm, BSD-3-Clause, sgsr/LICENSE): single-pass spatial
// upscaling with a 12-tap Lanczos-like filter and adaptive sharpening. The algorithm is
// sgsr1_shader_mobile.frag (snapdragon-gsr, sgsr/v1/include/glsl) as published, in RGBA mode;
// only its inputs follow the presenter's passes (a push constant, the full-screen triangle's UV).
//
//                  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
precision mediump float;
precision highp int;

#define OperationMode 1
#define EdgeThreshold 8.0/255.0
#define EdgeSharpness 2.0

layout(set = 0, binding = 0) uniform mediump sampler2D uTex;
// viewport: 1 / input width, 1 / input height, input width, input height
layout(push_constant) uniform Push { highp vec4 viewport; } pc;
layout(location = 0) in highp vec2 vUV;
layout(location = 0) out vec4 outColor;

float fastLanczos2(float x)
{
	float wA = x-4.0;
	float wB = x*wA-wA;
	wA *= wA;
	return wB*wA;
}
vec2 weightY(float dx, float dy,float c, float std)
{
	float x = ((dx*dx)+(dy* dy))* 0.55 + clamp(abs(c)*std, 0.0, 1.0);
	float w = fastLanczos2(x);
	return vec2(w, w * c);	
}

void main()
{
	int mode = OperationMode;
	float edgeThreshold = EdgeThreshold;
	float edgeSharpness = EdgeSharpness;

	vec4 color;
	if(mode == 1)
		color.xyz = textureLod(uTex,vUV,0.0).xyz;
	else
		color.xyzw = textureLod(uTex,vUV,0.0).xyzw;

	highp float xCenter;
	xCenter = abs(vUV.x+-0.5);
	highp float yCenter;
	yCenter = abs(vUV.y+-0.5);

	//todo: config the SR region based on needs
	//if ( mode!=4 && xCenter*xCenter+yCenter*yCenter<=0.4 * 0.4)
	if ( mode!=4)
	{
		highp vec2 imgCoord = ((vUV*pc.viewport.zw)+vec2(-0.5,0.5));
		highp vec2 imgCoordPixel = floor(imgCoord);
		highp vec2 coord = (imgCoordPixel*pc.viewport.xy);
		vec2 pl = (imgCoord+(-imgCoordPixel));
		vec4  left = textureGather(uTex,coord, OperationMode);

		float edgeVote = abs(left.z - left.y) + abs(color[mode] - left.y)  + abs(color[mode] - left.z) ;
		if(edgeVote > edgeThreshold)
		{
			coord.x += pc.viewport.x;

			vec4 right = textureGather(uTex,coord + vec2(pc.viewport.x, 0.0), OperationMode);
			vec4 upDown;
			upDown.xy = textureGather(uTex,coord + vec2(0.0, -pc.viewport.y), OperationMode).wz;
			upDown.zw  = textureGather(uTex,coord+ vec2(0.0, pc.viewport.y), OperationMode).yx;

			float mean = (left.y+left.z+right.x+right.w)*0.25;
			left = left - vec4(mean);
			right = right - vec4(mean);
			upDown = upDown - vec4(mean);
			color.w =color[mode] - mean;

			float sum = (((((abs(left.x)+abs(left.y))+abs(left.z))+abs(left.w))+(((abs(right.x)+abs(right.y))+abs(right.z))+abs(right.w)))+(((abs(upDown.x)+abs(upDown.y))+abs(upDown.z))+abs(upDown.w)));				
			float std = 2.181818/sum;
			
			vec2 aWY = weightY(pl.x, pl.y+1.0, upDown.x,std);				
			aWY += weightY(pl.x-1.0, pl.y+1.0, upDown.y,std);
			aWY += weightY(pl.x-1.0, pl.y-2.0, upDown.z,std);
			aWY += weightY(pl.x, pl.y-2.0, upDown.w,std);			
			aWY += weightY(pl.x+1.0, pl.y-1.0, left.x,std);
			aWY += weightY(pl.x, pl.y-1.0, left.y,std);
			aWY += weightY(pl.x, pl.y, left.z,std);
			aWY += weightY(pl.x+1.0, pl.y, left.w,std);
			aWY += weightY(pl.x-1.0, pl.y-1.0, right.x,std);
			aWY += weightY(pl.x-2.0, pl.y-1.0, right.y,std);
			aWY += weightY(pl.x-2.0, pl.y, right.z,std);
			aWY += weightY(pl.x-1.0, pl.y, right.w,std);

			float finalY = aWY.y/aWY.x;

			float maxY = max(max(left.y,left.z),max(right.x,right.w));
			float minY = min(min(left.y,left.z),min(right.x,right.w));
			finalY = clamp(edgeSharpness*finalY, minY, maxY);
					
			float deltaY = finalY -color.w;	
			
			//smooth high contrast input
			deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

			color.x = clamp((color.x+deltaY),0.0,1.0);
			color.y = clamp((color.y+deltaY),0.0,1.0);
			color.z = clamp((color.z+deltaY),0.0,1.0);
		}
	}

	color.w = 1.0;  //assume alpha channel is not used
	outColor.xyzw = color;
}