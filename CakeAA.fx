//================================================================================================================================
// Continuous Adaptive Kerneling Eigen AA (Cake AA) — SCALE-INVARIANT VERSION
// OPTIMIZED WITH SMART CONSENSUS PRE-FILTERING
// ALL THREE TENSORS ARE SCALE-INVARIANT (operate at all frequencies)
// TWO-TENSOR BUILD: 5x5 + 7x7 (triangular lattice removed)
// OBLIQUE-AWARE TUNING (5x5 oblique bias, dirWeight relaxation)
// BALANCED COLOR/JUMP SENSITIVITY
// Copyright © 2026 Alex Aaron Goven
// License: GNU General Public License v3
// CHROMA-AWARE NEARBY-ALIAS SENSITIVE CONSENSUS DETECTOR
//
// DEBUG VIEW:
//   R = 5x5 strength, B = 7x7 strength, additive.
//   Black background; overlaps sum toward white.
//================================================================================================================================

#ifndef __RESHADE_EMBEDDED__
#define __RESHADE_EMBEDDED__

#ifndef BUFFER_WIDTH
#define BUFFER_WIDTH 1920
#endif
#ifndef BUFFER_HEIGHT
#define BUFFER_HEIGHT 1080
#endif
#ifndef BUFFER_RCP_WIDTH
#define BUFFER_RCP_WIDTH (1.0f / BUFFER_WIDTH)
#endif
#ifndef BUFFER_RCP_HEIGHT
#define BUFFER_RCP_HEIGHT (1.0f / BUFFER_HEIGHT)
#endif

void PostProcessVS(in uint id : SV_VertexID, out float4 position : SV_Position, out float2 texcoord : TEXCOORD)
{
	texcoord.x = (id == 2) ? 2.0 : 0.0;
	texcoord.y = (id == 1) ? 2.0 : 0.0;
	position = float4(texcoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

#endif // __RESHADE_EMBEDDED__

// ================================ UI CONTROLS ================================

uniform bool Debug <
	label = "Debug";
	description = "Shows tensor strength visualization (Red=5x5, Blue=7x7, additive to white on black)";
> = false;

// ================================ Parameters ================================
static const float  AA_STRENGTH          = 1.0f;
static const float  RELATIVE_KNEE        = 0.005f;
static const float  NORMALIZED_DIST      = 3.5f;
static const float  COLOR_SENSITIVITY    = 7.0f;   // modest, safe edge-stop sensitivity
static const float  DIRECTION_ALIGNMENT_MIN = 0.7f;
static const float  MIN_COHERENCE        = 0.08f;

static const float  CONSENSUS_THRESHOLD  = 0.6f;
static const float  EDGE_SENSITIVITY     = 1.2f;
static const float  MIN_CONSENSUS_EDGE   = 0.15f;

static const float  QUALITY_TARGET_HIGH  = 0.98f;
static const float  QUALITY_TARGET_LOW   = 0.65f;
static const float  QUALITY_ADAPT_RATE   = 0.7f;
static const float  QUALITY_BOOST_STRONG = 0.15f;
static const float  QUALITY_CLAMP_MIN    = 0.4f;
static const float  QUALITY_CLAMP_MAX    = 0.98f;

// Coherence damping disabled (floor = 1.0). Kept as a tuning knob in case
// corner behavior needs adjustment. Set to 0.3 or lower to re-enable.
static const float  CORNER_CONFIDENCE_FLOOR = 1.0f;

// Constant-folded product (was JUMP_GAIN * EDGE_SENSITIVITY at runtime).
static const float  JUMP_GAIN_EDGE       = 3.0f;   // balanced

// Oblique tuning
static const float  OBLIQUE_BLEND_FLOOR  = 0.80f;
static const float  OBLIQUE_DIRMIN_FLOOR = 0.55f;
static const float  OBLIQUE_DIRMAX_WIDTH = 0.30f;
static const float  OBLIQUE_5X5_BIAS     = 0.60f;

#define RT_METRICS float4(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT, BUFFER_WIDTH, BUFFER_HEIGHT)
#define LUM_WEIGHTS float3(0.2126f, 0.7152f, 0.0722f)
#define INV_SQRT2 0.7071067811865475f

texture BackBufferTex : COLOR;

// Packed: R=strength5, G=strength7, B=angleQuality, A=consensus
texture EdgeDataTex0
{
	Width  = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = RGBA16F;
};

// R=tangent5.x, G=tangent5.y, B=tangent7.x, A=tangent7.y
texture EdgeDataTex1
{
	Width  = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = RGBA16F;
};

// R=coherence_7 (rest zero — triangular lattice removed)
texture EdgeDataTex2
{
	Width  = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = RGBA16F;
};

// Unused (kept for pass compatibility)
texture EdgeDataTex3
{
	Width  = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = RGBA16F;
};

sampler ColorSampler
{
	Texture = BackBufferTex;
	MinFilter = Linear;
	MagFilter = Linear;
	AddressU = Clamp;
	AddressV = Clamp;
};

sampler EdgeDataSampler0
{
	Texture = EdgeDataTex0;
	MinFilter = Point;
	MagFilter = Point;
};

sampler EdgeDataSampler1
{
	Texture = EdgeDataTex1;
	MinFilter = Point;
	MagFilter = Point;
};

sampler EdgeDataSampler2
{
	Texture = EdgeDataTex2;
	MinFilter = Point;
	MagFilter = Point;
};

sampler EdgeDataSampler3
{
	Texture = EdgeDataTex3;
	MinFilter = Point;
	MagFilter = Point;
};

// =============================================================================
// Utility functions
// =============================================================================
float max3(float3 v) { return max(v.x, max(v.y, v.z)); }

static const float DEG2RAD = 0.017453292519943295f;

static const float KERNEL_5x5[5] = { 1.0f, 4.0f, 6.0f, 4.0f, 1.0f };
static const float KERNEL_7x7[7] = { 1.0f, 3.0f, 6.0f, 7.0f, 6.0f, 3.0f, 1.0f };

// =============================================================================
// FAST CONSENSUS EDGE DETECTOR - CHROMA-AWARE, NEARBY-ALIAS SENSITIVE
// =============================================================================
float fast_edge_consensus(float2 tex, float2 offset, float3 centerColor, float centerLum)
{
	float3 c[8];
	[unroll]
	for (int i = 0; i < 8; i++) {
		int row = i / 3 - 1;
		int col = i % 3 - 1;
		if (i >= 4) {
			row = (i + 1) / 3 - 1;
			col = (i + 1) % 3 - 1;
		}
		float2 uv = clamp(tex + float2((float)col, (float)row) * offset, 0.0f, 1.0f);
		c[i] = tex2Dlod(ColorSampler, float4(uv, 0.0f, 0.0f)).rgb;
	}

	const float3 cNW = c[0];
	const float3 cN  = c[1];
	const float3 cNE = c[2];
	const float3 cW  = c[3];
	const float3 cE  = c[4];
	const float3 cSW = c[5];
	const float3 cS  = c[6];
	const float3 cSE = c[7];

	float3 cN2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( 0.0f, -2.0f) * offset, 0.0f, 1.0f), 0.0f, 0.0f)).rgb;
	float3 cS2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( 0.0f,  2.0f) * offset, 0.0f, 1.0f), 0.0f, 0.0f)).rgb;
	float3 cE2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( 2.0f,  0.0f) * offset, 0.0f, 1.0f), 0.0f, 0.0f)).rgb;
	float3 cW2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2(-2.0f,  0.0f) * offset, 0.0f, 1.0f), 0.0f, 0.0f)).rgb;

	float3 gx = cNW * -1.0f + cNE * 1.0f
	          + cW  * -2.0f + cE  * 2.0f
	          + cSW * -1.0f + cSE * 1.0f;

	float3 gy = cNW * -1.0f + cN * -2.0f + cNE * -1.0f
	          + cSW *  1.0f + cS *  2.0f + cSE *  1.0f;

	float3 sobelCh = sqrt(gx * gx + gy * gy);
	float sobelMag = max3(sobelCh);
	float lumaSobel = dot(abs(gx) + abs(gy), LUM_WEIGHTS) * 0.5f;
	sobelMag = max(sobelMag, lumaSobel);

	float lumN2 = dot(cN2, LUM_WEIGHTS);
	float lumS2 = dot(cS2, LUM_WEIGHTS);
	float lumE2 = dot(cE2, LUM_WEIGHTS);
	float lumW2 = dot(cW2, LUM_WEIGHTS);
	float farGrad = max(abs(lumN2 - centerLum), max(abs(lumS2 - centerLum),
	                 max(abs(lumE2 - centerLum), abs(lumW2 - centerLum))));
	sobelMag = max(sobelMag, farGrad * 0.75f);

	float edge1 = saturate(sobelMag * EDGE_SENSITIVITY);

	float axN = max3(abs(cN - centerColor));
	float axS = max3(abs(cS - centerColor));
	float axE = max3(abs(cE - centerColor));
	float axW = max3(abs(cW - centerColor));
	float axialJump = max(max(axN, axS), max(axE, axW));

	float dgNW = max3(abs(cNW - centerColor));
	float dgNE = max3(abs(cNE - centerColor));
	float dgSW = max3(abs(cSW - centerColor));
	float dgSE = max3(abs(cSE - centerColor));
	float diagJump = max(max(dgNW, dgNE), max(dgSW, dgSE));

	float edge2 = saturate(axialJump * JUMP_GAIN_EDGE);
	float edge3 = saturate(diagJump  * JUMP_GAIN_EDGE);

	float consensus = edge1 * edge2 * edge3;
	float threshold = CONSENSUS_THRESHOLD * CONSENSUS_THRESHOLD * CONSENSUS_THRESHOLD;
	consensus = smoothstep(threshold * 0.5f, threshold * 1.5f, consensus);

	float minEdge = max(edge1, max(edge2, edge3));
	consensus *= smoothstep(MIN_CONSENSUS_EDGE * 0.5f, MIN_CONSENSUS_EDGE * 1.5f, minEdge);

	return consensus;
}

// =============================================================================
// FUSED TENSOR COMPUTATION (5x5 + 7x7 only)
// =============================================================================
void compute_all_tensors(
	float2 tex,
	float2 offset,
	float3 centerColor,
	out float3 gx5, out float3 gy5, out float w5,
	out float3 gx7, out float3 gy7, out float w7)
{
	// 5x5 Rotated body
	gx5 = float3(0.0f, 0.0f, 0.0f);
	gy5 = float3(0.0f, 0.0f, 0.0f);
	float totalWeight5 = 0.0f;

	[unroll]
	for (int row = -2; row <= 2; row++)
	{
		[unroll]
		for (int col = -2; col <= 2; col++)
		{
			float2 screenOffset = float2(
				(float)(col - row) * INV_SQRT2,
				(float)(col + row) * INV_SQRT2
			);
			float2 sampleUV = clamp(tex + screenOffset * offset, 0.0f, 1.0f);
			float w = KERNEL_5x5[row + 2] * KERNEL_5x5[col + 2] * 0.00390625f;
			float3 sampleColor = tex2Dlod(ColorSampler, float4(sampleUV, 0.0f, 0.0f)).rgb;
			gx5 += sampleColor * screenOffset.x * w;
			gy5 += sampleColor * screenOffset.y * w;
			totalWeight5 += w;
		}
	}

	// 7x7 Axis-aligned body
	gx7 = float3(0.0f, 0.0f, 0.0f);
	gy7 = float3(0.0f, 0.0f, 0.0f);
	float totalWeight7 = 0.0f;

	[unroll]
	for (int dy = -3; dy <= 3; dy++)
	{
		[unroll]
		for (int dx = -3; dx <= 3; dx++)
		{
			float2 sampleUV = clamp(tex + float2((float)dx, (float)dy) * offset, 0.0f, 1.0f);
			float w = KERNEL_7x7[dy + 3] * KERNEL_7x7[dx + 3] / 729.0f;
			float3 sampleColor = tex2Dlod(ColorSampler, float4(sampleUV, 0.0f, 0.0f)).rgb;
			gx7 += sampleColor * (float)dx * w;
			gy7 += sampleColor * (float)dy * w;
			totalWeight7 += w;
		}
	}

	if (totalWeight5 > 0.0f) { float inv = 1.0f / totalWeight5; gx5 *= inv; gy5 *= inv; }
	if (totalWeight7 > 0.0f) { float inv = 1.0f / totalWeight7; gx7 *= inv; gy7 *= inv; }

	w5 = totalWeight5;
	w7 = totalWeight7;
}

// =============================================================================
// Structure tensor functions
// =============================================================================
void normalize_and_regularize_tensor(
	inout float S11, inout float S12, inout float S22,
	float weightSum,
	float gradientScale)
{
	const float EPSILON = 1e-6f;

	if (weightSum > 0.0f)
	{
		float invWeightSum = 1.0f / weightSum;
		S11 = S11 * invWeightSum + EPSILON;
		S12 *= invWeightSum;
		S22 = S22 * invWeightSum + EPSILON;
	}
	else
	{
		S11 += EPSILON;
		S22 += EPSILON;
	}

	if (gradientScale > 0.0f)
	{
		float scaleFactor = 1.0f / (gradientScale * gradientScale);
		S11 *= scaleFactor;
		S12 *= scaleFactor;
		S22 *= scaleFactor;
	}
}

void decompose_structure_tensor(
	float S11, float S12, float S22,
	out float lambda1, out float lambda2, out float2 tangent)
{
	float trace = S11 + S22;
	float det = S11 * S22 - S12 * S12;
	float disc = sqrt(max(trace * trace - 4.0f * det, 0.0f));

	lambda1 = (trace + disc) * 0.5f;
	lambda2 = (trace - disc) * 0.5f;

	if (disc < 1e-6f)
	{
		tangent = float2(1.0f, 0.0f);
	}
	else
	{
		tangent = float2(S12, lambda2 - S11);
		float len = length(tangent);
		if (len > 1e-6f)
			tangent /= len;
		else
			tangent = float2(1.0f, 0.0f);
	}
}

// =============================================================================
// Angle quality
// =============================================================================
float compute_angle_quality(float2 tangent, float angle)
{
	float cosA = abs(tangent.x);
	float sinA = abs(tangent.y);
	float maxBasis = max(cosA, sinA);
	float maxDiag = max(abs(cosA + sinA), abs(cosA - sinA)) * INV_SQRT2;
	float primaryQuality = max(maxBasis, maxDiag);
	float nearestAxis = round(angle / (DEG2RAD * 45.0f)) * (DEG2RAD * 45.0f);
	float obliqueness = 1.0f - abs(cos(angle - nearestAxis));
	float obliquenessBoost = obliqueness * obliqueness;
	return lerp(primaryQuality, 1.0f, obliquenessBoost * 0.5f);
}

// =============================================================================
// Adaptive blend (two-tensor version)
// =============================================================================
float compute_adaptive_blend(
	float strength5,
	float strength7,
	float angleQuality,
	float longEdgeFactor)
{
	float edgeStrength = sqrt((strength5*strength5 + strength7*strength7) * 0.5f);
	edgeStrength = clamp(edgeStrength, 0.0f, 1.0f);

	float coherence = clamp(longEdgeFactor, 0.0f, 1.0f);

	float strengthQuality = lerp(QUALITY_TARGET_LOW, QUALITY_TARGET_HIGH, edgeStrength);

	float structureConfidence = lerp(CORNER_CONFIDENCE_FLOOR, 1.0f, coherence);
	strengthQuality *= structureConfidence;

	float coherenceBoost = coherence * QUALITY_BOOST_STRONG;
	float angleFactor = lerp(OBLIQUE_BLEND_FLOOR, 1.0f, angleQuality);

	float blend = strengthQuality + coherenceBoost;
	blend *= angleFactor;
	blend = lerp(blend, 0.5f, 1.0f - QUALITY_ADAPT_RATE);
	blend = clamp(blend, QUALITY_CLAMP_MIN, QUALITY_CLAMP_MAX);

	if (edgeStrength < 0.1f)
		blend = lerp(blend, 0.1f, 1.0f - edgeStrength * 10.0f);
	if (edgeStrength > 0.8f)
		blend = lerp(blend, 0.9f, (edgeStrength - 0.8f) * 2.5f);

	return clamp(blend, 0.0f, 1.0f);
}

// =============================================================================
// Edge Detection Pass
// =============================================================================
void PS_DetectEdges(
	float4 pos : SV_Position,
	float2 tex : TEXCOORD,
	out float4 out0 : SV_Target0,
	out float4 out1 : SV_Target1,
	out float4 out2 : SV_Target2,
	out float4 out3 : SV_Target3)
{
	float2 offset = RT_METRICS.xy;
	float3 centerColor = tex2D(ColorSampler, tex).rgb;
	float centerLum = dot(centerColor, LUM_WEIGHTS);

	float consensus = fast_edge_consensus(tex, offset, centerColor, centerLum);

	if (consensus < 0.1f) {
		out0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
		out1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
		out2 = float4(0.0f, 0.0f, 0.0f, 0.0f);
		out3 = float4(0.0f, 0.0f, 0.0f, 0.0f);
		return;
	}

	float3 gx5, gy5, gx7, gy7;
	float w5, w7;

	compute_all_tensors(tex, offset, centerColor, gx5, gy5, w5, gx7, gy7, w7);

	float S11_5 = dot(gx5, gx5);
	float S12_5 = dot(gx5, gy5);
	float S22_5 = dot(gy5, gy5);

	float S11_7 = dot(gx7, gx7);
	float S12_7 = dot(gx7, gy7);
	float S22_7 = dot(gy7, gy7);

	normalize_and_regularize_tensor(S11_5, S12_5, S22_5, w5, 2.0f);
	normalize_and_regularize_tensor(S11_7, S12_7, S22_7, w7, 3.0f);

	float lambda1_5, lambda2_5;
	float2 tangent5;
	decompose_structure_tensor(S11_5, S12_5, S22_5, lambda1_5, lambda2_5, tangent5);

	float lambda1_7, lambda2_7;
	float2 tangent7;
	decompose_structure_tensor(S11_7, S12_7, S22_7, lambda1_7, lambda2_7, tangent7);

	float adaptedKnee = RELATIVE_KNEE * max(centerLum, 0.02f);

	float meanGradMag_5 = sqrt(max(lambda1_5 + lambda2_5, 0.0f));
	float meanGradMag_7 = sqrt(max(lambda1_7 + lambda2_7, 0.0f));

	float strength5 = meanGradMag_5 / (meanGradMag_5 + adaptedKnee);
	float strength7 = meanGradMag_7 / (meanGradMag_7 + adaptedKnee);

	float coherence_7 = (lambda1_7 - lambda2_7) / max(lambda1_7 + lambda2_7, 1e-8f);
	float longEdgeFactor = coherence_7;

	float angle7 = atan2(tangent7.y, tangent7.x);
	float angleQuality = compute_angle_quality(tangent7, angle7);

	strength5 *= AA_STRENGTH;
	strength7 *= AA_STRENGTH;

	out0 = float4(strength5, strength7, angleQuality, consensus);
	out1 = float4(tangent5.x, tangent5.y, tangent7.x, tangent7.y);
	out2 = float4(coherence_7, 0.0f, 0.0f, 0.0f);
	out3 = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// =============================================================================
// Edge stop weight
// =============================================================================
float edge_stop_weight(float3 sampleColor, float3 centerColor, float sampleEdge)
{
	float colorDiff = max(abs(sampleColor.x - centerColor.x),
	                     max(abs(sampleColor.y - centerColor.y),
	                         abs(sampleColor.z - centerColor.z)));
	float x = colorDiff * COLOR_SENSITIVITY;
	float colorWeight = 1.0f / (1.0f + x * x * 2.0f);
	return lerp(colorWeight, 1.0f, sampleEdge);
}

// =============================================================================
// Anti-Aliasing Pass
// =============================================================================
float4 PS_ApplyAA(float4 pos : SV_Position, float2 tex : TEXCOORD) : SV_Target
{
	float4 edgeData0 = tex2D(EdgeDataSampler0, tex);
	float consensus = edgeData0.a;

	// Debug: R=5x5, B=7x7
	if (Debug) {
		float s5 = edgeData0.r;
		float s7 = edgeData0.g;
		float3 dbg = float3(s5, 0.0f, s7);
		return float4(saturate(dbg), 1.0f);
	}

	float3 centerColor = tex2D(ColorSampler, tex).rgb;

	if (consensus < 0.05f) {
		return float4(centerColor, 1.0f);
	}

	float4 edgeData1 = tex2D(EdgeDataSampler1, tex);
	float4 edgeData2 = tex2D(EdgeDataSampler2, tex);

	float strength5 = edgeData0.r;
	float strength7 = edgeData0.g;
	float angleQuality = edgeData0.b;

	float2 tangent5 = edgeData1.rg;
	float2 tangent7 = edgeData1.ba;
	float coherence_7 = edgeData2.r;

	float blend = compute_adaptive_blend(strength5, strength7, angleQuality, coherence_7);
	blend *= AA_STRENGTH;

	if (blend < 0.01f)
		return float4(centerColor, 1.0f);

	float safeLongEdgeFactor = max(coherence_7, MIN_COHERENCE);
	float totalStrength = strength5 + strength7;
	float kernelPref7 = strength7 / max(totalStrength, 1e-6f);
	float boost7 = safeLongEdgeFactor * angleQuality;
	float kernelMix = lerp(kernelPref7, 1.0f, boost7);
	kernelMix = smoothstep(0.15f, 0.65f, kernelMix);

	float obliqueBias = saturate((0.7f - angleQuality) * 2.0f)
	                  * saturate((0.5f - coherence_7) * 2.0f);
	kernelMix = lerp(kernelMix, 0.0f, obliqueBias * OBLIQUE_5X5_BIAS);

	float angleScale = angleQuality;
	float longScale = 1.0f + safeLongEdgeFactor * 0.45f;

	float2 step5 = tangent5 * (float2)RT_METRICS.xy * (NORMALIZED_DIST / 2.5f) * angleScale * longScale;
	float2 step7 = tangent7 * (float2)RT_METRICS.xy * (NORMALIZED_DIST / 3.8f) * angleScale * longScale;

	float initWeight = lerp(16.0f, 32.0f, kernelMix);

	float3 acc = centerColor * initWeight;
	float totalWeight = initWeight;

	float dirAlignMin = lerp(OBLIQUE_DIRMIN_FLOOR, DIRECTION_ALIGNMENT_MIN, angleQuality);
	float dirAlignMax = dirAlignMin + OBLIQUE_DIRMAX_WIDTH;

	float kw5_1 = 9.0f * (1.0f - kernelMix);
	float kw5_2 = 2.0f * (1.0f - kernelMix);

	float kw7_1 = 24.0f * kernelMix;
	float kw7_2 = 12.0f * kernelMix;
	float kw7_3 = 3.0f  * kernelMix;

	// 5x5 Kernel
	if (strength5 > 0.01f)
	{
		[unroll]
		for (int i = -3; i <= 3; i++)
		{
			if (i == 0) continue;

			float idx = (float)i;
			float wEdge = (abs(i) == 3) ? kw5_2 : kw5_1;

			float2 uv = clamp(tex + step5 * idx, 0.0f, 1.0f);
			float3 color = tex2Dlod(ColorSampler, float4(uv, 0.0f, 0.0f)).rgb;
			float4 se0 = tex2Dlod(EdgeDataSampler0, float4(uv, 0.0f, 0.0f));
			float4 se1 = tex2Dlod(EdgeDataSampler1, float4(uv, 0.0f, 0.0f));
			float2 dir = se1.rg;
			float edge = max3(se0.rgb);

			float dirWeight = smoothstep(dirAlignMin, dirAlignMax, dot(tangent5, dir));
			float stopWeight = edge_stop_weight(color, centerColor, edge);

			float w = wEdge * dirWeight * stopWeight;
			acc += color * w;
			totalWeight += w;
		}
	}

	// 7x7 Kernel
	if (strength7 > 0.01f)
	{
		[unroll]
		for (int i = -3; i <= 3; i++)
		{
			if (i == 0) continue;

			float idx = (float)i;
			float wEdge = (abs(i) == 3) ? kw7_3 : ((abs(i) == 2) ? kw7_2 : kw7_1);

			float2 uv = clamp(tex + step7 * idx, 0.0f, 1.0f);
			float3 color = tex2Dlod(ColorSampler, float4(uv, 0.0f, 0.0f)).rgb;
			float4 se0 = tex2Dlod(EdgeDataSampler0, float4(uv, 0.0f, 0.0f));
			float4 se1 = tex2Dlod(EdgeDataSampler1, float4(uv, 0.0f, 0.0f));
			float2 dir = se1.ba;
			float edge = max3(se0.rgb);

			float dirWeight = smoothstep(dirAlignMin, dirAlignMax, dot(tangent7, dir));
			float stopWeight = edge_stop_weight(color, centerColor, edge);

			float w = wEdge * dirWeight * stopWeight;
			acc += color * w;
			totalWeight += w;
		}
	}

	float3 finalColor = (totalWeight > 0.0f) ? (acc / totalWeight) : centerColor;
	return float4(lerp(centerColor, finalColor, blend), 1.0f);
}

// =============================================================================
technique CakeAA
{
	pass EdgeDetection
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_DetectEdges;
		RenderTarget0 = EdgeDataTex0;
		RenderTarget1 = EdgeDataTex1;
		RenderTarget2 = EdgeDataTex2;
		RenderTarget3 = EdgeDataTex3;
		ClearRenderTargets = true;
	}
	pass AntiAliasing
	{
		VertexShader = PostProcessVS;
		PixelShader = PS_ApplyAA;
	}
}
