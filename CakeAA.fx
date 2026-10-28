//================================================================================================================================
// Continuous Adaptive Kerneling Eigen AA (Cake AA) — SCALE-INVARIANT VERSION
// OPTIMIZED WITH SMART CONSENSUS PRE-FILTERING
// ALL THREE TENSORS ARE SCALE-INVARIANT (operate at all frequencies)
// OBLIQUE-AWARE TUNING (tri kernel presence, 5x5 oblique bias, dirWeight relaxation)
// BALANCED COLOR/JUMP SENSITIVITY (suppresses grain without gating faint real edges)
// Copyright © 2026 Alex Aaron Goven
// License: GNU General Public License v3
// CHROMA-AWARE NEARBY-ALIAS SENSITIVE CONSENSUS DETECTOR
//
// DEBUG VIEW:
//   R = 5x5 strength, G = triangular strength, B = 7x7 strength, additive.
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
	description = "Shows tensor strength visualization (Red=5x5, Green=triangular, Blue=7x7, additive to white on black)";
> = false;

// ================================ Parameters ================================
static const float  AA_STRENGTH          = 1.0f;
static const float  RELATIVE_KNEE        = 0.005f;
static const float  NORMALIZED_DIST      = 3.5f;
static const float  COLOR_SENSITIVITY    = 7.0f;   // modest, safe edge-stop sensitivity
static const float  DIRECTION_ALIGNMENT_MIN = 0.7f;
static const float  MIN_COHERENCE        = 0.08f;
static const float  TRIANGULAR_WEIGHT    = 0.45f;

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
static const float  JUMP_GAIN_EDGE       = 3.0f;   // was 4.0 → 3.4 → 2.8 → 2.2 → back to 3.0, balanced

// Oblique tuning
static const float  OBLIQUE_BLEND_FLOOR  = 0.80f;
static const float  TRI_KERNEL_PRESENCE  = 0.75f;
static const float  OBLIQUE_DIRMIN_FLOOR = 0.55f;
static const float  OBLIQUE_DIRMAX_WIDTH = 0.30f;
static const float  OBLIQUE_5X5_BIAS     = 0.60f;

#define RT_METRICS float4(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT, BUFFER_WIDTH, BUFFER_HEIGHT)
#define LUM_WEIGHTS float3(0.2126f, 0.7152f, 0.0722f)
#define INV_THIRD 0.3333333f
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

// R=strength_tri, G=lambda1_tri, B=lambda2_tri, A=coherence_tri
texture EdgeDataTex2
{
	Width  = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;
	Format = RGBA16F;
};

// R=tangent_tri.x, G=tangent_tri.y, B=longEdgeFactor, A=unused (kept at 0)
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

// Ring-2 hexagonal directions for the triangular tensor (precomputed).
static const float2 HEX_RING2[6] = {
	float2( 0.8660254037844386f,  0.5f),
	float2( 0.0f,                 1.0f),
	float2(-0.8660254037844386f,  0.5f),
	float2(-0.8660254037844386f, -0.5f),
	float2( 0.0f,                -1.0f),
	float2( 0.8660254037844386f, -0.5f)
};

// =============================================================================
// FAST CONSENSUS EDGE DETECTOR - CHROMA-AWARE, NEARBY-ALIAS SENSITIVE
// =============================================================================
float fast_edge_consensus(float2 tex, float2 offset, float3 centerColor, float centerLum)
{
	// ---- 3x3 core samples, excluding center (8 taps) ----
	// Index map: c[0]=NW c[1]=N c[2]=NE c[3]=W c[4]=E c[5]=SW c[6]=S c[7]=SE
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

	// ---- Sparse 2px outer ring (4 taps: N,S,E,W at distance 2) ----
	float3 cN2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( 0.0f, -2.0f) * offset, 0.0f, 1.0f), 0.0f, 0.0f)).rgb;
	float3 cS2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( 0.0f,  2.0f) * offset, 0.0f, 1.0f), 0.0f, 0.0f)).rgb;
	float3 cE2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( 2.0f,  0.0f) * offset, 0.0f, 1.0f), 0.0f, 0.0f)).rgb;
	float3 cW2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2(-2.0f,  0.0f) * offset, 0.0f, 1.0f), 0.0f, 0.0f)).rgb;

	// =====================================================================
	// GATE 1: Per-channel Sobel magnitude (luma + chroma)
	// =====================================================================
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

	// =====================================================================
	// GATE 2: Axial color-jump (N/S/E/W at 1px) — chroma-aware
	// =====================================================================
	float axN = max3(abs(cN - centerColor));
	float axS = max3(abs(cS - centerColor));
	float axE = max3(abs(cE - centerColor));
	float axW = max3(abs(cW - centerColor));
	float axialJump = max(max(axN, axS), max(axE, axW));

	// =====================================================================
	// GATE 3: Diagonal color-jump (corners at 1px) — chroma-aware
	// =====================================================================
	float dgNW = max3(abs(cNW - centerColor));
	float dgNE = max3(abs(cNE - centerColor));
	float dgSW = max3(abs(cSW - centerColor));
	float dgSE = max3(abs(cSE - centerColor));
	float diagJump = max(max(dgNW, dgNE), max(dgSW, dgSE));

	float edge2 = saturate(axialJump * JUMP_GAIN_EDGE);
	float edge3 = saturate(diagJump  * JUMP_GAIN_EDGE);

	// =====================================================================
	// AND-gate consensus
	// =====================================================================
	float consensus = edge1 * edge2 * edge3;
	float threshold = CONSENSUS_THRESHOLD * CONSENSUS_THRESHOLD * CONSENSUS_THRESHOLD;
	consensus = smoothstep(threshold * 0.5f, threshold * 1.5f, consensus);

	float minEdge = max(edge1, max(edge2, edge3));
	consensus *= smoothstep(MIN_CONSENSUS_EDGE * 0.5f, MIN_CONSENSUS_EDGE * 1.5f, minEdge);

	return consensus;
}

// =============================================================================
// Triangular Lattice directions
// =============================================================================
static const float2 HEX_DIRECTIONS[6] = {
	float2( 1.0f,  0.0f),
	float2( 0.5f,  0.8660254f),
	float2(-0.5f,  0.8660254f),
	float2(-1.0f,  0.0f),
	float2(-0.5f, -0.8660254f),
	float2( 0.5f, -0.8660254f)
};

// =============================================================================
// FUSED TENSOR COMPUTATION (bodies only — no fans)
// =============================================================================
void compute_all_tensors(
	float2 tex,
	float2 offset,
	float3 centerColor,
	out float3 gx5, out float3 gy5, out float w5,
	out float3 gx7, out float3 gy7, out float w7,
	out float3 gxT, out float3 gyT, out float wT)
{
	// -------------------------------------------------------------------------
	// 5x5 Rotated body
	// -------------------------------------------------------------------------
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

	// -------------------------------------------------------------------------
	// 7x7 Axis-aligned body
	// -------------------------------------------------------------------------
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

	// -------------------------------------------------------------------------
	// Triangular Lattice body
	// -------------------------------------------------------------------------
	gxT = float3(0.0f, 0.0f, 0.0f);
	gyT = float3(0.0f, 0.0f, 0.0f);
	const float TRI_RING_WEIGHT        = 0.45f;
	const float TRI_CENTER_WEIGHT      = 0.1f;
	const float TRI_SECOND_RING_WEIGHT = 0.15f;
	const float HEX_RADIUS             = 3.4641016f;
	const float HEX_RADIUS2            = 6.0f;
	float totalWeightT = TRI_CENTER_WEIGHT;

	[unroll]
	for (int i = 0; i < 6; i++)
	{
		float2 dir = HEX_DIRECTIONS[i];
		float2 sampleUV = clamp(tex + dir * HEX_RADIUS * offset, 0.0f, 1.0f);
		float3 sampleColor = tex2Dlod(ColorSampler, float4(sampleUV, 0.0f, 0.0f)).rgb;
		float3 diff = sampleColor - centerColor;
		gxT += diff * dir.x * TRI_RING_WEIGHT;
		gyT += diff * dir.y * TRI_RING_WEIGHT;
		totalWeightT += TRI_RING_WEIGHT;
	}

	[unroll]
	for (int j = 0; j < 6; j++)
	{
		float2 dir = HEX_RING2[j];
		float2 sampleUV = clamp(tex + dir * HEX_RADIUS2 * offset, 0.0f, 1.0f);
		float3 sampleColor = tex2Dlod(ColorSampler, float4(sampleUV, 0.0f, 0.0f)).rgb;
		float3 diff = sampleColor - centerColor;
		gxT += diff * dir.x * TRI_SECOND_RING_WEIGHT;
		gyT += diff * dir.y * TRI_SECOND_RING_WEIGHT;
		totalWeightT += TRI_SECOND_RING_WEIGHT;
	}

	// -------------------------------------------------------------------------
	// Normalize
	// -------------------------------------------------------------------------
	if (totalWeight5 > 0.0f) { float inv = 1.0f / totalWeight5; gx5 *= inv; gy5 *= inv; }
	if (totalWeight7 > 0.0f) { float inv = 1.0f / totalWeight7; gx7 *= inv; gy7 *= inv; }
	if (totalWeightT > 0.0f) { float inv = 1.0f / totalWeightT; gxT *= inv; gyT *= inv; }

	w5 = totalWeight5;
	w7 = totalWeight7;
	wT = totalWeightT;
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
// Angle quality functions
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

float compute_triangular_angle_quality(float lambda1, float lambda2, float2 tangent)
{
	float coherence = (lambda1 - lambda2) / max(lambda1 + lambda2, 1e-8f);
	float angle = atan2(tangent.y, tangent.x);
	float alignment12fold = abs(cos(6.0f * angle));
	float quality = alignment12fold + (1.0f - alignment12fold) * INV_SQRT2;
	float nearestTriAxis = round(angle / (DEG2RAD * 30.0f)) * (DEG2RAD * 30.0f);
	float triObliqueness = 1.0f - abs(cos(angle - nearestTriAxis));
	float triObliqueBoost = triObliqueness * triObliqueness;
	quality = lerp(quality, max(quality, 0.95f), triObliqueBoost * 0.4f);
	quality = lerp(0.85f, quality, coherence);
	return quality;
}

// =============================================================================
// Adaptive blend
// =============================================================================
float compute_adaptive_blend(
	float strength5,
	float strength7,
	float strength_tri,
	float angleQuality,
	float longEdgeFactor,
	float coherence_tri)
{
	float edgeStrength = sqrt((strength5*strength5 + strength7*strength7 + strength_tri*strength_tri) * INV_THIRD);
	edgeStrength = clamp(edgeStrength, 0.0f, 1.0f);

	float coherence = max(longEdgeFactor, coherence_tri);
	coherence = clamp(coherence, 0.0f, 1.0f);

	float strengthQuality = lerp(QUALITY_TARGET_LOW, QUALITY_TARGET_HIGH, edgeStrength);

	float structureConfidence = lerp(CORNER_CONFIDENCE_FLOOR, 1.0f, coherence);
	strengthQuality *= structureConfidence;

	float coherenceBoost = coherence * QUALITY_BOOST_STRONG;
	float angleFactor = lerp(OBLIQUE_BLEND_FLOOR, 1.0f, angleQuality);
	float triBoost = coherence_tri * 0.08f;

	float blend = strengthQuality + coherenceBoost + triBoost;
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

	float3 gx5, gy5, gx7, gy7, gx_tri, gy_tri;
	float w5, w7, wTri;

	compute_all_tensors(
		tex, offset, centerColor,
		gx5, gy5, w5,
		gx7, gy7, w7,
		gx_tri, gy_tri, wTri);

	float S11_5 = dot(gx5, gx5);
	float S12_5 = dot(gx5, gy5);
	float S22_5 = dot(gy5, gy5);

	float S11_7 = dot(gx7, gx7);
	float S12_7 = dot(gx7, gy7);
	float S22_7 = dot(gy7, gy7);

	float S11_tri = dot(gx_tri, gx_tri);
	float S12_tri = dot(gx_tri, gy_tri);
	float S22_tri = dot(gy_tri, gy_tri);

	normalize_and_regularize_tensor(S11_5, S12_5, S22_5, w5, 2.0f);
	normalize_and_regularize_tensor(S11_7, S12_7, S22_7, w7, 3.0f);
	normalize_and_regularize_tensor(S11_tri, S12_tri, S22_tri, wTri, 3.0f);

	float lambda1_5, lambda2_5;
	float2 tangent5;
	decompose_structure_tensor(S11_5, S12_5, S22_5, lambda1_5, lambda2_5, tangent5);

	float lambda1_7, lambda2_7;
	float2 tangent7;
	decompose_structure_tensor(S11_7, S12_7, S22_7, lambda1_7, lambda2_7, tangent7);

	float lambda1_tri, lambda2_tri;
	float2 tangent_tri;
	decompose_structure_tensor(S11_tri, S12_tri, S22_tri, lambda1_tri, lambda2_tri, tangent_tri);

	float adaptedKnee = RELATIVE_KNEE * max(centerLum, 0.02f);

	float meanGradMag_5   = sqrt(max(lambda1_5   + lambda2_5,   0.0f));
	float meanGradMag_7   = sqrt(max(lambda1_7   + lambda2_7,   0.0f));
	float meanGradMag_tri = sqrt(max(lambda1_tri + lambda2_tri, 0.0f));

	float strength5     = meanGradMag_5   / (meanGradMag_5   + adaptedKnee);
	float strength7     = meanGradMag_7   / (meanGradMag_7   + adaptedKnee);
	float strength_tri  = meanGradMag_tri / (meanGradMag_tri + adaptedKnee);

	float coherence_7   = (lambda1_7   - lambda2_7)   / max(lambda1_7   + lambda2_7,   1e-8f);
	float coherence_tri = (lambda1_tri - lambda2_tri) / max(lambda1_tri + lambda2_tri, 1e-8f);
	float longEdgeFactor = max(coherence_7, coherence_tri * 0.85f);

	float angle7 = atan2(tangent7.y, tangent7.x);

	float baseAngleQuality = compute_angle_quality(tangent7, angle7);
	float triAngleQuality  = compute_triangular_angle_quality(lambda1_tri, lambda2_tri, tangent_tri);
	float angleQuality     = lerp(baseAngleQuality, triAngleQuality, TRIANGULAR_WEIGHT);

	strength5    *= AA_STRENGTH;
	strength7    *= AA_STRENGTH;
	strength_tri *= AA_STRENGTH;

	out0 = float4(strength5, strength7, angleQuality, consensus);
	out1 = float4(tangent5.x, tangent5.y, tangent7.x, tangent7.y);
	out2 = float4(strength_tri, lambda1_tri, lambda2_tri, coherence_tri);
	out3 = float4(tangent_tri.x, tangent_tri.y, longEdgeFactor, 0.0f);
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
// Anti-Aliasing Pass — along-tangent taps only, no fans.
// =============================================================================
float4 PS_ApplyAA(float4 pos : SV_Position, float2 tex : TEXCOORD) : SV_Target
{
	float4 edgeData0 = tex2D(EdgeDataSampler0, tex);
	float consensus = edgeData0.a;

	// ---- DEBUG VIEW ----
	if (Debug) {
		float s5   = edgeData0.r;
		float s7   = edgeData0.g;
		float sTri = tex2D(EdgeDataSampler2, tex).r;
		float3 dbg = float3(s5, sTri, s7);
		return float4(saturate(dbg), 1.0f);
	}

	float3 centerColor = tex2D(ColorSampler, tex).rgb;

	if (consensus < 0.05f) {
		return float4(centerColor, 1.0f);
	}

	float4 edgeData1 = tex2D(EdgeDataSampler1, tex);
	float4 edgeData2 = tex2D(EdgeDataSampler2, tex);
	float4 edgeData3 = tex2D(EdgeDataSampler3, tex);

	float strength5 = edgeData0.r;
	float strength7 = edgeData0.g;
	float angleQuality = edgeData0.b;

	float2 tangent5 = edgeData1.rg;
	float2 tangent7 = edgeData1.ba;
	float strength_tri = edgeData2.r;
	float coherence_tri = edgeData2.a;
	float2 tangent_tri = edgeData3.rg;
	float longEdgeFactor = edgeData3.b;

	float blend = compute_adaptive_blend(
		strength5, strength7, strength_tri,
		angleQuality, longEdgeFactor, coherence_tri
	);
	blend *= AA_STRENGTH;

	if (blend < 0.01f)
		return float4(centerColor, 1.0f);

	float safeLongEdgeFactor = max(longEdgeFactor, MIN_COHERENCE);
	float totalStrength = strength5 + strength7 + strength_tri;
	float kernelPref7 = strength7 / max(totalStrength, 1e-6f);
	float boost7 = safeLongEdgeFactor * angleQuality;
	float triBoost = coherence_tri * 0.15f;
	float kernelMix = lerp(kernelPref7, 1.0f, boost7 + triBoost);
	kernelMix = smoothstep(0.15f, 0.65f, kernelMix);

	float obliqueBias = saturate((0.7f - angleQuality) * 2.0f)
	                  * saturate((0.5f - coherence_tri) * 2.0f);
	kernelMix = lerp(kernelMix, 0.0f, obliqueBias * OBLIQUE_5X5_BIAS);

	float triAngleQuality = lerp(0.85f, 1.0f, coherence_tri);
	float angleScale = lerp(triAngleQuality, angleQuality, 0.5f);
	float longScale = 1.0f + safeLongEdgeFactor * 0.45f;

	float2 step5   = tangent5   * (float2)RT_METRICS.xy * (NORMALIZED_DIST / 2.5f) * angleScale * longScale;
	float2 step7   = tangent7   * (float2)RT_METRICS.xy * (NORMALIZED_DIST / 3.8f) * angleScale * longScale;
	float2 step_tri= tangent_tri* (float2)RT_METRICS.xy * (NORMALIZED_DIST / 3.2f) * angleScale * longScale;

	float triWeight = TRIANGULAR_WEIGHT * TRI_KERNEL_PRESENCE;
	float initWeight = lerp(lerp(16.0f, 32.0f, kernelMix), 24.0f, triWeight);

	float3 acc = centerColor * initWeight;
	float totalWeight = initWeight;

	float dirAlignMin = lerp(OBLIQUE_DIRMIN_FLOOR, DIRECTION_ALIGNMENT_MIN, angleQuality);
	float dirAlignMax = dirAlignMin + OBLIQUE_DIRMAX_WIDTH;

	float kw5_1 = 9.0f * (1.0f - kernelMix);
	float kw5_2 = 2.0f * (1.0f - kernelMix);

	float kw7_1 = 24.0f * kernelMix;
	float kw7_2 = 12.0f * kernelMix;
	float kw7_3 = 3.0f  * kernelMix;

	float kwTri_1 = 18.0f * triWeight;
	float kwTri_2 = 8.0f  * triWeight;
	float kwTri_3 = 2.0f  * triWeight;

	// ========== 5x5 Kernel ==========
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

	// ========== 7x7 Kernel ==========
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

	// ========== Triangular Kernel ==========
	if (strength_tri > 0.01f)
	{
		[unroll]
		for (int i = -3; i <= 3; i++)
		{
			if (i == 0) continue;

			float idx = (float)i;
			float wEdge = (abs(i) == 3) ? kwTri_3 : ((abs(i) == 2) ? kwTri_2 : kwTri_1);

			float2 uv = clamp(tex + step_tri * idx, 0.0f, 1.0f);
			float3 color = tex2Dlod(ColorSampler, float4(uv, 0.0f, 0.0f)).rgb;
			float4 se2 = tex2Dlod(EdgeDataSampler2, float4(uv, 0.0f, 0.0f));
			float4 se3 = tex2Dlod(EdgeDataSampler3, float4(uv, 0.0f, 0.0f));
			float2 dir = se3.rg;
			float edge = se2.r;

			float dirWeight = smoothstep(dirAlignMin, dirAlignMax, dot(tangent_tri, dir));
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
