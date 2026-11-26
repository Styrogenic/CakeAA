//================================================================================================================================
// Cake AA — SCALE-INVARIANT, LOW-CONTRAST RESCUE (REV 3, tightened)
// Copyright © 2026 Alex Aaron Goven — GPLv3
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

uniform bool Debug <
	label = "Debug";
	description = "R=5x5, B=7x7 additive on black";
> = false;

// ================================ Parameters ================================
static const float AA_STRENGTH          = 1.0f;
static const float RELATIVE_KNEE        = 0.005f;
static const float NORMALIZED_DIST      = 3.5f;
static const float COLOR_SENSITIVITY    = 7.0f;
static const float DIRECTION_ALIGNMENT_MIN = 0.7f;
static const float MIN_COHERENCE        = 0.08f;

// REV 3 consensus (geometric mean scale)
static const float CONSENSUS_LO         = 0.03f;
static const float CONSENSUS_HI         = 0.10f;
static const float EDGE_SENSITIVITY     = 4.0f;
static const float JUMP_GAIN_EDGE       = 3.0f;
static const float MIN_CONSENSUS_EDGE_LO = 0.02f;
static const float MIN_CONSENSUS_EDGE_HI = 0.06f;
static const float GAIN_LUMA_BASE       = 0.30f;
static const float GAIN_MAX             = 6.0f;
static const float EARLYOUT_DETECT      = 0.02f;
static const float EARLYOUT_APPLY       = 0.015f;

static const float QUALITY_TARGET_HIGH  = 0.98f;
static const float QUALITY_TARGET_LOW   = 0.65f;
static const float QUALITY_ADAPT_RATE   = 0.7f;
static const float QUALITY_BOOST_STRONG = 0.15f;
static const float QUALITY_CLAMP_MIN    = 0.4f;
static const float QUALITY_CLAMP_MAX    = 0.98f;
static const float CORNER_CONFIDENCE_FLOOR = 1.0f;

static const float OBLIQUE_BLEND_FLOOR  = 0.80f;
static const float OBLIQUE_DIRMIN_FLOOR = 0.55f;
static const float OBLIQUE_DIRMAX_WIDTH = 0.30f;
static const float OBLIQUE_5X5_BIAS     = 0.60f;

static const float KERNEL_MIX_LO        = 0.10f;
static const float KERNEL_MIX_HI        = 0.75f;
static const float STABILITY_FLOOR      = 0.35f;

#define RT_METRICS float4(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT, BUFFER_WIDTH, BUFFER_HEIGHT)
#define LUM_WEIGHTS float3(0.2126f, 0.7152f, 0.0722f)
#define INV_SQRT2 0.7071067811865475f

texture BackBufferTex : COLOR;

texture EdgeDataTex0 { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
texture EdgeDataTex1 { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
texture EdgeDataTex2 { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
texture EdgeDataTex3 { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };

sampler ColorSampler    { Texture = BackBufferTex; MinFilter = Linear; MagFilter = Linear; AddressU = Clamp; AddressV = Clamp; };
sampler EdgeDataSampler0{ Texture = EdgeDataTex0;  MinFilter = Point;  MagFilter = Point; };
sampler EdgeDataSampler1{ Texture = EdgeDataTex1;  MinFilter = Point;  MagFilter = Point; };
sampler EdgeDataSampler2{ Texture = EdgeDataTex2;  MinFilter = Point;  MagFilter = Point; };
sampler EdgeDataSampler3{ Texture = EdgeDataTex3;  MinFilter = Point;  MagFilter = Point; };

// =============================================================================
float max3(float3 v) { return max(v.x, max(v.y, v.z)); }
float min3(float3 v) { return min(v.x, min(v.y, v.z)); }

static const float DEG2RAD = 0.017453292519943295f;
static const float DIAMOND_W[3] = { 1.0f, 0.55f, 0.10f };
static const float KERNEL_7x7[7] = { 1.0f, 3.0f, 6.0f, 7.0f, 6.0f, 3.0f, 1.0f };

// =============================================================================
// Consensus detector (REV 3: geo-mean + contrast-relative gains)
// =============================================================================
float fast_edge_consensus(float2 tex, float2 offset, float3 centerColor)
{
	float3 c[8];
	[unroll]
	for (int i = 0; i < 8; i++) {
		int row = i / 3 - 1;
		int col = i % 3 - 1;
		if (i >= 4) { row = (i + 1) / 3 - 1; col = (i + 1) % 3 - 1; }
		float2 uv = clamp(tex + float2((float)col, (float)row) * offset, 0.0f, 1.0f);
		c[i] = tex2Dlod(ColorSampler, float4(uv, 0.0f, 0.0f)).rgb;
	}
	const float3 cNW = c[0], cN = c[1], cNE = c[2], cW = c[3], cE = c[4], cSW = c[5], cS = c[6], cSE = c[7];

	float3 cN2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( 0.0f, -2.0f) * offset, 0, 1), 0, 0)).rgb;
	float3 cS2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( 0.0f,  2.0f) * offset, 0, 1), 0, 0)).rgb;
	float3 cE2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( 2.0f,  0.0f) * offset, 0, 1), 0, 0)).rgb;
	float3 cW2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2(-2.0f,  0.0f) * offset, 0, 1), 0, 0)).rgb;
	float2 d2off = float2(1.41421356f, 1.41421356f);
	float3 cNW2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2(-d2off.x, -d2off.y) * offset, 0, 1), 0, 0)).rgb;
	float3 cNE2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( d2off.x, -d2off.y) * offset, 0, 1), 0, 0)).rgb;
	float3 cSW2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2(-d2off.x,  d2off.y) * offset, 0, 1), 0, 0)).rgb;
	float3 cSE2 = tex2Dlod(ColorSampler, float4(clamp(tex + float2( d2off.x,  d2off.y) * offset, 0, 1), 0, 0)).rgb;

	// Gate 1: distance-2 color jump
	float d2 = max3(abs(cN2  - centerColor));
	d2 = max(d2, max3(abs(cS2  - centerColor)));
	d2 = max(d2, max3(abs(cE2  - centerColor)));
	d2 = max(d2, max3(abs(cW2  - centerColor)));
	d2 = max(d2, max3(abs(cNW2 - centerColor)));
	d2 = max(d2, max3(abs(cNE2 - centerColor)));
	d2 = max(d2, max3(abs(cSW2 - centerColor)));
	d2 = max(d2, max3(abs(cSE2 - centerColor)));

	// Gate 2: 1px axial
	float axialJump = max(max(max3(abs(cN - centerColor)), max3(abs(cS - centerColor))),
	                      max(max3(abs(cE - centerColor)), max3(abs(cW - centerColor))));

	// Gate 3: 1px diagonal
	float diagJump = max(max(max3(abs(cNW - centerColor)), max3(abs(cNE - centerColor))),
	                     max(max3(abs(cSW - centerColor)), max3(abs(cSE - centerColor))));

	// Contrast-relative gain from 3x3 luma range + center luma
	float lumaC = dot(centerColor, LUM_WEIGHTS);
	float lumaMax9 = max(lumaC, max(max(max(dot(cN, LUM_WEIGHTS), dot(cS, LUM_WEIGHTS)),
	                                       max(dot(cE, LUM_WEIGHTS), dot(cW, LUM_WEIGHTS))),
	                                  max(max(dot(cNW, LUM_WEIGHTS), dot(cNE, LUM_WEIGHTS)),
	                                      max(dot(cSW, LUM_WEIGHTS), dot(cSE, LUM_WEIGHTS)))));
	float lumaMin9 = min(lumaC, min(min(min(dot(cN, LUM_WEIGHTS), dot(cS, LUM_WEIGHTS)),
	                                       min(dot(cE, LUM_WEIGHTS), dot(cW, LUM_WEIGHTS))),
	                                  min(min(dot(cNW, LUM_WEIGHTS), dot(cNE, LUM_WEIGHTS)),
	                                      min(dot(cSW, LUM_WEIGHTS), dot(cSE, LUM_WEIGHTS)))));
	float localRange = lumaMax9 - lumaMin9;

	float gainDenom = GAIN_LUMA_BASE + max(lumaC, 0.0f) + localRange * 0.5f;
	float gainScale = saturate(1.0f / max(gainDenom, 1e-4f) * GAIN_LUMA_BASE);
	gainScale = clamp(gainScale * (1.0f + localRange * 4.0f), 0.5f, GAIN_MAX);

	float e1 = saturate(d2        * EDGE_SENSITIVITY * gainScale);
	float e2 = saturate(axialJump * JUMP_GAIN_EDGE  * gainScale);
	float e3 = saturate(diagJump  * JUMP_GAIN_EDGE  * gainScale);

	// Geo-mean consensus; gate1 carries, gates 2/3 boost
	float geoMean = pow(max(e1 * e2 * e3, 1e-8f), 1.0f / 3.0f);
	float gate23  = max(e2, e3);
	float consensus = geoMean * lerp(0.40f, 1.0f, gate23);
	consensus = smoothstep(CONSENSUS_LO, CONSENSUS_HI, consensus);

	float minEdge = max(e1, gate23);
	float floorEdge = lerp(MIN_CONSENSUS_EDGE_LO, MIN_CONSENSUS_EDGE_HI, saturate(localRange * 4.0f));
	consensus *= smoothstep(floorEdge * 0.5f, floorEdge * 1.5f, minEdge);

	return saturate(consensus);
}

// =============================================================================
void compute_all_tensors(float2 tex, float2 offset, out float3 gx5, out float3 gy5, out float3 gx7, out float3 gy7)
{
	gx5 = 0; gy5 = 0;
	float totalWeight5 = 0.0f;
	[unroll]
	for (int row = -2; row <= 2; row++) {
		[unroll]
		for (int col = -2; col <= 2; col++) {
			float2 so = float2((float)(col - row) * INV_SQRT2, (float)(col + row) * INV_SQRT2);
			float w = DIAMOND_W[max(abs(row), abs(col))];
			float2 uv = clamp(tex + so * offset, 0.0f, 1.0f);
			float3 sc = tex2Dlod(ColorSampler, float4(uv, 0, 0)).rgb;
			gx5 += sc * so.x * w; gy5 += sc * so.y * w;
			totalWeight5 += w;
		}
	}

	gx7 = 0; gy7 = 0;
	float totalWeight7 = 0.0f;
	[unroll]
	for (int dy = -3; dy <= 3; dy++) {
		[unroll]
		for (int dx = -3; dx <= 3; dx++) {
			float2 uv = clamp(tex + float2((float)dx, (float)dy) * offset, 0.0f, 1.0f);
			float w = KERNEL_7x7[dy + 3] * KERNEL_7x7[dx + 3] / 729.0f;
			float3 sc = tex2Dlod(ColorSampler, float4(uv, 0, 0)).rgb;
			gx7 += sc * (float)dx * w; gy7 += sc * (float)dy * w;
			totalWeight7 += w;
		}
	}

	if (totalWeight5 > 0.0f) { float inv = 1.0f / totalWeight5; gx5 *= inv; gy5 *= inv; }
	if (totalWeight7 > 0.0f) { float inv = 1.0f / totalWeight7; gx7 *= inv; gy7 *= inv; }
}

void normalize_and_regularize_tensor(inout float S11, inout float S12, inout float S22, float gradientScale)
{
	const float EPSILON = 1e-6f;
	S11 += EPSILON; S22 += EPSILON;
	if (gradientScale > 0.0f) {
		float sf = 1.0f / (gradientScale * gradientScale);
		S11 *= sf; S12 *= sf; S22 *= sf;
	}
}

void decompose_structure_tensor(float S11, float S12, float S22, out float lambda1, out float lambda2, out float2 tangent)
{
	float trace = S11 + S22;
	float det = S11 * S22 - S12 * S12;
	float disc = sqrt(max(trace * trace - 4.0f * det, 0.0f));
	lambda1 = (trace + disc) * 0.5f;
	lambda2 = (trace - disc) * 0.5f;
	if (disc < 1e-6f) { tangent = float2(1.0f, 0.0f); }
	else {
		tangent = float2(S12, lambda2 - S11);
		float len = length(tangent);
		tangent = (len > 1e-6f) ? tangent / len : float2(1.0f, 0.0f);
	}
}

float compute_angle_quality(float2 tangent, float angle)
{
	float cosA = abs(tangent.x), sinA = abs(tangent.y);
	float maxBasis = max(cosA, sinA);
	float maxDiag = max(abs(cosA + sinA), abs(cosA - sinA)) * INV_SQRT2;
	float primaryQuality = max(maxBasis, maxDiag);
	float nearestAxis = round(angle / (DEG2RAD * 45.0f)) * (DEG2RAD * 45.0f);
	float obliqueness = 1.0f - abs(cos(angle - nearestAxis));
	return lerp(primaryQuality, 1.0f, obliqueness * obliqueness * 0.5f);
}

float compute_adaptive_blend(float strength5, float strength7, float angleQuality, float longEdgeFactor)
{
	float edgeStrength = clamp(sqrt((strength5*strength5 + strength7*strength7) * 0.5f), 0.0f, 1.0f);
	float coherence = clamp(longEdgeFactor, 0.0f, 1.0f);
	float strengthQuality = lerp(QUALITY_TARGET_LOW, QUALITY_TARGET_HIGH, edgeStrength);
	strengthQuality *= lerp(CORNER_CONFIDENCE_FLOOR, 1.0f, coherence);
	float blend = strengthQuality + coherence * QUALITY_BOOST_STRONG;
	blend *= lerp(OBLIQUE_BLEND_FLOOR, 1.0f, angleQuality);
	blend = lerp(blend, 0.5f, 1.0f - QUALITY_ADAPT_RATE);
	blend = clamp(blend, QUALITY_CLAMP_MIN, QUALITY_CLAMP_MAX);
	if (edgeStrength < 0.1f) blend = lerp(blend, 0.1f, 1.0f - edgeStrength * 10.0f);
	if (edgeStrength > 0.8f) blend = lerp(blend, 0.9f, (edgeStrength - 0.8f) * 2.5f);
	return clamp(blend, 0.0f, 1.0f);
}

float apply_stability_damping(float blend, float consensus, float coherence)
{
	float reliability = clamp(consensus * lerp(0.5f, 1.0f, coherence), 0.0f, 1.0f);
	return lerp(0.5f, blend, lerp(STABILITY_FLOOR, 1.0f, reliability));
}

// =============================================================================
void PS_DetectEdges(float4 pos : SV_Position, float2 tex : TEXCOORD,
	out float4 out0 : SV_Target0, out float4 out1 : SV_Target1,
	out float4 out2 : SV_Target2, out float4 out3 : SV_Target3)
{
	float2 offset = RT_METRICS.xy;
	float3 centerColor = tex2D(ColorSampler, tex).rgb;
	float consensus = fast_edge_consensus(tex, offset, centerColor);

	if (consensus < EARLYOUT_DETECT) {
		out0 = out1 = out2 = out3 = float4(0, 0, 0, 0);
		return;
	}

	float3 gx5, gy5, gx7, gy7;
	compute_all_tensors(tex, offset, gx5, gy5, gx7, gy7);

	float S11_5 = dot(gx5, gx5), S12_5 = dot(gx5, gy5), S22_5 = dot(gy5, gy5);
	float S11_7 = dot(gx7, gx7), S12_7 = dot(gx7, gy7), S22_7 = dot(gy7, gy7);

	normalize_and_regularize_tensor(S11_5, S12_5, S22_5, 2.0f);
	normalize_and_regularize_tensor(S11_7, S12_7, S22_7, 3.0f);

	float l1_5, l2_5, l1_7, l2_7;
	float2 tangent5, tangent7;
	decompose_structure_tensor(S11_5, S12_5, S22_5, l1_5, l2_5, tangent5);
	decompose_structure_tensor(S11_7, S12_7, S22_7, l1_7, l2_7, tangent7);

	float adaptedKnee = RELATIVE_KNEE * max(dot(centerColor, LUM_WEIGHTS), 0.02f);
	float mag5 = sqrt(max(l1_5 + l2_5, 0.0f));
	float mag7 = sqrt(max(l1_7 + l2_7, 0.0f));
	float strength5 = mag5 / (mag5 + adaptedKnee);
	float strength7 = mag7 / (mag7 + adaptedKnee);
	float coherence_7 = (l1_7 - l2_7) / max(l1_7 + l2_7, 1e-8f);
	float angle7 = atan2(tangent7.y, tangent7.x);
	float angleQuality = compute_angle_quality(tangent7, angle7);

	strength5 *= AA_STRENGTH;
	strength7 *= AA_STRENGTH;

	out0 = float4(strength5, strength7, angleQuality, consensus);
	out1 = float4(tangent5.x, tangent5.y, tangent7.x, tangent7.y);
	out2 = float4(coherence_7, 0.0f, 0.0f, 0.0f);
	out3 = float4(0, 0, 0, 0);
}

// =============================================================================
float edge_stop_weight(float3 sampleColor, float3 centerColor, float sampleEdge)
{
	float colorDiff = max(abs(sampleColor.x - centerColor.x),
	                     max(abs(sampleColor.y - centerColor.y), abs(sampleColor.z - centerColor.z)));
	float x = colorDiff * COLOR_SENSITIVITY;
	return lerp(1.0f / (1.0f + x * x * 2.0f), 1.0f, sampleEdge);
}

// =============================================================================
float4 PS_ApplyAA(float4 pos : SV_Position, float2 tex : TEXCOORD) : SV_Target
{
	float4 edgeData0 = tex2D(EdgeDataSampler0, tex);
	float consensus = edgeData0.a;

	if (Debug) return float4(saturate(float3(edgeData0.r, 0.0f, edgeData0.g)), 1.0f);

	float3 centerColor = tex2D(ColorSampler, tex).rgb;
	if (consensus < EARLYOUT_APPLY) return float4(centerColor, 1.0f);

	float4 edgeData1 = tex2D(EdgeDataSampler1, tex);
	float4 edgeData2 = tex2D(EdgeDataSampler2, tex);

	float strength5 = edgeData0.r;
	float strength7 = edgeData0.g;
	float angleQuality = edgeData0.b;
	float2 tangent5 = edgeData1.rg;
	float2 tangent7 = edgeData1.ba;
	float coherence_7 = edgeData2.r;

	float blend = compute_adaptive_blend(strength5, strength7, angleQuality, coherence_7);
	blend = apply_stability_damping(blend, consensus, coherence_7) * AA_STRENGTH;
	if (blend < 0.01f) return float4(centerColor, 1.0f);

	float safeLongEdgeFactor = max(coherence_7, MIN_COHERENCE);
	float totalStrength = strength5 + strength7;
	float kernelPref7 = strength7 / max(totalStrength, 1e-6f);
	float kernelMix = lerp(kernelPref7, 1.0f, safeLongEdgeFactor * angleQuality);
	kernelMix = smoothstep(KERNEL_MIX_LO, KERNEL_MIX_HI, kernelMix);

	float obliqueBias = saturate((0.7f - angleQuality) * 2.0f) * saturate((0.5f - coherence_7) * 2.0f);
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

	float kw5_1 =  9.0f * (1.0f - kernelMix);
	float kw5_2 =  2.0f * (1.0f - kernelMix);
	float kw7_1 = 24.0f * kernelMix;
	float kw7_2 = 12.0f * kernelMix;
	float kw7_3 =  3.0f * kernelMix;

	if (strength5 > 0.01f) {
		[unroll]
		for (int i = -3; i <= 3; i++) {
			if (i == 0) continue;
			float idx = (float)i;
			float wEdge = (abs(i) == 3) ? kw5_2 : kw5_1;
			float2 uv = clamp(tex + step5 * idx, 0.0f, 1.0f);
			float3 color = tex2Dlod(ColorSampler, float4(uv, 0, 0)).rgb;
			float4 se0 = tex2Dlod(EdgeDataSampler0, float4(uv, 0, 0));
			float2 dir = tex2Dlod(EdgeDataSampler1, float4(uv, 0, 0)).rg;
			float dirAlign = (dot(tangent5, tangent5) > 1e-6f) ? dot(tangent5, dir) : 1.0f;
			float w = wEdge * smoothstep(dirAlignMin, dirAlignMax, dirAlign)
			         * edge_stop_weight(color, centerColor, max3(se0.rgb));
			acc += color * w; totalWeight += w;
		}
	}

	if (strength7 > 0.01f) {
		[unroll]
		for (int i = -3; i <= 3; i++) {
			if (i == 0) continue;
			float idx = (float)i;
			float wEdge = (abs(i) == 3) ? kw7_3 : ((abs(i) == 2) ? kw7_2 : kw7_1);
			float2 uv = clamp(tex + step7 * idx, 0.0f, 1.0f);
			float3 color = tex2Dlod(ColorSampler, float4(uv, 0, 0)).rgb;
			float4 se0 = tex2Dlod(EdgeDataSampler0, float4(uv, 0, 0));
			float2 dir = tex2Dlod(EdgeDataSampler1, float4(uv, 0, 0)).ba;
			float dirAlign = (dot(tangent7, tangent7) > 1e-6f) ? dot(tangent7, dir) : 1.0f;
			float w = wEdge * smoothstep(dirAlignMin, dirAlignMax, dirAlign)
			         * edge_stop_weight(color, centerColor, max3(se0.rgb));
			acc += color * w; totalWeight += w;
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
