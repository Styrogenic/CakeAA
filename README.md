# Cake AA — Continuous Adaptive Kerneling Eigen Anti-Aliasing

A ReShade post-process anti-aliasing shader in HLSL implementing a multi-tensor, edge-directed AA filter. It detects edges, builds local gradient tensors with different footprints, then blurs **along** edges — never across them — with per-pixel strength and kernel selection driven by edge confidence.

- **Author:** Alex Aaron Goven
- **Copyright © 2026**
- **License:** GNU General Public License v3

---

## Overview

Cake AA is a two-pass filter:

1. **EdgeDetection** — writes RGBA16F render targets containing tensor strengths, tangents, coherence, and angle quality.
2. **AntiAliasing** — reads those targets plus the backbuffer and blends in an edge-directed average.

The filter never samples perpendicular to the local edge, so it smooths along edges rather than across them.

Scale invariance in the shipped shader is carried by two mechanisms, both downstream of detection. First, `compute_all_tensors` normalizes each gradient sum by its kernel weight total before returning, so tensor components are weight-sum-invariant. Second, `normalize_and_regularize_tensor` divides by `gradientScale` squared (2.0 for the 5×5, 3.0 for the 7×7), and the edge strength uses a constant, luminance-independent knee (`ABSOLUTE_KNEE = RELATIVE_KNEE = 0.005`), so a given absolute gradient magnitude produces a comparable strength regardless of where the pixel sits in the tonal range. As of REV 3 the knee no longer tracks center luminance; the detector treats near-white and near-black pixels identically.

Detection itself — the consensus gates and `edge_stop_weight` — uses absolute per-channel difference, not relative. The constant knee applies after detection, in the strength and blend, not in the gates. This matters for tuning: the gate constants are in absolute units.

---

## Pipeline

### Pass 1 — Edge Detection (PS_DetectEdges)

**Step 1: Consensus pre-filter (`fast_edge_consensus`)**

A cheap three-gate AND detector decides whether a pixel is worth the expensive tensor math. If consensus is below 0.1, all outputs are zeroed and the pixel exits early. All three gates measure the maximum per-channel absolute color difference across RGB.

| Gate | Signal | Notes |
|------|--------|-------|
| 1 | Distance-2 color jump | Max per-channel absolute color difference between center and its eight distance-2 neighbors (4 axial + 4 diagonal) — chroma-aware |
| 2 | 1px axial color jump | Max per-channel absolute color difference between center and its four axial 1px neighbors — chroma-aware |
| 3 | 1px diagonal color jump | Same for the four diagonal 1px neighbors |

Consensus combines the three gates with a soft geometric-mean AND — `pow(edge1 * edge2 * edge3, 1/3)` — so all three must agree while a triple-weak-but-present signal still produces a usable value. It is smoothstepped around the consensus threshold and scaled by a smoothstep on the max of the three to suppress weak edges. This kills flat regions and isolated single-pixel noise before the tensor loops run.

The REV 4 geometric mean replaces the earlier hard product, which collapsed on low-contrast edges because it required all three gates to be independently strong at the same time. The geometric mean preserves moderate agreement across all three gates, so sub-10% delta edges that were previously falling through the AND now pass. The min-edge gate on `max(edge1, edge2, edge3)` still rejects uniform areas.

Gate 1's distance-2 footprint decorrelates it from the two 1px jump gates, so the AND remains discriminating rather than collapsing into a single correlated test. Gates 2 and 3 are distance-normalized to Gate 1's color-delta scale via `JUMP_GAIN_EDGE` (default 6.0, raised from 3.0 in REV 4), so 1px jumps register equally against the distance-2 baseline instead of being systematically under-weighted.

**Step 2: Structure tensors (`compute_all_tensors`)**

| Tensor | Footprint | Side length | Weights | Notes |
|--------|-----------|-------------|---------|-------|
| 5×5 | Rotated 45° grid, coords (col−row)/√2 and (col+row)/√2 | 5 taps | Iso-diamond radial profile `{1.0, 0.55, 0.10}` indexed by `r = max(\|row\|,\|col\|)` | Rotational symmetry on the diamond — no rotated-square corner bias |
| 7×7 | Axis-aligned square | 7 taps (1.4× the 5×5) | Separable binomial `1, 3, 6, 7, 6, 3, 1` | Genuine separable grid |

The 7×7 tensor's side length is 1.4× the 5×5's — a 7:5 ratio — giving it proportionally larger edge-sampling reach. The 5×5 samples on a rotated 45° grid, so it resolves diagonal structure densely and axis-aligned structure coarsely; the 7×7 samples on an axis-aligned square grid and does the reverse. The two are complementary.

Each tensor accumulates gx and gy (3-channel gradient sums) and a weight total, then normalizes gx and gy internally by that weight sum before returning. The function yields only the normalized gradient sums; weight totals are not passed out. Tensor components are S11 = gx·gx, S12 = gx·gy, S22 = gy·gy, computed on the normalized gradients.

The 5×5 weight profile is an iso-diamond radial profile, so the tensor's support is rotationally symmetric about the pixel on the diamond lattice. This removes a structural oblique bias that the oblique-tuning constants would otherwise have to compensate for; those constants are now a refinement rather than a correction.

**Step 3: Normalize and regularize**

`normalize_and_regularize_tensor` adds epsilon to the diagonal and divides by `gradientScale` squared (2.0 for the 5×5, 3.0 for the 7×7). Weight-sum normalization already happened in Step 2, so this step only handles epsilon and gradient-scale invariance. The result is tensors that respond to relative structure rather than absolute brightness, suited to HDR and game content, and functional as subpixel AA.

**Step 4: Eigendecomposition**

`decompose_structure_tensor` extracts:

- λ1 — largest eigenvalue, gradient energy **across** the edge
- λ2 — smallest eigenvalue, gradient energy **along** the edge
- tangent — eigenvector for λ2, the direction to blur along

Derived quantities:

| Quantity | Formula | Meaning |
|----------|---------|---------|
| meanGradMag | √(λ1 + λ2) | Gradient magnitude |
| strength | meanGradMag / (meanGradMag + knee) | Edge strength, with constant knee = ABSOLUTE_KNEE = 0.005 |
| coherence | (λ1 − λ2) / (λ1 + λ2) | Edge-like vs. corner-like |
| longEdgeFactor | coherence_7 | Confidence this is a long edge rather than a corner |

The knee is a constant as of REV 3. Earlier revisions used `RELATIVE_KNEE × max(centerLum, 0.02)`, which made the detector more sensitive near black than near white; that luminance dependence is gone, and `LUM_WEIGHTS` is no longer used by the strength path.

**Step 5: Angle quality**

`compute_angle_quality` rewards tangents aligned to image axes or 45° diagonals, boosting oblique angles toward 1.0.

**Outputs**

| Target | R | G | B | A |
|--------|---|---|---|---|
| EdgeDataTex0 | strength5 | strength7 | angleQuality | consensus |
| EdgeDataTex1 | tangent5.x | tangent5.y | tangent7.x | tangent7.y |
| EdgeDataTex2 | coherence_7 | 0 | 0 | 0 |
| EdgeDataTex3 | 0 | 0 | 0 | 0 |

### Pass 2 — Anti-Aliasing (PS_ApplyAA)

**Debug view:** if Debug is on, output is strength5 in R and strength7 in B, visualizing which tensor fires where. Overlaps sum additively toward magenta on a black background.

**Adaptive blend strength (`compute_adaptive_blend`)**

Combines the two strengths via RMS, coherence_7, and angleQuality into a per-pixel blend amount. Stronger edges produce more blend; high coherence produces a coherenceBoost.

**Corner damping** — `structureConfidence = lerp(CORNER_CONFIDENCE_FLOOR, 1.0, coherence)` multiplies strengthQuality. The floor sets the confidence multiplier at zero coherence, scaling up with coherence, so corner-like pixels get reduced overall blend, preventing corner smearing and junction blow-outs. At the default `CORNER_CONFIDENCE_FLOOR = 1.0` the damping is a no-op; lower the floor (e.g. 0.85) to activate it. This is complementary to the per-tap dirWeight gate below.

Very weak edges (edgeStrength < 0.1) are pulled toward 0.1 and very strong edges (> 0.8) toward 0.9, to avoid over-blur and under-blur at the extremes.

**Stability damping (`apply_stability_damping`)**

Before the blend is scaled by AA_STRENGTH, it is soft-pulled toward 0.5 on pixels where the local estimate is unreliable. Reliability is `clamp(consensus * lerp(0.5, 1.0, coherence), 0, 1)`; damp is `lerp(STABILITY_FLOOR, 1.0, reliability)`; blend becomes `lerp(0.5, blend, damp)`. Reliable pixels (`reliability ≈ 1`) keep their full blend. Unreliable pixels (`reliability ≈ 0`) keep only `STABILITY_FLOOR` of their blend delta, biased toward the midpoint. This is the local, history-free analogue of temporal hysteresis: it damps the per-pixel blend on noisy estimates without requiring a history buffer, reprojection, or disocclusion handling.

**Kernel mix**

`kernelPref7 = strength7 / (strength5 + strength7)`, boosted by longEdgeFactor × angleQuality, then smoothstepped over `[KERNEL_MIX_LO, KERNEL_MIX_HI]` (default `[0.10, 0.75]`). Confident long edges get the long 7×7 kernel; short and fine features get the 5×5.

The transition range is deliberately wide. A narrow ramp on a noisy quantity like `strength7/(strength5+strength7)` causes per-pixel flicker near the crossover; widening it trades a little selectivity for a much smoother derivative of kernelMix with respect to the underlying strengths, so the mixed kernel evolves continuously instead of snapping.

Because both kernelPref7 and the boost rise with coherence and angle quality, the 7×7 gets more attention as an edge becomes longer and more coherent; the 5×5 is the sharper, diagonal fallback for short and fine features. An oblique bias additionally pulls kernelMix toward the 5×5 when angleQuality < 0.7 and coherence_7 < 0.5, using the rotated diagonal lattice for its geometric purpose.

**Along-tangent taps**

For each kernel, six non-center taps are taken along the tangent only (offsets −3, −2, −1, +1, +2, +3), never across it. The step is tangent × texelSize × NORMALIZED_DIST / scale × angleScale × longScale:

| Kernel | Scale |
|--------|-------|
| 5×5 | 2.5 |
| 7×7 | 3.8 |

longScale = 1 + longEdgeFactor × 0.45, so coherent long edges sample farther out. angleQuality scales the step, so oblique edges step more cautiously. The differing per-kernel scale constants compensate for the different footprint sizes so the physical sampling distances stay comparable.

**Per-tap weighting — three gates**

Each tap weight is kernelWeight × dirWeight × stopWeight.

| Gate | Formula | Purpose |
|------|---------|---------|
| Kernel weight | Radial falloff: 9/2 for 5×5, 24/12/3 for 7×7; scaled by kernelMix | Tap falloff |
| dirWeight | smoothstep(dirAlignMin, dirAlignMax, dot(centerTangent, sampleTangent)) | Down-weights neighbors whose tangent points differently — the primary guard against bleeding across edges and corners |
| stopWeight | 1 / (1 + 2x²) where x = max3(abs(sample − center)) × COLOR_SENSITIVITY, relaxed by sampleEdge | Suppresses taps with large absolute color differences; relaxation prevents halos |

`stopWeight` uses the same absolute per-channel difference as the consensus gates. The guard against near-white boundaries comes from the constant knee in the strength, not from the stop gate.

dirWeight guards against degenerate tangents: if the center tangent's squared length is below `1e-6`, `dirAlign` defaults to 1.0 rather than being driven by a meaningless dot product. Same guard for both kernels.

dirAlignMin is relaxed toward 0.55 on oblique edges (`lerp(0.55, 0.7, angleQuality)`) to admit more oblique-aligned neighbors without opening the gate on corners.

**Final blend**

The center is preloaded with a large initial weight of 16 to 32 (by kernelMix), keeping the result near the original unless neighbors strongly agree. `finalColor = acc / totalWeight`, output = `lerp(centerColor, finalColor, blend)`. The final blend is clamped to `[0, 1]` by the stability damping and the scaling step, so `AA_STRENGTH` values above 1 remain safe.

---

## Key Design Choices

**Scale invariance.** The tensors normalize by weight sum (in Step 2) and gradient scale (in Step 3), and the edge strength uses a constant, luminance-independent knee. Combined with sub-texel sampling steps, this makes Cake AA functional as subpixel AA and lets the strength and blend respond proportionally to structure at any brightness level. Detection (the consensus gates and `edge_stop_weight`) is absolute; the constant knee applies downstream of it.

**Two complementary kernels.** Rotated 5×5 and axis-aligned 7×7. The 7×7's side length is 1.4× the 5×5's, giving it proportionally larger reach; the 5×5's rotated grid resolves diagonals that the 7×7 resolves poorly, and vice versa. The 5×5 uses an iso-diamond radial weight profile, so its support is rotationally symmetric on the diamond lattice rather than rotated-square biased.

**Consensus gating.** The cheap three-gate soft-AND detector skips most pixels — including flat regions and noise — before the expensive tensor loops. Gate 1's distance-2 footprint keeps the AND discriminating, and the geometric-mean combination admits low-contrast edges that a hard product would reject.

**Along-tangent-only blur.** The filter never samples perpendicular to the local edge, so it smooths along edges, not across them.

**Layered artifact prevention.** Corner damping in the blend strength pulls back overall on low-coherence pixels, per-tap direction alignment rejects disagreeing neighbors, and edge-stop rejects large color jumps. These are three complementary guards. Stability damping is a fourth: it soft-pulls the blend toward 0.5 on pixels whose consensus and coherence are both weak, suppressing sub-pixel crawl from noisy strengths without a history buffer.

**Adaptive everything.** Blend adapts to strength / coherence / angle, kernel mix adapts to which tensor dominates, and step size adapts to coherence and angle quality. The result is aggressive on clean, long, axis-aligned edges and conservative on corners, fine detail, and oblique boundaries. The knee is fixed by design (REV 3) so that tonal position does not bias sensitivity.

---

## Oblique-Aware Tuning

The default parameters include several oblique-specific adjustments:

| Parameter | Default | Effect |
|-----------|---------|--------|
| OBLIQUE_BLEND_FLOOR | 0.80 | Oblique edges retain more of their blend instead of being throttled to 0.6× |
| OBLIQUE_DIRMIN_FLOOR | 0.55 | Widens the tangent-agreement window on oblique edges |
| OBLIQUE_5X5_BIAS | 0.60 | Pulls kernelMix toward the rotated 5×5 on oblique edges with moderate coherence |

These stay inside the tensor/blur paradigm — no pattern matching — and are intended to stop the filter from fighting its own oblique machinery. With the iso-diamond 5×5 weights the oblique bias is now a refinement rather than a correction of a structural support bias.

---

## Stacking with Other AA

Cake does not require SMAA or in-game non-temporal AA earlier in the pipeline, but pairing is beneficial in either of two ways:

- **Post-process AA (SMAA preferred over FXAA for sharpness).** SMAA's pattern-based edge reconstruction preserves more detail and introduces less blur than FXAA's single-pass luma blur, so it can cover edge cases Cake would miss — very thin sub-pixel features below Cake's consensus gate, or edges whose chroma signature is too subtle for the jump gates — without eroding sharpness.
- **Information-dense AA (MSAA, SSAA).** A cleaner, less aliased input with fewer spurious high-frequency edges means Cake's consensus pre-filter sees more genuine structure and less single-pixel noise, and its tensors resolve cleaner gradients.

Either way, Cake's along-tangent-only behavior composes well: it refines and extends whatever edge information the earlier stage leaves, rather than fighting it.

---

## Tuning Notes

### Edge sensitivity (gate 1)

`EDGE_SENSITIVITY` (default **4.0**) scales gate 1's distance-2 color jump. Higher values make gate 1 fire on smaller absolute color deltas.

- Conservative: ~2–3
- Default: 4.0
- Aggressive: ~5–8
- Above ~8, gate 1 saturates on most non-flat content and stops discriminating; the detector effectively becomes the two-gate AND

### Consensus floor

`MIN_CONSENSUS_EDGE` (default **0.05**, lowered from 0.10 in REV 4) gates the whole consensus on max(edge1, edge2, edge3). Lowering it admits pixels where all three gates are only moderately strong — the real lever for more sensitivity once gate 1 is near saturation, and the primary knob for pulling in low-contrast edges.

- Conservative: ~0.10
- Default: 0.05
- Aggressive: ~0.03
- Below ~0.02, flat-region noise can start to pass the min-edge gate

### Color sensitivity

`COLOR_SENSITIVITY` (default 7.0) controls `edge_stop_weight`. Higher values suppress taps with larger absolute color deltas more aggressively; lower values make the filter more forgiving of near-edge color differences.

- If grain near edges persists: step down toward 6.5
- If halos appear: step up toward 7.5

### Jump-gate sensitivity

`JUMP_GAIN_EDGE` (default **6.0**, raised from 3.0 in REV 4) scales the axial and diagonal jump gates in pass 1 (edge2 and edge3). The higher default distance-normalizes the 1px gates to Gate 1's distance-2 color-delta scale, so low-contrast edges register across all three gates instead of dying in the AND. Lower values make the gates require a larger color jump before they fire, suppressing grainy detail close to edges at the detection stage.

- Conservative: ~4
- Default: 6.0
- Aggressive: ~7–8
- If grain near edges persists: step down toward 4
- If faint thin edges start to disappear: step up toward 8

### Kernel-mix transition

`KERNEL_MIX_LO` (default 0.10) and `KERNEL_MIX_HI` (default 0.75) set the smoothstep bounds on the kernelMix quantity. A narrower range increases selectivity between the two kernels; a wider range smooths the crossover and reduces per-pixel flicker at the boundary.

- If 5×5/7×7 selection looks sluggish: narrow toward `[0.15, 0.65]`
- If flicker persists at the crossover: widen toward `[0.05, 0.85]`

### Stability damping

`STABILITY_FLOOR` (default 0.35) sets how much of the blend delta survives on the least-reliable pixels. Lower values damp more aggressively toward the midpoint (more AA on noisy edges); higher values leave more of the original blend intact.

- More damping: ~0.20
- Default: 0.35
- Less damping: ~0.50
- At 1.0 the stage is a no-op

### Corner damping

`CORNER_CONFIDENCE_FLOOR` controls the corner-damping mechanism. The floor sets the confidence multiplier at zero coherence; it scales up to 1.0 with coherence. Lower values damp corners more aggressively; higher values reduce the damping. At the default 1.0 the mechanism is disabled (multiplier is always 1.0); set to ~0.85 to activate.

---

## File Structure

CakeAA.fx

- `PostProcessVS` — fullscreen vertex shader
- UI Controls — Debug toggle
- Parameters — tuning constants
- Textures / Samplers — backbuffer + four edge data targets
- Utility functions — `max3`, kernel tables (`DIAMOND_W`, `KERNEL_7x7`), DEG2RAD
- `fast_edge_consensus` — three-gate soft-AND detector, absolute per-channel difference
- `compute_all_tensors` — fused 5×5 (iso-diamond) and 7×7 (separable binomial) tensor bodies; normalizes internally, returns normalized gradients only
- `normalize_and_regularize_tensor` — epsilon + gradient-scale normalization (weight-sum normalization happens in `compute_all_tensors`)
- `decompose_structure_tensor`
- `compute_angle_quality`
- `compute_adaptive_blend`
- `apply_stability_damping` — local, history-free blend damping on unreliable pixels
- `PS_DetectEdges` — pass 1
- `edge_stop_weight` — absolute per-channel color-stop gate for AA taps
- `PS_ApplyAA` — pass 2
- `technique CakeAA` — EdgeDetection + AntiAliasing passes

---

## Requirements

- ReShade with HLSL support
- A GPU supporting RGBA16F render targets and Shader Model 3.0+

---

## Debug View

Enable the Debug checkbox in the UI. The output shows:

- **R** = 5×5 strength
- **B** = 7×7 strength
- **Additive** — overlaps show as magenta

This visualizes which tensor fires where, useful for understanding why a particular edge gets a particular blend.

---

## License

GNU General Public License v3 — Copyright © 2026 Alex Aaron Goven.
