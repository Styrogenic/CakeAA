# Cake AA — Continuous Adaptive Kerneling Eigen Anti-Aliasing

A ReShade post-process anti-aliasing shader in HLSL implementing a multi-tensor, edge-directed AA filter.

It detects edges, builds three local gradient tensors with different footprints, then blurs **along** edges — never across them — with per-pixel strength and kernel selection driven by edge confidence.

- **Author:** Alex Aaron Goven
- **Copyright © 2026**
- **License:** GNU General Public License v3

---

## Overview

Cake AA is a two-pass filter:

1. **EdgeDetection** — writes four RGBA16F render targets containing tensor strengths, tangents, coherence, and angle quality.
2. **AntiAliasing** — reads those targets plus the backbuffer and blends in an edge-directed average.

The filter never samples perpendicular to the local edge, so it smooths along edges rather than across them.

---

## Pipeline

### Pass 1 — Edge Detection (PS_DetectEdges)

**Step 1: Consensus pre-filter (fast_edge_consensus)**

A cheap three-gate AND detector decides whether a pixel is worth the expensive tensor math. If consensus is below 0.1, all four outputs are zeroed and the pixel exits early.

| Gate | Signal | Notes |
|------|--------|-------|
| 1 | Sobel magnitude | Per-channel Sobel on the 3×3 neighborhood, plus a luma term and a sparse 2px outer-ring gradient (N/S/E/W at distance 2) |
| 2 | Axial jump | Max per-channel color difference between center and its four axial 1px neighbors — chroma-aware |
| 3 | Diagonal jump | Same for the four diagonal 1px neighbors |

Consensus = edge1 × edge2 × edge3, so all three must agree. It is smoothstepped around a cubed threshold and scaled by a smoothstep on the max of the three to suppress weak edges. This kills flat regions and isolated single-pixel noise before the 25 + 49 + 12 tap tensor loops run.

**Step 2: Three structure tensors (compute_all_tensors)**

| Tensor | Footprint | Lattice | Weights | Outer product values |
|--------|-----------|---------|---------|----------------------|
| 5×5 | Rotated 45° grid, coords (col−row)/√2 and (col+row)/√2 | Diagonal basis; spatial tap spacing 1 px | Separable binomial 1, 4, 6, 4, 1 | From 11^4 |
| 7×7 | Axis-aligned square | Square basis; spatial tap spacing 1 px | Separable 1, 3, 6, 7, 6, 3, 1 | From 111^3 |
| Triangular | Six hex-ring directions at radius 3.46 (weight 0.45), six ring-2 directions at radius 6 (weight 0.15), center (weight 0.1) | Hexagonal lattice | Radial differences from center color | — |

Both the 5×5 and the 7×7 sample at a **spatial tap spacing of 1 px** — a unit step along a rotated axis of the 5×5 maps to a screen displacement of (1/√2, 1/√2), whose length is 1 px, the same as a unit step along an axis of the 7×7.

The √2 is an **axis ratio between the two kernels' coordinate frames**, not a spatial distance. The 5×5's rotated coordinate axis spans √2 per unit step (because the transform divides by √2), while the 7×7's square coordinate axis spans 1 per unit step. That axis ratio of √2, combined with the 5×5 spanning ±2 rotated steps and the 7×7 spanning ±3 square steps, is what makes the 5×5's corner taps land at 2√2 ≈ 2.83 px along the screen axes and the 7×7's side taps land at 3 px — a ratio of 3/(2√2) ≈ 1.06.

So the two kernels have comparable spatial reach on complementary lattices: the rotated 5×5 resolves 45° structure densely and axis-aligned structure coarsely, while the 7×7 does the reverse. The triangular lattice fills in what the other two lack, catching 30°/60° directions the square grids miss.

Each tensor yields gx and gy (3-channel gradient sums) and a weight total. Tensor components are S11 = gx·gx, S12 = gx·gy, S22 = gy·gy.

**Step 3: Normalize and regularize**

normalize_and_regularize_tensor divides by weight sum, adds epsilon to the diagonal, and divides by gradientScale² (2.0, 3.0, 3.0). This makes the tensors scale-invariant, so they respond to relative structure rather than absolute brightness — suited to HDR and game content, and functional as subpixel AA.

**Step 4: Eigendecomposition**

decompose_structure_tensor extracts:

- λ1 — largest eigenvalue, gradient energy **across** the edge
- λ2 — smallest eigenvalue, gradient energy **along** the edge
- tangent — eigenvector for λ2, the direction to blur along

Derived quantities:

| Quantity | Formula | Meaning |
|----------|---------|---------|
| meanGradMag | √(λ1 + λ2) | Gradient magnitude |
| strength | meanGradMag / (meanGradMag + knee) | Edge strength, with adaptive knee = RELATIVE_KNEE × max(centerLum, 0.02) |
| coherence | (λ1 − λ2) / (λ1 + λ2) | Edge-like vs. corner-like |
| longEdgeFactor | max(coherence_7, coherence_tri × 0.85) | Confidence this is a long edge rather than a corner |

**Step 5: Angle quality**

- compute_angle_quality rewards tangents aligned to image axes or 45° diagonals, boosting oblique angles toward 1.0.
- compute_triangular_angle_quality rewards alignment to 30°/60° hexagonal axes, blended by coherence.

The two are mixed by TRIANGULAR_WEIGHT = 0.45 into a single angleQuality.

**Outputs**

| Target | R | G | B | A |
|--------|---|---|---|---|
| EdgeDataTex0 | strength5 | strength7 | angleQuality | consensus |
| EdgeDataTex1 | tangent5.x | tangent5.y | tangent7.x | tangent7.y |
| EdgeDataTex2 | strength_tri | λ1_tri | λ2_tri | coherence_tri |
| EdgeDataTex3 | tangent_tri.x | tangent_tri.y | longEdgeFactor | 0 |

### Pass 2 — Anti-Aliasing (PS_ApplyAA)

**Debug view:** if Debug is on, output is strength5 / strength_tri / strength7 in R / G / B, visualizing which tensor fires where.

**Adaptive blend strength (compute_adaptive_blend)**

Combines the three strengths via RMS, the max of longEdgeFactor and coherence_tri, and angleQuality into a per-pixel blend amount. Stronger edges produce more blend; high coherence produces a coherenceBoost.

**Corner damping** — structureConfidence = lerp(CORNER_CONFIDENCE_FLOOR, 1.0, coherence) multiplies strengthQuality. The floor sets the confidence multiplier at zero coherence, scaling up with coherence, so corner-like pixels get reduced overall blend, preventing corner smearing and junction blow-outs. **This build ships with CORNER_CONFIDENCE_FLOOR = 1.0, which neutralizes the mechanism at the default; lowering the floor re-enables it.**

Very weak edges (edgeStrength < 0.1) are pulled toward 0.1 and very strong edges (> 0.8) toward 0.9, to avoid over-blur and under-blur at the extremes.

**Kernel mix**

kernelPref7 = strength7 / (strength5 + strength7 + strength_tri), boosted by longEdgeFactor × angleQuality and coherence_tri × 0.15, then smoothstepped. Confident long edges get the long 7×7 kernel; short and fine features get the 5×5.

Because both kernelPref7 and the boost rise with coherence and angle quality, the 7×7 gets more attention as an edge becomes longer and more coherent; the 5×5 is the sharper, diagonal fallback for short and fine features. An oblique bias additionally pulls kernelMix toward the 5×5 when angleQuality < 0.7 and coherence_tri < 0.5, using the rotated diagonal lattice for its geometric purpose.

**Along-tangent taps**

For each of the three kernels, six non-center taps are taken along the tangent only (offsets −3, −2, −1, +1, +2, +3), never across it. The step is tangent × texelSize × NORMALIZED_DIST / scale × angleScale × longScale:

| Kernel | Scale | Notes |
|--------|-------|-------|
| 5×5 | 2.5 | |
| 7×7 | 3.8 | |
| Triangular | 3.2 | |

longScale = 1 + longEdgeFactor × 0.45, so coherent long edges sample farther out. angleQuality scales the step, so oblique edges step more cautiously. The differing per-kernel scale constants compensate for the different coordinate-axis ratios so the physical sampling distances stay comparable.

**Per-tap weighting — three gates**

Each tap weight is kernelWeight × dirWeight × stopWeight.

| Gate | Formula | Purpose |
|------|---------|---------|
| Kernel weight | Radial falloff: 9/2 for 5×5, 24/12/3 for 7×7, 18/8/2 for triangular; scaled by kernelMix or triWeight | Tap falloff |
| dirWeight | smoothstep(dirAlignMin, dirAlignMax, dot(centerTangent, sampleTangent)) | Down-weights neighbors whose tangent points differently — the primary guard against bleeding across edges and corners |
| stopWeight | 1 / (1 + 2x²) where x = colorDiff × COLOR_SENSITIVITY, relaxed by sampleEdge | Suppresses taps with large color differences; relaxation prevents halos |

dirAlignMin is relaxed toward 0.55 on oblique edges (lerp(0.55, 0.7, angleQuality)) to admit more oblique-aligned neighbors without opening the gate on corners.

**Final blend**

The center is preloaded with a large initial weight of 16 to 32 (by kernelMix), 24 at full tri weight, keeping the result near the original unless neighbors strongly agree. finalColor = acc / totalWeight, output = lerp(centerColor, finalColor, blend).

---

## Key Design Choices

**Scale invariance.** All three tensors normalize by weight sum and gradient scale, so they respond to structure at any contrast level. Combined with sub-texel sampling steps, this makes Cake AA functional as subpixel AA.

**Three complementary kernels.** Rotated 5×5, axis-aligned 7×7, and a hexagonal triangular lattice. The 5×5 and 7×7 differ by basis orientation — a √2 axis ratio between their coordinate frames, both at 1 px spatial tap spacing — giving comparable edge-sampling reach (5×5 corners at 2√2 ≈ 2.83 px vs. 7×7 sides at 3 px, ratio ≈ 1.06) on complementary lattices. The triangular kernel fills in 30°/60° directions the square grids miss and feeds coherence_tri into the mix.

**Consensus gating.** The cheap three-gate AND detector skips most pixels — including flat regions and noise — before the expensive tensor loops.

**Along-tangent-only blur.** The filter never samples perpendicular to the local edge, so it smooths along edges, not across them.

**Layered artifact prevention.** Corner damping in the blend strength pulls back overall on low-coherence pixels, per-tap direction alignment rejects disagreeing neighbors, and edge-stop rejects large color jumps. Corner damping is neutralized at the default CORNER_CONFIDENCE_FLOOR of 1.0, leaving the per-tap direction alignment and edge-stop as the two guards active as shipped.

**Adaptive everything.** Knee adapts to luminance, blend adapts to strength / coherence / angle, kernel mix adapts to which tensor dominates, and step size adapts to coherence and angle quality. The result is aggressive on clean, long, axis-aligned edges and conservative on corners, fine detail, and oblique boundaries.

---

## Oblique-Aware Tuning

The default parameters include several oblique-specific adjustments:

| Parameter | Default | Effect |
|-----------|---------|--------|
| OBLIQUE_BLEND_FLOOR | 0.80 | Oblique edges retain more of their blend instead of being throttled to 0.6× |
| TRI_KERNEL_PRESENCE | 0.75 | Raises the hexagonal kernel's effective tap weight from ~0.22 to ~0.34 |
| OBLIQUE_DIRMIN_FLOOR | 0.55 | Widens the tangent-agreement window on oblique edges |
| OBLIQUE_5X5_BIAS | 0.60 | Pulls kernelMix toward the rotated 5×5 on oblique edges with moderate coherence |

These stay inside the tensor/blur paradigm — no pattern matching — and are intended to stop the filter from fighting its own oblique machinery.

---

## Stacking with Other AA

Cake does not require SMAA or in-game non-temporal AA earlier in the pipeline, but pairing is beneficial in either of two ways:

- **Post-process AA (SMAA preferred over FXAA for sharpness).** SMAA's pattern-based edge reconstruction preserves more detail and introduces less blur than FXAA's single-pass luma blur, so it can cover edge cases Cake would miss — very thin sub-pixel features below Cake's consensus gate, or edges whose chroma signature is too subtle for the jump gates even though the luma Sobel fires — without eroding sharpness.
- **Information-dense AA (MSAA, SSAA).** A cleaner, less aliased input with fewer spurious high-frequency edges means Cake's consensus pre-filter sees more genuine structure and less single-pixel noise, and its tensors resolve cleaner gradients.

Either way, Cake's along-tangent-only, scale-invariant behavior composes well: it refines and extends whatever edge information the earlier stage leaves, rather than fighting it.

---

## Tuning Notes

### Color sensitivity

COLOR_SENSITIVITY (default 7.0) controls edge_stop_weight. Higher values suppress taps with larger color deltas more aggressively; lower values make the filter more forgiving of near-edge color differences.

- Original: 8.0
- Default: 7.0
- If grain near edges persists: step down toward 6.5
- If halos appear: step up toward 7.5

### Jump-gate sensitivity

JUMP_GAIN_EDGE (default 3.0) scales the axial and diagonal jump gates in pass 1 (edge2 and edge3). Lower values make the gates require a larger color jump before they fire, suppressing grainy detail close to edges at the detection stage.

- Original: 4.0
- Default: 3.0
- If grain near edges persists: step down toward 2.5
- If faint thin edges start to disappear: step up toward 3.4

### Corner damping

CORNER_CONFIDENCE_FLOOR (default 1.0) controls the corner-damping mechanism. At 1.0, the mechanism is present but neutralized. Set to 0.3 or lower to re-enable corner damping at the cost of some blend reduction on low-coherence pixels.

---

## File Structure

```
CakeAA.fx
├── PostProcessVS                 — fullscreen vertex shader
├── UI Controls                   — Debug toggle
├── Parameters                    — tuning constants
├── Textures / Samplers           — backbuffer + four edge data targets
├── Utility functions             — max3, kernel tables, HEX_RING2
├── fast_edge_consensus           — three-gate AND detector
├── compute_all_tensors           — fused 5×5, 7×7, triangular tensor bodies
├── normalize_and_regularize_tensor
├── decompose_structure_tensor
├── compute_angle_quality
├── compute_triangular_angle_quality
├── compute_adaptive_blend
├── PS_DetectEdges                — pass 1
├── edge_stop_weight
├── PS_ApplyAA                    — pass 2
└── technique CakeAA              — EdgeDetection + AntiAliasing passes
```

---

## Requirements

- ReShade with HLSL support
- A GPU supporting RGBA16F render targets and Shader Model 3.0+

---

## Debug View

Enable the Debug checkbox in the UI. The output shows:

- **R** = 5×5 strength
- **G** = triangular strength
- **B** = 7×7 strength
- **Additive** — overlaps sum toward white on a black background

This visualizes which tensor fires where, useful for understanding why a particular edge gets a particular blend.

---

## License

GNU General Public License v3 — Copyright © 2026 Alex Aaron Goven.

The key correction: both kernels sample at 1 px spatial tap spacing; √2 is the ratio between their coordinate-frame axis lengths, which combined with the 5×5's ±2 rotated steps and the 7×7's ±3 square steps produces the 2√2 vs 3 near-match at the extremes.
