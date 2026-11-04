# Cake AA — Continuous Adaptive Kerneling Eigen Anti-Aliasing

A ReShade post-process anti-aliasing shader in HLSL implementing a multi-tensor, edge-directed AA filter.

It detects edges, builds local gradient tensors with different footprints, then blurs **along** edges — never across them — with per-pixel strength and kernel selection driven by edge confidence.

- **Author:** Alex Aaron Goven
- **Copyright © 2026**
- **License:** GNU General Public License v3

---

## Overview

Cake AA is a two-pass filter:

1. **EdgeDetection** — writes RGBA16F render targets containing tensor strengths, tangents, coherence, and angle quality.
2. **AntiAliasing** — reads those targets plus the backbuffer and blends in an edge-directed average.

The filter never samples perpendicular to the local edge, so it smooths along edges rather than across them.

---

## Pipeline

### Pass 1 — Edge Detection (PS_DetectEdges)

**Step 1: Consensus pre-filter (fast_edge_consensus)**

A cheap three-gate AND detector decides whether a pixel is worth the expensive tensor math. If consensus is below 0.1, all outputs are zeroed and the pixel exits early.

| Gate | Signal | Notes |
|------|--------|-------|
| 1 | Distance-2 color jump | Max per-channel color difference between center and its eight distance-2 neighbors (4 axial + 4 diagonal) — chroma-aware |
| 2 | 1px axial jump | Max per-channel color difference between center and its four axial 1px neighbors — chroma-aware |
| 3 | 1px diagonal jump | Same for the four diagonal 1px neighbors |

Consensus = edge1 × edge2 × edge3, so all three must agree. It is smoothstepped around a cubed threshold and scaled by a smoothstep on the max of the three to suppress weak edges. This kills flat regions and isolated single-pixel noise before the tensor loops run.

Gate 1's distance-2 footprint decorrelates it from the two 1px jump gates, so the AND remains discriminating rather than collapsing into a single correlated test.

**Step 2: Structure tensors (compute_all_tensors)**

| Tensor | Footprint | Side length | Weights | Outer product values |
|--------|-----------|-------------|---------|----------------------|
| 5×5 | Rotated 45° grid, coords (col−row)/√2 and (col+row)/√2 | 5 taps | Separable binomial 1, 4, 6, 4, 1 | From 11^4 |
| 7×7 | Axis-aligned square | 7 taps (1.4× the 5×5) | Separable 1, 3, 6, 7, 6, 3, 1 | From 111^3 |

The 7×7 tensor's side length is 1.4× the 5×5's — a 7:5 ratio — giving it proportionally larger edge-sampling reach. The 5×5 samples on a rotated 45° grid, so it resolves diagonal structure densely and axis-aligned structure coarsely; the 7×7 samples on an axis-aligned square grid and does the reverse. The two are complementary.

Each tensor yields gx and gy (3-channel gradient sums) and a weight total. Tensor components are S11 = gx·gx, S12 = gx·gy, S22 = gy·gy.

**Step 3: Normalize and regularize**

normalize_and_regularize_tensor divides by weight sum, adds epsilon to the diagonal, and divides by gradientScale² (2.0, 3.0). This makes the tensors scale-invariant, so they respond to relative structure rather than absolute brightness — suited to HDR and game content, and functional as subpixel AA.

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
| longEdgeFactor | coherence_7 | Confidence this is a long edge rather than a corner |

**Step 5: Angle quality**

compute_angle_quality rewards tangents aligned to image axes or 45° diagonals, boosting oblique angles toward 1.0.

**Outputs**

| Target | R | G | B | A |
|--------|---|---|---|---|
| EdgeDataTex0 | strength5 | strength7 | angleQuality | consensus |
| EdgeDataTex1 | tangent5.x | tangent5.y | tangent7.x | tangent7.y |
| EdgeDataTex2 | coherence_7 | 0 | 0 | 0 |
| EdgeDataTex3 | 0 | 0 | 0 | 0 |

### Pass 2 — Anti-Aliasing (PS_ApplyAA)

**Debug view:** if Debug is on, output is strength5 in R and strength7 in B, visualizing which tensor fires where.

**Adaptive blend strength (compute_adaptive_blend)**

Combines the two strengths via RMS, coherence_7, and angleQuality into a per-pixel blend amount. Stronger edges produce more blend; high coherence produces a coherenceBoost.

**Corner damping** — structureConfidence = lerp(CORNER_CONFIDENCE_FLOOR, 1.0, coherence) multiplies strengthQuality. The floor sets the confidence multiplier at zero coherence, scaling up with coherence, so corner-like pixels get reduced overall blend, preventing corner smearing and junction blow-outs. **This build ships with CORNER_CONFIDENCE_FLOOR = 1.0, which neutralizes the mechanism at the default; lowering the floor re-enables it.**

Very weak edges (edgeStrength < 0.1) are pulled toward 0.1 and very strong edges (> 0.8) toward 0.9, to avoid over-blur and under-blur at the extremes.

**Kernel mix**

kernelPref7 = strength7 / (strength5 + strength7), boosted by longEdgeFactor × angleQuality, then smoothstepped. Confident long edges get the long 7×7 kernel; short and fine features get the 5×5.

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
| stopWeight | 1 / (1 + 2x²) where x = colorDiff × COLOR_SENSITIVITY, relaxed by sampleEdge | Suppresses taps with large color differences; relaxation prevents halos |

dirAlignMin is relaxed toward 0.55 on oblique edges (lerp(0.55, 0.7, angleQuality)) to admit more oblique-aligned neighbors without opening the gate on corners.

**Final blend**

The center is preloaded with a large initial weight of 16 to 32 (by kernelMix), keeping the result near the original unless neighbors strongly agree. finalColor = acc / totalWeight, output = lerp(centerColor, finalColor, blend).

---

## Key Design Choices

**Scale invariance.** All tensors normalize by weight sum and gradient scale, so they respond to structure at any contrast level. Combined with sub-texel sampling steps, this makes Cake AA functional as subpixel AA.

**Two complementary kernels.** Rotated 5×5 and axis-aligned 7×7. The 7×7's side length is 1.4× the 5×5's, giving it proportionally larger reach; the 5×5's rotated grid resolves diagonals that the 7×7 resolves poorly, and vice versa.

**Consensus gating.** The cheap three-gate AND detector skips most pixels — including flat regions and noise — before the expensive tensor loops. Gate 1's distance-2 footprint keeps the AND discriminating.

**Along-tangent-only blur.** The filter never samples perpendicular to the local edge, so it smooths along edges, not across them.

**Layered artifact prevention.** Corner damping in the blend strength pulls back overall on low-coherence pixels, per-tap direction alignment rejects disagreeing neighbors, and edge-stop rejects large color jumps. Corner damping is neutralized at the default CORNER_CONFIDENCE_FLOOR of 1.0, leaving the per-tap direction alignment and edge-stop as the two guards active as shipped.

**Adaptive everything.** Knee adapts to luminance, blend adapts to strength / coherence / angle, kernel mix adapts to which tensor dominates, and step size adapts to coherence and angle quality. The result is aggressive on clean, long, axis-aligned edges and conservative on corners, fine detail, and oblique boundaries.

---

## Oblique-Aware Tuning

The default parameters include several oblique-specific adjustments:

| Parameter | Default | Effect |
|-----------|---------|--------|
| OBLIQUE_BLEND_FLOOR | 0.80 | Oblique edges retain more of their blend instead of being throttled to 0.6× |
| OBLIQUE_DIRMIN_FLOOR | 0.55 | Widens the tangent-agreement window on oblique edges |
| OBLIQUE_5X5_BIAS | 0.60 | Pulls kernelMix toward the rotated 5×5 on oblique edges with moderate coherence |

These stay inside the tensor/blur paradigm — no pattern matching — and are intended to stop the filter from fighting its own oblique machinery.

---

## Curve Formation

This section documents how each curve in the shader is actually formed — which are derived, which are hand-tuned, and which are standard named curves.

**Tensor weight curves**

- **5×5:** `1, 4, 6, 4, 1` — Pascal's triangle, row 4, `C(4,k)`. Binomial. Row sum 16, outer product normalized by 16² = 256. The discrete binomial is the standard finite-difference approximation to a Gaussian.
- **7×7:** `1, 3, 6, 7, 6, 3, 1` — a hand-selected symmetric, unimodal row. **Not** a Pascal row (row 6 is `1, 6, 15, 20, 15, 6, 1`). **Not** a Gaussian sample. Flatter and wider than Pascal's row 6, with a gentler peak-to-shoulder ratio and a deliberately softened center for the long-edge kernel role. Row sum 27, outer product normalized by 27² = 729.

The `11^4` and `111^3` designations are the basis labels for the tensor weight structure. They are characterizations of the construction, not numeric sums of the rows or outer products — the 5×5 row sums to 16 and the 7×7 row sums to 27; the outer products sum to 256 and 729 respectively. The `11^4` and `111^3` naming captures the basis structure, not an arithmetic identity of the row entries.

**Radial falloff curves (pass 2 tap weights)**

- 5×5: `9, 9, 2` at |i| = 1, 2, 3
- 7×7: `24, 12, 3` at |i| = 1, 2, 3

These are separate from the tensor weights and are hand-tuned per tap offset. They are not from any named sequence; they were chosen so the tap loop's center-bias matches the kernel-mix intent.

**Smoothstep curves**

Standard Hermite cubic: `smoothstep(a, b, x) = t²(3 − 2t)` with `t = saturate((x − a)/(b − a))`. Used in four places, each with a different window `[a, b]`:
- Consensus threshold softening
- Minimum-edge floor
- Direction alignment gate
- Kernel mix

**Rational edge-stop suppression**

`1 / (1 + 2x²)` where `x = colorDiff × COLOR_SENSITIVITY`. Cauchy/Lorentzian-shaped: heavy-tailed compared to a Gaussian, so it never truly reaches zero — a tap with a moderately large color difference retains a small weight rather than being cut off sharply. The `lerp(..., 1.0, sampleEdge)` relaxation lets strong edges partially bypass the suppression.

**Linear interpolation and adaptation curves**

Standard `lerp(a, b, t) = a + (b − a)t`, plus linear tapers toward clamp endpoints for very weak and very strong edges. These are trivial interpolations; the endpoints are chosen per parameter.

**Summary**

| Curve | Formation method | Named ancestor? |
|---|---|---|
| 5×5 tensor weights | Pascal row 4, `C(4,k)` | Binomial |
| 7×7 tensor weights | Hand-selected symmetric row | None (`111^3` is a basis label) |
| 5×5 radial falloff | Hand-tuned `9, 9, 2` | None |
| 7×7 radial falloff | Hand-tuned `24, 12, 3` | None |
| Smoothstep gates | Hermite cubic `t²(3−2t)` | Standard smoothstep |
| Edge-stop suppression | `1/(1+2x²)` | Lorentzian-like |
| Blend/angle/dir lerps | Linear `a+(b−a)t` | Linear interpolation |
| Adaptation clamps | Linear tapers to endpoints | None |

The pattern across all of these: the tensor kernel rows are the only mathematically-derived curves (and only the 5×5 is a standard derivation; the 7×7 is chosen). Everything else is hand-tuned — either a custom profile, a named standard curve, or a linear taper.

---

## Originality

Cake AA combines established mathematical components into a specific architecture that is not represented among existing ReShade anti-aliasing shaders. The distinction between what is novel and what is not is stated plainly.

**Standard components** (not novel on their own):
- Structure tensors for orientation analysis
- Eigen-decomposition to find the edge tangent
- Along-tangent (edge-directed) blurring
- Post-process edge detection with blur
- Binomial kernel weights
- Smoothstep, lerp, and Lorentzian-shaped falloff curves

**Novel to Cake AA** (the specific combination and structures):
- **Consensus-gated dual-tensor detection** — a cheap three-gate AND pre-filter deciding whether a pixel is worth running *two different* tensor footprints on, with gate 1's distance-2 footprint chosen specifically to decorrelate it from the two 1px jump gates so the AND remains discriminating.
- **Complementary lattice pairing** — a rotated 5×5 and an axis-aligned 7×7 tensor, using two lattice geometries (diagonal-dense vs. axis-dense) that cover complementary orientation families, arbitrated by a `kernelMix` driven by coherence and angle quality.
- **Per-tap directional gating along a single tangent** — the `dirWeight × stopWeight` structure that samples only along the estimated edge, never across it, with direction-alignment and edge-stop as separate rejection gates.
- **Oblique-aware tuning layer** — the `OBLIQUE_5X5_BIAS`, `OBLIQUE_DIRMIN_FLOOR`, and angle-quality-driven step scaling that specifically address the failure mode of oblique edges in a tensor-based blur.
- **Tensor basis characterization** — the `11^4` and `111^3` naming of the tensor weight basis structures.

The individual vocabulary is standard graphics mathematics. The sentence structure — the specific assembly of consensus gating, dual-tensor arbitration, and per-tap direction/stop gating in a ReShade post-process — is the novel contribution. This is architecture-level originality: a new arrangement of known primitives with a specific performance and quality trade-off, rather than a new mathematical primitive.

---

## Stacking with Other AA

Cake does not require SMAA or in-game non-temporal AA earlier in the pipeline, but pairing is beneficial in either of two ways:

- **Post-process AA (SMAA preferred over FXAA for sharpness).** SMAA's pattern-based edge reconstruction preserves more detail and introduces less blur than FXAA's single-pass luma blur, so it can cover edge cases Cake would miss — very thin sub-pixel features below Cake's consensus gate, or edges whose chroma signature is too subtle for the jump gates — without eroding sharpness.
- **Information-dense AA (MSAA, SSAA).** A cleaner, less aliased input with fewer spurious high-frequency edges means Cake's consensus pre-filter sees more genuine structure and less single-pixel noise, and its tensors resolve cleaner gradients.

Either way, Cake's along-tangent-only, scale-invariant behavior composes well: it refines and extends whatever edge information the earlier stage leaves, rather than fighting it.

---

## Tuning Notes

### Edge sensitivity (gate 1)

EDGE_SENSITIVITY (default 4.0) scales gate 1's distance-2 color jump. Higher values make gate 1 fire on smaller distance-2 color deltas.

- Conservative: ~1.2–2.0
- Aggressive: ~4.0–6.0
- Above ~6.0, gate 1 saturates on most non-flat content and stops discriminating; the detector effectively becomes the two-gate AND

### Consensus floor

MIN_CONSENSUS_EDGE (default 0.10) gates the whole consensus on max(edge1, edge2, edge3). Lowering it admits pixels where all three gates are only moderately strong — the real lever for more sensitivity once gate 1 is near saturation.

- Original: 0.15
- Default: 0.10

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

---

## File Structure

CakeAA.fx

- PostProcessVS — fullscreen vertex shader
- UI Controls — Debug toggle
- Parameters — tuning constants
- Textures / Samplers — backbuffer + four edge data targets
- Utility functions — max3, kernel tables
- fast_edge_consensus — three-gate AND detector
- compute_all_tensors — fused 5×5 and 7×7 tensor bodies
- normalize_and_regularize_tensor
- decompose_structure_tensor
- compute_angle_quality
- compute_adaptive_blend
- PS_DetectEdges — pass 1
- edge_stop_weight
- PS_ApplyAA — pass 2
- technique CakeAA — EdgeDetection + AntiAliasing passes

---

## Requirements

- ReShade with HLSL support
- A GPU supporting RGBA16F render targets and Shader Model 3.0+

---

## Debug View

Enable the Debug checkbox in the UI. The output shows:

- **R** = 5×5 strength
- **B** = 7×7 strength
- **Additive** — overlaps sum toward white on a black background

This visualizes which tensor fires where, useful for understanding why a particular edge gets a particular blend.

---

## License

GNU General Public License v3 — Copyright © 2026 Alex Aaron Goven.
