# 🧁 Cake AA

Continuous Adaptive Kerneling Eigen AA — a scale-invariant anti-aliasing technique built on consensus-filtered structure tensors. Two complementary kernels (a rotated 5×5 and an axis-aligned 7×7) are decomposed into eigenvalues and tangents; an adaptive blend steers along-tangent sampling, with edge-stop weighting to prevent bleeding across boundaries.

## How It Works

A three-gate soft-AND consensus pre-filter (distance-2 color jump, 1px axial jump, 1px diagonal jump) is combined by a **geometric mean** so that low-contrast edges — where all three gates are only moderately positive — still fire, instead of being annihilated by a hard product. Gates use **contrast-relative gains** (scaled up in dark / flat regions), and the minimum-edge floor scales with local luma range. A **directional-consistency rescue** admits genuinely stair-stepped low-contrast edges (concentrated 1px axial jump on one side of the step) while rejecting diffuse texture/noise, and downstream edge-stop weighting is tightened so the rescued detections do not thicken or smear. Where consensus is low, pixels pass through untouched. Where it fires, two structure tensors are normalized, regularized, and decomposed into λ1, λ2, and the edge tangent. Blend strength, kernel mix, and step size all adapt to coherence and angle quality.

## Kernel Profiles

**5×5 rotated kernel** — the `1,4,6,4,1` binomial profile, selected by **diamond ring** on the rotated lattice:

| Diamond ring `r` | Binomial index | Weight |
|---|---|---|
| `r = 0` (center) | 6 | 6.0 |
| `r = 1` (4 diamond neighbors) | 4 | 4.0 |
| `r = 2` (outer ring) | 1 | 1.0 |

i.e. `DIAMOND_W = {6, 4, 1}`. Absolute scale cancels in the per-pixel normalization, so only the `6:4:1` ratio matters. This is the correct binomial coefficient sequence for a 5-tap kernel, yielding a proper separable derivative-of-Gaussian on the rotated diamond (and avoiding the rotated-square support bias of a literal 5×5 outer product).

**7×7 axis-aligned kernel** — the 7-tap binomial, applied as a separable outer product:

`KERNEL_7x7 = {1, 3, 6, 7, 6, 3, 1}`, normalized by `729 = 27²`.

## Outer Products & Separability

The 7×7 kernel is a true **separable** 2-D kernel: the outer product of `KERNEL_7x7` with itself. The outer product of `a = [a₁ … aₙ]` with itself is the matrix whose `(i, j)` entry is `aᵢ · aⱼ`. Every row is a scaled copy of `a`; every column is a scaled copy of `a` — which is why a 2-D Gaussian can be applied as two 1-D passes.

The 5×5 kernel is **not** the literal `{1,4,6,4,1}` outer product. It is the same binomial profile indexed by diamond radius on the rotated lattice, which gives an isotropic diamond response instead of a rotated square. This is deliberately different: the literal 5×5 outer product would put too much weight in the corners of the rotated support.

## Repunit Identity

The 1-D 5-tap profile `{1, 4, 6, 4, 1}` is row 4 of Pascal's triangle, and its digits spell `14641 = 11⁴`.

The 1-D 7-tap profile `{1, 3, 6, 7, 6, 3, 1}` read as a decimal digit string is from `1367631 = 111³`.

- **11⁴** = 14641
- **111³** = 1367631

Both are digit-palindromes whose entries are the 1-D tap weights. Their row sums `16` and `27` are the normalizers for the 1-D profiles, and for the 7×7 separable kernel the square `27² = 729` is the denominator in the tensor accumulation loop. (The 5×5 diamond kernel is normalized by its diamond-ring weight sum, not by `16²`, because the tap weights are selected by radius rather than by the full outer product.)
