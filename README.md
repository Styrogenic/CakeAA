# 🧁 Cake AA

Continuous Adaptive Kerneling Eigen AA — a scale-invariant anti-aliasing technique built on consensus-filtered structure tensors. Two complementary kernels (a rotated 5×5 and an axis-aligned 7×7) are decomposed into eigenvalues and tangents; an adaptive blend steers along-tangent sampling, with edge-stop weighting to prevent bleeding across boundaries.

## How It Works

A three-gate soft-AND consensus pre-filter (distance-2 color jump, 1px axial jump, 1px diagonal jump) is combined by a **geometric mean** so that low-contrast edges — where all three gates are only moderately positive — still fire, instead of being annihilated by a hard product. Gates use **contrast-relative gains** (scaled up in dark / flat regions), and the minimum-edge floor scales with local luma range. Where consensus is low, pixels pass through untouched. Where it fires, two structure tensors are normalized, regularized, and decomposed into λ1, λ2, and the edge tangent. Blend strength, kernel mix, and step size all adapt to coherence and angle quality.

## Outer Products & Separability

Both 2-D kernels are **separable**: each is the *outer product* of a 1-D profile with itself. The outer product of `a = [a₁ … aₙ]` with itself is the matrix whose `(i, j)` entry is `aᵢ · aⱼ`. Every row is a scaled copy of `a`; every column is a scaled copy of `a`. This is why a 2-D Gaussian can be applied as two 1-D passes — and why the tap values below form a rank-1 matrix.

**1-D profile · 5-tap:** `{1, 4, 6, 4, 1}` — outer product → rotated 45°

**1-D profile · 7-tap:** `{1, 3, 6, 7, 6, 3, 1}` — outer product → axis-aligned

## Repunit Identity

The 5-tap profile `{1, 4, 6, 4, 1}` is row 4 of Pascal's triangle, and its digits spell `14641 = 11⁴`.

The 7-tap profile `{1, 3, 6, 7, 6, 3, 1}` read as a decimal digit string is from `1367631 = 111³`.

- **11⁴** = 14641
- **111³** = 1367631

Both are digit-palindromes whose entries are the coefficients that weight the taps. The row sums `16` and `27` are the normalizers used in the shader, and their squares `16² = 256` and `27² = 729` are the denominators in the tensor accumulation loops.
