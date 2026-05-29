# Drift Report — breton

- Sample size A: 1,000
- Sample size B: 1,000

## Per-tag distribution shift
- KL(A || B): **0.0089** (lower = closer; 0 = identical; >1 = material drift)
- Total variation: **0.0409** (0..1; 0 = identical; 1 = disjoint supports)

## Top-N name overlap
- Jaccard(top-100): **0.0204** (0..1; 1 = same top names; 0 = disjoint)

## Decomposition rate
- A: **1.0000**
- B: **1.0000**
- Delta (B - A): **+0.0000**

## Position distribution
- A: bare=0.344, inner=0.126, post=0.242, pre=0.288
- B: bare=0.344, inner=0.126, post=0.242, pre=0.288

## Morpheme rank correlation
- Spearman ρ (top-100): **+0.7855** (-1..+1; +1 = identical ranking; 0 = unrelated; -1 = inverse)