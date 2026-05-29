# Drift Report — irish

- Sample size A: 1,000
- Sample size B: 1,000

## Per-tag distribution shift
- KL(A || B): **0.0227** (lower = closer; 0 = identical; >1 = material drift)
- Total variation: **0.0548** (0..1; 0 = identical; 1 = disjoint supports)

## Top-N name overlap
- Jaccard(top-100): **0.0050** (0..1; 1 = same top names; 0 = disjoint)

## Decomposition rate
- A: **1.0000**
- B: **1.0000**
- Delta (B - A): **+0.0000**

## Position distribution
- A: bare=0.474, inner=0.191, post=0.166, pre=0.170
- B: bare=0.474, inner=0.191, post=0.166, pre=0.170

## Morpheme rank correlation
- Spearman ρ (top-100): **+0.7611** (-1..+1; +1 = identical ranking; 0 = unrelated; -1 = inverse)