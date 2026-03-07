# Concept Probe Steering: Notes for Future Reference

*Saved 2026-03-04. Discussion of mean-vector vs probe-based concept steering.*

## Context

When designing concept steering for Exp 3, we considered two approaches:
1. **Mean vector steering** (activation addition with concept directions) — CHOSEN for initial implementation
2. **Concept probe steering** (train linear probes on concept activations, steer with probe weights)

## Why Mean Vectors First

- Concept directions are already computed (19 dims × 41 layers × 5120)
- Matches Anthropic's methodology in their introspection paper (Lindsey, Jan 2025)
- Matches the ActAdd / representation engineering literature
- With only 80 prompts per dimension (40H + 40A), a linear probe converges to approximately the mean difference direction anyway (Fisher LDA ≈ μ₁ - μ₂ when you can't estimate Σ)

## Why Probes May Be Worth Revisiting

1. **Per-layer accuracy** gives principled layer selection (which layers linearly separate H/A?)
2. **Cross-prediction** (script 7 already written): can concept probes classify conversational data? This directly tests the shared-representation hypothesis
3. **Direction comparison**: if probe weight ≠ mean vector, that's informative about noise structure
4. **Consistency with exp_2**: probes enable direct comparison of concept-steered vs conversation-steered results

## Feasibility Concerns

- **Sample size**: 80 prompts → 64 train / 16 test (5120-d space)
  - Massively underdetermined — linear classifier can perfectly separate 64 points in 5120-d
  - Test accuracy on 16 samples: variance ~±25%
  - Risk: probe "works" but learned direction is noisy
- **Exception**: dim 16 (mind) has 800 prompts — much more reliable
- **Mitigation**: L2 regularization, cross-validation, compare to mean vector

## Implementation Path (when ready)

1. Run existing script: `exp_3/code/3_train_concept_probes.py` (array job, one per dim)
2. Check per-layer accuracy vs mean vector alignment
3. Run cross-prediction: `exp_3/code/7_cross_prediction.py` (needs probes + GPU)
4. If probes look clean, run steering generation with probe weights for comparison

## What Anthropic Did

- Paper: "Emergent Introspective Awareness in LLMs" (Lindsey, Anthropic, Jan 2025)
- Used contrastive activation subtraction (≈ mean vectors), NOT trained probes
- Optimal layer: ~2/3 through model depth
- Multipliers: 2-4 (higher → incoherence)
- 50 prompts per concept
- ~20% detection rate at optimal settings

## Key Math

For balanced binary classification with linear probe:
- Optimal direction = Σ⁻¹(μ₁ - μ₂) (Fisher LDA)
- When n << d (our case: 64 << 5120), can't estimate Σ reliably
- Reduces to ≈ (μ₁ - μ₂) — the mean difference vector
- So probes ≈ mean vectors for this dataset size
