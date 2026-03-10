# Exp 3: Pairwise Dimension Comparison — Standalone Concept Alignment (Partner Identity)

*Generated: 2026-03-08 14:56 | Turn: 5*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 13 | 2 | 12 | 4 |
| Emotions | Mental | 9 | 3 | 8 | 8 |
| Agency | Mental | 7 | 5 | 7 | 9 |
| Intentions | Mental | 5 | 8 | 5 | 13 |
| Prediction | Mental | 6 | 5 | 5 | 11 |
| Cognitive | Mental | 3 | 13 | 5 | 9 |
| Social | Mental | 1 | 19 | 2 | 17 |
| Attention | Mental | 9 | 3 | 10 | 5 |
| Embodiment | Physical | 15 | 1 | 13 | 2 |
| Roles | Physical | 6 | 5 | 8 | 8 |
| Animacy | Physical | 15 | 2 | 15 | 1 |
| Formality | Pragmatic | 3 | 11 | 2 | 17 |
| Expertise | Pragmatic | 3 | 14 | 5 | 13 |
| Helpfulness | Pragmatic | 3 | 14 | 2 | 17 |
| Biological | Control | 7 | 4 | 15 | 1 |
| Shapes | Control | 7 | 7 | 12 | 4 |
| Human (concept) | Entity | 18 | 1 | 16 | 1 |
| AI (concept) | Entity | 21 | 0 | 21 | 0 |
| SysPrompt (talkto human) | SysPrompt | 0 | 21 | 0 | 21 |
| SysPrompt (talkto AI) | SysPrompt | 3 | 6 | 7 | 2 |
| SysPrompt (bare human) | SysPrompt | 1 | 19 | 1 | 20 |
| SysPrompt (bare AI) | SysPrompt | 9 | 1 | 13 | 1 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | AI (concept) | SysPrompt (talkto human) | 1.172 | 0.0000 | *** |
| Operational | AI (concept) | SysPrompt (bare human) | 1.085 | 0.0000 | *** |
| Operational | Social | AI (concept) | -0.901 | 0.0000 | *** |
| Operational | Helpfulness | AI (concept) | -0.846 | 0.0000 | *** |
| Operational | Formality | AI (concept) | -0.842 | 0.0000 | *** |
| Operational | Human (concept) | SysPrompt (talkto human) | 0.802 | 0.0000 | *** |
| Operational | SysPrompt (talkto human) | SysPrompt (bare AI) | -0.800 | 0.0000 | *** |
| Operational | Intentions | AI (concept) | -0.765 | 0.0000 | *** |
| Operational | Expertise | AI (concept) | -0.757 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (talkto human) | 0.750 | 0.0000 | *** |
| Operational | Prediction | AI (concept) | -0.739 | 0.0000 | *** |
| Operational | Cognitive | AI (concept) | -0.738 | 0.0000 | *** |
| Operational | Biological | SysPrompt (talkto human) | 0.730 | 0.0000 | *** |
| Operational | Human (concept) | SysPrompt (bare human) | 0.715 | 0.0000 | *** |
| Operational | SysPrompt (bare human) | SysPrompt (bare AI) | -0.713 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (talkto human) | 0.693 | 0.0000 | *** |
| Operational | Agency | AI (concept) | -0.682 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (bare human) | 0.663 | 0.0000 | *** |
| Operational | Emotions | AI (concept) | -0.654 | 0.0000 | *** |
| Operational | Roles | AI (concept) | -0.652 | 0.0000 | *** |

## Methods

- **Analysis**: Standalone Concept Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
