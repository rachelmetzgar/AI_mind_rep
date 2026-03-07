# Exp 3: Pairwise Dimension Comparison — Standalone Concept Alignment (Partner Identity)

*Generated: 2026-03-04 11:38 | Turn: 3*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 11 | 2 | 12 | 5 |
| Emotions | Mental | 7 | 5 | 6 | 9 |
| Agency | Mental | 7 | 2 | 6 | 9 |
| Intentions | Mental | 7 | 8 | 4 | 12 |
| Prediction | Mental | 12 | 1 | 5 | 10 |
| Cognitive | Mental | 4 | 8 | 5 | 9 |
| Social | Mental | 2 | 15 | 2 | 18 |
| Attention | Mental | 10 | 3 | 11 | 5 |
| Embodiment | Physical | 13 | 1 | 15 | 2 |
| Roles | Physical | 6 | 8 | 7 | 7 |
| Animacy | Physical | 13 | 2 | 17 | 1 |
| Formality | Pragmatic | 6 | 6 | 2 | 17 |
| Expertise | Pragmatic | 2 | 12 | 5 | 9 |
| Helpfulness | Pragmatic | 2 | 14 | 3 | 15 |
| Biological | Control | 14 | 1 | 16 | 1 |
| Shapes | Control | 7 | 7 | 15 | 3 |
| Human (concept) | Entity | 21 | 0 | 12 | 5 |
| AI (concept) | Entity | 17 | 1 | 20 | 0 |
| SysPrompt (talkto human) | SysPrompt | 0 | 21 | 0 | 21 |
| SysPrompt (talkto AI) | SysPrompt | 2 | 14 | 4 | 2 |
| SysPrompt (bare human) | SysPrompt | 1 | 20 | 1 | 20 |
| SysPrompt (bare AI) | SysPrompt | 2 | 15 | 12 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | AI (concept) | SysPrompt (talkto human) | 1.073 | 0.0000 | *** |
| Operational | AI (concept) | SysPrompt (bare human) | 0.973 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (talkto human) | 0.802 | 0.0000 | *** |
| Operational | Biological | SysPrompt (talkto human) | 0.773 | 0.0000 | *** |
| Operational | SysPrompt (talkto human) | SysPrompt (bare AI) | -0.771 | 0.0000 | *** |
| Operational | Social | AI (concept) | -0.733 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (talkto human) | 0.727 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (talkto human) | 0.716 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (bare human) | 0.703 | 0.0000 | *** |
| Operational | Formality | AI (concept) | -0.678 | 0.0000 | *** |
| Operational | Biological | SysPrompt (bare human) | 0.674 | 0.0000 | *** |
| Operational | SysPrompt (bare human) | SysPrompt (bare AI) | -0.672 | 0.0000 | *** |
| Operational | Helpfulness | AI (concept) | -0.651 | 0.0000 | *** |
| Operational | SysPrompt (talkto human) | SysPrompt (talkto AI) | -0.639 | 0.0000 | *** |
| Operational | Human (concept) | SysPrompt (talkto human) | 0.631 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (bare human) | 0.628 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (talkto human) | 0.624 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (bare human) | 0.617 | 0.0000 | *** |
| Operational | Attention | SysPrompt (talkto human) | 0.609 | 0.0000 | *** |
| Operational | Intentions | AI (concept) | -0.605 | 0.0000 | *** |

## Methods

- **Analysis**: Standalone Concept Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
