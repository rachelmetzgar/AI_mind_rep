# Exp 3: Pairwise Dimension Comparison — Standalone Concept Alignment (Partner Identity)

*Generated: 2026-03-04 11:37 | Turn: 2*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 9 | 3 | 13 | 5 |
| Emotions | Mental | 8 | 4 | 5 | 8 |
| Agency | Mental | 8 | 4 | 5 | 8 |
| Intentions | Mental | 8 | 4 | 5 | 9 |
| Prediction | Mental | 8 | 2 | 5 | 7 |
| Cognitive | Mental | 3 | 12 | 5 | 7 |
| Social | Mental | 2 | 15 | 2 | 19 |
| Attention | Mental | 8 | 4 | 13 | 4 |
| Embodiment | Physical | 9 | 2 | 14 | 2 |
| Roles | Physical | 5 | 12 | 8 | 7 |
| Animacy | Physical | 15 | 1 | 17 | 1 |
| Formality | Pragmatic | 2 | 13 | 3 | 14 |
| Expertise | Pragmatic | 2 | 16 | 4 | 9 |
| Helpfulness | Pragmatic | 2 | 15 | 3 | 16 |
| Biological | Control | 18 | 0 | 15 | 1 |
| Shapes | Control | 8 | 4 | 15 | 1 |
| Human (concept) | Entity | 16 | 0 | 8 | 7 |
| AI (concept) | Entity | 19 | 0 | 21 | 0 |
| SysPrompt (talkto human) | SysPrompt | 0 | 21 | 0 | 20 |
| SysPrompt (talkto AI) | SysPrompt | 6 | 4 | 3 | 3 |
| SysPrompt (bare human) | SysPrompt | 1 | 20 | 0 | 20 |
| SysPrompt (bare AI) | SysPrompt | 5 | 6 | 5 | 1 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | AI (concept) | SysPrompt (talkto human) | 1.912 | 0.0000 | *** |
| Operational | AI (concept) | SysPrompt (bare human) | 1.903 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (talkto human) | 1.502 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (bare human) | 1.493 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (talkto human) | 1.438 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (bare human) | 1.428 | 0.0000 | *** |
| Operational | Biological | SysPrompt (talkto human) | 1.428 | 0.0000 | *** |
| Operational | Biological | SysPrompt (bare human) | 1.418 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (talkto human) | 1.391 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (bare human) | 1.382 | 0.0000 | *** |
| Operational | Social | AI (concept) | -1.341 | 0.0000 | *** |
| Operational | Attention | SysPrompt (talkto human) | 1.301 | 0.0000 | *** |
| Operational | Attention | SysPrompt (bare human) | 1.291 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (talkto human) | 1.276 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (bare human) | 1.267 | 0.0000 | *** |
| Operational | Helpfulness | AI (concept) | -1.170 | 0.0000 | *** |
| Operational | SysPrompt (talkto human) | SysPrompt (bare AI) | -1.153 | 0.0000 | *** |
| Operational | SysPrompt (bare human) | SysPrompt (bare AI) | -1.143 | 0.0000 | *** |
| Operational | Human (concept) | SysPrompt (talkto human) | 1.115 | 0.0000 | *** |
| Operational | Roles | SysPrompt (talkto human) | 1.111 | 0.0000 | *** |

## Methods

- **Analysis**: Standalone Concept Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
