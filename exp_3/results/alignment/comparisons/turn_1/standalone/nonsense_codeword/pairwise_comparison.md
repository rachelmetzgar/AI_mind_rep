# Exp 3: Pairwise Dimension Comparison — Standalone Concept Alignment (Control)

*Generated: 2026-03-04 11:36 | Turn: 1*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 5 | 5 | 0 | 14 |
| Emotions | Mental | 8 | 2 | 0 | 15 |
| Agency | Mental | 8 | 2 | 2 | 12 |
| Intentions | Mental | 4 | 10 | 0 | 13 |
| Prediction | Mental | 9 | 2 | 7 | 11 |
| Cognitive | Mental | 5 | 9 | 10 | 7 |
| Social | Mental | 5 | 3 | 12 | 4 |
| Attention | Mental | 5 | 10 | 5 | 12 |
| Embodiment | Physical | 10 | 2 | 0 | 13 |
| Roles | Physical | 0 | 17 | 9 | 9 |
| Animacy | Physical | 5 | 4 | 0 | 15 |
| Formality | Pragmatic | 0 | 17 | 11 | 5 |
| Expertise | Pragmatic | 0 | 15 | 13 | 3 |
| Helpfulness | Pragmatic | 0 | 17 | 11 | 4 |
| Biological | Control | 8 | 2 | 0 | 14 |
| Shapes | Control | 12 | 2 | 0 | 14 |
| Human (concept) | Entity | 4 | 6 | 10 | 6 |
| AI (concept) | Entity | 0 | 17 | 19 | 0 |
| SysPrompt (talkto human) | SysPrompt | 20 | 0 | 17 | 3 |
| SysPrompt (talkto AI) | SysPrompt | 20 | 0 | 19 | 1 |
| SysPrompt (bare human) | SysPrompt | 7 | 4 | 14 | 4 |
| SysPrompt (bare AI) | SysPrompt | 13 | 2 | 20 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Emotions | AI (concept) | -0.282 | 0.0000 | *** |
| Operational | Animacy | AI (concept) | -0.280 | 0.0000 | *** |
| Operational | Shapes | AI (concept) | -0.274 | 0.0000 | *** |
| Operational | Phenomenology | AI (concept) | -0.268 | 0.0000 | *** |
| Operational | Emotions | SysPrompt (bare AI) | -0.263 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (bare AI) | -0.262 | 0.0000 | *** |
| Operational | Biological | AI (concept) | -0.257 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (bare AI) | -0.256 | 0.0000 | *** |
| Operational | Embodiment | AI (concept) | -0.253 | 0.0000 | *** |
| Operational | Intentions | AI (concept) | -0.253 | 0.0000 | *** |
| Operational | Emotions | SysPrompt (talkto AI) | -0.252 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (talkto AI) | -0.250 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (bare AI) | -0.250 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (talkto AI) | -0.244 | 0.0000 | *** |
| Operational | Biological | SysPrompt (bare AI) | -0.239 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (talkto AI) | -0.238 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (bare AI) | -0.235 | 0.0000 | *** |
| Operational | Intentions | SysPrompt (bare AI) | -0.234 | 0.0000 | *** |
| Operational | Agency | AI (concept) | -0.231 | 0.0000 | *** |
| Operational | Biological | SysPrompt (talkto AI) | -0.227 | 0.0000 | *** |

## Methods

- **Analysis**: Standalone Concept Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
