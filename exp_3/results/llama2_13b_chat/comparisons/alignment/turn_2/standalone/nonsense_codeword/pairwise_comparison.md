# Exp 3: Pairwise Dimension Comparison — Standalone Concept Alignment (Control)

*Generated: 2026-03-04 11:37 | Turn: 2*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 1 | 11 | 4 | 9 |
| Emotions | Mental | 3 | 7 | 9 | 6 |
| Agency | Mental | 7 | 5 | 9 | 6 |
| Intentions | Mental | 5 | 6 | 3 | 9 |
| Prediction | Mental | 16 | 2 | 16 | 0 |
| Cognitive | Mental | 10 | 3 | 16 | 1 |
| Social | Mental | 11 | 4 | 16 | 0 |
| Attention | Mental | 7 | 5 | 5 | 6 |
| Embodiment | Physical | 5 | 7 | 5 | 6 |
| Roles | Physical | 7 | 5 | 5 | 6 |
| Animacy | Physical | 0 | 17 | 1 | 17 |
| Formality | Pragmatic | 2 | 13 | 16 | 0 |
| Expertise | Pragmatic | 5 | 7 | 16 | 0 |
| Helpfulness | Pragmatic | 2 | 14 | 17 | 0 |
| Biological | Control | 0 | 19 | 1 | 16 |
| Shapes | Control | 0 | 15 | 1 | 17 |
| Human (concept) | Entity | 1 | 12 | 0 | 21 |
| AI (concept) | Entity | 7 | 5 | 1 | 13 |
| SysPrompt (talkto human) | SysPrompt | 15 | 3 | 4 | 10 |
| SysPrompt (talkto AI) | SysPrompt | 18 | 2 | 10 | 6 |
| SysPrompt (bare human) | SysPrompt | 20 | 0 | 6 | 7 |
| SysPrompt (bare AI) | SysPrompt | 20 | 0 | 4 | 9 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Metacognitive | Biological | SysPrompt (bare human) | -0.162 | 0.0000 | *** |
| Metacognitive | Biological | SysPrompt (bare AI) | -0.160 | 0.0000 | *** |
| Metacognitive | Animacy | SysPrompt (bare human) | -0.159 | 0.0000 | *** |
| Metacognitive | Animacy | SysPrompt (bare AI) | -0.157 | 0.0000 | *** |
| Metacognitive | Shapes | SysPrompt (bare human) | -0.145 | 0.0000 | *** |
| Metacognitive | Shapes | SysPrompt (bare AI) | -0.144 | 0.0000 | *** |
| Metacognitive | Human (concept) | SysPrompt (bare human) | -0.139 | 0.0000 | *** |
| Metacognitive | Human (concept) | SysPrompt (bare AI) | -0.137 | 0.0000 | *** |
| Metacognitive | Formality | SysPrompt (bare human) | -0.136 | 0.0000 | *** |
| Metacognitive | Phenomenology | SysPrompt (bare human) | -0.135 | 0.0000 | *** |
| Metacognitive | Helpfulness | SysPrompt (bare human) | -0.135 | 0.0000 | *** |
| Metacognitive | Formality | SysPrompt (bare AI) | -0.134 | 0.0000 | *** |
| Metacognitive | Phenomenology | SysPrompt (bare AI) | -0.134 | 0.0000 | *** |
| Metacognitive | Helpfulness | SysPrompt (bare AI) | -0.133 | 0.0000 | *** |
| Metacognitive | Emotions | SysPrompt (bare human) | -0.118 | 0.0000 | *** |
| Metacognitive | Emotions | SysPrompt (bare AI) | -0.116 | 0.0000 | *** |
| Metacognitive | Expertise | SysPrompt (bare human) | -0.114 | 0.0000 | *** |
| Metacognitive | Embodiment | SysPrompt (bare human) | -0.113 | 0.0000 | *** |
| Metacognitive | Expertise | SysPrompt (bare AI) | -0.113 | 0.0000 | *** |
| Operational | Formality | Human (concept) | 0.112 | 0.0000 | *** |

## Methods

- **Analysis**: Standalone Concept Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
