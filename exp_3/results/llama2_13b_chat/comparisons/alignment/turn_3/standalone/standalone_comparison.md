# Exp 3: Standalone Concept Alignment — Cross-Version Comparison

*Generated: 2026-03-04 11:38*

> Alignment between standalone mean activation vectors and probe weights. Standalone prompts describe concepts without entity framing (no 'human' or 'AI' words). Concept vector = mean activation across all prompts for that concept. Tests whether alignment is driven by concept content rather than entity labels.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.477 | 1.599 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.320 | 0.119 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.499 | 0.322 |
| Emotions | Mental | 0.478 | 0.331 |
| Agency | Mental | 0.493 | 0.329 |
| Intentions | Mental | 0.465 | 0.322 |
| Prediction | Mental | 0.525 | 0.327 |
| Cognitive | Mental | 0.456 | 0.304 |
| Social | Mental | 0.411 | 0.310 |
| Attention | Mental | 0.492 | 0.318 |
| Embodiment | Physical | 0.518 | 0.326 |
| Roles | Physical | 0.464 | 0.311 |
| Animacy | Physical | 0.515 | 0.313 |
| Formality | Pragmatic | 0.463 | 0.291 |
| Expertise | Pragmatic | 0.438 | 0.283 |
| Helpfulness | Pragmatic | 0.429 | 0.284 |
| Human (concept) | Entity | 0.593 | 0.283 |
| AI (concept) | Entity | 0.543 | 0.293 |
| Biological | Control | 0.523 | 0.300 |
| Shapes | Control | 0.477 | 0.314 |
| SysPrompt (talkto human) | SysPrompt | 0.319 | 0.313 |
| SysPrompt (talkto AI) | SysPrompt | 0.424 | 0.312 |
| SysPrompt (bare human) | SysPrompt | 0.350 | 0.359 |
| SysPrompt (bare AI) | SysPrompt | 0.411 | 0.345 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 1.715 | 0.106 |
| Emotions | Mental | 1.619 | 0.119 |
| Agency | Mental | 1.607 | 0.115 |
| Intentions | Mental | 1.559 | 0.107 |
| Prediction | Mental | 1.575 | 0.106 |
| Cognitive | Mental | 1.592 | 0.127 |
| Social | Mental | 1.430 | 0.171 |
| Attention | Mental | 1.698 | 0.098 |
| Embodiment | Physical | 1.818 | 0.100 |
| Roles | Physical | 1.642 | 0.119 |
| Animacy | Physical | 1.893 | 0.103 |
| Formality | Pragmatic | 1.485 | 0.146 |
| Expertise | Pragmatic | 1.587 | 0.168 |
| Helpfulness | Pragmatic | 1.512 | 0.165 |
| Human (concept) | Entity | 1.721 | 0.165 |
| AI (concept) | Entity | 2.163 | 0.133 |
| Biological | Control | 1.863 | 0.110 |
| Shapes | Control | 1.807 | 0.094 |
| SysPrompt (talkto human) | SysPrompt | 1.089 | 0.335 |
| SysPrompt (talkto AI) | SysPrompt | 1.723 | 0.266 |
| SysPrompt (bare human) | SysPrompt | 1.189 | 0.251 |
| SysPrompt (bare AI) | SysPrompt | 1.856 | 0.247 |

## Methods

- **Analysis**: Standalone Concept Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
