# Exp 3: Standalone Concept Alignment — Cross-Version Comparison

*Generated: 2026-03-04 11:37*

> Alignment between standalone mean activation vectors and probe weights. Standalone prompts describe concepts without entity framing (no 'human' or 'AI' words). Concept vector = mean activation across all prompts for that concept. Tests whether alignment is driven by concept content rather than entity labels.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.453 | 3.520 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.417 | 0.132 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.476 | 0.383 |
| Emotions | Mental | 0.470 | 0.401 |
| Agency | Mental | 0.469 | 0.415 |
| Intentions | Mental | 0.458 | 0.407 |
| Prediction | Mental | 0.477 | 0.452 |
| Cognitive | Mental | 0.425 | 0.429 |
| Social | Mental | 0.393 | 0.430 |
| Attention | Mental | 0.459 | 0.416 |
| Embodiment | Physical | 0.486 | 0.406 |
| Roles | Physical | 0.435 | 0.413 |
| Animacy | Physical | 0.499 | 0.359 |
| Formality | Pragmatic | 0.414 | 0.383 |
| Expertise | Pragmatic | 0.401 | 0.404 |
| Helpfulness | Pragmatic | 0.402 | 0.384 |
| Human (concept) | Entity | 0.517 | 0.380 |
| AI (concept) | Entity | 0.531 | 0.412 |
| Biological | Control | 0.512 | 0.357 |
| Shapes | Control | 0.463 | 0.373 |
| SysPrompt (talkto human) | SysPrompt | 0.339 | 0.435 |
| SysPrompt (talkto AI) | SysPrompt | 0.449 | 0.463 |
| SysPrompt (bare human) | SysPrompt | 0.370 | 0.518 |
| SysPrompt (bare AI) | SysPrompt | 0.437 | 0.518 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 3.773 | 0.108 |
| Emotions | Mental | 3.525 | 0.126 |
| Agency | Mental | 3.512 | 0.127 |
| Intentions | Mental | 3.465 | 0.106 |
| Prediction | Mental | 3.518 | 0.160 |
| Cognitive | Mental | 3.508 | 0.147 |
| Social | Mental | 3.068 | 0.167 |
| Attention | Mental | 3.794 | 0.117 |
| Embodiment | Physical | 3.887 | 0.119 |
| Roles | Physical | 3.606 | 0.120 |
| Animacy | Physical | 3.998 | 0.091 |
| Formality | Pragmatic | 3.316 | 0.169 |
| Expertise | Pragmatic | 3.410 | 0.157 |
| Helpfulness | Pragmatic | 3.239 | 0.169 |
| Human (concept) | Entity | 3.610 | 0.057 |
| AI (concept) | Entity | 4.408 | 0.098 |
| Biological | Control | 3.924 | 0.097 |
| Shapes | Control | 3.935 | 0.092 |
| SysPrompt (talkto human) | SysPrompt | 2.494 | 0.110 |
| SysPrompt (talkto AI) | SysPrompt | 3.564 | 0.129 |
| SysPrompt (bare human) | SysPrompt | 2.504 | 0.117 |
| SysPrompt (bare AI) | SysPrompt | 3.643 | 0.111 |

## Methods

- **Analysis**: Standalone Concept Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
