# Exp 3: Raw Contrast Alignment — Cross-Version Comparison

*Generated: 2026-03-04 11:38*

> Raw cosine alignment between human-AI contrast vectors and probe weight vectors. No entity baseline subtraction. Concept direction = mean(human) - mean(AI) per layer.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.378 | 5.543 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.317 | 1.071 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.246 | 0.302 |
| Emotions | Mental | 0.302 | 0.232 |
| Agency | Mental | 0.294 | 0.272 |
| Intentions | Mental | 0.437 | 0.394 |
| Prediction | Mental | 0.311 | 0.337 |
| Cognitive | Mental | 0.435 | 0.355 |
| Social | Mental | 0.503 | 0.298 |
| Attention | Mental | 0.496 | 0.347 |
| Embodiment | Physical | 0.278 | 0.317 |
| Roles | Physical | 0.273 | 0.304 |
| Animacy | Physical | 0.376 | 0.215 |
| Formality | Pragmatic | 0.457 | 0.272 |
| Expertise | Pragmatic | 0.483 | 0.245 |
| Helpfulness | Pragmatic | 0.399 | 0.237 |
| Baseline | Control | 0.589 | 0.360 |
| Biological | Control | 0.245 | 0.293 |
| Shapes | Control | 0.464 | 0.176 |
| SysPrompt (labeled) | SysPrompt | 0.629 | 0.238 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 4.273 | 0.986 |
| Emotions | Mental | 4.735 | 0.683 |
| Agency | Mental | 5.456 | 1.195 |
| Intentions | Mental | 5.194 | 1.460 |
| Prediction | Mental | 6.145 | 0.959 |
| Cognitive | Mental | 5.530 | 1.522 |
| Social | Mental | 6.591 | 0.743 |
| Attention | Mental | 6.424 | 1.022 |
| Embodiment | Physical | 4.108 | 1.073 |
| Roles | Physical | 5.204 | 0.741 |
| Animacy | Physical | 3.565 | 0.449 |
| Formality | Pragmatic | 1.280 | 0.507 |
| Expertise | Pragmatic | 2.351 | 0.941 |
| Helpfulness | Pragmatic | 0.570 | 0.582 |
| Baseline | Control | 5.164 | 0.683 |
| Biological | Control | 2.494 | 0.814 |
| Shapes | Control | 0.517 | 0.345 |
| SysPrompt (labeled) | SysPrompt | 4.982 | 0.457 |

## Methods

- **Analysis**: Raw Contrast Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
