# Exp 3: Raw Contrast Alignment — Cross-Version Comparison

*Generated: 2026-03-04 11:36*

> Raw cosine alignment between human-AI contrast vectors and probe weight vectors. No entity baseline subtraction. Concept direction = mean(human) - mean(AI) per layer.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.338 | 3.191 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.474 | 4.906 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.253 | 0.376 |
| Emotions | Mental | 0.247 | 0.366 |
| Agency | Mental | 0.271 | 0.444 |
| Intentions | Mental | 0.302 | 0.560 |
| Prediction | Mental | 0.336 | 0.464 |
| Cognitive | Mental | 0.403 | 0.525 |
| Social | Mental | 0.493 | 0.564 |
| Attention | Mental | 0.401 | 0.497 |
| Embodiment | Physical | 0.250 | 0.350 |
| Roles | Physical | 0.282 | 0.459 |
| Animacy | Physical | 0.306 | 0.441 |
| Formality | Pragmatic | 0.322 | 0.239 |
| Expertise | Pragmatic | 0.330 | 0.439 |
| Helpfulness | Pragmatic | 0.375 | 0.437 |
| Baseline | Control | 0.319 | 0.760 |
| Biological | Control | 0.273 | 0.265 |
| Shapes | Control | 0.272 | 0.475 |
| SysPrompt (labeled) | SysPrompt | 0.638 | 0.569 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 2.553 | 4.541 |
| Emotions | Mental | 2.679 | 3.947 |
| Agency | Mental | 3.309 | 5.253 |
| Intentions | Mental | 2.846 | 4.652 |
| Prediction | Mental | 3.693 | 5.612 |
| Cognitive | Mental | 3.100 | 4.323 |
| Social | Mental | 3.654 | 5.813 |
| Attention | Mental | 3.695 | 5.107 |
| Embodiment | Physical | 2.560 | 5.092 |
| Roles | Physical | 3.474 | 6.367 |
| Animacy | Physical | 2.878 | 6.167 |
| Formality | Pragmatic | 0.201 | 0.284 |
| Expertise | Pragmatic | 0.318 | 0.569 |
| Helpfulness | Pragmatic | 0.452 | 0.317 |
| Baseline | Control | 3.993 | 9.654 |
| Biological | Control | 2.058 | 4.084 |
| Shapes | Control | 0.520 | 0.641 |
| SysPrompt (labeled) | SysPrompt | 5.145 | 3.795 |

## Methods

- **Analysis**: Raw Contrast Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
