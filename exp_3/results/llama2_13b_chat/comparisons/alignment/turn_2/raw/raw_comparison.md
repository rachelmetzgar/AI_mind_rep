# Exp 3: Raw Contrast Alignment — Cross-Version Comparison

*Generated: 2026-03-04 11:37*

> Raw cosine alignment between human-AI contrast vectors and probe weight vectors. No entity baseline subtraction. Concept direction = mean(human) - mean(AI) per layer.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.412 | 7.754 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.648 | 1.276 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.276 | 0.633 |
| Emotions | Mental | 0.257 | 0.528 |
| Agency | Mental | 0.344 | 0.607 |
| Intentions | Mental | 0.453 | 0.789 |
| Prediction | Mental | 0.365 | 0.633 |
| Cognitive | Mental | 0.529 | 0.632 |
| Social | Mental | 0.481 | 0.655 |
| Attention | Mental | 0.590 | 0.707 |
| Embodiment | Physical | 0.191 | 0.469 |
| Roles | Physical | 0.292 | 0.573 |
| Animacy | Physical | 0.310 | 0.256 |
| Formality | Pragmatic | 0.464 | 0.425 |
| Expertise | Pragmatic | 0.590 | 0.213 |
| Helpfulness | Pragmatic | 0.346 | 0.249 |
| Baseline | Control | 0.516 | 0.479 |
| Biological | Control | 0.209 | 0.405 |
| Shapes | Control | 0.653 | 0.348 |
| SysPrompt (labeled) | SysPrompt | 1.002 | 0.293 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 5.739 | 1.307 |
| Emotions | Mental | 6.696 | 1.172 |
| Agency | Mental | 7.906 | 1.351 |
| Intentions | Mental | 7.491 | 1.406 |
| Prediction | Mental | 8.696 | 1.240 |
| Cognitive | Mental | 7.500 | 0.966 |
| Social | Mental | 9.183 | 1.602 |
| Attention | Mental | 8.817 | 1.163 |
| Embodiment | Physical | 5.838 | 1.435 |
| Roles | Physical | 7.672 | 1.230 |
| Animacy | Physical | 5.201 | 1.095 |
| Formality | Pragmatic | 1.569 | 0.606 |
| Expertise | Pragmatic | 4.327 | 0.124 |
| Helpfulness | Pragmatic | 0.794 | 0.742 |
| Baseline | Control | 7.302 | 1.412 |
| Biological | Control | 3.487 | 1.454 |
| Shapes | Control | 0.565 | 0.298 |
| SysPrompt (labeled) | SysPrompt | 7.380 | 0.407 |

## Methods

- **Analysis**: Raw Contrast Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
