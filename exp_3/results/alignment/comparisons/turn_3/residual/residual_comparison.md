# Exp 3: Residual Alignment (Entity Baseline Projected Out) — Cross-Version Comparison

*Generated: 2026-03-04 11:38*

> Same as raw alignment, but with the entity baseline direction (dim 0: 'this is a human/AI') projected out of each concept vector before computing alignment. This removes shared entity-level variance, isolating concept-specific alignment.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.698 | 1.925 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.243 | 0.719 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.583 | 0.206 |
| Emotions | Mental | 0.791 | 0.189 |
| Agency | Mental | 0.667 | 0.209 |
| Intentions | Mental | 0.728 | 0.276 |
| Prediction | Mental | 0.630 | 0.251 |
| Cognitive | Mental | 0.452 | 0.269 |
| Social | Mental | 0.951 | 0.292 |
| Attention | Mental | 0.781 | 0.251 |
| Embodiment | Physical | 0.336 | 0.209 |
| Roles | Physical | 0.824 | 0.202 |
| Animacy | Physical | 0.277 | 0.246 |
| Formality | Pragmatic | 0.540 | 0.297 |
| Expertise | Pragmatic | 0.584 | 0.255 |
| Helpfulness | Pragmatic | 0.361 | 0.263 |
| Baseline | Control | 0.280 | 0.244 |
| Biological | Control | 0.178 | 0.248 |
| Shapes | Control | 0.480 | 0.192 |
| SysPrompt (labeled) | SysPrompt | 0.622 | 0.195 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 1.049 | 0.507 |
| Emotions | Mental | 1.571 | 0.383 |
| Agency | Mental | 1.909 | 0.762 |
| Intentions | Mental | 1.870 | 1.058 |
| Prediction | Mental | 2.278 | 0.514 |
| Cognitive | Mental | 1.928 | 1.361 |
| Social | Mental | 2.310 | 0.441 |
| Attention | Mental | 2.485 | 0.722 |
| Embodiment | Physical | 0.878 | 0.688 |
| Roles | Physical | 1.318 | 0.398 |
| Animacy | Physical | 0.525 | 0.302 |
| Formality | Pragmatic | 0.916 | 0.661 |
| Expertise | Pragmatic | 1.544 | 1.069 |
| Helpfulness | Pragmatic | 0.579 | 0.798 |
| Baseline | Control | 2.900 | 0.399 |
| Biological | Control | 0.485 | 0.461 |
| Shapes | Control | 0.487 | 0.417 |
| SysPrompt (labeled) | SysPrompt | 1.888 | 0.362 |

## Entity Overlap (Mean |cosine| with baseline)

| Dimension | Entity Overlap |
|---|---|
| Phenomenology | 0.5669 |
| Emotions | 0.4941 |
| Agency | 0.5390 |
| Intentions | 0.4166 |
| Prediction | 0.5678 |
| Cognitive | 0.2396 |
| Social | 0.5568 |
| Attention | 0.4183 |
| Embodiment | 0.6324 |
| Roles | 0.6834 |
| Animacy | 0.7429 |
| Formality | 0.2335 |
| Expertise | 0.1731 |
| Helpfulness | 0.1964 |
| Baseline | 0.9756 |
| Biological | 0.5146 |
| Shapes | 0.1273 |
| SysPrompt (labeled) | 0.2829 |

## Methods

- **Analysis**: Residual Alignment (Entity Baseline Projected Out)
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
