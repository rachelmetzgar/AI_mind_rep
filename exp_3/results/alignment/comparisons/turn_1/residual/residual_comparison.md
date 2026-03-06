# Exp 3: Residual Alignment (Entity Baseline Projected Out) — Cross-Version Comparison

*Generated: 2026-03-04 11:36*

> Same as raw alignment, but with the entity baseline direction (dim 0: 'this is a human/AI') projected out of each concept vector before computing alignment. This removes shared entity-level variance, isolating concept-specific alignment.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.357 | 0.862 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.485 | 0.693 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.210 | 0.337 |
| Emotions | Mental | 0.320 | 0.537 |
| Agency | Mental | 0.292 | 0.455 |
| Intentions | Mental | 0.310 | 0.554 |
| Prediction | Mental | 0.388 | 0.459 |
| Cognitive | Mental | 0.344 | 0.401 |
| Social | Mental | 0.586 | 0.675 |
| Attention | Mental | 0.403 | 0.460 |
| Embodiment | Physical | 0.196 | 0.259 |
| Roles | Physical | 0.298 | 0.394 |
| Animacy | Physical | 0.241 | 0.346 |
| Formality | Pragmatic | 0.333 | 0.295 |
| Expertise | Pragmatic | 0.366 | 0.504 |
| Helpfulness | Pragmatic | 0.366 | 0.444 |
| Baseline | Control | 0.227 | 0.357 |
| Biological | Control | 0.195 | 0.175 |
| Shapes | Control | 0.275 | 0.483 |
| SysPrompt (labeled) | SysPrompt | 0.555 | 0.394 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.471 | 0.554 |
| Emotions | Mental | 0.672 | 0.607 |
| Agency | Mental | 0.956 | 0.797 |
| Intentions | Mental | 0.812 | 0.830 |
| Prediction | Mental | 1.064 | 0.746 |
| Cognitive | Mental | 0.844 | 0.741 |
| Social | Mental | 0.991 | 0.606 |
| Attention | Mental | 1.085 | 0.661 |
| Embodiment | Physical | 0.446 | 0.831 |
| Roles | Physical | 0.689 | 0.876 |
| Animacy | Physical | 0.464 | 0.744 |
| Formality | Pragmatic | 0.246 | 0.745 |
| Expertise | Pragmatic | 0.369 | 0.613 |
| Helpfulness | Pragmatic | 0.711 | 0.527 |
| Baseline | Control | 2.261 | 5.855 |
| Biological | Control | 0.442 | 0.512 |
| Shapes | Control | 0.613 | 0.796 |
| SysPrompt (labeled) | SysPrompt | 2.366 | 0.467 |

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
