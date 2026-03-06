# Exp 3: Residual Alignment (Entity Baseline Projected Out) — Cross-Version Comparison

*Generated: 2026-03-04 11:37*

> Same as raw alignment, but with the entity baseline direction (dim 0: 'this is a human/AI') projected out of each concept vector before computing alignment. This removes shared entity-level variance, isolating concept-specific alignment.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.596 | 2.318 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.398 | 0.661 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.441 | 0.302 |
| Emotions | Mental | 0.526 | 0.281 |
| Agency | Mental | 0.584 | 0.321 |
| Intentions | Mental | 0.640 | 0.515 |
| Prediction | Mental | 0.541 | 0.379 |
| Cognitive | Mental | 0.556 | 0.498 |
| Social | Mental | 0.699 | 0.419 |
| Attention | Mental | 0.781 | 0.467 |
| Embodiment | Physical | 0.389 | 0.297 |
| Roles | Physical | 0.469 | 0.229 |
| Animacy | Physical | 0.451 | 0.224 |
| Formality | Pragmatic | 0.527 | 0.538 |
| Expertise | Pragmatic | 0.737 | 0.203 |
| Helpfulness | Pragmatic | 0.357 | 0.315 |
| Baseline | Control | 0.316 | 0.326 |
| Biological | Control | 0.242 | 0.253 |
| Shapes | Control | 0.669 | 0.385 |
| SysPrompt (labeled) | SysPrompt | 0.993 | 0.219 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.905 | 0.552 |
| Emotions | Mental | 1.782 | 0.578 |
| Agency | Mental | 2.326 | 0.654 |
| Intentions | Mental | 2.356 | 0.792 |
| Prediction | Mental | 2.841 | 0.492 |
| Cognitive | Mental | 2.355 | 0.733 |
| Social | Mental | 2.946 | 0.871 |
| Attention | Mental | 3.033 | 0.620 |
| Embodiment | Physical | 0.979 | 0.645 |
| Roles | Physical | 1.733 | 0.387 |
| Animacy | Physical | 0.601 | 0.523 |
| Formality | Pragmatic | 0.879 | 1.020 |
| Expertise | Pragmatic | 3.028 | 0.160 |
| Helpfulness | Pragmatic | 0.782 | 0.930 |
| Baseline | Control | 4.140 | 0.696 |
| Biological | Control | 0.468 | 0.884 |
| Shapes | Control | 0.520 | 0.326 |
| SysPrompt (labeled) | SysPrompt | 2.942 | 0.296 |

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
