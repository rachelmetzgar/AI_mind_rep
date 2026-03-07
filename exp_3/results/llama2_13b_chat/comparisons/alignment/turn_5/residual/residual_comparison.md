# Exp 3: Residual Alignment (Entity Baseline Projected Out) — Cross-Version Comparison

*Generated: 2026-03-04 11:34*

> Same as raw alignment, but with the entity baseline direction (dim 0: 'this is a human/AI') projected out of each concept vector before computing alignment. This removes shared entity-level variance, isolating concept-specific alignment.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.507 | 0.992 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.217 | 0.313 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.324 | 0.250 |
| Emotions | Mental | 0.355 | 0.176 |
| Agency | Mental | 0.486 | 0.142 |
| Intentions | Mental | 0.540 | 0.191 |
| Prediction | Mental | 0.445 | 0.248 |
| Cognitive | Mental | 0.589 | 0.263 |
| Social | Mental | 0.663 | 0.202 |
| Attention | Mental | 0.656 | 0.265 |
| Embodiment | Physical | 0.249 | 0.203 |
| Roles | Physical | 0.380 | 0.151 |
| Animacy | Physical | 0.377 | 0.219 |
| Formality | Pragmatic | 0.342 | 0.241 |
| Expertise | Pragmatic | 0.435 | 0.288 |
| Helpfulness | Pragmatic | 0.301 | 0.235 |
| Baseline | Control | 0.400 | 0.159 |
| Biological | Control | 0.313 | 0.164 |
| Shapes | Control | 0.329 | 0.194 |
| SysPrompt (labeled) | SysPrompt | 0.667 | 0.189 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.542 | 0.254 |
| Emotions | Mental | 0.541 | 0.342 |
| Agency | Mental | 0.891 | 0.414 |
| Intentions | Mental | 1.073 | 0.362 |
| Prediction | Mental | 0.936 | 0.327 |
| Cognitive | Mental | 1.255 | 0.255 |
| Social | Mental | 1.269 | 0.293 |
| Attention | Mental | 1.430 | 0.257 |
| Embodiment | Physical | 0.313 | 0.266 |
| Roles | Physical | 0.584 | 0.300 |
| Animacy | Physical | 0.566 | 0.247 |
| Formality | Pragmatic | 0.341 | 0.227 |
| Expertise | Pragmatic | 0.780 | 0.229 |
| Helpfulness | Pragmatic | 0.327 | 0.232 |
| Baseline | Control | 1.539 | 0.174 |
| Biological | Control | 0.473 | 0.232 |
| Shapes | Control | 0.316 | 0.205 |
| SysPrompt (labeled) | SysPrompt | 1.315 | 0.080 |

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
