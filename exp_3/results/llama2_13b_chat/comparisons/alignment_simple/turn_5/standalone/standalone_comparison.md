# Exp 3: Standalone Concept Alignment — Cross-Version Comparison

*Generated: 2026-03-10 11:05*

> Alignment between standalone mean activation vectors and probe weights. Standalone prompts describe concepts without entity framing (no 'human' or 'AI' words). Concept vector = mean activation across all prompts for that concept. Tests whether alignment is driven by concept content rather than entity labels.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 1.984 | 2.427 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.229 | 0.154 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 2.009 | 0.231 |
| Emotions | Mental | 1.895 | 0.230 |
| Agency | Mental | 2.096 | 0.232 |
| Intentions | Mental | 1.870 | 0.228 |
| Prediction | Mental | 2.122 | 0.226 |
| Cognitive | Mental | 2.032 | 0.228 |
| Social | Mental | 1.865 | 0.227 |
| Embodiment | Physical | 1.959 | 0.225 |
| Roles | Physical | 1.943 | 0.224 |
| Animacy | Physical | 1.837 | 0.225 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 2.458 | 0.152 |
| Emotions | Mental | 2.339 | 0.157 |
| Agency | Mental | 2.540 | 0.153 |
| Intentions | Mental | 2.287 | 0.154 |
| Prediction | Mental | 2.560 | 0.151 |
| Cognitive | Mental | 2.512 | 0.152 |
| Social | Mental | 2.292 | 0.159 |
| Embodiment | Physical | 2.384 | 0.152 |
| Roles | Physical | 2.344 | 0.156 |
| Animacy | Physical | 2.264 | 0.155 |

## Methods

- **Analysis**: Standalone Concept Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/2a_alignment_analysis.py`
