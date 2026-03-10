# Exp 3: Standalone Concept Alignment — Cross-Version Comparison

*Generated: 2026-03-09 14:58*

> Alignment between standalone mean activation vectors and probe weights. Standalone prompts describe concepts without entity framing (no 'human' or 'AI' words). Concept vector = mean activation across all prompts for that concept. Tests whether alignment is driven by concept content rather than entity labels.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 1.561 | 1.877 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.220 | 0.161 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 1.678 | 0.223 |
| Emotions | Mental | 1.562 | 0.229 |
| Agency | Mental | 1.575 | 0.223 |
| Intentions | Mental | 1.654 | 0.225 |
| Prediction | Mental | 1.598 | 0.217 |
| Cognitive | Mental | 1.532 | 0.223 |
| Social | Mental | 1.273 | 0.205 |
| Attention | Mental | 1.703 | 0.229 |
| Beliefs | Mental | 1.432 | 0.224 |
| Desires | Mental | 1.644 | 0.211 |
| Goals | Mental | 1.523 | 0.215 |
| Embodiment | Physical | 1.672 | 0.219 |
| Roles | Physical | 1.779 | 0.221 |
| Animacy | Physical | 1.660 | 0.212 |
| Formality | Pragmatic | 1.484 | 0.233 |
| Expertise | Pragmatic | 1.274 | 0.218 |
| Helpfulness | Pragmatic | 1.402 | 0.233 |
| Human (concept) | Entity | 1.679 | 0.212 |
| AI (concept) | Entity | 1.842 | 0.222 |
| Biological | Control | 1.684 | 0.215 |
| Shapes | Control | 1.664 | 0.240 |
| Granite/Sandstone | Control | 1.706 | 0.237 |
| Squares/Triangles | Control | 1.481 | 0.237 |
| Horizontal/Vertical | Control | 1.542 | 0.241 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 1.942 | 0.162 |
| Emotions | Mental | 1.949 | 0.154 |
| Agency | Mental | 1.891 | 0.164 |
| Intentions | Mental | 1.872 | 0.166 |
| Prediction | Mental | 1.916 | 0.168 |
| Cognitive | Mental | 1.904 | 0.157 |
| Social | Mental | 1.693 | 0.164 |
| Attention | Mental | 1.958 | 0.162 |
| Beliefs | Mental | 1.748 | 0.160 |
| Desires | Mental | 1.914 | 0.163 |
| Goals | Mental | 1.867 | 0.150 |
| Embodiment | Physical | 1.998 | 0.162 |
| Roles | Physical | 2.169 | 0.146 |
| Animacy | Physical | 2.078 | 0.169 |
| Formality | Pragmatic | 1.799 | 0.142 |
| Expertise | Pragmatic | 1.703 | 0.167 |
| Helpfulness | Pragmatic | 1.630 | 0.164 |
| Human (concept) | Entity | 2.129 | 0.145 |
| AI (concept) | Entity | 2.556 | 0.146 |
| Biological | Control | 2.193 | 0.178 |
| Shapes | Control | 2.206 | 0.166 |
| Granite/Sandstone | Control | 2.226 | 0.158 |
| Squares/Triangles | Control | 2.081 | 0.166 |
| Horizontal/Vertical | Control | 2.143 | 0.170 |

## Methods

- **Analysis**: Standalone Concept Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/2a_alignment_analysis.py`
