# Exp 3: Standalone Concept Alignment — Cross-Version Comparison

*Generated: 2026-03-08 14:55*

> Alignment between standalone mean activation vectors and probe weights. Standalone prompts describe concepts without entity framing (no 'human' or 'AI' words). Concept vector = mean activation across all prompts for that concept. Tests whether alignment is driven by concept content rather than entity labels.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 1.649 | 2.004 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.216 | 0.157 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 1.762 | 0.227 |
| Emotions | Mental | 1.726 | 0.232 |
| Agency | Mental | 1.689 | 0.212 |
| Intentions | Mental | 1.637 | 0.216 |
| Prediction | Mental | 1.665 | 0.215 |
| Cognitive | Mental | 1.562 | 0.205 |
| Social | Mental | 1.456 | 0.205 |
| Attention | Mental | 1.707 | 0.218 |
| Beliefs | Mental | 1.585 | 0.212 |
| Desires | Mental | 1.705 | 0.217 |
| Goals | Mental | 1.647 | 0.215 |
| Embodiment | Physical | 1.806 | 0.217 |
| Roles | Physical | 1.663 | 0.206 |
| Animacy | Physical | 1.782 | 0.215 |
| Formality | Pragmatic | 1.589 | 0.223 |
| Expertise | Pragmatic | 1.543 | 0.211 |
| Helpfulness | Pragmatic | 1.531 | 0.217 |
| Human (concept) | Entity | 1.880 | 0.213 |
| AI (concept) | Entity | 1.940 | 0.214 |
| Biological | Control | 1.702 | 0.217 |
| Shapes | Control | 1.648 | 0.228 |
| Granite/Sandstone | Control | 1.611 | 0.225 |
| Squares/Triangles | Control | 1.602 | 0.224 |
| Horizontal/Vertical | Control | 1.592 | 0.234 |
| SysPrompt (talkto human) | SysPrompt | 1.320 | 0.226 |
| SysPrompt (talkto AI) | SysPrompt | 1.619 | 0.218 |
| SysPrompt (bare human) | SysPrompt | 1.453 | 0.205 |
| SysPrompt (bare AI) | SysPrompt | 1.751 | 0.197 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 2.161 | 0.158 |
| Emotions | Mental | 2.066 | 0.154 |
| Agency | Mental | 2.038 | 0.159 |
| Intentions | Mental | 1.956 | 0.162 |
| Prediction | Mental | 1.982 | 0.153 |
| Cognitive | Mental | 1.982 | 0.158 |
| Social | Mental | 1.819 | 0.158 |
| Attention | Mental | 2.124 | 0.160 |
| Beliefs | Mental | 1.936 | 0.155 |
| Desires | Mental | 1.995 | 0.157 |
| Goals | Mental | 1.989 | 0.158 |
| Embodiment | Physical | 2.242 | 0.168 |
| Roles | Physical | 2.067 | 0.160 |
| Animacy | Physical | 2.298 | 0.162 |
| Formality | Pragmatic | 1.878 | 0.155 |
| Expertise | Pragmatic | 1.962 | 0.155 |
| Helpfulness | Pragmatic | 1.874 | 0.156 |
| Human (concept) | Entity | 2.348 | 0.148 |
| AI (concept) | Entity | 2.719 | 0.148 |
| Biological | Control | 2.277 | 0.175 |
| Shapes | Control | 2.173 | 0.163 |
| Granite/Sandstone | Control | 2.099 | 0.168 |
| Squares/Triangles | Control | 2.133 | 0.162 |
| Horizontal/Vertical | Control | 2.112 | 0.165 |
| SysPrompt (talkto human) | SysPrompt | 1.546 | 0.163 |
| SysPrompt (talkto AI) | SysPrompt | 2.170 | 0.154 |
| SysPrompt (bare human) | SysPrompt | 1.633 | 0.195 |
| SysPrompt (bare AI) | SysPrompt | 2.344 | 0.177 |

## Methods

- **Analysis**: Standalone Concept Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/2a_alignment_analysis.py`
