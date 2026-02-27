# Exp 3: Raw Contrast Alignment — Cross-Version Comparison

*Generated: 2026-02-27 12:44*

> Raw cosine alignment between human-AI contrast vectors and probe weight vectors. No entity baseline subtraction. Concept direction = mean(human) - mean(AI) per layer.

## Summary (Mental Dimensions)

| Version | Reading R² (×10⁻³) | Control R² (×10⁻³) | Description |
|---------|---|---|---|
| Labels | 0.251 | 0.325 | Partner identified as 'human' or 'AI' (no names) |
| Balanced Names | 0.454 | 1.415 | Gender-balanced names (e.g., Alex/Jordan) |
| Balanced GPT | 0.631 | 2.810 | Balanced names with GPT-4 replacing 'AI' partner |
| Names (orig.) | 0.435 | 1.263 | Original Sam/Casey names (deprecated due to name confound) |
| Nonsense Codeword | 0.190 | 0.313 | Nonsense codewords replacing identity labels |
| Nonsense Ignore | 0.177 | 0.411 | Nonsense labels with instruction to ignore them |

## Per-Dimension: Reading Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore |
|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.229 | 0.219 | 0.400 | 0.264 | 0.201 | 0.202 |
| Emotions | Mental | 0.210 | 0.219 | 0.422 | 0.305 | 0.167 | 0.137 |
| Agency | Mental | 0.189 | 0.470 | 0.558 | 0.347 | 0.143 | 0.126 |
| Intentions | Mental | 0.255 | 0.571 | 0.673 | 0.613 | 0.172 | 0.158 |
| Prediction | Mental | 0.197 | 0.577 | 0.671 | 0.501 | 0.189 | 0.187 |
| Cognitive | Mental | 0.274 | 0.438 | 0.693 | 0.484 | 0.251 | 0.207 |
| Social | Mental | 0.350 | 0.644 | 0.773 | 0.493 | 0.164 | 0.178 |
| Attention | Mental | 0.305 | 0.494 | 0.862 | 0.469 | 0.230 | 0.222 |
| Embodiment | Physical | 0.216 | 0.183 | 0.404 | 0.338 | 0.207 | 0.205 |
| Roles | Physical | 0.261 | 0.393 | 0.611 | 0.410 | 0.173 | 0.189 |
| Animacy | Physical | 0.197 | 0.202 | 0.398 | 0.208 | 0.254 | 0.274 |
| Formality | Pragmatic | 0.314 | 0.254 | 0.244 | 0.305 | 0.245 | 0.316 |
| Expertise | Pragmatic | 0.349 | 0.614 | 0.372 | 0.529 | 0.272 | 0.212 |
| Helpfulness | Pragmatic | 0.236 | 0.559 | 0.220 | 0.593 | 0.290 | 0.216 |
| Baseline | Control | 0.216 | 0.331 | 0.596 | 0.250 | 0.187 | 0.187 |
| Biological | Control | 0.182 | 0.139 | 0.293 | 0.177 | 0.207 | 0.219 |
| Shapes | Control | 0.266 | 0.198 | 0.234 | 0.182 | 0.215 | 0.275 |
| SysPrompt (labeled) | SysPrompt | 0.237 | 0.674 | 0.936 | 0.385 | 0.183 | 0.221 |

## Per-Dimension: Control Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore |
|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.263 | 0.962 | 2.105 | 0.996 | 0.258 | 0.441 |
| Emotions | Mental | 0.355 | 0.930 | 1.988 | 1.093 | 0.333 | 0.376 |
| Agency | Mental | 0.355 | 1.386 | 2.565 | 1.142 | 0.377 | 0.449 |
| Intentions | Mental | 0.408 | 1.503 | 2.820 | 1.540 | 0.378 | 0.476 |
| Prediction | Mental | 0.302 | 1.751 | 2.982 | 1.324 | 0.312 | 0.446 |
| Cognitive | Mental | 0.265 | 1.350 | 3.138 | 1.260 | 0.276 | 0.335 |
| Social | Mental | 0.341 | 1.737 | 3.341 | 1.219 | 0.292 | 0.314 |
| Attention | Mental | 0.314 | 1.704 | 3.541 | 1.532 | 0.275 | 0.451 |
| Embodiment | Physical | 0.255 | 0.810 | 1.757 | 0.865 | 0.269 | 0.471 |
| Roles | Physical | 0.274 | 1.235 | 2.504 | 1.071 | 0.253 | 0.387 |
| Animacy | Physical | 0.121 | 0.626 | 1.386 | 0.576 | 0.179 | 0.354 |
| Formality | Pragmatic | 0.298 | 0.248 | 0.353 | 0.308 | 0.232 | 0.301 |
| Expertise | Pragmatic | 0.289 | 0.874 | 1.100 | 0.784 | 0.214 | 0.213 |
| Helpfulness | Pragmatic | 0.253 | 0.732 | 0.269 | 0.636 | 0.203 | 0.174 |
| Baseline | Control | 0.165 | 1.538 | 2.920 | 0.792 | 0.143 | 0.293 |
| Biological | Control | 0.191 | 0.393 | 1.032 | 0.727 | 0.217 | 0.398 |
| Shapes | Control | 0.264 | 0.259 | 0.248 | 0.177 | 0.191 | 0.208 |
| SysPrompt (labeled) | SysPrompt | 0.189 | 2.751 | 2.959 | 1.683 | 0.077 | 0.200 |

## Methods

- **Analysis**: Raw Contrast Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, turn 5, logistic per-layer (reading + control)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
