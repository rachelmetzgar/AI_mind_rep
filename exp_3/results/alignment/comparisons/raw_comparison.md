# Exp 3: Raw Contrast Alignment — Cross-Version Comparison

*Generated: 2026-02-26 15:47*

> Raw cosine alignment between human-AI contrast vectors and probe weight vectors. No entity baseline subtraction. Concept direction = mean(human) - mean(AI) per layer.

## Summary (Mental Dimensions)

| Version | Reading R² (×10⁻³) | Control R² (×10⁻³) | Description |
|---------|---|---|---|
| Labels | 0.302 | 0.303 | Partner identified as 'human' or 'AI' (no names) |
| Balanced Names | 0.460 | 1.464 | Gender-balanced names (e.g., Alex/Jordan) |
| Balanced GPT | 0.708 | 2.941 | Balanced names with GPT-4 replacing 'AI' partner |
| Names (orig.) | 0.464 | 1.608 | Original Sam/Casey names (deprecated due to name confound) |
| Nonsense Codeword | 0.257 | 0.346 | Nonsense codewords replacing identity labels |
| Nonsense Ignore | 0.228 | 0.372 | Nonsense labels with instruction to ignore them |

## Per-Dimension: Reading Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore |
|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.283 | 0.239 | 0.505 | 0.311 | 0.261 | 0.234 |
| Emotions | Mental | 0.265 | 0.254 | 0.506 | 0.362 | 0.243 | 0.212 |
| Agency | Mental | 0.270 | 0.461 | 0.603 | 0.388 | 0.229 | 0.187 |
| Intentions | Mental | 0.330 | 0.571 | 0.739 | 0.621 | 0.261 | 0.217 |
| Prediction | Mental | 0.241 | 0.561 | 0.789 | 0.509 | 0.238 | 0.220 |
| Cognitive | Mental | 0.323 | 0.452 | 0.732 | 0.504 | 0.329 | 0.258 |
| Social | Mental | 0.357 | 0.661 | 0.854 | 0.511 | 0.204 | 0.198 |
| Attention | Mental | 0.350 | 0.478 | 0.935 | 0.505 | 0.295 | 0.294 |
| Embodiment | Physical | 0.310 | 0.220 | 0.446 | 0.396 | 0.280 | 0.250 |
| Roles | Physical | 0.306 | 0.400 | 0.671 | 0.434 | 0.245 | 0.218 |
| Animacy | Physical | 0.253 | 0.214 | 0.462 | 0.232 | 0.271 | 0.255 |
| Formality | Pragmatic | 0.340 | 0.287 | 0.355 | 0.362 | 0.295 | 0.370 |
| Expertise | Pragmatic | 0.337 | 0.589 | 0.508 | 0.556 | 0.277 | 0.292 |
| Helpfulness | Pragmatic | 0.215 | 0.494 | 0.216 | 0.549 | 0.319 | 0.217 |
| Baseline | Control | 0.220 | 0.319 | 0.599 | 0.255 | 0.215 | 0.178 |
| Biological | Control | 0.291 | 0.183 | 0.321 | 0.273 | 0.234 | 0.262 |
| Shapes | Control | 0.337 | 0.271 | 0.310 | 0.269 | 0.309 | 0.362 |
| SysPrompt (labeled) | SysPrompt | 0.253 | 0.694 | 0.923 | 0.368 | 0.179 | 0.212 |

## Per-Dimension: Control Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore |
|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.250 | 1.108 | 2.397 | 1.471 | 0.308 | 0.395 |
| Emotions | Mental | 0.328 | 1.087 | 2.265 | 1.578 | 0.361 | 0.348 |
| Agency | Mental | 0.333 | 1.409 | 2.651 | 1.526 | 0.410 | 0.418 |
| Intentions | Mental | 0.381 | 1.516 | 2.894 | 1.803 | 0.407 | 0.426 |
| Prediction | Mental | 0.279 | 1.810 | 3.268 | 1.645 | 0.340 | 0.405 |
| Cognitive | Mental | 0.246 | 1.344 | 3.109 | 1.487 | 0.321 | 0.300 |
| Social | Mental | 0.319 | 1.709 | 3.303 | 1.403 | 0.294 | 0.280 |
| Attention | Mental | 0.289 | 1.729 | 3.638 | 1.950 | 0.323 | 0.403 |
| Embodiment | Physical | 0.247 | 0.856 | 1.809 | 1.248 | 0.300 | 0.441 |
| Roles | Physical | 0.271 | 1.261 | 2.596 | 1.421 | 0.288 | 0.356 |
| Animacy | Physical | 0.141 | 0.726 | 1.506 | 0.990 | 0.204 | 0.325 |
| Formality | Pragmatic | 0.314 | 0.634 | 1.261 | 1.169 | 0.256 | 0.292 |
| Expertise | Pragmatic | 0.273 | 0.899 | 1.470 | 1.047 | 0.235 | 0.239 |
| Helpfulness | Pragmatic | 0.229 | 0.695 | 0.502 | 0.641 | 0.248 | 0.178 |
| Baseline | Control | 0.190 | 1.365 | 2.697 | 0.853 | 0.141 | 0.445 |
| Biological | Control | 0.226 | 0.513 | 1.020 | 1.138 | 0.236 | 0.496 |
| Shapes | Control | 0.253 | 0.275 | 0.308 | 0.320 | 0.286 | 0.187 |
| SysPrompt (labeled) | SysPrompt | 0.183 | 2.492 | 2.698 | 1.558 | 0.121 | 0.213 |

## Methods

- **Analysis**: Raw Contrast Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, turn 5, logistic per-layer (reading + control)
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
