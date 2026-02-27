# Exp 3: Residual Alignment (Entity Baseline Projected Out) — Cross-Version Comparison

*Generated: 2026-02-26 15:47*

> Same as raw alignment, but with the entity baseline direction (dim 0: 'this is a human/AI') projected out of each concept vector before computing alignment. This removes shared entity-level variance, isolating concept-specific alignment.

## Summary (Mental Dimensions)

| Version | Reading R² (×10⁻³) | Control R² (×10⁻³) | Description |
|---------|---|---|---|
| Labels | 0.277 | 0.324 | Partner identified as 'human' or 'AI' (no names) |
| Balanced Names | 0.388 | 0.806 | Gender-balanced names (e.g., Alex/Jordan) |
| Balanced GPT | 0.443 | 1.267 | Balanced names with GPT-4 replacing 'AI' partner |
| Names (orig.) | 0.403 | 1.165 | Original Sam/Casey names (deprecated due to name confound) |
| Nonsense Codeword | 0.271 | 0.350 | Nonsense codewords replacing identity labels |
| Nonsense Ignore | 0.228 | 0.302 | Nonsense labels with instruction to ignore them |

## Per-Dimension: Reading Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore |
|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.278 | 0.215 | 0.276 | 0.217 | 0.268 | 0.203 |
| Emotions | Mental | 0.250 | 0.218 | 0.302 | 0.292 | 0.226 | 0.194 |
| Agency | Mental | 0.237 | 0.383 | 0.371 | 0.339 | 0.222 | 0.162 |
| Intentions | Mental | 0.279 | 0.473 | 0.467 | 0.544 | 0.273 | 0.209 |
| Prediction | Mental | 0.202 | 0.465 | 0.465 | 0.541 | 0.287 | 0.230 |
| Cognitive | Mental | 0.287 | 0.398 | 0.432 | 0.406 | 0.341 | 0.276 |
| Social | Mental | 0.374 | 0.575 | 0.656 | 0.467 | 0.228 | 0.246 |
| Attention | Mental | 0.313 | 0.380 | 0.574 | 0.422 | 0.318 | 0.303 |
| Embodiment | Physical | 0.264 | 0.191 | 0.322 | 0.289 | 0.262 | 0.224 |
| Roles | Physical | 0.274 | 0.276 | 0.323 | 0.446 | 0.218 | 0.187 |
| Animacy | Physical | 0.257 | 0.515 | 0.399 | 0.167 | 0.230 | 0.253 |
| Formality | Pragmatic | 0.349 | 0.280 | 0.380 | 0.371 | 0.274 | 0.352 |
| Expertise | Pragmatic | 0.337 | 0.677 | 0.499 | 0.549 | 0.271 | 0.299 |
| Helpfulness | Pragmatic | 0.234 | 0.585 | 0.256 | 0.639 | 0.294 | 0.222 |
| Baseline | Control | 0.192 | 0.289 | 0.337 | 0.172 | 0.176 | 0.209 |
| Biological | Control | 0.243 | 0.289 | 0.253 | 0.213 | 0.182 | 0.204 |
| Shapes | Control | 0.336 | 0.277 | 0.337 | 0.264 | 0.299 | 0.350 |
| SysPrompt (labeled) | SysPrompt | 0.250 | 0.544 | 0.627 | 0.308 | 0.191 | 0.240 |

## Per-Dimension: Control Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore |
|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.276 | 0.641 | 0.939 | 1.080 | 0.305 | 0.268 |
| Emotions | Mental | 0.382 | 0.638 | 0.931 | 1.241 | 0.374 | 0.300 |
| Agency | Mental | 0.377 | 0.820 | 1.116 | 1.114 | 0.450 | 0.344 |
| Intentions | Mental | 0.396 | 0.897 | 1.287 | 1.335 | 0.403 | 0.351 |
| Prediction | Mental | 0.279 | 0.975 | 1.376 | 1.208 | 0.358 | 0.299 |
| Cognitive | Mental | 0.244 | 0.748 | 1.383 | 1.031 | 0.310 | 0.257 |
| Social | Mental | 0.361 | 0.793 | 1.408 | 0.834 | 0.290 | 0.267 |
| Attention | Mental | 0.279 | 0.935 | 1.697 | 1.475 | 0.309 | 0.330 |
| Embodiment | Physical | 0.283 | 0.312 | 0.504 | 0.833 | 0.295 | 0.390 |
| Roles | Physical | 0.380 | 0.507 | 0.847 | 1.005 | 0.335 | 0.323 |
| Animacy | Physical | 0.240 | 0.684 | 0.740 | 0.572 | 0.260 | 0.321 |
| Formality | Pragmatic | 0.328 | 0.706 | 1.246 | 1.205 | 0.254 | 0.234 |
| Expertise | Pragmatic | 0.289 | 0.916 | 1.113 | 0.945 | 0.247 | 0.263 |
| Helpfulness | Pragmatic | 0.235 | 0.898 | 0.535 | 0.760 | 0.273 | 0.232 |
| Baseline | Control | 0.159 | 0.880 | 1.351 | 0.470 | 0.161 | 0.430 |
| Biological | Control | 0.243 | 0.442 | 0.540 | 1.017 | 0.246 | 0.380 |
| Shapes | Control | 0.270 | 0.314 | 0.350 | 0.292 | 0.300 | 0.198 |
| SysPrompt (labeled) | SysPrompt | 0.194 | 1.489 | 1.227 | 1.150 | 0.139 | 0.163 |

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
- **Probes**: Exp 2, turn 5, logistic per-layer (reading + control)
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
