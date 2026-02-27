# Exp 3: Residual Alignment (Entity Baseline Projected Out) — Cross-Version Comparison

*Generated: 2026-02-27 12:44*

> Same as raw alignment, but with the entity baseline direction (dim 0: 'this is a human/AI') projected out of each concept vector before computing alignment. This removes shared entity-level variance, isolating concept-specific alignment.

## Summary (Mental Dimensions)

| Version | Reading R² (×10⁻³) | Control R² (×10⁻³) | Description |
|---------|---|---|---|
| Labels | 0.242 | 0.335 | Partner identified as 'human' or 'AI' (no names) |
| Balanced Names | 0.393 | 0.665 | Gender-balanced names (e.g., Alex/Jordan) |
| Balanced GPT | 0.380 | 0.985 | Balanced names with GPT-4 replacing 'AI' partner |
| Names (orig.) | 0.397 | 0.880 | Original Sam/Casey names (deprecated due to name confound) |
| Nonsense Codeword | 0.210 | 0.314 | Nonsense codewords replacing identity labels |
| Nonsense Ignore | 0.166 | 0.324 | Nonsense labels with instruction to ignore them |

## Per-Dimension: Reading Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore |
|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.241 | 0.211 | 0.188 | 0.184 | 0.214 | 0.162 |
| Emotions | Mental | 0.208 | 0.197 | 0.228 | 0.252 | 0.153 | 0.108 |
| Agency | Mental | 0.172 | 0.399 | 0.332 | 0.321 | 0.139 | 0.088 |
| Intentions | Mental | 0.220 | 0.480 | 0.407 | 0.558 | 0.192 | 0.137 |
| Prediction | Mental | 0.172 | 0.489 | 0.368 | 0.574 | 0.254 | 0.192 |
| Cognitive | Mental | 0.249 | 0.397 | 0.393 | 0.403 | 0.266 | 0.212 |
| Social | Mental | 0.384 | 0.567 | 0.614 | 0.478 | 0.197 | 0.223 |
| Attention | Mental | 0.287 | 0.402 | 0.509 | 0.410 | 0.264 | 0.202 |
| Embodiment | Physical | 0.188 | 0.171 | 0.297 | 0.250 | 0.195 | 0.157 |
| Roles | Physical | 0.243 | 0.274 | 0.261 | 0.460 | 0.145 | 0.145 |
| Animacy | Physical | 0.217 | 0.577 | 0.373 | 0.162 | 0.220 | 0.265 |
| Formality | Pragmatic | 0.328 | 0.251 | 0.276 | 0.322 | 0.230 | 0.288 |
| Expertise | Pragmatic | 0.358 | 0.726 | 0.392 | 0.537 | 0.268 | 0.210 |
| Helpfulness | Pragmatic | 0.255 | 0.662 | 0.268 | 0.691 | 0.262 | 0.221 |
| Baseline | Control | 0.199 | 0.307 | 0.346 | 0.175 | 0.163 | 0.207 |
| Biological | Control | 0.129 | 0.268 | 0.223 | 0.123 | 0.151 | 0.146 |
| Shapes | Control | 0.263 | 0.211 | 0.272 | 0.187 | 0.198 | 0.254 |
| SysPrompt (labeled) | SysPrompt | 0.186 | 0.518 | 0.650 | 0.327 | 0.180 | 0.226 |

## Per-Dimension: Control Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore |
|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.283 | 0.430 | 0.541 | 0.683 | 0.249 | 0.288 |
| Emotions | Mental | 0.406 | 0.427 | 0.543 | 0.826 | 0.342 | 0.308 |
| Agency | Mental | 0.393 | 0.712 | 0.891 | 0.798 | 0.421 | 0.363 |
| Intentions | Mental | 0.405 | 0.794 | 1.060 | 1.124 | 0.362 | 0.373 |
| Prediction | Mental | 0.290 | 0.794 | 0.929 | 0.955 | 0.331 | 0.333 |
| Cognitive | Mental | 0.245 | 0.667 | 1.245 | 0.851 | 0.255 | 0.279 |
| Social | Mental | 0.362 | 0.685 | 1.252 | 0.683 | 0.295 | 0.294 |
| Attention | Mental | 0.296 | 0.811 | 1.422 | 1.125 | 0.259 | 0.354 |
| Embodiment | Physical | 0.286 | 0.201 | 0.327 | 0.513 | 0.266 | 0.364 |
| Roles | Physical | 0.384 | 0.368 | 0.579 | 0.724 | 0.305 | 0.321 |
| Animacy | Physical | 0.228 | 0.583 | 0.597 | 0.238 | 0.251 | 0.332 |
| Formality | Pragmatic | 0.309 | 0.342 | 0.390 | 0.413 | 0.223 | 0.245 |
| Expertise | Pragmatic | 0.300 | 0.905 | 0.760 | 0.737 | 0.228 | 0.235 |
| Helpfulness | Pragmatic | 0.261 | 0.967 | 0.326 | 0.778 | 0.232 | 0.231 |
| Baseline | Control | 0.166 | 1.008 | 1.547 | 0.527 | 0.172 | 0.253 |
| Biological | Control | 0.195 | 0.320 | 0.500 | 0.670 | 0.233 | 0.276 |
| Shapes | Control | 0.275 | 0.306 | 0.321 | 0.180 | 0.206 | 0.221 |
| SysPrompt (labeled) | SysPrompt | 0.162 | 1.604 | 1.314 | 1.285 | 0.081 | 0.161 |

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
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
