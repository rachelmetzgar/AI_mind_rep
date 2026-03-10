# Concept-Conversation Alignment: balanced_gpt, turn 5

**Model:** llama2_13b_chat

## Cross-Approach Comparison

| Dimension | Category | A diff | A p_FDR | A d | C diff | C p_FDR | C d | D diff | D p_FDR | D d |
|---|---|---|---|---|---|---|---|---|---|---|
| 10_animacy | Physical | +0.02838 | 0.0001 *** | +0.180 | -0.00402 | 0.0063 ** | -0.140 | +0.02788 | 0.0001 *** | +0.180 |
| 11_formality | Pragmatic | +0.02922 | 0.0001 *** | +0.181 | +0.00170 | 0.0123 * | +0.128 | +0.02855 | 0.0001 *** | +0.181 |
| 12_expertise | Pragmatic | +0.02896 | 0.0001 *** | +0.180 | +0.00128 | 0.1294  | +0.079 | +0.02838 | 0.0001 *** | +0.180 |
| 13_helpfulness | Pragmatic | +0.02894 | 0.0001 *** | +0.181 | +0.00055 | 0.3730  | +0.042 | +0.02838 | 0.0001 *** | +0.181 |
| 14_biological | Bio Ctrl | +0.02816 | 0.0001 *** | +0.179 | -0.00383 | 0.0002 *** | -0.186 | +0.02776 | 0.0001 *** | +0.179 |
| 15_shapes | Shapes | +0.02875 | 0.0001 *** | +0.180 | -0.00094 | 0.1542  | -0.073 | +0.02837 | 0.0001 *** | +0.180 |
| 16_human | Meta | +0.02802 | 0.0001 *** | +0.177 | -0.00380 | 0.0000 *** | -0.279 | +0.02734 | 0.0001 *** | +0.177 |
| 17_ai | Mental | +0.02811 | 0.0001 *** | +0.176 | -0.00380 | 0.0000 *** | -0.244 | +0.02757 | 0.0001 *** | +0.176 |
| 18_attention | SysPrompt | +0.02889 | 0.0001 *** | +0.180 | +0.00066 | 0.3646  | +0.044 | +0.02840 | 0.0001 *** | +0.180 |
| 1_phenomenology | Mental | +0.02883 | 0.0001 *** | +0.181 | -0.00112 | 0.3030  | -0.052 | +0.02832 | 0.0001 *** | +0.181 |
| 20_sysprompt_talkto_human | SysPrompt | +0.02908 | 0.0001 *** | +0.182 | +0.00202 | 0.0564  | +0.098 | +0.02904 | 0.0001 *** | +0.182 |
| 21_sysprompt_talkto_ai | SysPrompt | +0.02842 | 0.0001 *** | +0.178 | -0.00088 | 0.3464  | -0.047 | +0.02825 | 0.0001 *** | +0.178 |
| 22_sysprompt_bare_human | SysPrompt | +0.02924 | 0.0001 *** | +0.181 | +0.00209 | 0.0564  | +0.097 | +0.02916 | 0.0001 *** | +0.181 |
| 23_sysprompt_bare_ai | SysPrompt | +0.02844 | 0.0001 *** | +0.177 | -0.00112 | 0.2334  | -0.062 | +0.02818 | 0.0001 *** | +0.177 |
| 25_beliefs | Mental | +0.02890 | 0.0001 *** | +0.181 | -0.00018 | 0.7530  | -0.014 | +0.02825 | 0.0001 *** | +0.180 |
| 26_desires | Mental | +0.02905 | 0.0001 *** | +0.182 | +0.00153 | 0.0977  | +0.086 | +0.02850 | 0.0001 *** | +0.182 |
| 27_goals | Mental | +0.02948 | 0.0001 *** | +0.182 | +0.00389 | 0.0009 *** | +0.167 | +0.02888 | 0.0001 *** | +0.182 |
| 2_emotions | Mental | +0.02928 | 0.0001 *** | +0.182 | +0.00098 | 0.2960  | +0.053 | +0.02867 | 0.0001 *** | +0.182 |
| 30_granite_sandstone | Orthogonal Ctrl | +0.02881 | 0.0001 *** | +0.181 | -0.00041 | 0.5593  | -0.027 | +0.02850 | 0.0001 *** | +0.181 |
| 31_squares_triangles | Orthogonal Ctrl | +0.02851 | 0.0001 *** | +0.180 | -0.00155 | 0.0288 * | -0.111 | +0.02815 | 0.0001 *** | +0.180 |
| 32_horizontal_vertical | Shapes | +0.02869 | 0.0001 *** | +0.180 | -0.00061 | 0.2960  | -0.054 | +0.02830 | 0.0001 *** | +0.180 |
| 3_agency | Mental | +0.02926 | 0.0001 *** | +0.181 | +0.00213 | 0.0205 * | +0.119 | +0.02845 | 0.0001 *** | +0.181 |
| 4_intentions | Mental | +0.02945 | 0.0001 *** | +0.182 | +0.00501 | 0.0001 *** | +0.194 | +0.02888 | 0.0001 *** | +0.182 |
| 5_prediction | Mental | +0.02890 | 0.0001 *** | +0.180 | +0.00139 | 0.0287 * | +0.112 | +0.02825 | 0.0001 *** | +0.180 |
| 6_cognitive | Mental | +0.02917 | 0.0001 *** | +0.180 | +0.00232 | 0.0114 * | +0.130 | +0.02844 | 0.0001 *** | +0.180 |
| 7_social | Mental | +0.02886 | 0.0001 *** | +0.180 | +0.00116 | 0.1542  | +0.073 | +0.02816 | 0.0001 *** | +0.180 |
| 8_embodiment | Physical | +0.02852 | 0.0001 *** | +0.180 | -0.00366 | 0.0009 *** | -0.169 | +0.02790 | 0.0001 *** | +0.180 |
| 9_roles | Physical | +0.02907 | 0.0001 *** | +0.180 | +0.00297 | 0.0057 ** | +0.143 | +0.02841 | 0.0001 *** | +0.180 |

## Approach A

| Dimension | Category | H-A diff | p | p_FDR | d |
|---|---|---|---|---|---|
| 27_goals | Mental | +0.02948 | 0.0000 | 0.0001 *** | +0.182 |
| 4_intentions | Mental | +0.02945 | 0.0000 | 0.0001 *** | +0.182 |
| 2_emotions | Mental | +0.02928 | 0.0000 | 0.0001 *** | +0.182 |
| 3_agency | Mental | +0.02926 | 0.0001 | 0.0001 *** | +0.181 |
| 22_sysprompt_bare_human | SysPrompt | +0.02924 | 0.0001 | 0.0001 *** | +0.181 |
| 11_formality | Pragmatic | +0.02922 | 0.0001 | 0.0001 *** | +0.181 |
| 6_cognitive | Mental | +0.02917 | 0.0001 | 0.0001 *** | +0.180 |
| 20_sysprompt_talkto_human | SysPrompt | +0.02908 | 0.0001 | 0.0001 *** | +0.182 |
| 9_roles | Physical | +0.02907 | 0.0001 | 0.0001 *** | +0.180 |
| 26_desires | Mental | +0.02905 | 0.0001 | 0.0001 *** | +0.182 |
| 12_expertise | Pragmatic | +0.02896 | 0.0001 | 0.0001 *** | +0.180 |
| 13_helpfulness | Pragmatic | +0.02894 | 0.0001 | 0.0001 *** | +0.181 |
| 5_prediction | Mental | +0.02890 | 0.0001 | 0.0001 *** | +0.180 |
| 25_beliefs | Mental | +0.02890 | 0.0001 | 0.0001 *** | +0.181 |
| 18_attention | SysPrompt | +0.02889 | 0.0001 | 0.0001 *** | +0.180 |
| 7_social | Mental | +0.02886 | 0.0001 | 0.0001 *** | +0.180 |
| 1_phenomenology | Mental | +0.02883 | 0.0001 | 0.0001 *** | +0.181 |
| 30_granite_sandstone | Orthogonal Ctrl | +0.02881 | 0.0001 | 0.0001 *** | +0.181 |
| 15_shapes | Shapes | +0.02875 | 0.0001 | 0.0001 *** | +0.180 |
| 32_horizontal_vertical | Shapes | +0.02869 | 0.0001 | 0.0001 *** | +0.180 |
| 8_embodiment | Physical | +0.02852 | 0.0001 | 0.0001 *** | +0.180 |
| 31_squares_triangles | Orthogonal Ctrl | +0.02851 | 0.0001 | 0.0001 *** | +0.180 |
| 23_sysprompt_bare_ai | SysPrompt | +0.02844 | 0.0001 | 0.0001 *** | +0.177 |
| 21_sysprompt_talkto_ai | SysPrompt | +0.02842 | 0.0001 | 0.0001 *** | +0.178 |
| 10_animacy | Physical | +0.02838 | 0.0001 | 0.0001 *** | +0.180 |
| 14_biological | Bio Ctrl | +0.02816 | 0.0001 | 0.0001 *** | +0.179 |
| 17_ai | Mental | +0.02811 | 0.0001 | 0.0001 *** | +0.176 |
| 16_human | Meta | +0.02802 | 0.0001 | 0.0001 *** | +0.177 |

28/28 significant (p<.05), 28/28 after FDR

## Approach C

| Dimension | Category | H-A diff | p | p_FDR | d |
|---|---|---|---|---|---|
| 4_intentions | Mental | +0.00501 | 0.0000 | 0.0001 *** | +0.194 |
| 27_goals | Mental | +0.00389 | 0.0002 | 0.0009 *** | +0.167 |
| 9_roles | Physical | +0.00297 | 0.0014 | 0.0057 ** | +0.143 |
| 6_cognitive | Mental | +0.00232 | 0.0037 | 0.0114 * | +0.130 |
| 3_agency | Mental | +0.00213 | 0.0081 | 0.0205 * | +0.119 |
| 22_sysprompt_bare_human | SysPrompt | +0.00209 | 0.0302 | 0.0564  | +0.097 |
| 20_sysprompt_talkto_human | SysPrompt | +0.00202 | 0.0284 | 0.0564  | +0.098 |
| 11_formality | Pragmatic | +0.00170 | 0.0044 | 0.0123 * | +0.128 |
| 26_desires | Mental | +0.00153 | 0.0559 | 0.0977  | +0.086 |
| 5_prediction | Mental | +0.00139 | 0.0123 | 0.0287 * | +0.112 |
| 12_expertise | Pragmatic | +0.00128 | 0.0786 | 0.1294  | +0.079 |
| 7_social | Mental | +0.00116 | 0.1015 | 0.1542  | +0.073 |
| 2_emotions | Mental | +0.00098 | 0.2326 | 0.2960  | +0.053 |
| 18_attention | SysPrompt | +0.00066 | 0.3255 | 0.3646  | +0.044 |
| 13_helpfulness | Pragmatic | +0.00055 | 0.3464 | 0.3730  | +0.042 |
| 25_beliefs | Mental | -0.00018 | 0.7530 | 0.7530  | -0.014 |
| 30_granite_sandstone | Orthogonal Ctrl | -0.00041 | 0.5393 | 0.5593  | -0.027 |
| 32_horizontal_vertical | Shapes | -0.00061 | 0.2274 | 0.2960  | -0.054 |
| 21_sysprompt_talkto_ai | SysPrompt | -0.00088 | 0.2969 | 0.3464  | -0.047 |
| 15_shapes | Shapes | -0.00094 | 0.1046 | 0.1542  | -0.073 |
| 1_phenomenology | Mental | -0.00112 | 0.2489 | 0.3030  | -0.052 |
| 23_sysprompt_bare_ai | SysPrompt | -0.00112 | 0.1667 | 0.2334  | -0.062 |
| 31_squares_triangles | Orthogonal Ctrl | -0.00155 | 0.0134 | 0.0288 * | -0.111 |
| 8_embodiment | Physical | -0.00366 | 0.0002 | 0.0009 *** | -0.169 |
| 17_ai | Mental | -0.00380 | 0.0000 | 0.0000 *** | -0.244 |
| 16_human | Meta | -0.00380 | 0.0000 | 0.0000 *** | -0.279 |
| 14_biological | Bio Ctrl | -0.00383 | 0.0000 | 0.0002 *** | -0.186 |
| 10_animacy | Physical | -0.00402 | 0.0018 | 0.0063 ** | -0.140 |

15/28 significant (p<.05), 13/28 after FDR

## Approach D

| Dimension | Category | H-A diff | p | p_FDR | d |
|---|---|---|---|---|---|
| 22_sysprompt_bare_human | SysPrompt | +0.02916 | 0.0001 | 0.0001 *** | +0.181 |
| 20_sysprompt_talkto_human | SysPrompt | +0.02904 | 0.0001 | 0.0001 *** | +0.182 |
| 27_goals | Mental | +0.02888 | 0.0000 | 0.0001 *** | +0.182 |
| 4_intentions | Mental | +0.02888 | 0.0000 | 0.0001 *** | +0.182 |
| 2_emotions | Mental | +0.02867 | 0.0000 | 0.0001 *** | +0.182 |
| 11_formality | Pragmatic | +0.02855 | 0.0001 | 0.0001 *** | +0.181 |
| 30_granite_sandstone | Orthogonal Ctrl | +0.02850 | 0.0001 | 0.0001 *** | +0.181 |
| 26_desires | Mental | +0.02850 | 0.0001 | 0.0001 *** | +0.182 |
| 3_agency | Mental | +0.02845 | 0.0001 | 0.0001 *** | +0.181 |
| 6_cognitive | Mental | +0.02844 | 0.0001 | 0.0001 *** | +0.180 |
| 9_roles | Physical | +0.02841 | 0.0001 | 0.0001 *** | +0.180 |
| 18_attention | SysPrompt | +0.02840 | 0.0001 | 0.0001 *** | +0.180 |
| 12_expertise | Pragmatic | +0.02838 | 0.0001 | 0.0001 *** | +0.180 |
| 13_helpfulness | Pragmatic | +0.02838 | 0.0001 | 0.0001 *** | +0.181 |
| 15_shapes | Shapes | +0.02837 | 0.0001 | 0.0001 *** | +0.180 |
| 1_phenomenology | Mental | +0.02832 | 0.0001 | 0.0001 *** | +0.181 |
| 32_horizontal_vertical | Shapes | +0.02830 | 0.0001 | 0.0001 *** | +0.180 |
| 21_sysprompt_talkto_ai | SysPrompt | +0.02825 | 0.0001 | 0.0001 *** | +0.178 |
| 25_beliefs | Mental | +0.02825 | 0.0001 | 0.0001 *** | +0.180 |
| 5_prediction | Mental | +0.02825 | 0.0001 | 0.0001 *** | +0.180 |
| 23_sysprompt_bare_ai | SysPrompt | +0.02818 | 0.0001 | 0.0001 *** | +0.177 |
| 7_social | Mental | +0.02816 | 0.0001 | 0.0001 *** | +0.180 |
| 31_squares_triangles | Orthogonal Ctrl | +0.02815 | 0.0001 | 0.0001 *** | +0.180 |
| 8_embodiment | Physical | +0.02790 | 0.0001 | 0.0001 *** | +0.180 |
| 10_animacy | Physical | +0.02788 | 0.0001 | 0.0001 *** | +0.180 |
| 14_biological | Bio Ctrl | +0.02776 | 0.0001 | 0.0001 *** | +0.179 |
| 17_ai | Mental | +0.02757 | 0.0001 | 0.0001 *** | +0.176 |
| 16_human | Meta | +0.02734 | 0.0001 | 0.0001 *** | +0.177 |

28/28 significant (p<.05), 28/28 after FDR

## Prompt-Level Analysis (Approach D)

- **10_animacy**: 40/40 prompts significant
- **11_formality**: 40/40 prompts significant
- **12_expertise**: 40/40 prompts significant
- **13_helpfulness**: 40/40 prompts significant
- **14_biological**: 40/40 prompts significant
- **15_shapes**: 40/40 prompts significant
- **16_human**: 40/40 prompts significant
- **17_ai**: 40/40 prompts significant
- **18_attention**: 40/40 prompts significant
- **1_phenomenology**: 40/40 prompts significant
- **20_sysprompt_talkto_human**: 14/14 prompts significant
- **21_sysprompt_talkto_ai**: 14/14 prompts significant
- **22_sysprompt_bare_human**: 14/14 prompts significant
- **23_sysprompt_bare_ai**: 14/14 prompts significant
- **25_beliefs**: 40/40 prompts significant
- **26_desires**: 40/40 prompts significant
- **27_goals**: 40/40 prompts significant
- **2_emotions**: 40/40 prompts significant
- **30_granite_sandstone**: 40/40 prompts significant
- **31_squares_triangles**: 40/40 prompts significant
- **32_horizontal_vertical**: 40/40 prompts significant
- **3_agency**: 40/40 prompts significant
- **4_intentions**: 40/40 prompts significant
- **5_prediction**: 40/40 prompts significant
- **6_cognitive**: 40/40 prompts significant
- **7_social**: 40/40 prompts significant
- **8_embodiment**: 40/40 prompts significant
- **9_roles**: 40/40 prompts significant
