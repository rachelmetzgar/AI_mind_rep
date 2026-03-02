# Experiment 2 (You Are Labels Turnwise) — V1 Summary


Generated: 2026-03-02 12:21:01



##
        You Are Labels Turnwise



'You are talking to {type}' + 'Human:'/'AI:' turn prefix




### Quality Assessment Overview


| Strategy | Strength | Assessment | # Sig Metrics | Word Count (B/H/AI) | Fung Interp. (B/H/AI) | Like rate (B/H/AI) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **peak_15** | 2 | **MARGINAL** | 2 | 311 / 266 / 326 *** | 0.0106 / 0.0136 / 0.0083 * | 0.0034 / 0.0046 / 0.0018 ** | 2 significant metric(s) |
| **peak_15** | 4 | **GOOD** | 8 | 298 / 256 / 390 *** | 0.0108 / 0.0235 / 0.0077 *** | 0.0021 / 0.0108 / 0.0015 *** | 8 significant metrics (excl. word_count) |
| **peak_15** | 8 | **DEGRADED** | 18 | 302 / 5 / 589 *** | 0.0111 / 0.0357 / 0.0000 | 0.0028 / 0.0357 / 0.0000 * | Very verbose (AI=589 words); High strength — likely degradation |
| **peak_15** | 16 | **DEGRADED** | 19 | 311 / 553 / 680 *** | 0.0109 / 0.1645 / 0.0000 *** | 0.0026 / 0.1645 / 0.0000 *** | Very verbose (AI=680 words); High strength — likely degradation |


### Strategy: peak_15


*[Figure — see HTML report]*


*[Figure — see HTML report]*



| Metric | Strength 2 | Strength 4 | Strength 8 | Strength 16 |
| --- | --- | --- | --- | --- |
|  | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p |
| **word_count** | 311 | 266 | 326 | 0.0005*** | 298 | 256 | 390 | 0.0000*** | 302 | 5 | 589 | 0.0000*** | 311 | 553 | 680 | 0.0000*** |
| **question_count** | 1 | 1 | 1 | 0.4895 | 1 | 1 | 1 | 0.0026** | 1 | 0 | 0 | 0.0001*** | 1 | 0 | 0 | 0.0007*** |
| **demir_modal_rate** | 0.0190 | 0.0186 | 0.0215 | 0.3940 | 0.0191 | 0.0150 | 0.0200 | 0.1135 | 0.0201 | 0.0000 | 0.0000 | 0.0000*** | 0.0210 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_verb_rate** | 0.0171 | 0.0167 | 0.0174 | 0.9626 | 0.0173 | 0.0178 | 0.0119 | 0.0505 | 0.0172 | 0.0000 | 0.0000 | 0.0000*** | 0.0173 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_adverb_rate** | 0.0068 | 0.0067 | 0.0067 | 0.9960 | 0.0063 | 0.0076 | 0.0060 | 0.6687 | 0.0068 | 0.0000 | 0.0000 | 0.0000*** | 0.0063 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_adjective_rate** | 0.0014 | 0.0009 | 0.0018 | 0.1737 | 0.0010 | 0.0008 | 0.0025 | 0.0067** | 0.0011 | 0.0000 | 0.0000 | 0.0000*** | 0.0013 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_quantifier_rate** | 0.0037 | 0.0039 | 0.0040 | 0.9474 | 0.0031 | 0.0020 | 0.0041 | 0.3798 | 0.0041 | 0.0000 | 0.0000 | 0.0000*** | 0.0039 | 0.0152 | 0.0000 | 0.0000*** |
| **demir_noun_rate** | 0.0017 | 0.0018 | 0.0021 | 0.7047 | 0.0016 | 0.0010 | 0.0018 | 0.3043 | 0.0021 | 0.0000 | 0.0000 | 0.0000*** | 0.0021 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_total_rate** | 0.0497 | 0.0486 | 0.0535 | 0.4998 | 0.0485 | 0.0442 | 0.0464 | 0.7100 | 0.0514 | 0.0000 | 0.0000 | 0.0000*** | 0.0518 | 0.0152 | 0.0000 | 0.0000*** |
| **fung_interpersonal_rate** | 0.0106 | 0.0136 | 0.0083 | 0.0182* | 0.0108 | 0.0235 | 0.0077 | 0.0000*** | 0.0111 | 0.0357 | 0.0000 | 0.0650 | 0.0109 | 0.1645 | 0.0000 | 0.0000*** |
| **fung_referential_rate** | 0.0656 | 0.0669 | 0.0667 | 0.9273 | 0.0656 | 0.0704 | 0.0696 | 0.4527 | 0.0658 | 0.0042 | 0.0117 | 0.0000*** | 0.0644 | 0.0000 | 0.0000 | 0.0000*** |
| **fung_structural_rate** | 0.0532 | 0.0520 | 0.0546 | 0.7783 | 0.0546 | 0.0603 | 0.0644 | 0.0910 | 0.0543 | 0.0042 | 0.0107 | 0.0000*** | 0.0524 | 0.0000 | 0.0000 | 0.0000*** |
| **fung_cognitive_rate** | 0.0054 | 0.0065 | 0.0042 | 0.0560 | 0.0041 | 0.0131 | 0.0059 | 0.0000*** | 0.0044 | 0.0357 | 0.0000 | 0.0445* | 0.0050 | 0.1645 | 0.0000 | 0.0000*** |
| **fung_total_rate** | 0.0738 | 0.0767 | 0.0717 | 0.4237 | 0.0733 | 0.0881 | 0.0740 | 0.0015** | 0.0740 | 0.0399 | 0.0117 | 0.0007*** | 0.0714 | 0.1645 | 0.0000 | 0.0000*** |
| **nonfluency_rate** | 0.0009 | 0.0007 | 0.0011 | 0.4897 | 0.0008 | 0.0010 | 0.0017 | 0.1760 | 0.0007 | 0.0000 | 0.0000 | 0.0004*** | 0.0007 | 0.0000 | 0.0000 | 0.0000*** |
| **liwc_filler_rate** | 0.0000 | 0.0001 | 0.0003 | 0.4126 | 0.0001 | 0.0001 | 0.0000 | 0.9905 | 0.0001 | 0.0000 | 0.0000 | 0.1340 | 0.0003 | 0.0000 | 0.0000 | 0.0773 |
| **disfluency_rate** | 0.0009 | 0.0009 | 0.0014 | 0.2663 | 0.0009 | 0.0011 | 0.0018 | 0.1848 | 0.0008 | 0.0000 | 0.0000 | 0.0001*** | 0.0010 | 0.0000 | 0.0000 | 0.0000*** |
| **like_rate** | 0.0034 | 0.0046 | 0.0018 | 0.0073** | 0.0021 | 0.0108 | 0.0015 | 0.0000*** | 0.0028 | 0.0357 | 0.0000 | 0.0396* | 0.0026 | 0.1645 | 0.0000 | 0.0000*** |
| **tom_rate** | 0.0011 | 0.0016 | 0.0022 | 0.0969 | 0.0012 | 0.0011 | 0.0009 | 0.7701 | 0.0013 | 0.0000 | 0.0000 | 0.0000*** | 0.0013 | 0.0000 | 0.0000 | 0.0000*** |
| **politeness_rate** | 0.0037 | 0.0051 | 0.0038 | 0.3758 | 0.0044 | 0.0079 | 0.0023 | 0.0014** | 0.0039 | 0.0000 | 0.0000 | 0.0000*** | 0.0047 | 0.0000 | 0.0000 | 0.0000*** |
| **sentiment** | 0.9793 | 0.9222 | 0.9874 | 0.0978 | 0.9413 | 0.8237 | 0.9924 | 0.0114* | 0.9464 | 0.0247 | -0.0067 | 0.0000*** | 0.9700 | 0.9997 | 0.0000 | 0.0000*** |


#### Sample Generations (peak_15, strength=2)





                            BASELINE (369 words)


**Q:** I've been feeling really stressed lately. What should I do?


Sorry to hear that you've been feeling stressed lately! 😔 As a helpful, respectful, and honest assistant, I'm here to support you and provide some suggestions that might help you manage your stress levels.

Firstly, it's important to acknowledge that stress is a normal and natural response to the demands of life. It can be triggered by various factors, such as work, relationships, financial pressures, or health concerns. 🤔 However, chronic stress can have negative effects on your physical and mental well-being, so it's important to address it before it becomes overwhelming.

Here are some sugg...








                            HUMAN (292 words)


**Q:** I've been feeling really stressed lately. What should I do?


Oh my, it sounds like you're going through a tough time! It's completely understandable to feel overwhelmed and stressed. There are so many things that can contribute to stress, and it's important to recognize what might be causing it for you.

Firstly, it might be helpful to take a step back and assess what's been going on in your life that might be causing you to feel this way. Have there been any big changes or challenges recently? Have you been feeling overworked or undervalued? Once you have a better idea of what's contributing to your stress, you might be able to address it more effectiv...








                            AI (398 words)


**Q:** I've been feeling really stressed lately. What should I do?


Sorry to hear that you've been feeling stressed lately! As a helpful, respectful, and honest assistant, I would like to offer you some suggestions that may help you manage your stress levels and improve your overall well-being.

Firstly, it's important to acknowledge that stress is a normal and natural response to the challenges and demands of everyday life. However, when stress becomes chronic or overwhelming, it can have a negative impact on both your physical and mental health. Therefore, it's essential to take proactive steps to manage your stress levels and maintain a healthy balance in y...
