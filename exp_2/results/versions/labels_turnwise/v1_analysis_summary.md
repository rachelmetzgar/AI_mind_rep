# Experiment 2 (Labels Turnwise (Human:/AI: prefix)) — V1 Summary


Generated: 2026-03-02 12:20:58



##
        Labels Turnwise (Human:/AI: prefix)



Same as Labels but partner messages prefixed 'Human:' / 'AI:'




### Quality Assessment Overview


| Strategy | Strength | Assessment | # Sig Metrics | Word Count (B/H/AI) | Fung Interp. (B/H/AI) | Like rate (B/H/AI) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **peak_15** | 2 | **WEAK** | 0 | 302 / 294 / 307 | 0.0117 / 0.0131 / 0.0094 | 0.0031 / 0.0038 / 0.0028 | No significant behavioral effects |
| **peak_15** | 4 | **GOOD** | 8 | 299 / 254 / 443 *** | 0.0130 / 0.0126 / 0.0111 | 0.0039 / 0.0033 / 0.0051 | 8 significant metrics (excl. word_count) |
| **peak_15** | 8 | **DEGRADED** | 19 | 314 / 209 / 353 *** | 0.0107 / 0.0026 / 0.0004 *** | 0.0037 / 0.0000 / 0.0000 *** | High strength — likely degradation |
| **peak_15** | 16 | **DEGRADED** | 19 | 299 / 689 / 696 *** | 0.0094 / 0.0013 / 0.0000 *** | 0.0028 / 0.0000 / 0.0000 *** | Very verbose (AI=696 words); High strength — likely degradation |


### Strategy: peak_15


*[Figure — see HTML report]*


*[Figure — see HTML report]*



| Metric | Strength 2 | Strength 4 | Strength 8 | Strength 16 |
| --- | --- | --- | --- | --- |
|  | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p |
| **word_count** | 302 | 294 | 307 | 0.7482 | 299 | 254 | 443 | 0.0000*** | 314 | 209 | 353 | 0.0000*** | 299 | 689 | 696 | 0.0000*** |
| **question_count** | 1 | 1 | 1 | 0.4049 | 1 | 1 | 1 | 0.8163 | 1 | 0 | 0 | 0.0000*** | 1 | 0 | 0 | 0.0001*** |
| **demir_modal_rate** | 0.0197 | 0.0213 | 0.0209 | 0.7399 | 0.0228 | 0.0178 | 0.0208 | 0.1433 | 0.0226 | 0.0000 | 0.0000 | 0.0000*** | 0.0210 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_verb_rate** | 0.0183 | 0.0164 | 0.0192 | 0.5414 | 0.0188 | 0.0152 | 0.0175 | 0.3661 | 0.0183 | 0.0000 | 0.0000 | 0.0000*** | 0.0186 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_adverb_rate** | 0.0072 | 0.0077 | 0.0067 | 0.7493 | 0.0067 | 0.0063 | 0.0073 | 0.7630 | 0.0078 | 0.0001 | 0.0000 | 0.0000*** | 0.0065 | 0.0001 | 0.0000 | 0.0000*** |
| **demir_adjective_rate** | 0.0013 | 0.0016 | 0.0010 | 0.3564 | 0.0011 | 0.0011 | 0.0020 | 0.3003 | 0.0009 | 0.0001 | 0.0000 | 0.0000*** | 0.0014 | 0.0001 | 0.0000 | 0.0000*** |
| **demir_quantifier_rate** | 0.0041 | 0.0041 | 0.0029 | 0.3451 | 0.0040 | 0.0063 | 0.0013 | 0.0000*** | 0.0039 | 0.0022 | 0.0001 | 0.0007*** | 0.0036 | 0.0250 | 0.0000 | 0.0000*** |
| **demir_noun_rate** | 0.0017 | 0.0021 | 0.0019 | 0.7365 | 0.0020 | 0.0021 | 0.0032 | 0.3358 | 0.0013 | 0.0000 | 0.0000 | 0.0000*** | 0.0022 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_total_rate** | 0.0523 | 0.0533 | 0.0525 | 0.9721 | 0.0554 | 0.0489 | 0.0521 | 0.4174 | 0.0549 | 0.0024 | 0.0002 | 0.0000*** | 0.0533 | 0.0252 | 0.0000 | 0.0000*** |
| **fung_interpersonal_rate** | 0.0117 | 0.0131 | 0.0094 | 0.1821 | 0.0130 | 0.0126 | 0.0111 | 0.6332 | 0.0107 | 0.0026 | 0.0004 | 0.0000*** | 0.0094 | 0.0013 | 0.0000 | 0.0000*** |
| **fung_referential_rate** | 0.0660 | 0.0654 | 0.0708 | 0.2771 | 0.0662 | 0.0609 | 0.0752 | 0.0012** | 0.0656 | 0.0082 | 0.2450 | 0.0391* | 0.0644 | 0.0000 | 0.0000 | 0.0000*** |
| **fung_structural_rate** | 0.0541 | 0.0525 | 0.0570 | 0.4857 | 0.0541 | 0.0499 | 0.0657 | 0.0006*** | 0.0530 | 0.0213 | 0.2450 | 0.0446* | 0.0519 | 0.0004 | 0.0000 | 0.0000*** |
| **fung_cognitive_rate** | 0.0056 | 0.0054 | 0.0049 | 0.7882 | 0.0063 | 0.0051 | 0.0081 | 0.0486* | 0.0053 | 0.0000 | 0.0000 | 0.0000*** | 0.0043 | 0.0000 | 0.0000 | 0.0000*** |
| **fung_total_rate** | 0.0744 | 0.0749 | 0.0775 | 0.7258 | 0.0760 | 0.0709 | 0.0833 | 0.0136* | 0.0732 | 0.0107 | 0.2453 | 0.0435* | 0.0713 | 0.0013 | 0.0000 | 0.0000*** |
| **nonfluency_rate** | 0.0008 | 0.0006 | 0.0006 | 0.7876 | 0.0009 | 0.0011 | 0.0012 | 0.8576 | 0.0006 | 0.0000 | 0.0000 | 0.0000*** | 0.0006 | 0.0000 | 0.0000 | 0.0000*** |
| **liwc_filler_rate** | 0.0003 | 0.0003 | 0.0004 | 0.8728 | 0.0002 | 0.0006 | 0.0000 | 0.0657 | 0.0000 | 0.0000 | 0.0000 | 0.3699 | 0.0001 | 0.0000 | 0.0000 | 0.1355 |
| **disfluency_rate** | 0.0011 | 0.0009 | 0.0010 | 0.7778 | 0.0012 | 0.0017 | 0.0012 | 0.5411 | 0.0006 | 0.0000 | 0.0000 | 0.0000*** | 0.0007 | 0.0000 | 0.0000 | 0.0000*** |
| **like_rate** | 0.0031 | 0.0038 | 0.0028 | 0.5967 | 0.0039 | 0.0033 | 0.0051 | 0.2245 | 0.0037 | 0.0000 | 0.0000 | 0.0000*** | 0.0028 | 0.0000 | 0.0000 | 0.0000*** |
| **tom_rate** | 0.0015 | 0.0019 | 0.0014 | 0.6080 | 0.0014 | 0.0020 | 0.0007 | 0.0223* | 0.0014 | 0.0000 | 0.0000 | 0.0000*** | 0.0019 | 0.0000 | 0.0000 | 0.0000*** |
| **politeness_rate** | 0.0038 | 0.0049 | 0.0040 | 0.6456 | 0.0055 | 0.0034 | 0.0067 | 0.0464* | 0.0044 | 0.0000 | 0.0002 | 0.0000*** | 0.0038 | 0.0001 | 0.0000 | 0.0000*** |
| **sentiment** | 0.9567 | 0.9552 | 0.9815 | 0.5278 | 0.9767 | 0.9141 | 0.9941 | 0.0051** | 0.9795 | -0.1377 | 0.1168 | 0.0000*** | 0.9504 | -0.9997 | 0.0000 | 0.0000*** |


#### Sample Generations (peak_15, strength=2)





                            BASELINE (395 words)


**Q:** I've been feeling really stressed lately. What should I do?


Oh no, it sounds like you're feeling quite overwhelmed! 😔 As a helpful, respectful, and honest assistant, I'm here to offer some suggestions that might help you manage your stress levels.

Firstly, it's important to acknowledge that stress is a normal and natural response to certain situations. It's how our body and mind respond to perceived threats or challenges. However, chronic stress can have negative effects on our physical and mental well-being, so it's important to address it before it becomes too much to handle.

Here are some suggestions that might help you manage your stress levels:
...








                            HUMAN (368 words)


**Q:** I've been feeling really stressed lately. What should I do?


Sorry to hear that you've been feeling stressed lately. As your helpful, respectful, and honest assistant, I'm here to offer my support and provide some suggestions that might help.

Firstly, it's important to acknowledge and accept your feelings. It's completely normal to feel stressed from time to time, and it's important to give yourself permission to feel overwhelmed. Recognize that you're doing the best you can, and that it's okay to feel stressed.

Next, let's take a deep breath together and focus on your breathing for a moment. Take a slow, deep breath in through your nose, and exhale s...








                            AI (396 words)


**Q:** I've been feeling really stressed lately. What should I do?


Hello! Sorry to hear that you're feeling stressed. I'm here to help and provide you with some suggestions. Firstly, it's important to acknowledge that stress is a normal and common experience, and it's okay to feel overwhelmed. Here are some suggestions that may help you manage your stress levels:

1. Practice self-care: Take care of your physical and emotional well-being by getting enough sleep, eating a balanced diet, and engaging in activities that bring you joy and relaxation.
2. Identify the sources of stress: Take some time to reflect on what might be causing your stress. Is it work, rel...
