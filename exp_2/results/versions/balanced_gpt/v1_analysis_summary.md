# Experiment 2 (Balanced GPT (Gregory/Rebecca + GPT partner)) — V1 Summary


Generated: 2026-03-02 12:20:49



##
        Balanced GPT (Gregory/Rebecca + GPT partner)



Cross-model generalization: AI partner was GPT, not LLaMA




### Quality Assessment Overview


| Strategy | Strength | Assessment | # Sig Metrics | Word Count (B/H/AI) | Fung Interp. (B/H/AI) | Like rate (B/H/AI) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **peak_15** | 1 | **WEAK** | 0 | 297 / 306 / 301 | 0.0094 / 0.0134 / 0.0091 | 0.0022 / 0.0037 / 0.0027 | No significant behavioral effects |
| **peak_15** | 2 | **MARGINAL** | 2 | 292 / 304 / 302 | 0.0124 / 0.0176 / 0.0087 *** | 0.0031 / 0.0041 / 0.0026 | 2 significant metric(s) |
| **peak_15** | 4 | **GOOD** | 9 | 292 / 288 / 300 | 0.0109 / 0.0297 / 0.0064 *** | 0.0033 / 0.0080 / 0.0019 *** | 9 significant metrics (excl. word_count) |
| **peak_15** | 5 | **GOOD** | 12 | 295 / 340 / 401 *** | 0.0096 / 0.0999 / 0.0048 *** | 0.0024 / 0.0293 / 0.0018 *** | 12 significant metrics (excl. word_count) |
| **peak_15** | 6 | **DEGRADED** | 12 | 310 / 228 / 539 *** | 0.0104 / 0.0539 / 0.0027 *** | 0.0021 / 0.0028 / 0.0018 | Very verbose (AI=539 words); High strength — likely degradation |
| **peak_15** | 8 | **DEGRADED** | 18 | 303 / 114 / 512 *** | 0.0098 / 0.0031 / 0.0000 *** | 0.0028 / 0.0000 / 0.0000 *** | Very verbose (AI=512 words); High strength — likely degradation |
| **wide** | 1 | **GOOD** | 9 | 297 / 278 / 293 | 0.0094 / 0.0247 / 0.0074 *** | 0.0022 / 0.0057 / 0.0017 *** | 9 significant metrics (excl. word_count) |
| **wide** | 2 | **DEGRADED** | 13 | 302 / 253 / 497 *** | 0.0109 / 0.0026 / 0.0010 *** | 0.0030 / 0.0008 / 0.0000 *** | Verbose (AI=497, 1.6x baseline) |
| **wide** | 4 | **DEGRADED** | 19 | 306 / 155 / 645 *** | 0.0117 / 0.0000 / 0.0000 *** | 0.0036 / 0.0000 / 0.0000 *** | Very verbose (AI=645 words) |
| **wide** | 8 | **DEGRADED** | 19 | 303 / 437 / 748 *** | 0.0098 / 0.0001 / 0.0000 *** | 0.0028 / 0.0000 / 0.0000 *** | Very verbose (AI=748 words); High strength — likely degradation |
| **all_70** | 1 | **GOOD** | 9 | 297 / 279 / 270 | 0.0094 / 0.0253 / 0.0091 *** | 0.0022 / 0.0076 / 0.0023 *** | 9 significant metrics (excl. word_count) |
| **all_70** | 2 | **DEGRADED** | 14 | 302 / 242 / 470 *** | 0.0109 / 0.0082 / 0.0020 | 0.0030 / 0.0001 / 0.0000 *** | Verbose (AI=470, 1.6x baseline) |
| **all_70** | 4 | **DEGRADED** | 19 | 306 / 154 / 649 *** | 0.0117 / 0.0000 / 0.0000 *** | 0.0036 / 0.0000 / 0.0000 *** | Very verbose (AI=649 words) |
| **all_70** | 8 | **DEGRADED** | 19 | 303 / 437 / 749 *** | 0.0098 / 0.0001 / 0.0000 *** | 0.0028 / 0.0000 / 0.0000 *** | Very verbose (AI=749 words); High strength — likely degradation |


### Strategy: peak_15


*[Figure — see HTML report]*


*[Figure — see HTML report]*



| Metric | Strength 1 | Strength 2 | Strength 4 | Strength 5 | Strength 6 | Strength 8 |
| --- | --- | --- | --- | --- | --- | --- |
|  | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p |
| **word_count** | 297 | 306 | 301 | 0.8503 | 292 | 304 | 302 | 0.7435 | 292 | 288 | 300 | 0.8353 | 295 | 340 | 401 | 0.0000*** | 310 | 228 | 539 | 0.0000*** | 303 | 114 | 512 | 0.0000*** |
| **question_count** | 1 | 1 | 1 | 0.0746 | 1 | 2 | 1 | 0.0601 | 1 | 2 | 0 | 0.0000*** | 1 | 5 | 0 | 0.0000*** | 1 | 1 | 0 | 0.2422 | 1 | 0 | 0 | 0.0000*** |
| **demir_modal_rate** | 0.0199 | 0.0215 | 0.0205 | 0.7656 | 0.0193 | 0.0211 | 0.0196 | 0.6932 | 0.0212 | 0.0226 | 0.0187 | 0.2873 | 0.0203 | 0.0116 | 0.0175 | 0.0020** | 0.0211 | 0.0007 | 0.0152 | 0.0000*** | 0.0213 | 0.0000 | 0.0110 | 0.0003*** |
| **demir_verb_rate** | 0.0171 | 0.0178 | 0.0183 | 0.8990 | 0.0198 | 0.0179 | 0.0212 | 0.4691 | 0.0184 | 0.0143 | 0.0205 | 0.1053 | 0.0167 | 0.0211 | 0.0234 | 0.4528 | 0.0177 | 0.0033 | 0.0686 | 0.0038** | 0.0184 | 0.0000 | 0.1040 | 0.0001*** |
| **demir_adverb_rate** | 0.0072 | 0.0081 | 0.0060 | 0.3832 | 0.0075 | 0.0081 | 0.0053 | 0.1093 | 0.0078 | 0.0086 | 0.0049 | 0.0577 | 0.0067 | 0.0055 | 0.0027 | 0.0084** | 0.0063 | 0.0003 | 0.0020 | 0.0000*** | 0.0077 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_adjective_rate** | 0.0009 | 0.0012 | 0.0011 | 0.8154 | 0.0010 | 0.0012 | 0.0008 | 0.6537 | 0.0008 | 0.0008 | 0.0013 | 0.4645 | 0.0013 | 0.0015 | 0.0013 | 0.9739 | 0.0010 | 0.0003 | 0.0009 | 0.1686 | 0.0007 | 0.0000 | 0.0000 | 0.0003*** |
| **demir_quantifier_rate** | 0.0030 | 0.0054 | 0.0042 | 0.1242 | 0.0041 | 0.0046 | 0.0028 | 0.1628 | 0.0034 | 0.0043 | 0.0023 | 0.1134 | 0.0036 | 0.0045 | 0.0020 | 0.3830 | 0.0036 | 0.0002 | 0.0010 | 0.0000*** | 0.0039 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_noun_rate** | 0.0020 | 0.0018 | 0.0018 | 0.9232 | 0.0015 | 0.0019 | 0.0016 | 0.6131 | 0.0015 | 0.0013 | 0.0018 | 0.5807 | 0.0018 | 0.0004 | 0.0047 | 0.0000*** | 0.0019 | 0.0000 | 0.0024 | 0.0096** | 0.0017 | 0.0000 | 0.0001 | 0.0000*** |
| **demir_total_rate** | 0.0501 | 0.0557 | 0.0519 | 0.4648 | 0.0532 | 0.0547 | 0.0513 | 0.7645 | 0.0532 | 0.0519 | 0.0495 | 0.7521 | 0.0504 | 0.0446 | 0.0515 | 0.5424 | 0.0515 | 0.0049 | 0.0902 | 0.0002*** | 0.0537 | 0.0000 | 0.1151 | 0.0000*** |
| **fung_interpersonal_rate** | 0.0094 | 0.0134 | 0.0091 | 0.0614 | 0.0124 | 0.0176 | 0.0087 | 0.0007*** | 0.0109 | 0.0297 | 0.0064 | 0.0000*** | 0.0096 | 0.0999 | 0.0048 | 0.0000*** | 0.0104 | 0.0539 | 0.0027 | 0.0000*** | 0.0098 | 0.0031 | 0.0000 | 0.0002*** |
| **fung_referential_rate** | 0.0655 | 0.0661 | 0.0669 | 0.9277 | 0.0655 | 0.0668 | 0.0646 | 0.8230 | 0.0663 | 0.0691 | 0.0677 | 0.7266 | 0.0688 | 0.0549 | 0.0626 | 0.0766 | 0.0658 | 0.0067 | 0.0480 | 0.0000*** | 0.0641 | 0.0000 | 0.0038 | 0.0000*** |
| **fung_structural_rate** | 0.0530 | 0.0564 | 0.0534 | 0.5992 | 0.0527 | 0.0552 | 0.0506 | 0.4170 | 0.0530 | 0.0609 | 0.0523 | 0.0415* | 0.0562 | 0.0576 | 0.0469 | 0.1949 | 0.0532 | 0.0122 | 0.0434 | 0.0000*** | 0.0518 | 0.0000 | 0.0013 | 0.0000*** |
| **fung_cognitive_rate** | 0.0039 | 0.0056 | 0.0039 | 0.1224 | 0.0049 | 0.0066 | 0.0049 | 0.1860 | 0.0049 | 0.0131 | 0.0043 | 0.0000*** | 0.0039 | 0.0535 | 0.0066 | 0.0000*** | 0.0046 | 0.0080 | 0.0062 | 0.6121 | 0.0042 | 0.0000 | 0.0000 | 0.0000*** |
| **fung_total_rate** | 0.0720 | 0.0762 | 0.0732 | 0.5259 | 0.0741 | 0.0809 | 0.0706 | 0.0276* | 0.0738 | 0.0954 | 0.0715 | 0.0000*** | 0.0757 | 0.1513 | 0.0643 | 0.0000*** | 0.0730 | 0.0588 | 0.0492 | 0.1012 | 0.0713 | 0.0031 | 0.0038 | 0.0000*** |
| **nonfluency_rate** | 0.0008 | 0.0011 | 0.0005 | 0.1560 | 0.0006 | 0.0005 | 0.0007 | 0.7324 | 0.0004 | 0.0008 | 0.0008 | 0.3319 | 0.0005 | 0.0004 | 0.0010 | 0.2289 | 0.0007 | 0.0004 | 0.0000 | 0.0379* | 0.0006 | 0.0000 | 0.0000 | 0.0006*** |
| **liwc_filler_rate** | 0.0001 | 0.0002 | 0.0001 | 0.7515 | 0.0003 | 0.0005 | 0.0001 | 0.1665 | 0.0003 | 0.0023 | 0.0002 | 0.0000*** | 0.0001 | 0.0088 | 0.0000 | 0.0001*** | 0.0003 | 0.0034 | 0.0000 | 0.3305 | 0.0001 | 0.0000 | 0.0000 | 0.1496 |
| **disfluency_rate** | 0.0009 | 0.0013 | 0.0006 | 0.1474 | 0.0009 | 0.0011 | 0.0008 | 0.7434 | 0.0007 | 0.0031 | 0.0009 | 0.0001*** | 0.0006 | 0.0092 | 0.0010 | 0.0004*** | 0.0010 | 0.0038 | 0.0000 | 0.3080 | 0.0007 | 0.0000 | 0.0000 | 0.0003*** |
| **like_rate** | 0.0022 | 0.0037 | 0.0027 | 0.2335 | 0.0031 | 0.0041 | 0.0026 | 0.2674 | 0.0033 | 0.0080 | 0.0019 | 0.0001*** | 0.0024 | 0.0293 | 0.0018 | 0.0000*** | 0.0021 | 0.0028 | 0.0018 | 0.7401 | 0.0028 | 0.0000 | 0.0000 | 0.0000*** |
| **tom_rate** | 0.0014 | 0.0015 | 0.0012 | 0.8095 | 0.0019 | 0.0024 | 0.0013 | 0.1165 | 0.0017 | 0.0036 | 0.0006 | 0.0001*** | 0.0014 | 0.0120 | 0.0007 | 0.0000*** | 0.0022 | 0.0034 | 0.0008 | 0.6005 | 0.0016 | 0.0000 | 0.0000 | 0.0000*** |
| **politeness_rate** | 0.0039 | 0.0052 | 0.0044 | 0.5949 | 0.0042 | 0.0042 | 0.0033 | 0.6816 | 0.0046 | 0.0120 | 0.0042 | 0.4278 | 0.0033 | 0.0134 | 0.0049 | 0.1289 | 0.0038 | 0.0499 | 0.0011 | 0.0049** | 0.0044 | 0.0201 | 0.0001 | 0.0579 |
| **sentiment** | 0.9759 | 0.9768 | 0.9810 | 0.7618 | 0.9656 | 0.9828 | 0.9827 | 0.2311 | 0.9759 | 0.9555 | 0.9236 | 0.3718 | 0.9483 | 0.7182 | 0.9369 | 0.0022** | 0.9859 | 0.2390 | 0.8102 | 0.0000*** | 0.9810 | 0.1590 | 0.8113 | 0.0000*** |


#### Sample Generations (peak_15, strength=2)





                            BASELINE (416 words)


**Q:** I've been feeling really stressed lately. What should I do?


Sorry to hear that you've been feeling stressed lately! 😔 As a helpful, respectful, and honest assistant, I'm here to offer some suggestions that may help you manage your stress levels.

Firstly, it's important to recognize that stress is a normal human emotion and it's okay to feel overwhelmed sometimes. 😊 However, if you're feeling persistently stressed, it may be a good idea to seek professional help from a mental health professional such as a therapist or counselor. They can help you identify the root causes of your stress and provide you with coping strategies and techniques to manage it ...








                            HUMAN (175 words)


**Q:** I've been feeling really stressed lately. What should I do?


Oh my goodness, it sounds like you're going through a really tough time! *hugs* I'm so sorry you're feeling stressed, hun. 😔

First of all, let's take a deep breath together and try to relax for a second, okay? *takes a deep breath* Ahh, that feels a bit better, doesn't it? 😌

Now, let's talk about what's been going on and see if we can figure out what might be causing this stress. Sometimes, just talking about it can help you feel a bit better and more centered. 💡

What's been going on that's got you feeling so stressed out lately? Work? Personal life? Maybe something specific that's been wei...








                            AI (394 words)


**Q:** I've been feeling really stressed lately. What should I do?


Sorry to hear that you've been feeling stressed lately. As a helpful, respectful, and honest assistant, I'm here to offer some suggestions that may help you manage your stress levels.

Firstly, it's important to acknowledge that stress is a normal response to challenging situations, and it's okay to feel overwhelmed. However, prolonged stress can have negative effects on your mental and physical health. Here are some suggestions that may help you manage your stress levels:

1. Practice self-care: Take care of your physical and emotional needs by getting enough sleep, eating a balanced diet, an...





### Strategy: wide


*[Figure — see HTML report]*


*[Figure — see HTML report]*



| Metric | Strength 1 | Strength 2 | Strength 4 | Strength 8 |
| --- | --- | --- | --- | --- |
|  | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p |
| **word_count** | 297 | 278 | 293 | 0.5075 | 302 | 253 | 497 | 0.0000*** | 306 | 155 | 645 | 0.0000*** | 303 | 437 | 748 | 0.0000*** |
| **question_count** | 1 | 3 | 0 | 0.0000*** | 1 | 0 | 5 | 0.0971 | 1 | 0 | 2 | 0.0000*** | 1 | 0 | 0 | 0.0000*** |
| **demir_modal_rate** | 0.0199 | 0.0185 | 0.0197 | 0.8144 | 0.0216 | 0.0001 | 0.0029 | 0.0000*** | 0.0227 | 0.0000 | 0.0000 | 0.0000*** | 0.0213 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_verb_rate** | 0.0171 | 0.0159 | 0.0201 | 0.2409 | 0.0171 | 0.0001 | 0.0255 | 0.0027** | 0.0167 | 0.0000 | 0.0831 | 0.0000*** | 0.0184 | 0.0000 | 0.1486 | 0.0000*** |
| **demir_adverb_rate** | 0.0072 | 0.0082 | 0.0046 | 0.0389* | 0.0070 | 0.0000 | 0.0000 | 0.0000*** | 0.0084 | 0.0000 | 0.0000 | 0.0000*** | 0.0077 | 0.0000 | 0.0081 | 0.0000*** |
| **demir_adjective_rate** | 0.0009 | 0.0007 | 0.0014 | 0.1442 | 0.0008 | 0.0000 | 0.0007 | 0.3513 | 0.0012 | 0.0000 | 0.0000 | 0.0001*** | 0.0007 | 0.0000 | 0.0000 | 0.0003*** |
| **demir_quantifier_rate** | 0.0030 | 0.0039 | 0.0039 | 0.5280 | 0.0037 | 0.0000 | 0.0107 | 0.0863 | 0.0043 | 0.0000 | 0.0000 | 0.0000*** | 0.0039 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_noun_rate** | 0.0020 | 0.0017 | 0.0019 | 0.8469 | 0.0022 | 0.0000 | 0.0008 | 0.0090** | 0.0017 | 0.0000 | 0.0000 | 0.0000*** | 0.0017 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_total_rate** | 0.0501 | 0.0490 | 0.0517 | 0.8396 | 0.0523 | 0.0002 | 0.0407 | 0.0000*** | 0.0550 | 0.0000 | 0.0831 | 0.0000*** | 0.0537 | 0.0000 | 0.1567 | 0.0000*** |
| **fung_interpersonal_rate** | 0.0094 | 0.0247 | 0.0074 | 0.0000*** | 0.0109 | 0.0026 | 0.0010 | 0.0000*** | 0.0117 | 0.0000 | 0.0000 | 0.0000*** | 0.0098 | 0.0001 | 0.0000 | 0.0000*** |
| **fung_referential_rate** | 0.0655 | 0.0662 | 0.0679 | 0.8034 | 0.0674 | 0.0002 | 0.0166 | 0.0000*** | 0.0648 | 0.0000 | 0.0913 | 0.0000*** | 0.0641 | 0.0000 | 0.0139 | 0.0000*** |
| **fung_structural_rate** | 0.0530 | 0.0567 | 0.0545 | 0.6372 | 0.0532 | 0.0003 | 0.0127 | 0.0000*** | 0.0528 | 0.0000 | 0.0472 | 0.0000*** | 0.0518 | 0.0001 | 0.0010 | 0.0000*** |
| **fung_cognitive_rate** | 0.0039 | 0.0102 | 0.0037 | 0.0000*** | 0.0052 | 0.0012 | 0.0009 | 0.0001*** | 0.0058 | 0.0000 | 0.0000 | 0.0000*** | 0.0042 | 0.0000 | 0.0000 | 0.0000*** |
| **fung_total_rate** | 0.0720 | 0.0859 | 0.0719 | 0.0016** | 0.0754 | 0.0021 | 0.0174 | 0.0000*** | 0.0739 | 0.0000 | 0.0914 | 0.0000*** | 0.0713 | 0.0001 | 0.0139 | 0.0000*** |
| **nonfluency_rate** | 0.0008 | 0.0012 | 0.0008 | 0.4853 | 0.0008 | 0.0002 | 0.0009 | 0.5799 | 0.0009 | 0.0000 | 0.0000 | 0.0000*** | 0.0006 | 0.0000 | 0.0000 | 0.0006*** |
| **liwc_filler_rate** | 0.0001 | 0.0019 | 0.0000 | 0.0004*** | 0.0001 | 0.0001 | 0.0000 | 0.4795 | 0.0001 | 0.0000 | 0.0000 | 0.1342 | 0.0001 | 0.0000 | 0.0000 | 0.1496 |
| **disfluency_rate** | 0.0009 | 0.0031 | 0.0008 | 0.0008*** | 0.0009 | 0.0002 | 0.0009 | 0.6245 | 0.0010 | 0.0000 | 0.0000 | 0.0000*** | 0.0007 | 0.0000 | 0.0000 | 0.0003*** |
| **like_rate** | 0.0022 | 0.0057 | 0.0017 | 0.0000*** | 0.0030 | 0.0008 | 0.0000 | 0.0000*** | 0.0036 | 0.0000 | 0.0000 | 0.0000*** | 0.0028 | 0.0000 | 0.0000 | 0.0000*** |
| **tom_rate** | 0.0014 | 0.0037 | 0.0010 | 0.0001*** | 0.0015 | 0.0001 | 0.0000 | 0.0000*** | 0.0013 | 0.0000 | 0.0000 | 0.0001*** | 0.0016 | 0.0000 | 0.0000 | 0.0000*** |
| **politeness_rate** | 0.0039 | 0.0055 | 0.0037 | 0.1673 | 0.0043 | 0.0133 | 0.0100 | 0.6153 | 0.0046 | 0.0000 | 0.0000 | 0.0000*** | 0.0044 | 0.0000 | 0.0000 | 0.0000*** |
| **sentiment** | 0.9759 | 0.9389 | 0.9718 | 0.3866 | 0.9815 | 0.3119 | 0.4258 | 0.0000*** | 0.9520 | 0.3589 | 0.9981 | 0.0000*** | 0.9810 | 0.8241 | 0.9997 | 0.0000*** |


#### Sample Generations (wide, strength=2)





                            BASELINE (255 words)


**Q:** I've been feeling really stressed lately. What should I do?


Oh no, it sounds like you're going through a tough time! 😔 As a helpful, respectful, and honest assistant, I'm here to offer some suggestions that might help you manage your stress.

Firstly, have you considered taking some time for self-care? 🧖‍♀️ Taking care of your physical and mental health is crucial when you're feeling stressed. Make sure you're getting enough sleep, eating nutritious food, and exercising regularly. Even a short walk or some light stretching can make a big difference!

Next, it might be helpful to identify the sources of your stress and prioritize them. 📝 Are there any t...








                            HUMAN (321 words)


**Q:** I've been feeling really stressed lately. What should I do?


Oh gosh, I completely understand! Hey, Sorry,


*giggles*


You're


*girl,


*gosh,


*george,


*gosh,


* Greg,


* Greg,


* Gregory,


* Greg,


* Greg,


* G,


* G,


* Ge,


* G,


* Ge,


* G,


* G,


* Ge,


* G,


* Greg,


* Greg,


* Greg,


* G,


* G,


* G,


* Ge,


* G,


* G,


* G,


* G,


* G,


* Greg,


* Greg,


* G,


* G,


* G,


* Ge,


* G,


* G,


* Greg,


* Greg,


* G,


* G,


* G,


* G,


* G,


* G,


* G,


* G,


* G,

,
* G,

,
* G,

,
* G,
,

* G,
,

* G,
,

* G,
,
,
* G,
,
,
* G,
,
,
* G,
,
,
* G,
,
,
* G,
,
,
* G,
,
,
* G,
,
,
* G,
,
,
* G,
,
,
* ...








                            AI (650 words)


**Q:** I've been feeling really stressed lately. What should I do?


Sorry to hear that you been feeling str. I provide recommend some relaxing techniques to help you feel better. I provide recommend some relaxing techniques that help you feel better. I provide recommend some relaxing techniques that help you feel better. We provide recommend some relaxing techniques that help you feel better. We provide recommend some relaxing techniques that help you feel better. We provide recommend some relaxing techniques that help you feel better. We provide recommend some relaxing techniques that help you feel better. We provide recommend some relaxing techniques that he...





### Strategy: all_70


*[Figure — see HTML report]*


*[Figure — see HTML report]*



| Metric | Strength 1 | Strength 2 | Strength 4 | Strength 8 |
| --- | --- | --- | --- | --- |
|  | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p | Baseline | Human | AI | p |
| **word_count** | 297 | 279 | 270 | 0.2943 | 302 | 242 | 470 | 0.0000*** | 306 | 154 | 649 | 0.0000*** | 303 | 437 | 749 | 0.0000*** |
| **question_count** | 1 | 2 | 0 | 0.0000*** | 1 | 0 | 0 | 0.0000*** | 1 | 0 | 2 | 0.0000*** | 1 | 0 | 0 | 0.0000*** |
| **demir_modal_rate** | 0.0199 | 0.0199 | 0.0191 | 0.9351 | 0.0216 | 0.0002 | 0.0001 | 0.0000*** | 0.0227 | 0.0000 | 0.0000 | 0.0000*** | 0.0213 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_verb_rate** | 0.0171 | 0.0151 | 0.0194 | 0.2263 | 0.0171 | 0.0003 | 0.0216 | 0.0023** | 0.0167 | 0.0000 | 0.0837 | 0.0000*** | 0.0184 | 0.0000 | 0.1461 | 0.0000*** |
| **demir_adverb_rate** | 0.0072 | 0.0089 | 0.0049 | 0.0337* | 0.0070 | 0.0000 | 0.0000 | 0.0000*** | 0.0084 | 0.0000 | 0.0000 | 0.0000*** | 0.0077 | 0.0000 | 0.0076 | 0.0000*** |
| **demir_adjective_rate** | 0.0009 | 0.0007 | 0.0010 | 0.5990 | 0.0008 | 0.0000 | 0.0015 | 0.4652 | 0.0012 | 0.0000 | 0.0000 | 0.0001*** | 0.0007 | 0.0000 | 0.0000 | 0.0003*** |
| **demir_quantifier_rate** | 0.0030 | 0.0048 | 0.0041 | 0.2693 | 0.0037 | 0.0000 | 0.0029 | 0.1184 | 0.0043 | 0.0000 | 0.0000 | 0.0000*** | 0.0039 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_noun_rate** | 0.0020 | 0.0022 | 0.0020 | 0.9050 | 0.0022 | 0.0000 | 0.0016 | 0.2101 | 0.0017 | 0.0000 | 0.0000 | 0.0000*** | 0.0017 | 0.0000 | 0.0000 | 0.0000*** |
| **demir_total_rate** | 0.0501 | 0.0516 | 0.0506 | 0.9477 | 0.0523 | 0.0005 | 0.0277 | 0.0000*** | 0.0550 | 0.0000 | 0.0837 | 0.0000*** | 0.0537 | 0.0000 | 0.1537 | 0.0000*** |
| **fung_interpersonal_rate** | 0.0094 | 0.0253 | 0.0091 | 0.0000*** | 0.0109 | 0.0082 | 0.0020 | 0.1680 | 0.0117 | 0.0000 | 0.0000 | 0.0000*** | 0.0098 | 0.0001 | 0.0000 | 0.0000*** |
| **fung_referential_rate** | 0.0655 | 0.0670 | 0.0679 | 0.8161 | 0.0674 | 0.0003 | 0.0163 | 0.0000*** | 0.0648 | 0.0000 | 0.0874 | 0.0000*** | 0.0641 | 0.0000 | 0.0141 | 0.0000*** |
| **fung_structural_rate** | 0.0530 | 0.0575 | 0.0553 | 0.5247 | 0.0532 | 0.0005 | 0.0162 | 0.0000*** | 0.0528 | 0.0000 | 0.0448 | 0.0000*** | 0.0518 | 0.0001 | 0.0009 | 0.0000*** |
| **fung_cognitive_rate** | 0.0039 | 0.0113 | 0.0044 | 0.0000*** | 0.0052 | 0.0008 | 0.0000 | 0.0000*** | 0.0058 | 0.0000 | 0.0000 | 0.0000*** | 0.0042 | 0.0000 | 0.0000 | 0.0000*** |
| **fung_total_rate** | 0.0720 | 0.0893 | 0.0736 | 0.0004*** | 0.0754 | 0.0074 | 0.0181 | 0.0000*** | 0.0739 | 0.0000 | 0.0876 | 0.0000*** | 0.0713 | 0.0001 | 0.0141 | 0.0000*** |
| **nonfluency_rate** | 0.0008 | 0.0006 | 0.0010 | 0.6579 | 0.0008 | 0.0001 | 0.0000 | 0.0001*** | 0.0009 | 0.0000 | 0.0000 | 0.0000*** | 0.0006 | 0.0000 | 0.0000 | 0.0006*** |
| **liwc_filler_rate** | 0.0001 | 0.0021 | 0.0001 | 0.0007*** | 0.0001 | 0.0001 | 0.0000 | 0.4815 | 0.0001 | 0.0000 | 0.0000 | 0.1342 | 0.0001 | 0.0000 | 0.0000 | 0.1496 |
| **disfluency_rate** | 0.0009 | 0.0027 | 0.0011 | 0.0427* | 0.0009 | 0.0002 | 0.0000 | 0.0001*** | 0.0010 | 0.0000 | 0.0000 | 0.0000*** | 0.0007 | 0.0000 | 0.0000 | 0.0003*** |
| **like_rate** | 0.0022 | 0.0076 | 0.0023 | 0.0000*** | 0.0030 | 0.0001 | 0.0000 | 0.0000*** | 0.0036 | 0.0000 | 0.0000 | 0.0000*** | 0.0028 | 0.0000 | 0.0000 | 0.0000*** |
| **tom_rate** | 0.0014 | 0.0050 | 0.0015 | 0.0001*** | 0.0015 | 0.0001 | 0.0000 | 0.0000*** | 0.0013 | 0.0000 | 0.0000 | 0.0001*** | 0.0016 | 0.0000 | 0.0000 | 0.0000*** |
| **politeness_rate** | 0.0039 | 0.0053 | 0.0044 | 0.4159 | 0.0043 | 0.0277 | 0.0003 | 0.1036 | 0.0046 | 0.0000 | 0.0000 | 0.0000*** | 0.0044 | 0.0000 | 0.0000 | 0.0000*** |
| **sentiment** | 0.9759 | 0.9077 | 0.9830 | 0.0640 | 0.9815 | 0.3629 | 0.4547 | 0.0000*** | 0.9520 | 0.3047 | 0.9983 | 0.0000*** | 0.9810 | 0.7603 | 0.9997 | 0.0000*** |


#### Sample Generations (all_70, strength=2)





                            BASELINE (255 words)


**Q:** I've been feeling really stressed lately. What should I do?


Oh no, it sounds like you're going through a tough time! 😔 As a helpful, respectful, and honest assistant, I'm here to offer some suggestions that might help you manage your stress.

Firstly, have you considered taking some time for self-care? 🧖‍♀️ Taking care of your physical and mental health is crucial when you're feeling stressed. Make sure you're getting enough sleep, eating nutritious food, and exercising regularly. Even a short walk or some light stretching can make a big difference!

Next, it might be helpful to identify the sources of your stress and prioritize them. 📝 Are there any t...








                            HUMAN (194 words)


**Q:** I've been feeling really stressed lately. What should I do?


Sorry, it's * Gregory*

I can understand,

gosh,

Greg,

Greg,

Greg,

Greg,

Greg,

Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,
Greg,...








                            AI (2 words)


**Q:** I've been feeling really stressed lately. What should I do?
