# AI Mind Rep

**Author:** Rachel C. Metzgar  
**Repo root:** `ai_mind_rep/`  
**Python:** 3.11+  

This repo contains experiments examining how AI generates conversations when it is instructed it is talking to a human and AI partner. 

exp_1: This project generates **synthetic human–AI conversations** and runs a suite of **behavioral text analyses** (sentiment, politeness, hedging, ToM, questions, word count, etc.) comparing how the conversations differ when the LLM generates text for human and AI conversation partners.

exp_2 (WIP): This project reproduces the *TalkTuner*–style probing pipeline (Chen et al., 2024) to test how LLaMA-2 internal representations distinguish **Human** vs **AI** partners during dialogue.

---

## Environments

Create each environment with:

```bash
conda env create -f envs/behavior_env.yml
conda env create -f envs/llama2_env.yml
```
