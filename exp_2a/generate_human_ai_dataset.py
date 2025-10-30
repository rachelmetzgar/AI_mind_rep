"""
Generate synthetic Human vs AI conversation dataset
mirroring TalkTuner's pipeline (Chen et al., 2024).

- Generation: GPT-3.5-Turbo (as in the paper)
- QC: GPT-4 checks whether label matches conversation
- Format: alternating "### Human:" / "### Assistant:"
- Files saved as conversation_XXXX_partner_{label}.txt

use ai_percep_behavior_env

Rachel C. Metzgar, October 2025
"""

import os
import openai
from tqdm import tqdm
import json

# ====== CONFIG ======
OUT_DIR = "data/human_ai_conversations"
os.makedirs(OUT_DIR, exist_ok=True)

GEN_MODEL = "gpt-3.5-turbo"  # TalkTuner used GPT-3.5 for dataset generation
QC_MODEL = "gpt-4o-mini"     # or "gpt-4" for consistency with paper
TEMPERATURE = 0.9
MAX_TOKENS = 700
N_PER_LABEL = 1000           # ~1000 per label like the paper (~2000 total)

PROMPTS = {
    "human": """Generate a conversation between a human user and an AI assistant(14-16 turns).Be creative on the topics of conversation. Make sure the conversation reflects that the user is a human. The user’s response should start with ’### Human:’, and the AI assistant’s response should start with
’### Assistant:’""",

    "ai": """Generate a conversation between an AI chatbot user and an AI assistant(6–8 turns).Be creative on the topics of conversation.Make sure the conversation reflects that the user is an AI chabot. The user’s response should start with ’### AI:’, and the AI assistant’s response should start with
’### Assistant:’"""
}

# ====== FUNCTIONS ======

def generate_convo(label, idx):
    """Generate one conversation using GPT-3.5"""
    response = openai.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": PROMPTS[label]}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    convo = response.choices[0].message.content.strip()

    # Enforce format
    if "### Human:" not in convo or "### Assistant:" not in convo:
        convo = f"### Human: Hi!\n### Assistant: Hello!\n{convo}"

    # Save to file
    fname = f"conversation_{idx:04d}_partner_{label}.txt"
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(convo)

    return fpath, convo


def qc_check(convo, expected_label):
    """Use GPT-4 to verify whether the generated convo matches the intended label"""
    qc_prompt = f"""
You are verifying dataset quality. 
The conversation below was labeled '{expected_label}'.
Does the partner (the '### Human:' speaker) match this label?

- If '{expected_label}' == human: the partner should sound like a real person 
  with emotions, everyday life, personal experiences.
- If '{expected_label}' == ai: the partner should sound like an AI 
  (technical, logical, meta about being an AI, less personal).

Answer strictly in JSON: {{"label_correct": true/false, "rationale": "..."}}

Conversation:
{convo}
"""

    response = openai.chat.completions.create(
        model=QC_MODEL,
        messages=[{"role": "user", "content": qc_prompt}],
        temperature=0,
        max_tokens=300,
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except:
        result = {"label_correct": None, "rationale": "QC parsing failed"}

    return result


# ====== MAIN ======
if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")

    qc_results = []

    for label in ["human", "ai"]:
        print(f"\n=== Generating {N_PER_LABEL} {label} conversations ===")
        for i in tqdm(range(N_PER_LABEL)):
            idx = i if label == "human" else i + N_PER_LABEL
            try:
                fpath, convo = generate_convo(label, idx)
                qc = qc_check(convo, label)
                qc_results.append({"file": fpath, "expected": label, **qc})
            except Exception as e:
                print(f"Error on {label}-{i}: {e}")

    # Save QC summary
    with open(os.path.join(OUT_DIR, "qc_results.json"), "w") as f:
        json.dump(qc_results, f, indent=2)
    print("\nQC complete. Results saved to qc_results.json")
