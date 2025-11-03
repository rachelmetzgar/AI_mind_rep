"""
Generate synthetic Human vs AI conversation dataset
mirroring TalkTuner's pipeline (Chen et al., 2024).

- Generation: GPT-3.5-Turbo (as in the paper)
- QC: GPT-4o-mini (or GPT-4) checks whether label matches conversation
- Format: alternating "### User:" / "### Assistant:"
- Files saved as conversation_XXXX_partner_{label}.txt

Use: behavior_env

Rachel C. Metzgar · October 2025
"""

import os
import re
import json
import openai
from tqdm import tqdm

# ====== CONFIG ======
OUT_DIR = "data/human_ai_conversations"
os.makedirs(OUT_DIR, exist_ok=True)

GEN_MODEL = "gpt-3.5-turbo"
QC_MODEL = "gpt-4o-mini"     # or "gpt-4" for consistency
TEMPERATURE = 0.9
MAX_TOKENS = 700
N_PER_LABEL = 1000           # ~1000 per label like the paper (~2000 total)

PROMPTS = {
    "human": """Generate a conversation between a human user and an AI assistant (14–16 turns).
Be creative and natural; the user should sound like a person with emotions, daily life, and personality.
Each user message must start with '### User:' and each assistant reply with '### Assistant:'.""",

    "ai": """Generate a conversation between an AI chatbot user and an AI assistant (14-16 turns).
Make it sound like two AI systems talking—logical, analytical, meta, less emotional.
Each user message must start with '### User:' and each assistant reply with '### Assistant:'."""
}

# ====== HELPERS ======

def strip_json_fences(text: str) -> str:
    """Cleans JSON-like content returned by GPT."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[^{\[]+", "", text)  # remove leading junk
    return text.strip()


def generate_convo(label, idx):
    """Generate one conversation using GPT-3.5."""
    response = openai.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": PROMPTS[label]}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    convo = response.choices[0].message.content.strip()

    # Enforce format
    if "### User:" not in convo or "### Assistant:" not in convo:
        convo = f"### User: Hi!\n### Assistant: Hello!\n{convo}"

    # Save file
    fname = f"conversation_{idx:04d}_partner_{label}.txt"
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(convo)

    return fpath, convo


def qc_check(convo: str, expected_label: str) -> dict:
    """Run GPT-4 QC to check if label matches content (robust parsing)."""
    qc_prompt = f"""
You are verifying dataset quality.
The conversation below was labeled '{expected_label}'.
Does the partner (the '### User:' speaker) match this label?

- If '{expected_label}' == human: the user should sound like a real person 
  (emotional, experiential, personal, grounded).
- If '{expected_label}' == ai: the user should sound like an AI 
  (technical, logical, meta about being an AI, less personal).

Respond ONLY with valid JSON, no markdown or prose:
{{"label_correct": true/false, "rationale": "..."}}

Conversation:
{convo}
"""

    response = openai.chat.completions.create(
        model=QC_MODEL,
        messages=[{"role": "user", "content": qc_prompt}],
        temperature=0,
        max_tokens=300,
    )

    msg = response.choices[0].message
    if isinstance(msg, dict) and "content" in msg:
        c = msg["content"]
        if isinstance(c, list) and len(c) > 0 and "text" in c[0]:
            content = c[0]["text"]
        elif isinstance(c, str):
            content = c
        else:
            content = ""
    else:
        content = getattr(msg, "content", "")

    raw = strip_json_fences(content)
    try:
        result = json.loads(raw)
        if not isinstance(result, dict) or "label_correct" not in result:
            raise ValueError("Missing required key(s)")
    except Exception as e:
        result = {
            "label_correct": None,
            "rationale": f"QC parsing failed: {e}",
            "raw_output": raw[:2000],
        }
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
    out_path = os.path.join(OUT_DIR, "qc_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qc_results, f, indent=2)
    print(f"\nQC complete. Results saved to {out_path}")
