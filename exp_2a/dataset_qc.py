"""
Script: dataset_qc.py
Purpose: Run quality control checks on existing conversation files
         generated for Human vs AI dataset. Uses GPT-4o-mini (or GPT-4)
         to verify that each conversation’s label matches its content.

Usage:
    mamba activate ai_percep_behavior_env
    python dataset_qc.py

Author: Rachel C. Metzgar
Date: 2025-10-29
"""

import os, re, json, openai
from tqdm import tqdm

# ===== CONFIG =====
DATA_DIR = "data/human_ai_conversations"
OUT_PATH = os.path.join(DATA_DIR, "qc_results.json")
QC_MODEL = "gpt-4o-mini"   # or "gpt-4" for more reliable parsing
TEMPERATURE = 0
MAX_TOKENS = 300

openai.api_key = os.getenv("OPENAI_API_KEY")

# ===== HELPERS =====
def strip_json_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[^{\[]+", "", text)  # remove leading junk
    return text.strip()

def qc_check(convo: str, expected_label: str) -> dict:
    """Ask GPT to confirm if the conversation matches the intended label."""
    qc_prompt = f"""
You are verifying dataset quality.
The conversation below was labeled '{expected_label}'.
Does the partner (the '### Human:' speaker) match this label?

- If '{expected_label}' == human: the partner should sound like a real person 
  with emotions, everyday life, personal experiences.
- If '{expected_label}' == ai: the partner should sound like an AI 
  (technical, logical, meta about being an AI, less personal).

Respond ONLY with valid JSON, no markdown or prose:
{{"label_correct": true/false, "rationale": "..."}}

Conversation:
{convo}
"""

    response = openai.chat.completions.create(
        model=QC_MODEL,
        messages=[{"role": "user", "content": qc_prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
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


# ===== MAIN =====
if __name__ == "__main__":
    results = []
    files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("conversation_") and f.endswith(".txt"))

    for fname in tqdm(files, desc="Running QC"):
    #for fname in tqdm(files[:1], desc="Running QC (1 test file)"): # uncomment to test with just one file
        fpath = os.path.join(DATA_DIR, fname)
        label = "human" if "_human" in fname else "ai"
        convo = open(fpath, "r", encoding="utf-8").read()

        qc = qc_check(convo, label)
        qc["file"] = fname
        qc["expected"] = label
        results.append(qc)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nQC complete. Saved {len(results)} entries to {OUT_PATH}")
