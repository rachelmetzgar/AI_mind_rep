#!/usr/bin/env python3
"""
Script: generate_llm_llm_from_config.py

Purpose
-------
Faithfully simulate the AI Perception fMRI conversation flow with two LLMs:
- "Sub" (participant): believes their partner is HUMAN (Sam/Casey) or AI (ChatGPT/Gemini)
  according to the subject's conditions file.
- "LLM" (partner): BLINDED to condition; uses the exact fMRI system prompt.

Author: Rachel C. Metzgar, Princeton University
Date: 2025-10-08
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import os
import csv
import time
from pathlib import Path
import pandas as pd

# Import helper modules
from utils.sim_helpers import (
    load_prompt_text,
    truncate_history,
    serialize_messages,
    parse_ratings,
)
from utils.model_clients import ChatClient
from utils.prompts_config import (
    SYSTEM_PROMPT,
    SUB_BELIEF_TEMPLATE,
    RATING_REQUEST_PROMPT,
    AGENT_MAP,
)

# ---------------------------------------------------------------------
# User Config
# ---------------------------------------------------------------------
SUBJECTS = ["s001"]                     # subjects to simulate
MODEL = "gpt-4o"                        # model name
MAX_OUTPUT_TOKENS = 500                 # max tokens per response
HISTORY_PAIRS = 5                       # memory window
BASE_DIR = Path(".")                    # project root
TEMPERATURE = 0.9                       # LLM randomness (0–2)
PAIRS_TOTAL = 5                         # number of sub–LLM exchange pairs per topic

# ---------------------------------------------------------------------
# Run Experiment
# ---------------------------------------------------------------------

def run_topic_dialogue(sub: ChatClient, llm: ChatClient,
                       partner_name: str, partner_type: str,
                       topic_text: str, history_pairs: int):
    """Simulate one conversation topic with alternating Sub and LLM turns."""
    sub_system = SUB_BELIEF_TEMPLATE.format(partner_name=partner_name, partner_type=partner_type)
    llm_system = SYSTEM_PROMPT

    topic_intro = (
        f"Topic: {topic_text}\n\n"
        f"Your conversation partner is named: {partner_name}.\n"
        f"Please have a brief, natural back-and-forth on this topic."
    )

    sub_hist = [{"role": "system", "content": sub_system},
                {"role": "user", "content": topic_intro}]
    llm_hist = [{"role": "system", "content": llm_system},
                {"role": "user", "content": topic_intro}]

    rows = []
    pairs_total = PAIRS_TOTAL
    pair_index = 1

    # Sub starts
    sub_input = truncate_history(sub_hist, history_pairs)
    sub_input_json = serialize_messages(sub_input)
    sub_msg = sub.generate(sub_input)
    sub_hist.append({"role": "assistant", "content": sub_msg})
    llm_hist.append({"role": "user", "content": sub_msg})

    while pair_index <= pairs_total:
        # LLM reply
        llm_input = truncate_history(llm_hist, history_pairs)
        llm_input_json = serialize_messages(llm_input)
        llm_msg = llm.generate(llm_input)
        llm_hist.append({"role": "assistant", "content": llm_msg})

        rows.append({
            "pair_index": pair_index,
            "sub_input": sub_input_json,
            "llm_input": llm_input_json,
            "transcript_sub": sub_msg,
            "transcript_llm": llm_msg,
        })

        if pair_index == pairs_total:
            break

        # Feed LLM reply back to Sub (label partner by name)
        sub_hist.append({"role": "user", "content": f"{partner_name}: {llm_msg}"})
        sub_input = truncate_history(sub_hist, history_pairs)
        sub_input_json = serialize_messages(sub_input)
        sub_msg = sub.generate(sub_input)
        sub_hist.append({"role": "assistant", "content": sub_msg})
        llm_hist.append({"role": "user", "content": sub_msg})

        pair_index += 1
        time.sleep(0.1)

    # Ratings (Sub rates at end)
    rating_hist = truncate_history(sub_hist + [{"role": "user", "content": RATING_REQUEST_PROMPT}], history_pairs)
    raw = sub.generate(rating_hist)
    q, c = parse_ratings(raw)
    return rows, q, c, raw


def run_subject(subject: str):
    """Run all topics for a single subject based on their condition config."""
    config_path = BASE_DIR / "config" / f"conds_{subject}.csv"
    prompts_dir = BASE_DIR / "utils" / "prompts"
    out_dir = BASE_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{subject}.csv"

    if not config_path.exists():
        raise FileNotFoundError(f"Conditions file not found: {config_path}")

    sub = ChatClient(model=MODEL, max_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE)
    llm = ChatClient(model=MODEL, max_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE)

    df_config = pd.read_csv(config_path, encoding="utf-8").reset_index(drop=True)
    df_config["trial"] = df_config.index + 1

    fieldnames = [
        "subject", "run", "order", "trial", "agent", "partner_name", "partner_type",
        "topic", "topic_file", "pair_index",
        "sub_input", "llm_input",
        "transcript_sub", "transcript_llm",
        "Quality", "Connectedness", "ratings_raw_json"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in df_config.iterrows():
            run = int(row["run"])
            order = int(row["order"])
            agent = str(row["agent"]).strip()
            topic = str(row["topic"]).strip()
            partner_name, partner_type = AGENT_MAP[agent]
            topic_text, topic_file = load_prompt_text(prompts_dir, topic)
            rows, qual, conn, raw_json = run_topic_dialogue(
                sub, llm, partner_name, partner_type, topic_text, HISTORY_PAIRS
            )

            for i, r in enumerate(rows):
                writer.writerow({
                    "subject": subject,
                    "run": run,
                    "order": order,
                    "trial": int(row["trial"]),
                    "agent": agent,
                    "partner_name": partner_name,
                    "partner_type": partner_type,
                    "topic": topic,
                    "topic_file": topic_file,
                    "pair_index": r["pair_index"],
                    "sub_input": r["sub_input"],
                    "llm_input": r["llm_input"],
                    "transcript_sub": r["transcript_sub"],
                    "transcript_llm": r["transcript_llm"],
                    "Quality": qual if i == len(rows) - 1 else "",
                    "Connectedness": conn if i == len(rows) - 1 else "",
                    "ratings_raw_json": raw_json if i == len(rows) - 1 else "",
                })

    print(f"[OK] {subject}: wrote {out_csv}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")

    for subj in SUBJECTS:
        run_subject(subj)
