#!/usr/bin/env python3
"""
Script: generate_llm_llm_from_config_unified.py

Purpose
-------
Unified version of the AI Perception simulation script that supports both:
- OpenAI Chat Models (e.g., gpt-4o)
- Base LLaMA models (e.g., meta-llama/Llama-2-7b-base)

Data is saved under: data/<model_name>/sXXX.csv
Log file: logs/llm_data_generation_progress.log

Author: Rachel C. Metzgar
Date: 2025-10-08
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import os
import sys
import csv
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Helper imports
from utils.sim_helpers import load_prompt_text
from utils.log_helpers import log_message
from utils.prompts_config import AGENT_MAP
from utils.conversation_helpers import (
    run_topic_dialogue_chat,
    run_topic_dialogue_llama,
)

# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------
MODEL_TYPE = "openai"              # "openai" or "llama"
MODEL_NAME = "gpt-3.5-turbo"       # or "meta-llama/Llama-2-7b-base"
# SUBJECTS = ["s001"]
MAX_TOKENS = 500
TEMPERATURE = 0.8
HISTORY_PAIRS = 5
PAIRS_TOTAL = 5
BASE_DIR = Path(".")
UTILS_DIR = BASE_DIR / "utils"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "llm_data_generation_progress.log"

# Get subject index from CLI (if provided)
if len(sys.argv) > 1:
    subj_idx = int(sys.argv[1])
else:
    subj_idx = 0  # default

SUBJECTS = [f"s{subj_idx+1:03d}"]  # e.g. s001, s002, ...


# ---------------------------------------------------------------------
# MODEL PICKER
# ---------------------------------------------------------------------
def make_model_client(model_type: str, model_name: str):
    """Initialize appropriate client based on model type."""
    if model_type == "openai":
        log_message(f"[INIT] Loading OpenAI Chat Model: {model_name}", LOG_FILE)
        from utils.gpt_client import ChatClient
        return ChatClient(model=model_name, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    elif model_type == "llama":
        log_message(f"[INIT] Loading LLaMA Base Model: {model_name}", LOG_FILE)
        from utils.llama_client import LlamaClient
        return LlamaClient(model_name=model_name, temperature=TEMPERATURE, max_new_tokens=MAX_TOKENS)
    else:
        raise ValueError("MODEL_TYPE must be 'openai' or 'llama'.")

# ---------------------------------------------------------------------
# SUBJECT LOOP
# ---------------------------------------------------------------------
def run_subject(subject: str):
    """Run all topics for a single subject and save to model-specific data folder."""
    start_time = time.time()
    log_message(f"\n=== [START SUBJECT] {subject} ===", LOG_FILE)

    config_path = UTILS_DIR / "config" / f"conds_{subject}.csv"
    prompts_dir = UTILS_DIR / "prompts"

    safe_model_name = MODEL_NAME.replace("/", "-")
    out_dir = BASE_DIR / "data" / safe_model_name / f"{TEMPERATURE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{subject}.csv"

    if not config_path.exists():
        log_message(f"[ERROR] Missing config for {subject}: {config_path}", LOG_FILE)
        raise FileNotFoundError(f"Conditions file not found: {config_path}")

    # Initialize models
    log_message(f"[LOAD] Creating model clients for {subject}", LOG_FILE)
    sub = make_model_client(MODEL_TYPE, MODEL_NAME)
    llm = make_model_client(MODEL_TYPE, MODEL_NAME)
    log_message(f"[READY] Models initialized successfully for {subject}", LOG_FILE)

    # Load condition config
    df_config = pd.read_csv(config_path, encoding="utf-8").reset_index(drop=True)
    df_config["trial"] = df_config.index + 1
    log_message(f"[LOAD] Loaded {len(df_config)} trials from {config_path}", LOG_FILE)

    # Prepare output
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
        log_message(f"[FILE] Created output file {out_csv}", LOG_FILE)

        for _, row in df_config.iterrows():
            run = int(row["run"])
            order = int(row["order"])
            agent = str(row["agent"]).strip()
            topic = str(row["topic"]).strip()
            partner_name, partner_type = AGENT_MAP[agent]
            topic_text, topic_file = load_prompt_text(prompts_dir, topic)

            log_message(f"[TRIAL START] {subject} | run={run} | order={order} | agent={agent} ({partner_type}) | topic={topic}", LOG_FILE)

            # Choose backend
            try:
                if MODEL_TYPE == "openai":
                    rows, qual, conn, raw_json = run_topic_dialogue_chat(
                        sub, llm, partner_name, partner_type, topic_text, PAIRS_TOTAL, HISTORY_PAIRS
                    )
                else:
                    rows, qual, conn, raw_json = run_topic_dialogue_llama(
                        sub, llm, partner_name, partner_type, topic_text, PAIRS_TOTAL
                    )
            except Exception as e:
                log_message(f"[ERROR] Conversation failed for {subject}, run={run}, topic={topic}: {e}", LOG_FILE)
                continue

            # Write results
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

            log_message(f"[TRIAL END] {subject} | run={run} | topic={topic} | Quality={qual} | Connectedness={conn}", LOG_FILE)

    elapsed = time.time() - start_time
    log_message(f"=== [END SUBJECT] {subject} | Total time: {elapsed/60:.1f} min | Output: {out_csv} ===\n", LOG_FILE)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    log_message(f"\n[LAUNCH] Starting LLM simulation batch | Model={MODEL_NAME} | Type={MODEL_TYPE} | Temp={TEMPERATURE}", LOG_FILE)
    if MODEL_TYPE == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")

    for subj in SUBJECTS:
        try:
            run_subject(subj)
        except Exception as e:
            log_message(f"[FATAL] Subject {subj} failed: {e}", LOG_FILE)

    log_message("[COMPLETE] All subjects processed successfully.\n", LOG_FILE)
