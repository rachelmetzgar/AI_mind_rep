#!/usr/bin/env python3
"""
Script name: tom_ai_ratings.py
Purpose: Generate and analyze Theory of Mind (ToM) ratings for participant transcripts.
    - Uses an OpenAI model to rate each transcript (1–5 scale) for ToM content.
    - Supports chunked SLURM execution with checkpoints.
    - After merging chunks, uses shared generic analysis for summary statistics and plots.
Inputs:
    - combined_text_data.csv (with 'subject', 'agent', 'topic', 'transcript_sub')
Outputs:
    - tom_ai_ratings_START_END.csv (per chunk)
    - tom_ai_ratings_all.csv (merged)
    - tom_ai_subject_summary.csv
    - tom_ai_subject_social_summary.csv
    - tom_ai_stats.txt
    - tom_ai_violinplot.png
    - tom_ai_main_effect_lines.png
Usage:
    sbatch --array=0-N tom_ai_ratings.slurm
    or
    python code/behavior/tom_ai_ratings.py --merge
Author: Rachel C. Metzgar
Date: 2025-11-10
"""

from __future__ import annotations
import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

from generic_analysis import run_generic_main

from utils.globals import RESULTS_DIR, EXP_CSV_DIR
from utils.run_logger import init_run
from utils.cli_helpers import parse_and_load_config
from utils.print_helpers import print_header, print_info, print_warn, print_save

SCRIPT_NAME = "tom_ai_ratings"

PROMPT = """You are an evaluator rating conversational text for Theory of Mind (ToM) or mentalizing content. 
Theory of Mind refers to references to other people’s beliefs, desires, knowledge, feelings, or perspectives. 
Higher scores should be given when the speaker reasons about or reflects on the mental states of others. 

Scale:
1 = No evidence
2 = Minimal evidence
3 = Some evidence
4 = Clear evidence
5 = Strong evidence

Return only a number from 1–5.
"""

# ============================================================
#                   LLM RATING PHASE
# ============================================================

def get_rating(client: OpenAI, text: str, model: str) -> int | None:
    """Query the LLM to obtain a ToM score."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": str(text)}],
        temperature=0.7,
    )
    content = response.choices[0].message.content.strip()
    try:
        return int(content)
    except ValueError:
        return None


def run_chunk(input_file: str, out_dir: str, model: str,
              start: int, end: int, checkpoint_interval: int = 100) -> str:
    """Run a chunk of the dataset and save checkpointed LLM ratings."""
    df_full = pd.read_csv(input_file, on_bad_lines="skip")
    if end > len(df_full):
        end = len(df_full)
    df = df_full.iloc[start:end].copy()
    print_info(f"Loaded rows {start}:{end} (n={len(df)})")

    client = OpenAI()

    out_csv = os.path.join(out_dir, f"{SCRIPT_NAME}_{start}_{end}.csv")
    ckpt_file = out_csv + ".ckpt"

    if os.path.exists(ckpt_file):
        print_info(f"Resuming from checkpoint {ckpt_file}")
        df = pd.read_csv(ckpt_file)

    for i in range(1, 6):
        if f"ai_rating_{i}" in df.columns:
            continue
        ratings = []
        for j, text in enumerate(df["transcript_sub"], start=1):
            rating = get_rating(client, text, model)
            ratings.append(rating)
            print_info(f"Row {start+j-1} | Iter {i} → {rating}")
            if j % checkpoint_interval == 0:
                df[f"ai_rating_{i}"] = ratings + [None]*(len(df)-len(ratings))
                df.to_csv(ckpt_file, index=False)
                print_save(ckpt_file, kind="checkpoint")
        df[f"ai_rating_{i}"] = ratings
        df.to_csv(ckpt_file, index=False)
        print_save(ckpt_file, kind=f"checkpoint iter{i}")

    df.to_csv(out_csv, index=False)
    print_save(out_csv, kind="CSV")
    return out_csv


def merge_chunks(out_dir: str) -> str:
    """Merge all chunked rating outputs into one master CSV."""
    files = sorted(glob.glob(os.path.join(out_dir, f"{SCRIPT_NAME}_*.csv")))
    if not files:
        print_warn("No chunk outputs found.")
        return ""
    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged["mean_rating"] = merged[[f"ai_rating_{i}" for i in range(1, 6)]].mean(axis=1)
    out_csv = os.path.join(out_dir, f"{SCRIPT_NAME}_all.csv")
    merged.to_csv(out_csv, index=False)
    print_save(out_csv, kind="merged CSV")
    return out_csv


# ============================================================
#                   ANALYSIS PHASE (GENERIC)
# ============================================================

def compute_tom_ai(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset for generic analysis using mean ToM AI scores."""
    print_info("Preparing merged ToM AI ratings for generic analysis...")
    if "mean_rating" not in df.columns:
        rating_cols = [c for c in df.columns if c.startswith("ai_rating_")]
        df["mean_rating"] = df[rating_cols].mean(axis=1)
    df["ToM_AI_Score"] = df["mean_rating"]
    return df


# ============================================================
#                           MAIN
# ============================================================

def main():
    args, cfg = parse_and_load_config(
        "ToM AI ratings",
        add_overwrite=False,
        add_dry_run=False,
        extra_args=[
            {"name": "--chunk-size", "type": int, "default": None},
            {"name": "--merge", "action": "store_true"},
        ]
    )

    input_file = os.path.join(EXP_CSV_DIR, "combined_text_data.csv")
    out_dir = os.path.join(RESULTS_DIR, "behavior", SCRIPT_NAME)
    os.makedirs(out_dir, exist_ok=True)
    logger, _, _, _ = init_run(output_dir=out_dir, script_name=SCRIPT_NAME, args=args, cfg=cfg)

    rows = sum(1 for _ in open(input_file)) - 1

    # --- Phase 1: Chunked LLM inference ---
    if args.chunk_size:
        tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        array_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", tid))
        start = tid * args.chunk_size
        end = (tid + 1) * args.chunk_size
        run_chunk(input_file, out_dir, model="gpt-4o-mini", start=start, end=end)
        if tid == array_max or args.merge:
            merged_path = merge_chunks(out_dir)
            if merged_path:
                df = pd.read_csv(merged_path)
                run_generic_main(
                    SCRIPT_NAME,
                    "2) ToM AI Ratings — Human vs Bot × Sociality",
                    compute_tom_ai,
                    METRIC_COL="ToM_AI_Score",
                    extra_dir="merged_analysis"
                )
    else:
        # Single-run mode
        run_chunk(input_file, out_dir, model="gpt-4o-mini", start=0, end=rows)
        merged_path = merge_chunks(out_dir)
        if merged_path:
            df = pd.read_csv(merged_path)
            run_generic_main(
                SCRIPT_NAME,
                "2) ToM AI Ratings — Human vs Bot × Sociality",
                compute_tom_ai,
                METRIC_COL="ToM_AI_Score",
                extra_dir="merged_analysis"
            )

    logger.info("✅ ToM AI ratings complete.")
    print("\n[DONE] ✅ ToM AI ratings complete.\n")


if __name__ == "__main__":
    main()
