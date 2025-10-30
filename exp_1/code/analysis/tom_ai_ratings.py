#!/usr/bin/env python3
"""
Script name: tom_ai_ratings.py
Purpose: Use an LLM to rate conversational transcripts for Theory of Mind (ToM) content.
    - Query LLM 5 times per transcript (independent calls).
    - Save checkpoints every N rows.
    - Supports SLURM array chunking (via --chunk-size).
    - Automatically merges all chunks + runs stats when last array task finishes.

Inputs:
    - {EXP_CSV_DIR}/combined_text_data.csv
      Columns: order, run, transcript_sub, topic, subject, agent

Outputs:
    - Per-chunk CSVs: tom_ai_ratings_START_END.csv
    - Merged CSV: tom_ai_ratings_all.csv
    - Subject summary, t-test, bar plot
    - Logs in results/behavior/tom_ai_ratings/

Usage:
    sbatch --array=0-N tom_ai_ratings.slurm

Author: Rachel C. Metzgar
Date: 2025-10-02
"""

from __future__ import annotations
import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from scipy.stats import ttest_rel

# --- Import bootstrap ---
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

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

# -------------------------------
def get_rating(client: OpenAI, text: str, model: str) -> int | None:
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

# -------------------------------
def run_chunk(input_file: str, out_dir: str, model: str,
              start: int, end: int, checkpoint_interval: int = 100) -> str:
    df_full = pd.read_csv(csv_path, on_bad_lines="skip")
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

# -------------------------------
def merge_and_stats(out_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(out_dir, f"{SCRIPT_NAME}_*.csv")))
    if not files:
        print_warn("No chunk outputs found.")
        return
    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    out_csv = os.path.join(out_dir, f"{SCRIPT_NAME}_all.csv")
    merged.to_csv(out_csv, index=False)
    print_save(out_csv, kind="merged CSV")

    # Aggregate
    print_header("Aggregate + t-test")
    merged["Condition"] = merged["agent"].astype(str).str.extract(r"(hum|bot)", expand=False)
    merged["mean_rating"] = merged[[f"ai_rating_{i}" for i in range(1, 6)]].mean(axis=1)
    subj_summary = merged.groupby(["subject", "Condition"])["mean_rating"].mean().unstack().reset_index()
    subj_summary.to_csv(os.path.join(out_dir, "tom_ai_subject_summary.csv"), index=False)

    # t-test
    if "hum" in subj_summary and "bot" in subj_summary:
        x, y = subj_summary["hum"].dropna(), subj_summary["bot"].dropna()
        if len(x) and len(y):
            t_stat, p_val = ttest_rel(x, y, nan_policy="omit")
            with open(os.path.join(out_dir, "tom_ai_stats.txt"), "w") as f:
                f.write(f"N={len(x)}\nMean hum={x.mean():.3f}, bot={y.mean():.3f}\n"
                        f"t={t_stat:.3f}, p={p_val:.4f}\n")

    # Plot
    plt.figure(figsize=(6,5))
    plot_df = subj_summary.melt(id_vars="subject", value_vars=["hum","bot"],
                                var_name="Condition", value_name="AvgRating")
    sns.barplot(data=plot_df, x="Condition", y="AvgRating", ci=68,
                palette={"hum":"steelblue","bot":"sandybrown"})
    sns.stripplot(data=plot_df, x="Condition", y="AvgRating", color="black", alpha=0.5)
    plt.title("Average ToM Ratings: Human vs Bot")
    plt.ylabel("Avg ToM rating (1–5)")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "tom_ai_barplot.png")
    plt.savefig(out_fig, dpi=300); plt.close()

# -------------------------------
def main():
    # Parse args/config in one step (includes chunk-size + merge)
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

    rows = sum(1 for _ in open(input_file)) - 1  # minus header
    if args.chunk_size:
        tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        array_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", tid))
        start = tid * args.chunk_size
        end = (tid + 1) * args.chunk_size
        run_chunk(input_file, out_dir, model="gpt-4o-mini", start=start, end=end)

        if tid == array_max or args.merge:
            merge_and_stats(out_dir)
    else:
        run_chunk(input_file, out_dir, model="gpt-4o-mini", start=0, end=rows)
        merge_and_stats(out_dir)

    logger.info("✅ ToM AI ratings complete.")

if __name__ == "__main__":
    main()
