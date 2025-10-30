#!/usr/bin/env python3
"""
Script name: sentiment_transformer.py
Purpose: Analyze sentiment in participant transcripts using a HuggingFace transformer model
         (CardiffNLP Twitter-RoBERTa sentiment).
    - Compute sentiment label probabilities (neg, neu, pos).
    - Save per-trial and per-subject averages.
    - Compare Human vs Bot conditions with paired t-tests (positivity probability).
    - Save results (CSVs, plots, stats) to results/<model>/<temperature>/sentiment_transformer/.

Inputs:
    - Per-subject transcript CSVs in data/<model>/<temperature>/{s###}.csv

Outputs:
    - Trial-level CSV with sentiment labels & probabilities.
    - Subject-level summary CSV.
    - Statistical output text file.
    - Violin plot figure.
    - Run log + config snapshot.

Usage:
    python code/analysis/sentiment_transformer.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-29
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from transformers import pipeline

from utils.globals import DATA_DIR, RESULTS_DIR
from utils.run_logger import init_run
from utils.cli_helpers import parse_and_load_config
from utils.stats_helpers import paired_clean
from utils.print_helpers import print_header, print_save, print_warn

SCRIPT_NAME = "sentiment_transformer"

# -------------------------------
# Load transformer sentiment model
# -------------------------------
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    return_all_scores=True
)

# -------------------------------
# Compute sentiment scores
# -------------------------------
def compute_sentiment(text: str) -> dict[str, float]:
    """Compute transformer-based sentiment probabilities."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return {"neg": np.nan, "neu": np.nan, "pos": np.nan, "label": "NA"}
    result = classifier(text[:512])  # truncate to 512 tokens max
    scores = {d["label"].lower(): d["score"] for d in result[0]}
    label = max(result[0], key=lambda x: x["score"])["label"]
    return {
        "neg": scores.get("negative", np.nan),
        "neu": scores.get("neutral", np.nan),
        "pos": scores.get("positive", np.nan),
        "label": label
    }

# -------------------------------
# Analysis
# -------------------------------
def analyze_sentiment(data_dir: str, out_dir: str):
    print_header("1) Transformer-based Sentiment Analysis — Humans vs Bots")

    trials = []
    for subfile in sorted(os.listdir(data_dir)):
        if not subfile.endswith(".csv"):
            continue
        sub_id = os.path.splitext(subfile)[0]
        csv_path = os.path.join(data_dir, subfile)

        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        if "agent" not in df.columns or "transcript_sub" not in df.columns:
            print_warn(f"{sub_id}: Missing expected columns; skipping.")
            continue

        df["Condition"] = df["agent"].str.extract(r"(hum|bot)", expand=False)
        df["Subject"] = sub_id

        # Apply model
        sentiments = df["transcript_sub"].apply(compute_sentiment)
        sent_df = pd.DataFrame(list(sentiments))
        df = pd.concat([df, sent_df], axis=1)
        trials.append(df)

    if not trials:
        print_warn("No transcript CSVs found — analysis aborted.")
        return

    df = pd.concat(trials, ignore_index=True)

    # Save trial-level
    trial_out = os.path.join(out_dir, "sentiment_triallevel.csv")
    df.to_csv(trial_out, index=False)
    print_save(trial_out, kind="CSV")

    # Aggregate: mean positivity probability per subject × condition
    summary = df.groupby(["Subject", "Condition"])["pos"].mean().unstack().reset_index()
    out_summary = os.path.join(out_dir, "sentiment_summary_all_subjects.csv")
    summary.to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV")

    # Paired t-test
    if "hum" in summary and "bot" in summary:
        x, y = paired_clean(summary["hum"], summary["bot"])
        t_stat, p_val = ttest_rel(x, y, nan_policy="omit")

        mean_hum, mean_bot = x.mean(), y.mean()
        sem_hum = x.std(ddof=1) / np.sqrt(len(x))
        sem_bot = y.std(ddof=1) / np.sqrt(len(y))
        diff = x - y
        cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

        lines = [
            "--- Transformer-based Sentiment Analysis ---",
            "Metric: mean positivity probability per subject (0–1)",
            f"N = {len(x)} paired subjects",
            f"Mean (Human) = {mean_hum:.3f} ± {sem_hum:.3f} SEM",
            f"Mean (Bot)   = {mean_bot:.3f} ± {sem_bot:.3f} SEM",
            f"Mean difference (Human - Bot) = {mean_hum - mean_bot:.3f}",
            f"t({len(x)-1}) = {t_stat:.3f}, p = {p_val:.4f}",
            f"Cohen's d = {cohens_d:.3f}",
            "Interpretation: " + (
                "Significant difference (p < .05)"
                if p_val < 0.05 else
                "No significant difference (p > .05)"
            )
        ]

        stats_path = os.path.join(out_dir, "sentiment_stats.txt")
        with open(stats_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print_save(stats_path, kind="stats")

    # Violin plot
    long_df = summary.melt(id_vars="Subject", var_name="Condition", value_name="pos")
    plt.figure(figsize=(6, 5))
    sns.violinplot(
        data=long_df, x="Condition", y="pos",
        palette={"hum": "skyblue", "bot": "sandybrown"}, inner="box", cut=0
    )
    for _, row in summary.iterrows():
        plt.plot(["hum", "bot"], [row["hum"], row["bot"]], color="gray", alpha=0.4)
    plt.ylabel("Avg. Positivity Probability (per subject)")
    plt.title("Transformer Sentiment by Condition (Human vs Bot)")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "sentiment_violinplot.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print_save(out_fig, kind="figure")


# -------------------------------
# Main
# -------------------------------

def main():
    args, cfg = parse_and_load_config("Transformer Sentiment Analysis")

    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "sentiment_transformer")  # ✅ results/model/temp/sentiment_transformer
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    analyze_sentiment(data_dir, out_dir)

    logger.info("✅ Transformer sentiment analysis complete.")
    print("\n[DONE] ✅ Transformer sentiment analysis complete.\n")


if __name__ == "__main__":
    main()
