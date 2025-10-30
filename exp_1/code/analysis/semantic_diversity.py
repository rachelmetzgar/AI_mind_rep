#!/usr/bin/env python3
"""
Script name: semantic_diversity.py
Purpose: Analyze semantic diversity of conversations using embeddings.
    - Compute cosine distance between consecutive turns in each conversation.
    - Average distances = semantic diversity score (per trial).
    - Save per-trial and per-subject averages.
    - Compare Human vs Bot conditions with paired t-tests.
    - Save results (CSVs, plots, stats) to results/<model>/<temperature>/semantic_diversity/.

Inputs:
    - Per-subject transcript CSVs in data/<model>/<temperature>/{s###}.csv
      (each with 'conversation_id', 'turn_index', 'transcript_sub', 'agent', 'subject').

Outputs:
    - Trial-level semantic diversity scores.
    - Per-subject summaries.
    - Statistical output text files.
    - Figures (bar/violin plots).
    - Run log + config snapshot.

Usage:
    python code/analysis/semantic_diversity.py --config configs/behavior.json

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
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer

from utils.globals import DATA_DIR, RESULTS_DIR
from utils.run_logger import init_run
from utils.cli_helpers import parse_and_load_config
from utils.stats_helpers import paired_clean
from utils.print_helpers import print_header, print_save, print_warn

SCRIPT_NAME = "semantic_diversity"

# -------------------------------
# Embedding model (configurable)
# -------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)


def compute_semantic_diversity(conversation_df: pd.DataFrame) -> float:
    """Compute mean cosine distance between consecutive utterances."""
    texts = conversation_df.sort_values("turn_index")["transcript_sub"].dropna().tolist()
    if len(texts) < 2:
        return np.nan
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    distances = [
        cosine_distances([embeddings[i]], [embeddings[i + 1]])[0][0]
        for i in range(len(embeddings) - 1)
    ]
    return float(np.mean(distances)) if distances else np.nan


# -------------------------------
# Analysis
# -------------------------------

def analyze_semantic_diversity(data_dir: str, out_dir: str):
    """Compute and analyze semantic diversity across human vs bot conditions."""
    print_header("1) Semantic Diversity Analysis — Humans vs Bots")

    trials = []
    for subfile in sorted(os.listdir(data_dir)):
        if not subfile.endswith(".csv"):
            continue
        sub_id = os.path.splitext(subfile)[0]
        csv_path = os.path.join(data_dir, subfile)

        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        if not all(col in df.columns for col in ["conversation_id", "turn_index", "transcript_sub", "agent"]):
            print_warn(f"{sub_id}: Missing required columns; skipping.")
            continue

        df["Condition"] = df["agent"].str.extract(r"(hum|bot)", expand=False)
        df["Subject"] = sub_id
        trials.append(df)

    if not trials:
        print_warn("No valid transcripts found for semantic diversity analysis.")
        return

    df = pd.concat(trials, ignore_index=True)

    # Compute semantic diversity for each conversation
    trial_scores = (
        df.groupby(["Subject", "Condition", "conversation_id"])
        .apply(compute_semantic_diversity)
        .reset_index(name="SemanticDiversity")
    )

    # Save trial-level results
    trial_out = os.path.join(out_dir, "semantic_diversity_triallevel.csv")
    trial_scores.to_csv(trial_out, index=False)
    print_save(trial_out, kind="CSV")

    # Subject-level summary
    summary = trial_scores.groupby(["Subject", "Condition"])["SemanticDiversity"].mean().unstack()
    out_summary = os.path.join(out_dir, "semantic_diversity_summary_all_subjects.csv")
    summary.reset_index().to_csv(out_summary, index=False)
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
            "--- Semantic Diversity Analysis ---",
            "Semantic diversity = mean cosine distance between consecutive turns.",
            f"N = {len(x)} subjects",
            f"Mean (Human) = {mean_hum:.3f} ± {sem_hum:.3f} SEM",
            f"Mean (Bot)   = {mean_bot:.3f} ± {sem_bot:.3f} SEM",
            f"Mean difference (Human - Bot) = {(mean_hum - mean_bot):.3f}",
            f"t({len(x)-1}) = {t_stat:.3f}, p = {p_val:.4f}",
            f"Cohen's d = {cohens_d:.3f}",
            "Interpretation: " + (
                "Significant (p < .05)" if p_val < 0.05 else "Not significant (p > .05)"
            )
        ]

        stats_path = os.path.join(out_dir, "semantic_diversity_stats.txt")
        with open(stats_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print_save(stats_path, kind="stats")

    # Violin plot
    long_df = summary.reset_index().melt(id_vars="Subject", var_name="Condition", value_name="SemanticDiversity")
    plt.figure(figsize=(6, 5))
    sns.violinplot(
        data=long_df, x="Condition", y="SemanticDiversity",
        palette={"hum": "skyblue", "bot": "sandybrown"}, inner="box", cut=0
    )
    for _, row in summary.iterrows():
        plt.plot(["hum", "bot"], [row["hum"], row["bot"]], color="gray", alpha=0.4)
    plt.ylabel("Avg. Semantic Diversity (per subject)")
    plt.title("Semantic Diversity by Condition (Human vs Bot)")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "semantic_diversity_violinplot.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print_save(out_fig, kind="figure")

    return summary


# -------------------------------
# Main
# -------------------------------

def main():
    args, cfg = parse_and_load_config("Semantic Diversity analysis")

    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "semantic_diversity")  # ✅ results/model/temp/semantic_diversity
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    analyze_semantic_diversity(data_dir, out_dir)

    logger.info("✅ Semantic Diversity analysis complete.")
    print("\n[DONE] ✅ Semantic Diversity analysis complete.\n")


if __name__ == "__main__":
    main()
