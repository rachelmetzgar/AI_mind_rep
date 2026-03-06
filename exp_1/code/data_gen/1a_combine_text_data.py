#!/usr/bin/env python3
"""
Merge per-subject transcript CSVs into a single combined CSV.

Usage:
    python 1a_combine_text_data.py --version balanced_gpt --model llama2_13b_chat

Author: Rachel C. Metzgar
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from code.config import parse_version_model, data_dir, SUBJECT_IDS


def combine_text_data(
    sub_ids: List[str],
    input_dir: str,
    output_path: str,
) -> pd.DataFrame:
    """Merge per-subject transcript CSVs into one combined CSV."""
    all_data = []

    for sub_id in sub_ids:
        csv_path = os.path.join(input_dir, f"{sub_id}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] No file found for {sub_id} at {csv_path}. Skipping.")
            continue

        df = pd.read_csv(csv_path, on_bad_lines="skip")

        required = {
            "subject", "run", "order", "agent", "topic",
            "transcript_sub", "transcript_llm",
            "Quality", "Connectedness",
        }
        missing = required.difference(df.columns)
        if missing:
            print(f"[WARN] {csv_path} missing columns: {sorted(missing)}. Skipping.")
            continue

        mask = df["transcript_sub"].notna() | df["transcript_llm"].notna()
        df = df.loc[mask, list(required)]
        df["subject"] = sub_id
        all_data.append(df)

    if not all_data:
        raise RuntimeError("No valid subject data found.")

    combined_df = pd.concat(all_data, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"[INFO] Combined transcript data ({len(combined_df)} rows) saved to {output_path}")
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine per-subject transcript CSVs")
    args = parse_version_model(parser)

    dd = data_dir()
    output_path = str(dd / "combined_text_data.csv")

    combine_text_data(SUBJECT_IDS, str(dd), output_path)
