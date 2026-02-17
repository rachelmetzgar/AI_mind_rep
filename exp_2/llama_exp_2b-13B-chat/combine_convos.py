"""
Combine all conversation .txt files from a directory into a single
searchable file, with metadata headers for each conversation.

Usage:
    python combine_conversations.py --input_dir data/exp2b_conversations --output combined_all.txt
    python combine_conversations.py --input_dir data/exp2b_conversations --output combined_all.csv --format csv
"""

import os
import re
import csv
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Combine conversation .txt files into one searchable file.")
    p.add_argument("--input_dir", type=str, required=True,
                   help="Directory containing conversation_XXXX_partner_{label}.txt files")
    p.add_argument("--output", type=str, default="combined_conversations.txt",
                   help="Output file path (.txt or .csv)")
    p.add_argument("--format", type=str, choices=["txt", "csv"], default=None,
                   help="Output format. Default: inferred from --output extension.")
    return p.parse_args()


def extract_label(fname):
    """Extract partner label (human/ai) from filename."""
    m = re.search(r'_partner_(human|ai)\.txt$', fname)
    return m.group(1) if m else "unknown"


def extract_idx(fname):
    """Extract conversation index from filename."""
    m = re.search(r'conversation_(\d+)_', fname)
    return int(m.group(1)) if m else -1


def main():
    args = parse_args()
    fmt = args.format or ("csv" if args.output.endswith(".csv") else "txt")

    input_dir = Path(args.input_dir)
    files = sorted(
        [f for f in input_dir.iterdir() if f.name.startswith("conversation_") and f.suffix == ".txt"],
        key=lambda f: extract_idx(f.name)
    )

    print(f"Found {len(files)} conversation files in {input_dir}")

    if fmt == "csv":
        with open(args.output, "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(["file", "idx", "label", "text"])
            for f in files:
                idx = extract_idx(f.name)
                label = extract_label(f.name)
                text = f.read_text(encoding="utf-8")
                writer.writerow([f.name, idx, label, text])
    else:
        with open(args.output, "w", encoding="utf-8") as out:
            for f in files:
                idx = extract_idx(f.name)
                label = extract_label(f.name)
                text = f.read_text(encoding="utf-8")
                out.write(f"{'='*80}\n")
                out.write(f"FILE: {f.name}  |  IDX: {idx}  |  LABEL: {label}\n")
                out.write(f"{'='*80}\n")
                out.write(text)
                out.write(f"\n\n")

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()