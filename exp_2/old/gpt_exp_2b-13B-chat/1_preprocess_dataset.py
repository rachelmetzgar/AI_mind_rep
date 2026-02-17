"""
Preprocess Experiment 1 conversation CSVs into .txt files
for Experiment 2b activation extraction pipeline.

Each trial (5 exchange pairs) → one conversation file in
    ### User: / ### Assistant:
format, named:
    conversation_{idx:04d}_partner_{label}.txt

where label ∈ {human, ai}.

The "User" is the SUBJECT LLM (whose activations we will extract).
The "Assistant" is the PARTNER LLM.

We use transcript_sub / transcript_llm columns (clean utterance text)
to avoid leaking the system prompt or partner name prefixes.

Usage:
    python preprocess_exp1_to_2b.py \
        --input_dir /path/to/csv_files \
        --output_dir /path/to/output \
        [--subjects s001 s002 ...] \
        [--verify]

Rachel C. Metzgar · February 2026
"""

import os
import csv
import argparse
import json
from collections import defaultdict


# ── Label mapping ────────────────────────────────────────────────────────────
PARTNER_TYPE_TO_LABEL = {
    "an AI": "ai",
    "a Human": "human",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Experiment 1 CSVs to Exp 2b .txt conversation files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing subject CSV files (e.g., s001.csv, s002.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/exp2b_conversations",
        help="Output directory for formatted .txt files.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Specific subject IDs to process (e.g., s001 s002). Default: all CSVs.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print summary statistics and sample conversations for verification.",
    )
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="Save a metadata JSON alongside each conversation file.",
    )
    return parser.parse_args()


def load_csv(filepath):
    """Load a subject CSV, handling potential BOM and Windows line endings."""
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def group_by_trial(rows):
    """
    Group rows by (subject, trial) and sort by pair_index within each trial.
    Returns dict: (subject, trial) → list of rows sorted by pair_index.
    """
    trials = defaultdict(list)
    for row in rows:
        key = (row["subject"], int(row["trial"]))
        trials[key].append(row)

    # Sort each trial's rows by pair_index
    for key in trials:
        trials[key].sort(key=lambda r: int(r["pair_index"]))

    return trials


def format_conversation(trial_rows):
    """
    Convert a trial's rows into ### User: / ### Assistant: format.
    
    User = subject LLM (transcript_sub)
    Assistant = partner LLM (transcript_llm)
    
    Returns: (formatted_text, label, metadata_dict)
    """
    lines = []
    for row in trial_rows:
        sub_text = row["transcript_sub"].strip()
        llm_text = row["transcript_llm"].strip()

        if sub_text:
            lines.append(f"### User: {sub_text}")
        if llm_text:
            lines.append(f"### Assistant: {llm_text}")

    formatted_text = "\n".join(lines)

    # Extract label from partner_type
    partner_type = trial_rows[0]["partner_type"].strip()
    label = PARTNER_TYPE_TO_LABEL.get(partner_type)
    if label is None:
        raise ValueError(
            f"Unknown partner_type '{partner_type}' in subject={trial_rows[0]['subject']}, "
            f"trial={trial_rows[0]['trial']}"
        )

    # Metadata for reference
    metadata = {
        "subject": trial_rows[0]["subject"],
        "run": int(trial_rows[0]["run"]),
        "trial": int(trial_rows[0]["trial"]),
        "partner_name": trial_rows[0]["partner_name"],
        "partner_type": partner_type,
        "label": label,
        "topic": trial_rows[0]["topic"],
        "n_exchanges": len(trial_rows),
        "quality": trial_rows[-1].get("Quality", ""),
        "connectedness": trial_rows[-1].get("Connectedness", ""),
    }

    return formatted_text, label, metadata


def scrub_partner_names(text, extra_names=None):
    """
    Replace partner names that could leak the label with 'user'.
    Case-insensitive replacement, preserves surrounding whitespace/punctuation.
    """
    names_to_scrub = ["Sam", "Casey", "ChatGPT", "Gemini"]
    if extra_names:
        names_to_scrub.extend(extra_names)

    import re
    for name in names_to_scrub:
        # Word-boundary match, case-insensitive
        text = re.sub(rf'\b{re.escape(name)}\b', 'partner', text, flags=re.IGNORECASE)
    return text


def check_for_identity_leakage(text, partner_names=None):
    """
    Check if conversation text contains explicit identity mentions
    that might leak the label.
    
    Returns list of detected leakage strings (empty if clean).
    """
    leakage_terms = [
        "I am an AI",
        "I'm an AI",
        "as an AI",
        "I am a human",
        "I'm a human",
        "as a human",
        "I'm not a real person",
        "I don't have feelings",
        "I'm a language model",
        "I'm a chatbot",
    ]
    if partner_names:
        # Don't flag these — partner names appear naturally in conversation
        # since subjects were told them. But note them for reference.
        pass

    found = []
    text_lower = text.lower()
    for term in leakage_terms:
        if term.lower() in text_lower:
            found.append(term)
    return found


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Find CSV files
    if args.subjects:
        csv_files = [
            os.path.join(args.input_dir, f"{s}.csv") for s in args.subjects
        ]
    else:
        import re as _re
        csv_files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".csv") and _re.match(r'^s\d+\.csv$', f)
        ])

    global_idx = 0
    all_metadata = []
    label_counts = defaultdict(int)
    leakage_log = []

    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping.")
            continue

        subject_id = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Processing {subject_id}...")

        rows = load_csv(csv_path)
        trials = group_by_trial(rows)

        for (subj, trial_num), trial_rows in sorted(trials.items()):
            formatted_text, label, metadata = format_conversation(trial_rows)

            # Scrub partner names (Sam, Casey, ChatGPT, Gemini → user)
            formatted_text = scrub_partner_names(formatted_text)

            # Check for identity leakage
            leaks = check_for_identity_leakage(formatted_text)
            if leaks:
                leakage_log.append({
                    "subject": subj,
                    "trial": trial_num,
                    "label": label,
                    "leaked_terms": leaks,
                })

            # Save conversation file
            fname = f"conversation_{global_idx:04d}_partner_{label}.txt"
            fpath = os.path.join(args.output_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(formatted_text)

            metadata["file"] = fname
            metadata["global_idx"] = global_idx
            all_metadata.append(metadata)
            label_counts[label] += 1
            global_idx += 1

    # Save master metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    # Save leakage log
    if leakage_log:
        leak_path = os.path.join(args.output_dir, "leakage_log.json")
        with open(leak_path, "w", encoding="utf-8") as f:
            json.dump(leakage_log, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Preprocessing complete.")
    print(f"  Total conversations: {global_idx}")
    print(f"  Label distribution:  {dict(label_counts)}")
    print(f"  Output directory:    {args.output_dir}")
    print(f"  Metadata saved to:   {meta_path}")
    if leakage_log:
        print(f"  ⚠ Identity leakage detected in {len(leakage_log)} conversations")
        print(f"    See: {leak_path}")
    else:
        print(f"  ✓ No identity leakage detected")
    print(f"{'='*60}")

    # ── Verification ─────────────────────────────────────────────────────────
    if args.verify and all_metadata:
        print("\n── Sample output (first conversation) ──")
        sample_file = os.path.join(args.output_dir, all_metadata[0]["file"])
        with open(sample_file, "r") as f:
            sample = f.read()
        print(f"File: {all_metadata[0]['file']}")
        print(f"Label: {all_metadata[0]['label']}")
        print(f"Topic: {all_metadata[0]['topic']}")
        print(f"Partner: {all_metadata[0]['partner_name']}")
        print("-" * 40)
        # Print first 500 chars
        print(sample[:500])
        if len(sample) > 500:
            print(f"... ({len(sample)} chars total)")

        # Per-subject breakdown
        print("\n── Per-subject breakdown ──")
        subj_stats = defaultdict(lambda: defaultdict(int))
        for m in all_metadata:
            subj_stats[m["subject"]][m["label"]] += 1
        for subj in sorted(subj_stats):
            counts = subj_stats[subj]
            print(f"  {subj}: {dict(counts)}")


if __name__ == "__main__":
    main()