#!/usr/bin/env python3
"""
Compute lexical distinctiveness measures for Exp 3 concept prompt pairs
and correlate with Exp 2 probe alignment.
"""

import json
import os
import re
import csv
from pathlib import Path
from scipy import stats
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path("/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat")
CONTRASTS = BASE / "data" / "concept_activations" / "contrasts"
DIM_TABLE  = BASE / "results" / "concept_probe_alignment" / "summaries" / "dimension_table.csv"

# ── stopwords ──────────────────────────────────────────────────────────────
STOPWORDS = {
    "a", "the", "an", "is", "are", "to", "of", "in", "that", "this",
    "it", "for", "with", "on", "at", "by", "from", "as", "or", "and",
    "but", "not", "be", "was", "were", "been", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "shall",
}

# ── entity-revealing words ─────────────────────────────────────────────────
HUMAN_ENTITY_WORDS = {"human", "person", "people", "man", "woman"}
AI_ENTITY_WORDS    = {"ai", "artificial", "machine", "robot", "system",
                      "computer", "algorithm", "bot", "chatbot"}

# ── helpers ────────────────────────────────────────────────────────────────
def tokenize(text):
    """Lowercase, split on non-alpha, drop stopwords."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def content_words(text):
    """Return set of content words."""
    return set(tokenize(text))

def words_in_prompt(text):
    """Return set of all lowercased words (for entity checks)."""
    return set(re.findall(r"[a-z]+", text.lower()))

# ── load alignment stats ──────────────────────────────────────────────────
alignment = {}  # dim_id -> observed_projection  (control_probe, all_layers)
with open(DIM_TABLE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["probe_type"] == "control_probe" and row["layer_range"] == "all_layers":
            alignment[int(row["dim_id"])] = float(row["observed_projection"])

# ── dim 16 excluded per project convention ─────────────────────────────────
EXCLUDED = {16}

# ── iterate over contrasts ─────────────────────────────────────────────────
dims = sorted(
    [d for d in CONTRASTS.iterdir() if d.is_dir()],
    key=lambda p: int(p.name.split("_")[0]),
)

results = []
for dim_dir in dims:
    dim_id = int(dim_dir.name.split("_")[0])
    if dim_id in EXCLUDED:
        continue

    prompts_file = dim_dir / "concept_prompts.json"
    if not prompts_file.exists():
        continue

    with open(prompts_file) as f:
        prompts = json.load(f)

    human_prompts = [p for p in prompts if p["label"] == 1]
    ai_prompts    = [p for p in prompts if p["label"] == 0]

    # ── (1) Unique content words & Jaccard ────────────────────────────────
    human_words = set()
    for p in human_prompts:
        human_words |= content_words(p["prompt"])

    ai_words = set()
    for p in ai_prompts:
        ai_words |= content_words(p["prompt"])

    intersection = human_words & ai_words
    union        = human_words | ai_words
    jaccard      = len(intersection) / len(union) if union else 0.0

    # ── (2) Entity word contamination ─────────────────────────────────────
    n_human_entity = sum(
        1 for p in human_prompts
        if words_in_prompt(p["prompt"]) & HUMAN_ENTITY_WORDS
    )
    n_ai_entity = sum(
        1 for p in ai_prompts
        if words_in_prompt(p["prompt"]) & AI_ENTITY_WORDS
    )
    pct_human_entity = n_human_entity / len(human_prompts) if human_prompts else 0.0
    pct_ai_entity    = n_ai_entity / len(ai_prompts) if ai_prompts else 0.0

    # ── (3) alignment projection ──────────────────────────────────────────
    proj = alignment.get(dim_id, float("nan"))

    # ── load dim_name / category from the CSV ─────────────────────────────
    # Re-read quickly (small file)
    dim_name = dim_dir.name
    category = ""
    with open(DIM_TABLE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["dim_id"]) == dim_id:
                dim_name = row["dim_name"]
                category = row["category"]
                break

    results.append({
        "dim_id":              dim_id,
        "dim_name":            dim_name,
        "category":            category,
        "n_human_prompts":     len(human_prompts),
        "n_ai_prompts":        len(ai_prompts),
        "n_human_unique":      len(human_words),
        "n_ai_unique":         len(ai_words),
        "jaccard":             jaccard,
        "lex_distinct":        1 - jaccard,
        "pct_human_entity":    pct_human_entity,
        "pct_ai_entity":       pct_ai_entity,
        "alignment_projection": proj,
    })

# ── print per-dimension table ─────────────────────────────────────────────
print("=" * 160)
print(f"{'dim':>4s}  {'name':<22s}  {'category':<22s}  {'nH':>4s} {'nA':>4s}  "
      f"{'H_uniq':>6s} {'A_uniq':>6s}  {'Jaccard':>7s} {'1-Jacc':>7s}  "
      f"{'%H_ent':>7s} {'%A_ent':>7s}  {'align_proj':>10s}")
print("-" * 160)
for r in results:
    print(f"{r['dim_id']:4d}  {r['dim_name']:<22s}  {r['category']:<22s}  "
          f"{r['n_human_prompts']:4d} {r['n_ai_prompts']:4d}  "
          f"{r['n_human_unique']:6d} {r['n_ai_unique']:6d}  "
          f"{r['jaccard']:7.4f} {r['lex_distinct']:7.4f}  "
          f"{r['pct_human_entity']:7.3f} {r['pct_ai_entity']:7.3f}  "
          f"{r['alignment_projection']:10.6f}")
print("=" * 160)

# ── correlations ──────────────────────────────────────────────────────────
lex_dist  = np.array([r["lex_distinct"] for r in results])
pct_h_ent = np.array([r["pct_human_entity"] for r in results])
pct_a_ent = np.array([r["pct_ai_entity"] for r in results])
avg_ent   = (pct_h_ent + pct_a_ent) / 2
align     = np.array([r["alignment_projection"] for r in results])

print("\n── Spearman correlations (N = {}) ──".format(len(results)))

rho, p = stats.spearmanr(lex_dist, np.abs(align))
print(f"  Lexical distinctiveness (1-Jaccard) vs |alignment|:   rho = {rho:+.4f},  p = {p:.4f}")

rho2, p2 = stats.spearmanr(lex_dist, align)
print(f"  Lexical distinctiveness (1-Jaccard) vs alignment:     rho = {rho2:+.4f},  p = {p2:.4f}")

rho3, p3 = stats.spearmanr(pct_h_ent, np.abs(align))
print(f"  % human entity words vs |alignment|:                  rho = {rho3:+.4f},  p = {p3:.4f}")

rho4, p4 = stats.spearmanr(pct_a_ent, np.abs(align))
print(f"  % AI entity words vs |alignment|:                     rho = {rho4:+.4f},  p = {p4:.4f}")

rho5, p5 = stats.spearmanr(avg_ent, np.abs(align))
print(f"  Avg entity contamination vs |alignment|:              rho = {rho5:+.4f},  p = {p5:.4f}")

rho6, p6 = stats.spearmanr(avg_ent, align)
print(f"  Avg entity contamination vs alignment:                rho = {rho6:+.4f},  p = {p6:.4f}")

# ── also save CSV ─────────────────────────────────────────────────────────
out_csv = BASE / "results" / "concept_probe_alignment" / "summaries" / "lexical_distinctiveness.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dim_id", "dim_name", "category",
        "n_human_prompts", "n_ai_prompts",
        "n_human_unique_words", "n_ai_unique_words",
        "jaccard", "lexical_distinctiveness",
        "pct_human_entity_words", "pct_ai_entity_words",
        "alignment_projection",
    ])
    writer.writeheader()
    for r in results:
        writer.writerow({
            "dim_id":                  r["dim_id"],
            "dim_name":                r["dim_name"],
            "category":                r["category"],
            "n_human_prompts":         r["n_human_prompts"],
            "n_ai_prompts":            r["n_ai_prompts"],
            "n_human_unique_words":    r["n_human_unique"],
            "n_ai_unique_words":       r["n_ai_unique"],
            "jaccard":                 f"{r['jaccard']:.6f}",
            "lexical_distinctiveness": f"{r['lex_distinct']:.6f}",
            "pct_human_entity_words":  f"{r['pct_human_entity']:.4f}",
            "pct_ai_entity_words":     f"{r['pct_ai_entity']:.4f}",
            "alignment_projection":    f"{r['alignment_projection']:.6f}",
        })
print(f"\nSaved to {out_csv}")
