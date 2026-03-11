#!/usr/bin/env python3
"""
One-time migration script: consolidate variant directories into base directories
using filename suffixes.

Usage:
    python migrate_variant_dirs.py --dry-run      # Preview changes
    python migrate_variant_dirs.py                 # Execute copies
    python migrate_variant_dirs.py --delete-old    # Execute copies, then delete old dirs
"""

import argparse
import os
import shutil
from pathlib import Path

BASE = Path("/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/results/llama2_13b_chat")
VERSIONS = ["balanced_gpt", "nonsense_codeword"]


def add_suffix(filename, suffix):
    """Insert suffix before extension: 'foo.npz' + '_top_align' -> 'foo_top_align.npz'"""
    p = Path(filename)
    return p.stem + suffix + p.suffix


class Migrator:
    def __init__(self, dry_run=False, delete_old=False):
        self.dry_run = dry_run
        self.delete_old = delete_old
        self.copied = 0
        self.skipped = 0
        self.dir_copies = 0
        self.created_dirs = set()
        self.old_dirs_to_delete = set()

    def copy_file(self, src, dst):
        src = Path(src)
        dst = Path(dst)
        if not src.exists():
            self.skipped += 1
            return
        if self.dry_run:
            print(f"  COPY {src}\n    -> {dst}")
            self.copied += 1
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.copied += 1

    def copy_dir(self, src, dst):
        """Copy entire directory tree."""
        src = Path(src)
        dst = Path(dst)
        if not src.exists():
            self.skipped += 1
            return
        if self.dry_run:
            count = sum(1 for _ in src.rglob("*") if _.is_file())
            print(f"  COPY DIR ({count} files) {src}\n    -> {dst}")
            self.dir_copies += count
            return
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        count = sum(1 for _ in dst.rglob("*") if _.is_file())
        self.dir_copies += count

    def move_file(self, src, dst):
        """Move a file within the same base directory (for reorganizing defaults)."""
        src = Path(src)
        dst = Path(dst)
        if not src.exists():
            self.skipped += 1
            return
        if dst.exists():
            # Already moved
            self.skipped += 1
            return
        if self.dry_run:
            print(f"  MOVE {src}\n    -> {dst}")
            self.copied += 1
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        src.unlink()
        self.copied += 1

    def mark_for_deletion(self, d):
        d = Path(d)
        if d.exists():
            self.old_dirs_to_delete.add(d)

    # ── A. concept_activations_1 → concept_activations with _top_align ──

    def migrate_concept_activations_1(self):
        print("\n=== A. concept_activations_1 -> concept_activations (_top_align) ===")
        src_base = BASE / "concept_activations_1"
        dst_base = BASE / "concept_activations"
        if not src_base.exists():
            print("  SKIP: source not found")
            return

        # top_prompt_selections.json
        self.copy_file(
            src_base / "top_prompt_selections.json",
            dst_base / "top_prompt_selections_top_align.json",
        )

        # contrasts/{dim}/
        contrasts_src = src_base / "contrasts"
        if contrasts_src.exists():
            for dim_dir in sorted(contrasts_src.iterdir()):
                if not dim_dir.is_dir():
                    continue
                dim = dim_dir.name
                for fname in ["concept_vector_per_layer.npz", "concept_prompts.json"]:
                    self.copy_file(
                        dim_dir / fname,
                        dst_base / "contrasts" / dim / add_suffix(fname, "_top_align"),
                    )

        # standalone/{dim}/
        standalone_src = src_base / "standalone"
        if standalone_src.exists():
            for dim_dir in sorted(standalone_src.iterdir()):
                if not dim_dir.is_dir():
                    continue
                dim = dim_dir.name
                self.copy_file(
                    dim_dir / "mean_vectors_per_layer.npz",
                    dst_base / "standalone" / dim / "mean_vectors_per_layer_top_align.npz",
                )

        self.mark_for_deletion(src_base)

    # ── B. concept_activations_simple → concept_activations with _simple ──

    def migrate_concept_activations_simple(self):
        print("\n=== B. concept_activations_simple -> concept_activations (_simple) ===")
        src_base = BASE / "concept_activations_simple"
        dst_base = BASE / "concept_activations"
        if not src_base.exists():
            print("  SKIP: source not found")
            return

        standalone_src = src_base / "standalone"
        if standalone_src.exists():
            for dim_dir in sorted(standalone_src.iterdir()):
                if not dim_dir.is_dir():
                    continue
                dim = dim_dir.name
                for fname in [
                    "concept_activations.npz",
                    "mean_vectors_per_layer.npz",
                    "concept_prompts.json",
                    "split_half_stability.json",
                ]:
                    src = dim_dir / fname
                    if src.exists():
                        self.copy_file(
                            src,
                            dst_base / "standalone" / dim / add_suffix(fname, "_simple"),
                        )

        self.mark_for_deletion(src_base)

    # ── C. concept_overlap_1 → concept_overlap with _top_align ──

    def migrate_concept_overlap_1(self):
        print("\n=== C. concept_overlap_1 -> concept_overlap (_top_align) ===")
        src_base = BASE / "concept_overlap_1" / "contrasts"
        dst_base = BASE / "concept_overlap" / "contrasts"
        if not src_base.exists():
            print("  SKIP: source not found")
            return

        # Data files → data/ subfolder with suffix
        for fname in ["overlap_matrix.npz", "overlap_matrix.csv", "layer_profiles.npz"]:
            self.copy_file(
                src_base / fname,
                dst_base / "data" / add_suffix(fname, "_top_align"),
            )

        # Reports → parent with suffix
        for fname in ["concept_overlap_report.html", "concept_overlap_report.md"]:
            self.copy_file(
                src_base / fname,
                dst_base / add_suffix(fname, "_top_align"),
            )

        # Figures directory
        self.copy_dir(
            src_base / "figures",
            dst_base / "figures_top_align",
        )

        # Also move DEFAULT overlap data to data/ subfolder
        print("\n  -- Moving default overlap data to data/ subfolder --")
        for fname in [
            "overlap_matrix.npz",
            "overlap_matrix.csv",
            "layer_profiles.npz",
            "baseline_overlap.csv",
            "sysprompt_baseline_overlap.csv",
        ]:
            self.move_file(
                dst_base / fname,
                dst_base / "data" / fname,
            )

        self.mark_for_deletion(BASE / "concept_overlap_1")

    # ── D. alignment_1 → alignment with _top_align ──

    def migrate_alignment_1(self):
        print("\n=== D. alignment_1 -> alignment (_top_align) ===")
        for version in VERSIONS:
            src_base = BASE / version / "alignment_1"
            dst_base = BASE / version / "alignment"
            if not src_base.exists():
                print(f"  SKIP {version}: source not found")
                continue

            print(f"\n  -- {version} --")
            for turn_dir in sorted(src_base.iterdir()):
                if not turn_dir.is_dir() or not turn_dir.name.startswith("turn_"):
                    continue
                turn = turn_dir.name

                # contrasts/raw/{dim}/alignment.npz
                raw_src = turn_dir / "contrasts" / "raw"
                if raw_src.exists():
                    for dim_dir in sorted(raw_src.iterdir()):
                        if not dim_dir.is_dir():
                            continue
                        self.copy_file(
                            dim_dir / "alignment.npz",
                            dst_base / turn / "contrasts" / "raw" / dim_dir.name / "alignment_top_align.npz",
                        )
                    # summary.json → data/
                    self.copy_file(
                        raw_src / "summary.json",
                        dst_base / turn / "contrasts" / "raw" / "data" / "summary_top_align.json",
                    )

                # standalone/{dim}/alignment.npz
                standalone_src = turn_dir / "standalone"
                if standalone_src.exists():
                    for dim_dir in sorted(standalone_src.iterdir()):
                        if not dim_dir.is_dir():
                            continue
                        self.copy_file(
                            dim_dir / "alignment.npz",
                            dst_base / turn / "standalone" / dim_dir.name / "alignment_top_align.npz",
                        )
                    # summary.json → data/
                    self.copy_file(
                        standalone_src / "summary.json",
                        dst_base / turn / "standalone" / "data" / "summary_top_align.json",
                    )

            self.mark_for_deletion(src_base)

        # Also move DEFAULT alignment summaries to data/
        print("\n  -- Moving default alignment summaries to data/ --")
        for version in VERSIONS:
            dst_base = BASE / version / "alignment"
            if not dst_base.exists():
                continue
            for turn_dir in sorted(dst_base.iterdir()):
                if not turn_dir.is_dir() or not turn_dir.name.startswith("turn_"):
                    continue
                turn = turn_dir.name
                # contrasts/raw/summary.json
                self.move_file(
                    dst_base / turn / "contrasts" / "raw" / "summary.json",
                    dst_base / turn / "contrasts" / "raw" / "data" / "summary.json",
                )
                # contrasts/residual/summary.json
                self.move_file(
                    dst_base / turn / "contrasts" / "residual" / "summary.json",
                    dst_base / turn / "contrasts" / "residual" / "data" / "summary.json",
                )
                # standalone/summary.json
                self.move_file(
                    dst_base / turn / "standalone" / "summary.json",
                    dst_base / turn / "standalone" / "data" / "summary.json",
                )

    # ── E. alignment_simple → alignment with _simple ──

    def migrate_alignment_simple(self):
        print("\n=== E. alignment_simple -> alignment (_simple) ===")
        for version in VERSIONS:
            src_base = BASE / version / "alignment_simple"
            dst_base = BASE / version / "alignment"
            if not src_base.exists():
                print(f"  SKIP {version}: source not found")
                continue

            print(f"\n  -- {version} --")
            for turn_dir in sorted(src_base.iterdir()):
                if not turn_dir.is_dir() or not turn_dir.name.startswith("turn_"):
                    continue
                turn = turn_dir.name

                # standalone/{dim}/alignment.npz
                standalone_src = turn_dir / "standalone"
                if standalone_src.exists():
                    for dim_dir in sorted(standalone_src.iterdir()):
                        if not dim_dir.is_dir():
                            continue
                        self.copy_file(
                            dim_dir / "alignment.npz",
                            dst_base / turn / "standalone" / dim_dir.name / "alignment_simple.npz",
                        )
                    # summary.json → data/
                    self.copy_file(
                        standalone_src / "summary.json",
                        dst_base / turn / "standalone" / "data" / "summary_simple.json",
                    )

            self.mark_for_deletion(src_base)

    # ── F. concept_steering_1 → concept_steering with _top_align ──

    def migrate_concept_steering_1(self):
        print("\n=== F. concept_steering_1 -> concept_steering (_top_align) ===")
        # balanced_gpt only
        src_base = BASE / "balanced_gpt" / "concept_steering_1" / "v1"
        dst_base = BASE / "balanced_gpt" / "concept_steering" / "v1"
        if not src_base.exists():
            print("  SKIP: source not found")
            return

        # Per-dim / per-strategy files
        for dim_dir in sorted(src_base.iterdir()):
            if not dim_dir.is_dir():
                continue
            dim = dim_dir.name
            for strategy_dir in sorted(dim_dir.iterdir()):
                if not strategy_dir.is_dir():
                    continue
                strategy = strategy_dir.name
                for f in sorted(strategy_dir.iterdir()):
                    if f.is_file():
                        self.copy_file(
                            f,
                            dst_base / dim / strategy / add_suffix(f.name, "_top_align"),
                        )

        # Top-level files
        for fname in ["behavioral_summary.csv", "concept_steering_summary.html"]:
            self.copy_file(
                src_base / fname,
                dst_base / add_suffix(fname, "_top_align"),
            )

        self.mark_for_deletion(BASE / "balanced_gpt" / "concept_steering_1")

    # ── G. concept_conversation_1 → concept_conversation with _top_align ──

    def migrate_concept_conversation_1(self):
        print("\n=== G. concept_conversation_1 -> concept_conversation (_top_align) ===")
        # balanced_gpt only
        src_base = BASE / "balanced_gpt" / "concept_conversation_1"
        dst_base = BASE / "balanced_gpt" / "concept_conversation"
        if not src_base.exists():
            print("  SKIP: source not found")
            return

        for turn_dir in sorted(src_base.iterdir()):
            if not turn_dir.is_dir() or not turn_dir.name.startswith("turn_"):
                continue
            turn = turn_dir.name

            # Per-approach data and figures
            for approach in ["approach_a", "approach_c", "approach_d"]:
                approach_src = turn_dir / approach
                if not approach_src.exists():
                    continue

                # Data files → data/ with suffix
                for fname in ["stats.csv", "alignment_scores.npz", "prompt_stats.csv"]:
                    self.copy_file(
                        approach_src / fname,
                        dst_base / turn / approach / "data" / add_suffix(fname, "_top_align"),
                    )

                # Figures directory
                if (approach_src / "figures").exists():
                    self.copy_dir(
                        approach_src / "figures",
                        dst_base / turn / approach / "figures_top_align",
                    )

            # Reports
            for fname in [
                "concept_conversation_report.html",
                "concept_conversation_report.md",
            ]:
                self.copy_file(
                    turn_dir / fname,
                    dst_base / turn / add_suffix(fname, "_top_align"),
                )

            # cross_approach_summary.csv → data/ with suffix
            self.copy_file(
                turn_dir / "cross_approach_summary.csv",
                dst_base / turn / "data" / "cross_approach_summary_top_align.csv",
            )

        # Also move DEFAULT conversation data to data/
        print("\n  -- Moving default conversation data to data/ --")
        for turn_dir in sorted(dst_base.iterdir()):
            if not turn_dir.is_dir() or not turn_dir.name.startswith("turn_"):
                continue
            turn = turn_dir.name

            for approach in ["approach_a", "approach_c", "approach_d"]:
                approach_dir = turn_dir / approach
                if not approach_dir.exists():
                    continue
                for fname in ["stats.csv", "alignment_scores.npz", "prompt_stats.csv"]:
                    self.move_file(
                        approach_dir / fname,
                        approach_dir / "data" / fname,
                    )

            # cross_approach_summary.csv
            self.move_file(
                turn_dir / "cross_approach_summary.csv",
                turn_dir / "data" / "cross_approach_summary.csv",
            )

        self.mark_for_deletion(src_base)

    # ── H. comparisons/alignment_1 → comparisons/alignment with _top_align ──

    def migrate_comparisons_alignment_1(self):
        print("\n=== H. comparisons/alignment_1 -> comparisons/alignment (_top_align) ===")
        src_base = BASE / "comparisons" / "alignment_1"
        dst_base = BASE / "comparisons" / "alignment"
        if not src_base.exists():
            print("  SKIP: source not found")
            return

        for turn_dir in sorted(src_base.iterdir()):
            if not turn_dir.is_dir() or not turn_dir.name.startswith("turn_"):
                continue
            turn = turn_dir.name

            for subdir_name in ["raw", "standalone"]:
                subdir = turn_dir / subdir_name
                if not subdir.exists():
                    continue

                # Reports
                for f in sorted(subdir.iterdir()):
                    if f.is_file():
                        self.copy_file(
                            f,
                            dst_base / turn / subdir_name / add_suffix(f.name, "_top_align"),
                        )

                # Figures
                if (subdir / "figures").exists():
                    self.copy_dir(
                        subdir / "figures",
                        dst_base / turn / subdir_name / "figures_top_align",
                    )

        self.mark_for_deletion(src_base)

    # ── I. comparisons/alignment_simple → comparisons/alignment with _simple ──

    def migrate_comparisons_alignment_simple(self):
        print("\n=== I. comparisons/alignment_simple -> comparisons/alignment (_simple) ===")
        src_base = BASE / "comparisons" / "alignment_simple"
        dst_base = BASE / "comparisons" / "alignment"
        if not src_base.exists():
            print("  SKIP: source not found")
            return

        for turn_dir in sorted(src_base.iterdir()):
            if not turn_dir.is_dir() or not turn_dir.name.startswith("turn_"):
                continue
            turn = turn_dir.name

            for subdir_name in ["raw", "standalone"]:
                subdir = turn_dir / subdir_name
                if not subdir.exists():
                    continue

                # Reports
                for f in sorted(subdir.iterdir()):
                    if f.is_file():
                        self.copy_file(
                            f,
                            dst_base / turn / subdir_name / add_suffix(f.name, "_simple"),
                        )

                # Figures
                if (subdir / "figures").exists():
                    self.copy_dir(
                        subdir / "figures",
                        dst_base / turn / subdir_name / "figures_simple",
                    )

        self.mark_for_deletion(src_base)

    def run_all(self):
        self.migrate_concept_activations_1()
        self.migrate_concept_activations_simple()
        self.migrate_concept_overlap_1()
        self.migrate_alignment_1()
        self.migrate_alignment_simple()
        self.migrate_concept_steering_1()
        self.migrate_concept_conversation_1()
        self.migrate_comparisons_alignment_1()
        self.migrate_comparisons_alignment_simple()

        print("\n" + "=" * 60)
        print(f"Files copied:      {self.copied}")
        print(f"Dir files copied:  {self.dir_copies}")
        print(f"Files skipped:     {self.skipped}")
        print(f"Old dirs to delete: {len(self.old_dirs_to_delete)}")

        if self.old_dirs_to_delete:
            print("\nDirectories marked for deletion:")
            for d in sorted(self.old_dirs_to_delete):
                print(f"  {d}")

        if self.delete_old and not self.dry_run:
            print("\nDeleting old variant directories...")
            for d in sorted(self.old_dirs_to_delete):
                if d.exists():
                    shutil.rmtree(d)
                    print(f"  DELETED {d}")
            print("Done.")
        elif self.dry_run:
            print("\n[DRY RUN] No files were actually copied or deleted.")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate variant directories into base directories with filename suffixes."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them.",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete old variant directories after copying (ignored with --dry-run).",
    )
    args = parser.parse_args()

    if args.delete_old and args.dry_run:
        print("NOTE: --delete-old is ignored with --dry-run\n")

    migrator = Migrator(dry_run=args.dry_run, delete_old=args.delete_old)
    migrator.run_all()


if __name__ == "__main__":
    main()
