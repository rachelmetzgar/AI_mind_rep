"""
Utility plotting functions for behavioral and encoding analyses.

Reusable functions:
    - plot_violin_basic: condition × condition violin plots (no sig stars)
    - plot_main_effect_lines: within-subject line plot for paired conditions (with optional sig stars)

Color palette matches Rachel C. Metzgar's Empath ToM convention.

Author: Rachel C. Metzgar
Date: 2025-10-31
"""

from __future__ import annotations
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Shared color palette
# ------------------------------------------------------------
DEFAULT_PALETTE = {
    "hum_nonsocial": "skyblue",
    "hum_social": "steelblue",
    "bot_nonsocial": "sandybrown",
    "bot_social": "peru",
    "Hum": "steelblue",
    "Bot": "sandybrown",
}


# ------------------------------------------------------------
# Helper: significance star formatter
# ------------------------------------------------------------
def p_to_star(p: float | None) -> str:
    """Convert p-value to standard star notation."""
    if p is None:
        return ""
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


# ------------------------------------------------------------
#  Simple violin plot (no significance stars)
# ------------------------------------------------------------
def plot_violin_basic(
    df_long: pd.DataFrame,
    y_col: str,
    x_col: str,
    out_path: str,
    title: str = "",
    palette: dict[str, str] | None = None,
    figsize: tuple[int, int] = (8, 6),
):
    """Draw clean violin plots with box overlays, using fixed color palette."""
    if palette is None:
        palette = DEFAULT_PALETTE

    plt.figure(figsize=figsize)
    sns.violinplot(
        data=df_long,
        x=x_col,
        y=y_col,
        palette=palette,
        inner="box",
        cut=0
    )
    plt.title(title)
    plt.ylabel(y_col.replace("_", " ").title())
    plt.xlabel(x_col.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ------------------------------------------------------------
#  Within-subject line plot for main effect (e.g., Human vs Bot)
# ------------------------------------------------------------
def main_effect_violin_lines(
    df_summary: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    y_col: str,
    out_path: str,
    title: str = "",
    palette: dict[str, str] | None = None,
    p_val: float | None = None,
):
    """Plot within-subject lines between two paired conditions and show significance stars.
       (Bot shown first / left)."""
    if palette is None:
        palette = DEFAULT_PALETTE

    # Melt into long form
    long_df = df_summary.melt(
        id_vars="Subject",
        value_vars=[cond_a, cond_b],
        var_name="Condition",
        value_name=y_col
    )

    # --- Enforce consistent order (Bot left, Hum right) ---
    cond_order = [cond_b, cond_a]  # Flip order so Bot is first (left)

    plt.figure(figsize=(6, 6))
    ax = sns.violinplot(
        data=long_df,
        x="Condition",
        y=y_col,
        order=cond_order,
        palette={cond_a: palette.get(cond_a, "steelblue"),
                 cond_b: palette.get(cond_b, "sandybrown")},
        inner="box",
        cut=0
    )

    # Within-subject connecting lines (respect order)
    for sub, g in long_df.groupby("Subject"):
        if g[y_col].isnull().any():
            continue
        # align values to cond_order (Bot → Hum)
        y_vals = [g.loc[g["Condition"] == c, y_col].values[0] for c in cond_order]
        plt.plot([0, 1], y_vals, color="gray", alpha=0.4, lw=1)

    # Add top bracket and significance star text
    y_max = long_df[y_col].max()
    y_min = long_df[y_col].min()
    y_offset = (y_max - y_min) * 0.08

    # Draw bracket
    ax.plot(
        [0, 0, 1, 1],
        [y_max + y_offset, y_max + 2 * y_offset,
         y_max + 2 * y_offset, y_max + y_offset],
        lw=1.2, c="k"
    )

    # Add star or 'n.s.'
    ax.text(0.5, y_max + 2.4 * y_offset, p_to_star(p_val),
            ha="center", va="bottom", fontsize=14)

    # --- Increase top margin so text isn't clipped ---
    ax.set_ylim(y_min, y_max + 4 * y_offset)   # Extra space above plot

    plt.title(title)
    plt.ylabel(y_col.replace("_", " ").title())
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    
# ------------------------------------------------------------
#  Bar plot with within-subject lines + significance stars
# ------------------------------------------------------------
def barplot_with_lines(
    df_long: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str,
    title: str = "",
    palette: dict[str, str] | None = None,
    p_val: float | None = None,
    figsize: tuple[int, int] = (6, 5),
):
    """Create a bar plot (mean ± SD) with gray subject-level lines connecting paired conditions and a significance bracket/star annotation."""
    if palette is None:
        palette = DEFAULT_PALETTE

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_long,
        x=x_col,
        y=y_col,
        ci="sd",
        palette=palette,
        capsize=0.15,
        errwidth=1.2,
        edgecolor="black"
    )

    # Draw gray within-subject lines
    if "Subject" in df_long.columns:
        for sub, g in df_long.groupby("Subject"):
            if g[y_col].isnull().any():
                continue
            y_vals = []
            for cond in df_long[x_col].unique():
                vals = g.loc[g[x_col] == cond, y_col].values
                if len(vals) > 0:
                    y_vals.append(vals[0])
            x_vals = list(range(len(y_vals)))
            plt.plot(x_vals, y_vals, color="gray", alpha=0.4, lw=1)

    # --- Add significance bracket + stars ---
    if p_val is not None:
        y_max = df_long[y_col].max()
        y_min = df_long[y_col].min()
        y_offset = (y_max - y_min) * 0.08

        # Draw bracket above bars
        ax.plot(
            [0, 0, 1, 1],
            [y_max + y_offset, y_max + 2 * y_offset,
             y_max + 2 * y_offset, y_max + y_offset],
            lw=1.2, c="k"
        )

        # Add significance label
        ax.text(
            0.5, y_max + 2.4 * y_offset,
            p_to_star(p_val),
            ha="center", va="bottom", fontsize=14
        )

        # Add vertical padding so label isn't clipped
        ax.set_ylim(y_min, y_max + 4 * y_offset)

    plt.title(title)
    plt.ylabel(y_col.replace("_", " ").title())
    plt.xlabel(x_col.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
