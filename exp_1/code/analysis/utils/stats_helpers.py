"""
Script name: stats_helpers.py
Purpose: Statistical helper functions for behavioral analyses.
    - Clean paired arrays for tests.
    - Compute effect sizes for paired samples.
    - Run paired t-tests with formatted output (console and file).
Author: Rachel C. Metzgar
Date: 2025-09-29
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, TextIO
from scipy.stats import ttest_rel

__all__ = ["paired_clean", "cohens_dz", "paired_ttest_report"]

# --------------------------
# Paired-sample helpers
# --------------------------


def paired_clean(a: List[float], b: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired arrays with NaNs removed."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    return a[mask], b[mask]


def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's dz effect size for paired samples."""
    diff = a - b
    if diff.size < 2 or np.nanstd(diff, ddof=1) == 0:
        return np.nan
    return np.nanmean(diff) / np.nanstd(diff, ddof=1)


def paired_ttest_report(
    a: List[float],
    b: List[float],
    label_a: str,
    label_b: str,
    measure_name: str,
    out_file: Optional[TextIO] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Run paired t-test, print formatted results, and optionally write to file.

    Returns (t, p, dz) or None if no valid pairs.
    """
    a, b = paired_clean(a, b)
    n = a.size
    if n == 0:
        msg = f"{measure_name}: no usable pairs."
        print(msg)
        if out_file:
            out_file.write(msg + "\n")
        return None

    t, p = ttest_rel(a, b, nan_policy="omit")
    dz = cohens_dz(a, b)

    msg = (f"{measure_name}: N={n}, "
           f"{label_a}={a.mean():.2f}, {label_b}={b.mean():.2f}, "
           f"t={t:.3f}, p={p:.4f}, dz={dz:.3f}")
    print(msg)
    if out_file:
        out_file.write(msg + "\n")

    return t, p, dz

# ============================================================
#                    STATISTICAL ANALYSES
# ============================================================

def run_simple_effects(aov_df, metric):
    """Run simple effects tests: Condition effect at each level of Sociality."""
    from scipy import stats
    
    simple_effects = {}
    
    for sociality in ['social', 'nonsocial']:
        soc_df = aov_df[aov_df['social_type'] == sociality]
        pivot = soc_df.pivot(index='subject', columns='condition', values=metric)
        
        if 'hum' not in pivot.columns or 'bot' not in pivot.columns:
            simple_effects[sociality] = {'error': 'Missing condition data'}
            continue
        
        pivot = pivot.dropna()
        
        if len(pivot) < 3:
            simple_effects[sociality] = {'error': f'Insufficient subjects: {len(pivot)}'}
            continue
        
        hum_vals = pivot['hum']
        bot_vals = pivot['bot']
        
        t_stat, p_val = stats.ttest_rel(hum_vals, bot_vals)
        
        simple_effects[sociality] = {
            'hum_mean': hum_vals.mean(),
            'hum_sem': hum_vals.sem(),
            'bot_mean': bot_vals.mean(),
            'bot_sem': bot_vals.sem(),
            'diff_mean': (hum_vals - bot_vals).mean(),
            'diff_sem': (hum_vals - bot_vals).sem(),
            't_stat': t_stat,
            'p_val': p_val,
            'n': len(pivot),
            'direction': 'Human > AI' if hum_vals.mean() > bot_vals.mean() else 'AI > Human',
            'sig': '*' if p_val < .05 else ''
        }
    
    return simple_effects


def run_within_experiment_anova(df, metric, experiment_name):
    """
    Run 2×2 RM-ANOVA (Condition × Sociality) for one experiment.
    """
    exp_df = df[df['experiment'] == experiment_name].copy()
    
    if exp_df.empty:
        return None
    
    aov_df = exp_df.copy()
    n_subjects = aov_df['subject'].nunique()
    n_cells = aov_df.groupby(['condition', 'social_type']).size()
    
    if n_subjects < 3:
        return {'error': f'Insufficient subjects: {n_subjects}'}
    
    if len(n_cells) < 4:
        return {'error': f'Missing cells: {n_cells.to_dict()}'}
    
    if aov_df[metric].isna().all():
        return {'error': f'All values are NaN for {metric}'}
    
    aov_df = aov_df.dropna(subset=[metric])
    
    complete_subjects = aov_df.groupby('subject').filter(lambda x: len(x) == 4)['subject'].unique()
    aov_df = aov_df[aov_df['subject'].isin(complete_subjects)]
    n_subjects = aov_df['subject'].nunique()
    
    if n_subjects < 3:
        return {'error': f'Insufficient complete subjects after filtering: {n_subjects}'}
    
    results = {
        'experiment': experiment_name,
        'metric': metric,
        'n_subjects': n_subjects,
    }
    
    try:
        wide = aov_df.pivot_table(
            index='subject',
            columns=['condition', 'social_type'],
            values=metric
        )
        
        Y_hs = wide[('hum', 'social')].values
        Y_hn = wide[('hum', 'nonsocial')].values
        Y_bs = wide[('bot', 'social')].values
        Y_bn = wide[('bot', 'nonsocial')].values
        
        n = len(Y_hs)
        
        GM = (Y_hs + Y_hn + Y_bs + Y_bn).mean() / 4
        
        M_hum = (Y_hs + Y_hn).mean() / 2
        M_bot = (Y_bs + Y_bn).mean() / 2
        M_social = (Y_hs + Y_bs).mean() / 2
        M_nonsocial = (Y_hn + Y_bn).mean() / 2
        
        M_subj = (Y_hs + Y_hn + Y_bs + Y_bn) / 4
        
        # Main effect of Condition
        SS_cond = 2 * n * ((M_hum - GM)**2 + (M_bot - GM)**2)
        df_cond = 1
        
        subj_hum = (Y_hs + Y_hn) / 2
        subj_bot = (Y_bs + Y_bn) / 2
        SS_cond_subj = 2 * np.sum((subj_hum - M_subj - M_hum + GM)**2 + (subj_bot - M_subj - M_bot + GM)**2)
        df_cond_subj = n - 1
        
        MS_cond = SS_cond / df_cond
        MS_cond_subj = SS_cond_subj / df_cond_subj
        F_cond = MS_cond / MS_cond_subj
        p_cond = 1 - f_dist.cdf(F_cond, df_cond, df_cond_subj)
        
        results['condition_F'] = F_cond
        results['condition_p'] = p_cond
        results['condition_df'] = (df_cond, df_cond_subj)
        
        # Main effect of Sociality
        SS_social = 2 * n * ((M_social - GM)**2 + (M_nonsocial - GM)**2)
        df_social = 1
        
        subj_soc = (Y_hs + Y_bs) / 2
        subj_nonsoc = (Y_hn + Y_bn) / 2
        SS_social_subj = 2 * np.sum((subj_soc - M_subj - M_social + GM)**2 + (subj_nonsoc - M_subj - M_nonsocial + GM)**2)
        df_social_subj = n - 1
        
        MS_social = SS_social / df_social
        MS_social_subj = SS_social_subj / df_social_subj
        F_social = MS_social / MS_social_subj
        p_social = 1 - f_dist.cdf(F_social, df_social, df_social_subj)
        
        results['sociality_F'] = F_social
        results['sociality_p'] = p_social
        results['sociality_df'] = (df_social, df_social_subj)
        
        # Condition × Sociality Interaction
        M_hs = Y_hs.mean()
        M_hn = Y_hn.mean()
        M_bs = Y_bs.mean()
        M_bn = Y_bn.mean()
        
        SS_inter = n * ((M_hs - M_hum - M_social + GM)**2 + 
                        (M_hn - M_hum - M_nonsocial + GM)**2 +
                        (M_bs - M_bot - M_social + GM)**2 + 
                        (M_bn - M_bot - M_nonsocial + GM)**2)
        df_inter = 1
        
        SS_inter_subj = np.sum(
            (Y_hs - subj_hum - subj_soc + M_subj - M_hs + M_hum + M_social - GM)**2 +
            (Y_hn - subj_hum - subj_nonsoc + M_subj - M_hn + M_hum + M_nonsocial - GM)**2 +
            (Y_bs - subj_bot - subj_soc + M_subj - M_bs + M_bot + M_social - GM)**2 +
            (Y_bn - subj_bot - subj_nonsoc + M_subj - M_bn + M_bot + M_nonsocial - GM)**2
        )
        df_inter_subj = n - 1
        
        MS_inter = SS_inter / df_inter
        MS_inter_subj = SS_inter_subj / df_inter_subj
        F_inter = MS_inter / MS_inter_subj
        p_inter = 1 - f_dist.cdf(F_inter, df_inter, df_inter_subj)
        
        results['interaction_F'] = F_inter
        results['interaction_p'] = p_inter
        results['interaction_df'] = (df_inter, df_inter_subj)
        
        # Descriptive statistics
        cond_means = aov_df.groupby('condition')[metric].agg(['mean', 'sem'])
        results['hum_mean'] = cond_means.loc['hum', 'mean'] if 'hum' in cond_means.index else np.nan
        results['hum_sem'] = cond_means.loc['hum', 'sem'] if 'hum' in cond_means.index else np.nan
        results['bot_mean'] = cond_means.loc['bot', 'mean'] if 'bot' in cond_means.index else np.nan
        results['bot_sem'] = cond_means.loc['bot', 'sem'] if 'bot' in cond_means.index else np.nan
        
        social_means = aov_df.groupby('social_type')[metric].agg(['mean', 'sem'])
        results['social_mean'] = social_means.loc['social', 'mean'] if 'social' in social_means.index else np.nan
        results['social_sem'] = social_means.loc['social', 'sem'] if 'social' in social_means.index else np.nan
        results['nonsocial_mean'] = social_means.loc['nonsocial', 'mean'] if 'nonsocial' in social_means.index else np.nan
        results['nonsocial_sem'] = social_means.loc['nonsocial', 'sem'] if 'nonsocial' in social_means.index else np.nan
        
        cell_means = aov_df.groupby(['condition', 'social_type'])[metric].agg(['mean', 'sem'])
        results['cell_means'] = cell_means.to_dict()
        
        results['anova_table'] = pd.DataFrame({
            'Source': ['condition', 'social_type', 'condition:social_type'],
            'SS': [SS_cond, SS_social, SS_inter],
            'DF1': [df_cond, df_social, df_inter],
            'DF2': [df_cond_subj, df_social_subj, df_inter_subj],
            'MS': [MS_cond, MS_social, MS_inter],
            'MS_error': [MS_cond_subj, MS_social_subj, MS_inter_subj],
            'F': [F_cond, F_social, F_inter],
            'p': [p_cond, p_social, p_inter]
        })
        
        if results.get('interaction_p', 1.0) < 0.05:
            results['simple_effects'] = run_simple_effects(aov_df, metric)
        
    except Exception as e:
        return {'error': str(e)}
    
    return results


def run_cross_experiment_analysis(combined_subject_df, metric):
    """Test whether the Condition effect differs between experiments."""
    df = combined_subject_df.copy()
    
    wide_df = df.pivot_table(
        index=['experiment', 'subject'],
        columns='condition',
        values=metric,
        aggfunc='mean'
    ).reset_index()
    
    if 'hum' not in wide_df.columns or 'bot' not in wide_df.columns:
        return {'error': 'Missing condition columns'}
    
    wide_df['condition_effect'] = wide_df['hum'] - wide_df['bot']
    
    human_data = wide_df[wide_df['experiment'] == 'Human'].dropna(subset=['condition_effect'])
    llm_data = wide_df[wide_df['experiment'] == 'LLM'].dropna(subset=['condition_effect'])
    
    human_effects = human_data['condition_effect']
    llm_effects = llm_data['condition_effect']
    
    results = {
        'metric': metric,
        'n_human': len(human_effects),
        'n_llm': len(llm_effects),
        'human_effect_mean': human_effects.mean() if len(human_effects) > 0 else np.nan,
        'human_effect_sem': human_effects.std(ddof=1) / np.sqrt(len(human_effects)) if len(human_effects) > 1 else np.nan,
        'llm_effect_mean': llm_effects.mean() if len(llm_effects) > 0 else np.nan,
        'llm_effect_sem': llm_effects.std(ddof=1) / np.sqrt(len(llm_effects)) if len(llm_effects) > 1 else np.nan,
    }
    
    results['human_hum_mean'] = human_data['hum'].mean() if len(human_data) > 0 else np.nan
    results['human_bot_mean'] = human_data['bot'].mean() if len(human_data) > 0 else np.nan
    results['llm_hum_mean'] = llm_data['hum'].mean() if len(llm_data) > 0 else np.nan
    results['llm_bot_mean'] = llm_data['bot'].mean() if len(llm_data) > 0 else np.nan
    
    human_sig = False
    llm_sig = False
    
    if len(human_data) > 1:
        t1, p1 = ttest_rel(human_data['hum'], human_data['bot'])
        results['human_within_t'] = t1
        results['human_within_p'] = p1
        human_sig = p1 < 0.05
    
    if len(llm_data) > 1:
        t2, p2 = ttest_rel(llm_data['hum'], llm_data['bot'])
        results['llm_within_t'] = t2
        results['llm_within_p'] = p2
        llm_sig = p2 < 0.05
    
    if len(human_effects) > 1 and len(llm_effects) > 1:
        t_stat, p_val = ttest_ind(human_effects, llm_effects)
        results['interaction_t'] = t_stat
        results['interaction_p'] = p_val
        results['interaction_df'] = len(human_effects) + len(llm_effects) - 2
    
    if not human_sig:
        human_dir = 'ns'
    elif results['human_effect_mean'] > 0:
        human_dir = 'Human > AI'
    else:
        human_dir = 'AI > Human'
    
    if not llm_sig:
        llm_dir = 'ns'
    elif results['llm_effect_mean'] > 0:
        llm_dir = 'Human > AI'
    else:
        llm_dir = 'AI > Human'
    
    results['human_direction'] = human_dir
    results['llm_direction'] = llm_dir
    
    if human_dir == llm_dir:
        results['pattern'] = 'Both ns' if human_dir == 'ns' else 'Similar'
    elif human_dir == 'ns' or llm_dir == 'ns':
        results['pattern'] = 'Different'
    elif (human_dir == 'Human > AI' and llm_dir == 'AI > Human') or \
         (human_dir == 'AI > Human' and llm_dir == 'Human > AI'):
        results['pattern'] = 'Flipped'
    else:
        results['pattern'] = 'Different'
    
    return results

