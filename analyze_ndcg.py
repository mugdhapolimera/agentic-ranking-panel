import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, cohen_kappa_score

def calculate_ndcg(scores: List[float], k: int = None) -> float:
    """
    Calculate nDCG (normalized Discounted Cumulative Gain) for a list of scores.
    
    Args:
        scores: List of relevance scores (0-3)
        k: Number of results to consider (None for all)
    
    Returns:
        nDCG score
    """
    if not scores:
        return 0.0
    
    # Use all scores if k is not specified
    if k is None:
        k = len(scores)
    else:
        k = min(k, len(scores))
    
    # Calculate DCG
    dcg = 0
    for i, score in enumerate(scores[:k]):
        dcg += (2 ** score - 1) / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate ideal DCG (IDCG)
    ideal_scores = sorted(scores, reverse=True)
    idcg = 0
    for i, score in enumerate(ideal_scores[:k]):
        idcg += (2 ** score - 1) / np.log2(i + 2)
    
    # Calculate nDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

def load_results(file_path: str) -> Dict:
    """Load results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_scores(results: Dict, score_type: str) -> List[float]:
    """
    Extract scores from results dictionary.
    
    Args:
        results: Dictionary containing search results
        score_type: Type of score to extract ('sme', 'consensus', 'claude', 'gemini', 'deepseek')
    
    Returns:
        List of scores
    """
    scores = []
    for result in results.get('search_results', []):
        if score_type == 'sme':
            score = result.get('sme_judgements')
        elif score_type == 'consensus':
            score = result.get('consensus_score')
        else:
            score = result.get(f'{score_type}_score')
        if score is not None:
            try:
                scores.append(float(score))
            except Exception:
                continue
    return scores

def extract_scores_pairwise(results: Dict, score_type: str) -> Tuple[List[float], List[float]]:
    """
    Returns (sme_judgements, score_type_scores) for all results where both are present.
    """
    sme = []
    llm = []
    for result in results.get('search_results', []):
        sme_score = result.get('sme_judgements')
        if score_type == 'consensus':
            llm_score = result.get('consensus_score')
        else:
            llm_score = result.get(f'{score_type}_score')
        if sme_score is not None and llm_score is not None:
            try:
                sme.append(float(sme_score))
                llm.append(float(llm_score))
            except Exception:
                continue
    return sme, llm

def plot_sme_vs_llm(ndcg_per_set, output_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr

    if not ndcg_per_set:
        print('No nDCG data to plot.')
        return

    df = pd.DataFrame(ndcg_per_set)
    judges = ['claude', 'gemini', 'deepseek', 'consensus']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    markers = ['o', 's', '^', 'D']

    plt.figure(figsize=(8, 8))
    corrs = {}
    for i, judge in enumerate(judges):
        x = df['sme']
        y = df[judge]
        plt.scatter(x, y, label=judge.capitalize(), color=colors[i], marker=markers[i], alpha=0.7)
        if len(x) > 1 and len(y) > 1:
            try:
                corr, _ = pearsonr(x, y)
                corrs[judge] = corr
            except Exception:
                corrs[judge] = None
        else:
            corrs[judge] = None

    # Plot y=x line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xlabel('SME nDCG')
    plt.ylabel('LLM/Consensus nDCG')
    plt.title('nDCG: SME vs LLM/Consensus (per set)')
    plt.legend()
    # Annotate correlation coefficients
    y0 = 1.02
    for i, judge in enumerate(judges):
        if corrs.get(judge) is not None:
            plt.text(0.02, y0 - 0.06 * i, f"{judge.capitalize()} r = {corrs[judge]:.3f}", color=colors[i])
    plt.grid(True, alpha=0.3)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'ndcg_sme_vs_llm.png'), dpi=300, bbox_inches='tight')
    plt.close()

def bland_altman_plot(sme, llm, label, output_dir=None):
    mean = (np.array(sme) + np.array(llm)) / 2
    diff = np.array(llm) - np.array(sme)
    md = np.mean(diff)
    sd = np.std(diff)
    plt.figure(figsize=(7, 5))
    plt.scatter(mean, diff, alpha=0.7)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='red', linestyle='--')
    plt.axhline(md - 1.96*sd, color='red', linestyle='--')
    plt.xlabel('Mean of SME and ' + label)
    plt.ylabel(f'{label} - SME')
    plt.title(f'Bland-Altman Plot: SME vs {label}')
    plt.grid(True, alpha=0.3)
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'bland_altman_{label.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return md, sd

def plot_confusion(sme, llm, label, output_dir=None):
    cm = confusion_matrix(sme, llm, labels=[0,1,2,3])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
    plt.xlabel(f'{label} Score')
    plt.ylabel('SME Score')
    plt.title(f'Confusion Matrix: SME vs {label}')
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{label.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return cm

def plot_distributions(sme, llm_dict, output_dir=None):
    plt.figure(figsize=(8, 6))
    data = {'SME': sme}
    data.update(llm_dict)
    df = []
    for k, v in data.items():
        for score in v:
            df.append({'Judge': k, 'Score': score})
    import pandas as pd
    df = pd.DataFrame(df)
    if not df.empty and 'Judge' in df.columns and 'Score' in df.columns:
        sns.violinplot(x='Judge', y='Score', data=df, inner='box', palette='Set2')
        plt.title('Score Distributions')
        plt.ylim(-0.2, 3.2)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print('No data for violin plot.')

def analyze_results(output_file: str, output_dir: str = None):
    """
    Analyze results and create visualizations.
    
    Args:
        output_file: Path to output JSON file with LLM scores
        output_dir: Directory to save visualizations
    """
    # Load files
    output_data = load_results(output_file)
    
    # Calculate nDCG for different score types
    score_types = ['sme', 'claude', 'gemini', 'deepseek', 'consensus']
    ndcg_scores = {}
    
    for score_type in score_types:
        # Extract scores
        scores = extract_scores(output_data, score_type)
        
        # Calculate nDCG
        ndcg = calculate_ndcg(scores)
        
        ndcg_scores[score_type] = ndcg
        
        # Print results
        print(f"\n{score_type.upper()} nDCG: {ndcg:.3f}")
    
    # Create scatter plot
    corrs = plot_sme_vs_llm(output_data, output_dir)
    
    # Save results to JSON
    if output_dir:
        results = {
            'correlations': corrs,
            'output_file': output_file
        }
        with open(os.path.join(output_dir, 'sme_vs_llm_correlation.json'), 'w') as f:
            json.dump(results, f, indent=2)
    print("\nPearson correlations (SME vs LLM/Consensus):")
    for k, v in corrs.items():
        if v is not None:
            print(f"{k.capitalize()}: r = {v:.3f}")
        else:
            print(f"{k.capitalize()}: Not enough data")

    # MAE, RMSE, t-test, Wilcoxon, Kappa, Bland-Altman, Confusion
    stats = {}
    all_sme = []
    all_llm = {k: [] for k in score_types}
    # For violin plot
    for result in output_data.get('search_results', []):
        sme_score = result.get('sme_judgements')
        if sme_score is not None:
            all_sme.append(float(sme_score))
            for k in score_types:
                if k == 'consensus':
                    llm_score = result.get('consensus_score')
                else:
                    llm_score = result.get(f'{k}_score')
                if llm_score is not None:
                    all_llm[k].append(float(llm_score))
                else:
                    all_llm[k].append(np.nan)
    # Correlation scatter plot
    corrs = plot_sme_vs_llm(output_data, output_dir)
    stats['correlations'] = corrs
    # MAE, RMSE, t-test, Wilcoxon, Kappa, Bland-Altman, Confusion
    for i, score_type in enumerate(score_types):
        sme, llm = extract_scores_pairwise(output_data, score_type)
        if len(sme) < 2:
            continue
        arr = np.array([(s, l) for s, l in zip(sme, llm) if not (np.isnan(s) or np.isnan(l))])
        if arr.shape[0] < 2:
            continue
        s, l = arr[:,0], arr[:,1]
        mae = mean_absolute_error(s, l)
        rmse = np.sqrt(mean_squared_error(s, l))
        try:
            t_stat, t_p = ttest_rel(s, l)
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            w_stat, w_p = wilcoxon(s, l)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        # Round to int for classification metrics
        s_int = np.round(s).astype(int)
        l_int = np.round(l).astype(int)
        try:
            kappa = cohen_kappa_score(s_int, l_int, weights='quadratic')
        except Exception:
            kappa = np.nan
        md, sd = bland_altman_plot(s, l, score_type.capitalize(), output_dir)
        try:
            cm = plot_confusion(s_int, l_int, score_type.capitalize(), output_dir)
        except Exception:
            cm = np.zeros((4,4), dtype=int)
        stats[score_type] = {
            'mae': mae,
            'rmse': rmse,
            't_stat': t_stat,
            't_p': t_p,
            'wilcoxon_stat': w_stat,
            'wilcoxon_p': w_p,
            'kappa': kappa,
            'bland_altman_mean_diff': md,
            'bland_altman_std_diff': sd,
            'confusion_matrix': cm.tolist()
        }
    # Distribution plot
    plot_distributions(all_sme, {k: [x for x in all_llm[k] if not np.isnan(x)] for k in score_types}, output_dir)
    if output_dir:
        with open(os.path.join(output_dir, 'sme_vs_llm_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    print("\nStatistics and figures saved to:", output_dir)
    print("\nSummary:")
    for k in score_types:
        if k in stats:
            print(f"{k.capitalize()} | MAE: {stats[k]['mae']:.3f} | RMSE: {stats[k]['rmse']:.3f} | Kappa: {stats[k]['kappa']:.3f} | t_p: {stats[k]['t_p']:.3g} | w_p: {stats[k]['wilcoxon_p']:.3g}")

def auto_analyze_examples(examples_dir: str = 'examples'):
    summary = []
    ndcg_per_set = []  # List of dicts: {'set': set_name, 'sme': ..., 'claude': ..., ...}
    # Find all llm_scored_results.json files recursively
    output_files = glob.glob(os.path.join(examples_dir, '**', '*llm_scored_results.json'), recursive=True)
    for output_file in output_files:
        # Try to infer bibcode and set name from the file path
        rel_path = os.path.relpath(output_file, examples_dir)
        parts = rel_path.split(os.sep)
        if len(parts) >= 3:
            bibcode = parts[0]
            set_name = os.path.splitext(os.path.basename(output_file))[0]
        else:
            bibcode = 'unknown'
            set_name = os.path.splitext(os.path.basename(output_file))[0]
        analysis_dir = os.path.join(os.path.dirname(output_file), set_name + '_analysis')
        print(f"\nAnalyzing bibcode: {bibcode}, set: {set_name}")
        try:
            # Compute nDCG for this set
            output_data = load_results(output_file)
            ndcg_row = {'set': set_name}
            for judge in ['sme', 'claude', 'gemini', 'deepseek', 'consensus']:
                scores = extract_scores(output_data, judge)
                ndcg_row[judge] = calculate_ndcg(scores)
            ndcg_per_set.append(ndcg_row)
            analyze_results(output_file, analysis_dir)
            summary.append({'bibcode': bibcode, 'set': set_name, 'output': output_file, 'analysis_dir': analysis_dir, 'status': 'success'})
        except Exception as e:
            print(f"Error analyzing {set_name}: {e}")
            summary.append({'bibcode': bibcode, 'set': set_name, 'output': output_file, 'analysis_dir': analysis_dir, 'status': f'error: {e}'})
    print("\n=== Auto Analysis Summary ===")
    for entry in summary:
        print(f"{entry['bibcode']} | {entry['set']} | {entry['status']} | {entry['analysis_dir']}")
    print(f"\nTotal analyzed: {len(summary)}")
    # After all, do global analysis
    aggregate_all_results(examples_dir, output_dir='examples/global_analysis')
    # Plot nDCG bar and line plots for all sets
    plot_ndcg_across_sets(ndcg_per_set, output_dir='examples/global_analysis')
    plot_sme_vs_llm(ndcg_per_set, output_dir='examples/global_analysis')

def plot_ndcg_across_sets(ndcg_per_set, output_dir):
    import pandas as pd
    if not ndcg_per_set:
        print('No nDCG data to plot.')
        return
    df = pd.DataFrame(ndcg_per_set)
    judges = ['sme', 'claude', 'gemini', 'deepseek', 'consensus']
    set_names = df['set'].tolist()
    # Bar plot
    plt.figure(figsize=(max(10, len(set_names)*0.6), 6))
    bar_width = 0.15
    x = np.arange(len(set_names))
    for i, judge in enumerate(judges):
        plt.bar(x + i*bar_width, df[judge], width=bar_width, label=judge.capitalize())
    plt.xticks(x + bar_width*2, set_names, rotation=90)
    plt.ylabel('nDCG')
    plt.ylim(0, 1.05)
    plt.title('nDCG per Set (Bar Plot)')
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'ndcg_barplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Line plot
    plt.figure(figsize=(max(10, len(set_names)*0.6), 6))
    for judge in judges:
        plt.plot(set_names, df[judge], marker='o', label=judge.capitalize())
    plt.xticks(rotation=90)
    plt.ylabel('nDCG')
    plt.ylim(0, 1.05)
    plt.title('nDCG per Set (Line Plot)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ndcg_lineplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Save the nDCG data as CSV for reference
    df.to_csv(os.path.join(output_dir, 'ndcg_per_set.csv'), index=False)

def aggregate_all_results(examples_dir: str = 'examples', output_dir: str = 'examples/global_analysis'):
    score_types = ['claude', 'gemini', 'deepseek', 'consensus']
    all_sme = []
    all_llm = {k: [] for k in score_types}
    all_pairs = {k: {'sme': [], 'llm': []} for k in score_types}
    # Find all llm_scored_results.json files recursively
    output_files = glob.glob(os.path.join(examples_dir, '**', '*llm_scored_results.json'), recursive=True)
    for output_file in output_files:
        try:
            output_data = load_results(output_file)
            for result in output_data.get('search_results', []):
                sme_score = result.get('sme_judgements')
                if sme_score is not None:
                    all_sme.append(float(sme_score))
                    for k in score_types:
                        if k == 'consensus':
                            llm_score = result.get('consensus_score')
                        else:
                            llm_score = result.get(f'{k}_score')
                        if llm_score is not None:
                            all_llm[k].append(float(llm_score))
                            all_pairs[k]['sme'].append(float(sme_score))
                            all_pairs[k]['llm'].append(float(llm_score))
        except Exception as e:
            print(f"Error loading {output_file}: {e}")
    # Now, use the same plotting/statistics functions as before
    # Correlation scatter plot
    corrs = plot_sme_vs_llm_global(all_pairs, output_dir)
    stats = {'correlations': corrs}
    # MAE, RMSE, t-test, Wilcoxon, Kappa, Bland-Altman, Confusion
    for k in score_types:
        s, l = np.array(all_pairs[k]['sme']), np.array(all_pairs[k]['llm'])
        if len(s) < 2:
            continue
        arr = np.array([(s_, l_) for s_, l_ in zip(s, l) if not (np.isnan(s_) or np.isnan(l_))])
        if arr.shape[0] < 2:
            continue
        s, l = arr[:,0], arr[:,1]
        mae = mean_absolute_error(s, l)
        rmse = np.sqrt(mean_squared_error(s, l))
        try:
            t_stat, t_p = ttest_rel(s, l)
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            w_stat, w_p = wilcoxon(s, l)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        # Round to int for classification metrics
        s_int = np.round(s).astype(int)
        l_int = np.round(l).astype(int)
        try:
            kappa = cohen_kappa_score(s_int, l_int, weights='quadratic')
        except Exception:
            kappa = np.nan
        md, sd = bland_altman_plot(s, l, k.capitalize(), output_dir)
        try:
            cm = plot_confusion(s_int, l_int, k.capitalize(), output_dir)
        except Exception:
            cm = np.zeros((4,4), dtype=int)
        stats[k] = {
            'mae': mae,
            'rmse': rmse,
            't_stat': t_stat,
            't_p': t_p,
            'wilcoxon_stat': w_stat,
            'wilcoxon_p': w_p,
            'kappa': kappa,
            'bland_altman_mean_diff': md,
            'bland_altman_std_diff': sd,
            'confusion_matrix': cm.tolist()
        }
    # Distribution plot
    plot_distributions(all_sme, {k: all_llm[k] for k in score_types}, output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'sme_vs_llm_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    print("\nGlobal statistics and figures saved to:", output_dir)
    print("\nGlobal Summary:")
    for k in score_types:
        if k in stats:
            print(f"{k.capitalize()} | MAE: {stats[k]['mae']:.3f} | RMSE: {stats[k]['rmse']:.3f} | Kappa: {stats[k]['kappa']:.3f} | t_p: {stats[k]['t_p']:.3g} | w_p: {stats[k]['wilcoxon_p']:.3g}")

def plot_sme_vs_llm_global(all_pairs, output_dir=None):
    score_types = ['claude', 'gemini', 'deepseek', 'consensus']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    markers = ['o', 's', '^', 'D']
    plt.figure(figsize=(8, 8))
    corrs = {}
    for i, score_type in enumerate(score_types):
        sme = all_pairs[score_type]['sme']
        llm = all_pairs[score_type]['llm']
        if len(sme) == 0:
            continue
        plt.scatter(sme, llm, label=f'{score_type.capitalize()}', color=colors[i], marker=markers[i], alpha=0.7)
        # Pearson correlation
        if len(sme) > 1:
            corr, pval = pearsonr(sme, llm)
            corrs[score_type] = corr
        else:
            corrs[score_type] = None
    plt.plot([0, 3], [0, 3], 'k--', alpha=0.3)
    plt.xlim(-0.2, 3.2)
    plt.ylim(-0.2, 3.2)
    plt.xlabel('SME Judgement')
    plt.ylabel('LLM/Consensus Score')
    plt.title('Global Correlation: SME vs LLM/Consensus Scores')
    plt.legend()
    y0 = 3.05
    for i, score_type in enumerate(score_types):
        if corrs.get(score_type) is not None:
            plt.text(0.05, y0 - 0.18 * i, f"{score_type.capitalize()} r = {corrs[score_type]:.3f}", color=colors[i])
    plt.grid(True, alpha=0.3)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'global_sme_vs_llm_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return corrs

def main():
    parser = argparse.ArgumentParser(description="Analyze correlation between SME and LLM/consensus scores from merged LLM results.")
    parser.add_argument('--output', help='Path to output JSON file with LLM scores')
    parser.add_argument('--output-dir', help='Directory to save visualizations and analysis results')
    parser.add_argument('--auto', action='store_true', help='Automatically analyze all <bibcode>_similar_llm_scored_results.json files in the examples directory')
    args = parser.parse_args()
    if args.auto:
        auto_analyze_examples('examples')
        return
    if not args.output:
        print("Error: --output is required unless --auto is set.")
        return
    if not args.output_dir:
        output_path = Path(args.output)
        args.output_dir = str(output_path.parent / 'analysis')
    analyze_results(args.output, args.output_dir)

if __name__ == "__main__":
    main() 