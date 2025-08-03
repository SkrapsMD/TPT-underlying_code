import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import country_converter as coco
from scipy.stats import spearmanr
from data_loader import load_combined_data

# Load the data
combined_data = load_combined_data()

baseline_scenario = 'TEN' # Default, but can be changed to 'TWT'

effect_cols = ['impVal','total','direct','indirect']
share_cols = ['impVal_share','total_share','direct_share','indirect_share']
rank_cols = ['impVal_country_rank','total_country_rank','direct_country_rank','indirect_country_rank']
global_rank_cols = ['impVal_global_rank','total_global_rank','direct_global_rank','indirect_global_rank']

def calculate_country_spearman(country, data = combined_data, remove_codes=None):
    """
    Calculate the spearman rank correlation between each effect and impVal_country_rank for a country.
    Returns correlations for total, direct, and indirect effects vs impVal baseline.
    """
    df = data[(country, baseline_scenario)]
    df = df[df['impVal_country_rank'].notna()]  # Filter out rows with NaN ranks
    
    # Remove specified usummary_codes if provided
    if remove_codes is not None and len(remove_codes) > 0:
        df = df[~df['usummary_code'].isin(remove_codes)]
    # Extract the rank columns
    effect_ranks = df[['impVal_country_rank', 'total_country_rank', 'direct_country_rank', 'indirect_country_rank']]
    
    # Calculate Spearman correlations between each effect and impVal baseline
    country_results = {}
    
    # Total vs impVal
    corr_total_impval, p_val_total_impval = spearmanr(effect_ranks['impVal_country_rank'], 
                                                      effect_ranks['total_country_rank'])
    country_results['total_vs_impval'] = {
        'correlation': corr_total_impval,
        'p_value': p_val_total_impval
    }
    
    # Direct vs impVal  
    corr_direct_impval, p_val_direct_impval = spearmanr(effect_ranks['impVal_country_rank'], 
                                                        effect_ranks['direct_country_rank'])
    country_results['direct_vs_impval'] = {
        'correlation': corr_direct_impval,
        'p_value': p_val_direct_impval
    }
    
    # Indirect vs impVal
    corr_indirect_impval, p_val_indirect_impval = spearmanr(effect_ranks['impVal_country_rank'], 
                                                            effect_ranks['indirect_country_rank'])
    country_results['indirect_vs_impval'] = {
        'correlation': corr_indirect_impval,
        'p_value': p_val_indirect_impval
    }
    
    return country_results

def run_all_spearman_calculations(data=combined_data, remove_codes=None):
    """
    Run Spearman rank correlation calculations for all countries in the dataset.
    Returns a dictionary with country names as keys and their Spearman results.
    """
    results = {}
    # Extract unique countries from the dictionary keys
    countries = set()
    for key in data.keys():
        if isinstance(key, tuple) and len(key) >= 1:
            if key[0] != 'All Countries':
                countries.add(key[0])
    
    for country in sorted(countries):
        results[country] = calculate_country_spearman(country, data, remove_codes=remove_codes)
        
    dir="Calculations/validations/08_weighted_cos_and_spearman_rank"
    os.makedirs(dir, exist_ok=True)
    
    # Create filename suffix based on removed codes
    filename_suffix = ""
    if remove_codes is not None and len(remove_codes) > 0:
        codes_str = "_".join(remove_codes)
        filename_suffix = f"_excluding_{codes_str}"
    
    # Save results to a CSV file (original format)
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(dir, f'01_spearman_rank_results{filename_suffix}.csv'))
    
    # Create detailed CSV with country names and full precision
    detailed_csv = create_detailed_csv(results)
    detailed_csv.to_csv(os.path.join(dir, f'01_spearman_rank_results_detailed{filename_suffix}.csv'), index=False)
    
    # Generate LaTeX longtable
    latex_table = generate_latex_table(results)
    with open(os.path.join(dir, f'01_spearman_rank_results{filename_suffix}.tex'), 'w') as f:
        f.write(latex_table)
    
    return results

def create_detailed_csv(results):
    """
    Create a detailed CSV with country names and full precision numbers.
    """
    csv_data = []
    
    for country_iso in sorted(results.keys()):
        country_results = results[country_iso]
        country_name = coco.convert(country_iso, to='short_name')
        
        row = {
            'country_name': country_name,
            'country_iso': country_iso,
            'total_vs_impval_correlation': country_results['total_vs_impval']['correlation'],
            'total_vs_impval_p_value': country_results['total_vs_impval']['p_value'],
            'direct_vs_impval_correlation': country_results['direct_vs_impval']['correlation'],
            'direct_vs_impval_p_value': country_results['direct_vs_impval']['p_value'],
            'indirect_vs_impval_correlation': country_results['indirect_vs_impval']['correlation'],
            'indirect_vs_impval_p_value': country_results['indirect_vs_impval']['p_value']
        }
        csv_data.append(row)
    
    return pd.DataFrame(csv_data)

def generate_latex_table(results):
    """
    Generate a LaTeX longtable from the Spearman correlation results.
    """
    latex_lines = [
        "\\begin{longtable}{p{4cm}p{3cm}p{3cm}p{3cm}}",
        "\\toprule",
        "\\textbf{Country} & \\textbf{Total vs ImpVal} & \\textbf{Direct vs ImpVal} & \\textbf{Indirect vs ImpVal} \\\\",
        "\\midrule",
        "\\endfirsthead",
        "",
        "\\toprule",
        "\\textbf{Country} & \\textbf{Total vs ImpVal} & \\textbf{Direct vs ImpVal} & \\textbf{Indirect vs ImpVal} \\\\",
        "\\midrule",
        "\\endhead",
        "",
        "\\bottomrule",
        "\\endfoot",
        "",
        "\\bottomrule",
        "\\endlastfoot",
        ""
    ]
    
    # Sort countries alphabetically
    sorted_countries = sorted(results.keys())
    
    for country in sorted_countries:
        country_results = results[country]
        
        # Get country name using country_converter
        country_name = coco.convert(country, to='short_name')
        country_display = f"{country_name} ({country})"
        
        # Format correlations and p-values with highlighting for significant results
        def format_cell(corr, pval):
            if pd.isna(corr) or pd.isna(pval):
                return "N/A"
            
            cell_text = f"{corr:.3f} ({pval:.3f})"
            
            # Highlight significant results
            if pval < 0.01:
                return f"\\textbf{{{cell_text}}}**"  # Bold with ** for p < 0.01
            elif pval < 0.05:
                return f"\\textbf{{{cell_text}}}*"   # Bold with * for p < 0.05
            else:
                return cell_text
        
        total_corr = country_results['total_vs_impval']['correlation']
        total_pval = country_results['total_vs_impval']['p_value']
        total_cell = format_cell(total_corr, total_pval)
        
        direct_corr = country_results['direct_vs_impval']['correlation']
        direct_pval = country_results['direct_vs_impval']['p_value']
        direct_cell = format_cell(direct_corr, direct_pval)
        
        indirect_corr = country_results['indirect_vs_impval']['correlation']
        indirect_pval = country_results['indirect_vs_impval']['p_value']
        indirect_cell = format_cell(indirect_corr, indirect_pval)
        
        latex_lines.append(f"{country_display} & {total_cell} & {direct_cell} & {indirect_cell} \\\\")
    
    latex_lines.append("\\end{longtable}")
    
    # Add significance note
    latex_lines.append("")
    latex_lines.append("\\textit{Note: ** indicates p < 0.01, * indicates p < 0.05. Bold values are statistically significant.}")
    
    return "\n".join(latex_lines)

def create_summary_table(results_dict):
    """
    Create a summary table with average coefficients for different runs.
    results_dict: dictionary with run names as keys and results as values
    """
    summary_data = []
    
    for run_name, results in results_dict.items():
        # Calculate average correlations across all countries
        total_corrs = []
        direct_corrs = []
        indirect_corrs = []
        
        for country_results in results.values():
            total_corr = country_results['total_vs_impval']['correlation']
            direct_corr = country_results['direct_vs_impval']['correlation']
            indirect_corr = country_results['indirect_vs_impval']['correlation']
            
            if not pd.isna(total_corr):
                total_corrs.append(total_corr)
            if not pd.isna(direct_corr):
                direct_corrs.append(direct_corr)
            if not pd.isna(indirect_corr):
                indirect_corrs.append(indirect_corr)
        
        summary_data.append({
            'run_name': run_name,
            'avg_total_corr': np.mean(total_corrs) if total_corrs else np.nan,
            'avg_direct_corr': np.mean(direct_corrs) if direct_corrs else np.nan,
            'avg_indirect_corr': np.mean(indirect_corrs) if indirect_corrs else np.nan,
            'n_countries': len(results)
        })
    
    return pd.DataFrame(summary_data)

def generate_summary_latex_table(summary_df):
    """
    Generate LaTeX table for the summary results.
    """
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Run} & \\textbf{Avg Total Corr} & \\textbf{Avg Direct Corr} & \\textbf{Avg Indirect Corr} \\\\",
        "\\midrule"
    ]
    
    for _, row in summary_df.iterrows():
        run_name = row['run_name']
        total_avg = f"{row['avg_total_corr']:.3f}" if not pd.isna(row['avg_total_corr']) else "N/A"
        direct_avg = f"{row['avg_direct_corr']:.3f}" if not pd.isna(row['avg_direct_corr']) else "N/A"
        indirect_avg = f"{row['avg_indirect_corr']:.3f}" if not pd.isna(row['avg_indirect_corr']) else "N/A"
        
        latex_lines.append(f"{run_name} & {total_avg} & {direct_avg} & {indirect_avg} \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Summary of Average Spearman Correlations by Run}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_lines)

def main(remove_codes=None):
    """
    Main function to run all Spearman rank correlation calculations and generate outputs.
    """
    print("Starting Spearman rank correlation analysis...")
    
    # Run calculations for all countries
    results = run_all_spearman_calculations(combined_data, remove_codes=remove_codes)
    
    print(f"Analysis complete for {len(results)} countries.")
    
    # Create filename suffix for output messages
    filename_suffix = ""
    if remove_codes is not None and len(remove_codes) > 0:
        codes_str = "_".join(remove_codes)
        filename_suffix = f"_excluding_{codes_str}"
        print(f"Excluded codes: {remove_codes}")
    
    print("Files generated:")
    print(f"- 01_spearman_rank_results{filename_suffix}.csv (original format)")
    print(f"- 01_spearman_rank_results_detailed{filename_suffix}.csv (with country names and full precision)")
    print(f"- 01_spearman_rank_results{filename_suffix}.tex (LaTeX longtable)")
    
    return results

def run_comparison():
    """
    Run both analyses and create a summary comparison table.
    """
    print("Running comparison analysis...")
    
    # Run complete analysis
    print("\n=== Running complete analysis ===")
    results_complete = run_all_spearman_calculations(combined_data, remove_codes=None)
    
    # Run analysis excluding codes
    print("\n=== Running analysis excluding '211' and 'Other' ===")
    results_excluding = run_all_spearman_calculations(combined_data, remove_codes=['211', 'Other'])
    
    print("\n=== Running analysis excluding '211', 'Other', and '3313NF' ===")
    results_excluding_NF = run_all_spearman_calculations(combined_data, remove_codes=['211', 'Other', '3313NF'])

    # Create summary comparison
    results_dict = {
        'Complete': results_complete,
        'Excluding Oil and Other': results_excluding,
        'Excluding Oil, Other, and Nonferrous Metals': results_excluding_NF
    }
    
    summary_df = create_summary_table(results_dict)
    summary_latex = generate_summary_latex_table(summary_df)
    
    # Save summary files
    dir_path = "Calculations/validations/08_weighted_cos_and_spearman_rank"
    summary_df.to_csv(os.path.join(dir_path, 'summary_comparison.csv'), index=False)
    
    with open(os.path.join(dir_path, 'summary_comparison.tex'), 'w') as f:
        f.write(summary_latex)
    
    return summary_df

if __name__ == "__main__":
    run_comparison()