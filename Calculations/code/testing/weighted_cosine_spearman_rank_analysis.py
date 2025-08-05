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
    
    # Handle duplicates for ISR by summing up duplicate usummary_codes
    if country == 'ISR':
        print(f"Processing ISR data - checking for duplicates...")
        original_len = len(df)
        # Group by usummary_code and sum the numeric columns
        numeric_cols = ['impVal', 'total', 'direct', 'indirect']
        rank_cols_subset = ['impVal_country_rank', 'total_country_rank', 'direct_country_rank', 'indirect_country_rank']
        
        # For ranks, take the first value (they should be the same for duplicates)
        df_grouped = df.groupby('usummary_code').agg({
            **{col: 'sum' for col in numeric_cols if col in df.columns},
            **{col: 'first' for col in rank_cols_subset if col in df.columns},
            'iso3': 'first'
        }).reset_index()
        
        new_len = len(df_grouped)
        if original_len != new_len:
            print(f"ISR: Consolidated {original_len} rows to {new_len} rows (removed {original_len - new_len} duplicates)")
        
        df = df_grouped
    
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

# Create a function to return the ranked order of the countries based on their import Values

def sort_countries_by_value(df, value_col = 'impVal', remove_codes = None):
    """
    Sort countries by a specific rank column - default to the impVal_country_rank. 

    Args:
        df (_type_): _description_
        value_col (str, optional): _description_. Defaults to 'impVal'.
    """
    country_rank = {}
    for key in df.keys(): 
        if isinstance(key, tuple) and len(key) >=1:
            if key[0] != 'All Countries':
                df_country = df[(key[0], baseline_scenario)]
                if remove_codes is not None:
                    df_country = df_country[~df_country['usummary_code'].isin(remove_codes)]
                df_country = df_country[['iso3',value_col]].dropna().groupby('iso3').sum().reset_index()
                country_rank[key[0]] = df_country[value_col]
    country_rank = pd.DataFrame(country_rank).T.reset_index()
    country_rank = country_rank.rename(columns={'index': 'iso3', 0: value_col})
    country_rank = country_rank.sort_values(by=value_col, ascending=False)
    country_rank['rank'] = np.arange(1, len(country_rank) + 1)
    country_rank = country_rank.reset_index()[['iso3','rank']]
    return country_rank

def run_all_spearman_calculations(data=combined_data, remove_codes=None, p1_color=None, p5_color=None, pInsig=None, csv = False, caption =None):
    """
    Run Spearman rank correlation calculations for all countries in the dataset.
    Returns a dictionary with country names as keys and their Spearman results.
    """
    results = {}
    # Extract the unique countries from the impVal sorted data -- THIS IS NEW
    sorted_countries = sort_countries_by_value(data, value_col='impVal', remove_codes=remove_codes)
    for country in sorted_countries['iso3']:
        results[country] = calculate_country_spearman(country, data, remove_codes=remove_codes)
    # Remove countries with missing or NaN correlation values
    countries_to_remove = [country for country in sorted_countries['iso3']
                           if pd.isna(results[country]['total_vs_impval']['correlation'])]
    for country in countries_to_remove:
        results.pop(country, None)

    dir="Calculations/validations/08_weighted_cos_and_spearman_rank"
    os.makedirs(dir, exist_ok=True)
    
    # Create filename suffix based on removed codes
    filename_suffix = ""
    if remove_codes is not None and len(remove_codes) > 0:
        codes_str = "_".join(remove_codes)
        filename_suffix = f"_excluding_{codes_str}"
    
    if csv:
        # Save results to a CSV file (original format)
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(os.path.join(dir, f'01_spearman_rank_results{filename_suffix}.csv'))
        
        # Create detailed CSV with country names and full precision
        detailed_csv = create_detailed_csv(results)
        detailed_csv.to_csv(os.path.join(dir, f'01_spearman_rank_results_detailed{filename_suffix}.csv'), index=False)
    
    # Generate LaTeX longtable -- No coloring 
    latex_table = generate_latex_table(results)
    with open(os.path.join(dir, f'01_spearman_rank_results{filename_suffix}.tex'), 'w') as f:
        f.write(latex_table)
    
    # Generate LaTeX longtable with coloring (if specified)
    if p1_color is not None or p5_color is not None or pInsig is not None:
            latex_table_colored = generate_latex_table(results, p1_color=p1_color, p5_color=p5_color, pInsig=pInsig, caption = caption)
            with open(os.path.join(dir, f'01_spearman_rank_results_detailed_highlighted{filename_suffix}.tex'), 'w') as f:
                f.write(latex_table_colored)
    
    return results

def create_detailed_csv(results):
    """
    Create a detailed CSV with country names and full precision numbers.
    """
    csv_data = []
    for country_iso in results.keys():
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

def generate_latex_table(results, p1_color = None, p5_color = None, pInsig = None, caption = None):
    """
    Generate a LaTeX longtable from the Spearman correlation results.
    """
    if caption is None: 
        latex_lines = [
            "\\begin{longtable}{p{4cm}p{3cm}p{3cm}p{3cm}}",
            "\\toprule",
            "\\textbf{Country (ISO)} & \\textbf{Total vs Import} & \\textbf{Direct vs Import} & \\textbf{Indirect vs Import} \\\\",
            "\\midrule",
            "\\endfirsthead",
            "",
            "\\toprule",
            "\\textbf{Country (ISO)} & \\textbf{Total vs Import} & \\textbf{Direct vs Import} & \\textbf{Indirect vs Import} \\\\",
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
    if caption is not None:
        latex_lines = [
            "\\begin{longtable}{p{5cm}p{3cm}p{3cm}p{5cm}}",
            "\\centering",
            "\\caption{"+caption+"} \\\\",
            "\\label{tab:"+caption.replace(' ', '_').replace(',', '').replace('-','')+"} \\\\",
            "\\toprule",
            "\\textbf{Country (ISO)} & \\textbf{Total vs Import} & \\textbf{Direct vs Import} & \\textbf{Indirect vs Import} \\\\",
            "\\midrule",
            "\\endfirsthead",
            "",
            "\\toprule",
            "\\textbf{Country (ISO)} & \\textbf{Total vs Import} & \\textbf{Direct vs Import} & \\textbf{Indirect vs Import} \\\\",
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
    sorted_countries = results.keys()
    
    for country in sorted_countries:
        country_results = results[country]
        
        # Get country name using country_converter
        country_name = coco.convert(country, to='short_name')
        country_display = f"{country_name} ({country})"
        
        # Format correlations and p-values with highlighting for significant results
        def format_cell(corr, pval, p1_color=p1_color, p5_color=p5_color, pInsig=pInsig):
            if pd.isna(corr) or pd.isna(pval):
                return None  # Skip this country by returning None
            cell_text = f"{corr:.3f} ({pval:.3f})"
            
            if p1_color is None and p5_color is None and pInsig is None:
                # Highlight significant results
                if pval < 0.01:
                    return f"\\textbf{{{cell_text}}}**"  # Bold with ** for p < 0.01
                elif pval < 0.05:
                    return f"\\textbf{{{cell_text}}}*"   # Bold with * for p < 0.05
                else:
                    return cell_text
            else:
                if p1_color is not None and pval <0.01:
                    return f"\\textcolor{{{p1_color}}}{{\\textbf{{{cell_text}}}**}}"
                elif p5_color is not None and pval < 0.05:
                    return f"\\textcolor{{{p5_color}}}{{\\textbf{{{cell_text}}}*}}"
                elif pInsig is not None and pval >= 0.05:
                    return f"\\textcolor{{{pInsig}}}{{{cell_text}}}"
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
    if p1_color is None and p5_color is None and pInsig is None:
        latex_lines.append("\\textit{Note: Bold values are statistically significant (** indicates p $<$ 0.01, * indicates p $<$ 0.05)). Insignificant values are normal font. Countries ranked by 2024 import values.}")
    else: 
        if p1_color is not None and p5_color is not None and pInsig is not None:
            latex_lines.append(f"\\textit{{Note: Statistically significant values are indicated by \\textcolor{{{p1_color}}}{{**}} (p $<$ 0.01) and \\textcolor{{{p5_color}}}{{*}} (p $<$ 0.05). Insignificant values are \\textcolor{{{pInsig}}}{{this color}}. Countries ranked by 2024 import values.}}")
        else:
            latex_lines.append("\\textit{Note:  Bold values are statistically significant (** indicates p $<$ 0.01, * indicates p $<$ 0.05).  Insignificant values are in normal font. Countries ranked by 2024 import values.}")

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
        
        total_significant = []
        direct_significant = []
        indirect_significant = []
        
        for country_results in results.values():
            total_corr = country_results['total_vs_impval']['correlation']
            direct_corr = country_results['direct_vs_impval']['correlation']
            indirect_corr = country_results['indirect_vs_impval']['correlation']
            
            # Create share of countries that are significant
            total_significant.append(country_results['total_vs_impval']['p_value'] < 0.05)
            direct_significant.append(country_results['direct_vs_impval']['p_value'] < 0.05)
            indirect_significant.append(country_results['indirect_vs_impval']['p_value'] < 0.05)
                
            
            if not pd.isna(total_corr):
                total_corrs.append(total_corr)
            if not pd.isna(direct_corr):
                direct_corrs.append(direct_corr)
            if not pd.isna(indirect_corr):
                indirect_corrs.append(indirect_corr)
        
        summary_data.append({
            'run_name': run_name,
            'avg_total_corr': np.mean(total_corrs) if total_corrs else np.nan,
            'share_total_significant': np.mean(total_significant) if total_significant else np.nan,
            'avg_direct_corr': np.mean(direct_corrs) if direct_corrs else np.nan,
            'share_direct_significant': np.mean(direct_significant) if direct_significant else np.nan,
            'avg_indirect_corr': np.mean(indirect_corrs) if indirect_corrs else np.nan,
            'share_indirect_significant': np.mean(indirect_significant) if indirect_significant else np.nan,
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
        "\\caption{Summary of Average Spearman Correlations by Run}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Run} & \\textbf{Avg Total Corr} & \\textbf{Avg Direct Corr} & \\textbf{Avg Indirect Corr} \\\\",
        "\\midrule"
    ]
    
    for _, row in summary_df.iterrows():
        run_name = row['run_name']
        total_avg = f"{row['avg_total_corr']:.3f} ({100*row['share_total_significant']:.2f})\%" if not pd.isna(row['avg_total_corr']) else "N/A"
        direct_avg = f"{row['avg_direct_corr']:.3f} ({100*row['share_direct_significant']:.2f})\%" if not pd.isna(row['avg_direct_corr']) else "N/A"
        indirect_avg = f"{row['avg_indirect_corr']:.3f} ({100*row['share_indirect_significant']:.2f})\%" if not pd.isna(row['avg_indirect_corr']) else "N/A"

        latex_lines.append(f"{run_name} & {total_avg} & {direct_avg} & {indirect_avg} \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular} \\\\ ",
        "\\scriptsize{Note: Values are the simple averages of the Spearman correlation coefficients across all countries. The share of countries with significant correlations (p <0.05) is shown in parentheses.}",
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
    color_maps ={
        'p1_color': 'atlLime600',
        'p5_color': 'atlLime600',
        'pinsig': 'atlOrange600'
    }
    
    print("Running comparison analysis...")
    
    # Run complete analysis
    print("\n=== Running complete analysis ===")
    results_complete = run_all_spearman_calculations(combined_data, remove_codes=None, p1_color = color_maps['p1_color'], p5_color = color_maps['p5_color'], pInsig = color_maps['pinsig'], caption = "Spearman Rank Correlations - All Commodities")
    
    # Run analysis excluding codes
    print("\n=== Running analysis excluding '211' and 'Other' ===")
    results_excluding = run_all_spearman_calculations(combined_data, remove_codes=['211', 'Other'],p1_color = color_maps['p1_color'], p5_color = color_maps['p5_color'], pInsig = color_maps['pinsig'], caption = "Spearman Rank Correlations - Excluding Oil and Other")

    print("\n=== Running analysis excluding '211', 'Other', and '3313NF' ===")
    results_excluding_NF = run_all_spearman_calculations(combined_data, remove_codes=['211', 'Other', '3313NF'],p1_color = color_maps['p1_color'], p5_color = color_maps['p5_color'], pInsig = color_maps['pinsig'], caption = "Spearman Rank Correlations - Excluding Oil, Nonferrous Metals, and Other")

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