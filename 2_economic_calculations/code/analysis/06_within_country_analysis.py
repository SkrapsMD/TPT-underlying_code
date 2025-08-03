import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import country_converter as coco
from data_loader import load_combined_data


baseline_scenario = 'TEN' # Default, but can be changed to 'TWT'
# Load the combined data structure
# combined_data[('country','TWT/TEN')]

cos_sim_df = pd.read_csv('Calculations/validations/05_cos_validation/within_country_cosine_similarity_TEN.csv')
cos_sim_df = cos_sim_df[cos_sim_df['iso3'] != 'All Countries']


def plot_cos_sim(df, variable, type='frequency', annotate_data = True): 
    """
    Description: Plots the cosine similarity for a given variable. 
    
    Args: 
        df (pd.DataFrame): DataFrame containing cosine similarity data.
        variable (str): Plot either 'total_sim','direct_sim', or 'indirect_sim'.
        type (str): Type of plot, default is 'frequency', takes 'frequency' or 'share'.
    """
    title_map = {
        'total_sim': 'Total effect Cosine Similarity',
        'direct_sim': 'Direct effect Cosine Similarity',
        'indirect_sim': 'Indirect effect Cosine Similarity'
    }
    colors = {
        'total_sim': '#3581b4',
        'direct_sim': '#cd590c',
        'indirect_sim': '#74ac1c',
        'median': '#580d10'
    }
    color = colors.get(variable)
    
    plt.figure(figsize=(10, 6))
    data = df[variable].dropna()
    bins = 30

    if type == 'share':
        counts, bin_edges = np.histogram(data, bins=bins)
        shares = counts / counts.sum() * 100  # Convert to percentage
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(bin_centers, shares, width=(bin_edges[1] - bin_edges[0]), color=color, alpha=0.7)
        plt.ylabel('Share (%)')
    else:
        plt.hist(data, bins=bins, color=color, alpha=0.7)
        plt.ylabel('Frequency')

    median_val = data.median()
    plt.axvline(median_val, color=colors['median'], linestyle="--", linewidth=2)
    plt.annotate(f"Median: {median_val:.2f}", 
                xy=(median_val, plt.ylim()[1]*0.95), 
                xytext=(50, -20), 
                textcoords='offset points',
                ha='center', color=colors['median'], fontsize=12,
                fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=colors['median']))
    plt.title(f'Histogram of {title_map[variable]} Values')
    plt.xlabel('Cosine Similarity')
    plt.grid(axis='y', alpha=0.75)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    # Remove legend for median
    if annotate_data: 
        bin_edges = np.histogram(data, bins=bins)[1]
        bin_indices = np.digitize(df[variable], bin_edges, right=False) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)
        df[f'{variable}_bins'] = bin_indices
    return plt

def create_histograms(cos_sim_df):
    """
    Description: Creates histograms for the cosine similarity variables in the DataFrame.
    
    Args:
        cos_sim_df (pd.DataFrame): DataFrame containing cosine similarity data.
    """
    # Create histograms for each variable
    for variable in ['total_sim','direct_sim', 'indirect_sim']:
        plt = plot_cos_sim(cos_sim_df, variable, type = 'share')
        dir = 'Calculations/validations/06_effect_validation_cont'
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, f'{variable}_histogram.png'))
        plt.close()
create_histograms(cos_sim_df)

def create_bins(df, variable, bins = 3): 
    """
    Description: Creates a table with the bottom {bins} for the variable from {variable}_bins. We want
    to look at the countries that we have the least cosing simiarity for. 
    Args: 
        df (pd.DataFrame): DataFrame containing cosine similarity data.
        variable (str): The variable to create bins for.
        bins (int): Number of bins to create, default is 3. 
    """
    effect_map = {
        'total_sim': 'Total Effect',
        'direct_sim': 'Direct Effect',
        'indirect_sim': 'Indirect Effect'
    }
    title = f"Bottom {bins} Cosine Similarity Bins for {effect_map[variable]}"
    
    df = df[df[f'{variable}_bins'] <= bins]
    df['country'] = coco.convert(df['iso3'], to='name_short', not_found=None)
    df = df.sort_values(by=f'{variable}_bins', ascending=True)
    df_output = df[['country','iso3',variable,f'{variable}_bins']].copy()
    df_output = df_output.rename(columns={'country': 'Country', 'iso3':'ISO' ,variable: 'Cosine Similarity', f'{variable}_bins': 'Bin No.'})

    dir = 'Calculations/validations/06_effect_validation_cont/tables'
    os.makedirs(dir, exist_ok=True)
    
    # Output to LaTeX table
    tex_file_path = os.path.join(dir, f'{variable}_bottom_bins.tex')
    with open(tex_file_path, 'w') as tex_file:
        tex_file.write('\\begin{table}[htbp]\n')
        tex_file.write('\\centering\n')
        tex_file.write(f"\\caption{{{title}}}\n")
        tex_file.write(f"\\label{{tab:bottom_cosine_similarity_bins_{variable}}}\n")
        tex_file.write("\\begin{tabular}{llcc}\n")
        tex_file.write("\\toprule\n")
        tex_file.write("Country & ISO & Cosine Similarity & Bin No. \\\\\n")
        tex_file.write("\\midrule\n")
        
        for _, row in df_output.iterrows():
            tex_file.write(f"{row['Country']} & {row['ISO']} &  {row['Cosine Similarity']:.5f} & {row['Bin No.']} \\\\\n")

        tex_file.write("\\bottomrule\n")
        tex_file.write("\\end{tabular}\n")
        tex_file.write('\\end{table}\n')
    return df
bottom_bins_direct = create_bins(cos_sim_df, 'direct_sim', bins = 4)
bottom_bins_indirect = create_bins(cos_sim_df, 'indirect_sim', bins = 4)
bottom_bins_total = create_bins(cos_sim_df, 'total_sim', bins = 4)
# Load the combined data structure with the imports and the effects data
combined_data = load_combined_data()

def evaluate_imports_and_effects(country, effect):
    """
    We know what the bottom countries are by the cosine similarity (bottom_bins). We now want 
    to see what is impelling this low cosine similarity between the imports and the effects. To do this, we can use our 
    combined_data[(country,scenario)] to get the import and effect data for each country. 
    
    This will set up the analysis for us: I'm thinking two kind of overlapping histograms: one for the imports and one for the effects. 
    WE do this using the shares. Creates both with and without oil versions.
    
    Args:
        country (str): The ISO3 code of the country to evaluate.
        effect (str): The type of effect to evaluate, e.g., 'direct','indirect','total'.
    """
    
    def create_plot(drop_oil_flag, drop_other_flag, suffix):
        df = combined_data[(country,'TEN')][['iso3','usummary_code','usummary_desc','impVal','impVal_share',f'{effect}',f'{effect}_share']]
        # Fill NaN values with 0 for numeric columns only
        df[f'{effect}_share'] = df[f'{effect}_share'].fillna(0)
        df['impVal_share'] = df['impVal_share'].fillna(0)
        df['impVal'] = df['impVal'].fillna(0)
        df[f'{effect}'] = df[f'{effect}'].fillna(0)
        # Fill NaN descriptions with 'Unknown'
        df['usummary_desc'] = df['usummary_desc'].fillna('Unknown')
        
        # Drop oil if requested
        if drop_oil_flag:
            df.loc[df['usummary_code'] == '211', 'impVal'] = 0
            df.loc[df['usummary_code'] == '211', f'{effect}'] = 0
        
        # Drop "Other" category if requested (usummary_code == 'Other')
        if drop_other_flag:
            df.loc[df['usummary_code'] == 'Other', 'impVal'] = 0
            df.loc[df['usummary_code'] == 'Other', f'{effect}'] = 0
        
        # Recalculate shares after dropping categories
        if drop_oil_flag or drop_other_flag:
            # Recalculate import shares
            total_imports = df['impVal'].sum()
            if total_imports > 0:
                df['impVal_share'] = df['impVal'] / total_imports
            else:
                df['impVal_share'] = 0
            
            # Recalculate effect shares
            total_effects = df[f'{effect}'].sum()
            if total_effects > 0:
                df[f'{effect}_share'] = df[f'{effect}'] / total_effects
            else:
                df[f'{effect}_share'] = 0
        
        # Aggregate duplicate commodities by summing their raw values first
        df_agg = df.groupby('usummary_desc').agg({
            'impVal': 'sum',
            f'{effect}': 'sum'
        }).reset_index()
        
        # Recalculate shares after aggregation
        total_imports_agg = df_agg['impVal'].sum()
        total_effects_agg = df_agg[f'{effect}'].sum()
        
        if total_imports_agg > 0:
            df_agg['impVal_share'] = df_agg['impVal'] / total_imports_agg
        else:
            df_agg['impVal_share'] = 0
            
        if total_effects_agg > 0:
            df_agg[f'{effect}_share'] = df_agg[f'{effect}'] / total_effects_agg
        else:
            df_agg[f'{effect}_share'] = 0
        
        # Filter to only commodities with positive import shares
        df_agg = df_agg[(df_agg['impVal_share'] > 0)]
        
        import_values = df_agg[['usummary_desc','impVal_share']]
        effect_values = df_agg[['usummary_desc', f'{effect}_share']]
        
        fig, ax = plt.subplots(figsize=(16,8))
        ax.barh(import_values['usummary_desc'], import_values['impVal_share'], color='#3581b4', alpha=0.3, label='Imports Share')
        ax.barh(effect_values['usummary_desc'], -effect_values[f'{effect}_share'], color='#cd590c', alpha=0.7, label=f'{effect.capitalize()} Share')
        
        # Add text labels positioned just inside the y-axis
        for i, country_name in enumerate(import_values['usummary_desc']):
            ax.text(0, i, country_name, ha='center', va='center', fontsize=16, alpha=0.75)

        # Hide y-axis labels since we're showing them as text on bars
        ax.set_yticklabels([])
        
        # Remove spines and reduce margins
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        ax.tick_params(left=False, right=False)
        plt.tight_layout(pad=0.1)
        
        # Save plot
        dir = f'Calculations/validations/06_effect_validation_cont/{effect}/country/{country}'
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, f'{effect}_imports_vs_effects{suffix}.png'))
        plt.close()
    
    # Create all four versions
    create_plot(False, False, '_complete')
    create_plot(True, False, '_no_oil') 
    create_plot(False, True, '_no_other')
    create_plot(True, True, '_no_oil_no_other')
    
    # Save CSV with aggregated data and no-oil import shares
    dir = f'Calculations/validations/06_effect_validation_cont/{effect}/country/{country}'
    df_csv = combined_data[(country,'TEN')][['iso3','usummary_code','usummary_desc','impVal','impVal_share',f'{effect}',f'{effect}_share']].copy()
    
    # Fill NaN values
    df_csv[f'{effect}_share'] = df_csv[f'{effect}_share'].fillna(0)
    df_csv['impVal_share'] = df_csv['impVal_share'].fillna(0)
    df_csv['impVal'] = df_csv['impVal'].fillna(0)
    df_csv[f'{effect}'] = df_csv[f'{effect}'].fillna(0)
    df_csv['usummary_desc'] = df_csv['usummary_desc'].fillna('Unknown')
    
    # Aggregate duplicate commodities by commodity description and code
    df_csv_agg = df_csv.groupby(['usummary_code', 'usummary_desc']).agg({
        'iso3': 'first',  # Keep the country code
        'impVal': 'sum',
        f'{effect}': 'sum'
    }).reset_index()
    
    # Recalculate shares after aggregation
    total_imports_agg = df_csv_agg['impVal'].sum()
    total_effects_agg = df_csv_agg[f'{effect}'].sum()
    
    if total_imports_agg > 0:
        df_csv_agg['impVal_share'] = df_csv_agg['impVal'] / total_imports_agg
    else:
        df_csv_agg['impVal_share'] = 0
        
    if total_effects_agg > 0:
        df_csv_agg[f'{effect}_share'] = df_csv_agg[f'{effect}'] / total_effects_agg
    else:
        df_csv_agg[f'{effect}_share'] = 0
    
    # Calculate import shares without oil
    df_csv_no_oil = df_csv_agg.copy()
    df_csv_no_oil.loc[df_csv_no_oil['usummary_code'] == '211', 'impVal'] = 0
    total_imports_no_oil = df_csv_no_oil['impVal'].sum()
    if total_imports_no_oil > 0:
        df_csv_agg['impVal_share_no_oil'] = df_csv_no_oil['impVal'] / total_imports_no_oil
    else:
        df_csv_agg['impVal_share_no_oil'] = 0
    
    # Reorder columns to match original format
    df_csv_agg = df_csv_agg[['iso3','usummary_code','usummary_desc','impVal','impVal_share',f'{effect}',f'{effect}_share','impVal_share_no_oil']]
    
    df_csv_agg.to_csv(os.path.join(dir,f'{effect}_imports_vs_effects.csv'), index=False)
    
    # Create LaTeX file for four-panel figure
    country_name = coco.convert(country, to='name_short', not_found=country)
    dir = f'Calculations/validations/06_effect_validation_cont/{effect}/country/{country}'
    tex_file_path = os.path.join(dir, f'{effect}_imports_vs_effects.tex')
    with open(tex_file_path, 'w') as tex_file:
        tex_file.write('\\begin{figure}[htbp]\n')
        tex_file.write(f'\\caption{{{country_name} Imports vs. {effect.capitalize()} Effects Shares}}\n')
        tex_file.write(f'\\label{{fig:{country.lower()}_{effect}_imports_vs_effects}}\n')
        tex_file.write('\\centering\n')
        
        # First row
        tex_file.write('\\begin{subfigure}[b]{0.48\\textwidth}\n')
        tex_file.write('\\centering\n')
        tex_file.write(f'\\includegraphics[width=\\textwidth]{{06_effect_validation_cont/{effect}/country/{country}/{effect}_imports_vs_effects_complete.png}}\n')
        tex_file.write('\\caption{Complete}\n')
        tex_file.write(f'\\label{{fig:{country.lower()}_{effect}_complete}}\n')
        tex_file.write('\\end{subfigure}\n')
        tex_file.write('\\hfill\n')
        tex_file.write('\\begin{subfigure}[b]{0.48\\textwidth}\n')
        tex_file.write('\\centering\n')
        tex_file.write(f'\\includegraphics[width=\\textwidth]{{06_effect_validation_cont/{effect}/country/{country}/{effect}_imports_vs_effects_no_oil.png}}\n')
        tex_file.write('\\caption{No Oil}\n')
        tex_file.write(f'\\label{{fig:{country.lower()}_{effect}_no_oil}}\n')
        tex_file.write('\\end{subfigure}\n')
        
        # Second row
        tex_file.write('\\\\[1em]\n')
        tex_file.write('\\begin{subfigure}[b]{0.48\\textwidth}\n')
        tex_file.write('\\centering\n')
        tex_file.write(f'\\includegraphics[width=\\textwidth]{{06_effect_validation_cont/{effect}/country/{country}/{effect}_imports_vs_effects_no_other.png}}\n')
        tex_file.write('\\caption{No Other}\n')
        tex_file.write(f'\\label{{fig:{country.lower()}_{effect}_no_other}}\n')
        tex_file.write('\\end{subfigure}\n')
        tex_file.write('\\hfill\n')
        tex_file.write('\\begin{subfigure}[b]{0.48\\textwidth}\n')
        tex_file.write('\\centering\n')
        tex_file.write(f'\\includegraphics[width=\\textwidth]{{06_effect_validation_cont/{effect}/country/{country}/{effect}_imports_vs_effects_no_oil_no_other.png}}\n')
        tex_file.write('\\caption{No Oil, No Other}\n')
        tex_file.write(f'\\label{{fig:{country.lower()}_{effect}_no_oil_no_other}}\n')
        tex_file.write('\\end{subfigure}\n')
        
        tex_file.write('\\end{figure}\n')
    return 

# Evaulate each country in the bottom bins (or just each country)
def eval_each_country(bottom_bins, effect= 'direct'):
    
    for country in bottom_bins['iso3'].unique():
        print(f"Evaluating {country} for {effect} effects...")
        evaluate_imports_and_effects(country, effect)

    # Create a master LaTeX file that includes all country .tex files
    master_dir = f'Calculations/validations/06_effect_validation_cont/{effect}'
    os.makedirs(master_dir, exist_ok=True)
    
    master_tex_path = os.path.join(master_dir, f'all_countries_{effect}_analysis.tex')
    
    with open(master_tex_path, 'w') as master_file:
        master_file.write(f'% Master file for all {effect} effect country analyses\n')
        master_file.write(f'% Generated automatically from country-specific .tex files\n\n')
        
        for country in sorted(bottom_bins['iso3'].unique()):
            country_name = coco.convert(country, to='name_short', not_found=country)
            master_file.write(f'% {country_name} ({country})\n')
            master_file.write(f'\\input{{06_effect_validation_cont/{effect}/country/{country}/{effect}_imports_vs_effects.tex}}\n')
            master_file.write('\\clearpage\n\n')
    
    print(f"Created master LaTeX file: {master_tex_path}")

def analyze_commodity_differences(bottom_bins, effect='direct'):
    """
    Analyzes the average difference between import shares and effect shares 
    (no oil, no other) across all countries for each commodity
    """
    commodity_diffs = {}
    
    for country in bottom_bins['iso3'].unique():
        df = combined_data[(country,'TEN')][['iso3','usummary_code','usummary_desc','impVal','impVal_share',f'{effect}',f'{effect}_share']].copy()
        
        # Fill NaN values
        df[f'{effect}_share'] = df[f'{effect}_share'].fillna(0)
        df['impVal_share'] = df['impVal_share'].fillna(0)
        df['impVal'] = df['impVal'].fillna(0)
        df[f'{effect}'] = df[f'{effect}'].fillna(0)
        df['usummary_desc'] = df['usummary_desc'].fillna('Unknown')
        
        # Remove oil and other
        df.loc[df['usummary_code'] == '211', 'impVal'] = 0
        df.loc[df['usummary_code'] == '211', f'{effect}'] = 0
        df.loc[df['usummary_code'] == 'Other', 'impVal'] = 0
        df.loc[df['usummary_code'] == 'Other', f'{effect}'] = 0
        
        # Recalculate shares
        total_imports = df['impVal'].sum()
        total_effects = df[f'{effect}'].sum()
        
        if total_imports > 0:
            df['impVal_share_adj'] = df['impVal'] / total_imports
        else:
            df['impVal_share_adj'] = 0
            
        if total_effects > 0:
            df[f'{effect}_share_adj'] = df[f'{effect}'] / total_effects
        else:
            df[f'{effect}_share_adj'] = 0
        
        # Calculate differences for each commodity
        for _, row in df.iterrows():
            code = row['usummary_code']
            desc = row['usummary_desc']
            
            if code not in ['211', 'Other'] and (row['impVal_share_adj'] > 0 or row[f'{effect}_share_adj'] > 0):
                diff = row[f'{effect}_share_adj'] - row['impVal_share_adj']
                
                if code not in commodity_diffs:
                    commodity_diffs[code] = {
                        'usummary_desc': desc,
                        'differences': [],
                        'import_shares': [],
                        'effect_shares': []
                    }
                
                commodity_diffs[code]['differences'].append(diff)
                commodity_diffs[code]['import_shares'].append(row['impVal_share_adj'])
                commodity_diffs[code]['effect_shares'].append(row[f'{effect}_share_adj'])
    
    # Calculate averages
    results = []
    for code, data in commodity_diffs.items():
        if len(data['differences']) > 0:
            avg_diff = np.mean(data['differences'])
            avg_import_share = np.mean(data['import_shares'])
            avg_effect_share = np.mean(data['effect_shares'])
            
            results.append({
                'usummary_code': code,
                'usummary_desc': data['usummary_desc'],
                'avg_effect_minus_import_share': avg_diff,
                'avg_import_share': avg_import_share,
                'avg_effect_share': avg_effect_share,
                'n_countries': len(data['differences'])
            })
    
    # Convert to DataFrame and sort by difference
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('avg_effect_minus_import_share', ascending=False)
    
    # Save to CSV
    output_dir = f'Calculations/validations/06_effect_validation_cont/{effect}'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'commodity_share_differences_{effect}.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"Created commodity differences analysis: {csv_path}")
    return results_df

eval_each_country(bottom_bins_direct, effect='direct')
# eval_each_country(bottom_bins_total, effect='total')
# eval_each_country(bottom_bins_indirect, effect='indirect')

# Create commodity difference analysis for direct effects
analyze_commodity_differences(bottom_bins_direct, effect='direct')

# Case study: Create analysis for specific countries (ISR and GBR)
def create_case_study_analysis(countries, effect='direct'):
    """
    Creates the same analysis outputs for specific countries as a case study.
    
    Args:
        countries (list): List of ISO3 country codes to analyze
        effect (str): The type of effect to evaluate ('direct', 'indirect', 'total')
    """
    
    # Create case study directory
    case_study_dir = f'Calculations/validations/06_effect_validation_cont/case_study/{effect}'
    os.makedirs(case_study_dir, exist_ok=True)
    
    for country in countries:
        print(f"Creating case study analysis for {country} ({effect} effects)...")
        
        # Create country-specific directory
        country_dir = os.path.join(case_study_dir, 'country', country)
        os.makedirs(country_dir, exist_ok=True)
        
        # Generate the same outputs as the regular analysis
        evaluate_imports_and_effects(country, effect)
        
        # Move the generated files to case study directory
        source_dir = f'Calculations/validations/06_effect_validation_cont/{effect}/country/{country}'
        if os.path.exists(source_dir):
            import shutil
            for file in os.listdir(source_dir):
                source_file = os.path.join(source_dir, file)
                dest_file = os.path.join(country_dir, file)
                shutil.copy2(source_file, dest_file)
    
    # Create master LaTeX file for case study
    master_tex_path = os.path.join(case_study_dir, f'case_study_{effect}_analysis.tex')
    
    with open(master_tex_path, 'w') as master_file:
        master_file.write(f'% Case Study: Israel and United Kingdom {effect.capitalize()} Effects Analysis\\n')
        master_file.write(f'% Generated automatically from country-specific .tex files\\n\\n')
        
        for country in sorted(countries):
            country_name = coco.convert(country, to='name_short', not_found=country)
            master_file.write(f'% {country_name} ({country})\\n')
            master_file.write(f'\\\\input{{06_effect_validation_cont/case_study/{effect}/country/{country}/{effect}_imports_vs_effects.tex}}\\n')
            master_file.write('\\\\clearpage\\n\\n')
    
    print(f"Created case study master LaTeX file: {master_tex_path}")

# Run case study for Israel (ISR) and United Kingdom (GBR)
case_study_countries = ['ISR', 'GBR']
create_case_study_analysis(case_study_countries, effect='direct')