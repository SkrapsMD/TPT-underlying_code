import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import mantel
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels to Calculations/
data_paths_file = os.path.join(project_dir, "data_paths.json")
with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Set up clean directory variables using project_root from config
project_root = data_paths['base_paths']['project_root']
raw_data_dir = os.path.join(project_root, data_paths['base_paths']['raw_data'])
working_data_dir = os.path.join(project_root, data_paths['base_paths']['working_data'])
calculations_dir = os.path.join(working_data_dir, "Components for Calculations")
validations_dir = os.path.join(project_root, data_paths['base_paths']['validations'])
hs_to_bea_data_dir = os.path.join(project_root, data_paths['base_paths']['hs_to_bea_data'])
figures_dir = os.path.join(validations_dir,'04_main_calculations')
final_data_dir = os.path.join(project_root, data_paths['base_paths']['final_data'])

# TWT Scenario
TWT_direct = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/TWT Data/BEA_direct_effects.csv'), index_col = False)
TWT_indirect = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/TWT Data/BEA_indirect_effects.csv'), index_col = False)
TWT_total = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/TWT Data/BEA_total_effects.csv'), index_col = False)

# Add ranking columns for TWT effects
def add_rankings(df, effect_type):
    country_cols = [col for col in df.columns if col not in ['BEA_Industry','BEA_Code', 'BEA_Description']]
    
    # Global rankings - rank all values across all countries
    all_values = df[country_cols].values.flatten()
    global_ranks = pd.Series(all_values).rank(method='dense', ascending=False)
    
    # Create all ranking columns at once to avoid fragmentation
    global_rank_dict = {}
    country_rank_dict = {}
    
    start_idx = 0
    for col in country_cols:
        end_idx = start_idx + len(df)
        global_rank_dict[f'{col}_{effect_type}_global_rank'] = global_ranks.iloc[start_idx:end_idx].values
        country_rank_dict[f'{col}_{effect_type}_country_rank'] = df[col].rank(method='dense', ascending=False)
        start_idx = end_idx
    
    # Add all ranking columns at once using pd.concat
    global_rank_df = pd.DataFrame(global_rank_dict)
    country_rank_df = pd.DataFrame(country_rank_dict)
    
    return pd.concat([df, global_rank_df, country_rank_df], axis=1)

TWT_direct = add_rankings(TWT_direct.copy(), 'direct')
TWT_indirect = add_rankings(TWT_indirect.copy(), 'indirect') 
TWT_total = add_rankings(TWT_total.copy(), 'total')
TWT_effects = {}
# Get all country columns (excluding BEA_Code, BEA_Description, and ranking columns)
country_cols = [col for col in TWT_total.columns if col not in ['BEA_Industry','BEA_Code', 'BEA_Description'] and not col.endswith('_rank')]
for country in country_cols:
    country_key = 'All Countries' if country == 'All Countries Effect' else country
    country_df = pd.DataFrame({
        'usummary_code': TWT_total['BEA_Code'],
        'iso3': country,
        'total': TWT_total[country],
        'indirect': TWT_indirect[country],
        'direct': TWT_direct[country],
        'direct_share': TWT_direct[country] / TWT_direct[country].sum(),
        'indirect_share': TWT_indirect[country] / TWT_indirect[country].sum(),
        'total_share': TWT_total[country] / TWT_total[country].sum(),
        'total_global_rank': TWT_total[f'{country}_total_global_rank'],
        'total_country_rank': TWT_total[f'{country}_total_country_rank'],
        'indirect_global_rank': TWT_indirect[f'{country}_indirect_global_rank'],
        'indirect_country_rank': TWT_indirect[f'{country}_indirect_country_rank'],
        'direct_global_rank': TWT_direct[f'{country}_direct_global_rank'],
        'direct_country_rank': TWT_direct[f'{country}_direct_country_rank']
    })
    TWT_effects[country_key] = country_df
# 10% Scenario
TEN_direct = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/Constant 10%/BEA_direct_effects.csv'), index_col = False)
TEN_indirect = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/Constant 10%/BEA_indirect_effects.csv'), index_col = False)
TEN_total = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/Constant 10%/BEA_total_effects.csv'), index_col = False)

# Add ranking columns for TEN effects
TEN_direct = add_rankings(TEN_direct.copy(), 'direct')
TEN_indirect = add_rankings(TEN_indirect.copy(), 'indirect')
TEN_total = add_rankings(TEN_total.copy(), 'total')
TEN_effects = {}
# Get all country columns (excluding BEA_Code, BEA_Description, and ranking columns)
country_cols = [col for col in TEN_total.columns if col not in ['BEA_Industry','BEA_Code', 'BEA_Description'] and not col.endswith('_rank')]
for country in country_cols:
    # Rename "All Countries Effect" to "All Countries"
    country_key = "All Countries" if country == "All Countries Effect" else country
    country_df_ten = pd.DataFrame({
        'usummary_code': TEN_total['BEA_Code'],
        'iso3': country,
        'total': TEN_total[country],
        'indirect': TEN_indirect[country],
        'direct': TEN_direct[country],
        'direct_share': TEN_direct[country] / TEN_direct[country].sum(),
        'indirect_share': TEN_indirect[country] / TEN_indirect[country].sum(),
        'total_share': TEN_total[country] / TEN_total[country].sum(),
        'total_global_rank': TEN_total[f'{country}_total_global_rank'],
        'total_country_rank': TEN_total[f'{country}_total_country_rank'],
        'indirect_global_rank': TEN_indirect[f'{country}_indirect_global_rank'],
        'indirect_country_rank': TEN_indirect[f'{country}_indirect_country_rank'],
        'direct_global_rank': TEN_direct[f'{country}_direct_global_rank'],
        'direct_country_rank': TEN_direct[f'{country}_direct_country_rank']
    })
    TEN_effects[country_key] = country_df_ten

# Import Data -- turn into a dictionary of dataframes by country so we can access them as import_country[country]
import_data = pd.read_csv(os.path.join(hs_to_bea_data_dir, 'data', 'working', '04_Aggregate_BEA_and_HS', 'aggregated_data', 'country_usummary', 'all_continents_usummary.csv'))
import_data = import_data[['iso3','usummary_code','impVal']]
import_data_all_countries = import_data.groupby('usummary_code')['impVal'].sum().reset_index()
import_data_all_countries['usummary_code'] = import_data_all_countries['usummary_code'].astype(str)
import_data_all_countries['iso3'] = 'All Countries'
import_data = pd.concat([import_data, import_data_all_countries], ignore_index=True)

# Add import rankings
import_data['impVal_global_rank'] = import_data['impVal'].rank(method='dense', ascending=False)
import_data['impVal_country_rank'] = import_data.groupby('iso3')['impVal'].rank(method='dense', ascending=False)

# Create import shares within country
import_data['impVal_share'] = import_data.groupby('iso3')['impVal'].transform(lambda x: x / x.sum())

import_country = {country: df for country, df in import_data.groupby('iso3')}

"""This script measures the degree of similarity between the import values and the three different effect values (direct, indirect, and total). 

Using cosine simialrity, first it creates vectors of import vlaues and effect values, calclates cosine simialrity, and this by country. We then 
want to try and cluster to see if countries with similar import patterns have similar effect values (k-means clustering?)
"""

# 1) Create datasets with combined effects and imports from the datasets from the TWT or TEN with the import data for a given country. 
def combine_data(country, scenario = 'TWT'):
    """
    Description: Combines import data with effect data for a given country and scenario
    
    Args:
        country (str): The ISO3 code of the country to combine data for
        scenario (str): The scenario to use ('TWT' or 'TEN')
    Returns: 
        dataframe with the columns for the direct, indirect,a nd total effects, the usummary code"""
    if scenario =='TWT':
        return pd.merge(import_country[country], TWT_effects[country], on = ['usummary_code', 'iso3'], how='outer')
    else:
        return pd.merge(import_country[country], TEN_effects[country], on = ['usummary_code', 'iso3'], how='outer')
# Construct combined data
combined_data = {}
for scenario in ['TWT', 'TEN']: 
    effects_dict = TWT_effects if scenario == 'TWT' else TEN_effects
    for country in effects_dict.keys():
        combined_data[(country, scenario)] = combine_data(country, scenario)

"""
Variables in combined_data[(country, scenario)]

iso3 -  country 
usummary_code - BEA code for the industry/commodity 
impVal - import value for the country and industry/commodity
impVal_global_rank - global rank of the import value (i.e. US import of 111 from Great Britain is the XXXth largest import category in the world)
impVal_country_rank - country rank of the import value. 

total - total effect for the country and industry/commodity
total_global_rank - global rank of the total effect
total_country_rank - country rank of the total effect

indirect - indirect effect for the country and industry/commodity
indirect_global_rank - global rank of the indirect effect
indirect_country_rank - country rank of the indirect effect

direct - direct effect for the country and industry/commodity
direct_global_rank - global rank of the direct effect
direct_country_rank - country rank of the direct effect

There are two things we kind of care about: does the ranking of the import values match the ranking of the effect? Do countries with 
similar import vectors have similar effect vectors (K-Means Clustering).

For the cosine similarity, we will produce vectors of the differences between import values for country a and import values for every other country, and do the same for the effects. 
Ideally, we would like to see that the cosine similarity between the import vector and the effect vector is high. 
"""
default_scenario ='TEN'

# 1 - Within country sanity -- Does a large import value lead to a large effect?
def within_country_cos_check(scenario = default_scenario):
    """
    Description: Checks the cosine similarity between the import shares and the effect shares for each country in the specified scenario.
    
    Goal -- if everything is working right, the cosine similarity between the import shares and the direct (at least) effect should be high
    
    Args:
        scenario (str): The scenario to use ('TWT' or 'TEN')
        
    Returns:
        DataFrame with cosine similarity results for each country
            
    """
    results = []
    top_industries = []
    
    for (country, scen), data in combined_data.items():
        if scen != scenario:
            continue
            
        # Fill NaN values with 0
        total_effects = data['total_share'].fillna(0).values
        direct_effects = data['direct_share'].fillna(0).values
        indirect_effects = data['indirect_share'].fillna(0).values
        import_values = data['impVal_share'].fillna(0).values
        
        # Calculate cosine similarity
        if len(import_values) > 0 and len(total_effects) > 0:
            total_sim = cosine_similarity([import_values], [total_effects])[0][0]
            direct_sim = cosine_similarity([import_values], [direct_effects])[0][0]
            indirect_sim = cosine_similarity([import_values], [indirect_effects])[0][0]
            results.append({'iso3': country, 'scenario': scenario, 'total_sim': total_sim, 'direct_sim': direct_sim, 'indirect_sim': indirect_sim})
            
            # Get top 5 industries for each effect type
            data_clean = data.dropna(subset=['usummary_code'])
            
            # Top 5 direct effects
            top_direct = data_clean.nlargest(5, 'direct')[['usummary_code', 'direct']].to_dict('records')
            # Top 5 indirect effects  
            top_indirect = data_clean.nlargest(5, 'indirect')[['usummary_code', 'indirect']].to_dict('records')
            # Top 5 total effects
            top_total = data_clean.nlargest(5, 'total')[['usummary_code', 'total']].to_dict('records')
            
            top_industries.append({
                'iso3': country,
                'scenario': scenario,
                'top_direct': top_direct,
                'top_indirect': top_indirect,
                'top_total': top_total
            })
            
    results.sort(key=lambda x: x['direct_sim'], reverse=False)
    
    # Print top industries for each country
    print(f"\nTop 5 Industries by Effect Type ({scenario} scenario):")
    print("="*60)
    
    for country_data in top_industries:
        country = country_data['iso3']
        print(f"\n{country}:")
        print("-" * 30)
        
        print("Top 5 Direct Effects:")
        for i, industry in enumerate(country_data['top_direct'], 1):
            print(f"  {i}. {industry['usummary_code']}: {industry['direct']:.6f}")
            
        print("\nTop 5 Indirect Effects:")
        for i, industry in enumerate(country_data['top_indirect'], 1):
            print(f"  {i}. {industry['usummary_code']}: {industry['indirect']:.6f}")
            
        print("\nTop 5 Total Effects:")
        for i, industry in enumerate(country_data['top_total'], 1):
            print(f"  {i}. {industry['usummary_code']}: {industry['total']:.6f}")
    
    return pd.DataFrame(results)
within_country_cos_check().to_csv(os.path.join(validations_dir, '05_cos_validation', f'within_country_cosine_similarity_{default_scenario}.csv'), index=False)

# 2 - Cross country similarity validation: if two countries have similar baskets do they see similar effects?
"""This one uses a distance matrix between countries, and then I run a Mantel test"""
def cross_country_cos_check(scenario='TEN'):
    
    # Create DataFrames for each metric - countries as columns, BEA codes as rows
    metric_dataframes = {}
    
    for metric_name in ['impVal_share','direct_share','indirect_share','total_share']:
        metric_data = {}
        
        # Get all countries for this scenario
        countries = [country for (country, scen) in combined_data.keys() if scen == scenario]
        
        for country in countries:
            if (country, scenario) in combined_data:
                data = combined_data[(country, scenario)]
                # Check for and handle duplicate usummary_codes
                if data['usummary_code'].duplicated().any():
                    # Aggregate duplicates by summing (or you could use mean)
                    data = data.groupby('usummary_code')[metric_name].sum().reset_index()
                    country_data = data.set_index('usummary_code')[metric_name]
                else:
                    country_data = data.set_index('usummary_code')[metric_name]
                metric_data[country] = country_data
        
        # Create DataFrame with countries as columns
        metric_dataframes[metric_name] = pd.DataFrame(metric_data).fillna(0)
    # Check if all matrices have same BEA codes in same order
    import_index = metric_dataframes['impVal_share'].index
    for effect_type in ['direct_share', 'indirect_share', 'total_share']:
        effect_index = metric_dataframes[effect_type].index
    # Create distance matrices using import values vs each effect type
    import_matrix = metric_dataframes['impVal_share'].T  # Countries as rows
    results = {}
    for effect_type in ['direct_share', 'indirect_share', 'total_share']:
        effect_matrix = metric_dataframes[effect_type].T  # Countries as rows
        
        # Filter out BEA codes where the effect value is 0
        non_zero_mask = (metric_dataframes[effect_type] != 0).any(axis=1)
        filtered_import = metric_dataframes['impVal_share'].loc[non_zero_mask].T
        filtered_effect = metric_dataframes[effect_type].loc[non_zero_mask].T
        
        # Create distance matrices
        import_dist = squareform(pdist(filtered_import, metric='cosine'))
        effect_dist = squareform(pdist(filtered_effect, metric='cosine'))

        r, p, _ = mantel(import_dist, effect_dist, method='pearson', permutations=999)
        print(f"Mantel r = {r:.3f},  p = {p:.4f}")

        results[effect_type] = {
            'import_dist': import_dist,
            'effect_dist': effect_dist,
            'countries': list(import_matrix.index),
            'mantel_r': r,
            'mantel_p': p
        }
        
        # Create scatter plot and save
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        # Scatter plot with 45-degree line and confidence bands
        import_flat = import_dist[np.triu_indices_from(import_dist, k=1)]
        effect_flat = effect_dist[np.triu_indices_from(effect_dist, k=1)]
        ax.scatter(import_flat, effect_flat, alpha=0.6, s=10)
        # Calculate residuals from 45-degree line and standard deviation
        residuals = effect_flat - import_flat
        std_residual = np.std(residuals)
        max_val = max(import_flat.max(), effect_flat.max())
        x_line = np.linspace(0, max_val, 100)
        # Plot 45-degree line
        ax.plot(x_line, x_line, 'r--', label='Perfect Equality')
        # Plot confidence bands (±1 and ±2 standard deviations)
        ax.fill_between(x_line, x_line - std_residual, x_line + std_residual, 
                       alpha=0.2, color='red', label=f'±1σ ({std_residual:.3f})')
        ax.fill_between(x_line, x_line - 2*std_residual, x_line + 2*std_residual, 
                       alpha=0.1, color='red', label=f'±2σ ({2*std_residual:.3f})')
        ax.set_xlabel('Import Distance')
        ax.set_ylabel(f'{effect_type.title()} Distance')
        ax.set_title(f'Import vs {effect_type.title()} Distance')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        # Save to validations directory
        save_dir = os.path.join(validations_dir, '05_cos_validation')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'import_vs_{effect_type}_distance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return results, metric_dataframes

cross_country_cos_check()


def cluster_by_imports_and_effects(scenario='TEN', n_clusters=3):
    """
    Description: Clusters countries based on their import and effect vectors using KMeans clustering.
    
    Args:
        scenario (str): The scenario to use ('TWT' or 'TEN')
        n_clusters (int): The number of clusters to form
    Returns:
        DataFrame with cluster assignments for each country
    """
    
    # Get countries for this scenario
    countries = [country for (country, scen) in combined_data.keys() if scen == scenario]
    
    # Prepare data for clustering - collect all metrics
    clustering_data = {}
    
    for country in countries:
        if (country, scenario) in combined_data:
            data = combined_data[(country, scenario)]
            # Remove duplicates if any
            if data['usummary_code'].duplicated().any():
                data = data.groupby('usummary_code', as_index=False).first()
            
            clustering_data[country] = {
                # Import shares and rankings
                'impVal_share': data['impVal_share'].values,
                'impVal_global_rank': data['impVal_global_rank'].values,
                'impVal_country_rank': data['impVal_country_rank'].values,
                'impVal': data['impVal'].values,  # Raw import values
                
                # Effect shares
                'direct_share': data['direct_share'].values,
                'indirect_share': data['indirect_share'].values, 
                'total_share': data['total_share'].values,
                
                # Raw effect values
                'direct': data['direct'].values,
                'indirect': data['indirect'].values,
                'total': data['total'].values,
                
                # Effect global rankings
                'direct_global_rank': data['direct_global_rank'].values,
                'indirect_global_rank': data['indirect_global_rank'].values,
                'total_global_rank': data['total_global_rank'].values,
                
                # Effect country rankings
                'direct_country_rank': data['direct_country_rank'].values,
                'indirect_country_rank': data['indirect_country_rank'].values,
                'total_country_rank': data['total_country_rank'].values
            }
    
    # Perform clustering for each metric type
    cluster_results = pd.DataFrame({'country': countries})
    
    for metric_name in ['impVal_share', 'impVal_global_rank', 'impVal_country_rank', 'impVal', 'direct_share', 'indirect_share', 
                       'total_share', 'direct', 'indirect', 'total', 'direct_global_rank', 'indirect_global_rank', 'total_global_rank',
                       'direct_country_rank', 'indirect_country_rank', 'total_country_rank']:
        
        # Create matrix with countries as rows, BEA codes as columns
        metric_matrix = []
        valid_countries = []
        
        for country in countries:
            if country in clustering_data:
                metric_vector = clustering_data[country][metric_name]
                # Handle NaN values
                metric_vector = np.nan_to_num(metric_vector, nan=0.0)
                metric_matrix.append(metric_vector)
                valid_countries.append(country)
        
        if len(metric_matrix) > 0:
            metric_matrix = np.array(metric_matrix)
            
            # Standardize the data before clustering
            scaler = StandardScaler()
            metric_matrix_scaled = scaler.fit_transform(metric_matrix)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(valid_countries)), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(metric_matrix_scaled)
            
            # Add cluster assignments to results
            cluster_df = pd.DataFrame({
                'country': valid_countries,
                f'{metric_name}_cluster': cluster_labels
            })
            
            cluster_results = cluster_results.merge(cluster_df, on='country', how='left')
            
            print(f"Clustered {len(valid_countries)} countries for {metric_name}")
    
    return cluster_results

def compare_import_vs_effect_clusters(cluster_results, scenario='TEN'):
    """
    Description: Creates visualizations comparing import-based clusters with effect-based clusters.
    
    Args:
        cluster_results (DataFrame): Results from cluster_by_imports_and_effects function
        scenario (str): The scenario used for labeling
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Effect types to compare against imports
    effect_types = ['direct', 'indirect', 'total']
    
    # Initialize LaTeX output
    save_dir = os.path.join(validations_dir, '05_cos_validation')
    os.makedirs(save_dir, exist_ok=True)
    tex_file_path = os.path.join(save_dir, f'cluster_concentration_analysis_{scenario}.tex')
    
    with open(tex_file_path, 'w') as tex_file:
        
        # Collect data for all effect types first
        all_concentrations = {}
        
        for effect_type in effect_types:
            # Create figure with single panel for share comparison only
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Compare share-based clusters
            share_data = cluster_results[['country', 'impVal_share_cluster', f'{effect_type}_share_cluster']].dropna()
            if len(share_data) > 0:
                # Create confusion matrix / cross-tabulation
                crosstab_share = pd.crosstab(share_data['impVal_share_cluster'], 
                                           share_data[f'{effect_type}_share_cluster'], 
                                           margins=True)
                
                # Plot as heatmap
                sns.heatmap(crosstab_share.iloc[:-1, :-1], annot=True, fmt='d', ax=ax, cmap='Blues')
                ax.set_title(f'Import Share vs {effect_type.title()} Share Clusters')
                ax.set_xlabel(f'{effect_type.title()} Share Cluster')
                ax.set_ylabel('Import Share Cluster')
                
                # Analyze concentration of blocks
                crosstab_data = crosstab_share.iloc[:-1, :-1]  # Remove margins
                
                # Calculate concentrations
                row_concentrations = []
                col_concentrations = []
                
                # Get row concentrations (import clusters)
                for i, row in crosstab_data.iterrows():
                    total_countries = row.sum()
                    if total_countries > 0:
                        max_concentration = row.max() / total_countries
                        row_concentrations.append(max_concentration)
                
                # Get column concentrations (effect clusters)
                for col_idx, col in crosstab_data.T.iterrows():
                    total_countries = col.sum()
                    if total_countries > 0:
                        max_concentration = col.max() / total_countries
                        col_concentrations.append(max_concentration)
                
                # Store concentrations for later use
                all_concentrations[effect_type] = {
                    'row_concentrations': row_concentrations,
                    'col_concentrations': col_concentrations,
                    'avg_row': np.mean(row_concentrations) if row_concentrations else 0,
                    'avg_col': np.mean(col_concentrations) if col_concentrations else 0
                }
                
                # Print console output
                print(f"\n{effect_type.title()} Effect Cluster Comparison:")
                print("Share-based clusters:")
                print(crosstab_share)
                print(f"\nConcentration Analysis for {effect_type.title()} Effects:")
                for i, conc in enumerate(row_concentrations):
                    print(f"Import cluster {i}: {conc:.2%} of countries go to dominant effect cluster")
                print(f"Average row concentration: {all_concentrations[effect_type]['avg_row']:.2%}")
                print(f"Average column concentration: {all_concentrations[effect_type]['avg_col']:.2%}")
                is_concentrated = all_concentrations[effect_type]['avg_row'] > 0.6
                print(f"Clustering appears {'CONCENTRATED' if is_concentrated else 'DISPERSED'} (threshold: 60%)")
            
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(save_dir, f'import_vs_{effect_type}_clusters_{scenario}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Now create the combined LaTeX table
        
        # Find the maximum number of clusters across all effect types
        max_clusters = 0
        for effect_type in effect_types:
            if effect_type in all_concentrations:
                max_clusters = max(max_clusters, 
                                 len(all_concentrations[effect_type]['row_concentrations']),
                                 len(all_concentrations[effect_type]['col_concentrations']))
        
        # Create the combined table with separators using longtable
        column_spec = "l"
        for i, effect_type in enumerate(effect_types):
            if i > 0:
                column_spec += "|"  # Add separator before each new effect type
            column_spec += "cc"
        tex_file.write(f"\\begin{{longtable}}{{{column_spec}}}\n")
        tex_file.write("\\toprule\n")
        
        # Header row
        header = "Cluster"
        for effect_type in effect_types:
            header += f" & \\multicolumn{{2}}{{c}}{{{effect_type.title()}}}"
        header += " \\\\\n"
        tex_file.write(header)
        
        # Subheader row
        subheader = ""
        for effect_type in effect_types:
            subheader += " & Import$\\rightarrow$Effect & Effect$\\rightarrow$Import"
        subheader += " \\\\\n"
        tex_file.write(subheader)
        tex_file.write("\\midrule\n")
        tex_file.write("\\endfirsthead\n")
        
        # Repeat header on subsequent pages
        tex_file.write("\\toprule\n")
        tex_file.write(header)
        tex_file.write(subheader)
        tex_file.write("\\midrule\n")
        tex_file.write("\\endhead\n")
        
        # Footer for page breaks
        tex_file.write("\\midrule\n")
        tex_file.write("\\multicolumn{" + str(1 + 2*len(effect_types)) + "}{r}{Continued on next page...} \\\\\n")
        tex_file.write("\\endfoot\n")
        
        # Final footer
        tex_file.write("\\bottomrule\n")
        tex_file.write("\\endlastfoot\n")
        
        # Data rows
        for i in range(max_clusters):
            row = str(i)
            for effect_type in effect_types:
                if effect_type in all_concentrations:
                    row_concs = all_concentrations[effect_type]['row_concentrations']
                    col_concs = all_concentrations[effect_type]['col_concentrations']
                    row_val = f"{row_concs[i]*100:.1f}\\%" if i < len(row_concs) else "---"
                    col_val = f"{col_concs[i]*100:.1f}\\%" if i < len(col_concs) else "---"
                    row += f" & {row_val} & {col_val}"
                else:
                    row += " & --- & ---"
            row += " \\\\\n"
            tex_file.write(row)
        
        # Average row
        tex_file.write("\\midrule\n")
        avg_row = "Average"
        for effect_type in effect_types:
            if effect_type in all_concentrations:
                avg_row_val = f"{all_concentrations[effect_type]['avg_row']*100:.1f}\\%"
                avg_col_val = f"{all_concentrations[effect_type]['avg_col']*100:.1f}\\%"
                avg_row += f" & {avg_row_val} & {avg_col_val}"
            else:
                avg_row += " & --- & ---"
        avg_row += " \\\\\n"
        tex_file.write(avg_row)
        
        tex_file.write("\\end{longtable}\n")            

cluster_results = cluster_by_imports_and_effects(scenario=default_scenario, n_clusters=30)
compare_import_vs_effect_clusters(cluster_results, scenario=default_scenario)