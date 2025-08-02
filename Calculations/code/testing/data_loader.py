import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_combined_data():
    """
    Loads and constructs the combined_data dictionary structure used across analysis scripts.
    
    Returns:
        dict: combined_data[(country, scenario)] containing merged import and effect data
    """
    
    # Get script directory and paths
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
            'usummary_desc': TWT_total['BEA_Description'],
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
            'usummary_desc': TEN_total['BEA_Description'],
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
    
    return combined_data