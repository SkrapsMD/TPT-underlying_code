# Compare original census mappings with hierarchical matches
# Merges data from 3_Hierarchical_Matches.csv with 04_hs_2024_to_naics_2022_mapping.csv

import os
import pandas as pd
import json

# Load data paths configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
data_paths_file = os.path.join(script_dir, '..', '..', 'data_paths.json')

with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Read 3_Hierarchical_Matches.csv
hierarchical_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                'validations', '03_Map_country_trade_data', '3_Hierarchical_Matches.csv')
hierarchical_df = pd.read_csv(hierarchical_path)

# Filter for primary matches only
primary_matches = hierarchical_df[hierarchical_df['match_type'] == 'primary'].copy()

# Keep only required columns
primary_subset = primary_matches[['hs_code', 'matched_bea_detail', 'match_level', 'mapping_strength']].copy()

# Read census mapping data
census_mapping_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                  'validations', 'Alternative_Census_Mappings', 
                                  '04_hs_2024_to_naics_2022_mapping.csv')
census_df = pd.read_csv(census_mapping_path)

# Merge on hs_code (from hierarchical) with hs_code_2024 (from census)
merged_df = pd.merge(primary_subset, census_df, 
                    left_on='hs_code', right_on='hs_code_2024', 
                    how='left')

# Read BEA NAICS crosswalk
bea_naics_path = os.path.join(data_paths['base_paths']['working_data'], 
                             '02_HS_to_Naics_to_BEA', '01_BEA_naics_mapping.csv')
bea_naics_df = pd.read_csv(bea_naics_path)

# Convert NAICS codes to string for consistent merging
merged_df['naics_2024'] = merged_df['naics_2024'].astype(str)
merged_df['naics_2022_mapped'] = merged_df['naics_2022_mapped'].astype(str)
bea_naics_df['naics'] = bea_naics_df['naics'].astype(str)

def find_bea_code_hierarchical(naics_code, bea_mapping_df):
    """Find BEA code using hierarchical matching (exact, then 4-digit, 3-digit, 2-digit)"""
    if pd.isna(naics_code) or naics_code == 'nan':
        return None
        
    naics_str = str(naics_code).strip()
    
    # Try exact match first
    exact_match = bea_mapping_df[bea_mapping_df['naics'] == naics_str]
    if not exact_match.empty:
        return exact_match.iloc[0]['Code']
    
    # Try hierarchical matching from most specific to least specific
    for length in [5, 4, 3, 2]:
        if len(naics_str) >= length:
            partial_code = naics_str[:length]
            partial_match = bea_mapping_df[bea_mapping_df['naics'] == partial_code]
            if not partial_match.empty:
                return partial_match.iloc[0]['Code']
    
    return None

# Apply hierarchical matching for both NAICS columns
merged_df['bea_code_2024'] = merged_df['naics_2024'].apply(
    lambda x: find_bea_code_hierarchical(x, bea_naics_df))
merged_df['bea_code_2022'] = merged_df['naics_2022_mapped'].apply(
    lambda x: find_bea_code_hierarchical(x, bea_naics_df))

# Save combined dataset
output_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                          'validations', 'Alternative_Census_Mappings', 
                          '09_combined_hierarchical_census_mappings.csv')

merged_df.to_csv(output_path, index=False)

print(f"Created combined mapping file with {len(merged_df)} records")