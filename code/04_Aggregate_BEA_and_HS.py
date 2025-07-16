import os
import pandas as pd
import json
from main_pipeline_run import get_data_path
import country_converter as coco

"""
Description: Aggregates processed continent trade data into different analytical formats.
Creates multiple aggregation levels for BEA and HS code analysis.
"""

data_paths_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_paths.json')
with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Load processed continent data
continents_to_process = ['Asia', 'Europe', 'North America', 'South America', 'Oceana']
processed_data = {}

print("Loading processed continent data...")
processed_data_dir = os.path.join(data_paths['base_paths']['working_data'], '03_Map_country_trade_data', 'processed_data')

for continent in continents_to_process:
    continent_filename = f"{continent.replace(' ', '_')}_processed.csv"
    continent_path = os.path.join(processed_data_dir, continent_filename)
    
    if os.path.exists(continent_path):
        df = pd.read_csv(continent_path)
        # Add alpha ISO3 country code column
        df['iso3'] = coco.convert(df['Country'], to='iso3') ### NEED TO DECIDE WHAT TO DO WITH ISRAEL AND PALESTINE HERE!!!!!!!!!
        
        # Handle cases where coco.convert returns lists or None
        def clean_iso3(iso3_value):
            if isinstance(iso3_value, list):
                return iso3_value[0] if iso3_value else 'UNK'
            elif iso3_value is None or pd.isna(iso3_value):
                return 'UNK'
            else:
                return iso3_value
        
        df['iso3'] = df['iso3'].apply(clean_iso3)
        
        # Print any countries that couldn't be converted
        unknown_countries = df[df['iso3'] == 'UNK']['Country'].unique()
        if len(unknown_countries) > 0:
            print(f"  Warning: Countries with unknown ISO3 codes in {continent}: {unknown_countries}")
        
        processed_data[continent] = df
        print(f"Loaded {continent}: {len(df)} rows")
    else:
        print(f"Warning: {continent_path} not found")

# Create validation directory
validation_dir = os.path.join(data_paths['base_paths']['validations'], '04_Aggregate_BEA_and_HS')
os.makedirs(validation_dir, exist_ok=True)

# Load BEA hierarchy mapping
bea_hierarchy_path = os.path.join(data_paths['base_paths']['working_data'], '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv')
bea_hierarchy = pd.read_csv(bea_hierarchy_path)

# Create mapping dictionaries with trimmed values
detail_to_usummary = dict(zip(bea_hierarchy['Detail'].str.strip(), bea_hierarchy['U.Summary'].str.strip()))
detail_to_summary = dict(zip(bea_hierarchy['Detail'].str.strip(), bea_hierarchy['Summary'].str.strip()))
detail_to_sector = dict(zip(bea_hierarchy['Detail'].str.strip(), bea_hierarchy['Sector'].str.strip()))

# Create output directories
base_output_dir = os.path.join(data_paths['base_paths']['working_data'], '04_Aggregate_BEA_and_HS', 'aggregated_data')
detail_dir = os.path.join(base_output_dir, 'country_detail')
usummary_dir = os.path.join(base_output_dir, 'country_usummary')
summary_dir = os.path.join(base_output_dir, 'country_summary')
sector_dir = os.path.join(base_output_dir, 'country_sector')

for dir_path in [detail_dir, usummary_dir, summary_dir, sector_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Store original country totals for validation
print("Calculating original country totals...")
original_country_totals = []
for continent, df in processed_data.items():
    country_totals = df.groupby('Country')['impVal'].sum().reset_index()
    country_totals['continent'] = continent
    original_country_totals.append(country_totals)

original_country_totals_df = pd.concat(original_country_totals, ignore_index=True)

# Version 1: Aggregate to Country X detail_code level
print("\nCreating Version 1: Country X detail_code aggregation...")
detail_aggregated_data = {}
for continent, df in processed_data.items():
    # Filter out rows with missing detail_code and trim detail_code
    df_with_bea = df[df['detail_code'].notna()].copy()
    df_with_bea['detail_code'] = df_with_bea['detail_code'].astype(str).str.strip()
    
    # Aggregate by Country and detail_code, preserving iso3
    country_bea_agg = df_with_bea.groupby(['Country', 'detail_code', 'iso3'])['impVal'].sum().reset_index()
    country_bea_agg['continent'] = continent
    
    detail_aggregated_data[continent] = country_bea_agg
    
    # Save aggregated data
    output_filename = f"{continent.replace(' ', '_')}_aggregated.csv"
    output_path = os.path.join(detail_dir, output_filename)
    country_bea_agg.to_csv(output_path, index=False)
    print(f"Saved {continent} detail aggregated data: {len(country_bea_agg)} rows")

# Combine all detail level data into one file
print("\nCombining all detail level data...")
all_detail_data = pd.concat(detail_aggregated_data.values(), ignore_index=True)
all_detail_output_path = os.path.join(detail_dir, 'all_continents_detail.csv')
all_detail_data.to_csv(all_detail_output_path, index=False)
print(f"Saved combined detail data: {len(all_detail_data)} rows")

# Version 2: Aggregate to Country X U.Summary level
print("\nCreating Version 2: Country X U.Summary aggregation...")
all_continents_detail = pd.concat(detail_aggregated_data.values(), ignore_index=True)
all_continents_detail['usummary_code'] = all_continents_detail['detail_code'].map(detail_to_usummary)
all_continents_detail = all_continents_detail[all_continents_detail['usummary_code'].notna()]
# Trim usummary_code
all_continents_detail['usummary_code'] = all_continents_detail['usummary_code'].astype(str).str.strip()

usummary_aggregated = all_continents_detail.groupby(['Country', 'usummary_code', 'iso3'])['impVal'].sum().reset_index()
usummary_output_path = os.path.join(usummary_dir, 'all_continents_usummary.csv')
usummary_aggregated.to_csv(usummary_output_path, index=False)
print(f"Saved U.Summary aggregated data: {len(usummary_aggregated)} rows")

# Version 3: Aggregate to Country X Summary level
print("\nCreating Version 3: Country X Summary aggregation...")
all_continents_detail['summary_code'] = all_continents_detail['detail_code'].map(detail_to_summary)
all_continents_detail = all_continents_detail[all_continents_detail['summary_code'].notna()]
# Trim summary_code
all_continents_detail['summary_code'] = all_continents_detail['summary_code'].astype(str).str.strip()

summary_aggregated = all_continents_detail.groupby(['Country', 'summary_code', 'iso3'])['impVal'].sum().reset_index()
summary_output_path = os.path.join(summary_dir, 'all_continents_summary.csv')
summary_aggregated.to_csv(summary_output_path, index=False)
print(f"Saved Summary aggregated data: {len(summary_aggregated)} rows")

# Version 4: Aggregate to Country X Sector level
print("\nCreating Version 4: Country X Sector aggregation...")
all_continents_detail['sector_code'] = all_continents_detail['detail_code'].map(detail_to_sector)
all_continents_detail = all_continents_detail[all_continents_detail['sector_code'].notna()]
# Trim sector_code
all_continents_detail['sector_code'] = all_continents_detail['sector_code'].astype(str).str.strip()

sector_aggregated = all_continents_detail.groupby(['Country', 'sector_code', 'iso3'])['impVal'].sum().reset_index()
sector_output_path = os.path.join(sector_dir, 'all_continents_sector.csv')
sector_aggregated.to_csv(sector_output_path, index=False)
print(f"Saved Sector aggregated data: {len(sector_aggregated)} rows")

# Validation: Check that country totals are preserved at each level
print("\nValidating country totals across all aggregation levels...")

# Detail level validation
detail_country_totals = []
for continent, df in detail_aggregated_data.items():
    country_totals = df.groupby('Country')['impVal'].sum().reset_index()
    country_totals['continent'] = continent
    country_totals = country_totals.rename(columns={'impVal': 'detail_impVal'})
    detail_country_totals.append(country_totals)

detail_country_totals_df = pd.concat(detail_country_totals, ignore_index=True)

# U.Summary level validation
usummary_country_totals = usummary_aggregated.groupby('Country')['impVal'].sum().reset_index()
usummary_country_totals = usummary_country_totals.rename(columns={'impVal': 'usummary_impVal'})

# Summary level validation
summary_country_totals = summary_aggregated.groupby('Country')['impVal'].sum().reset_index()
summary_country_totals = summary_country_totals.rename(columns={'impVal': 'summary_impVal'})

# Sector level validation
sector_country_totals = sector_aggregated.groupby('Country')['impVal'].sum().reset_index()
sector_country_totals = sector_country_totals.rename(columns={'impVal': 'sector_impVal'})

# Merge all validation data
validation_df = original_country_totals_df.merge(
    detail_country_totals_df, 
    on=['Country', 'continent'], 
    how='left'
).merge(
    usummary_country_totals, 
    on='Country', 
    how='left'
).merge(
    summary_country_totals, 
    on='Country', 
    how='left'
).merge(
    sector_country_totals, 
    on='Country', 
    how='left'
)

# Calculate differences
validation_df['detail_difference'] = validation_df['impVal'] - validation_df['detail_impVal']
validation_df['detail_pct_difference'] = (validation_df['detail_difference'] / validation_df['impVal']) * 100
validation_df['usummary_difference'] = validation_df['impVal'] - validation_df['usummary_impVal']
validation_df['usummary_pct_difference'] = (validation_df['usummary_difference'] / validation_df['impVal']) * 100
validation_df['summary_difference'] = validation_df['impVal'] - validation_df['summary_impVal']
validation_df['summary_pct_difference'] = (validation_df['summary_difference'] / validation_df['impVal']) * 100
validation_df['sector_difference'] = validation_df['impVal'] - validation_df['sector_impVal']
validation_df['sector_pct_difference'] = (validation_df['sector_difference'] / validation_df['impVal']) * 100

# Save validation results
validation_path = os.path.join(validation_dir, '1_Country_Aggregation_Validation.csv')
validation_df.to_csv(validation_path, index=False)
total_original = validation_df['impVal'].sum()
total_detail = validation_df['detail_impVal'].sum()
total_usummary = validation_df['usummary_impVal'].sum()
total_summary = validation_df['summary_impVal'].sum()
total_sector = validation_df['sector_impVal'].sum()

print(f"Total original import value: ${total_original:,.0f}")
print(f"Total detail aggregated import value: ${total_detail:,.0f}")
print(f"Total U.Summary aggregated import value: ${total_usummary:,.0f}")
print(f"Total Summary aggregated import value: ${total_summary:,.0f}")
print(f"Total Sector aggregated import value: ${total_sector:,.0f}")
print(f"\nDetail level difference: ${total_original - total_detail:,.0f} ({((total_original - total_detail)/total_original)*100:.6f}%)")
print(f"U.Summary level difference: ${total_original - total_usummary:,.0f} ({((total_original - total_usummary)/total_original)*100:.6f}%)")
print(f"Summary level difference: ${total_original - total_summary:,.0f} ({((total_original - total_summary)/total_original)*100:.6f}%)")
print(f"Sector level difference: ${total_original - total_sector:,.0f} ({((total_original - total_sector)/total_original)*100:.6f}%)")

# Check for significant differences at any level
levels = ['detail', 'usummary', 'summary', 'sector']
for level in levels:
    significant_diffs = validation_df[abs(validation_df[f'{level}_pct_difference']) > 0.01]
    if len(significant_diffs) > 0:
        print(f"\nWarning: {len(significant_diffs)} countries have {level} level differences > 0.01%")
    else:
        print(f"\n{level.capitalize()} level validation passed: No significant differences found")

print(f"\nValidation results saved to: {validation_path}")

#################################################################################
# Step 2: Create HS Code Hierarchy and Calculate Weights within BEA Codes
#################################################################################

print("\n" + "="*80)
print("STEP 2: CREATING HS CODE HIERARCHY AND CALCULATING WEIGHTS")
print("="*80)

# Load HS section mapping
hs_section_path = get_data_path('raw', 'hs10', 'hs_section_chapter_mapping')
hs_section_mapping = pd.read_csv(hs_section_path)
# Create HS Chapter to Section mapping (HS_Chapter is integer without leading zeros)
chapter_to_section = dict(zip(hs_section_mapping['HS_Chapter'], hs_section_mapping['HS_Section']))
def create_hs_hierarchy(df):
    """
    Create HS code hierarchy from the existing hs_code (commodity) field.
    
    Logic:
    - If hs_code is 9 digits: HS8=first 7, HS6=first 5, HS4=first 3, HS2=first 1
    - If hs_code is 10 digits: HS8=first 8, HS6=first 6, HS4=first 4, HS2=first 2
    """
    df = df.copy()
    
    # Ensure hs_code is string and handle any missing values
    df['hs_code'] = df['hs_code'].astype(str).str.strip()
    
    # Rename hs_code to HS10 for clarity
    df = df.rename(columns={'hs_code': 'HS10'})
    
    # Create HS hierarchy based on length
    df['HS10_length'] = df['HS10'].str.len()
    # 9 digit HS codes (i.e. no leading 0 )
    mask_9 = df['HS10_length'] == 9
    df.loc[mask_9, 'HS8'] = df.loc[mask_9, 'HS10'].str[:7]
    df.loc[mask_9, 'HS6'] = df.loc[mask_9, 'HS10'].str[:5]
    df.loc[mask_9, 'HS4'] = df.loc[mask_9, 'HS10'].str[:3]
    df.loc[mask_9, 'HS2'] = df.loc[mask_9, 'HS10'].str[:1]
    mask_10 = df['HS10_length'] == 10
    df.loc[mask_10, 'HS8'] = df.loc[mask_10, 'HS10'].str[:8]
    df.loc[mask_10, 'HS6'] = df.loc[mask_10, 'HS10'].str[:6]
    df.loc[mask_10, 'HS4'] = df.loc[mask_10, 'HS10'].str[:4]
    df.loc[mask_10, 'HS2'] = df.loc[mask_10, 'HS10'].str[:2]
    if 'hs2_code' in df.columns:
        df['HS2_check'] = df['hs2_code'].astype(str).str.zfill(2)
        validation_issues = df[df['HS2'] != df['HS2_check']]
        if len(validation_issues) > 0:
            print(f"Warning: {len(validation_issues)} rows have HS2 validation issues")
        df = df.drop(columns=['HS2_check'])
    
    # Create HS Section mapping
    df['HS2_int'] = df['HS2'].astype(int)
    df['HS_Section'] = df['HS2_int'].map(chapter_to_section)
    
    # Drop temporary columns
    df = df.drop(columns=['HS10_length', 'HS2_int'])
    
    return df

def calculate_hs_weights_within_bea(df, hs_level, bea_level):
    """
    Calculate weights of each HS code within each BEA code for a given country.
    
    Parameters:
    - df: DataFrame with HS hierarchy and BEA codes
    - hs_level: string, the HS level to calculate weights for (e.g., 'HS8', 'HS6', etc.)
    - bea_level: string, the BEA level to group by (e.g., 'detail_code', 'usummary_code', etc.)
    
    Returns:
    - DataFrame with weights showing how much each HS code contributes to each BEA code
    """
    if hs_level not in df.columns or bea_level not in df.columns:
        print(f"Error: Missing columns {hs_level} or {bea_level}")
        return None
    
    # Filter out missing values
    df_clean = df[(df[hs_level].notna()) & (df[bea_level].notna())].copy()
    
    if len(df_clean) == 0:
        print(f"Warning: No valid data for {hs_level} and {bea_level}")
        return None
    
    # Calculate total import value for each Country-BEA combination, preserving iso3
    bea_totals = df_clean.groupby(['Country', bea_level, 'iso3'])['impVal'].sum().reset_index()
    bea_totals = bea_totals.rename(columns={'impVal': 'bea_total'})
    hs_within_bea = df_clean.groupby(['Country', bea_level, hs_level, 'iso3'])['impVal'].sum().reset_index()
    
    # Merge to get totals
    hs_weights = hs_within_bea.merge(bea_totals, on=['Country', bea_level, 'iso3'], how='left')
    hs_weights['weight'] = hs_weights['impVal'] / hs_weights['bea_total']
    col_order = ['Country', 'iso3', hs_level, bea_level, 'impVal', 'bea_total', 'weight']
    hs_weights = hs_weights[col_order]
    
    return hs_weights
weights_base_dir = os.path.join(data_paths['base_paths']['working_data'], '04_Aggregate_BEA_and_HS', 'hs_weights')
os.makedirs(weights_base_dir, exist_ok=True)

# Create subdirectories for each BEA level
weights_dirs = {}
for bea_level in ['detail', 'usummary', 'summary', 'sector']:
    weights_dirs[bea_level] = os.path.join(weights_base_dir, bea_level)
    os.makedirs(weights_dirs[bea_level], exist_ok=True)

bea_levels = {
    'detail': ('detail_code', detail_aggregated_data),
    'usummary': ('usummary_code', {'all_continents': all_continents_detail}),
    'summary': ('summary_code', {'all_continents': all_continents_detail}),
    'sector': ('sector_code', {'all_continents': all_continents_detail})
}

for bea_level_name, (bea_column, bea_data) in bea_levels.items():
    print(f"\nProcessing {bea_level_name} level...")
    
    # Prepare data for this BEA level
    if bea_level_name == 'detail':
        # For detail level, we need to go back to original processed data with HS codes
        bea_data_with_hs = {}
        for continent, df in processed_data.items():
            df_with_bea = df[df['detail_code'].notna()].copy()
            df_with_bea['detail_code'] = df_with_bea['detail_code'].astype(str).str.strip()
            df_with_hs = create_hs_hierarchy(df_with_bea)
            bea_data_with_hs[continent] = df_with_hs
        
        # Combine all continents
        all_data = pd.concat(bea_data_with_hs.values(), ignore_index=True)
    else:
        # For higher levels, use the already aggregated data and add HS hierarchy
        # We need to go back to the detail level and add the appropriate BEA mapping
        all_data_list = []
        for continent, df in processed_data.items():
            df_with_bea = df[df['detail_code'].notna()].copy()
            df_with_bea['detail_code'] = df_with_bea['detail_code'].astype(str).str.strip()
            df_with_hs = create_hs_hierarchy(df_with_bea)
            
            # Add the appropriate BEA level mapping
            if bea_level_name == 'usummary':
                df_with_hs['usummary_code'] = df_with_hs['detail_code'].map(detail_to_usummary)
                df_with_hs['usummary_code'] = df_with_hs['usummary_code'].astype(str).str.strip()
            elif bea_level_name == 'summary':
                df_with_hs['summary_code'] = df_with_hs['detail_code'].map(detail_to_summary)
                df_with_hs['summary_code'] = df_with_hs['summary_code'].astype(str).str.strip()
            elif bea_level_name == 'sector':
                df_with_hs['sector_code'] = df_with_hs['detail_code'].map(detail_to_sector)
                df_with_hs['sector_code'] = df_with_hs['sector_code'].astype(str).str.strip()
            
            all_data_list.append(df_with_hs)
        
        all_data = pd.concat(all_data_list, ignore_index=True)
        all_data = all_data[all_data[bea_column].notna()]
    
    # Calculate weights for each HS level (excluding HS10)
    hs_levels = ['HS2', 'HS4', 'HS6', 'HS8', 'HS_Section']
    
    for hs_level in hs_levels:
        print(f"  Calculating {hs_level} weights...")
        
        weights_df = calculate_hs_weights_within_bea(all_data, hs_level, bea_column)
        
        if weights_df is not None:
            # Save weights to the appropriate BEA level subdirectory
            weights_filename = f"{hs_level.lower()}_weights.csv"
            weights_path = os.path.join(weights_dirs[bea_level_name], weights_filename)
            weights_df.to_csv(weights_path, index=False)
            
            print(f"    Saved {weights_filename}: {len(weights_df)} rows")
            
            # Print summary statistics
            print(f"    Weight statistics: min={weights_df['weight'].min():.4f}, max={weights_df['weight'].max():.4f}, mean={weights_df['weight'].mean():.4f}")
        else:
            print(f"    Error calculating weights for {hs_level}")

print(f"\nHS weights calculation completed. Files saved to: {weights_base_dir}")

# Create a summary of what was created
print("\n" + "="*80)
print("SUMMARY OF OUTPUTS CREATED")
print("="*80)
print("BEA Aggregation Files:")
print(f"  Detail level: {detail_dir}")
print(f"  U.Summary level: {usummary_dir}")
print(f"  Summary level: {summary_dir}")
print(f"  Sector level: {sector_dir}")
print(f"\nHS Weights Files:")
print(f"  {weights_base_dir}")
for bea_level in ['detail', 'usummary', 'summary', 'sector']:
    print(f"    {bea_level}/")
    for hs_level in ['HS2', 'HS4', 'HS6', 'HS8', 'HS_Section']:
        print(f"      {hs_level.lower()}_weights.csv")
print(f"  - Total files created: {len(bea_levels) * len(['HS2', 'HS4', 'HS6', 'HS8', 'HS_Section'])}")
print(f"\nValidation Files:")
print(f"  {validation_dir}")
print("="*80)

#################################################################################
# Step 3: Create Final JSON Output
#################################################################################

print("\n" + "="*80)
print("STEP 3: CREATING FINAL JSON OUTPUT")
print("="*80)

def create_bea_json_simple(df):
    """
    Create final JSON structure for BEA-HS section weights.
    
    Structure: {country_iso3: {usummary_code: {hs_section: weight}}}
    """
    result = {}
    for country_iso3 in df['iso3'].unique():
        country_data = df[df['iso3'] == country_iso3]
        result[country_iso3] = {}
        for usummary in country_data['usummary_code'].unique():
            usummary_data = country_data[country_data['usummary_code'] == usummary]
            # Just the section weights
            section_weights = {}
            for _, row in usummary_data.iterrows():
                section = str(row['HS_Section'])
                weight = row['weight']
                section_weights[section] = weight
            result[country_iso3][usummary] = section_weights
    return result
# Load the HS section weights from usummary level
hs_section_weights_path = os.path.join(weights_dirs['usummary'], 'hs_section_weights.csv')

if os.path.exists(hs_section_weights_path):
    print(f"Loading HS section weights from: {hs_section_weights_path}")
    section_weights_df = pd.read_csv(hs_section_weights_path)
    # Create the final JSON structure
    final_json = create_bea_json_simple(section_weights_df)
    final_output_dir = data_paths['base_paths']['final_data']
    os.makedirs(final_output_dir, exist_ok=True)
    final_output_path = os.path.join(final_output_dir, 'bea_hs_section_weights.json')
    # Write the final data to a json.
    import json
    with open(final_output_path, 'w') as f:
        json.dump(final_json, f, indent=2)
    total_entries = 0
    for country, usummary_data in final_json.items():
        for usummary, sections in usummary_data.items():
            total_entries += len(sections)
    print(f"Total country-usummary-section combinations: {total_entries}")
else:
    print(f"Warning: HS section weights file not found at {hs_section_weights_path}")

print("="*80)