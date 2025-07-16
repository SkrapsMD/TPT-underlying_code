import os
import pandas as pd
import json
from main_pipeline_run import get_data_path

"""
Description: Processes 2024 trade data by continent and maps HS commodity codes to BEA economic categories.
This script combines raw trade data from multiple continents, applies the HS-to-BEA mapping created in
02_HS_to_Naics_to_BEA, and produces clean country-level trade data ready for economic analysis.

The core challenge we're solving:
- Raw trade data is fragmented across continents and multiple files per continent
- HS commodity codes need to be mapped to BEA economic categories for policy analysis
- Data quality issues require cleaning and validation
- We need consistent country-level aggregation across all trade relationships

This script bridges the gap between raw trade data and analytical formats:
1. RAW DATA: Continent-specific CSV files with HS10 codes and trade values
2. MAPPING: HS-to-BEA bridge from 02_HS_to_Naics_to_BEA
3. PROCESSED DATA: Clean country-level trade data with BEA economic categories

The approach:
1. Load 2024 trade data by continent (Asia, Europe, North America, South America, Oceana)
2. Skip combined_data files and process individual continent files
3. Apply HS-to-BEA mapping to add economic sector information
4. Clean and validate data quality (missing values, duplicate entries)
5. Aggregate to country level while preserving HS commodity detail
6. Create processed datasets ready for BEA aggregation

Data transformation:
- Input: HS10 commodity codes, import values, country names
- Mapping: HS10 → NAICS → BEA Detail codes
- Output: Country-HS-BEA linked trade data

Main outputs:
- Combined data: {continent}_combined.csv files with raw trade data
- Processed data: {continent}_processed.csv files with BEA mappings applied
- Validation: Data quality checks and mapping success rates

"""

data_paths_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_paths.json')
with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Step 2 -- Load the 2024 trade data by continent 
continents_to_process = ['Asia', 'Europe', 'North America', 'South America', 'Oceana']
#continents_to_process = ['Oceana']

all_continent_data = {}
base_path = data_paths['raw_data_sources']['hs10']['2024_trade_data']['base_path']

for continent in continents_to_process:
    continent_files = data_paths['raw_data_sources']['hs10']['2024_trade_data']['regions'][continent]
    continent_folder = os.path.join(data_paths['base_paths']['underlying_data_root'], base_path, continent)
    
    # Set up output directory and file path
    output_dir = os.path.join(data_paths['base_paths']['working_data'], '03_Map_country_trade_data', 'combined_data')
    os.makedirs(output_dir, exist_ok=True)
    combined_file_path = os.path.join(output_dir, f'{continent.replace(" ", "_")}_combined.csv')
    
    # Check if combined file already exists
    if os.path.exists(combined_file_path):
        # Use existing combined file
        continent_data = pd.read_csv(combined_file_path)
        # Add hs2_code column if it doesn't exist
        if 'hs_code' in continent_data.columns and 'hs2_code' not in continent_data.columns:
            continent_data['hs2_code'] = continent_data['hs_code'].str[:2]
    else:
        # Create combined file from individual files
        continent_data_list = []
        for filename in continent_files:
            if filename == 'combined_data.csv':
                continue
            file_path = os.path.join(continent_folder, filename)
            df = pd.read_csv(file_path, skiprows=2)
            # Drop any unnamed columns (like 'Unnamed: 4')
            unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
            df = df.drop(columns=['Time'])
            df = df.rename(columns={'Customs Value (Gen) ($US)': 'impVal'})
            # Extract HS code from commodity (first 10 characters)
            df['hs_code'] = df['Commodity'].str[:10]
            # Extract HS-2 code (first 2 characters)
            df['hs2_code'] = df['hs_code'].str[:2]

            df = df.drop(columns=['Commodity'])
            df['impVal'] = df['impVal'].astype(str).str.replace(',', '').astype(float)
            
            continent_data_list.append(df)
        continent_data = pd.concat(continent_data_list, ignore_index=True)
        continent_data.to_csv(combined_file_path, index=False)
    all_continent_data[continent] = continent_data

# Quick Validation: Do the sums make sense for each country? 
# Answer - yes!
country_sums_list = []
for continent, data in all_continent_data.items():
    country_sums = data.groupby('Country')['impVal'].sum().reset_index()
    country_sums['continent'] = continent
    country_sums_list.append(country_sums)
country_import_sums = pd.concat(country_sums_list, ignore_index=True)
country_import_sums = country_import_sums.sort_values('impVal', ascending=False)
validation_dir = os.path.join(data_paths['base_paths']['validations'], '03_Map_country_trade_data')
os.makedirs(validation_dir, exist_ok=True)
validation_path = os.path.join(validation_dir, '1_country_import_sums.csv')
country_import_sums.to_csv(validation_path, index=False)
print("Top 3 countries by import value:")
for i, row in country_import_sums.head(3).iterrows():
    print(f"{row['Country']}: ${row['impVal']:,.0f}")

# Step 2 - Merge in the data/working/02_HS_to_Naics_to_BEA/03_complete_hs_to_bea_mapping.csv file
hs_to_bea_mapping_path = os.path.join(data_paths['base_paths']['working_data'], '02_HS_to_Naics_to_BEA', '03_complete_hs_to_bea_mapping.csv')
hs_to_bea_mapping = pd.read_csv(hs_to_bea_mapping_path)
hs_to_bea_mapping = hs_to_bea_mapping[['commodity', 'matched_bea_detail']]

# Merge with each continent's data
for continent in continents_to_process:
    if continent in all_continent_data:
        merged_data = all_continent_data[continent].merge(
            hs_to_bea_mapping, 
            left_on='hs_code', 
            right_on='commodity', 
            how='left'
        )
        all_continent_data[continent] = merged_data

# "This is a check to see if we have any unmapped hs commodities... Because these are 2024 values, but we use schotts 2023 values, there may be missings here we need to control for."
failed_merge_data = []
for continent, data in all_continent_data.items():
    failed_data = data[data['matched_bea_detail'].isna()]
    failed_merge_data.append(failed_data)

if failed_merge_data:
    all_failed_data = pd.concat(failed_merge_data, ignore_index=True)
    failed_hs_sums = all_failed_data.groupby('hs_code')['impVal'].sum().reset_index()
    failed_hs_sums = failed_hs_sums.sort_values('impVal', ascending=False)
    
    failed_merge_path = os.path.join(validation_dir, '2_Failed_Merge_HS_Codes.csv')
    failed_hs_sums['hs_code'] = failed_hs_sums['hs_code'].astype(str)
    failed_hs_sums.to_csv(failed_merge_path, index=False)
    
    print(f"Number of unmapped HS codes: {len(failed_hs_sums)}")
    print(f"Total import value of unmapped codes: ${failed_hs_sums['impVal'].sum():,.0f}")
    print("Top 3 unmapped HS codes by import value:")
    for i, row in failed_hs_sums.head(3).iterrows():
        print(f"{row['hs_code']}: ${row['impVal']:,.0f}")
    
    # Try hierarchical matching using complete HS to BEA mapping data
    print("\nAttempting hierarchical matching for unmapped HS codes...")
    complete_mapping_path = os.path.join(data_paths['base_paths']['working_data'], '02_HS_to_Naics_to_BEA', '03_complete_hs_to_bea_mapping.csv')
    complete_mapping = pd.read_csv(complete_mapping_path, dtype={'commodity': str})
    
    # Filter to only successfully mapped entries
    complete_mapping = complete_mapping[complete_mapping['matched_bea_detail'].notna()]
    
    # Create hierarchical lookup dictionaries
    hs8_lookup = {}
    hs6_lookup = {}
    hs4_lookup = {}
    for _, row in complete_mapping.iterrows():
        commodity = row['commodity']
        bea_detail = row['matched_bea_detail']
        # Create hierarchical codes
        if len(commodity) >= 8:
            hs8 = commodity[:8]
            if hs8 not in hs8_lookup:
                hs8_lookup[hs8] = []
            hs8_lookup[hs8].append(bea_detail)
        if len(commodity) >= 6:
            hs6 = commodity[:6]
            if hs6 not in hs6_lookup:
                hs6_lookup[hs6] = []
            hs6_lookup[hs6].append(bea_detail)
        if len(commodity) >= 4:
            hs4 = commodity[:4]
            if hs4 not in hs4_lookup:
                hs4_lookup[hs4] = []
            hs4_lookup[hs4].append(bea_detail)
    
    # Get the ACTUAL unmapped HS codes from continent data
    actual_failed_list = []
    for continent in continents_to_process:
        if continent in all_continent_data:
            unmapped_data = all_continent_data[continent][all_continent_data[continent]['matched_bea_detail'].isna()]
            if not unmapped_data.empty:
                hs_sums = unmapped_data.groupby('hs_code')['impVal'].sum().reset_index()
                actual_failed_list.append(hs_sums)
    
    if actual_failed_list:
        actual_failed_hs_sums = pd.concat(actual_failed_list, ignore_index=True)
        actual_failed_hs_sums = actual_failed_hs_sums.groupby('hs_code')['impVal'].sum().reset_index()
        
        # Filter out NaN HS codes and ensure they're strings
        actual_failed_hs_sums = actual_failed_hs_sums[actual_failed_hs_sums['hs_code'].notna()]
        actual_failed_hs_sums['hs_code'] = actual_failed_hs_sums['hs_code'].astype(str)
        actual_failed_hs_sums = actual_failed_hs_sums.sort_values('impVal', ascending=False)
        
        print(f"Found {len(actual_failed_hs_sums)} actual unmapped HS codes")
    else:
        actual_failed_hs_sums = pd.DataFrame(columns=['hs_code', 'impVal'])
    
    # Try to match ACTUAL unmapped HS codes
    hierarchical_matches = []
    for _, row in actual_failed_hs_sums.iterrows():
        hs_code = row['hs_code']
        match_found = False
        # Try HS-8 first
        if len(hs_code) >= 8:
            hs8 = hs_code[:8]
            if hs8 in hs8_lookup:
                from collections import Counter
                naics_options = Counter(hs8_lookup[hs8])
                most_common_naics = naics_options.most_common(1)[0][0]
                modal_count = naics_options.most_common(1)[0][1]
                total_matches = len(hs8_lookup[hs8])
                mapping_strength = modal_count / total_matches
                hierarchical_matches.append({
                    'hs_code': hs_code,
                    'matched_bea_detail': most_common_naics,
                    'match_level': 'hs8',
                    'mapping_strength': mapping_strength,
                    'modal_count': modal_count,
                    'total_matches': total_matches,
                    'impVal': row['impVal']
                })
                match_found = True
        # Try HS-6 if HS-8 didn't work
        if not match_found and len(hs_code) >= 6:
            hs6 = hs_code[:6]
            if hs6 in hs6_lookup:
                from collections import Counter
                naics_options = Counter(hs6_lookup[hs6])
                most_common_naics = naics_options.most_common(1)[0][0]
                modal_count = naics_options.most_common(1)[0][1]
                total_matches = len(hs6_lookup[hs6])
                mapping_strength = modal_count / total_matches
                hierarchical_matches.append({
                    'hs_code': hs_code,
                    'matched_bea_detail': most_common_naics,
                    'match_level': 'hs6',
                    'mapping_strength': mapping_strength,
                    'modal_count': modal_count,
                    'total_matches': total_matches,
                    'impVal': row['impVal']
                })
                match_found = True
        # Try HS-4 if HS-6 didn't work
        if not match_found and len(hs_code) >= 4:
            hs4 = hs_code[:4]
            if hs4 in hs4_lookup:
                from collections import Counter
                naics_options = Counter(hs4_lookup[hs4])
                most_common_naics = naics_options.most_common(1)[0][0]
                modal_count = naics_options.most_common(1)[0][1]
                total_matches = len(hs4_lookup[hs4])
                mapping_strength = modal_count / total_matches
                hierarchical_matches.append({
                    'hs_code': hs_code,
                    'matched_bea_detail': most_common_naics,
                    'match_level': 'hs4',
                    'mapping_strength': mapping_strength,
                    'modal_count': modal_count,
                    'total_matches': total_matches,
                    'impVal': row['impVal']
                })
                
    # Save the Hierarchical Matches and then apply them to the continent data
    if hierarchical_matches:
        hierarchical_df = pd.DataFrame(hierarchical_matches)
        hierarchical_df = hierarchical_df.sort_values('impVal', ascending=False)
        
        # Save hierarchical matches
        hierarchical_path = os.path.join(validation_dir, '3_Hierarchical_Matches.csv')
        hierarchical_df.to_csv(hierarchical_path, index=False)
        
        print(f"Found hierarchical matches for {len(hierarchical_df)} HS codes")
        print(f"Total import value of hierarchical matches: ${hierarchical_df['impVal'].sum():,.0f}")
        
        # Calculate remaining unmapped value
        remaining_unmapped_value = failed_hs_sums['impVal'].sum() - hierarchical_df['impVal'].sum()
        print(f"Remaining unmapped import value: ${remaining_unmapped_value:,.0f}")
        
        # Show match level distribution
        level_counts = hierarchical_df['match_level'].value_counts()
        print("Match level distribution:")
        for level, count in level_counts.items():
            print(f"  {level}: {count} codes")
        
        # Apply hierarchical mappings back to continent data with threshold
        
####### This is a parameter you might want to adjust, or think about an alternative (i.e. maybe a naics5 with wildcard if applicable, I don't know)
        mapping_threshold = 0.5 ## Silly parameter value
        high_confidence_matches = hierarchical_df[hierarchical_df['mapping_strength'] >= mapping_threshold]
        
        print(f"\nApplying {len(high_confidence_matches)} high-confidence matches (strength >= {mapping_threshold}) to continent data...")
        
        # Create lookup dictionary for high-confidence matches (now directly to BEA codes)
        hs_to_bea_lookup = dict(zip(high_confidence_matches['hs_code'], high_confidence_matches['matched_bea_detail']))
        
        # Apply matches to each continent's data
        total_applied = 0
        
        for continent in continents_to_process:
            if continent in all_continent_data:
                # Convert continent HS codes to strings for matching
                continent_hs_str = all_continent_data[continent]['hs_code'].astype(str)
                
                # Update matched_bea_detail for high-confidence hierarchical matches
                mask = (all_continent_data[continent]['matched_bea_detail'].isna()) & \
                       (continent_hs_str.isin(hs_to_bea_lookup.keys()))
                
                matches_in_continent = mask.sum()
                
                if matches_in_continent > 0:
                    # Use vectorized operations for better performance
                    matched_rows = all_continent_data[continent][mask].copy()
                    matched_rows['hs_code_str'] = matched_rows['hs_code'].astype(str)
                    matched_rows['matched_bea_detail'] = matched_rows['hs_code_str'].map(hs_to_bea_lookup)
                    
                    # Update the original dataframe
                    all_continent_data[continent].loc[mask, 'matched_bea_detail'] = matched_rows['matched_bea_detail']
                    total_applied += matches_in_continent
        
        print(f"Applied {total_applied} hierarchical BEA matches directly")
        
        applied_value = high_confidence_matches['impVal'].sum()
        print(f"Applied hierarchical matches worth ${applied_value:,.0f} in import value")
        
        # Save processed continent data
        print("\nSaving processed continent data...")
        processed_data_dir = os.path.join(data_paths['base_paths']['working_data'], '03_Map_country_trade_data', 'processed_data')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        for continent in continents_to_process:
            if continent in all_continent_data:
                # Clean up the data before saving
                continent_df = all_continent_data[continent].copy()
                
                # Drop commodity column if it exists
                if 'commodity' in continent_df.columns:
                    continent_df = continent_df.drop(columns=['commodity'])
                
                # Rename matched_bea_detail to detail_code
                if 'matched_bea_detail' in continent_df.columns:
                    continent_df = continent_df.rename(columns={'matched_bea_detail': 'detail_code'})
                
                continent_filename = f"{continent.replace(' ', '_')}_processed.csv"
                continent_path = os.path.join(processed_data_dir, continent_filename)
                continent_df.to_csv(continent_path, index=False)
                print(f"Saved {continent} data: {len(continent_df)} rows")
        
        # Update remaining unmapped value
        remaining_unmapped_value = failed_hs_sums['impVal'].sum() - applied_value
        print(f"Final remaining unmapped import value: ${remaining_unmapped_value:,.0f}")
        
    else:
        print("No hierarchical matches found")
        print(f"Remaining unmapped import value: ${failed_hs_sums['impVal'].sum():,.0f}")

#################################################################################
# Step 3: Create BEA-level aggregated datasets for each continent
#################################################################################

# Create BEA-level aggregated datasets for each continent
continent_bea_data = {}
country_bea_sums_list = []

for continent in continents_to_process:
    if continent in all_continent_data:
        # Create a copy of the continent data
        continent_df = all_continent_data[continent].copy()
        # Filter out rows with missing BEA codes
        mapped_data = continent_df[continent_df['matched_bea_detail'].notna()]
        # Group by Country and matched_bea_detail, sum impVal
        bea_aggregated = mapped_data.groupby(['Country', 'matched_bea_detail'])['impVal'].sum().reset_index()
        continent_bea_data[continent] = bea_aggregated
        
        country_bea_sums = mapped_data.groupby('Country')['impVal'].sum().reset_index()
        country_bea_sums['continent'] = continent
        country_bea_sums = country_bea_sums.rename(columns={'impVal': 'bea_mapped_impVal'})
        country_bea_sums_list.append(country_bea_sums)

# Combine all country sums and save validation file
if country_bea_sums_list:
    country_bea_sums = pd.concat(country_bea_sums_list, ignore_index=True)
    country_bea_sums = country_bea_sums.merge(
        country_import_sums[['Country', 'continent', 'impVal']], 
        on=['Country', 'continent'], 
        how='left'
    )
    country_bea_sums = country_bea_sums.rename(columns={'impVal': 'original_impVal'})
    country_bea_sums['unmapped_impVal'] = country_bea_sums['original_impVal'] - country_bea_sums['bea_mapped_impVal']
    country_bea_sums['mapping_coverage'] = country_bea_sums['bea_mapped_impVal'] / country_bea_sums['original_impVal']
    country_bea_sums = country_bea_sums.sort_values('original_impVal', ascending=False)

    # Save BEA-level country sums validation
    bea_validation_path = os.path.join(validation_dir, '4_Country_BEA_Mapped_Sums.csv')
    country_bea_sums.to_csv(bea_validation_path, index=False)
    print(f"BEA-level country sums saved to validation file")
    print(f"Total original import value: ${country_bea_sums['original_impVal'].sum():,.0f}")
    print(f"Total BEA-mapped import value: ${country_bea_sums['bea_mapped_impVal'].sum():,.0f}")
    print(f"Overall mapping coverage: {country_bea_sums['bea_mapped_impVal'].sum() / country_bea_sums['original_impVal'].sum():.1%}")

    # Create detailed breakdown of unmapped HS codes by country (using UPDATED continent data)
    unmapped_details_list = []
    for continent in continents_to_process:
        if continent in all_continent_data:
            continent_df = all_continent_data[continent].copy()
            
            # Get unmapped data
            unmapped_data = continent_df[continent_df['matched_bea_detail'].isna()]
            
            if not unmapped_data.empty:
                # Group by Country and HS code, sum impVal
                unmapped_by_country_hs = unmapped_data.groupby(['Country', 'hs_code'])['impVal'].sum().reset_index()
                unmapped_by_country_hs['continent'] = continent
                unmapped_details_list.append(unmapped_by_country_hs)
    
    # Save detailed unmapped breakdown
    if unmapped_details_list:
        unmapped_details = pd.concat(unmapped_details_list, ignore_index=True)
        unmapped_details = unmapped_details.sort_values('impVal', ascending=False)
        
        # Save detailed unmapped file
        unmapped_details_path = os.path.join(validation_dir, '5_Unmapped_Country_HS_Details.csv')
        unmapped_details['hs_code'] = unmapped_details['hs_code'].astype(str)
        unmapped_details.to_csv(unmapped_details_path, index=False)
        
        print(f"Detailed unmapped breakdown saved: {len(unmapped_details)} country-HS combinations")
        print(f"Top 5 unmapped country-HS combinations:")
        for i, row in unmapped_details.head(5).iterrows():
            print(f"  {row['Country']} - {row['hs_code']}: ${row['impVal']:,.0f}")
        # Summary by country
        country_unmapped_sums = unmapped_details.groupby('Country')['impVal'].sum().reset_index()
        country_unmapped_sums = country_unmapped_sums.sort_values('impVal', ascending=False)
        
        print(f"\nTop 5 countries by unmapped import value:")
        for i, row in country_unmapped_sums.head(5).iterrows():
            print(f"  {row['Country']}: ${row['impVal']:,.0f}")
            
        # Summary by HS code
        hs_unmapped_sums = unmapped_details.groupby('hs_code')['impVal'].sum().reset_index()
        hs_unmapped_sums = hs_unmapped_sums.sort_values('impVal', ascending=False)
        
        print(f"\nTop 5 HS codes by unmapped import value:")
        for i, row in hs_unmapped_sums.head(5).iterrows():
            print(f"  {row['hs_code']}: ${row['impVal']:,.0f}")
