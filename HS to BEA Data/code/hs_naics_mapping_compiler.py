import os
import pandas as pd
from main_pipeline_run import get_data_path

"""
DESCRIPTION: This code addresses the complex challenge of mapping 2023 HS commodity codes to 2017 NAICS codes
for trade analysis, despite simultaneous changes in both HS classifications and NAICS classifications.

The primary data output of this process is data/working/01_Schott_Data_Compiler/03_hs_naics_mapping_2023_corrected_naicsMDS. 
For extra validation, we also create a set of validation files in the validations/01_Schott_Data_Compiler/ folder. 

THE CORE PROBLEM:
We need to map 2023 HS codes to 2017 NAICS codes for BEA analysis, but face two simultaneous challenges:
1. HS codes change over time (new codes appear, old codes disappear, classifications evolve)
2. NAICS codes changed between 2017 and 2022 (requiring crosswalk mapping)

The raw Schott data gives us:
- 2022 mappings: HS codes → 2017 NAICS codes (what we want for consistency)
- 2023 mappings: HS codes → 2022 NAICS codes (what we have for current data)

SOLUTION APPROACH:
We use a 3-step hierarchical approach to create corrected 2023→2017 mappings:

1. DIRECT_2017 (lines 544-556): For HS codes that exist in both 2022 and 2023 data,
    directly use the 2022 mapping to 2017 NAICS codes (most reliable)

2. SIMPLE_CROSSWALK (lines 558-577): For HS codes only in 2023 data where the 2022 NAICS
    maps to exactly one 2017 NAICS code, use the official crosswalk

3. HIERARCHICAL_HS (lines 579-616): For complex cases where 2022 NAICS maps to multiple
    2017 NAICS codes, use hierarchical matching at HS-8, HS-6, and HS-4 levels to find
    similar historical patterns (lines 363-500)

CODE STRUCTURE:
- Lines 34-66: Basic data extraction by year
- Lines 71-159: Validation functions for distinct counts
- Lines 174-361: Analysis of many-to-many NAICS mappings and overlap issues
- Lines 363-500: Hierarchical HS matching algorithm
- Lines 517-637: Main correction algorithm with 3-step approach
- Lines 640-737: Validation CSV creation for mapping issues
- Lines 756-864: Final validation against 2022 NAICS codes

Inputs: {
    import mapping: "data/raw/hs_sic_naics_imports_89_123_20240801.csv",
    export mapping: "data/raw/hs_sic_naics_exports_89_123_20240801.csv"
}

Outputs: { 
    Data: {
        01_hs_naics_mapping_2022_imports.csv: "2022 HS codes → 2017 NAICS codes",
        02_hs_naics_mapping_2023_imports.csv: "2023 HS codes → 2022 NAICS codes", 
        03_hs_naics_mapping_2023_corrected_naicsMDS.csv: "2023 HS codes → 2017 NAICS codes (corrected)"
    },
    Validations: {
        1_Schott_Import_Distinct_Counts.csv: "Year-by-year distinct value counts",
        2_hs_overlap_analysis_all_naics.csv: "HS commodity overlap for many-to-many NAICS mappings",
        3_hs_mapping_analysis_simplified.csv: "HS commodities with mapping inconsistencies between years",
        4_mapping_validation_issues.csv: "Hierarchical matching details and mapping problems",
        5_FINAL_schott_validation.txt: "Final validation results with method breakdown and descriptions"
    }
}
"""


imports_file_raw = get_data_path('raw', 'hs10', 'imports')
exports_file_raw = get_data_path('raw', 'hs10', 'exports')

## IMPORTANT: This is the main function that compiles the data.
def save_year_data(file_path, year, output_suffix):
    """
    MAIN FUNCTION: Extracts mappings for a specific year and saves the csv to the data/working/Schott_Concordances directory. 
    
    Args: 
        file_path (str): Path to the input csv file
        year (int): Year to extract the data mapping for
        output_suffix (str): Suffix for the output file name (imports or exports)
        
    NOTE: WE only use the imports mappings. The exports are nice for completeness, but notused in any step of the current analysis. 
    """
    print(f"Processing {year} mappings from {file_path}...")
    try:
        df = pd.read_csv(file_path, delimiter = '\t')
        year_data = df[df['year'] == year]
        if year_data.empty:
            # No year data found, skip
            print(f"No data found for year {year} in {file_path}. Skipping...")
            return
        # Keep only some relevant columns from the data
        cols_to_keep = ['commodity','naics','naicsX'] 
        final_data = year_data[cols_to_keep].copy()
        # naicsX are broader NAICS codes, I use the more precisely defined naics codes, but I'll keep the NAICSX in here in case you guys want to take a look
        
        # When we have missing NAICS or NAICSX, we drop them. 
        final_data = final_data.dropna(subset=['naics', 'naicsX'])
        
        # Save final output to the working directory (Notice this uses the get_data_path function defined in the 00_main_pipeline_run.py)
        output_dir = get_data_path('working', '01_Schott_Data_Compiler')
        output_file = os.path.join(output_dir, f'0{year-2021}_hs_naics_mapping_{year}_{output_suffix}.csv')
        final_data.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error in extracting {year} data: {e}")
        
## VALIDATIONS: We want a function that will construct distinct values by year just to make sure that we are pulling the data we 
## think we are pulling... 

def analyze_distinct_counts(file_path, trade_direction):
    """
    Analyzes a Schott HS-NAICS mapping data file and counts the number of distinct values by year.
    This allows us to be sure that we can pull just one year of the data at a time. 
    
    Args: 
        file_path (str): Path to the input csv file
        trade_direction (str): 'imports' or 'exports' to specify the type of mapping
    """
    print(f"Analyzing distinct {trade_direction} values in {file_path}...")
    try: 
        df = pd.read_csv(file_path, delimiter='\t')
        print(f'Successfully loaded data with {len(df)} rows.')
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame() # Return an empty data frame on error in loading the data. 
    
    # Initialize results to store counts.
    results = []
    # Group by year and count distinct values. 
    years = sorted(df['year'].unique())
    for year in years: 
        year_data = df[df['year'] == year]
        # Construct counts -- SAFELY in case something is broken down here. 
        try:
            commodity_count = year_data['commodity'].nunique() if 'commodity' in df.columns else 0
        except Exception as e:
            print(f"Error counting commodities: {e}")
            commodity_count = 0
            
        try:
            sic_count = year_data['sic'].nunique() if 'sic' in df.columns else 0
        except Exception as e:
            print(f"Error counting SIC: {e}")
            sic_count = 0
            
        try:
            naics_count = year_data['naics'].nunique() if 'naics' in df.columns else 0
        except Exception as e:
            print(f"Error counting NAICS: {e}")
            naics_count = 0
            
        try:
            naicsx_count = year_data['naicsX'].nunique() if 'naicsX' in df.columns else 0
        except Exception as e:
            print(f"Error counting NAICSX: {e}")
            naicsx_count = 0
        
        # Store results
        results.append({
            'Year': year,
            'Commodities': commodity_count,
            'SIC': sic_count,
            'NAICS': naics_count,
            'NAICSX': naicsx_count
        })
    # Convert the resutls to a handy dandy dataframe 
    results_df = pd.DataFrame(results)
    
    # Add some statistics to make these more comprehensible
    try: 
        all_commodity_count = df['commodity'].nunique() if 'commodity' in df.columns else 0
        all_sic_count = df['sic'].nunique() if 'sic' in df.columns else 0
        all_naics_count = df['naics'].nunique() if 'naics' in df.columns else 0
        all_naicsx_count = df['naicsX'].nunique() if 'naicsX' in df.columns else 0
    except Exception as e:
        print(f"Error counting all values: {e}")
        all_commodity_count = all_sic_count = all_naics_count = all_naicsx_count = 0
        
    # Add Rows for Averages and for counts: 
    avg_row = {
        'Year': 'Average',
        'Commodities': results_df['Commodities'].mean(),
        'SIC': results_df['SIC'].mean(),
        'NAICS': results_df['NAICS'].mean(),
        'NAICSX': results_df['NAICSX'].mean()
    }
    all_row={
        'Year': 'All Years',
        'Commodities': all_commodity_count,
        'SIC': all_sic_count,
        'NAICS': all_naics_count,
        'NAICSX': all_naicsx_count
    }
    
    # Append and return the results from the function
    results_df = pd.concat([results_df, pd.DataFrame([avg_row, all_row])])
    return results_df


### MAIN EXECUTION of the mapping: This is the code that we run to execute the above functions and save the output data
#years = range(2018, 2024)
years = [2022, 2023]
for year in years:
    save_year_data(imports_file_raw, year, 'imports')

# # # Analyze the imports data using the analyze_distinct_counts function -- All years have approximately the same numhbers, so they completely replace on another. 
import_results = analyze_distinct_counts(imports_file_raw, 'Imports').to_csv(get_data_path('validation', '01_Schott_Data_Compiler', '1_Schott_Import_Distinct_Counts.csv'), index = False)


###################################################################################################################################

# Load NAICS crosswalk and identify many-to-many mappings
try:
    naics_cw = pd.read_csv(os.path.join(get_data_path('raw','naics_crosswalks'), 'NAICS22-17_cw_complete.csv'))
    print("Using complete NAICS crosswalk (includes self-mappings)")
    # Ensure consistent column naming
    if 'naics' in naics_cw.columns and 'naics17' not in naics_cw.columns:
        naics_cw = naics_cw.rename(columns={'naics': 'naics17'})
except FileNotFoundError:
    print("Complete crosswalk not found, using original incomplete crosswalk")
    print("Run 00_Naics_crosswalk_formation.py first to create complete crosswalk")
    naics_cw = pd.read_csv(get_data_path('raw','naics_crosswalks','naics_2022_to_2017_crosswalk'))
    naics_cw = naics_cw.rename(columns={'naics': 'naics17'})
naics_cw_grouped_22 = naics_cw.groupby('naics22')['naics17'].nunique().reset_index()
many_to_many_naics22 = naics_cw_grouped_22[naics_cw_grouped_22['naics17'] > 1]['naics22'].tolist()

print(f"Found {len(many_to_many_naics22)} NAICS22 codes that map to multiple NAICS17 codes")
print("First few:", many_to_many_naics22[:5] if len(many_to_many_naics22) > 5 else many_to_many_naics22)

# Load HS concordance data for analysis
hs_concord = {}
for year in [2023, 2022]: 
    hs_concord[year] = pd.read_csv(os.path.join(get_data_path('working', '01_Schott_Data_Compiler'), f'0{year-2021}_hs_naics_mapping_{year}_imports.csv'))

def get_commodities_for_naics22(hs_concord_df, naics22_code): 
    """
    Returns the set of commodities that map to a given NAICS 22 code in the HS concordance dataframe.
    
    Args: 
        hs_concord_df (pd.DataFrame): The HS concordance dataframe containing NAICS22 codes and commodities.
        naics22_code (str or int): The NAICS22 code to filter by.
    """
    naics22_code_str = str(naics22_code) 
    filtered_df = hs_concord_df[hs_concord_df['naics'] == naics22_code_str]
    return set(filtered_df['commodity'].unique())

# Create consolidated overlap analysis for NAICS22 codes that map to multiple NAICS17 codes
# This validates mapping consistency by comparing HS commodities across years
all_overlap_rows = []

for naics22_code in many_to_many_naics22:
    current_filtered_cw = naics_cw[naics_cw['naics22'] == naics22_code]
    current_naics17_values = current_filtered_cw['naics17'].tolist()
    current_naics22_commodities_2023 = get_commodities_for_naics22(hs_concord[2023], naics22_code)
    
    current_naics17_commodities = {}
    current_all_naics17_commodities = set()
    for naics17_code in current_naics17_values:
        commodities = get_commodities_for_naics22(hs_concord[2022], naics17_code)
        current_naics17_commodities[naics17_code] = commodities
        current_all_naics17_commodities.update(commodities)
    
    current_overlap = current_naics22_commodities_2023.intersection(current_all_naics17_commodities)
    current_hs_only_in_2023 = current_naics22_commodities_2023 - current_all_naics17_commodities
    current_hs_only_in_2022 = current_all_naics17_commodities - current_naics22_commodities_2023
    for hs_code in current_overlap:
        matching_naics17 = []
        for naics17_code in current_naics17_values:
            if hs_code in current_naics17_commodities[naics17_code]:
                matching_naics17.append(str(naics17_code))
        
        all_overlap_rows.append({
            'naics22_code': naics22_code,
            'hs_commodity': hs_code,
            'naics22_2023': naics22_code,
            'naics17_2022': ','.join(matching_naics17),
            'overlap_type': 'both_years'
        })
    # Add HS codes only in 2023
    for hs_code in current_hs_only_in_2023:
        all_overlap_rows.append({
            'naics22_code': naics22_code,
            'hs_commodity': hs_code,
            'naics22_2023': naics22_code,
            'naics17_2022': '',
            'overlap_type': 'only_2023'
        })
    # Add HS codes only in 2022
    for hs_code in current_hs_only_in_2022:
        matching_naics17 = []
        for naics17_code in current_naics17_values:
            if hs_code in current_naics17_commodities[naics17_code]:
                matching_naics17.append(str(naics17_code))
        all_overlap_rows.append({
            'naics22_code': naics22_code,
            'hs_commodity': hs_code,
            'naics22_2023': '',
            'naics17_2022': ','.join(matching_naics17),
            'overlap_type': 'only_2022'
        })

# Save consolidated overlap analysis
if all_overlap_rows:
    all_overlap_df = pd.DataFrame(all_overlap_rows)
    overlap_csv_path = get_data_path('validation', '01_Schott_Data_Compiler', '2_hs_overlap_analysis_all_naics.csv')
    all_overlap_df.to_csv(overlap_csv_path, index=False)
    print(f"\nConsolidated overlap analysis saved to: {overlap_csv_path}")
    print(f"Total rows in consolidated analysis: {len(all_overlap_df)}")

# Create simplified HS commodity mapping analysis (addresses the filtering issue)
print("\n" + "="*50)
print("CREATING SIMPLIFIED HS COMMODITY MAPPING ANALYSIS")
print("="*50)

# Get all unique HS commodities from both years
all_hs_commodities_2023 = set(hs_concord[2023]['commodity'].unique())
all_hs_commodities_2022 = set(hs_concord[2022]['commodity'].unique())
all_hs_commodities = all_hs_commodities_2023.union(all_hs_commodities_2022)

# Create mapping analysis for each HS commodity
hs_mapping_rows = []

for hs_code in all_hs_commodities:
    # Find NAICS mapping in 2023 data
    naics_2023_matches = hs_concord[2023][hs_concord[2023]['commodity'] == hs_code]['naics'].unique()
    naics_2023_str = ','.join(naics_2023_matches) if len(naics_2023_matches) > 0 else ''
    
    # Find NAICS mapping in 2022 data
    naics_2022_matches = hs_concord[2022][hs_concord[2022]['commodity'] == hs_code]['naics'].unique()
    naics_2022_str = ','.join(naics_2022_matches) if len(naics_2022_matches) > 0 else ''
    
    # Determine overlap type
    if naics_2023_str and naics_2022_str:
        overlap_type = 'both_years'
    elif naics_2023_str and not naics_2022_str:
        overlap_type = 'only_2023'
    elif not naics_2023_str and naics_2022_str:
        overlap_type = 'only_2022'
    else:
        continue  # Skip if no mapping in either year
    
    hs_mapping_rows.append({
        'hs_commodity': hs_code,
        'naics_2023': naics_2023_str,
        'naics_2022': naics_2022_str,
        'overlap_type': overlap_type
    })

# Save simplified HS mapping analysis
if hs_mapping_rows:
    hs_mapping_df = pd.DataFrame(hs_mapping_rows)
    
    # Filter to include:
    # 1. Cases where NAICS codes are different between years (Case 1, Case 2)
    # 2. Cases where commodity appears in only one year (potential Case 3)
    # 3. Cases where NAICS codes are the same but we have both only_2023 and only_2022 (Case 3)
    
    different_naics_mappings = hs_mapping_df[
        (hs_mapping_df['naics_2023'] != hs_mapping_df['naics_2022']) & 
        (hs_mapping_df['naics_2023'] != '') & 
        (hs_mapping_df['naics_2022'] != '')
    ].copy()
    
    # Cases where commodity appears in only one year
    one_year_only = hs_mapping_df[
        (hs_mapping_df['overlap_type'] == 'only_2023') | 
        (hs_mapping_df['overlap_type'] == 'only_2022')
    ].copy()
    
    # Cases where NAICS codes are the same but we have HS changes (potential Case 3)
    same_naics_with_changes = hs_mapping_df[
        (hs_mapping_df['naics_2023'] == hs_mapping_df['naics_2022']) & 
        (hs_mapping_df['naics_2023'] != '') & 
        (hs_mapping_df['naics_2022'] != '') &
        (hs_mapping_df['overlap_type'].isin(['only_2023', 'only_2022']))
    ].copy()
    
    # Combine all types of mappings we want to analyze
    filtered_df = pd.concat([different_naics_mappings, one_year_only, same_naics_with_changes]).drop_duplicates()
    filtered_df = filtered_df.sort_values(['naics_2023', 'naics_2022'])
    
    hs_mapping_csv_path = get_data_path('validation', '01_Schott_Data_Compiler', '3_hs_mapping_analysis_simplified.csv')
    filtered_df.to_csv(hs_mapping_csv_path, index=False)
    print(f"\nSimplified HS mapping analysis saved to: {hs_mapping_csv_path}")
    print(f"Total HS commodities with mapping issues: {len(filtered_df)}")
    print(f"Original total HS commodities: {len(hs_mapping_df)}")
    print(f"Commodities with consistent mappings (filtered out): {len(hs_mapping_df) - len(filtered_df)}")
    
    # Print distinct NAICS counts for problematic cases only
    distinct_naics_2023 = filtered_df[filtered_df['naics_2023'] != '']['naics_2023'].nunique()
    distinct_naics_2022 = filtered_df[filtered_df['naics_2022'] != '']['naics_2022'].nunique()
    print(f"Distinct NAICS codes in 2023 (problematic only): {distinct_naics_2023}")
    print(f"Distinct NAICS codes in 2022 (problematic only): {distinct_naics_2022}")
    
    # Print summary statistics for problematic cases
    overlap_summary = filtered_df['overlap_type'].value_counts()
    print("\nOverlap type summary (problematic cases only):")
    for overlap_type, count in overlap_summary.items():
        print(f"  {overlap_type}: {count} HS commodities")

def assign_unmapped_hs_via_hierarchy(naics_2023_code, hs_mapping_df):
    """
    For a given NAICS 2023 code with some HS commodities with no naics22 mappings, assign unmapped HS commodities
    to NAICS 2022 codes using hierarchical HS code matching.
    
    Args:
        naics_2023_code: The NAICS 2023 code to process
        hs_mapping_df: DataFrame with HS commodity mappings
        
    Returns:
        Dictionary with assignments and diagnostic information
    """
    # Get all HS commodities for this NAICS 2023 code
    hs_for_naics = hs_mapping_df[hs_mapping_df['naics_2023'] == naics_2023_code].copy()
    
    # Split into mapped and unmapped
    mapped_hs = hs_for_naics[hs_for_naics['overlap_type'] == 'both_years'].copy()
    unmapped_hs = hs_for_naics[hs_for_naics['overlap_type'] == 'only_2023'].copy()
    
    if len(unmapped_hs) == 0:
        return {
            'assignments': [],
            'summary': {
                'multiple_mappings_available': False,
                'hs_level_used': None,
                'mapping_strength': None
            }
        }
    
    def create_hierarchical_codes(hs_code):
        """Create HS-8, HS-6, and HS-4 codes from full HS code"""
        hs_str = str(hs_code)
        length = len(hs_str)
        
        if length == 9:
            return {
                'hs8': hs_str[:7],
                'hs6': hs_str[:5], 
                'hs4': hs_str[:3]
            }
        elif length == 10:
            return {
                'hs8': hs_str[:8],
                'hs6': hs_str[:6],
                'hs4': hs_str[:4]
            }
        else:
            # Handle other lengths
            return {
                'hs8': hs_str[:min(8, length)],
                'hs6': hs_str[:min(6, length)],
                'hs4': hs_str[:min(4, length)]
            }
    
    # Add hierarchical codes to mapped HS commodities
    mapped_hs['hierarchy'] = mapped_hs['hs_commodity'].apply(create_hierarchical_codes)
    
    # For each unmapped HS commodity, try to find a match
    assignments = []
    level_used_counts = {}
    all_unique_naics = set()
    
    for _, unmapped_row in unmapped_hs.iterrows():
        hs_code = unmapped_row['hs_commodity']
        hierarchy = create_hierarchical_codes(hs_code)
        assigned_naics_2022 = None
        match_level = None
        match_details = None
        # Try each hierarchical level (HS-8, then HS-6, then HS-4)
        for level in ['hs8', 'hs6', 'hs4']:
            target_code = hierarchy[level]
            # Find mapped HS commodities with same hierarchical code
            matches = []
            matching_commodities = []
            for _, mapped_row in mapped_hs.iterrows():
                if mapped_row['hierarchy'][level] == target_code:
                    # Extract NAICS 2022 codes (handle comma-separated)
                    naics_2022_codes = [code.strip() for code in mapped_row['naics_2022'].split(',') if code.strip()]
                    matches.extend(naics_2022_codes)
                    matching_commodities.append(mapped_row['hs_commodity'])
            if matches:
                # Take modal (most common) value
                from collections import Counter
                counter = Counter(matches)
                modal_naics_2022 = counter.most_common(1)[0][0]
                modal_count = counter.most_common(1)[0][1]
                # Calculate mapping strength
                total_matches = len(matches)
                mapping_strength = modal_count / total_matches if total_matches > 0 else 0
                # Count total HS commodities at this level for this NAICS 2023
                total_hs_at_level = 0
                for _, mapped_row in mapped_hs.iterrows():
                    if mapped_row['hierarchy'][level] == target_code:
                        total_hs_at_level += 1
                
                assigned_naics_2022 = modal_naics_2022
                match_level = level
                match_details = {
                    'unique_naics_options': len(counter),
                    'modal_count': modal_count,
                    'total_matches': total_matches,
                    'mapping_strength': mapping_strength,
                    'total_hs_at_level': total_hs_at_level
                }
                all_unique_naics.update(counter.keys())
                break
        assignments.append({
            'naics_2023': naics_2023_code,
            'hs_commodity': hs_code,
            'assigned_naics_2022': assigned_naics_2022,
            'match_level': match_level,
            'match_details': match_details
        })
        
        if match_level:
            level_used_counts[match_level] = level_used_counts.get(match_level, 0) + 1
    
    # Calculate summary statistics
    successful_assignments = [a for a in assignments if a['assigned_naics_2022'] is not None]
    
    if successful_assignments:
        # Most common level used
        most_common_level = max(level_used_counts, key=level_used_counts.get) if level_used_counts else None
        avg_mapping_strength = sum(a['match_details']['mapping_strength'] for a in successful_assignments) / len(successful_assignments)
        multiple_mappings = len(all_unique_naics) > 1
    else:
        most_common_level = None
        avg_mapping_strength = None
        multiple_mappings = False
    
    return {
        'assignments': assignments,
        'summary': {
            'multiple_mappings_available': multiple_mappings,
            'hs_level_used': most_common_level,
            'mapping_strength': avg_mapping_strength
        }
    }

###### ATTEMPT TO CREATE THE NAICSMDS MAPPING WHICH IS A 2017 NAICS MAPPING FOR THE 2023 HS CODES  ##########

"""
How should we approach these mappings? Well, we really only have a bea mapping for the 2017 naics codes, so we can impute the 
2017 naics codes mappings by just applying them directly to the HS commodities. This should work <=> there are not HUGE changes in the
HS codes. For those cases, we will need to be a bit more discretionary and maybe even do some manual work. Looking at the validation data (validations/01_Schott_Data_Compiler/2_hs_overlap_analysis_all_naics.csv)

First, we NEED to have the mappings for the 2017 NAICS codes in order to map to the BEA codes. So how can we construct this? 

a.) if it is an HS code that shows up in the 2017 data AND the 2023 data, we can just use the 2017 mapping (we create a new kind of pseudo-2023 mapping)... 
b.) if it is an old HS code that does not exist in the 2023 data, we can see what the old mapping used to be and see if that is mapped to a different value. 
c.) if it is a new hs code that does not appear in the 2017 data, we can figure out a mapping to it based on other codes around it at the HS-8, HS-6, etc levels. 

"""

def construct_corrected_naics_mappings():
    """
    Construct corrected NAICS mappings using a comprehensive 3-step approach:
    1. Direct 2017 mapping (if HS code exists in 2022 data)
    2. Simple crosswalk mapping (if 2023 NAICS maps to exactly one 2017 NAICS)
    3. Hierarchical HS matching (for complex many-to-many cases)
    
    Creates a new 'naicsMDS' column with corrected mappings.
    """
    print("\n" + "="*60)
    print("CONSTRUCTING CORRECTED NAICS MAPPINGS")
    print("="*60)
    
    # Start with 2023 data as base
    corrected_df = hs_concord[2023].copy()
    corrected_df['naicsMDS'] = None
    corrected_df['mapping_method'] = None
    
    # Create lookup dictionaries for efficiency
    hs_2022_to_naics = {row['commodity']: row['naics'] for _, row in hs_concord[2022].iterrows()}
    
    step1_count = 0
    step2_count = 0
    step3_count = 0
    
    print(f"Total HS commodities in 2023: {len(corrected_df)}")
    
    # Step 1: Direct 2017 mapping
    print("\nStep 1: Direct 2017 mapping...")
    for idx, row in corrected_df.iterrows():
        hs_code = row['commodity']
        
        # Check if this HS code exists in 2022 data
        if hs_code in hs_2022_to_naics:
            naics_2017 = hs_2022_to_naics[hs_code]
            corrected_df.at[idx, 'naicsMDS'] = naics_2017
            corrected_df.at[idx, 'mapping_method'] = 'direct_2017'
            step1_count += 1
    
    print(f"  Applied direct 2017 mappings: {step1_count}")
    
    # Step 2: Simple crosswalk mapping
    print("\nStep 2: Simple crosswalk mapping...")
    unmapped_rows = corrected_df[corrected_df['naicsMDS'].isna()]
    
    for idx, row in unmapped_rows.iterrows():
        naics_2023 = row['naics']
        
        try:
            # Check if this NAICS 2023 code maps to exactly one NAICS 2017 code
            naics_2017_options = naics_cw[naics_cw['naics22'] == int(naics_2023)]['naics17'].unique()
            
            if len(naics_2017_options) == 1:
                corrected_df.at[idx, 'naicsMDS'] = str(naics_2017_options[0])
                corrected_df.at[idx, 'mapping_method'] = 'simple_crosswalk'
                step2_count += 1
        except (ValueError, IndexError):
            # Handle cases where NAICS code can't be converted to int or not found
            continue
    
    print(f"  Applied simple crosswalk mappings: {step2_count}")
    
    # Step 3: Hierarchical HS matching for remaining unmapped cases
    print("\nStep 3: Hierarchical HS matching...")
    still_unmapped = corrected_df[corrected_df['naicsMDS'].isna()]
    
    if len(still_unmapped) > 0:
        # Group unmapped HS codes by their NAICS 2023 code
        unmapped_by_naics = still_unmapped.groupby('naics')['commodity'].apply(list).to_dict()
        
        for naics_2023, hs_codes in unmapped_by_naics.items():
            # Check if this NAICS 2023 has multiple NAICS 2017 mappings
            try:
                naics_2017_options = naics_cw[naics_cw['naics22'] == int(naics_2023)]['naics17'].unique()
                
                if len(naics_2017_options) > 1:
                    # Use hierarchical matching for this NAICS 2023 code
                    hierarchical_result = assign_unmapped_hs_via_hierarchy(naics_2023, hs_mapping_df)
                    
                    if hierarchical_result and hierarchical_result['assignments']:
                        for assignment in hierarchical_result['assignments']:
                            if assignment['assigned_naics_2022'] is not None:
                                # Find the corresponding row in corrected_df
                                mask = corrected_df['commodity'] == assignment['hs_commodity']
                                if mask.any():
                                    corrected_df.loc[mask, 'naicsMDS'] = assignment['assigned_naics_2022']
                                    corrected_df.loc[mask, 'mapping_method'] = 'hierarchical_hs'
                                    step3_count += 1
                else:
                    # Shouldn't happen (should have been caught in Step 2), but handle gracefully
                    if len(naics_2017_options) == 1:
                        for hs_code in hs_codes:
                            mask = corrected_df['commodity'] == hs_code
                            if mask.any():
                                corrected_df.loc[mask, 'naicsMDS'] = str(naics_2017_options[0])
                                corrected_df.loc[mask, 'mapping_method'] = 'late_simple_crosswalk'
                                step2_count += 1
            except (ValueError, IndexError):
                continue
    
    print(f"  Applied hierarchical HS mappings: {step3_count}")
    
    # Summary statistics
    final_unmapped = corrected_df[corrected_df['naicsMDS'].isna()]
    print(f"\nFinal mapping summary:")
    print(f"  Direct 2017 mappings: {step1_count}")
    print(f"  Simple crosswalk mappings: {step2_count}")
    print(f"  Hierarchical HS mappings: {step3_count}")
    print(f"  Still unmapped: {len(final_unmapped)}")
    print(f"  Total mapped: {step1_count + step2_count + step3_count}")
    print(f"  Success rate: {(step1_count + step2_count + step3_count) / len(corrected_df) * 100:.1f}%")
    
    # Show method breakdown
    method_counts = corrected_df['mapping_method'].value_counts()
    print(f"\nMapping method breakdown:")
    for method, count in method_counts.items():
        print(f"  {method}: {count}")
    
    # Create validation CSV for mapping issues and hierarchical details
    create_mapping_validation_csv(corrected_df, hs_2022_to_naics)
    
    return corrected_df

def create_mapping_validation_csv(corrected_df, hs_2022_to_naics):
    """
    Create validation CSV identifying mapping issues and hierarchical matching details.
    """
    print(f"\nCreating mapping validation CSV...")
    
    validation_rows = []
    
    for idx, row in corrected_df.iterrows():
        hs_code = row['commodity']
        naics_2023 = row['naics']
        mapping_method = row['mapping_method']
        naics_mds = row['naicsMDS']
        
        # Base row data
        row_data = {
            'hs_commodity': hs_code,
            'naics_2023': naics_2023,
            'naics_mds': naics_mds,
            'mapping_method': mapping_method,
            'issue_type': None,
            'issue_description': None,
            'hs_level_used': None,
            'mapping_strength': None,
            'multiple_naics_options': None
        }
        
        # Check for Issue 1: New HS codes with no historical context
        if hs_code not in hs_2022_to_naics and naics_mds is None:
            # Check if NAICS 2023 exists in crosswalk
            try:
                naics_2017_options = naics_cw[naics_cw['naics22'] == int(naics_2023)]['naics17'].unique()
                if len(naics_2017_options) == 0:
                    row_data['issue_type'] = 'issue_1'
                    row_data['issue_description'] = 'New HS code with NAICS not in crosswalk'
            except (ValueError, IndexError):
                row_data['issue_type'] = 'issue_1'
                row_data['issue_description'] = 'New HS code with invalid NAICS format'
        
        # Check for Issue 2: Hierarchical matching failures
        elif mapping_method is None and naics_mds is None:
            # This HS code should have been handled by hierarchical matching but wasn't
            try:
                naics_2017_options = naics_cw[naics_cw['naics22'] == int(naics_2023)]['naics17'].unique()
                if len(naics_2017_options) > 1:
                    row_data['issue_type'] = 'issue_2'
                    row_data['issue_description'] = 'Hierarchical matching failed - no similar HS codes found'
                    row_data['multiple_naics_options'] = len(naics_2017_options)
            except (ValueError, IndexError):
                pass
        
        # For successful hierarchical matches, get detailed matching info
        elif mapping_method == 'hierarchical_hs' and naics_mds is not None:
            # Run hierarchical matching again to get details
            hierarchical_result = assign_unmapped_hs_via_hierarchy(naics_2023, hs_mapping_df)
            
            if hierarchical_result and hierarchical_result['assignments']:
                # Find the assignment for this specific HS code
                for assignment in hierarchical_result['assignments']:
                    if assignment['hs_commodity'] == hs_code and assignment['assigned_naics_2022'] == naics_mds:
                        if assignment['match_details']:
                            row_data['hs_level_used'] = assignment['match_level']
                            row_data['mapping_strength'] = round(assignment['match_details']['mapping_strength'], 3)
                            row_data['multiple_naics_options'] = assignment['match_details']['unique_naics_options']
                        break
        
        validation_rows.append(row_data)
    
    # Create DataFrame and save
    validation_df = pd.DataFrame(validation_rows)
    
    # Filter to only include rows with issues or hierarchical details
    filtered_validation = validation_df[
        (validation_df['issue_type'].notna()) | 
        (validation_df['mapping_method'] == 'hierarchical_hs')
    ].copy()
    
    # Sort by issue type and HS commodity
    filtered_validation = filtered_validation.sort_values(['issue_type', 'hs_commodity'])
    
    # Save to CSV
    validation_csv_path = get_data_path('validation', '01_Schott_Data_Compiler', '4_mapping_validation_issues.csv')
    filtered_validation.to_csv(validation_csv_path, index=False)
    
    print(f"Mapping validation CSV saved to: {validation_csv_path}")
    print(f"Total validation records: {len(filtered_validation)}")
    
    # Print summary statistics
    issue_counts = filtered_validation['issue_type'].value_counts()
    print(f"\nIssue breakdown:")
    for issue, count in issue_counts.items():
        if issue:
            print(f"  {issue}: {count}")
    
    hierarchical_count = len(filtered_validation[filtered_validation['mapping_method'] == 'hierarchical_hs'])
    print(f"  Hierarchical matches with details: {hierarchical_count}")
    
    return filtered_validation

# Create corrected mappings on the whole dataset
print("RUNNING CORRECTED MAPPINGS ON FULL 2023 DATASET")
corrected_mappings = construct_corrected_naics_mappings()
print(f"\nCreated corrected mappings with {len(corrected_mappings)} rows")
# Save the corrected mappings to the working directory
corrected_output_dir = get_data_path('working', '01_Schott_Data_Compiler')
corrected_output_path = os.path.join(corrected_output_dir, '03_hs_naics_mapping_2023_corrected_naicsMDS.csv')
corrected_mappings.to_csv(corrected_output_path, index=False)
print(f"Corrected mappings saved to: {corrected_output_path}")
# Show some example rows
print("\nExample corrected mappings:")
print(corrected_mappings[['commodity', 'naics', 'naicsMDS', 'mapping_method']].head(10))

# Final validation of the corrected mappings
print("\n" + "="*60)
print("FINAL VALIDATION OF CORRECTED MAPPINGS")
print("="*60)

def create_final_validation():
    """
    Create final validation of corrected mappings to ensure:
    1. All naicsMDS codes exist in 2017 NAICS data
    2. HS commodity count matches 2023 raw data
    3. Count mapping methods used
    """
    validation_results = []
    
    # Load 2022 NAICS codes from the actual mapping file for validation
    naics_2022_mapping_path = os.path.join(get_data_path('working', '01_Schott_Data_Compiler'), '01_hs_naics_mapping_2022_imports.csv')
    naics_2022_df = pd.read_csv(naics_2022_mapping_path)
    valid_naics_2022 = set(naics_2022_df['naics'].astype(str))
    
    # Load raw 2023 data for comparison
    raw_2023_df = pd.read_csv(os.path.join(get_data_path('working', '01_Schott_Data_Compiler'), '02_hs_naics_mapping_2023_imports.csv'))
    raw_hs_count = len(raw_2023_df['commodity'].unique())
    
    validation_results.append("="*60)
    validation_results.append("FINAL VALIDATION REPORT - CORRECTED SCHOTT MAPPINGS")
    validation_results.append("="*60)
    validation_results.append("")
    
    # Check 1: Validate all naicsMDS codes exist in 2022 NAICS mapping
    validation_results.append("CHECK 1: NAICS 2022 CODE VALIDATION")
    validation_results.append("-" * 40)
    
    corrected_naics_mds = corrected_mappings['naicsMDS'].dropna().astype(str)
    invalid_naics = corrected_naics_mds[~corrected_naics_mds.isin(valid_naics_2022)]
    
    if len(invalid_naics) == 0:
        validation_results.append("✓ PASS: All naicsMDS codes exist in 2022 NAICS mapping")
    else:
        validation_results.append(f"✗ FAIL: {len(invalid_naics)} invalid naicsMDS codes found:")
        for naics in invalid_naics.unique()[:10]:  # Show first 10
            validation_results.append(f"  - {naics}")
        if len(invalid_naics.unique()) > 10:
            validation_results.append(f"  ... and {len(invalid_naics.unique()) - 10} more")
    
    validation_results.append("")
    
    # Check 2: Validate HS commodity count matches raw data
    validation_results.append("CHECK 2: HS COMMODITY COUNT VALIDATION")
    validation_results.append("-" * 40)
    
    corrected_hs_count = len(corrected_mappings['commodity'].unique())
    
    if corrected_hs_count == raw_hs_count:
        validation_results.append("✓ PASS: HS commodity count matches raw 2023 data")
    else:
        validation_results.append(f"✗ FAIL: HS commodity count mismatch")
        validation_results.append(f"  Raw 2023 data: {raw_hs_count} unique HS codes")
        validation_results.append(f"  Corrected data: {corrected_hs_count} unique HS codes")
        validation_results.append(f"  Difference: {corrected_hs_count - raw_hs_count}")
    
    validation_results.append("")
    
    # Check 3: Count mapping methods
    validation_results.append("CHECK 3: MAPPING METHOD BREAKDOWN")
    validation_results.append("-" * 40)
    
    method_counts = corrected_mappings['mapping_method'].value_counts()
    total_mapped = len(corrected_mappings[corrected_mappings['naicsMDS'].notna()])
    total_unmapped = len(corrected_mappings[corrected_mappings['naicsMDS'].isna()])
    
    validation_results.append(f"Total HS commodities: {len(corrected_mappings)}")
    validation_results.append(f"Successfully mapped: {total_mapped}")
    validation_results.append(f"Unmapped: {total_unmapped}")
    validation_results.append(f"Success rate: {(total_mapped / len(corrected_mappings) * 100):.1f}%")
    validation_results.append("")
    
    validation_results.append("Mapping method counts:")
    for method, count in method_counts.items():
        if method:
            percentage = (count / len(corrected_mappings)) * 100
            validation_results.append(f"  {method}: {count} ({percentage:.1f}%)")
    
    validation_results.append("")
    
    # Add method descriptions
    validation_results.append("MAPPING METHOD DESCRIPTIONS")
    validation_results.append("-" * 40)
    validation_results.append("")
    
    validation_results.append("1. direct_2017:")
    validation_results.append("   - HS commodity exists in both 2023 and 2022 data")
    validation_results.append("   - Uses the existing 2022 NAICS mapping directly")
    validation_results.append("   - Most reliable method as it uses historical mappings")
    validation_results.append("")
    
    validation_results.append("2. simple_crosswalk:")
    validation_results.append("   - HS commodity only exists in 2023 data")
    validation_results.append("   - 2023 NAICS code maps to exactly one 2017 NAICS code")
    validation_results.append("   - Uses official NAICS 2022-to-2017 crosswalk")
    validation_results.append("")
    
    validation_results.append("3. hierarchical_hs:")
    validation_results.append("   - HS commodity only exists in 2023 data")
    validation_results.append("   - 2023 NAICS code maps to multiple 2017 NAICS codes")
    validation_results.append("   - Uses hierarchical matching (HS-8, HS-6, HS-4 levels)")
    validation_results.append("   - Finds similar HS codes and uses modal 2017 NAICS mapping")
    validation_results.append("")
    
    validation_results.append("="*60)
    validation_results.append("VALIDATION COMPLETE")
    validation_results.append("="*60)
    
    return validation_results

# Create and save final validation
final_validation = create_final_validation()
validation_output_path = get_data_path('validation', '01_Schott_Data_Compiler', '5_FINAL_schott_validation.txt')

with open(validation_output_path, 'w') as f:
    f.write('\n'.join(final_validation))

print(f"Final validation saved to: {validation_output_path}")
print("\nFinal validation summary:")
for line in final_validation:
    if line.startswith('✓') or line.startswith('✗') or 'Success rate:' in line:
        print(line)
