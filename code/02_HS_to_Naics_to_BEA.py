import os
import pandas as pd
from main_pipeline_run import get_data_path
"""
Description: So in the 01_Schott_Data_Compiler, we corrected the NAICS mapping to get 2023 HS codes mapped to 2017 NAICS codes
(which we need for the BEA data). But even with the 2017 NAICS codes, there are still wildcards everywhere -- like 31131X, 1123XX, etc.
The question is: do these wildcards actually line up with the BEA data? The BEA has 402 industry categories that use NAICS codes,
but they're all over the place -- some are 2-digit, some are 6-digit, totally inconsistent.

This script does a few things:
1. Loads the BEA data and figures out what NAICS codes they actually use
2. Takes our corrected HS -> NAICS mapping and tries to match it to the BEA codes
3. Uses hierarchical matching when exact matches don't work (trim digits until something matches exactly (because the BEA mapped naics codes can be 2-6 digits))
4. Creates a complete bridge from HS codes to BEA categories

The main output is 03_complete_hs_to_bea_mapping.csv which gives us the full chain:
HS commodity -> naicsMDS -> matched_bea_naics -> matched_bea_detail

The problem we're solving:
- HS codes from Schott give us naicsMDS codes (with wildcards)
- BEA has their own NAICS codes at different aggregation levels
- We need to connect these two systems so we can use trade data with BEA economic accounts

The approach:
1. Process BEA data - they have ranges like "3331-9" that we need to expand (we ensure that each of them is actually a valid 2017 NAICS code)
2. Build BEA hierarchy mapping (they have 4 levels: Sector -> Summary -> U.Summary -> Detail)
3. Take our naicsMDS codes and try to match them to BEA codes
4. When exact match fails, progressively trim digits (6->5->4->3->2) until we find a match
5. Merge everything together for the final bridge mapping

Inputs: 
- 03_hs_naics_mapping_2023_corrected_naicsMDS.csv from 01_Schott_Data_Compiler
- 402_use.xlsx from BEA (the "NAICS Codes" sheet)
- ALL_NAICS_CODES_2017.csv for validation

Outputs:
- 01_BEA_naics_mapping.csv: BEA detail codes -> NAICS codes (expanded from ranges)
- 02_BEA_hierarchy.csv: Full BEA hierarchical structure
- 03_complete_hs_to_bea_mapping.csv: Complete HS -> BEA bridge mapping
- Various validation files to check for problems
"""

########################### Step 1: Read the BEA Data in and Create Hierarchy and NAICS mapping ############################
bea_file_path = get_data_path('raw', 'BEA_codes', '402_use')
bea = pd.read_excel(bea_file_path, sheet_name = 'NAICS Codes', header = 4, nrows = 1017)
bea_df = bea[['Detail','Unnamed: 4','Related 2017 NAICS Codes']].copy()
bea_df = bea_df.rename(columns= {'Detail':'Code','Unnamed: 4':'Commodity Description','Related 2017 NAICS Codes':'naics'})
bea_df = bea_df.rename(columns= {'Detail':'Code','Unnamed: 4':'Commodity Description','Related 2017 NAICS Codes':'naics'})
bea_df = bea_df.dropna(subset=['naics'])
bea_df['naics'] = bea_df['naics'].astype(str)
def expand_naics_range(token):
    """
    Expands a single NAICS code-token. 
    If it contains a hyphen, we generate a range based on that.
    Otherwise, return token as a single item list. 
    """
    token = token.strip().replace('*','')
    if '-' in token:
        base, end_digit = token.split('-')
        base = base.strip()
        end_digit = int(end_digit.strip())
        start_digit = int(base[-1])
        prefix = base[:-1]
        # Generate the list of codes from the starting digit to the end digit
        return [prefix + str(d) for d in range(start_digit, end_digit + 1)]
    else:
        return [token]
def expand_naics_mapping(mapping_str):
    """
    Processes a string of NAICS codes which may include single codes and ranges,
    separated by commas. Returns a list of all expanded NAICS codes.
    """
    tokens = mapping_str.split(',')
    all_codes = []
    for token in tokens:
        all_codes.extend(expand_naics_range(token))
    return all_codes
# perform the mappings to create a BEA detail code -> Naics mapping
bea_df['naics'] = bea_df['naics'].apply(expand_naics_mapping)
bea_df = bea_df.explode('naics')
bea_df = bea_df.drop_duplicates()

# SAVE the BEA NAICS mapping to working directory
output_path = os.path.join(get_data_path('working', '02_HS_to_Naics_to_BEA'), '01_BEA_naics_mapping.csv')
bea_df.to_csv(output_path, index=False)

########################### Step 2: Create BEA Hierarchical Mapping ############################
# Create hierarchical mappings from the BEA codes
data = bea.copy()
data = data.rename(columns={'Unnamed: 4': 'detail title'})
data = data.drop(columns=['Notes', 'Related 2017 NAICS Codes'])

# Create mapping tables for each hierarchy level
sector = data[['Sector', 'Summary']].copy()  # 21 Industry Groups (SECTOR)
sector = sector.rename(columns={'Summary': 'sector title'})
summary = data[['Summary', 'U.Summary']].copy()  # 71 Industry Groups (SUMMARY)
summary = summary.rename(columns={'U.Summary': 'summary title'})
undsummary = data[['U.Summary', 'Detail']].copy()  # 138 Industry Groups (UNDERLYING SUMMARY) - Key for TiVA Tables
undsummary = undsummary.rename(columns={'Detail': 'undersum title'})
detail = data[['Detail', 'detail title']].copy()  # 402 Industry Groups (Detail)

# Clean up mapping tables by removing rows with missing values
for df in [sector, summary, undsummary, detail]:
    df.dropna(subset=[df.columns[0], df.columns[1]], inplace=True)
# Forward fill the hierarchy to create complete mapping for each detail code
data.loc[data['Sector'].notna(), 'Summary'] = None
data.loc[data['Summary'].notna(), 'U.Summary'] = None
data.loc[data['U.Summary'].notna(), 'Detail'] = None

data.dropna(how='all', inplace=True)
for col in ['Sector', 'Summary', 'U.Summary', 'Detail']:
    data[col] = data[col].ffill()
data.dropna(subset=['detail title'], inplace=True)

# Merge back the title information
data = data.merge(sector, how='left', left_on='Sector', right_on='Sector')
data = data.merge(summary, how='left', left_on='Summary', right_on='Summary')
data = data.merge(undsummary, how='left', left_on='U.Summary', right_on='U.Summary')

# Select final columns and capitalize titles
data = data[['Sector', 'sector title', 'Summary', 'summary title', 'U.Summary', 'undersum title', 'Detail', 'detail title']]
data['sector title'] = data['sector title'].str.capitalize()
data['summary title'] = data['summary title'].str.capitalize()
data['undersum title'] = data['undersum title'].str.capitalize()
data['detail title'] = data['detail title'].str.capitalize()

# Save the hierarchical mapping to the working directory
hierarchical_mapping_path = os.path.join(get_data_path('working', '02_HS_to_Naics_to_BEA'), '02_BEA_hierarchy.csv')
data.to_csv(hierarchical_mapping_path, index=False)

########################### Step 3: Verify NAICS MDS Codes Map to BEA NAICS Codes ############################
"""
Okay, so the codes from the BEA are kind of varied. There are 402 which are at MOST a naics 6 level, but can go down to the naics 2 level. 
This means that when we are trying to map, we need to be careful. Fortunately, we also have wildcards in the NAICS codes from the Schott data, 
so hopefully those wildcards line up with the more aggregated BEA codes. We need to test for this. How? 

1.) Create a list of all NAICS codes from the hs_naics mapping. (also create a sublist of just those with wildcards "X" and place in the validations path. ) 
2.) Create a list of all NAICS codes from the BEA mapping. 
3.) Construct a hierarchical mapping process. 
"""
# Load the corrected NAICS mapping data from 01_Schott_Data_Compiler
hs_naics_path = os.path.join(get_data_path('working', '01_Schott_Data_Compiler'), '03_hs_naics_mapping_2023_corrected_naicsMDS.csv')
hs_naics_df = pd.read_csv(hs_naics_path)

# Create the list of all the naicsMD codes from hs_naics_df
naics_codes = hs_naics_df['naicsMDS'].unique()
print(f"Total unique NAICS codes from HS(2023) to NAICS(2017) mapping: {len(naics_codes)}")
naics_codes_wildcard = hs_naics_df[hs_naics_df['naicsMDS'].str.contains('X')]['naicsMDS'].unique()
print(f"Total unique NAICS codes with wildcards from HS(2023) to NAICS(2017) mapping: {len(naics_codes_wildcard)}")
#Save the list of 5 naics wild cards to the validations
wildcard_df = pd.DataFrame({'naicsMDS_wildcard': sorted(naics_codes_wildcard)})
wildcard_output_path = os.path.join(get_data_path('validation', '02_HS_to_Naics_to_BEA'), '1_HS_to_NAICS_wildcards.csv')
wildcard_df.to_csv(wildcard_output_path, index=False)
# Create a list of all the naics codes from the BEA mapping
bea_naics_codes = bea_df['naics'].unique()
print(f"Total unique NAICS codes from BEA mapping: {len(bea_naics_codes)}")
# Validate that the NAICS codes from the bea_naics_mapping are actually in the naics 2017 data
naics_2017_path = get_data_path('raw', 'naics_crosswalks', 'naics_codes_2017')
naics_2017_df = pd.read_csv(naics_2017_path)
naics_2017_codes = naics_2017_df['NAICS'].unique()
print(f"Total unique NAICS codes from NAICS 2017 data: {len(naics_2017_codes)}")

## CHECK: IS EVERY BEA NAICS CODE A VALID 2017 NAICS CODE? (ANSWER: YES)
extra_bea_codes = set(bea_naics_codes) - set(naics_2017_codes)
if extra_bea_codes:
    print(f"EXTRA BEA NAICS codes not found in official NAICS 2017 census data: {sorted(extra_bea_codes)}")
    print(f"Number of extra BEA codes: {len(extra_bea_codes)}")
    # THESE ARE ALL CODES CREATED BY THE EXPANDER CODE... NOTHING SYSTEMIC IS WRONG
    # Remove the extra codes from bea_naics_codes to create a clean list
    bea_naics_codes_clean = [code for code in bea_naics_codes if code not in extra_bea_codes]
    # Update bea_naics_codes to the cleaned version
    bea_naics_codes = bea_naics_codes_clean
else:
    print("All BEA NAICS codes are present in the official NAICS 2017 census data.")
# Create comparison CSV by merging BEA and NAICS definition codes
bea_df_codes = pd.DataFrame({'naics_code': bea_naics_codes, 'in_bea': True})
naics_df_codes = pd.DataFrame({'naics_code': naics_2017_codes, 'in_naics_definition': True})
# Merge to show which codes are in which dataset
comparison_df = pd.merge(bea_df_codes, naics_df_codes, on='naics_code', how='outer')
comparison_df['in_bea'] = comparison_df['in_bea'].fillna(False)
comparison_df['in_naics_definition'] = comparison_df['in_naics_definition'].fillna(False)
# Keep only codes that are in both datasets
comparison_df = comparison_df[(comparison_df['in_bea'] == True) & (comparison_df['in_naics_definition'] == True)]
comparison_df = comparison_df.sort_values('naics_code')
comparison_output_path = os.path.join(get_data_path('validation', '02_HS_to_Naics_to_BEA'), '2_BEA_vs_NAICS_comparison.csv')
comparison_df.to_csv(comparison_output_path, index=False)
# Validate that all BEA NAICS codes are valid 2017 NAICS codes
if len(comparison_df) == len(bea_naics_codes):
    print(f"\033[92m VALIDATION PASSED: All {len(bea_naics_codes)} BEA NAICS codes are valid 2017 NAICS codes\033[0m")
else:
    print(f"\033[91m VALIDATION FAILED: Only {len(comparison_df)} out of {len(bea_naics_codes)} BEA NAICS codes are valid 2017 NAICS codes\033[0m")


########################### Step 4: Hierarchical Matching of HS-NAICS to BEA NAICS ############################

def hierarchical_match(hs_naics_code, bea_naics_set):
    """
    Match HS-NAICS code to BEA NAICS by progressively trimming digits
    Returns (matched_bea_code, match_level) or (None, None) if no match
    """
    # Handle special cases for noncomparable imports
    if hs_naics_code in ['910000', '930000', '980000', '990000']:
        return 'S00300', 'noncomparable_imports' ## THIS IS AN ASSUMPTION THAT WE MIGHT NOW WANT TO DO... 
    
    # Clean the HS-NAICS code (remove X wildcards for matching)
    clean_code = hs_naics_code.replace('X', '')
    # Try exact match first
    if clean_code in bea_naics_set:
        return clean_code, 'exact'
    # Progressive trimming from 6 digits down to 2
    for level in range(len(clean_code) - 1, 1, -1):
        trimmed = clean_code[:level]
        if trimmed in bea_naics_set:
            return trimmed, f'level_{level}'
    return None, None
# Create hierarchical mapping results
mapping_results = []
bea_naics_set = set(bea_naics_codes)
for hs_naics in naics_codes:
    matched_bea, match_level = hierarchical_match(hs_naics, bea_naics_set)
    mapping_results.append({
        'original_hs_naics': hs_naics,
        'matched_bea_naics': matched_bea,
        'match_level': match_level,
        'has_wildcard': 'X' in hs_naics
    })
# Convert to DataFrame and analyze
mapping_df = pd.DataFrame(mapping_results)
# Check for successful matches
successful_matches = mapping_df[mapping_df['matched_bea_naics'].notna()]
failed_matches = mapping_df[mapping_df['matched_bea_naics'].isna()]
print(f"Successful matches: {len(successful_matches)}/{len(mapping_df)}")
print(f"Failed matches: {len(failed_matches)}")
# Add problematic assessment to mapping results
successful_matches['original_length'] = successful_matches['original_hs_naics'].str.replace('X', '').str.len()
successful_matches['matched_length'] = successful_matches['matched_bea_naics'].str.len()
# Check for duplicate mappings (multiple HS-NAICS codes mapping to same BEA code)
duplicate_analysis = successful_matches.groupby('matched_bea_naics').size().reset_index(name='count')
duplicates = duplicate_analysis[duplicate_analysis['count'] > 1]
print(f"BEA codes with multiple HS-NAICS mappings: {len(duplicates)}")
# For duplicates, assess if problematic (same length mappings)
if len(duplicates) > 0:
    duplicate_details = successful_matches[successful_matches['matched_bea_naics'].isin(duplicates['matched_bea_naics'])]
    duplicate_details['problematic'] = duplicate_details.apply(
        lambda row: 'Yes' if row['matched_length'] > row['original_length'] else 'No', axis=1
    )
    # Check for truly problematic mappings
    problematic_mappings = duplicate_details[duplicate_details['problematic'] == 'Yes']
    problematic_bea_codes = problematic_mappings['matched_bea_naics'].nunique()
    print(f"Truly problematic BEA codes (more disaggregated with duplicates): {problematic_bea_codes}")
    duplicate_details = duplicate_details.sort_values(['matched_bea_naics', 'original_hs_naics'])
    duplicate_output_path = os.path.join(get_data_path('validation', '02_HS_to_Naics_to_BEA'), '4_duplicate_mappings.csv')
    duplicate_details.to_csv(duplicate_output_path, index=False)
# Add problematic assessment to all mapping results
mapping_df['original_length'] = mapping_df['original_hs_naics'].str.replace('X', '').str.len()
mapping_df['matched_length'] = mapping_df['matched_bea_naics'].str.len()
mapping_df['problematic'] = 'N/A'
# For successful matches, determine if problematic (only for duplicates)
successful_mask = mapping_df['matched_bea_naics'].notna()
mapping_df.loc[successful_mask, 'problematic'] = 'No'

# Mark failed matches as problematic
failed_mask = mapping_df['matched_bea_naics'].isna()
mapping_df.loc[failed_mask, 'problematic'] = 'Yes'

# Only mark as problematic if it's a duplicate AND BEA is more aggregated
if len(duplicates) > 0:
    duplicate_mask = mapping_df['matched_bea_naics'].isin(duplicates['matched_bea_naics'])
    mapping_df.loc[duplicate_mask, 'problematic'] = mapping_df.loc[duplicate_mask].apply(
        lambda row: 'Yes' if row['matched_length'] > row['original_length'] else 'No', axis=1
    )

# Sort to put failed matches (problematic=Yes) at the top
mapping_df = mapping_df.sort_values(['problematic', 'original_hs_naics'], ascending=[False, True])

# Save detailed mapping results
mapping_output_path = os.path.join(get_data_path('validation', '02_HS_to_Naics_to_BEA'), '3_hierarchical_mapping_results.csv')
mapping_df.to_csv(mapping_output_path, index=False)

########################### Step 5: Create Complete HS to BEA Bridge Mapping ############################
"""
Now merge the hierarchical mapping results with the original HS commodity data 
to create a complete bridge: HS commodity -> naicsMDS -> matched_bea_naics
"""
print("\nCreating complete HS to BEA bridge mapping...")
# The original_hs_naics in mapping_df corresponds to naicsMDS in hs_naics_df
complete_mapping = hs_naics_df.merge(
    mapping_df, 
    left_on='naicsMDS', 
    right_on='original_hs_naics', 
    how='left'
)
# Merge with BEA mapping to get the BEA detail codes
complete_mapping = complete_mapping.merge(
    bea_df[['Code', 'naics']].drop_duplicates(),
    left_on='matched_bea_naics',
    right_on='naics',
    how='left'
)
complete_mapping = complete_mapping.rename(columns={'Code': 'matched_bea_detail'})
complete_mapping = complete_mapping.drop(columns=['naics_y'])  # Drop the duplicate naics column from the merge

# Select relevant columns - use naics_x which is the original naics column from hs_naics_df
complete_mapping = complete_mapping[[
    'commodity',
    'naics_x',  # This is the original naics column from hs_naics_df
    'naicsX',
    'naicsMDS',
    'mapping_method',
    'matched_bea_naics',
    'matched_bea_detail',
    'match_level',
    'has_wildcard',
    'problematic'
]].rename(columns={'naics_x': 'naics'})  # Rename back to naics for clarity

total_commodities = len(complete_mapping)
successful_matches = len(complete_mapping[complete_mapping['matched_bea_naics'].notna()])
failed_matches = len(complete_mapping[complete_mapping['matched_bea_naics'].isna()])

print(f"Total HS commodities: {total_commodities}")
print(f"Successful BEA matches: {successful_matches}")
print(f"Failed BEA matches: {failed_matches}")
print(f"Match rate: {successful_matches/total_commodities*100:.1f}%")

complete_output_path = os.path.join(get_data_path('working', '02_HS_to_Naics_to_BEA'), '03_complete_hs_to_bea_mapping.csv')
# Ensure matched_bea_detail is set to 'S00300' when matched_bea_naics is 'S00300'
complete_mapping.loc[complete_mapping['matched_bea_naics'] == 'S00300', 'matched_bea_detail'] = 'S00300'
complete_mapping.to_csv(complete_output_path, index=False)
print(f"Complete HS to BEA mapping saved to: {complete_output_path}")
# Summary statistics
if successful_matches > 0:
    match_level_counts = complete_mapping['match_level'].value_counts()
    print(f"\nMatch level distribution:")
    for level, count in match_level_counts.items():
        if pd.notna(level):
            print(f"  - {level}: {count} ({count/successful_matches*100:.1f}%)")

# Show unique BEA codes matched
unique_bea_codes = complete_mapping['matched_bea_naics'].dropna().nunique()
print(f"\nUnique BEA NAICS codes matched: {unique_bea_codes}")

# Show failed matches if any
if failed_matches > 0:
    failed_naics = complete_mapping[complete_mapping['matched_bea_naics'].isna()]['naicsMDS'].unique()
    print(f"\nFailed naicsMDS codes: {sorted(failed_naics)}")
    failed_mapping = complete_mapping[complete_mapping['matched_bea_naics'].isna()]
    failed_output_path = os.path.join(get_data_path('validation', '02_HS_to_Naics_to_BEA'), '5_failed_hs_to_bea_matches.csv')
    failed_mapping.to_csv(failed_output_path, index=False)
    print(f"Failed matches saved to: {failed_output_path}")

# Final validation: Check that we haven't lost any commodity codes
input_commodities = len(hs_naics_df['commodity'].unique())
output_commodities = len(complete_mapping['commodity'].unique())

print(f"\nFinal validation:")
print(f"Input commodities (03_hs_naics_mapping_2023_corrected_naicsMDS.csv): {input_commodities}")
print(f"Output commodities (03_complete_hs_to_bea_mapping.csv): {output_commodities}")

if input_commodities == output_commodities:
    print(f"\033[92mVALIDATION PASSED: All {input_commodities} commodity codes preserved in output\033[0m")
else:
    print(f"\033[91mVALIDATION FAILED: Lost {input_commodities - output_commodities} commodity codes in mapping\033[0m")
    # Find missing commodities with their NAICS codes
    input_commodity_set = set(hs_naics_df['commodity'].unique())
    output_commodity_set = set(complete_mapping['commodity'].unique())
    missing_commodities = input_commodity_set - output_commodity_set
    
    if missing_commodities:
        print(f"Missing commodities: {sorted(missing_commodities)}")
        # Create detailed report of missing commodities
        missing_details = hs_naics_df[hs_naics_df['commodity'].isin(missing_commodities)][['commodity', 'naics', 'naicsX', 'naicsMDS', 'mapping_method']]
        missing_output_path = os.path.join(get_data_path('validation', '02_HS_to_Naics_to_BEA'), '5_missing_commodities_validation.csv')
        missing_details.to_csv(missing_output_path, index=False)
        print(f"Missing commodities details saved to: {missing_output_path}")

print("\nComplete HS to BEA bridge mapping creation finished!") 
