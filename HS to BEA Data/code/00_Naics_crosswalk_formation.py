import os
import pandas as pd
from main_pipeline_run import get_data_path

"""
DESCRIPTION: This code creates a complete NAICS 2022 to 2017 crosswalk by combining:
1. The official crosswalk file (NAICS22-17_cw.csv) - contains mappings for codes that changed
2. Self-mappings for codes that exist in both years but didn't change (missing from official crosswalk but present in both the 2017 and 2022 lists)

The issue: The official crosswalk only includes codes that changed between 2017 and 2022, 
but many codes stayed the same and thus aren't included. This causes mapping failures 
for valid NAICS codes like 111422 that exist in both years.

Inputs: {
    ALL_NAICS_CODES_2017.csv: Complete list of 2017 NAICS codes
    ALL_NAICS_CODES_2022.csv: Complete list of 2022 NAICS codes  
    NAICS22-17_cw.csv: Official crosswalk for changed codes
}

Outputs: {
    NAICS22-17_cw_complete.csv: Complete crosswalk including unchanged codes
}
"""

print("="*60)
print("CREATING COMPLETE NAICS 2022-2017 CROSSWALK")
print("="*60)

# Load the NAICS code files
print("\nLoading NAICS code files...")
naics_2017_path = get_data_path('raw', 'naics_crosswalks', 'naics_codes_2017')
naics_2022_path = get_data_path('raw', 'naics_crosswalks', 'naics_codes_2022')
crosswalk_path = get_data_path('raw', 'naics_crosswalks', 'naics_2022_to_2017_crosswalk')

naics_2017 = pd.read_csv(naics_2017_path)
naics_2022 = pd.read_csv(naics_2022_path)
official_crosswalk = pd.read_csv(crosswalk_path)

# Assume the NAICS code column is named 'naics' - adjust if different
naics_2017_col = 'naics' if 'naics' in naics_2017.columns else naics_2017.columns[0]
naics_2022_col = 'naics' if 'naics' in naics_2022.columns else naics_2022.columns[0]

# Filter to 6-digit codes only
print(f"\nFiltering to 6-digit NAICS codes only...")
naics_2017_6digit = naics_2017[naics_2017[naics_2017_col].astype(str).str.len() == 6].copy()
naics_2022_6digit = naics_2022[naics_2022[naics_2022_col].astype(str).str.len() == 6].copy()

# Find codes that exist in both years (these should map to themselves if not in official crosswalk)
common_codes = set(naics_2017_6digit[naics_2017_col].astype(str)) & set(naics_2022_6digit[naics_2022_col].astype(str))

# Check which common codes are already in the official crosswalk
official_crosswalk_codes = set(official_crosswalk['naics22'].astype(str))
missing_from_crosswalk = common_codes - official_crosswalk_codes

# Create self-mappings for missing codes (naics22 -> naics17 = same code)
missing_mappings = []
for code in missing_from_crosswalk:
    missing_mappings.append({
        'naics22': int(code),
        'naics': int(code),  # Assuming 'naics' is the naics17 column name
        'mapping_type': 'self_mapping'
    })

missing_mappings_df = pd.DataFrame(missing_mappings)

# Add mapping type to official crosswalk
official_crosswalk['mapping_type'] = 'official_crosswalk'

# Ensure column names match
if 'naics' not in official_crosswalk.columns:
    # Find the column that represents naics17
    naics17_col = [col for col in official_crosswalk.columns if 'naics' in col.lower() and col != 'naics22'][0]
    official_crosswalk = official_crosswalk.rename(columns={naics17_col: 'naics'})

# Combine official crosswalk with missing mappings
complete_crosswalk = pd.concat([official_crosswalk, missing_mappings_df], ignore_index=True)

# Sort by naics22 for easier inspection
complete_crosswalk = complete_crosswalk.sort_values('naics22')

# Verify no duplicates -- There will be duplicates from the actual crosswalk when they decide to change the mappings,but there shouldnt be any from the actual tables
duplicates = complete_crosswalk['naics22'].duplicated().sum()
if duplicates > 0:
    print(f"WARNING: {duplicates} duplicate naics22 codes found.")
    print("Duplicate codes:")
    print(complete_crosswalk[complete_crosswalk['naics22'].duplicated(keep=False)].sort_values('naics22'))
else:
    print("No duplicate naics22 codes found - good!")

# Save the complete crosswalk
output_path = get_data_path('raw', 'naics_crosswalks')
complete_crosswalk_path = os.path.join(output_path, 'NAICS22-17_cw_complete.csv')
complete_crosswalk.to_csv(complete_crosswalk_path, index=False)
# Summary statistics
mapping_type_counts = complete_crosswalk['mapping_type'].value_counts()
print(f"\nMapping type summary:")
for mapping_type, count in mapping_type_counts.items():
    print(f"  {mapping_type}: {count}")
