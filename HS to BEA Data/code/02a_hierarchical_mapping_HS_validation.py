import os
import pandas as pd
from main_pipeline_run import get_data_path

"""
DESCRIPTION: This script creates a validation dataset showing HS codes that went through 
hierarchical mapping in the 02_HS_to_NAICS_to_BEA.py process.

The hierarchical mapping occurs when NAICS codes from the HS-to-NAICS mapping don't 
exactly match BEA NAICS codes, requiring progressive digit trimming (6->5->4->3->2 digits)
until a match is found.

This script creates 3a_hierarchical_mapping_HS_codes.csv which shows:
- hs_code: The original HS commodity code
- hs_description: Description of the HS code (when available)
- original_naics: The original NAICS code from HS-to-NAICS mapping
- matched_bea_naics: The BEA NAICS code found through hierarchical matching
- matched_bea_detail: The corresponding BEA detail code
- match_type: 'primary' (the applied mapping) or 'alternative' (showing the original before trimming)

The style follows 03_Map_country_trade_data.py's 3_Hierarchical_Matches.csv format.
"""

def create_hierarchical_hs_validation():
    print("Creating hierarchical mapping HS codes validation dataset...")
    
    # Load the complete HS to BEA mapping (output of 02_HS_to_NAICS_to_BEA.py)
    complete_mapping_path = os.path.join(get_data_path('working', '02_HS_to_Naics_to_BEA'), '03_complete_hs_to_bea_mapping.csv')
    complete_mapping = pd.read_csv(complete_mapping_path)
    
    # Load the hierarchical mapping results to understand the matching process
    hierarchical_results_path = os.path.join(get_data_path('validation', '02_HS_to_Naics_to_BEA'), '3_hierarchical_mapping_results.csv')
    hierarchical_results = pd.read_csv(hierarchical_results_path)
    
    print(f"Loaded {len(complete_mapping)} HS commodity mappings")
    print(f"Loaded {len(hierarchical_results)} hierarchical mapping results")
    
    # Filter for codes that went through hierarchical matching (not exact matches)
    hierarchical_codes = hierarchical_results[
        hierarchical_results['match_level'].notna() & 
        (hierarchical_results['match_level'] != 'exact')
    ].copy()
    
    print(f"Found {len(hierarchical_codes)} NAICS codes that required hierarchical matching")
    
    # Join with complete mapping to get HS commodity codes
    hs_hierarchical = complete_mapping.merge(
        hierarchical_codes,
        left_on='naicsMDS',
        right_on='original_hs_naics',
        how='inner',
        suffixes=('', '_hier')
    )
    
    print(f"Found {len(hs_hierarchical)} HS commodity codes involved in hierarchical mapping")
    
    # Get HS descriptions from the trade data mapping
    try:
        hs_desc_path = os.path.join(get_data_path('working', '03_Map_country_trade_data'), '01_hs10_descriptions_full.csv')
        hs_descriptions = pd.read_csv(hs_desc_path)
        hs_descriptions = hs_descriptions.rename(columns={'hs10': 'commodity'})
        
        hs_hierarchical = hs_hierarchical.merge(
            hs_descriptions[['commodity', 'description']],
            on='commodity',
            how='left'
        )
        hs_hierarchical['hs_description'] = hs_hierarchical['description'].fillna('No description available')
        print(f"Added descriptions for {len(hs_hierarchical[hs_hierarchical['description'].notna()])} HS codes")
        
    except Exception as e:
        print(f"Could not load HS descriptions: {e}")
        hs_hierarchical['hs_description'] = 'No description available'
    
    # Create the validation dataset in the style of 3_Hierarchical_Matches.csv
    validation_data = []
    
    for _, row in hs_hierarchical.iterrows():
        # Add the primary mapping (the one that was actually used)
        validation_data.append({
            'hs_code': row['commodity'],
            'hs_description': row.get('hs_description', 'No description available'),
            'original_naics': row['naicsMDS'],
            'matched_bea_naics': row['matched_bea_naics'],
            'matched_bea_detail': row['matched_bea_detail'],
            'match_level': row['match_level'],
            'has_wildcard': row['has_wildcard'],
            'match_type': 'primary'
        })
        
        # Add the alternative mapping showing what the original NAICS was before trimming
        if row['naicsMDS'] != row['matched_bea_naics']:
            validation_data.append({
                'hs_code': row['commodity'],
                'hs_description': row.get('hs_description', 'No description available'),
                'original_naics': row['naicsMDS'],
                'matched_bea_naics': row['naicsMDS'],  # Show the original as alternative
                'matched_bea_detail': 'N/A - No direct BEA mapping',
                'match_level': 'original',
                'has_wildcard': row['has_wildcard'],
                'match_type': 'alternative'
            })
    
    validation_df = pd.DataFrame(validation_data)
    
    # Sort by HS code and match type to group primary/alternative pairs
    validation_df = validation_df.sort_values(['hs_code', 'match_type'])
    
    # Filter out unnecessary "N/A - No direct BEA mapping" alternatives
    print("Filtering out unnecessary N/A alternative mappings...")
    
    # Group by HS code to analyze alternatives
    filtered_data = []
    for hs_code in validation_df['hs_code'].unique():
        hs_entries = validation_df[validation_df['hs_code'] == hs_code]
        
        primary_entry = hs_entries[hs_entries['match_type'] == 'primary']
        alternative_entries = hs_entries[hs_entries['match_type'] == 'alternative']
        
        # Always keep the primary mapping
        filtered_data.extend(primary_entry.to_dict('records'))
        
        # Logic for alternatives:
        if len(alternative_entries) > 0:
            # If primary mapping is "N/A", keep all alternatives
            if len(primary_entry) > 0 and 'N/A - No direct BEA mapping' in primary_entry['matched_bea_detail'].values:
                filtered_data.extend(alternative_entries.to_dict('records'))
            else:
                # If there are multiple alternatives, keep all
                if len(alternative_entries) > 1:
                    filtered_data.extend(alternative_entries.to_dict('records'))
                else:
                    # If only one alternative and it's NOT "N/A", keep it
                    # If only one alternative and it IS "N/A", skip it (this is the main filtering logic)
                    single_alt = alternative_entries.iloc[0]
                    if single_alt['matched_bea_detail'] != 'N/A - No direct BEA mapping':
                        filtered_data.append(single_alt.to_dict())
                    # Skip the "N/A" alternative when it's the only one and primary worked
    
    validation_df_filtered = pd.DataFrame(filtered_data)
    validation_df_filtered = validation_df_filtered.sort_values(['hs_code', 'match_type'])
    
    print(f"Filtered from {len(validation_df)} to {len(validation_df_filtered)} entries")
    
    # Save the filtered validation dataset
    validation_output_path = os.path.join(get_data_path('validation', '02_HS_to_Naics_to_BEA'), '3a_hierarchical_mapping_HS_codes.csv')
    validation_df_filtered.to_csv(validation_output_path, index=False)
    
    print(f"Created filtered validation dataset with {len(validation_df_filtered)} entries")
    print(f"Saved to: {validation_output_path}")
    
    # Summary statistics
    unique_hs_codes = validation_df_filtered['hs_code'].nunique()
    primary_mappings = len(validation_df_filtered[validation_df_filtered['match_type'] == 'primary'])
    alternative_mappings = len(validation_df_filtered[validation_df_filtered['match_type'] == 'alternative'])
    
    print(f"\nSummary:")
    print(f"- Unique HS codes with hierarchical mappings: {unique_hs_codes}")
    print(f"- Primary mappings (applied): {primary_mappings}")
    print(f"- Alternative mappings (original): {alternative_mappings}")
    
    # Show match level distribution
    if primary_mappings > 0:
        match_level_counts = validation_df_filtered[validation_df_filtered['match_type'] == 'primary']['match_level'].value_counts()
        print(f"\nHierarchical match level distribution:")
        for level, count in match_level_counts.items():
            print(f"  - {level}: {count} HS codes ({count/primary_mappings*100:.1f}%)")
    
    # Show filtering results
    na_alternatives_removed = len(validation_df) - len(validation_df_filtered)
    print(f"\nFiltering results:")
    print(f"- Removed {na_alternatives_removed} unnecessary 'N/A - No direct BEA mapping' alternatives")
    print(f"- Kept {alternative_mappings} meaningful alternative mappings")
    
    # Show some examples
    print(f"\nFirst 10 examples of filtered data:")
    print(validation_df_filtered[['hs_code', 'original_naics', 'matched_bea_naics', 'matched_bea_detail', 'match_level', 'match_type']].head(10))
    
    return validation_df_filtered

if __name__ == "__main__":
    validation_df = create_hierarchical_hs_validation()
    print("\nHierarchical mapping HS codes validation completed!")