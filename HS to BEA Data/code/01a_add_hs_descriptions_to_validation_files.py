import os
import pandas as pd
from main_pipeline_run import get_data_path

"""
DESCRIPTION: This script adds HS descriptions to the Schott Data Compiler validation files
to make them more readable and useful for analysis.

The script enhances:
1. 4_mapping_validation_issues.csv - HS codes that went through hierarchical mapping
2. 3_hs_mapping_analysis_simplified.csv - HS codes with 2022/2023 overlap issues

It uses the same HS description source as 3a_hierarchical_mapping_HS_codes.csv
(01_hs10_descriptions_full.csv from the trade data mapping).
"""

def add_hs_descriptions():
    print("Adding HS descriptions to Schott Data Compiler validation files...")
    
    # Load HS descriptions
    try:
        hs_desc_path = os.path.join(get_data_path('working', '03_Map_country_trade_data'), '01_hs10_descriptions_full.csv')
        hs_descriptions = pd.read_csv(hs_desc_path)
        hs_descriptions = hs_descriptions.rename(columns={'hs10': 'hs_commodity'})
        print(f"Loaded {len(hs_descriptions)} HS descriptions")
        
    except Exception as e:
        print(f"Could not load HS descriptions: {e}")
        return
    
    # Process 4_mapping_validation_issues.csv
    validation_issues_path = os.path.join(get_data_path('validation', '01_Schott_Data_Compiler'), '4_mapping_validation_issues.csv')
    try:
        validation_issues = pd.read_csv(validation_issues_path)
        print(f"Loaded {len(validation_issues)} validation issues")
        
        # Add HS descriptions
        validation_issues_enhanced = validation_issues.merge(
            hs_descriptions[['hs_commodity', 'description']],
            on='hs_commodity',
            how='left'
        )
        validation_issues_enhanced['hs_description'] = validation_issues_enhanced['description'].fillna('No description available')
        validation_issues_enhanced = validation_issues_enhanced.drop(columns=['description'])
        
        # Reorder columns to put description right after hs_commodity
        cols = list(validation_issues_enhanced.columns)
        cols.remove('hs_description')
        cols.insert(1, 'hs_description')
        validation_issues_enhanced = validation_issues_enhanced[cols]
        
        # Save enhanced file
        enhanced_output_path = validation_issues_path.replace('.csv', '_with_descriptions.csv')
        validation_issues_enhanced.to_csv(enhanced_output_path, index=False)
        
        descriptions_added = len(validation_issues_enhanced[validation_issues_enhanced['hs_description'] != 'No description available'])
        print(f"Enhanced 4_mapping_validation_issues.csv: Added descriptions for {descriptions_added}/{len(validation_issues)} HS codes")
        print(f"Saved to: {enhanced_output_path}")
        
    except Exception as e:
        print(f"Could not process 4_mapping_validation_issues.csv: {e}")
    
    # Process 3_hs_mapping_analysis_simplified.csv
    mapping_analysis_path = os.path.join(get_data_path('validation', '01_Schott_Data_Compiler'), '3_hs_mapping_analysis_simplified.csv')
    try:
        mapping_analysis = pd.read_csv(mapping_analysis_path)
        print(f"Loaded {len(mapping_analysis)} mapping analysis records")
        
        # Add HS descriptions
        mapping_analysis_enhanced = mapping_analysis.merge(
            hs_descriptions[['hs_commodity', 'description']],
            on='hs_commodity',
            how='left'
        )
        mapping_analysis_enhanced['hs_description'] = mapping_analysis_enhanced['description'].fillna('No description available')
        mapping_analysis_enhanced = mapping_analysis_enhanced.drop(columns=['description'])
        
        # Reorder columns to put description right after hs_commodity
        cols = list(mapping_analysis_enhanced.columns)
        cols.remove('hs_description')
        cols.insert(1, 'hs_description')
        mapping_analysis_enhanced = mapping_analysis_enhanced[cols]
        
        # Save enhanced file
        enhanced_output_path = mapping_analysis_path.replace('.csv', '_with_descriptions.csv')
        mapping_analysis_enhanced.to_csv(enhanced_output_path, index=False)
        
        descriptions_added = len(mapping_analysis_enhanced[mapping_analysis_enhanced['hs_description'] != 'No description available'])
        print(f"Enhanced 3_hs_mapping_analysis_simplified.csv: Added descriptions for {descriptions_added}/{len(mapping_analysis)} HS codes")
        print(f"Saved to: {enhanced_output_path}")
        
        # Show summary statistics for mapping analysis
        print(f"\nMapping analysis summary:")
        overlap_counts = mapping_analysis_enhanced['overlap_type'].value_counts()
        for overlap_type, count in overlap_counts.items():
            print(f"  - {overlap_type}: {count} HS codes")
            
    except Exception as e:
        print(f"Could not process 3_hs_mapping_analysis_simplified.csv: {e}")
    
    print("\nHS descriptions added successfully!")
    
    # Show some examples from the validation issues (hierarchical mapping cases)
    try:
        print(f"\nFirst 10 hierarchical mapping examples from validation issues:")
        validation_issues_sample = validation_issues_enhanced[validation_issues_enhanced['mapping_method'] == 'hierarchical_hs'].head(10)
        print(validation_issues_sample[['hs_commodity', 'hs_description', 'naics_2023', 'naics_mds', 'mapping_method']].to_string(index=False))
        
        # Show statistics
        hierarchical_count = len(validation_issues_enhanced[validation_issues_enhanced['mapping_method'] == 'hierarchical_hs'])
        print(f"\nTotal HS codes that used hierarchical mapping: {hierarchical_count}")
        
    except Exception as e:
        print(f"Could not show examples: {e}")

if __name__ == "__main__":
    add_hs_descriptions()