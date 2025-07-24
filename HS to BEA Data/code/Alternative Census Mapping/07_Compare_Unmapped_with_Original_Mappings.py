# Compare unmapped HS codes from Alternative Census mapping with original Schott mappings
# This script enhances the 07_comprehensive_unmapped_records.csv with original Schott mapping data
# to understand mapping differences between the two approaches

import json
import os
import pandas as pd

# Load data paths configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
data_paths_file = os.path.join(script_dir, '..', '..', 'data_paths.json')

with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

def load_unmapped_records():
    """Load the comprehensive unmapped records from Alternative Census mapping"""
    print("Loading unmapped records from Alternative Census mapping...")
    
    validation_base = data_paths['validation_outputs']['base_path']
    validation_subdir = data_paths['validation_outputs']['subdirectories']['Alternative_Census_Mappings']
    unmapped_path = os.path.join(
        data_paths['base_paths']['underlying_data_root'], 
        validation_base, 
        validation_subdir, 
        '07_comprehensive_unmapped_records.csv'
    )
    
    unmapped_df = pd.read_csv(unmapped_path, dtype={'hs_code': str})
    print(f"Loaded {len(unmapped_df):,} unmapped HS codes")
    print(f"Total unmapped import value: ${unmapped_df['impVal'].sum():,.0f}")
    
    return unmapped_df

def load_original_schott_mappings():
    """Load the original Schott HS-to-BEA mappings"""
    print("Loading original Schott HS-to-BEA mappings...")
    
    schott_path = os.path.join(
        data_paths['base_paths']['underlying_data_root'],
        'data', 'working', '02_HS_to_Naics_to_BEA', '03_complete_hs_to_bea_mapping.csv'
    )
    
    schott_df = pd.read_csv(schott_path, dtype={'commodity': str})
    
    # Rename commodity column to hs_code for consistency
    schott_df = schott_df.rename(columns={'commodity': 'hs_code'})
    
    # Select only the columns we need
    schott_columns = ['hs_code', 'matched_bea_naics', 'matched_bea_detail', 'naicsMDS']
    schott_df = schott_df[schott_columns].copy()
    
    # Add prefix to column names to distinguish from Alternative Census mapping
    schott_df = schott_df.rename(columns={
        'matched_bea_naics': 'original_bea_naics',
        'matched_bea_detail': 'original_bea_code', 
        'naicsMDS': 'original_naics_mds'
    })
    
    print(f"Loaded {len(schott_df):,} original Schott mappings")
    
    return schott_df

def merge_and_analyze():
    """Merge unmapped records with original mappings and analyze differences"""
    print("\n" + "=" * 70)
    print("MERGING UNMAPPED RECORDS WITH ORIGINAL SCHOTT MAPPINGS")
    print("=" * 70)
    
    # Load data
    unmapped_df = load_unmapped_records()
    schott_df = load_original_schott_mappings()
    
    # Merge the datasets
    print("Merging unmapped records with original Schott mappings...")
    merged_df = unmapped_df.merge(
        schott_df, 
        on='hs_code', 
        how='left'
    )
    
    print(f"Successfully merged: {len(merged_df):,} records")
    
    # Calculate match statistics
    has_original_mapping = merged_df['original_bea_code'].notna()
    print(f"HS codes with original Schott mappings: {has_original_mapping.sum():,} ({has_original_mapping.mean()*100:.1f}%)")
    print(f"HS codes without original mappings: {(~has_original_mapping).sum():,}")
    
    # Analyze differences by mapping type
    print("\n" + "=" * 50)
    print("ANALYSIS BY ORIGINAL BEA CODE")
    print("=" * 50)
    
    # Group by original BEA code to see patterns
    bea_analysis = merged_df[has_original_mapping].groupby('original_bea_code').agg({
        'impVal': ['sum', 'count'],
        'hs_code': 'nunique'
    }).round(0)
    
    # Flatten column names
    bea_analysis.columns = ['total_import_value', 'record_count', 'unique_hs_codes']
    bea_analysis = bea_analysis.reset_index()
    bea_analysis = bea_analysis.sort_values('total_import_value', ascending=False)
    
    print("Top original BEA codes for unmapped HS codes:")
    for _, row in bea_analysis.head(15).iterrows():
        print(f"  {row['original_bea_code']}: ${row['total_import_value']:,.0f} "
              f"({row['unique_hs_codes']:,.0f} HS codes, {row['record_count']:,.0f} records)")
    
    # Analyze noncomparable imports vs. real industries
    print("\n" + "=" * 50)
    print("NONCOMPARABLE VS. REAL INDUSTRY MAPPINGS")
    print("=" * 50)
    
    # Check how many unmapped codes had real industry mappings in original
    noncomparable_codes = ['S00300', '910000', '930000', '980000', '990000']
    real_industry_mask = ~merged_df['original_bea_code'].isin(noncomparable_codes) & merged_df['original_bea_code'].notna()
    
    real_industry_count = real_industry_mask.sum()
    real_industry_value = merged_df[real_industry_mask]['impVal'].sum()
    
    print(f"Unmapped HS codes that had REAL INDUSTRY mappings in original: {real_industry_count:,}")
    print(f"Import value for real industry losses: ${real_industry_value:,.0f}")
    
    if real_industry_count > 0:
        print(f"\nTop real industry mappings lost in Alternative Census approach:")
        real_industry_analysis = merged_df[real_industry_mask].groupby('original_bea_code').agg({
            'impVal': 'sum',
            'hs_code': 'nunique'
        }).reset_index()
        real_industry_analysis = real_industry_analysis.sort_values('impVal', ascending=False)
        
        for _, row in real_industry_analysis.head(10).iterrows():
            print(f"  {row['original_bea_code']}: ${row['impVal']:,.0f} ({row['hs_code']} HS codes)")
    
    # Save the enhanced dataset
    print("\n" + "=" * 50)
    print("SAVING ENHANCED UNMAPPED RECORDS")
    print("=" * 50)
    
    validation_base = data_paths['validation_outputs']['base_path']
    validation_subdir = data_paths['validation_outputs']['subdirectories']['Alternative_Census_Mappings']
    output_path = os.path.join(
        data_paths['base_paths']['underlying_data_root'], 
        validation_base, 
        validation_subdir, 
        '08_unmapped_records_with_original_mappings.csv'
    )
    
    # Reorder columns for better readability
    column_order = [
        'hs_code', 
        'naics_2022_mapped', 
        'naics_2024', 
        'mapping_source',
        'original_bea_naics',
        'original_bea_code', 
        'original_naics_mds',
        'impVal'
    ]
    
    merged_df[column_order].to_csv(output_path, index=False)
    print(f"Enhanced unmapped records saved: {output_path}")
    print(f"Columns included: {', '.join(column_order)}")
    
    # Create summary comparison
    print("\n" + "=" * 50)
    print("SUMMARY COMPARISON")
    print("=" * 50)
    
    # Compare Alternative Census mapping results vs Original Schott results
    alt_census_noncomparable = merged_df['naics_2022_mapped'].isin(['910000', '930000', '980000', '990000'])
    orig_noncomparable = merged_df['original_bea_code'].isin(['S00300', '910000', '930000', '980000', '990000'])
    
    print("Mapping approach comparison:")
    print(f"  Alternative Census → Noncomparable: {alt_census_noncomparable.sum():,} HS codes, ${merged_df[alt_census_noncomparable]['impVal'].sum():,.0f}")
    print(f"  Original Schott → Noncomparable: {orig_noncomparable.sum():,} HS codes, ${merged_df[orig_noncomparable]['impVal'].sum():,.0f}")
    
    # Find cases where approaches differ
    different_approaches = (alt_census_noncomparable & ~orig_noncomparable) | (~alt_census_noncomparable & orig_noncomparable)
    print(f"  HS codes with different classification approaches: {different_approaches.sum():,}")
    
    if different_approaches.sum() > 0:
        print(f"  Import value affected by different approaches: ${merged_df[different_approaches]['impVal'].sum():,.0f}")
    
    return merged_df, bea_analysis

def main():
    """Main execution function"""
    print("COMPARING UNMAPPED HS CODES WITH ORIGINAL SCHOTT MAPPINGS")
    print("=" * 70)
    
    # Create output directory if needed
    validation_base = data_paths['validation_outputs']['base_path']
    validation_subdir = data_paths['validation_outputs']['subdirectories']['Alternative_Census_Mappings']
    validation_path = os.path.join(data_paths['base_paths']['underlying_data_root'], validation_base, validation_subdir)
    os.makedirs(validation_path, exist_ok=True)
    
    # Run analysis
    merged_df, bea_analysis = merge_and_analyze()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("This analysis helps identify:")
    print("  1. Which unmapped HS codes had real industry mappings in the original approach")
    print("  2. Whether the Alternative Census mapping is more/less restrictive")
    print("  3. Specific industry sectors most affected by mapping differences")
    print(f"\nOutput saved to: validations/Alternative_Census_Mappings/08_unmapped_records_with_original_mappings.csv")

if __name__ == "__main__":
    main()