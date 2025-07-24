# Validate which 2022 NAICS codes and HS codes are not getting mapped to BEA codes
# This script analyzes the unmapped records to understand what's causing the mapping failures

import json
import os
import pandas as pd
import glob

# Load data paths configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
data_paths_file = os.path.join(script_dir, '..', '..', 'data_paths.json')

with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Set up paths
aggregated_data_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                   'data', 'working', 'Alternative_Census_Mapping', 
                                   '02_aggregated_with_bea_mapping.csv')

combined_data_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                 'data', 'working', 'Alternative_Census_Mapping', 'combined_data')

validation_base = data_paths['validation_outputs']['base_path']
validation_subdir = data_paths['validation_outputs']['subdirectories']['Alternative_Census_Mappings']
validation_path = os.path.join(data_paths['base_paths']['underlying_data_root'], validation_base, validation_subdir)

def analyze_unmapped_naics_codes():
    """Analyze which 2022 NAICS codes are not mapping to BEA codes"""
    print("=" * 70)
    print("ANALYZING UNMAPPED 2022 NAICS CODES")
    print("=" * 70)
    
    # Load the aggregated data
    print(f"Loading aggregated data from: {aggregated_data_path}")
    aggregated_df = pd.read_csv(aggregated_data_path, dtype=str)
    
    # Convert impVal to numeric
    aggregated_df['impVal'] = pd.to_numeric(aggregated_df['impVal'])
    
    print(f"Total records: {len(aggregated_df):,}")
    print(f"Total import value: ${aggregated_df['impVal'].sum():,.0f}")
    
    # Find records where 2022 NAICS didn't map to BEA
    unmapped_2022 = aggregated_df[aggregated_df['bea_2022'].isna()].copy()
    
    print(f"\nUnmapped 2022 NAICS records: {len(unmapped_2022):,}")
    print(f"Unmapped import value: ${unmapped_2022['impVal'].sum():,.0f}")
    
    # Analyze by NAICS code
    naics_analysis = unmapped_2022.groupby('naics_2022_mapped').agg({
        'impVal': ['sum', 'count'],
        'Country': 'nunique'
    }).round(0)
    
    # Flatten column names
    naics_analysis.columns = ['total_import_value', 'record_count', 'country_count']
    naics_analysis = naics_analysis.reset_index()
    naics_analysis = naics_analysis.sort_values('total_import_value', ascending=False)
    
    print(f"\nTop unmapped 2022 NAICS codes by import value:")
    for _, row in naics_analysis.head(10).iterrows():
        print(f"  {row['naics_2022_mapped']}: ${row['total_import_value']:,.0f} "
              f"({row['record_count']:,.0f} records, {row['country_count']:,.0f} countries)")
    
    # Save detailed NAICS analysis
    naics_output_path = os.path.join(validation_path, '05_unmapped_naics_2022_analysis.csv')
    naics_analysis.to_csv(naics_output_path, index=False)
    print(f"\nDetailed NAICS analysis saved: {naics_output_path}")
    
    return unmapped_2022, naics_analysis

def analyze_unmapped_hs_codes():
    """Analyze which HS codes contribute to unmapped records"""
    print("\n" + "=" * 70)
    print("ANALYZING UNMAPPED HS CODES")
    print("=" * 70)
    
    # Load all the combined data files to get HS code details
    print(f"Loading combined trade data from: {combined_data_path}")
    
    trade_files = glob.glob(os.path.join(combined_data_path, "*_with_naics_mapping.csv"))
    print(f"Found {len(trade_files)} trade data files")
    
    # Load and combine all trade data
    all_trade_data = []
    for trade_file in trade_files:
        df = pd.read_csv(trade_file, dtype={'hs_code': str, 'naics_2022_mapped': str})
        df['impVal'] = pd.to_numeric(df['impVal'])
        all_trade_data.append(df)
    
    combined_trade_df = pd.concat(all_trade_data, ignore_index=True)
    print(f"Total trade records: {len(combined_trade_df):,}")
    
    # Load the aggregated data to identify unmapped NAICS codes
    aggregated_df = pd.read_csv(aggregated_data_path, dtype=str)
    unmapped_naics_codes = set(aggregated_df[aggregated_df['bea_2022'].isna()]['naics_2022_mapped'])
    
    print(f"Unmapped 2022 NAICS codes: {len(unmapped_naics_codes)}")
    
    # Filter trade data to only unmapped NAICS codes
    unmapped_trade_df = combined_trade_df[
        combined_trade_df['naics_2022_mapped'].isin(unmapped_naics_codes)
    ].copy()
    
    print(f"Trade records with unmapped NAICS: {len(unmapped_trade_df):,}")
    print(f"Import value for unmapped records: ${unmapped_trade_df['impVal'].sum():,.0f}")
    
    # Analyze by HS code
    hs_analysis = unmapped_trade_df.groupby(['hs_code', 'naics_2022_mapped']).agg({
        'impVal': 'sum',
        'Country': 'nunique'
    }).reset_index()
    
    hs_analysis = hs_analysis.sort_values('impVal', ascending=False)
    hs_analysis.columns = ['hs_code', 'naics_2022_mapped', 'total_import_value', 'country_count']
    
    print(f"\nTop HS codes contributing to unmapped values:")
    for _, row in hs_analysis.head(15).iterrows():
        print(f"  HS {row['hs_code']} (NAICS {row['naics_2022_mapped']}): "
              f"${row['total_import_value']:,.0f} ({row['country_count']} countries)")
    
    # Save detailed HS analysis
    hs_output_path = os.path.join(validation_path, '06_unmapped_hs_codes_analysis.csv')
    hs_analysis.to_csv(hs_output_path, index=False)
    print(f"\nDetailed HS code analysis saved: {hs_output_path}")
    
    # Create summary by NAICS code
    naics_summary = unmapped_trade_df.groupby('naics_2022_mapped').agg({
        'impVal': 'sum',
        'hs_code': 'nunique',
        'Country': 'nunique'
    }).reset_index()
    
    naics_summary.columns = ['naics_2022_mapped', 'total_import_value', 'unique_hs_codes', 'unique_countries']
    naics_summary = naics_summary.sort_values('total_import_value', ascending=False)
    
    print(f"\nUnmapped NAICS summary:")
    for _, row in naics_summary.iterrows():
        print(f"  NAICS {row['naics_2022_mapped']}: ${row['total_import_value']:,.0f} "
              f"({row['unique_hs_codes']} HS codes, {row['unique_countries']} countries)")
    
    return unmapped_trade_df, hs_analysis, naics_summary

def create_comprehensive_unmapped_report():
    """Create a comprehensive report of all unmapped records"""
    print("\n" + "=" * 70)
    print("CREATING COMPREHENSIVE UNMAPPED REPORT")
    print("=" * 70)
    
    # Analyze unmapped NAICS codes from aggregated data
    unmapped_aggregated, naics_analysis = analyze_unmapped_naics_codes()
    
    # Analyze unmapped HS codes from detailed data
    unmapped_trade_df, hs_analysis, naics_summary = analyze_unmapped_hs_codes()
    
    # Validation: Check that totals match
    print(f"\n" + "=" * 70)
    print("VALIDATION: CHECKING TOTALS")
    print("=" * 70)
    
    aggregated_unmapped_value = unmapped_aggregated['impVal'].sum()
    detailed_unmapped_value = unmapped_trade_df['impVal'].sum()
    
    print(f"Unmapped value from aggregated data: ${aggregated_unmapped_value:,.0f}")
    print(f"Unmapped value from detailed data: ${detailed_unmapped_value:,.0f}")
    print(f"Difference: ${abs(aggregated_unmapped_value - detailed_unmapped_value):,.0f}")
    
    if abs(aggregated_unmapped_value - detailed_unmapped_value) < 1000:  # Allow for small rounding differences
        print("✓ VALIDATION PASSED: Totals match!")
    else:
        print("⚠ VALIDATION WARNING: Totals don't match exactly")
    
    # Save comprehensive unmapped records aggregated by HS code
    comprehensive_output_path = os.path.join(validation_path, '07_comprehensive_unmapped_records.csv')
    
    # Aggregate by HS code level (sum import values across countries)
    # First, let's check for any inconsistencies in the grouping columns
    print("Checking for inconsistencies in grouping columns...")
    for hs_code in unmapped_trade_df['hs_code'].unique():
        hs_subset = unmapped_trade_df[unmapped_trade_df['hs_code'] == hs_code]
        unique_naics_2022 = hs_subset['naics_2022_mapped'].nunique()
        unique_naics_2024 = hs_subset['naics_2024'].nunique()
        unique_mapping_source = hs_subset['mapping_source'].nunique()
        
        if unique_naics_2022 > 1 or unique_naics_2024 > 1 or unique_mapping_source > 1:
            print(f"  HS {hs_code}: naics_2022({unique_naics_2022}), naics_2024({unique_naics_2024}), mapping_source({unique_mapping_source})")
    
    # Simple aggregation by HS code only, taking the first values for other columns
    hs_aggregated = unmapped_trade_df.groupby('hs_code').agg({
        'naics_2022_mapped': 'first',
        'naics_2024': 'first', 
        'mapping_source': 'first',
        'impVal': 'sum'
    }).reset_index()
    
    # Sort by import value descending
    hs_aggregated = hs_aggregated.sort_values('impVal', ascending=False)
    
    # Save the aggregated dataset with only the requested columns
    columns_to_save = ['hs_code', 'naics_2022_mapped', 'naics_2024', 'impVal', 'mapping_source']
    hs_aggregated[columns_to_save].to_csv(comprehensive_output_path, index=False)
    
    print(f"Comprehensive unmapped records (aggregated by HS code) saved: {comprehensive_output_path}")
    print(f"  Unique HS codes: {len(hs_aggregated):,}")
    print(f"  Total import value: ${hs_aggregated['impVal'].sum():,.0f}")
    
    # Summary statistics
    print(f"\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total unmapped 2022 NAICS codes: {len(naics_analysis)}")
    print(f"Total unmapped HS codes: {hs_analysis['hs_code'].nunique():,}")
    print(f"Total unmapped import value: ${detailed_unmapped_value:,.0f}")
    print(f"Countries affected: {unmapped_trade_df['Country'].nunique()}")
    print(f"Continents affected: {unmapped_trade_df['continent'].nunique() if 'continent' in unmapped_trade_df.columns else 'N/A'}")

def main():
    """Main execution function"""
    print("VALIDATION: ANALYZING UNMAPPED 2022 NAICS AND HS CODES")
    
    # Create output directory if needed
    os.makedirs(validation_path, exist_ok=True)
    
    # Run comprehensive analysis
    create_comprehensive_unmapped_report()
    
    print(f"\nAll validation files saved to: {validation_path}")

if __name__ == "__main__":
    main()