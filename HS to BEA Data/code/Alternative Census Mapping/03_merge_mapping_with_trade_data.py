# Merge the HS-to-NAICS mapping with combined trade data files
# This script takes the 04_hs_2024_to_naics_2022_mapping.csv and merges it with
# each continent's combined trade data to add NAICS mapping information

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
validation_base = data_paths['validation_outputs']['base_path']
validation_subdir = data_paths['validation_outputs']['subdirectories']['Alternative_Census_Mappings']
validation_path = os.path.join(data_paths['base_paths']['underlying_data_root'], validation_base, validation_subdir)

# Path to the HS-to-NAICS mapping file
mapping_file_path = os.path.join(validation_path, '04_hs_2024_to_naics_2022_mapping.csv')

# Path to the combined trade data files
combined_data_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                 'data', 'working', '03_Map_country_trade_data', 'combined_data')

# Output path for merged files
output_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                          'data', 'working', 'Alternative_Census_Mapping', 'combined_data')

def load_mapping_data(mapping_file_path):
    """Load the HS-to-NAICS mapping data"""
    print(f"Loading mapping data from: {mapping_file_path}")
    
    try:
        mapping_df = pd.read_csv(mapping_file_path, dtype=str)
        
        # Clean up any whitespace
        for col in mapping_df.columns:
            if mapping_df[col].dtype == 'object':
                mapping_df[col] = mapping_df[col].str.strip()
        
        print(f"Loaded {len(mapping_df):,} HS-to-NAICS mappings")
        print(f"Columns: {list(mapping_df.columns)}")
        
        return mapping_df
    except Exception as e:
        print(f"Error loading mapping data: {e}")
        return None

def merge_mapping_with_trade_data(trade_file_path, mapping_df, output_file_path):
    """Merge trade data with HS-to-NAICS mapping"""
    
    continent_name = os.path.basename(trade_file_path).replace('_combined.csv', '')
    print(f"\nProcessing {continent_name} trade data...")
    
    try:
        # Load trade data
        trade_df = pd.read_csv(trade_file_path, dtype={'hs_code': str})
        print(f"  Loaded {len(trade_df):,} trade records")
        
        # Ensure hs_code is properly formatted (10 digits with leading zeros)
        trade_df['hs_code'] = trade_df['hs_code'].str.zfill(10)
        
        # Merge with mapping data on hs_code
        merged_df = trade_df.merge(
            mapping_df, 
            left_on='hs_code', 
            right_on='hs_code_2024', 
            how='left'
        )
        
        # Count successful mappings
        mapped_count = merged_df['naics_2024'].notna().sum()
        unmapped_count = len(merged_df) - mapped_count
        mapping_rate = (mapped_count / len(merged_df)) * 100
        
        print(f"  Mapping results:")
        print(f"    Successfully mapped: {mapped_count:,} ({mapping_rate:.1f}%)")
        print(f"    No mapping found: {unmapped_count:,}")
        
        # Show examples of unmapped HS codes
        if unmapped_count > 0:
            unmapped_hs_codes = merged_df[merged_df['naics_2024'].isna()]['hs_code'].unique()
            print(f"    Example unmapped HS codes: {list(unmapped_hs_codes[:5])}")
        
        # Save merged data
        merged_df.to_csv(output_file_path, index=False)
        print(f"  Saved merged data: {output_file_path}")
        
        return {
            'continent': continent_name,
            'total_records': len(merged_df),
            'mapped_records': mapped_count,
            'unmapped_records': unmapped_count,
            'mapping_rate': mapping_rate,
            'unique_hs_codes': merged_df['hs_code'].nunique(),
            'unique_mapped_hs_codes': merged_df[merged_df['naics_2024'].notna()]['hs_code'].nunique(),
            'unique_naics_codes': merged_df['naics_2024'].nunique()
        }
        
    except Exception as e:
        print(f"  Error processing {continent_name}: {e}")
        return None

def main():
    """Main execution function"""
    print("=" * 60)
    print("MERGING HS-TO-NAICS MAPPING WITH TRADE DATA")
    print("=" * 60)
    
    # Load the mapping data
    mapping_df = load_mapping_data(mapping_file_path)
    if mapping_df is None:
        print("Failed to load mapping data. Exiting.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    
    # Find all combined trade data files
    trade_files = glob.glob(os.path.join(combined_data_path, "*_combined.csv"))
    print(f"Found {len(trade_files)} trade data files:")
    for file in trade_files:
        print(f"  - {os.path.basename(file)}")
    
    if not trade_files:
        print("No trade data files found. Exiting.")
        return
    
    # Process each trade data file
    results = []
    
    for trade_file in trade_files:
        continent_name = os.path.basename(trade_file).replace('_combined.csv', '')
        output_file = os.path.join(output_path, f"{continent_name}_with_naics_mapping.csv")
        
        result = merge_mapping_with_trade_data(trade_file, mapping_df, output_file)
        if result:
            results.append(result)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    if results:
        total_records = sum(r['total_records'] for r in results)
        total_mapped = sum(r['mapped_records'] for r in results)
        overall_mapping_rate = (total_mapped / total_records) * 100 if total_records > 0 else 0
        
        print(f"Overall Statistics:")
        print(f"  Total trade records processed: {total_records:,}")
        print(f"  Successfully mapped records: {total_mapped:,}")
        print(f"  Overall mapping rate: {overall_mapping_rate:.1f}%")
        
        print(f"\nBy Continent:")
        for result in sorted(results, key=lambda x: x['mapping_rate'], reverse=True):
            print(f"  {result['continent']}:")
            print(f"    Records: {result['total_records']:,}")
            print(f"    Mapped: {result['mapped_records']:,} ({result['mapping_rate']:.1f}%)")
            print(f"    Unique HS codes: {result['unique_hs_codes']:,}")
            print(f"    Unique mapped HS codes: {result['unique_mapped_hs_codes']:,}")
            print(f"    Unique NAICS codes: {result['unique_naics_codes']:,}")
        
        # Identify potential issues
        print(f"\nData Quality Observations:")
        low_mapping_continents = [r for r in results if r['mapping_rate'] < 90]
        if low_mapping_continents:
            print(f"  Continents with <90% mapping rate:")
            for result in low_mapping_continents:
                print(f"    - {result['continent']}: {result['mapping_rate']:.1f}%")
        else:
            print(f"  All continents have >90% mapping rate ✓")
        
        # Save summary to CSV
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(output_path, 'mapping_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
    
    print(f"\nAll merged files saved to: {output_path}")

if __name__ == "__main__":
    main()