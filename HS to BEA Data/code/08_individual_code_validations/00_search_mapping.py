# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from pathlib import Path
import country_converter as coco

"""
HS CODE MAPPING AND TRADE DATA EXTRACTION FUNCTIONS

This module provides functions to extract HS-level trade data for specific NAICS codes
by leveraging our HS-to-NAICS-to-BEA mapping system and regional trade data files.

CORE FUNCTIONALITY:
1. Load HS-to-NAICS mapping data to identify relevant HS codes
2. Load regional trade data files (Africa_combined.csv, Asia_combined.csv, etc.)
3. Apply BEA region definitions to aggregate countries into regions
4. Filter HS codes that map to specified NAICS codes
5. Return HS x Region level aggregated trade data

KEY USE CASE:
Extract semiconductor trade data (NAICS 334413, 334418, etc.) at HS code level
to analyze mapping accuracy and identify specific HS codes driving trade patterns.
"""

def get_bea_regions():
    """
    Get BEA region definitions from the actual mapping files used in 05_Trade_weights.py
    Lines 98-100 and 105-106
    Returns mappings for countries to BEA regions: CAN, MEX, CHN, JPN, Europe, RoAsia, RoWorld
    """
    # Get the project root directory structure like in 05_Trade_weights.py
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent  # Go up to "HS to BEA Data" directory
    underlying_data_root = project_root.parent  # Go up to "Underlying_Data_Construction" directory
    
    # Load BEA region mapping files (from 05_Trade_weights.py lines 98-99)
    bea_europe_path = underlying_data_root / 'Map BEA Regions' / 'data' / 'final' / 'BEA_TiVA_Europe.csv'
    bea_asia_path = underlying_data_root / 'Map BEA Regions' / 'data' / 'final' / 'BEA_TiVA_Asia_and_Pacific.csv'
    
    print(f"Loading BEA region mappings from:")
    print(f"  Europe: {bea_europe_path}")
    print(f"  Asia: {bea_asia_path}")
    
    # Load the CSV files
    bea_europe_df = pd.read_csv(bea_europe_path)
    bea_asia_df = pd.read_csv(bea_asia_path)
    
    # Create sets of ISO3 codes for BEA regions (from 05_Trade_weights.py lines 105-106)
    bea_europe_iso3 = set(bea_europe_df['iso3'].dropna().unique())
    bea_asia_iso3 = set(bea_asia_df['iso3'].dropna().unique())
    
    print(f"  Loaded Europe: {len(bea_europe_iso3)} countries")
    print(f"  Loaded Asia and Pacific: {len(bea_asia_iso3)} countries")
    
    return bea_europe_iso3, bea_asia_iso3

def assign_bea_region(country_name, bea_europe_iso3, bea_asia_iso3):
    """
    Assign BEA region based on country name using country_converter
    Exactly like 04_Aggregate_BEA_and_HS.py line 71
    """
    # Use country_converter to get ISO3 code
    iso3 = coco.convert(country_name, to='iso3')
    
    # Handle cases where coco.convert returns lists or None (from 04_Aggregate_BEA_and_HS.py)
    def clean_iso3(iso3_value):
        if isinstance(iso3_value, list):
            return iso3_value[0] if iso3_value else 'UNK'
        elif iso3_value is None or pd.isna(iso3_value):
            return 'UNK'
        else:
            return iso3_value
    
    iso3 = clean_iso3(iso3)
    
    # Apply BEA region mapping logic (from 05_Trade_weights.py line 126)
    if iso3 in ['CAN', 'MEX', 'CHN', 'JPN']:
        return iso3
    elif iso3 in bea_europe_iso3:
        return 'Europe'
    elif iso3 in bea_asia_iso3:
        return 'RoAsia'
    else:
        return 'RoWorld'

def load_hs_naics_mapping(project_root):
    """
    Load the complete HS-to-NAICS-to-BEA mapping file
    Returns DataFrame with HS codes and their corresponding NAICS codes
    """
    mapping_path = project_root / "data" / "working" / "02_HS_to_Naics_to_BEA" / "03_complete_hs_to_bea_mapping.csv"
    
    print(f"Loading HS-to-NAICS mapping from: {mapping_path}")
    mapping_df = pd.read_csv(mapping_path)
    
    print(f"Loaded {len(mapping_df)} HS code mappings")
    return mapping_df

def load_regional_trade_data(project_root):
    """
    Load all regional trade data files and combine them
    Returns DataFrame with columns: Country, impVal, hs_code, hs2_code, region
    """
    combined_data_path = project_root / "data" / "working" / "03_Map_country_trade_data" / "combined_data"
    
    regional_files = [
        'Africa_combined.csv',
        'Asia_combined.csv', 
        'Europe_combined.csv',
        'North_America_combined.csv',
        'Oceana_combined.csv',
        'South_America_combined.csv'
    ]
    
    all_data = []
    bea_europe_iso3, bea_asia_iso3 = get_bea_regions()
    
    # Create a country-to-region mapping once to avoid repeated country_converter calls
    all_countries = set()
    print("Pre-loading to identify unique countries...")
    for file in regional_files:
        file_path = combined_data_path / file
        df = pd.read_csv(file_path)
        all_countries.update(df['Country'].unique())
    
    print(f"Creating country-to-region mapping for {len(all_countries)} unique countries...")
    country_to_region = {}
    for country in all_countries:
        country_to_region[country] = assign_bea_region(country, bea_europe_iso3, bea_asia_iso3)
    
    print("Loading and processing trade data files...")
    for file in regional_files:
        file_path = combined_data_path / file
        print(f"Loading trade data from: {file_path}")
        
        df = pd.read_csv(file_path)
        df['source_file'] = file  # Track which file data came from
        
        # Assign BEA regions using the pre-computed mapping
        df['bea_region'] = df['Country'].map(country_to_region)
        
        all_data.append(df)
        print(f"  Loaded {len(df)} records from {file}")
    
    # Combine all regional data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total trade records loaded: {len(combined_df)}")
    
    return combined_df

def extract_hs_data_for_naics_codes(naics_codes, project_root=None):
    """
    Extract HS-level trade data for specified NAICS codes
    
    Parameters:
    -----------
    naics_codes : set or list
        Set of NAICS codes to extract data for (e.g., {334413, 334418, 334412})
    project_root : Path, optional
        Root path to project. If None, will be inferred from script location
        
    Returns:
    --------
    pandas.DataFrame
        HS x Region level trade data with columns:
        - hs_code: HS commodity code
        - bea_region: BEA region (CAN, MEX, CHN, JPN, Europe, RoAsia, RoWorld)
        - total_impVal: Total import value for that HS code in that region
        - naics_codes: NAICS code(s) this HS code maps to
        - country_count: Number of countries contributing to this HS x Region total
    """
    
    # Set up paths
    if project_root is None:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent  # Go up to "HS to BEA Data" directory
    
    print(f"Extracting HS data for NAICS codes: {sorted(naics_codes)}")
    print("="*60)
    
    # Load HS-to-NAICS mapping
    mapping_df = load_hs_naics_mapping(project_root)
    
    # Filter mapping to only HS codes that map to our target NAICS codes
    # Convert naics_codes to strings for comparison since mapping file may have strings
    naics_codes_str = {str(code) for code in naics_codes}
    
    relevant_hs_codes = mapping_df[
        mapping_df['matched_bea_naics'].astype(str).isin(naics_codes_str) |
        mapping_df['naics'].astype(str).isin(naics_codes_str) |
        mapping_df['naicsMDS'].astype(str).isin(naics_codes_str)
    ]
    
    print(f"Found {len(relevant_hs_codes)} HS codes mapping to target NAICS codes")
    
    if len(relevant_hs_codes) == 0:
        print("WARNING: No HS codes found for the specified NAICS codes!")
        return pd.DataFrame()
    
    # Get the set of HS codes we're interested in
    target_hs_codes = set(relevant_hs_codes['commodity'].astype(str))
    print(f"Target HS codes: {len(target_hs_codes)} unique codes")
    
    # Load regional trade data
    print("\nLoading regional trade data...")
    trade_df = load_regional_trade_data(project_root)
    
    # Filter trade data to only include our target HS codes
    trade_df['hs_code'] = trade_df['hs_code'].astype(str)
    filtered_trade = trade_df[trade_df['hs_code'].isin(target_hs_codes)]
    
    print(f"Filtered to {len(filtered_trade)} trade records matching target HS codes")
    
    if len(filtered_trade) == 0:
        print("WARNING: No trade data found for the target HS codes!")
        return pd.DataFrame()
    
    # Aggregate to HS x BEA Region level
    print("\nAggregating to HS x Region level...")
    aggregated = filtered_trade.groupby(['hs_code', 'bea_region']).agg({
        'impVal': 'sum',
        'Country': 'count'  # Count number of countries
    }).reset_index()
    
    aggregated = aggregated.rename(columns={
        'impVal': 'total_impVal',
        'Country': 'country_count'
    })
    
    # Add NAICS code information
    hs_to_naics = relevant_hs_codes.groupby('commodity')['matched_bea_naics'].apply(
        lambda x: ', '.join(x.dropna().astype(str).unique())
    ).to_dict()
    
    aggregated['naics_codes'] = aggregated['hs_code'].map(hs_to_naics)
    
    # Sort by total import value descending
    aggregated = aggregated.sort_values('total_impVal', ascending=False)
    
    print(f"\nFinal result: {len(aggregated)} HS x Region combinations")
    print(f"Total import value: ${aggregated['total_impVal'].sum():,.0f}")
    
    # Create the detailed CSV data with individual HS codes (not aggregated)
    # This will have naics_code, hs_code, impVal, bea_region for each row
    csv_data = []
    for _, row in filtered_trade.iterrows():
        hs_code = row['hs_code']
        # Find the NAICS codes for this HS code
        hs_naics_info = relevant_hs_codes[relevant_hs_codes['commodity'].astype(str) == hs_code]
        if not hs_naics_info.empty:
            naics_code = hs_naics_info['matched_bea_naics'].iloc[0]
            csv_data.append({
                'naics_code': naics_code,
                'hs_code': hs_code,
                'impVal': row['impVal'],
                'bea_region': row['bea_region']
            })
    
    csv_df = pd.DataFrame(csv_data)
    
    # Save to CSV
    if project_root is None:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent
    
    csv_dir = project_root / "code" / "08_individual_code_validations" / "csvs"
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / "02_semiconductors.csv"
    
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved detailed data to: {csv_path}")
    print(f"CSV contains {len(csv_df)} individual HS x Region x NAICS records")
    
    return aggregated

def extract_semiconductor_data():
    """
    Convenience function to extract semiconductor data using the NAICS codes
    from line 99 of 02_semiconductors.py
    """
    # NAICS codes from 02_semiconductors.py line 99
    bea_3344 = {334413, 334418, 334412, 334416, 334417, 334419}  # Semiconductors
    bea_334X = {3343, 33461}  # Other Computer Manufacturing  
    bea_3341 = {334111, 334112, 334118}  # Computer Equipment
    
    print("="*60)
    print("SEMICONDUCTOR DATA EXTRACTION BY BEA CATEGORY")
    print("="*60)
    
    # Extract data for each BEA category separately to get totals
    print("\n1. Extracting BEA 3344 (Semiconductors)...")
    data_3344 = extract_hs_data_for_naics_codes(bea_3344)
    total_3344 = data_3344['total_impVal'].sum() if not data_3344.empty else 0
    
    print("\n2. Extracting BEA 334X (Other Computer Manufacturing)...")
    data_334X = extract_hs_data_for_naics_codes(bea_334X)
    total_334X = data_334X['total_impVal'].sum() if not data_334X.empty else 0
    
    print("\n3. Extracting BEA 3341 (Computer Equipment)...")
    data_3341 = extract_hs_data_for_naics_codes(bea_3341)
    total_3341 = data_3341['total_impVal'].sum() if not data_3341.empty else 0
    
    print("\n" + "="*60)
    print("BEA CATEGORY TOTALS")
    print("="*60)
    print(f"BEA 3344 (Semiconductors):           ${total_3344:>15,.0f}")
    print(f"BEA 334X (Other Computer Mfg):       ${total_334X:>15,.0f}")
    print(f"BEA 3341 (Computer Equipment):       ${total_3341:>15,.0f}")
    print("-" * 60)
    print(f"TOTAL ALL CATEGORIES:                ${total_3344 + total_334X + total_3341:>15,.0f}")
    print("="*60)
    
    # Now extract all combined for the CSV
    print("\n4. Extracting combined data for CSV...")
    all_semiconductor_naics = bea_3344 | bea_334X | bea_3341
    combined_data = extract_hs_data_for_naics_codes(all_semiconductor_naics)
    
    return combined_data

# Example usage and testing
if __name__ == "__main__":
    print("SEMICONDUCTOR DATA EXTRACTION AND CSV GENERATION")
    print("="*60)
    
    # Extract all semiconductor data and save to CSV
    semiconductor_data = extract_semiconductor_data()
    
    print(f"\nFinal aggregated results: {len(semiconductor_data)} HS x Region combinations")
    print(f"Process completed successfully!")