import os
import pandas as pd
import json
import numpy as np
from main_pipeline_run import get_data_path

"""
DESCRIPTION: This script helps investigate potential mapping issues by analyzing how 
HS codes are currently mapped to BEA codes and exploring alternative mappings.

The script addresses a key question: Are weak hierarchical mappings contributing to the 
large discrepancies identified in 07_TiVA_Import_Values_Comparison.py?

KEY INSIGHT: We perform hierarchical mapping in TWO stages:
1. 01_Schott_Data_Compiler.py: HS codes -> NAICS codes (hierarchical_hs method)
2. 03_Map_country_trade_data.py: HS codes -> BEA detail codes (hierarchical matching with strength scores)

This creates potential inconsistencies where:
- An HS code gets mapped to NAICS X in stage 1
- The same HS code gets mapped to BEA detail Y in stage 2 (with weak strength)
- BEA detail Y might correspond to a different NAICS code than X

FUNCTIONALITY:
- Given a BEA U.Summary code, find all HS codes currently mapped to it
- Show the trade values for these HS codes across regions
- Identify which mappings are "weak" (low mapping_strength from stage 2)
- Explore alternative BEA mappings for weak codes
- Calculate how changing mappings would affect regional totals

This helps answer: "If I remap weak HS codes to their alternative BEA codes, 
does it reduce the discrepancies with TiVA data?"
"""

class BEAMappingExplorer:
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        """Load all necessary data files"""
        print("Loading data files...")
        
        # Load BEA hierarchy for code relationships
        bea_hierarchy_path = os.path.join(get_data_path('working', '02_HS_to_Naics_to_BEA'), '02_BEA_hierarchy.csv')
        self.bea_hierarchy = pd.read_csv(bea_hierarchy_path)
        
        # Load complete HS to BEA mapping (from stage 1: Schott)
        hs_bea_path = os.path.join(get_data_path('working', '02_HS_to_Naics_to_BEA'), '03_complete_hs_to_bea_mapping.csv')
        self.hs_bea_mapping = pd.read_csv(hs_bea_path)
        
        # Load hierarchical matches from stage 2 (trade data mapping)
        hierarchical_path = os.path.join(get_data_path('validation', '03_Map_country_trade_data'), '3_Hierarchical_Matches.csv')
        self.hierarchical_matches = pd.read_csv(hierarchical_path)
        
        # Load trade weights (final aggregated values by region) - optional
        try:
            trade_weights_path = os.path.join(get_data_path('working', '05_Trade_weights'), 'usummary_trade_weights.csv')
            self.trade_weights = pd.read_csv(trade_weights_path)
            print(f"Loaded {len(self.trade_weights)} trade weight records")
        except FileNotFoundError:
            print("Warning: Trade weights file not found - some functions may be limited")
            self.trade_weights = pd.DataFrame()
        
        # Load HS descriptions
        try:
            hs_desc_path = os.path.join(get_data_path('working', '03_Map_country_trade_data'), '01_hs10_descriptions_full.csv')
            self.hs_descriptions = pd.read_csv(hs_desc_path)
            self.hs_descriptions = self.hs_descriptions.rename(columns={'hs10': 'commodity'})
        except:
            print("Warning: Could not load HS descriptions")
            self.hs_descriptions = pd.DataFrame(columns=['commodity', 'description'])
        
        # Load trade data from combined files
        self.trade_data = self.load_trade_data()
        
        # Create lookup dictionaries
        self.detail_to_usummary = dict(zip(self.bea_hierarchy['Detail'], self.bea_hierarchy['U.Summary']))
        
        print(f"Loaded {len(self.hs_bea_mapping)} HS-to-BEA mappings")
        print(f"Loaded {len(self.hierarchical_matches)} hierarchical matches")
        if hasattr(self, 'trade_data') and self.trade_data is not None:
            print(f"Loaded {len(self.trade_data)} HS-level trade records")
    
    def load_trade_data(self):
        """Load and combine all regional trade data files"""
        print("Loading trade data from combined files...")
        
        combined_data_dir = os.path.join(get_data_path('working', '03_Map_country_trade_data'), 'combined_data')
        
        # Define region mapping based on the files and expected regions
        region_files = {
            'CAN': 'North_America_combined.csv',  # Will filter for Canada
            'MEX': 'North_America_combined.csv',  # Will filter for Mexico  
            'CHN': 'Asia_combined.csv',           # Will filter for China
            'Europe': 'Europe_combined.csv',      # All European countries
            'JPN': 'Asia_combined.csv',           # Will filter for Japan
            'RoAsia': 'Asia_combined.csv',        # Rest of Asia (excluding China, Japan)
            'RoWorld': ['Africa_combined.csv', 'Oceana_combined.csv', 'South_America_combined.csv']  # Rest of world
        }
        
        all_trade_data = []
        
        try:
            # Process each region
            for region, file_or_files in region_files.items():
                if isinstance(file_or_files, str):
                    file_or_files = [file_or_files]
                
                region_data = []
                for filename in file_or_files:
                    filepath = os.path.join(combined_data_dir, filename)
                    if os.path.exists(filepath):
                        df = pd.read_csv(filepath)
                        
                        # Apply country filtering based on region
                        if region == 'CAN':
                            df = df[df['Country'] == 'Canada']
                        elif region == 'MEX':
                            df = df[df['Country'] == 'Mexico'] 
                        elif region == 'CHN':
                            df = df[df['Country'] == 'China']
                        elif region == 'JPN':
                            df = df[df['Country'] == 'Japan']
                        elif region == 'RoAsia':
                            # Rest of Asia excluding China and Japan
                            df = df[~df['Country'].isin(['China', 'Japan'])]
                        # For Europe and RoWorld, take all countries in the file
                        
                        if len(df) > 0:
                            df['region'] = region
                            region_data.append(df)
                
                if region_data:
                    combined_region = pd.concat(region_data, ignore_index=True)
                    all_trade_data.append(combined_region)
            
            if all_trade_data:
                combined_trade = pd.concat(all_trade_data, ignore_index=True)
                
                # Aggregate by HS code and region
                trade_summary = combined_trade.groupby(['hs_code', 'region'])['impVal'].sum().reset_index()
                
                print(f"Successfully loaded trade data for {len(trade_summary['hs_code'].unique())} unique HS codes")
                return trade_summary
            else:
                print("Warning: No trade data files found")
                return pd.DataFrame(columns=['hs_code', 'region', 'impVal'])
                
        except Exception as e:
            print(f"Error loading trade data: {e}")
            return pd.DataFrame(columns=['hs_code', 'region', 'impVal'])
    
    def explore_bea_code(self, usummary_code, show_weak_only=False, strength_threshold=0.8):
        """
        Explore all HS codes mapped to a given BEA U.Summary code
        
        Args:
            usummary_code: BEA U.Summary code to investigate
            show_weak_only: If True, only show HS codes with weak mappings
            strength_threshold: Mapping strength below this is considered "weak"
        """
        print(f"\n{'='*60}")
        print(f"EXPLORING BEA U.SUMMARY CODE: {usummary_code}")
        print(f"{'='*60}")
        
        # Get BEA hierarchy info
        bea_info = self.bea_hierarchy[self.bea_hierarchy['U.Summary'] == usummary_code]
        if len(bea_info) > 0:
            print(f"BEA Category: {bea_info['undersum title'].iloc[0]}")
            print(f"Detail codes: {sorted(bea_info['Detail'].unique())}")
        
        # Find all HS codes currently mapped to this U.Summary code
        # Step 1: Get detail codes for this U.Summary
        detail_codes = self.bea_hierarchy[self.bea_hierarchy['U.Summary'] == usummary_code]['Detail'].unique()
        
        # Step 2: Find HS codes mapped to these detail codes
        hs_codes_stage1 = self.hs_bea_mapping[self.hs_bea_mapping['matched_bea_detail'].isin(detail_codes)]['commodity'].unique()
        
        print(f"\nHS codes mapped via Stage 1 (Schott): {len(hs_codes_stage1)}")
        
        # Step 3: Find HS codes mapped to these detail codes via Stage 2 (hierarchical)
        hs_codes_stage2 = self.hierarchical_matches[self.hierarchical_matches['matched_bea_detail'].isin(detail_codes)]['hs_code'].unique()
        
        print(f"HS codes mapped via Stage 2 (hierarchical): {len(hs_codes_stage2)}")
        
        # Combine and analyze
        all_hs_codes = set(list(hs_codes_stage1) + list(hs_codes_stage2))
        print(f"Total unique HS codes: {len(all_hs_codes)}")
        
        # Create detailed analysis
        analysis_data = []
        
        for hs_code in all_hs_codes:
            # Get Stage 1 mapping
            stage1_info = self.hs_bea_mapping[self.hs_bea_mapping['commodity'] == hs_code]
            stage1_detail = stage1_info['matched_bea_detail'].iloc[0] if len(stage1_info) > 0 else None
            
            # Get Stage 2 mapping (hierarchical)
            stage2_info = self.hierarchical_matches[self.hierarchical_matches['hs_code'] == hs_code]
            
            if len(stage2_info) > 0:
                # Get primary mapping
                primary_mapping = stage2_info[stage2_info['match_type'] == 'primary']
                if len(primary_mapping) > 0:
                    stage2_detail = primary_mapping['matched_bea_detail'].iloc[0]
                    mapping_strength = primary_mapping['mapping_strength'].iloc[0]
                    match_level = primary_mapping['match_level'].iloc[0]
                else:
                    stage2_detail = None
                    mapping_strength = None
                    match_level = None
                
                # Get alternative mappings
                alternatives = stage2_info[stage2_info['match_type'] == 'alternative']
                alt_details = alternatives['matched_bea_detail'].tolist() if len(alternatives) > 0 else []
            else:
                stage2_detail = None
                mapping_strength = None
                match_level = None
                alt_details = []
            
            # Get trade values for this HS code
            trade_values = self.get_hs_trade_values(hs_code)
            
            # Get description
            desc_info = self.hs_descriptions[self.hs_descriptions['commodity'] == hs_code]
            description = desc_info['description'].iloc[0] if len(desc_info) > 0 else 'No description'
            
            # Determine mapping consistency
            mapping_consistent = (stage1_detail == stage2_detail) if (stage1_detail and stage2_detail) else None
            
            # Check if mapping is weak
            is_weak = mapping_strength < strength_threshold if mapping_strength else False
            
            if show_weak_only and not is_weak:
                continue
            
            analysis_data.append({
                'hs_code': hs_code,
                'description': description,
                'stage1_detail': stage1_detail,
                'stage2_detail': stage2_detail,
                'mapping_strength': mapping_strength,
                'match_level': match_level,
                'mapping_consistent': mapping_consistent,
                'is_weak': is_weak,
                'alternatives': alt_details,
                'total_trade_value': sum(trade_values.values()) if trade_values else 0,
                'trade_values': trade_values
            })
        
        # Convert to DataFrame and sort by trade value
        analysis_df = pd.DataFrame(analysis_data)
        if len(analysis_df) > 0:
            analysis_df = analysis_df.sort_values('total_trade_value', ascending=False)
        
        # Display results
        if len(analysis_df) == 0:
            print("No matching HS codes found.")
            return analysis_df
        
        # Summary statistics
        total_codes = len(analysis_df)
        weak_codes = len(analysis_df[analysis_df['is_weak'] == True])
        inconsistent_codes = len(analysis_df[analysis_df['mapping_consistent'] == False])
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"Total HS codes: {total_codes}")
        print(f"Weak mappings (strength < {strength_threshold}): {weak_codes}")
        print(f"Inconsistent mappings (Stage1 ≠ Stage2): {inconsistent_codes}")
        
        # Show top trade value contributors
        print(f"\nTOP 10 TRADE VALUE CONTRIBUTORS:")
        top_10 = analysis_df.head(10)
        for _, row in top_10.iterrows():
            status = []
            if row['is_weak']:
                status.append('WEAK')
            if row['mapping_consistent'] == False:
                status.append('INCONSISTENT')
            status_str = f" [{'/'.join(status)}]" if status else ""
            
            print(f"  {row['hs_code']}: ${row['total_trade_value']:,.0f}{status_str}")
            print(f"    {row['description'][:80]}...")
            
            if len(row['alternatives']) > 0:
                alt_usummary = [self.detail_to_usummary.get(alt, 'Unknown') for alt in row['alternatives']]
                print(f"    Alternatives: {row['alternatives']} -> {alt_usummary}")
        
        return analysis_df
    
    def get_hs_trade_values(self, hs_code):
        """Get trade values for a specific HS code across all regions"""
        if not hasattr(self, 'trade_data') or self.trade_data is None:
            return {'CAN': 0, 'CHN': 0, 'Europe': 0, 'JPN': 0, 'MEX': 0, 'RoAsia': 0, 'RoWorld': 0}
        
        # Convert HS code to integer for matching (trade data has integer HS codes)
        try:
            hs_code_int = int(str(hs_code).zfill(10))
        except ValueError:
            return {'CAN': 0, 'CHN': 0, 'Europe': 0, 'JPN': 0, 'MEX': 0, 'RoAsia': 0, 'RoWorld': 0}
        
        # Filter trade data for this HS code
        hs_trade = self.trade_data[self.trade_data['hs_code'] == hs_code_int]
        
        # Create dictionary with values for each region
        trade_values = {'CAN': 0, 'CHN': 0, 'Europe': 0, 'JPN': 0, 'MEX': 0, 'RoAsia': 0, 'RoWorld': 0}
        
        for _, row in hs_trade.iterrows():
            if row['region'] in trade_values:
                trade_values[row['region']] = row['impVal']
        
        return trade_values
    
    def simulate_mapping_change(self, usummary_code, hs_code, new_detail_code):
        """
        Simulate how changing one HS code's mapping would affect regional totals
        
        Args:
            usummary_code: Current BEA U.Summary code
            hs_code: HS code to remap
            new_detail_code: New BEA detail code to map to
        """
        print(f"\nSIMULATING MAPPING CHANGE:")
        print(f"HS Code: {hs_code}")
        print(f"From U.Summary: {usummary_code}")
        
        new_usummary = self.detail_to_usummary.get(new_detail_code, 'Unknown')
        print(f"To Detail: {new_detail_code} (U.Summary: {new_usummary})")
        
        # Get current trade values for this HS code
        trade_values = self.get_hs_trade_values(hs_code)
        
        if sum(trade_values.values()) == 0:
            print("No trade data found for this HS code")
            return
        
        # Show impact on regional totals
        print(f"\nTRADE VALUES TO BE MOVED:")
        for region, value in trade_values.items():
            if value > 0:
                print(f"  {region}: ${value:,.0f}")
        
        print(f"\nThis would:")
        print(f"  - Reduce {usummary_code} totals by ${sum(trade_values.values()):,.0f}")
        print(f"  - Increase {new_usummary} totals by ${sum(trade_values.values()):,.0f}")
    
    def find_problematic_bea_codes(self, min_discrepancy_pct=30, top_n=10):
        """
        Find BEA codes with large discrepancies from TiVA data
        Uses the discrepancies table from 07_TiVA_Import_Values_Comparison.py
        """
        print(f"\nFINDING BEA CODES WITH >{min_discrepancy_pct}% DISCREPANCY FROM TiVA:")
        print("="*60)
        
        try:
            # Load the discrepancies table from 07_TiVA analysis
            discrepancies_path = os.path.join(get_data_path('validation', '07_TiVA_Import_Values_Comparison'), '03_large_discrepancies_table.csv')
            discrepancies = pd.read_csv(discrepancies_path)
            
            # Filter for large discrepancies
            large_disc = discrepancies[discrepancies['pct_difference'] >= min_discrepancy_pct]
            
            # Group by BEA code and calculate average discrepancy across regions
            bea_discrepancies = large_disc.groupby('usummary_code').agg({
                'pct_difference': ['mean', 'max', 'count'],
                'HS_total_imports': 'sum',
                'TiVA_total_imports': 'sum'
            }).round(2)
            
            bea_discrepancies.columns = ['avg_pct_diff', 'max_pct_diff', 'region_count', 'total_hs_imports', 'total_tiva_imports']
            bea_discrepancies = bea_discrepancies.sort_values('avg_pct_diff', ascending=False)
            
            print(f"Found {len(bea_discrepancies)} BEA codes with >{min_discrepancy_pct}% discrepancies")
            print(f"\nTop {top_n} problematic BEA codes:")
            print("-" * 80)
            
            for i, (bea_code, row) in enumerate(bea_discrepancies.head(top_n).iterrows()):
                # Get BEA category name
                bea_info = self.bea_hierarchy[self.bea_hierarchy['U.Summary'] == bea_code]
                bea_name = bea_info['undersum title'].iloc[0] if len(bea_info) > 0 else 'Unknown'
                
                print(f"{i+1:2d}. {bea_code} - {bea_name}")
                print(f"    Avg discrepancy: {row['avg_pct_diff']:.1f}% (max: {row['max_pct_diff']:.1f}%)")
                print(f"    Regions affected: {row['region_count']}")
                print(f"    HS total: ${row['total_hs_imports']:,.0f}, TiVA total: ${row['total_tiva_imports']:,.0f}")
                print()
            
            return bea_discrepancies.head(top_n)
            
        except Exception as e:
            print(f"Could not load discrepancies data: {e}")
            print("Make sure 07_TiVA_Import_Values_Comparison.py has been run")
            return pd.DataFrame()
    
    def comprehensive_analysis(self, bea_code):
        """
        Run a comprehensive analysis of a BEA code including:
        1. Current HS mappings
        2. Weak mappings identification  
        3. Trade value breakdown
        4. TiVA discrepancy context
        """
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE ANALYSIS FOR BEA CODE: {bea_code}")
        print(f"{'='*80}")
        
        # Step 1: Basic exploration
        result = self.explore_bea_code(bea_code, show_weak_only=False, strength_threshold=0.8)
        
        # Step 2: Show weak mappings specifically
        if len(result) > 0:
            weak_mappings = result[result['is_weak'] == True]
            if len(weak_mappings) > 0:
                print(f"\n{len(weak_mappings)} WEAK MAPPINGS FOUND:")
                print("-" * 50)
                for _, row in weak_mappings.head(5).iterrows():
                    print(f"HS {row['hs_code']}: {row['description'][:60]}...")
                    print(f"  Strength: {row['mapping_strength']:.3f}, Trade: ${row['total_trade_value']:,.0f}")
                    if len(row['alternatives']) > 0:
                        alt_usummary = [self.detail_to_usummary.get(alt, 'Unknown') for alt in row['alternatives']]
                        print(f"  Alternatives: {alt_usummary}")
                    print()
                
                # Automatically test alternative mappings
                print(f"\n{'='*50}")
                print("TESTING ALTERNATIVE MAPPINGS FOR WEAK CODES")
                print(f"{'='*50}")
                alt_results = self.test_alternative_mappings(bea_code, strength_threshold=0.8)
        
        # Step 3: Show TiVA discrepancy context if available
        try:
            discrepancies_path = os.path.join(get_data_path('validation', '07_TiVA_Import_Values_Comparison'), '03_large_discrepancies_table.csv')
            discrepancies = pd.read_csv(discrepancies_path)
            bea_disc = discrepancies[discrepancies['usummary_code'] == bea_code]
            
            if len(bea_disc) > 0:
                print(f"\nTiVA DISCREPANCY CONTEXT:")
                print("-" * 30)
                for _, row in bea_disc.iterrows():
                    print(f"{row['region']}: {row['pct_difference']:.1f}% discrepancy")
                    print(f"  HS: ${row['HS_total_imports']:,.0f}, TiVA: ${row['TiVA_total_imports']:,.0f}")
        except:
            print("\nTiVA discrepancy data not available")
        
        return result
    
    def test_alternative_mappings(self, usummary_code, strength_threshold=0.8):
        """
        For weak mappings in a BEA U.Summary code, test using alternative mappings
        and show the regional trade value differences
        """
        print(f"\n{'='*80}")
        print(f"TESTING ALTERNATIVE MAPPINGS FOR BEA CODE: {usummary_code}")
        print(f"{'='*80}")
        
        # First get the basic exploration
        result = self.explore_bea_code(usummary_code, show_weak_only=False, strength_threshold=strength_threshold)
        
        if len(result) == 0:
            print("No HS codes found for this BEA code.")
            return
        
        # Find weak mappings
        weak_mappings = result[result['is_weak'] == True]
        
        if len(weak_mappings) == 0:
            print("No weak mappings found - all mappings are strong.")
            return
        
        print(f"Found {len(weak_mappings)} weak mappings to test alternatives for:")
        print("-" * 60)
        
        # Track regional impacts
        regional_changes = {'CAN': 0, 'CHN': 0, 'Europe': 0, 'JPN': 0, 'MEX': 0, 'RoAsia': 0, 'RoWorld': 0}
        alternative_bea_impacts = {}
        
        for i, (_, weak_row) in enumerate(weak_mappings.iterrows()):
            hs_code = weak_row['hs_code']
            current_strength = weak_row['mapping_strength']
            alternatives = weak_row['alternatives']
            trade_values = weak_row['trade_values']
            total_trade = weak_row['total_trade_value']
            
            print(f"\n{i+1}. HS Code: {hs_code}")
            print(f"   Description: {weak_row['description'][:70]}...")
            print(f"   Current mapping strength: {current_strength:.3f}")
            print(f"   Total trade value: ${total_trade:,.0f}")
            
            if len(alternatives) == 0:
                print("   No alternatives available")
                continue
            
            # Show trade values by region for this HS code
            print(f"   Regional trade breakdown:")
            for region, value in trade_values.items():
                if value > 0:
                    print(f"     {region}: ${value:,.0f}")
                    regional_changes[region] += value
            
            print(f"   Alternative BEA detail codes: {alternatives}")
            
            # Convert alternative detail codes to U.Summary codes
            alt_usummary_codes = []
            for alt_detail in alternatives:
                alt_usummary = self.detail_to_usummary.get(alt_detail, 'Unknown')
                alt_usummary_codes.append(alt_usummary)
                
                # Track impact on alternative BEA codes
                if alt_usummary not in alternative_bea_impacts:
                    alternative_bea_impacts[alt_usummary] = {
                        'total_value': 0,
                        'hs_codes': [],
                        'regional_values': {'CAN': 0, 'CHN': 0, 'Europe': 0, 'JPN': 0, 'MEX': 0, 'RoAsia': 0, 'RoWorld': 0}
                    }
                
                alternative_bea_impacts[alt_usummary]['total_value'] += total_trade
                alternative_bea_impacts[alt_usummary]['hs_codes'].append(hs_code)
                
                for region, value in trade_values.items():
                    alternative_bea_impacts[alt_usummary]['regional_values'][region] += value
            
            print(f"   Would move to U.Summary: {alt_usummary_codes}")
        
        # Summary of regional impact
        total_moved = sum(regional_changes.values())
        print(f"\n{'='*60}")
        print(f"SUMMARY: ALTERNATIVE MAPPING IMPACT")
        print(f"{'='*60}")
        
        print(f"\nTrade values that would move FROM {usummary_code}:")
        for region, value in regional_changes.items():
            if value > 0:
                print(f"  {region}: ${value:,.0f}")
        print(f"  TOTAL: ${total_moved:,.0f}")
        
        print(f"\nTrade values that would move TO alternative BEA codes:")
        for alt_bea, impact_data in alternative_bea_impacts.items():
            if impact_data['total_value'] > 0:
                print(f"\n  {alt_bea}: ${impact_data['total_value']:,.0f} ({len(impact_data['hs_codes'])} HS codes)")
                
                # Get BEA name for the alternative
                alt_bea_info = self.bea_hierarchy[self.bea_hierarchy['U.Summary'] == alt_bea]
                alt_name = alt_bea_info['undersum title'].iloc[0] if len(alt_bea_info) > 0 else 'Unknown'
                print(f"    Category: {alt_name}")
                
                print(f"    Regional breakdown:")
                for region, value in impact_data['regional_values'].items():
                    if value > 0:
                        print(f"      {region}: ${value:,.0f}")
        
        # Calculate potential TiVA impact if TiVA data is available
        try:
            discrepancies_path = os.path.join(get_data_path('validation', '07_TiVA_Import_Values_Comparison'), '03_large_discrepancies_table.csv')
            discrepancies = pd.read_csv(discrepancies_path)
            
            print(f"\n{'='*60}")
            print(f"POTENTIAL TiVA DISCREPANCY IMPACT")
            print(f"{'='*60}")
            
            # Check current discrepancies for this BEA code
            current_disc = discrepancies[discrepancies['usummary_code'] == usummary_code]
            if len(current_disc) > 0:
                print(f"\nCurrent {usummary_code} TiVA discrepancies:")
                for _, row in current_disc.iterrows():
                    region_change = regional_changes.get(row['region'], 0)
                    old_hs_total = row['HS_total_imports']
                    new_hs_total = old_hs_total - region_change
                    old_pct = row['pct_difference']
                    
                    if old_hs_total > 0:
                        new_pct = abs(new_hs_total - row['TiVA_total_imports']) / max(new_hs_total, row['TiVA_total_imports']) * 100
                        impact = new_pct - old_pct
                        direction = "↓" if impact < 0 else "↑"
                        print(f"  {row['region']}: {old_pct:.1f}% → {new_pct:.1f}% ({direction}{abs(impact):.1f}%)")
                    else:
                        print(f"  {row['region']}: {old_pct:.1f}% → unchanged (no trade data)")
            
            # Check discrepancies for alternative BEA codes
            for alt_bea, impact_data in alternative_bea_impacts.items():
                alt_disc = discrepancies[discrepancies['usummary_code'] == alt_bea]
                if len(alt_disc) > 0:
                    print(f"\nAlternative {alt_bea} would get additional trade:")
                    for _, row in alt_disc.iterrows():
                        if row['region'] in impact_data['regional_values']:
                            region_addition = impact_data['regional_values'][row['region']]
                            if region_addition > 0:
                                old_hs_total = row['HS_total_imports']
                                new_hs_total = old_hs_total + region_addition
                                old_pct = row['pct_difference']
                                new_pct = abs(new_hs_total - row['TiVA_total_imports']) / max(new_hs_total, row['TiVA_total_imports']) * 100
                                impact = new_pct - old_pct
                                direction = "↓" if impact < 0 else "↑" 
                                print(f"  {row['region']}: {old_pct:.1f}% → {new_pct:.1f}% ({direction}{abs(impact):.1f}%)")
                
        except Exception as e:
            print(f"\nTiVA discrepancy analysis not available: {e}")
        
        return {
            'weak_mappings_count': len(weak_mappings),
            'total_trade_moved': total_moved,
            'regional_changes': regional_changes,
            'alternative_impacts': alternative_bea_impacts
        }

def main():
    """Main function demonstrating the mapping explorer"""
    explorer = BEAMappingExplorer()
    
    print("BEA Mapping Explorer loaded successfully!")
    print("\nExample usage:")
    print("1. explorer.find_problematic_bea_codes() - Find BEA codes with large TiVA discrepancies")
    print("2. explorer.explore_bea_code('334X') - Explore all HS codes mapped to BEA code 334X")
    print("3. explorer.test_alternative_mappings('336111') - Test alternative mappings for weak codes")
    print("4. explorer.comprehensive_analysis('334X') - Full analysis with weak mappings and alternatives")
    print("5. explorer.simulate_mapping_change('334X', '1234567890', '335') - Simulate remapping")
    
    # Find problematic codes first
    print("\nFinding BEA codes with largest TiVA discrepancies...")
    try:
        problematic_codes = explorer.find_problematic_bea_codes(min_discrepancy_pct=50, top_n=5)
        
        if len(problematic_codes) > 0:
            example_code = input(f"\nEnter a BEA code to analyze (or press Enter for {problematic_codes.index[0]}): ").strip()
            if not example_code:
                example_code = problematic_codes.index[0]
            
            print(f"\nRunning comprehensive analysis for {example_code}...")
            result = explorer.comprehensive_analysis(example_code)
            
        else:
            print("No problematic codes found or TiVA data not available")
            
    except Exception as e:
        print(f"Could not run automatic analysis: {e}")
        print("You can still use the explorer manually")
    
    return explorer

if __name__ == "__main__":
    explorer = main()