# Compare Schott vs Census mappings at regional level using BEA region definitions
# Creates regional aggregations and validates totals match between mapping approaches
# Generates HTML visualization with regional comparison and country-level difference tabs

import os
import sys
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import country_converter as coco

# Add path to shared validation styles
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'shared_utilities'))
from shared_validation_styles import get_shared_css, get_shared_javascript

# Load data paths configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
data_paths_file = os.path.join(script_dir, '..', '..', 'data_paths.json')

with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Load BEA region mapping files (from 05_Trade_Weights.py approach)
bea_europe_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'Map BEA Regions', 'data', 'final', 'BEA_TiVA_Europe.csv')
bea_asia_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'Map BEA Regions', 'data', 'final', 'BEA_TiVA_Asia_and_Pacific.csv')

bea_europe_df = pd.read_csv(bea_europe_path)
bea_asia_df = pd.read_csv(bea_asia_path)

# Extract ISO3 codes from the loaded data
bea_europe_iso3 = set(bea_europe_df['iso3'].dropna().unique())
bea_asia_iso3 = set(bea_asia_df['iso3'].dropna().unique())

print(f"Loaded BEA region mappings:")
print(f"  Europe: {len(bea_europe_iso3)} countries")
print(f"  Asia and Pacific: {len(bea_asia_iso3)} countries")

def assign_region_bea(row):
    """BEA-specific region mapping (from 05_Trade_Weights.py)"""
    iso3 = row['iso3']
    
    if iso3 in ['CAN', 'MEX', 'CHN', 'JPN']:
        return iso3
    elif iso3 in bea_europe_iso3:
        return 'Europe'
    elif iso3 in bea_asia_iso3:
        return 'RoAsia'
    else:
        return 'RoWorld'

def load_original_bea_data():
    """Load original BEA aggregated data"""
    print("Loading original BEA data...")
    
    base_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                            'data', 'working', '04_Aggregate_BEA_and_HS', 'aggregated_data', 'country_usummary')
    bea_file = os.path.join(base_path, 'all_continents_usummary.csv')
    
    df = pd.read_csv(bea_file, dtype=str)
    df['impVal'] = pd.to_numeric(df['impVal'])
    df['Country'] = df['Country'].str.strip()
    
    # Add ISO3 codes for region assignment
    df['iso3'] = coco.convert(df['Country'], to='iso3')
    
    # Handle cases where coco.convert returns lists or None
    def clean_iso3(iso3_value):
        if isinstance(iso3_value, list):
            return iso3_value[0] if iso3_value else 'UNK'
        elif iso3_value is None or pd.isna(iso3_value):
            return 'UNK'
        else:
            return iso3_value
    
    df['iso3'] = df['iso3'].apply(clean_iso3)
    
    return df

def load_alternative_census_data():
    """Load Alternative Census aggregated data"""
    print("Loading Alternative Census data...")
    
    base_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                            'data', 'working', 'Alternative_Census_Mapping', 'bea_aggregations', 'country_usummary')
    census_file = os.path.join(base_path, 'all_continents_usummary.csv')
    
    df = pd.read_csv(census_file, dtype=str)
    df['impVal'] = pd.to_numeric(df['impVal'])
    df['Country'] = df['Country'].str.strip()
    
    # Add ISO3 codes for region assignment (should already have them but ensure consistency)
    if 'iso3' not in df.columns:
        df['iso3'] = coco.convert(df['Country'], to='iso3')
        df['iso3'] = df['iso3'].apply(lambda x: x[0] if isinstance(x, list) and x else ('UNK' if pd.isna(x) else x))
    
    return df

def create_regional_aggregations():
    """Create regional aggregations for both datasets"""
    
    # Load both datasets
    original_df = load_original_bea_data()
    census_df = load_alternative_census_data()
    
    # Add region assignments
    original_df['region'] = original_df.apply(assign_region_bea, axis=1)
    census_df['region'] = census_df.apply(assign_region_bea, axis=1)
    
    # Create regional aggregations
    original_regional = original_df.groupby(['region', 'usummary_code'])['impVal'].sum().reset_index()
    census_regional = census_df.groupby(['region', 'usummary_code'])['impVal'].sum().reset_index()
    
    # Validate totals match
    print("\nValidating regional totals match:")
    original_totals = original_df.groupby('region')['impVal'].sum()
    census_totals = census_df.groupby('region')['impVal'].sum()
    
    for region in original_totals.index:
        original_total = original_totals[region]
        census_total = census_totals.get(region, 0)
        difference = abs(original_total - census_total)
        print(f"{region}: Original ${original_total:,.0f}, Census ${census_total:,.0f}, Diff ${difference:,.0f}")
    
    return original_regional, census_regional

def load_usummary_comparison_data():
    """Load the usummary level comparison CSV"""
    comparison_file = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                  'validations', 'Alternative_Census_Mappings', 
                                  'comparisons with original mapping', '08_usummary_level_comparison.csv')
    
    df = pd.read_csv(comparison_file)
    return df

def create_tiva_census_comparison():
    """Create TiVA vs Census mapping comparison (like 07_TiVA_Import_Values_Comparison.py)"""
    print("Creating TiVA vs Census mapping comparison...")
    
    # Load Census regional data
    census_df = load_alternative_census_data()
    census_df['region'] = census_df.apply(assign_region_bea, axis=1)
    
    # Load TiVA data
    tiva_dir = os.path.join(data_paths['base_paths']['underlying_data_root'], 'data', 'raw', 'TiVA Tables')
    region_mapping = {
        'CAN.csv': 'CAN',
        'CHN.csv': 'CHN', 
        'EUR.csv': 'Europe',
        'JPN.csv': 'JPN',
        'MEX.csv': 'MEX',
        'RoAsia.csv': 'RoAsia',
        'RoW.csv': 'RoWorld',
        'WholeWorld.csv': 'world'
    }
    
    all_comparisons = {}
    
    # Create comparisons for each region
    for tiva_file, region_key in region_mapping.items():
        tiva_path = os.path.join(tiva_dir, tiva_file)
        if not os.path.exists(tiva_path):
            print(f"Warning: TiVA file not found: {tiva_path}")
            continue
            
        tiva_df = pd.read_csv(tiva_path)
        
        # Keep only first and last columns
        tiva_comparison = tiva_df.iloc[:, [0, -1]].copy()
        tiva_comparison.columns = ['usummary_code', 'TiVA_total_imports']
        
        # Convert usummary_code to string and multiply TiVA values by 1,000,000
        tiva_comparison['usummary_code'] = tiva_comparison['usummary_code'].astype(str)
        tiva_comparison['TiVA_total_imports'] = tiva_comparison['TiVA_total_imports'] * 1000000
        
        # Get corresponding Census data
        if region_key == 'world':
            census_region_data = census_df.groupby('usummary_code')['impVal'].sum().reset_index()
            census_region_data.columns = ['usummary_code', 'Census_total_imports']
        else:
            region_data = census_df[census_df['region'] == region_key]
            census_region_data = region_data.groupby('usummary_code')['impVal'].sum().reset_index()
            census_region_data.columns = ['usummary_code', 'Census_total_imports']
        
        # Merge TiVA and Census data
        comparison_df = pd.merge(tiva_comparison, census_region_data, on='usummary_code', how='outer')
        comparison_df['TiVA_total_imports'] = comparison_df['TiVA_total_imports'].fillna(0)
        comparison_df['Census_total_imports'] = comparison_df['Census_total_imports'].fillna(0)
        comparison_df['difference'] = comparison_df['Census_total_imports'] - comparison_df['TiVA_total_imports']
        comparison_df['region'] = region_key
        
        all_comparisons[region_key] = comparison_df
        
        print(f"  {region_key}: {len(comparison_df)} usummary codes")
    
    return all_comparisons

def create_html_visualization():
    """Create HTML visualization with two tabs matching 07_TiVA_Import_Values_Comparison.py structure"""
    
    # Get regional data
    original_regional, census_regional = create_regional_aggregations()
    
    # Get comparison data
    comparison_df = load_usummary_comparison_data()
    
    # Get TiVA comparison data
    tiva_comparisons = create_tiva_census_comparison()
    
    # Load BEA hierarchy for code names
    bea_hierarchy_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                                     'data', 'working', '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv')
    bea_hierarchy = pd.read_csv(bea_hierarchy_path)
    usummary_names = bea_hierarchy[['U.Summary', 'undersum title']].drop_duplicates()
    usummary_names = usummary_names.rename(columns={'U.Summary': 'usummary_code', 'undersum title': 'usummary_name'})
    usummary_names['usummary_code'] = usummary_names['usummary_code'].astype(str)
    
    # Tab 1: Regional comparison
    # Merge regional data for plotting
    merged_regional = pd.merge(original_regional, census_regional, 
                              on=['region', 'usummary_code'], 
                              suffixes=('_original', '_census'))
    
    # Add BEA code names
    merged_regional = merged_regional.merge(usummary_names, on='usummary_code', how='left')
    merged_regional['usummary_name'] = merged_regional['usummary_name'].fillna('Unknown')
    
    # Create main regional scatter plot
    fig = px.scatter(
        merged_regional,
        x='impVal_original',
        y='impVal_census',
        color='region',
        hover_data={'usummary_code': True, 'usummary_name': True},
        title='Regional Usummary Code Comparison: Schott vs Census Mapping',
        labels={
            'impVal_original': 'Original Schott Mapping Import Value ($)',
            'impVal_census': 'Census Mapping Import Value ($)',
            'region': 'Region'
        },
        template='plotly_white'
    )
    
    # Add 45-degree line
    max_val = max(merged_regional[['impVal_original', 'impVal_census']].max())
    fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                  line=dict(color='red', dash='dash', width=2),
                  name='Perfect Agreement')
    
    fig.update_layout(
        height=600,
        legend=dict(
            x=1.02, y=1, xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1
        ),
        margin=dict(r=200)
    )
    
    # Tab 2: Country-level comparison
    # Filter for significant differences only
    significant_diffs = comparison_df[abs(comparison_df['impVal_pct_difference']) > 1].copy()
    significant_diffs = significant_diffs.head(100)  # Top 100 differences
    
    fig2 = px.scatter(
        significant_diffs,
        x='impVal_original',
        y='impVal_alternative',
        color='impVal_pct_difference',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        hover_data={'Country': True, 'bea_code': True, 'impVal_pct_difference': ':.1f'},
        title='Country-Level Usummary Differences (Top 100, >1% Difference)',
        labels={
            'impVal_original': 'Original Mapping Import Value ($)',
            'impVal_alternative': 'Alternative Mapping Import Value ($)',
            'impVal_pct_difference': '% Difference'
        },
        template='plotly_white'
    )
    
    # Add 45-degree line
    max_val2 = max(significant_diffs[['impVal_original', 'impVal_alternative']].max())
    fig2.add_shape(type='line', x0=0, y0=0, x1=max_val2, y1=max_val2,
                   line=dict(color='red', dash='dash', width=2),
                   name='Perfect Agreement')
    
    fig2.update_layout(height=600)
    
    # Tab 3: TiVA vs Census comparison - create individual plots for each region
    tiva_figures = {}
    region_order = ['CAN', 'CHN', 'Europe', 'JPN', 'MEX', 'RoAsia', 'RoWorld', 'world']
    
    for region_key in region_order:
        if region_key not in tiva_comparisons:
            continue
            
        tiva_df = tiva_comparisons[region_key].copy()
        tiva_df = tiva_df.merge(usummary_names, on='usummary_code', how='left')
        tiva_df['usummary_name'] = tiva_df['usummary_name'].fillna('Unknown')
        
        # Filter out zero values for better visualization
        tiva_filtered = tiva_df[
            (tiva_df['Census_total_imports'] > 0) & 
            (tiva_df['TiVA_total_imports'] > 0)
        ].copy()
        
        if len(tiva_filtered) == 0:
            continue
            
        region_title = 'World Total' if region_key == 'world' else region_key
        
        fig_tiva = px.scatter(
            tiva_filtered,
            x='Census_total_imports',
            y='TiVA_total_imports',
            hover_data={'usummary_code': True, 'usummary_name': True},
            title=f'{region_title}: Census Mapping vs TiVA Benchmark',
            labels={
                'Census_total_imports': 'Census Mapping Import Value ($)',
                'TiVA_total_imports': 'TiVA Import Value ($)'
            },
            template='plotly_white'
        )
        
        # Add 45-degree line
        max_val_region = max(tiva_filtered[['Census_total_imports', 'TiVA_total_imports']].max())
        fig_tiva.add_shape(type='line', x0=0, y0=0, x1=max_val_region, y1=max_val_region,
                          line=dict(color='red', dash='dash', width=2),
                          name='Perfect Agreement')
        
        fig_tiva.update_layout(
            height=500,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        tiva_figures[region_key] = fig_tiva
    
    # Create HTML with tabs using shared styling
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Regional Mapping Comparison: Schott vs Census</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            {get_shared_css()}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Regional Mapping Comparison</h1>
            <p>Schott vs Census Mapping - Import Value Analysis</p>
        </div>
        
        <a href="../../../validation_index.html" class="back-button">Back to Validation Dashboard</a>
        
        <div class="tabs">
            <button class="tablinks" onclick="openTab(event, 'Regional')" id="defaultOpen">Regional Comparison</button>
            <button class="tablinks" onclick="openTab(event, 'Country')">Country-Level Differences</button>
            <button class="tablinks" onclick="openTab(event, 'TiVA')">Census vs TiVA Benchmark</button>
        </div>
        
        <div id="Regional" class="tabcontent">
            <h2>Regional Aggregation Comparison</h2>
            <div class="summary">
                <p>This chart compares regional import values aggregated by usummary codes between the original Schott mapping and the new Census mapping.</p>
                <p>Each point represents a (Region, Usummary Code) combination. Points on the 45° line indicate perfect agreement between the two mapping approaches.</p>
                <p>Total regional import values are validated to match exactly between both approaches.</p>
            </div>
            <div id="regional-plot"></div>
        </div>
        
        <div id="Country" class="tabcontent">
            <h2>Country-Level Usummary Differences</h2>
            <div class="summary">
                <p>This chart shows country-level differences in usummary codes where the percentage difference exceeds 1%.</p>
                <p>Color indicates the percentage difference: blue for Census mapping higher, red for original mapping higher.</p>
                <p>Only significant differences (>1%) are shown to focus on meaningful changes.</p>
            </div>
            <div id="country-plot"></div>
        </div>
        
        <div id="TiVA" class="tabcontent">
            <h2>Census Mapping vs TiVA Benchmark</h2>
            <div class="summary">
                <p>These charts compare our Census mapping import values against the official TiVA benchmark data for each region and usummary code.</p>
                <p>Points on the 45° line indicate perfect agreement with TiVA benchmarks. This validates the accuracy of our Census-based mapping approach.</p>
                <p>Only positive values from both sources are shown for meaningful comparison.</p>
            </div>
            {"".join([f'<div id="tiva-plot-{region}"></div>' for region in region_order if region in tiva_figures])}
        </div>
        
        
        <div class="footer">
            <p>Generated for FRBAtl TariffPricePulse Project | Regional Mapping Validation</p>
        </div>
        
        <script>
            {get_shared_javascript()}
            
            // Plot the figures
            var regionalData = {fig.to_json()};
            Plotly.newPlot('regional-plot', regionalData.data, regionalData.layout);
            
            var countryData = {fig2.to_json()};
            Plotly.newPlot('country-plot', countryData.data, countryData.layout);
            
            // Plot TiVA comparison charts for each region
            {"; ".join([f"var tivaData_{region} = {tiva_figures[region].to_json()}; Plotly.newPlot('tiva-plot-{region}', tivaData_{region}.data, tivaData_{region}.layout)" for region in region_order if region in tiva_figures])};
        </script>
    </body>
    </html>
    """
    
    # Save HTML file  
    output_path = os.path.join(data_paths['base_paths']['underlying_data_root'], 
                              'validations', 'Alternative_Census_Mappings', 
                              '11_Regional_Mapping_Comparison.html')
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nHTML visualization saved to: {output_path}")

if __name__ == "__main__":
    create_html_visualization()