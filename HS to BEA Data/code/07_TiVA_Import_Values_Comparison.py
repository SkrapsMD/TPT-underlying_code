import os
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from main_pipeline_run import get_data_path

"""
DESCRIPTION: This code validates our constructed trade weights and import values by comparing them
against benchmark data from the TiVA (Trade in Value Added) database for 2023.

THE CORE VALIDATION CHALLENGE:
We've constructed regional trade weights and import values by aggregating HS commodity codes 
to BEA economic categories for 7 regions (CAN, CHN, Europe, JPN, MEX, RoAsia, RoWorld). 
We need to validate these constructed values against established benchmarks to ensure accuracy.

DATA SOURCES:
1. CONSTRUCTED DATA: data/working/05_Trade_weights/usummary_trade_weights.csv
   - Contains regional_denominator (our calculated import values by region and BEA category)
   - Contains world_denominator (our calculated total world import values by BEA category)
   - Built from HS commodity codes mapped through NAICS to BEA categories

2. BENCHMARK DATA: data/raw/TiVA Tables/*.csv (CAN.csv, CHN.csv, EUR.csv, etc.)
   - Contains official TiVA 2023 import values by BEA category for comparison
   - Values are in millions of dollars (converted to dollars for comparison)

VALIDATION PROCESS:
1. Regional Sum Validation (lines 31-34): Verify that sum of 7 regional denominators 
   equals world denominator for each BEA category (should be mathematically exact)

2. TiVA Benchmark Comparison (lines 65-124): Compare our constructed values against 
   TiVA benchmarks for each region and BEA category:
   - Read TiVA data and convert millions to dollars
   - Merge with our constructed values
   - Calculate differences and create comparison files
   - Generate interactive scatter plots showing correlation

3. Data Consistency Mapping (lines 26-29): Map S003 codes to "Other" category
   to maintain consistency with 05_Trade_weights.py processing

MAIN OUTPUTS:
- 01_regional_world_denominator_comparison.csv: Validates regional sums equal world totals
- 02_TiVA_vs_HS_Import_Charts.html: Interactive dashboard showing all regional comparisons
- regional_HS_BEA_mapping/: Organized comparison files, plots, and data by region
  - csv/: Detailed comparison data with differences
  - html/: Interactive scatter plots with hover info showing BEA category names
  - png/: Static versions of scatter plots

The scatter plots show correlation between our constructed values (x-axis) and TiVA benchmarks
(y-axis), with a red dashed line indicating perfect correlation. Points close to the line
indicate good agreement between our methodology and established benchmarks.
"""

# Manual USATradeOnline data for comparison
usa_trade_online_data = {
    'CHN': {'2023': 427246582836, '2024': 438741998078},
    'JPN': {'2023': 147206391795, '2024': 148370517793},
    'Europe': {'2023': 725524830151, '2024': 770782051188},
    'CAN': {'2023': 418010321977, '2024': 411886683368},
    'MEX': {'2023': 472907368135, '2024': 505523174525}
}

# Load data paths configuration
data_paths_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_paths.json')
with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Read the trade weights data
trade_weights_path = os.path.join(data_paths['base_paths']['working_data'], '05_Trade_weights', 'usummary_trade_weights.csv')
df = pd.read_csv(trade_weights_path)

# Get unique regions (excluding 'region' header)
regions = df['region'].unique()
regions = [r for r in regions if r != 'region']
regional_dfs = {}
for region in regions:
    regional_dfs[region] = df[df['region'] == region][['usummary_code', 'regional_denominator']].drop_duplicates()
# World denominator mapping
world_df = df[['usummary_code', 'world_denominator']].drop_duplicates()

# Map S003 to Other like in 05_Trade_weights.py
for region in regions:
    regional_dfs[region]['usummary_code'] = regional_dfs[region]['usummary_code'].replace({'S003': 'Other'})
world_df['usummary_code'] = world_df['usummary_code'].replace({'S003': 'Other'})

# Create validation analysis -- Do the 7 regions sum to the world total? They should. And they Do!!! 
validation_df = df.groupby(['usummary_code', 'region'])['regional_denominator'].first().unstack(fill_value=0).reset_index()
validation_df['sum_7_regions'] = validation_df[regions].sum(axis=1)
validation_df = validation_df.merge(world_df, on='usummary_code')
validation_df['difference'] = validation_df['world_denominator'] - validation_df['sum_7_regions']
validation_dir = os.path.join(data_paths['base_paths']['validations'], '07_TiVA_Import_Values_Comparison')
os.makedirs(validation_dir, exist_ok=True)
validation_path = os.path.join(validation_dir, '01_regional_world_denominator_comparison.csv')
validation_df.to_csv(validation_path, index=False)

# Read BEA hierarchy for usummary_code names
bea_hierarchy_path = os.path.join(data_paths['base_paths']['working_data'], '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv')
bea_hierarchy = pd.read_csv(bea_hierarchy_path)
usummary_names = bea_hierarchy[['U.Summary', 'undersum title']].drop_duplicates()
usummary_names = usummary_names.rename(columns={'U.Summary': 'usummary_code', 'undersum title': 'usummary_name'})
usummary_names['usummary_code'] = usummary_names['usummary_code'].astype(str)

# Read TiVA Tables and compare with regional denominators
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

# Create subfolders for regional comparisons
regional_comparison_dir = os.path.join(validation_dir, 'regional_HS_BEA_mapping')
csv_dir = os.path.join(regional_comparison_dir, 'csv')
html_dir = os.path.join(regional_comparison_dir, 'html')
html_log_dir = os.path.join(regional_comparison_dir, 'html_log')
png_dir = os.path.join(regional_comparison_dir, 'png')
png_log_dir = os.path.join(regional_comparison_dir, 'png_log')
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)
os.makedirs(html_log_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)
os.makedirs(png_log_dir, exist_ok=True)

# Store all comparison data for the discrepancies table
all_comparisons = {}

for tiva_file, region_key in region_mapping.items():
    tiva_path = os.path.join(tiva_dir, tiva_file)
    tiva_df = pd.read_csv(tiva_path)
    
    # Keep only first and last columns
    tiva_comparison = tiva_df.iloc[:, [0, -1]].copy()
    tiva_comparison.columns = ['usummary_code', 'TiVA_total_imports']
    
    # Convert usummary_code to string and multiply TiVA values by 1,000,000
    tiva_comparison['usummary_code'] = tiva_comparison['usummary_code'].astype(str)
    tiva_comparison['TiVA_total_imports'] = tiva_comparison['TiVA_total_imports'] * 1000000
    
    # Get corresponding regional/world data
    if region_key == 'world':
        comparison_data = world_df.copy()
        comparison_data.columns = ['usummary_code', 'HS_total_imports']
    else:
        comparison_data = regional_dfs[region_key].copy()
        comparison_data.columns = ['usummary_code', 'HS_total_imports']
    
    # Merge and calculate difference
    comparison_data['usummary_code'] = comparison_data['usummary_code'].astype(str)
    merged_comparison = comparison_data.merge(tiva_comparison, on='usummary_code', how='outer')
    merged_comparison = merged_comparison.fillna(0)
    merged_comparison['difference'] = merged_comparison['HS_total_imports'] - merged_comparison['TiVA_total_imports']
    
    # Add usummary_code names for hover info
    merged_comparison = merged_comparison.merge(usummary_names, on='usummary_code', how='left')
    merged_comparison['usummary_name'] = merged_comparison['usummary_name'].fillna('Unknown')
    
    # Save comparison CSV
    output_filename = f'{region_key}_HS_TiVA_comparison.csv'
    output_path = os.path.join(csv_dir, output_filename)
    merged_comparison.to_csv(output_path, index=False)
    
    # Store for discrepancies table
    all_comparisons[region_key] = merged_comparison.copy()
    
    # Calculate correlation statistics
    valid_data = merged_comparison[(merged_comparison['HS_total_imports'] > 0) | (merged_comparison['TiVA_total_imports'] > 0)]
    if len(valid_data) > 1:
        correlation = np.corrcoef(valid_data['HS_total_imports'], valid_data['TiVA_total_imports'])[0, 1]
        r2 = r2_score(valid_data['TiVA_total_imports'], valid_data['HS_total_imports'])
        title_text = f'{region_key} - HS to BEA vs TiVA Imports (R² = {r2:.3f}, r = {correlation:.3f})'
    else:
        title_text = f'{region_key} - HS to BEA vs TiVA Imports'
    
    # Create interactive scatter plot with Plotly (regular scale)
    fig = px.scatter(merged_comparison, 
                     x='HS_total_imports', 
                     y='TiVA_total_imports',
                     hover_data=['usummary_code', 'usummary_name'],
                     labels={'HS_total_imports': 'HS to BEA Imports (2024)',
                             'TiVA_total_imports': 'TiVA Imports (2023)'},
                     title=title_text,
                     template='plotly_dark')
    
    # Add 45-degree line for reference
    max_val = max(merged_comparison['HS_total_imports'].max(), merged_comparison['TiVA_total_imports'].max())
    fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                  line=dict(color='red', dash='dash', width=2),
                  name='Perfect correlation')
    
    # Save interactive HTML plot
    html_filename = f'{region_key}_HS_TiVA_scatter.html'
    html_path = os.path.join(html_dir, html_filename)
    fig.write_html(html_path)
    
    # Save static PNG plot
    png_filename = f'{region_key}_HS_TiVA_scatter.png'
    png_path = os.path.join(png_dir, png_filename)
    fig.write_image(png_path, width=1200, height=800)
    
    # Create logged version plots
    # Filter out zero values for log scale
    log_data = merged_comparison[
        (merged_comparison['HS_total_imports'] > 0) & 
        (merged_comparison['TiVA_total_imports'] > 0)
    ].copy()
    
    if len(log_data) > 0:
        # Create logged scatter plot
        fig_log = px.scatter(log_data, 
                             x='HS_total_imports', 
                             y='TiVA_total_imports',
                             hover_data=['usummary_code', 'usummary_name'],
                             labels={'HS_total_imports': 'HS to BEA Imports (2024)',
                                     'TiVA_total_imports': 'TiVA Imports (2023)'},
                             title=title_text + ' (Log Scale)',
                             template='plotly_dark',
                             log_x=True,
                             log_y=True)
        
        # Add 45-degree line for reference on log scale
        min_val = min(log_data['HS_total_imports'].min(), log_data['TiVA_total_imports'].min())
        max_val_log = max(log_data['HS_total_imports'].max(), log_data['TiVA_total_imports'].max())
        fig_log.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val_log, y1=max_val_log,
                          line=dict(color='red', dash='dash', width=2),
                          name='Perfect correlation')
        
        # Save logged HTML plot
        html_log_filename = f'{region_key}_HS_TiVA_scatter_log.html'
        html_log_path = os.path.join(html_log_dir, html_log_filename)
        fig_log.write_html(html_log_path)
        
        # Save logged PNG plot
        png_log_filename = f'{region_key}_HS_TiVA_scatter_log.png'
        png_log_path = os.path.join(png_log_dir, png_log_filename)
        fig_log.write_image(png_log_path, width=1200, height=800)

# Create regional aggregate data for the new tab
regional_aggregate_data = []
regional_aggregate_goods_only_data = []

for region_key, comparison_df in all_comparisons.items():
    if region_key != 'world':  # Skip world total for regional aggregates
        # Full aggregates (all BEA codes)
        hs_total = comparison_df['HS_total_imports'].sum()
        tiva_total = comparison_df['TiVA_total_imports'].sum()
        regional_aggregate_data.append({
            'region': region_key,
            'HS_total_imports': hs_total,
            'TiVA_total_imports': tiva_total,
            'difference': hs_total - tiva_total
        })
        
        # Goods-only aggregates (only BEA codes with non-zero HS imports)
        goods_only_df = comparison_df[comparison_df['HS_total_imports'] > 0]
        hs_goods_total = goods_only_df['HS_total_imports'].sum()
        tiva_goods_total = goods_only_df['TiVA_total_imports'].sum()
        regional_aggregate_goods_only_data.append({
            'region': region_key,
            'HS_total_imports': hs_goods_total,
            'TiVA_total_imports': tiva_goods_total,
            'difference': hs_goods_total - tiva_goods_total,
            'bea_codes_count': len(goods_only_df)
        })

# Add world total to regional aggregate data
if 'world' in all_comparisons:
    world_comparison = all_comparisons['world']
    
    # Full world aggregates
    world_hs_total = world_comparison['HS_total_imports'].sum()
    world_tiva_total = world_comparison['TiVA_total_imports'].sum()
    regional_aggregate_data.append({
        'region': 'World Total',
        'HS_total_imports': world_hs_total,
        'TiVA_total_imports': world_tiva_total,
        'difference': world_hs_total - world_tiva_total
    })
    
    # Goods-only world aggregates
    world_goods_only_df = world_comparison[world_comparison['HS_total_imports'] > 0]
    world_hs_goods_total = world_goods_only_df['HS_total_imports'].sum()
    world_tiva_goods_total = world_goods_only_df['TiVA_total_imports'].sum()
    regional_aggregate_goods_only_data.append({
        'region': 'World Total',
        'HS_total_imports': world_hs_goods_total,
        'TiVA_total_imports': world_tiva_goods_total,
        'difference': world_hs_goods_total - world_tiva_goods_total,
        'bea_codes_count': len(world_goods_only_df)
    })

regional_aggregate_df = pd.DataFrame(regional_aggregate_data)
regional_aggregate_goods_only_df = pd.DataFrame(regional_aggregate_goods_only_data)

# Create regional aggregate scatter plots with USATradeOnline data
if len(regional_aggregate_df) > 0:
    # Calculate correlation statistics for regional aggregates
    valid_regional_data = regional_aggregate_df[
        (regional_aggregate_df['HS_total_imports'] > 0) | 
        (regional_aggregate_df['TiVA_total_imports'] > 0)
    ]
    
    if len(valid_regional_data) > 1:
        correlation_regional = np.corrcoef(valid_regional_data['HS_total_imports'], valid_regional_data['TiVA_total_imports'])[0, 1]
        r2_regional = r2_score(valid_regional_data['TiVA_total_imports'], valid_regional_data['HS_total_imports'])
        title_regional = f'Regional Aggregates - HS to BEA vs TiVA Imports (R² = {r2_regional:.3f}, r = {correlation_regional:.3f})'
    else:
        title_regional = 'Regional Aggregates - HS to BEA vs TiVA Imports'
    
    # Create regular scale regional aggregate plot with multiple data sources
    fig_regional = go.Figure()
    
    # Add HS to BEA mapped data (original dots)
    fig_regional.add_trace(go.Scatter(
        x=regional_aggregate_df['HS_total_imports'],
        y=regional_aggregate_df['TiVA_total_imports'],
        mode='markers+text',
        text=regional_aggregate_df['region'],
        textposition='top center',
        marker=dict(size=10, color='#636EFA'),
        name='HS to BEA Mapped (2024)',
        hovertemplate='<b>%{text}</b><br>HS to BEA: $%{x:,.0f}<br>TiVA: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add USATradeOnline 2024 data (X markers, same color)
    for _, row in regional_aggregate_df.iterrows():
        region = row['region']
        tiva_value = row['TiVA_total_imports']
        
        # Map region names to USATradeOnline data keys
        region_mapping = {'CAN': 'CAN', 'CHN': 'CHN', 'Europe': 'Europe', 'JPN': 'JPN', 'MEX': 'MEX'}
        
        if region in region_mapping and region_mapping[region] in usa_trade_online_data:
            usa_2024_value = usa_trade_online_data[region_mapping[region]]['2024']
            fig_regional.add_trace(go.Scatter(
                x=[usa_2024_value],
                y=[tiva_value],
                mode='markers+text',
                text=[region],
                textposition='bottom center',
                marker=dict(size=12, color='#636EFA', symbol='x'),
                name='USATradeOnline 2024' if region == 'CAN' else '',
                showlegend=region == 'CAN',
                hovertemplate=f'<b>{region}</b><br>USATradeOnline 2024: $%{{x:,.0f}}<br>TiVA: $%{{y:,.0f}}<extra></extra>'
            ))
    
    # Add USATradeOnline 2023 data (circles, different color)
    for _, row in regional_aggregate_df.iterrows():
        region = row['region']
        tiva_value = row['TiVA_total_imports']
        
        # Map region names to USATradeOnline data keys
        region_mapping = {'CAN': 'CAN', 'CHN': 'CHN', 'Europe': 'Europe', 'JPN': 'JPN', 'MEX': 'MEX'}
        
        if region in region_mapping and region_mapping[region] in usa_trade_online_data:
            usa_2023_value = usa_trade_online_data[region_mapping[region]]['2023']
            fig_regional.add_trace(go.Scatter(
                x=[usa_2023_value],
                y=[tiva_value],
                mode='markers+text',
                text=[region],
                textposition='middle right',
                marker=dict(size=10, color='#FF6692'),
                name='USATradeOnline 2023' if region == 'CAN' else '',
                showlegend=region == 'CAN',
                hovertemplate=f'<b>{region}</b><br>USATradeOnline 2023: $%{{x:,.0f}}<br>TiVA: $%{{y:,.0f}}<extra></extra>'
            ))
    
    # Add 45-degree line for reference
    max_val_regional = max(
        regional_aggregate_df['HS_total_imports'].max(), 
        regional_aggregate_df['TiVA_total_imports'].max(),
        max([usa_trade_online_data[region]['2024'] for region in usa_trade_online_data.keys()]) if usa_trade_online_data else 0
    )
    fig_regional.add_shape(type='line', x0=0, y0=0, x1=max_val_regional, y1=max_val_regional,
                          line=dict(color='red', dash='dash', width=2),
                          name='Perfect correlation')
    
    # Update layout
    fig_regional.update_layout(
        title=title_regional,
        xaxis_title='Import Values (USD)',
        yaxis_title='TiVA Imports (2023, USD)',
        template='plotly_dark',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )
    
    # Save regional aggregate HTML plot
    regional_aggregate_path = os.path.join(html_dir, 'regional_aggregate_scatter.html')
    fig_regional.write_html(regional_aggregate_path)
    
    # Create log scale regional aggregate plot
    log_regional_data = regional_aggregate_df[
        (regional_aggregate_df['HS_total_imports'] > 0) & 
        (regional_aggregate_df['TiVA_total_imports'] > 0)
    ].copy()
    
    if len(log_regional_data) > 0:
        # Create log scale plot with multiple data sources
        fig_regional_log = go.Figure()
        
        # Add HS to BEA mapped data (original dots)
        fig_regional_log.add_trace(go.Scatter(
            x=log_regional_data['HS_total_imports'],
            y=log_regional_data['TiVA_total_imports'],
            mode='markers+text',
            text=log_regional_data['region'],
            textposition='top center',
            marker=dict(size=10, color='#636EFA'),
            name='HS to BEA Mapped (2024)',
            hovertemplate='<b>%{text}</b><br>HS to BEA: $%{x:,.0f}<br>TiVA: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add USATradeOnline 2024 data (X markers, same color)
        for _, row in log_regional_data.iterrows():
            region = row['region']
            tiva_value = row['TiVA_total_imports']
            
            # Map region names to USATradeOnline data keys
            region_mapping = {'CAN': 'CAN', 'CHN': 'CHN', 'Europe': 'Europe', 'JPN': 'JPN', 'MEX': 'MEX'}
            
            if region in region_mapping and region_mapping[region] in usa_trade_online_data:
                usa_2024_value = usa_trade_online_data[region_mapping[region]]['2024']
                if usa_2024_value > 0:  # Only add if positive for log scale
                    fig_regional_log.add_trace(go.Scatter(
                        x=[usa_2024_value],
                        y=[tiva_value],
                        mode='markers+text',
                        text=[region],
                        textposition='bottom center',
                        marker=dict(size=12, color='#636EFA', symbol='x'),
                        name='USATradeOnline 2024' if region == 'CAN' else '',
                        showlegend=region == 'CAN',
                        hovertemplate=f'<b>{region}</b><br>USATradeOnline 2024: $%{{x:,.0f}}<br>TiVA: $%{{y:,.0f}}<extra></extra>'
                    ))
        
        # Add USATradeOnline 2023 data (circles, different color)
        for _, row in log_regional_data.iterrows():
            region = row['region']
            tiva_value = row['TiVA_total_imports']
            
            # Map region names to USATradeOnline data keys
            region_mapping = {'CAN': 'CAN', 'CHN': 'CHN', 'Europe': 'Europe', 'JPN': 'JPN', 'MEX': 'MEX'}
            
            if region in region_mapping and region_mapping[region] in usa_trade_online_data:
                usa_2023_value = usa_trade_online_data[region_mapping[region]]['2023']
                if usa_2023_value > 0:  # Only add if positive for log scale
                    fig_regional_log.add_trace(go.Scatter(
                        x=[usa_2023_value],
                        y=[tiva_value],
                        mode='markers+text',
                        text=[region],
                        textposition='middle right',
                        marker=dict(size=10, color='#FF6692'),
                        name='USATradeOnline 2023' if region == 'CAN' else '',
                        showlegend=region == 'CAN',
                        hovertemplate=f'<b>{region}</b><br>USATradeOnline 2023: $%{{x:,.0f}}<br>TiVA: $%{{y:,.0f}}<extra></extra>'
                    ))
        
        # Add 45-degree line for reference on log scale
        min_val_regional = min(
            log_regional_data['HS_total_imports'].min(), 
            log_regional_data['TiVA_total_imports'].min(),
            min([usa_trade_online_data[region]['2023'] for region in usa_trade_online_data.keys()]) if usa_trade_online_data else 1
        )
        max_val_regional_log = max(
            log_regional_data['HS_total_imports'].max(), 
            log_regional_data['TiVA_total_imports'].max(),
            max([usa_trade_online_data[region]['2024'] for region in usa_trade_online_data.keys()]) if usa_trade_online_data else 1
        )
        fig_regional_log.add_shape(type='line', x0=min_val_regional, y0=min_val_regional, x1=max_val_regional_log, y1=max_val_regional_log,
                                  line=dict(color='red', dash='dash', width=2),
                                  name='Perfect correlation')
        
        # Update layout for log scale
        fig_regional_log.update_layout(
            title=title_regional + ' (Log Scale)',
            xaxis_title='Import Values (USD)',
            yaxis_title='TiVA Imports (2023, USD)',
            template='plotly_dark',
            xaxis_type='log',
            yaxis_type='log',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            )
        )
        
        # Save regional aggregate log HTML plot
        regional_aggregate_log_path = os.path.join(html_log_dir, 'regional_aggregate_scatter_log.html')
        fig_regional_log.write_html(regional_aggregate_log_path)

# Create goods-only regional aggregate scatter plots
if len(regional_aggregate_goods_only_df) > 0:
    # Calculate correlation statistics for goods-only regional aggregates
    valid_goods_only_data = regional_aggregate_goods_only_df[
        (regional_aggregate_goods_only_df['HS_total_imports'] > 0) | 
        (regional_aggregate_goods_only_df['TiVA_total_imports'] > 0)
    ]
    
    if len(valid_goods_only_data) > 1:
        correlation_goods_only = np.corrcoef(valid_goods_only_data['HS_total_imports'], valid_goods_only_data['TiVA_total_imports'])[0, 1]
        r2_goods_only = r2_score(valid_goods_only_data['TiVA_total_imports'], valid_goods_only_data['HS_total_imports'])
        title_goods_only = f'Goods-Only Regional Aggregates - HS to BEA vs TiVA Imports (R² = {r2_goods_only:.3f}, r = {correlation_goods_only:.3f})'
    else:
        title_goods_only = 'Goods-Only Regional Aggregates - HS to BEA vs TiVA Imports'
    
    # Create regular scale goods-only regional aggregate plot
    fig_goods_only = go.Figure()
    
    # Add HS to BEA mapped data (original dots)
    fig_goods_only.add_trace(go.Scatter(
        x=regional_aggregate_goods_only_df['HS_total_imports'],
        y=regional_aggregate_goods_only_df['TiVA_total_imports'],
        mode='markers+text',
        text=regional_aggregate_goods_only_df['region'],
        textposition='top center',
        marker=dict(size=10, color='#636EFA'),
        name='HS to BEA Mapped (2024) - Goods Only',
        hovertemplate='<b>%{text}</b><br>HS to BEA: $%{x:,.0f}<br>TiVA: $%{y:,.0f}<br>BEA Codes: %{customdata}<extra></extra>',
        customdata=regional_aggregate_goods_only_df['bea_codes_count']
    ))
    
    # Add 45-degree line for reference
    max_val_goods_only = max(
        regional_aggregate_goods_only_df['HS_total_imports'].max(), 
        regional_aggregate_goods_only_df['TiVA_total_imports'].max()
    )
    fig_goods_only.add_shape(type='line', x0=0, y0=0, x1=max_val_goods_only, y1=max_val_goods_only,
                             line=dict(color='red', dash='dash', width=2),
                             name='Perfect correlation')
    
    # Update layout
    fig_goods_only.update_layout(
        title=title_goods_only,
        xaxis_title='HS to BEA Import Values (USD) - Goods Only',
        yaxis_title='TiVA Imports (2023, USD) - Goods Only',
        template='plotly_dark',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )
    
    # Save goods-only regional aggregate HTML plot
    goods_only_aggregate_path = os.path.join(html_dir, 'regional_aggregate_goods_only_scatter.html')
    fig_goods_only.write_html(goods_only_aggregate_path)
    
    # Create log scale goods-only regional aggregate plot
    log_goods_only_data = regional_aggregate_goods_only_df[
        (regional_aggregate_goods_only_df['HS_total_imports'] > 0) & 
        (regional_aggregate_goods_only_df['TiVA_total_imports'] > 0)
    ].copy()
    
    if len(log_goods_only_data) > 0:
        # Create log scale plot
        fig_goods_only_log = go.Figure()
        
        # Add HS to BEA mapped data (original dots)
        fig_goods_only_log.add_trace(go.Scatter(
            x=log_goods_only_data['HS_total_imports'],
            y=log_goods_only_data['TiVA_total_imports'],
            mode='markers+text',
            text=log_goods_only_data['region'],
            textposition='top center',
            marker=dict(size=10, color='#636EFA'),
            name='HS to BEA Mapped (2024) - Goods Only',
            hovertemplate='<b>%{text}</b><br>HS to BEA: $%{x:,.0f}<br>TiVA: $%{y:,.0f}<br>BEA Codes: %{customdata}<extra></extra>',
            customdata=log_goods_only_data['bea_codes_count']
        ))
        
        # Add 45-degree line for reference on log scale
        min_val_goods_only = min(
            log_goods_only_data['HS_total_imports'].min(), 
            log_goods_only_data['TiVA_total_imports'].min()
        )
        max_val_goods_only_log = max(
            log_goods_only_data['HS_total_imports'].max(), 
            log_goods_only_data['TiVA_total_imports'].max()
        )
        fig_goods_only_log.add_shape(type='line', x0=min_val_goods_only, y0=min_val_goods_only, x1=max_val_goods_only_log, y1=max_val_goods_only_log,
                                     line=dict(color='red', dash='dash', width=2),
                                     name='Perfect correlation')
        
        # Update layout for log scale
        fig_goods_only_log.update_layout(
            title=title_goods_only + ' (Log Scale)',
            xaxis_title='HS to BEA Import Values (USD) - Goods Only',
            yaxis_title='TiVA Imports (2023, USD) - Goods Only',
            template='plotly_dark',
            xaxis_type='log',
            yaxis_type='log',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            )
        )
        
        # Save goods-only regional aggregate log HTML plot
        goods_only_aggregate_log_path = os.path.join(html_log_dir, 'regional_aggregate_goods_only_scatter_log.html')
        fig_goods_only_log.write_html(goods_only_aggregate_log_path)

# Create discrepancies table (BEA codes with >30% difference)
discrepancies_list = []
for region_key, comparison_df in all_comparisons.items():
    # Calculate percentage difference
    valid_comparison = comparison_df[
        (comparison_df['HS_total_imports'] > 0) | (comparison_df['TiVA_total_imports'] > 0)
    ].copy()
    
    # Calculate percentage difference (use max of the two values as denominator to avoid division by zero)
    valid_comparison['max_value'] = np.maximum(valid_comparison['HS_total_imports'], valid_comparison['TiVA_total_imports'])
    valid_comparison['pct_difference'] = np.abs(valid_comparison['difference']) / valid_comparison['max_value'] * 100
    
    # Filter for >30% difference and non-zero values
    large_discrepancies = valid_comparison[
        (valid_comparison['pct_difference'] > 30) & 
        (valid_comparison['max_value'] > 0)
    ].copy()
    
    for _, row in large_discrepancies.iterrows():
        # Check if this is likely a services category (TiVA > 0, HS = 0)
        likely_services = (row['TiVA_total_imports'] > 0) and (row['HS_total_imports'] == 0)
        
        discrepancies_list.append({
            'region': region_key,
            'usummary_code': row['usummary_code'],
            'usummary_name': row['usummary_name'],
            'HS_total_imports': row['HS_total_imports'],
            'TiVA_total_imports': row['TiVA_total_imports'],
            'difference': row['difference'],
            'pct_difference': row['pct_difference'],
            'likely_services': '✓' if likely_services else ''
        })

discrepancies_df = pd.DataFrame(discrepancies_list)
discrepancies_df = discrepancies_df.sort_values('pct_difference', ascending=False)

# Cross-reference with hierarchical matches to identify mapping uncertainty
# Source 1: Hierarchical matches from 03_Map_country_trade_data (2024 unmapped HS codes)
hierarchical_matches_path = os.path.join(data_paths['base_paths']['validations'], '03_Map_country_trade_data', '3_Hierarchical_Matches.csv')
hierarchical_matches_2024 = pd.DataFrame()
if os.path.exists(hierarchical_matches_path):
    hierarchical_matches_2024 = pd.read_csv(hierarchical_matches_path)
    print(f"Found {len(hierarchical_matches_2024)} hierarchical matches from 2024 trade data")

# Source 2: Hierarchical matches from 01_Schott_Data_Compiler (2023 HS codes)
schott_hierarchical_path = os.path.join(data_paths['base_paths']['validations'], '01_Schott_Data_Compiler', '4_mapping_validation_issues.csv')
schott_hierarchical_matches = pd.DataFrame()
if os.path.exists(schott_hierarchical_path):
    schott_data = pd.read_csv(schott_hierarchical_path)
    print(f"Found {len(schott_data)} hierarchical matches from Schott data")
    
    # Load BEA mappings to convert NAICS to BEA codes
    bea_naics_path = os.path.join(data_paths['base_paths']['working_data'], '02_HS_to_Naics_to_BEA', '01_BEA_naics_mapping.csv')
    bea_hierarchy_path = os.path.join(data_paths['base_paths']['working_data'], '02_HS_to_Naics_to_BEA', '02_BEA_hierarchy.csv')
    
    if os.path.exists(bea_naics_path) and os.path.exists(bea_hierarchy_path):
        bea_naics_mapping = pd.read_csv(bea_naics_path)
        bea_hierarchy = pd.read_csv(bea_hierarchy_path)
        
        # Create NAICS to BEA detail code lookup
        naics_to_bea = dict(zip(bea_naics_mapping['naics'], bea_naics_mapping['Code']))
        
        # Create BEA detail to U.Summary lookup
        bea_detail_to_usummary = dict(zip(bea_hierarchy['Detail'], bea_hierarchy['U.Summary']))
        
        # Convert Schott hierarchical matches to BEA codes
        schott_bea_matches = []
        for _, row in schott_data.iterrows():
            if 'naics_mds' in row and pd.notna(row['naics_mds']):
                naics_code = str(row['naics_mds'])
                if naics_code in naics_to_bea:
                    bea_detail = naics_to_bea[naics_code]
                    bea_usummary = bea_detail_to_usummary.get(bea_detail, bea_detail)
                    
                    schott_bea_matches.append({
                        'hs_code': row.get('hs_code', ''),
                        'matched_bea_detail': bea_detail,
                        'matched_bea_usummary': bea_usummary,
                        'source': 'schott_hierarchical'
                    })
        
        schott_hierarchical_matches = pd.DataFrame(schott_bea_matches)
        print(f"Converted {len(schott_hierarchical_matches)} Schott matches to BEA codes")
    else:
        print("BEA mapping files not found - skipping Schott hierarchical analysis")

# Combine both sources of hierarchical matches
all_hierarchical_bea_codes = set()
if not hierarchical_matches_2024.empty:
    all_hierarchical_bea_codes.update(hierarchical_matches_2024['matched_bea_detail'].unique())
if not schott_hierarchical_matches.empty:
    all_hierarchical_bea_codes.update(schott_hierarchical_matches['matched_bea_detail'].unique())

print(f"Total unique BEA codes from hierarchical matching: {len(all_hierarchical_bea_codes)}")

if len(all_hierarchical_bea_codes) > 0 or not hierarchical_matches_2024.empty:
    
    # Add mapping uncertainty analysis to discrepancies
    discrepancies_enhanced = []
    
    for _, row in discrepancies_df.iterrows():
        bea_code = row['usummary_code']
        
        # Check if this BEA code appears in 2024 hierarchical matches
        matching_entries_2024 = hierarchical_matches_2024[hierarchical_matches_2024['matched_bea_detail'] == bea_code] if not hierarchical_matches_2024.empty else pd.DataFrame()
        
        # Check if this BEA code appears in Schott hierarchical matches
        # Try matching both the detail code and the usummary code
        matching_entries_schott = pd.DataFrame()
        if not schott_hierarchical_matches.empty:
            matching_entries_schott = schott_hierarchical_matches[
                (schott_hierarchical_matches['matched_bea_usummary'] == bea_code) |
                (schott_hierarchical_matches['matched_bea_detail'] == bea_code)
            ]
            
        print(f"Debug: For BEA code {bea_code}, found {len(matching_entries_2024)} 2024 matches and {len(matching_entries_schott)} Schott matches")
        
        if len(matching_entries_2024) > 0 or len(matching_entries_schott) > 0:
            hs_codes = []
            hs_descriptions = []
            primary_mappings = []
            alternative_mappings = []
            sources = []
            
            # Process 2024 hierarchical matches
            if len(matching_entries_2024) > 0:
                hs_groups_2024 = matching_entries_2024.groupby('hs_code')
                
                for hs_code, group in hs_groups_2024:
                    hs_codes.append(str(hs_code))
                    sources.append('2024')
                    
                    # Get description
                    desc = group['hs10_description'].iloc[0] if 'hs10_description' in group.columns else ''
                    hs_descriptions.append(desc)
                    
                    # Get primary mapping
                    primary_match = group[group['match_type'] == 'primary']
                    if len(primary_match) > 0:
                        primary_mappings.append(primary_match['matched_bea_detail'].iloc[0])
                    else:
                        primary_mappings.append(bea_code)
                    
                    # Get alternative mappings
                    alternatives = group[group['match_type'] == 'alternative']['matched_bea_detail'].tolist()
                    alternative_mappings.append('; '.join(alternatives) if alternatives else '')
            
            # Process Schott hierarchical matches
            if len(matching_entries_schott) > 0:
                for _, schott_row in matching_entries_schott.iterrows():
                    hs_code = str(schott_row['hs_code'])
                    if hs_code and hs_code != 'nan':  # Only add non-empty HS codes
                        hs_codes.append(hs_code)
                        sources.append('Schott')
                        hs_descriptions.append('(No description from Schott data)')
                        primary_mappings.append(schott_row['matched_bea_detail'])
                        alternative_mappings.append('(No alternatives from Schott data)')
            
            # Debug: Print what we have before filtering
            print(f"Debug for BEA code {bea_code}:")
            print(f"  hs_codes: {hs_codes}")
            print(f"  hs_descriptions: {hs_descriptions}")
            print(f"  primary_mappings: {primary_mappings}")
            print(f"  alternative_mappings: {alternative_mappings}")
            print(f"  sources: {sources}")
            
            # Create enhanced row with simpler logic
            enhanced_row = row.to_dict()
            enhanced_row.update({
                'has_hierarchical_match': '✓',
                'hs_codes': '; '.join(hs_codes) if hs_codes else 'N/A',
                'hs_descriptions': '; '.join(hs_descriptions) if hs_descriptions else 'N/A',
                'primary_bea_mappings': '; '.join(primary_mappings) if primary_mappings else 'N/A',
                'alternative_bea_mappings': '; '.join(alternative_mappings) if alternative_mappings else 'N/A',
                'mapping_sources': '; '.join(sources) if sources else 'N/A'
            })
            discrepancies_enhanced.append(enhanced_row)
        else:
            # No hierarchical match found
            enhanced_row = row.to_dict()
            enhanced_row.update({
                'has_hierarchical_match': '',
                'hs_codes': '',
                'hs_descriptions': '',
                'primary_bea_mappings': '',
                'alternative_bea_mappings': '',
                'mapping_sources': ''
            })
            discrepancies_enhanced.append(enhanced_row)
    
    discrepancies_df = pd.DataFrame(discrepancies_enhanced)
    print(f"Enhanced discrepancies table with hierarchical mapping analysis")
    
    # Count how many discrepancies have hierarchical matches
    hierarchical_matches_count = (discrepancies_df['has_hierarchical_match'] == '✓').sum()
    print(f"Found {hierarchical_matches_count} out of {len(discrepancies_df)} discrepancies with hierarchical mapping uncertainty")
else:
    print("Hierarchical matches file not found - skipping mapping uncertainty analysis")

# Save enhanced discrepancies table
discrepancies_path = os.path.join(validation_dir, '03_large_discrepancies_table.csv')
discrepancies_df.to_csv(discrepancies_path, index=False)

# Create discrepancies HTML table
discrepancies_html_table = ""
if len(discrepancies_df) > 0:
    discrepancies_html_table = f"""
    <h2>BEA Codes with >30% Difference from TiVA</h2>
    <p>The ✓ symbol in "Hierarchical?" indicates BEA codes that originated from uncertain hierarchical HS-to-BEA mappings from either 2024 trade data or Schott 2023 data.</p>
    <p>The ✓ symbol in "Likely Services" indicates BEA codes where TiVA has non-zero imports but our HS-to-BEA mapping has zero imports, suggesting these are services categories.</p>
    <table border="1" style="border-collapse: collapse; width: 100%; color: #ffffff; font-size: 11px;">
        <tr style="background-color: #333333;">
            <th>Region</th>
            <th>BEA Code</th>
            <th>BEA Name</th>
            <th>HS Total Imports</th>
            <th>TiVA Total Imports</th>
            <th>Difference</th>
            <th>% Difference</th>
            <th>Likely Services</th>
            <th>Hierarchical?</th>
            <th>HS Codes</th>
            <th>HS Descriptions</th>
            <th>Alternative BEA Codes</th>
            <th>Data Source</th>
        </tr>
    """
    
    for _, row in discrepancies_df.iterrows():
        # Handle new columns safely
        hierarchical_match = row.get('has_hierarchical_match', '')
        hs_codes = row.get('hs_codes', '')
        hs_descriptions = row.get('hs_descriptions', '')
        alternative_mappings = row.get('alternative_bea_mappings', '')
        mapping_sources = row.get('mapping_sources', '')
        likely_services = row.get('likely_services', '')
        
        # Truncate long descriptions for display
        if len(hs_descriptions) > 80:
            hs_descriptions = hs_descriptions[:80] + '...'
        if len(alternative_mappings) > 60:
            alternative_mappings = alternative_mappings[:60] + '...'
        if len(hs_codes) > 60:
            hs_codes = hs_codes[:60] + '...'
        
        discrepancies_html_table += f"""
        <tr>
            <td>{row['region']}</td>
            <td>{row['usummary_code']}</td>
            <td>{row['usummary_name']}</td>
            <td>${row['HS_total_imports']:,.0f}</td>
            <td>${row['TiVA_total_imports']:,.0f}</td>
            <td>${row['difference']:,.0f}</td>
            <td>{row['pct_difference']:.1f}%</td>
            <td>{likely_services}</td>
            <td>{hierarchical_match}</td>
            <td>{hs_codes}</td>
            <td>{hs_descriptions}</td>
            <td>{alternative_mappings}</td>
            <td>{mapping_sources}</td>
        </tr>
        """
    
    discrepancies_html_table += "</table>"
else:
    discrepancies_html_table = "<h2>No BEA codes with >30% difference found</h2>"

# Create a master HTML file with tabs
master_html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TiVA Import Values Comparison - All Regions</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #111111;
            color: #ffffff;
        }}
        .plot-container {{
            margin-bottom: 40px;
            border: 1px solid #333333;
            padding: 20px;
            border-radius: 8px;
        }}
        .intro-text {{
            background-color: #222222;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        h1 {{
            text-align: center;
            color: #ffffff;
        }}
        h2 {{
            color: #ffffff;
            margin-bottom: 10px;
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
        }}
        .tabs {{
            overflow: hidden;
            border: 1px solid #333333;
            background-color: #222222;
            margin-bottom: 20px;
        }}
        .tabs button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            color: #ffffff;
            font-size: 16px;
        }}
        .tabs button:hover {{
            background-color: #333333;
        }}
        .tabs button.active {{
            background-color: #444444;
        }}
        .tabcontent {{
            display: none;
            padding: 12px;
            border: 1px solid #333333;
            border-top: none;
            background-color: #111111;
        }}
        .tabcontent.active {{
            display: block;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #333333;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #333333;
        }}
    </style>
</head>
<body>
    <h1>TiVA Import Values Comparison - All Regions</h1>
    <div class="intro-text">
        <p>This presents a validation of our HS to BEA level imports mapping by comparing imports across our codes, and the BEA's TiVA tables. While our data was created using 2024 import data, the BEA TiVA Tables use 2023 data resulting in some bias. Also, the TiVA tables include values like services (notice the values on the x = 0 line). This is simply a sanity check.</p>
    </div>
    
    <div class="tabs">
        <button class="tablinks active" onclick="openTab(event, 'regional-aggregates')">Regional Aggregates</button>
        <button class="tablinks" onclick="openTab(event, 'regular-plots')">Regular Scale</button>
        <button class="tablinks" onclick="openTab(event, 'log-plots')">Log Scale</button>
        <button class="tablinks" onclick="openTab(event, 'discrepancies')">Large Discrepancies</button>
    </div>
    
    <div id="regional-aggregates" class="tabcontent active">
        <h2>Regional Aggregates</h2>
        <p>This tab shows the sum of all BEA-level imports for each region, with one point per region. Both regular and log scale plots are shown.</p>
        
        <div class="intro-text">
            <h3>Understanding the Regional Comparison Plot</h3>
            <p><strong>Y-axis (TiVA Import Values):</strong> These represent the TiVA table import values, which are the sums of imported commodities across all industries and final uses for 2023 as described by the BEA. These values are <strong>inclusive of both goods and services trades</strong>.</p>
            
            <p><strong>X-axis (Our Import Values):</strong> These are the import values for each region that we compute via our concordance from HS-to-BEA codes. These values are denominated in 2024 dollars and are <strong>non-inclusive of services imports</strong>, which will lead to some deviation from the 45-degree line.</p>
            
            <p><strong>Three Types of Data Points:</strong></p>
            <ul>
                <li><strong>Blue dots (HS to BEA Mapped):</strong> These represent our actual estimates from our crosswalk methodology.</li>
                <li><strong>Blue X markers (USATradeOnline 2024):</strong> These are hard-coded values extracted from USA Trade Online for 2024, providing a contemporary comparison point for regions like Europe, Mexico, Canada, China, and Japan.</li>
                <li><strong>Pink circles (USATradeOnline 2023):</strong> These are hard-coded values extracted from USA Trade Online for 2023, acting as baseline comparisons for what we should be seeing in the absence of our crosswalk.</li>
            </ul>
            
            <p><strong>Goods Only:</strong> Assuming the data construction is correct (i.e. that the correct countries are included where they should be), then the deviations here should be the 
            result of the inclusion of services imports. I identified the ~30 services exports (they are, functionally, the 'commodities' with non-zero import value from the TiVA
            Tables, but with 0 values from the HS to BEA crosswalk. When we remove these, we can similarly create a scatter plot of the goods-only TiVA tables vs the hs-to-bea crosswalk which itself should be goods-only. These are the final two scatters.<p>
        </div>
        
        <div class="plot-container">
            <h3>Regional Aggregates - Regular Scale</h3>
            <iframe src="regional_HS_BEA_mapping/html/regional_aggregate_scatter.html"></iframe>
        </div>
        
        <div class="plot-container">
            <h3>Regional Aggregates - Log Scale</h3>
            <iframe src="regional_HS_BEA_mapping/html_log/regional_aggregate_scatter_log.html"></iframe>
        </div>
        
        <div class="plot-container">
            <h3>Goods-Only Regional Aggregates - Regular Scale</h3>
            <p>This plot shows only the BEA codes that have non-zero values in our HS-to-BEA mapping, providing a more direct goods-to-goods comparison by excluding services-only BEA categories from the TiVA totals.</p>
            <iframe src="regional_HS_BEA_mapping/html/regional_aggregate_goods_only_scatter.html"></iframe>
        </div>
        
        <div class="plot-container">
            <h3>Goods-Only Regional Aggregates - Log Scale</h3>
            <iframe src="regional_HS_BEA_mapping/html_log/regional_aggregate_goods_only_scatter_log.html"></iframe>
        </div>
    </div>
    
    <div id="regular-plots" class="tabcontent">
        <h2>Regular Scale Plots</h2>
"""

# Add each region's regular plot
for region_key in ['world', 'CAN', 'CHN', 'Europe', 'JPN', 'MEX', 'RoAsia', 'RoWorld']:
    html_filename = f'{region_key}_HS_TiVA_scatter.html'
    
    # Special handling for world total
    if region_key == 'world':
        title = "World Total - HS to BEA vs TiVA Imports"
        description = "<p>This is the comparison of the world total from our HS mapping to the world total imports from the TiVA tables.</p>"
    else:
        title = f"{region_key} - HS to BEA vs TiVA Imports"
        description = ""
    
    master_html_content += f"""
        <div class="plot-container">
            <h3>{title}</h3>
            {description}
            <iframe src="regional_HS_BEA_mapping/html/{html_filename}"></iframe>
        </div>
    """

master_html_content += """
    </div>
    
    <div id="log-plots" class="tabcontent">
        <h2>Log Scale Plots</h2>
"""

# Add each region's log plot
for region_key in ['world', 'CAN', 'CHN', 'Europe', 'JPN', 'MEX', 'RoAsia', 'RoWorld']:
    html_log_filename = f'{region_key}_HS_TiVA_scatter_log.html'
    
    # Special handling for world total
    if region_key == 'world':
        title = "World Total - HS to BEA vs TiVA Imports (Log Scale)"
        description = "<p>This is the comparison of the world total from our HS mapping to the world total imports from the TiVA tables on a log scale.</p>"
    else:
        title = f"{region_key} - HS to BEA vs TiVA Imports (Log Scale)"
        description = ""
    
    master_html_content += f"""
        <div class="plot-container">
            <h3>{title}</h3>
            {description}
            <iframe src="regional_HS_BEA_mapping/html_log/{html_log_filename}"></iframe>
        </div>
    """

master_html_content += f"""
    </div>
    
    <div id="discrepancies" class="tabcontent">
        {discrepancies_html_table}
    </div>
    
    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].classList.remove("active");
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].classList.remove("active");
            }}
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }}
    </script>
</body>
</html>
"""

# Save the master HTML file
master_html_path = os.path.join(validation_dir, '02_TiVA_vs_HS_Import_Charts.html')
with open(master_html_path, 'w') as f:
    f.write(master_html_content)