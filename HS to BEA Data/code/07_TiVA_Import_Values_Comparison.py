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
        discrepancies_list.append({
            'region': region_key,
            'usummary_code': row['usummary_code'],
            'usummary_name': row['usummary_name'],
            'HS_total_imports': row['HS_total_imports'],
            'TiVA_total_imports': row['TiVA_total_imports'],
            'difference': row['difference'],
            'pct_difference': row['pct_difference']
        })

discrepancies_df = pd.DataFrame(discrepancies_list)
discrepancies_df = discrepancies_df.sort_values('pct_difference', ascending=False)

# Save discrepancies table
discrepancies_path = os.path.join(validation_dir, '03_large_discrepancies_table.csv')
discrepancies_df.to_csv(discrepancies_path, index=False)

# Create discrepancies HTML table
discrepancies_html_table = ""
if len(discrepancies_df) > 0:
    discrepancies_html_table = f"""
    <h2>BEA Codes with >30% Difference from TiVA</h2>
    <table border="1" style="border-collapse: collapse; width: 100%; color: #ffffff;">
        <tr style="background-color: #333333;">
            <th>Region</th>
            <th>BEA Code</th>
            <th>BEA Name</th>
            <th>HS Total Imports</th>
            <th>TiVA Total Imports</th>
            <th>Difference</th>
            <th>% Difference</th>
        </tr>
    """
    
    for _, row in discrepancies_df.iterrows():
        discrepancies_html_table += f"""
        <tr>
            <td>{row['region']}</td>
            <td>{row['usummary_code']}</td>
            <td>{row['usummary_name']}</td>
            <td>${row['HS_total_imports']:,.0f}</td>
            <td>${row['TiVA_total_imports']:,.0f}</td>
            <td>${row['difference']:,.0f}</td>
            <td>{row['pct_difference']:.1f}%</td>
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
        <button class="tablinks active" onclick="openTab(event, 'regular-plots')">Regular Scale</button>
        <button class="tablinks" onclick="openTab(event, 'log-plots')">Log Scale</button>
        <button class="tablinks" onclick="openTab(event, 'discrepancies')">Large Discrepancies</button>
    </div>
    
    <div id="regular-plots" class="tabcontent active">
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