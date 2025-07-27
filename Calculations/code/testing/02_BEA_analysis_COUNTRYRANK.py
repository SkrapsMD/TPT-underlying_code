import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy import linalg
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import country_converter as coco 
# Load data paths and set up standard directory variables
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels to Calculations/
data_paths_file = os.path.join(project_dir, "data_paths.json")
with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)
# Set up clean directory variables using project_root from config
project_root = data_paths['base_paths']['project_root']
raw_data_dir = os.path.join(project_root, data_paths['base_paths']['raw_data'])
working_data_dir = os.path.join(project_root, data_paths['base_paths']['working_data'])
calculations_dir = os.path.join(working_data_dir, "Components for Calculations")
validations_dir = os.path.join(project_root, data_paths['base_paths']['validations'])
hs_to_bea_data_dir = os.path.join(project_root, data_paths['base_paths']['hs_to_bea_data'])
figures_dir = os.path.join(validations_dir,'04_main_calculations')
final_data_dir = os.path.join(project_root, data_paths['base_paths']['final_data'])

# TWT Scenario 
TWT_direct = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/TWT Data/BEA_direct_effects.csv'))
TWT_indirect = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/TWT Data/BEA_indirect_effects.csv'))
TWT_total = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/TWT Data/BEA_total_effects.csv'))

# 10% Scenario
TEN_direct = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/Constant 10%/BEA_direct_effects.csv'))
TEN_indirect = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/Constant 10%/BEA_indirect_effects.csv'))
TEN_total = pd.read_csv(os.path.join(validations_dir,'BEA Results Validations/Constant 10%/BEA_total_effects.csv'))

def create_top20_stacked_bar_chart(direct_df, indirect_df, scenario_name, output_path, result_type="BEA"):
    """
    Create a horizontal stacked bar chart for the top 20 countries by total effect.
    
    Args:
        direct_df: DataFrame with direct effects by country (columns are ISO codes)
        indirect_df: DataFrame with indirect effects by country (columns are ISO codes)
        scenario_name: Name of the scenario (e.g., "TWT Data", "Constant 10%")
        output_path: Full path where to save the HTML file
        result_type: Type of results ("NIPA" or "BEA")
    """
    # Get ISO codes (exclude non-country columns) - use only numeric columns
    exclude_cols = ['BEA_Industry', 'BEA_Code', 'BEA_Description', 'NIPA_Industry', 'NIPA_Code', 'NIPA_Description', 
                   'NIPA Line', 'Description', 'leading_spaces', 'Level_0', 'Level_1', 'Level_2', 'Level_3', 
                   'Level_4', 'Level_5', 'Level_6', 'Level_7', 'Level_8', 'Level_9', 'Index', 'All Countries Effect']
    
    # Select only numeric columns for ISO codes
    numeric_cols = direct_df.select_dtypes(include=[np.number]).columns
    iso_codes = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate totals for each ISO code
    direct_totals = direct_df[iso_codes].sum().to_dict()
    indirect_totals = indirect_df[iso_codes].sum().to_dict()
    
    # Create combined data
    country_data = []
    for iso in iso_codes:
        direct_val = direct_totals.get(iso, 0)
        indirect_val = indirect_totals.get(iso, 0)
        total_val = direct_val + indirect_val
        
        country_data.append({
            'ISO': iso,
            'Direct': direct_val * 100,  # Convert to percentage
            'Indirect': indirect_val * 100,
            'Total': total_val * 100
        })

    
    # Sort by total and get top 20
    country_df = pd.DataFrame(country_data)
    country_df['country_name'] = coco.convert(country_df['ISO'], to='name_short', not_found=None)

    top20 = country_df.nlargest(20, 'Total').sort_values('Total')
    # Create stacked horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Direct Effects',
        y=top20['country_name'],
        x=top20['Direct'],
        orientation='h',
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        name='Indirect Effects',
        y=top20['country_name'],
        x=top20['Indirect'],
        orientation='h',
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        barmode='stack',
        template='plotly_white',
        title=f'Top 20 Countries by Total Effect - {scenario_name} Scenario ({result_type} Results)',
        xaxis_title='Price Effect (%)',
        yaxis_title='Country',
        height=600,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as HTML
    fig.write_html(output_path)
    print(f"Saved {scenario_name} top 20 chart to: {output_path}")
    
    # Save as PNG
    png_path = output_path.replace('.html', '.png')
    fig.write_image(png_path, width=1000, height=600, scale=2)
    print(f"Saved {scenario_name} top 20 chart to: {png_path}")
    
    # Save complete country effects data as CSV
    csv_path = output_path.replace('02_top20.html', '02_country_effects.csv')
    country_df_sorted = country_df.sort_values('Total', ascending=False)
    # Reorder columns for better readability
    country_df_export = country_df_sorted[['ISO', 'country_name', 'Direct', 'Indirect', 'Total']]
    country_df_export.to_csv(csv_path, index=False)
    print(f"Saved {scenario_name} complete country effects data to: {csv_path}")

# Create charts for both scenarios
# TWT Scenario
twt_output_path = os.path.join(validations_dir, 'BEA Results Validations/TWT Data/figures/02_top20.html')
create_top20_stacked_bar_chart(TWT_direct, TWT_indirect, "TWT Data", twt_output_path, "BEA")

# 10% Scenario  
ten_output_path = os.path.join(validations_dir, 'BEA Results Validations/Constant 10%/figures/02_top20.html')
create_top20_stacked_bar_chart(TEN_direct, TEN_indirect, "Constant 10%", ten_output_path, "BEA")