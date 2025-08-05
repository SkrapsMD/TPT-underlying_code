import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy import linalg
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

def create_top25_product_stacked_bar_chart(direct_df, indirect_df, scenario_name, output_path, result_type="BEA"):
    """
    Create a horizontal stacked bar chart for the top 25 products by total effect using 'All Countries Effect' column.
    
    Args:
        direct_df: DataFrame with direct effects by product
        indirect_df: DataFrame with indirect effects by product
        scenario_name: Name of the scenario (e.g., "TWT Data", "Constant 10%")
        output_path: Full path where to save the HTML file
        result_type: Type of results ("NIPA" or "BEA")
    """
    # Extract product information and effects
    product_data = []
    
    for i, row in direct_df.iterrows():
        # Get product identifiers based on result type
        if result_type == "BEA":
            product_code = row['BEA_Code']
            product_name = row['BEA_Description']
        else:  # NIPA
            product_code = row['NIPA Line']
            product_name = row['Description']
        
        # Get effects from 'All Countries Effect' column
        direct_val = row['All Countries Effect']
        indirect_val = indirect_df.iloc[i]['All Countries Effect']
        total_val = direct_val + indirect_val
        
        product_data.append({
            'Product_Code': product_code,
            'Product_Name': product_name,
            'Direct': direct_val * 100,  # Convert to percentage
            'Indirect': indirect_val * 100,
            'Total': total_val * 100
        })
    
    # Create DataFrame and sort by total effect
    product_df = pd.DataFrame(product_data)
    
    # Get top 25 products
    top25 = product_df.nlargest(25, 'Total').sort_values('Total')
    
    # Create product labels that include both code and name (truncated for readability)
    if result_type == "BEA":
        top25['Product_Label'] = top25.apply(lambda x: f"{x['Product_Code']}: {x['Product_Name'][:40]}{'...' if len(x['Product_Name']) > 40 else ''}", axis=1)
    else:  # NIPA
        top25['Product_Label'] = top25.apply(lambda x: f"{x['Product_Code']}: {x['Product_Name'][:40]}{'...' if len(x['Product_Name']) > 40 else ''}", axis=1)
    
    # Create stacked horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Direct Effects',
        y=top25['Product_Label'],
        x=top25['Direct'],
        orientation='h',
        marker_color='#3581b4'
    ))
    
    fig.add_trace(go.Bar(
        name='Indirect Effects',
        y=top25['Product_Label'],
        x=top25['Indirect'],
        orientation='h',
        marker_color='#ca590c'
    ))
    
    fig.update_layout(
        barmode='stack',
        template='plotly_white',
        #title=f'Top 25 Products by Total Effect - {scenario_name} Scenario ({result_type} Results)',
        xaxis_title='Price Effect (%)',
        yaxis_title='Product',
        height=800,  # Taller for 25 products
        width=1200,  # Wider for longer product names
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=20, color='black')
        ),
        margin=dict(l=400),  # More left margin for product names
        font=dict(color='black', size=22),
            xaxis=dict(
                title_font=dict(color='black', size=20),
                tickfont=dict(color='black', size=14),
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title_font=dict(color='black', size=20),
                tickfont=dict(color='black', size=14),
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
    )
    
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as HTML
    fig.write_html(output_path)
    print(f"Saved {scenario_name} top 25 product chart to: {output_path}")
    
    # Save as PNG
    png_path = output_path.replace('.html', '.png')
    fig.write_image(png_path, width=1200, height=800, scale=2)
    print(f"Saved {scenario_name} top 25 product chart to: {png_path}")
    
    # Save complete product effects data as CSV
    csv_path = output_path.replace('03_top25_products.html', '03_product_effects.csv')
    product_df_sorted = product_df.sort_values('Total', ascending=False)
    # Reorder columns for better readability
    if result_type == "BEA":
        product_df_export = product_df_sorted[['Product_Code', 'Product_Name', 'Direct', 'Indirect', 'Total']]  
        product_df_export = product_df_export.rename(columns={'Product_Code': 'BEA_Code', 'Product_Name': 'BEA_Description'})
    else:  # NIPA  
        product_df_export = product_df_sorted[['Product_Code', 'Product_Name', 'Direct', 'Indirect', 'Total']]
        product_df_export = product_df_export.rename(columns={'Product_Code': 'NIPA_Line', 'Product_Name': 'Description'})
    
    product_df_export.to_csv(csv_path, index=False)
    print(f"Saved {scenario_name} complete product effects data to: {csv_path}")

# Create charts for both scenarios
# TWT Scenario
twt_output_path = os.path.join(validations_dir, 'BEA Results Validations/TWT Data/figures/03_top25_products.html')
create_top25_product_stacked_bar_chart(TWT_direct, TWT_indirect, "TWT Data", twt_output_path, "BEA")

# 10% Scenario  
ten_output_path = os.path.join(validations_dir, 'BEA Results Validations/Constant 10%/figures/03_top25_products.html')
create_top25_product_stacked_bar_chart(TEN_direct, TEN_indirect, "Constant 10%", ten_output_path, "BEA")