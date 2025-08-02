import os
import pandas as pd
import numpy as np

def flatten_pce_diagonal_matrix():
    """
    Flattens the diagonal PCE matrix from C_BEA.csv and adds percentile rankings.
    Also merges with BEA hierarchy data to get commodity descriptions.
    
    The input file is a diagonal matrix where each row represents a BEA industry code
    and the diagonal values represent the PCE share for that industry.
    """
    
    # Read the diagonal matrix
    file_path = "Calculations/data/working/Components for Calculations/TiVA/138/2023/C_BEA.csv"
    df = pd.read_csv(file_path, index_col=0)
    
    # Extract diagonal values (PCE shares)
    diagonal_values = np.diag(df.values)
    industry_codes = df.index.tolist()
    
    # Create flattened dataframe
    flattened_df = pd.DataFrame({
        'usummary_code': industry_codes,
        'PCE_share': diagonal_values
    })
    
    # Calculate percentiles for each PCE_share value
    flattened_df['PCE_percentile'] = flattened_df['PCE_share'].rank(pct=True) * 100
    
    # Read BEA hierarchy to get descriptions
    bea_hierarchy_path = "HS to BEA Data/data/working/02_HS_to_Naics_to_BEA/02_BEA_hierarchy.csv"
    bea_hierarchy = pd.read_csv(bea_hierarchy_path)
    
    # Create mapping from U.Summary codes to undersum title descriptions
    bea_mapping = bea_hierarchy[['U.Summary', 'undersum title']].drop_duplicates()
    bea_mapping = bea_mapping.rename(columns={'U.Summary': 'usummary_code', 'undersum title': 'usummary_desc'})
    
    # Merge with BEA descriptions
    flattened_df = pd.merge(flattened_df, bea_mapping, on='usummary_code', how='left')
    
    # Sort by PCE_share in descending order
    flattened_df = flattened_df.sort_values('PCE_share', ascending=False).reset_index(drop=True)
    
    # Reorder columns
    flattened_df = flattened_df[['usummary_code', 'usummary_desc', 'PCE_share', 'PCE_percentile']]
    
    # Create output directory
    output_dir = "Calculations/validations/07_PCE_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the flattened data
    output_path = os.path.join(output_dir, "PCE_shares_flattened.csv")
    flattened_df.to_csv(output_path, index=False)
    
    print(f"Flattened PCE data saved to: {output_path}")
    print(f"Shape: {flattened_df.shape}")
    print("\nTop 10 industries by PCE share:")
    print(flattened_df.head(10))
    
    return flattened_df

if __name__ == "__main__":
    pce_data = flatten_pce_diagonal_matrix()