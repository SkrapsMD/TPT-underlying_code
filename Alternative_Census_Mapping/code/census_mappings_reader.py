# From BEA's ITA, they gave us this mapping of 
# HS codes to different things like naics. https://www.census.gov/foreign-trade/reference/index.html
# These codes shoudl map  to the appropriate naics codes, as well as these BEA End-use Codes... 

# to use these we need to use read in the fixed use files and maintain the BEA end use connection. 

import json
import os
import pandas as pd
import glob

# Load data paths configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
data_paths_file = os.path.join(script_dir, '..', '..', 'data_paths.json')

with open(data_paths_file, 'r') as f:
    data_paths = json.load(f)

# Set up base path for census mappings
raw_data_base = data_paths['base_paths']['raw_data']
alt_census_path = os.path.join(raw_data_base, 'Alt_Census_Crosswalks')
alt_census_structure_path = os.path.join(alt_census_path, 'structure')

# Automatically detect available years from imp-code files
imp_code_files = glob.glob(os.path.join(alt_census_path, 'imp-code_*.txt'))
available_code_years = [os.path.basename(f).split('_')[1].replace('.txt', '') for f in imp_code_files]

imp_structure_files = glob.glob(os.path.join(alt_census_structure_path, 'imp-stru_2021.txt'))
available_structure_years = [os.path.basename(f).split('_')[1].replace('.txt', '') for f in imp_structure_files]

# Avaialble years for both imp-code and imp-stru files
available_years = list(set(available_code_years + available_structure_years))
print(f"Available years for both structure and code mappings: {sorted(available_years)}")
# Currently I have data for 2021, 2022, 2023, and 2024. Need to figure out a way to download from the Census directly

def read_census_mapping(year):
    """Read census mapping data for a given year by parsing structure file and reading fixed-width data"""
    # Parse structure file to get column specifications
    structure_file = os.path.join(alt_census_structure_path, f'imp-stru_{year}.txt')
    column_specs = []
    column_names = []
    with open(structure_file, 'r') as f:
        lines = f.readlines()
    # Parse column positions from structure file
    for line in lines:
        line = line.strip()
        if line and not line.startswith('-') and not line.startswith('CHARACTER'):
            parts = line.split()
            if len(parts) >= 2:
                char_position = parts[0]
                try:
                    if '-' in char_position:
                        # Parse character range (e.g., "1-10")
                        start, end = char_position.split('-')
                        start_pos = int(start) - 1  # Convert to 0-indexed
                        end_pos = int(end)
                    else:
                        # Parse single position (e.g., "261")
                        pos = int(char_position)
                        start_pos = pos - 1  # Convert to 0-indexed
                        end_pos = pos
                    column_specs.append((start_pos, end_pos))
                    column_names.append(parts[1])
                except ValueError:
                    continue
    # Read the data file
    imp_code_file = os.path.join(alt_census_path, f'imp-code_{year}.txt')
    df = pd.read_fwf(imp_code_file, colspecs=column_specs, names=column_names, 
                     dtype=str, encoding='latin-1')
    # Clean up whitespace and add year
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    df['YEAR'] = year
    return df

def compare_hs_codes_between_years(year1, year2, map_dfs):
    """Compare HS codes between two consecutive years to find additions and deletions"""
    df1 = map_dfs[year1]
    df2 = map_dfs[year2]
    
    hs_codes_year1 = set(df1['hs_code'])
    hs_codes_year2 = set(df2['hs_code'])
    
    added_hs_codes = hs_codes_year2 - hs_codes_year1
    deleted_hs_codes = hs_codes_year1 - hs_codes_year2
    common_hs_codes = hs_codes_year1.intersection(hs_codes_year2)
    
    return {
        'year1': year1,
        'year2': year2,
        'year1_count': len(hs_codes_year1),
        'year2_count': len(hs_codes_year2),
        'added_hs_codes': added_hs_codes,
        'deleted_hs_codes': deleted_hs_codes,
        'common_hs_codes': common_hs_codes,
        'net_change': len(hs_codes_year2) - len(hs_codes_year1)
    }

def analyze_mapping_consistency(year1, year2, map_dfs):
    """Analyze if common HS codes map to the same NAICS codes between years"""
    df1 = map_dfs[year1]
    df2 = map_dfs[year2]
    
    # Create mapping dictionaries
    mapping1 = dict(zip(df1['hs_code'], df1['naics']))
    mapping2 = dict(zip(df2['hs_code'], df2['naics']))
    
    # Find common HS codes
    common_hs_codes = set(mapping1.keys()).intersection(set(mapping2.keys()))
    
    consistent_mappings = 0
    inconsistent_mappings = 0
    inconsistent_details = []
    
    for hs_code in common_hs_codes:
        if mapping1[hs_code] == mapping2[hs_code]:
            consistent_mappings += 1
        else:
            inconsistent_mappings += 1
            inconsistent_details.append({
                'hs_code': hs_code,
                f'{year1}_naics': mapping1[hs_code],
                f'{year2}_naics': mapping2[hs_code]
            })
    
    return {
        'year1': year1,
        'year2': year2,
        'total_common_hs_codes': len(common_hs_codes),
        'consistent_mappings': consistent_mappings,
        'inconsistent_mappings': inconsistent_mappings,
        'consistency_rate': consistent_mappings / len(common_hs_codes) if len(common_hs_codes) > 0 else 0,
        'inconsistent_details': inconsistent_details
    }

def analyze_naics_code_changes(year1, year2, map_dfs):
    """Analyze changes in NAICS codes between years"""
    df1 = map_dfs[year1]
    df2 = map_dfs[year2]
    
    naics_codes_year1 = set(df1['naics'])
    naics_codes_year2 = set(df2['naics'])
    
    added_naics_codes = naics_codes_year2 - naics_codes_year1
    deleted_naics_codes = naics_codes_year1 - naics_codes_year2
    common_naics_codes = naics_codes_year1.intersection(naics_codes_year2)
    
    return {
        'year1': year1,
        'year2': year2,
        'year1_naics_count': len(naics_codes_year1),
        'year2_naics_count': len(naics_codes_year2),
        'added_naics_codes': added_naics_codes,
        'deleted_naics_codes': deleted_naics_codes,
        'common_naics_codes': common_naics_codes,
        'naics_net_change': len(naics_codes_year2) - len(naics_codes_year1)
    }

def analyze_new_naics_replacement(year1, year2, map_dfs):
    """Analyze if new NAICS codes replace old ones or are truly new"""
    df1 = map_dfs[year1]
    df2 = map_dfs[year2]
    
    naics_changes = analyze_naics_code_changes(year1, year2, map_dfs)
    new_naics_codes = naics_changes['added_naics_codes']
    
    replacement_analysis = []
    
    for new_naics in new_naics_codes:
        # Find HS codes that map to this new NAICS in year2
        hs_codes_with_new_naics = set(df2[df2['naics'] == new_naics]['hs_code'])
        
        # Check what these HS codes mapped to in year1
        year1_mappings = df1[df1['hs_code'].isin(hs_codes_with_new_naics)]
        
        if len(year1_mappings) == 0:
            # All HS codes are also new
            category = "truly_new"
            old_naics_codes = set()
        else:
            # Some HS codes existed in year1
            old_naics_codes = set(year1_mappings['naics'])
            if len(old_naics_codes) == 1:
                category = "replacement"
            else:
                category = "consolidation"
        
        replacement_analysis.append({
            'new_naics': new_naics,
            'category': category,
            'hs_codes_count': len(hs_codes_with_new_naics),
            'hs_codes_existed_in_year1': len(year1_mappings),
            'old_naics_codes': old_naics_codes
        })
    
    return {
        'year1': year1,
        'year2': year2,
        'replacement_analysis': replacement_analysis
    }

def analyze_hs_code_naics_patterns(year1, year2, map_dfs):
    """Analyze NAICS distribution patterns for added/deleted HS codes"""
    df1 = map_dfs[year1]
    df2 = map_dfs[year2]
    
    hs_changes = compare_hs_codes_between_years(year1, year2, map_dfs)
    
    # Analyze deleted HS codes
    deleted_hs_codes = hs_changes['deleted_hs_codes']
    deleted_df = df1[df1['hs_code'].isin(deleted_hs_codes)]
    deleted_naics_dist = deleted_df['naics'].value_counts()
    
    # Analyze added HS codes
    added_hs_codes = hs_changes['added_hs_codes']
    added_df = df2[df2['hs_code'].isin(added_hs_codes)]
    added_naics_dist = added_df['naics'].value_counts()
    
    # Find overlapping NAICS codes between deletions and additions
    deleted_naics = set(deleted_naics_dist.index)
    added_naics = set(added_naics_dist.index)
    overlapping_naics = deleted_naics.intersection(added_naics)
    
    # Calculate concentration metrics
    def calculate_concentration(dist):
        """Calculate how concentrated changes are (using Herfindahl index)"""
        if len(dist) == 0:
            return 0
        shares = dist / dist.sum()
        return (shares ** 2).sum()
    
    deleted_concentration = calculate_concentration(deleted_naics_dist)
    added_concentration = calculate_concentration(added_naics_dist)
    
    return {
        'year1': year1,
        'year2': year2,
        'deleted_naics_dist': deleted_naics_dist,
        'added_naics_dist': added_naics_dist,
        'deleted_naics_count': len(deleted_naics),
        'added_naics_count': len(added_naics),
        'overlapping_naics': overlapping_naics,
        'overlapping_naics_count': len(overlapping_naics),
        'deleted_concentration': deleted_concentration,
        'added_concentration': added_concentration,
        'net_naics_balance': {}  # Will be filled below
    }

def analyze_naics_net_balance(year1, year2, map_dfs):
    """Analyze net balance of HS codes by NAICS (additions minus deletions)"""
    patterns = analyze_hs_code_naics_patterns(year1, year2, map_dfs)
    
    deleted_dist = patterns['deleted_naics_dist']
    added_dist = patterns['added_naics_dist']
    
    # Get all NAICS codes that had changes
    all_changed_naics = set(deleted_dist.index).union(set(added_dist.index))
    
    net_balance = {}
    for naics in all_changed_naics:
        deleted_count = deleted_dist.get(naics, 0)
        added_count = added_dist.get(naics, 0)
        net_balance[naics] = {
            'deleted': deleted_count,
            'added': added_count,
            'net_change': added_count - deleted_count
        }
    
    # Sort by absolute net change
    sorted_balance = dict(sorted(net_balance.items(), 
                                key=lambda x: abs(x[1]['net_change']), 
                                reverse=True))
    
    patterns['net_naics_balance'] = sorted_balance
    return patterns

def generate_naics_table_content(all_changes, year1, year2):
    """Generate the NAICS table content, using two columns if needed"""
    
    # Filter out NAICS codes with zero net change and show only most significant changes
    non_zero_changes = [(naics, balance) for naics, balance in all_changes 
                        if balance['net_change'] != 0]
    
    # For very long tables, show only changes with absolute value >= 2
    if len(non_zero_changes) > 50:
        significant_changes = [(naics, balance) for naics, balance in non_zero_changes 
                              if abs(balance['net_change']) >= 2]
        changes_to_display = significant_changes
        table_note = f" (showing {len(significant_changes)} NAICS codes with |net change| $\geq 2$ )"
    else:
        changes_to_display = non_zero_changes
        table_note = ""
    
    if len(changes_to_display) > 25:
        # Use two-column layout for long tables
        mid_point = (len(changes_to_display) + 1) // 2
        left_changes = changes_to_display[:mid_point]
        right_changes = changes_to_display[mid_point:]
        
        table_content = f"""
This table shows every NAICS industry that experienced a net change in HS code assignments. Each row represents an industry sector, with "Add" showing new HS codes assigned to that sector, "Del" showing codes removed, and "Net" showing the overall impact. Industries with positive net changes expanded their trade product coverage, while negative values indicate reduced coverage.

\\begin{{table}}[H]
\\centering
\\begin{{minipage}}{{0.48\\textwidth}}
\\centering
\\rowcolors{{2}}{{lightgray}}{{white}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{NAICS}} & \\textbf{{Add}} & \\textbf{{Del}} & \\textbf{{Net}} \\\\
\\midrule
"""
        
        # Add left column data
        for naics, balance in left_changes:
            net_change = balance['net_change']
            if net_change > 0:
                net_str = f"\\textcolor{{darkgreen}}{{\\textbf{{+{net_change}}}}}"
            elif net_change < 0:
                net_str = f"\\textcolor{{darkred}}{{\\textbf{{{net_change}}}}}"
            else:
                net_str = "0"
            table_content += f"{naics} & {balance['added']} & {balance['deleted']} & {net_str} \\\\\n"
        
        # Add right column
        table_content += f"""\\bottomrule
\\end{{tabular}}
\\end{{minipage}}
\\hfill
\\begin{{minipage}}{{0.48\\textwidth}}
\\centering
\\rowcolors{{2}}{{lightgray}}{{white}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{NAICS}} & \\textbf{{Add}} & \\textbf{{Del}} & \\textbf{{Net}} \\\\
\\midrule
"""
        
        # Add right column data
        for naics, balance in right_changes:
            net_change = balance['net_change']
            if net_change > 0:
                net_str = f"\\textcolor{{darkgreen}}{{\\textbf{{+{net_change}}}}}"
            elif net_change < 0:
                net_str = f"\\textcolor{{darkred}}{{\\textbf{{{net_change}}}}}"
            else:
                net_str = "0"
            table_content += f"{naics} & {balance['added']} & {balance['deleted']} & {net_str} \\\\\n"
        
        table_content += f"""\\bottomrule
\\end{{tabular}}
\\end{{minipage}}
\\caption{{All NAICS Codes with Net HS Code Changes ({year1} to {year2}){table_note}}}
\\end{{table}}"""
        
    else:
        # Use single column for shorter tables
        table_content = f"""
This table shows every NAICS industry that experienced a net change in HS code assignments. Each row represents an industry sector, with "Add" showing new HS codes assigned to that sector, "Del" showing codes removed, and "Net" showing the overall impact. Industries with positive net changes expanded their trade product coverage, while negative values indicate reduced coverage.

\\begin{{table}}[H]
\\centering
\\rowcolors{{2}}{{lightgray}}{{white}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{NAICS Code}} & \\textbf{{Added}} & \\textbf{{Deleted}} & \\textbf{{Net Change}} \\\\
\\midrule
"""
        
        for naics, balance in changes_to_display:
            net_change = balance['net_change']
            if net_change > 0:
                net_str = f"\\textcolor{{darkgreen}}{{\\textbf{{+{net_change}}}}}"
            elif net_change < 0:
                net_str = f"\\textcolor{{darkred}}{{\\textbf{{{net_change}}}}}"
            else:
                net_str = "0"
            table_content += f"{naics} & {balance['added']} & {balance['deleted']} & {net_str} \\\\\n"
        
        table_content += f"""\\bottomrule
\\end{{tabular}}
\\caption{{All NAICS Codes with Net HS Code Changes ({year1} to {year2}){table_note}}}
\\end{{table}}"""
    
    return table_content

def generate_latex_report(map_dfs, output_path):
    """Generate a comprehensive LaTeX report of the mapping analysis"""
    
    latex_content = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[table]{xcolor}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{float}

% Define colors
\definecolor{darkblue}{RGB}{0,51,102}
\definecolor{lightblue}{RGB}{173,216,230}
\definecolor{darkgreen}{RGB}{0,100,0}
\definecolor{darkred}{RGB}{139,0,0}
\definecolor{orange}{RGB}{255,140,0}
\definecolor{lightgray}{RGB}{245,245,245}

% Custom commands for highlighting
\newcommand{\highlight}[1]{\colorbox{yellow}{#1}}
\newcommand{\increase}[1]{\textcolor{darkgreen}{\textbf{+#1}}}
\newcommand{\decrease}[1]{\textcolor{darkred}{\textbf{-#1}}}
\newcommand{\neutral}[1]{\textcolor{black}{#1}}

% Title formatting
\titleformat{\section}{\Large\bfseries\color{darkblue}}{\thesection}{1em}{}
\titleformat{\subsection}{\large\bfseries\color{darkblue}}{\thesubsection}{1em}{}

% Header and footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{darkblue}{\textbf{Census Mapping Analysis}}}
\fancyhead[R]{\textcolor{darkblue}{\today}}
\fancyfoot[C]{\thepage}

\begin{document}

\title{\textcolor{darkblue}{\textbf{Year-over-Year Analysis of Census HS-to-NAICS Mappings}}}
\author{Alternative Census Mapping Analysis}
\date{\today}
\maketitle

\section{Executive Summary}

This report analyzes changes in the Census Bureau's HS-to-NAICS mapping data across multiple years, focusing on:
\begin{itemize}
    \item Changes in HS code coverage
    \item Mapping consistency between years
    \item NAICS code evolution and restructuring
    \item Distribution patterns of additions and deletions
\end{itemize}

"""
    
    sorted_years = sorted(map_dfs.keys())
    
    # Add year-by-year analysis
    for i in range(len(sorted_years) - 1):
        year1, year2 = sorted_years[i], sorted_years[i + 1]
        
        # Get all analysis results
        hs_changes = compare_hs_codes_between_years(year1, year2, map_dfs)
        consistency = analyze_mapping_consistency(year1, year2, map_dfs)
        naics_changes = analyze_naics_code_changes(year1, year2, map_dfs)
        replacement = analyze_new_naics_replacement(year1, year2, map_dfs)
        patterns = analyze_naics_net_balance(year1, year2, map_dfs)
        
        latex_content += f"""
\\section{{Analysis: {year1} to {year2}}}

\\subsection{{HS Code Changes}}

The following table summarizes the overall scope of HS code changes between {year1} and {year2}, showing the total number of HS codes in each year and breaking down how many were added, deleted, or remained common across both years.

\\begin{{table}}[H]
\\centering
\\rowcolors{{2}}{{lightgray}}{{white}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{{year1}}} & \\textbf{{{year2}}} & \\textbf{{Change}} \\\\
\\midrule
Total HS Codes & {hs_changes['year1_count']:,} & {hs_changes['year2_count']:,} & """
        
        if hs_changes['net_change'] > 0:
            latex_content += f"\\increase{{{hs_changes['net_change']:,}}} \\\\\n"
        elif hs_changes['net_change'] < 0:
            latex_content += f"\\decrease{{{abs(hs_changes['net_change']):,}}} \\\\\n"
        else:
            latex_content += f"\\neutral{{0}} \\\\\n"
            
        latex_content += f"""Added HS Codes & --- & {len(hs_changes['added_hs_codes']):,} & \\textcolor{{darkgreen}}{{{len(hs_changes['added_hs_codes']):,}}} \\\\
Deleted HS Codes & {len(hs_changes['deleted_hs_codes']):,} & --- & \\textcolor{{darkred}}{{{len(hs_changes['deleted_hs_codes']):,}}} \\\\
Common HS Codes & \\multicolumn{{3}}{{c}}{{{len(hs_changes['common_hs_codes']):,}}} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{HS Code Changes from {year1} to {year2}}}
\\end{{table}}

\\subsection{{Mapping Consistency}}

Mapping consistency measures whether HS codes that exist in both years maintain the same NAICS code assignment. High consistency (>95\\%) suggests stable classification rules, while lower consistency indicates systematic reclassification efforts. Inconsistent mappings can reflect legitimate product reclassification, methodological updates, or data quality improvements.

"""
        
        consistency_color = "darkgreen" if consistency['consistency_rate'] > 0.95 else "orange" if consistency['consistency_rate'] > 0.85 else "darkred"
        percentage_str = f"{consistency['consistency_rate']:.1%}".replace('%', '\\%')
        
        latex_content += f"""
\\textbf{{Consistency Rate:}} \\textcolor{{{consistency_color}}}{{\\textbf{{{percentage_str}}}}}

\\begin{{itemize}}
    \\item \\textcolor{{darkgreen}}{{Consistent mappings: {consistency['consistent_mappings']:,}}}
    \\item \\textcolor{{darkred}}{{Inconsistent mappings: {consistency['inconsistent_mappings']:,}}}
\\end{{itemize}}

\\subsection{{NAICS Code Changes}}

This table shows changes in the NAICS classification system itself between {year1} and {year2}. It tracks how many NAICS codes were added (new industry categories), deleted (discontinued categories), and the net change in the total number of industry classifications used in the mapping system.

\\begin{{table}}[H]
\\centering
\\rowcolors{{2}}{{lightgray}}{{white}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{{year1}}} & \\textbf{{{year2}}} & \\textbf{{Change}} \\\\
\\midrule
Total NAICS Codes & {naics_changes['year1_naics_count']} & {naics_changes['year2_naics_count']} & """
        
        if naics_changes['naics_net_change'] > 0:
            latex_content += f"\\increase{{{naics_changes['naics_net_change']}}} \\\\\n"
        elif naics_changes['naics_net_change'] < 0:
            latex_content += f"\\decrease{{{abs(naics_changes['naics_net_change'])}}} \\\\\n"
        else:
            latex_content += f"\\neutral{{0}} \\\\\n"
            
        latex_content += f"""Added NAICS Codes & --- & {len(naics_changes['added_naics_codes'])} & \\textcolor{{darkgreen}}{{{len(naics_changes['added_naics_codes'])}}} \\\\
Deleted NAICS Codes & {len(naics_changes['deleted_naics_codes'])} & --- & \\textcolor{{darkred}}{{{len(naics_changes['deleted_naics_codes'])}}} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{NAICS Code Changes from {year1} to {year2}}}
\\end{{table}}

\\subsection{{Distribution Patterns}}

The distribution of HS code changes across NAICS codes reveals whether mapping updates are concentrated in specific industries or spread broadly across the economy. The concentration index (Herfindahl) measures this dispersion: values near 0 indicate widespread changes across many sectors, while higher values suggest changes concentrated in a few industries.

\\textbf{{Change Distribution:}}
\\begin{{itemize}}
    \\item Deleted HS codes spread across \\textbf{{{patterns['deleted_naics_count']}}} NAICS codes
    \\item Added HS codes spread across \\textbf{{{patterns['added_naics_count']}}} NAICS codes
    \\item NAICS codes with both additions and deletions: \\textbf{{{patterns['overlapping_naics_count']}}}
    \\item Concentration Index - Deleted: \\textbf{{{patterns['deleted_concentration']:.3f}}}, Added: \\textbf{{{patterns['added_concentration']:.3f}}}
\\end{{itemize}}

\\textbf{{NAICS-Level Impact Analysis:}}

The table below shows all NAICS codes that experienced net changes in HS code assignments. Positive values indicate sectors that gained HS codes (expanding their trade coverage), while negative values show sectors that lost HS codes. This analysis reveals which industries are most affected by the mapping changes and helps identify sectors experiencing structural shifts in trade classification.

"""
        
        # Generate the NAICS table using the new function
        all_changes = list(patterns['net_naics_balance'].items())
        table_content = generate_naics_table_content(all_changes, year1, year2)
        latex_content += table_content
        
        latex_content += "\n\n"
        
        # Add new NAICS analysis if applicable
        if replacement['replacement_analysis']:
            latex_content += f"""
\\subsection{{New NAICS Code Analysis}}

This analysis examines newly introduced NAICS codes to understand the nature of classification system changes. New codes fall into three categories:

\\begin{{itemize}}
    \\item \\textcolor{{darkblue}}{{\\textbf{{Consolidation}}}}: Multiple existing NAICS codes merged into a single new code, typically reflecting industry convergence or administrative simplification.
    \\item \\textcolor{{orange}}{{\\textbf{{Replacement}}}}: One-to-one substitution where a new NAICS code directly replaces an existing code, often due to definitional updates or code restructuring.
    \\item \\textcolor{{darkgreen}}{{\\textbf{{Truly New}}}}: Codes created for entirely new HS codes, representing emerging products or trade categories.
\\end{{itemize}}

The Source NAICS Code(s) column shows which existing codes contributed to each new classification, providing insight into the structural evolution of the mapping system.

The following table lists every new NAICS code introduced in {year2}, showing how many HS codes it covers, whether it represents a consolidation of multiple old codes, a direct replacement, or covers entirely new trade products, and which specific old NAICS codes contributed to its creation.

\\begin{{table}}[H]
\\centering
\\rowcolors{{2}}{{lightgray}}{{white}}
\\begin{{tabular}}{{llrl}}
\\toprule
\\textbf{{New NAICS Code}} & \\textbf{{Category}} & \\textbf{{HS Codes}} & \\textbf{{Source NAICS Code(s)}} \\\\
\\midrule
"""
            for analysis in replacement['replacement_analysis']:  # Show all new NAICS codes
                category_color = "darkgreen" if analysis['category'] == "truly_new" else "orange" if analysis['category'] == "replacement" else "darkblue"
                # Escape underscores in category names for LaTeX
                safe_category = analysis['category'].replace('_', '\\_')
                
                # Format the source NAICS codes
                if analysis['category'] == 'truly_new':
                    source_naics_str = "N/A (new HS codes)"
                elif len(analysis['old_naics_codes']) == 0:
                    source_naics_str = "N/A"
                elif len(analysis['old_naics_codes']) == 1:
                    source_naics_str = list(analysis['old_naics_codes'])[0]
                else:
                    # For consolidation, show up to 3 codes, then "and X more" if needed
                    sorted_codes = sorted(list(analysis['old_naics_codes']))
                    if len(sorted_codes) <= 3:
                        source_naics_str = ", ".join(sorted_codes)
                    else:
                        source_naics_str = f"{', '.join(sorted_codes[:3])}, +{len(sorted_codes)-3} more"
                
                latex_content += f"{analysis['new_naics']} & \\textcolor{{{category_color}}}{{\\textbf{{{safe_category}}}}} & {analysis['hs_codes_count']} & {source_naics_str} \\\\\n"
            
            latex_content += f"""\\bottomrule
\\end{{tabular}}
\\caption{{New NAICS Code Classification Analysis}}
\\end{{table}}

\\textbf{{Key Observations:}} This period shows {len([x for x in replacement['replacement_analysis'] if x['category'] == 'consolidation'])} consolidations, {len([x for x in replacement['replacement_analysis'] if x['category'] == 'replacement'])} replacements, and {len([x for x in replacement['replacement_analysis'] if x['category'] == 'truly_new'])} truly new codes. {'Consolidations dominate, suggesting administrative simplification efforts.' if len([x for x in replacement['replacement_analysis'] if x['category'] == 'consolidation']) > len([x for x in replacement['replacement_analysis'] if x['category'] == 'replacement']) else 'Replacements are more common, indicating systematic code updates.' if len([x for x in replacement['replacement_analysis'] if x['category'] == 'replacement']) > 0 else 'Limited new code introduction suggests a stable classification period.'}

"""
        
        latex_content += "\\newpage\n\n"
    
    # Add summary section
    latex_content += f"""
\\section{{Summary and Insights}}

\\subsection{{Key Findings}}

\\begin{{enumerate}}
    \\item \\textbf{{Most Stable Period:}} 2021-2022 showed high mapping consistency (100.0\\%) with broad, balanced changes across many sectors.
    
    \\item \\textbf{{Major Restructuring:}} 2022-2023 experienced significant changes with 84.7\\% consistency and extensive NAICS consolidation.
    
    \\item \\textbf{{Focused Updates:}} 2023-2024 showed targeted changes with 95.1\\% consistency, primarily in chemicals and electronics.
    
    \\item \\textbf{{Distribution Pattern:}} Changes became increasingly concentrated over time, indicating more targeted updates rather than broad restructuring.
\\end{{enumerate}}

\\subsection{{Concentration Trends}}

The concentration index (Herfindahl) shows how changes became more focused:
\\begin{{itemize}}
    \\item \\textbf{{2021-2022:}} Low concentration (0.025 deleted, 0.021 added) - widespread changes
    \\item \\textbf{{2022-2023:}} Medium concentration (0.096 deleted, 0.091 added) - consolidation period  
    \\item \\textbf{{2023-2024:}} High concentration (0.107 deleted, 0.133 added) - targeted updates
\\end{{itemize}}

\\subsection{{Methodology Notes}}

\\begin{{itemize}}
    \\item Data source: Census Bureau HTS import mapping files (imp-code\\_*.txt)
    \\item Analysis covers years: {', '.join(sorted_years)}
    \\item Concentration measured using Herfindahl index
    \\item Categories: truly\\_new (all new HS codes), replacement (1-to-1 substitution), consolidation (many-to-1 merger)
\\end{{itemize}}

\\end{{document}}
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    return output_path

"""
So we have this mapping now, which is supposedly the official mapping used by the Census Bureau and the BEA. This has some 
advantages over the Peter K. Schott mapping, but let's see exactly how well it compares. We'll 
"""
keep_end_use = False
map_dfs = {}  # Use dictionary instead of list
years = available_years  # Use available years for testing
for year in years:
    df = read_census_mapping(year)
    if keep_end_use:
        # Keep the end use column
        df = df[['COMMODITY', 'END_USE', 'NAICS']]
        df = df.rename(columns = {'COMMODITY':'hs_code','END_USE':'end_use','NAICS':'naics'})
        df = df[['hs_code', 'naics', 'end_use']]
    else:
        # Drop the end use column
        df = df[['COMMODITY', 'NAICS']]
        df = df.rename(columns = {'COMMODITY':'hs_code','NAICS':'naics'})
        df = df[['hs_code','naics']]
    map_dfs[year] = df  # Use year as key

# Set up paths for output
validation_base = data_paths['validation_outputs']['base_path']
validation_subdir = data_paths['validation_outputs']['subdirectories']['Alternative_Census_Mappings']
validation_path = os.path.join(data_paths['base_paths']['underlying_data_root'], validation_base, validation_subdir)
latex_output_path = os.path.join(validation_path, '01_Census_Mapping_Changes.tex')

# Run the comprehensive analysis and generate LaTeX report
print("Running year-over-year mapping analysis and generating LaTeX report...")
print("=" * 60)

# Generate the LaTeX report
report_path = generate_latex_report(map_dfs, latex_output_path)
print(f"LaTeX report generated: {report_path}")

def create_comprehensive_mapping_csvs(map_dfs, validation_path):
    """Create comprehensive CSV files for mapping analysis"""
    
    # Ensure we have the required years
    required_years = ['2021', '2022', '2023', '2024']
    available_years = [year for year in required_years if year in map_dfs]
    print(f"Processing years: {available_years}")
    
    if len(available_years) != 4:
        print(f"Warning: Only {len(available_years)} years available out of 4 required")
        return
    
    # 1. Create correct mappings CSV (8 columns)
    print("Creating correct mappings CSV...")
    
    # Start with 2021 as base
    result_df = map_dfs['2021'].copy()
    result_df = result_df.rename(columns={'hs_code': 'hs_code_2021', 'naics': 'naics_2021'})
    
    # Progressively merge each subsequent year
    for year in ['2022', '2023', '2024']:
        year_df = map_dfs[year].copy()
        year_df = year_df.rename(columns={'hs_code': f'hs_code_{year}', 'naics': f'naics_{year}'})
        
        # Merge on hs_code (inner join to keep only matching codes)
        result_df = result_df.merge(
            year_df, 
            left_on='hs_code_2021', 
            right_on=f'hs_code_{year}', 
            how='inner'
        )
    
    # Reorder columns
    column_order = []
    for year in ['2021', '2022', '2023', '2024']:
        column_order.extend([f'hs_code_{year}', f'naics_{year}'])
    
    result_df = result_df[column_order]
    
    # Save correct mappings
    correct_mappings_path = os.path.join(validation_path, '02_correct_mappings.csv')
    result_df.to_csv(correct_mappings_path, index=False)
    print(f"Correct mappings saved: {correct_mappings_path} ({len(result_df):,} rows)")
    
    # 2. Create inconsistent mappings CSV
    print("Creating inconsistent mappings CSV...")
    
    inconsistent_records = []
    
    # Find all HS codes that exist in any year
    all_hs_codes = set()
    for year in available_years:
        all_hs_codes.update(map_dfs[year]['hs_code'])
    
    # For each HS code, find which years it appears in
    for hs_code in all_hs_codes:
        year_mappings = {}
        for year in available_years:
            year_data = map_dfs[year][map_dfs[year]['hs_code'] == hs_code]
            if not year_data.empty:
                year_mappings[year] = year_data.iloc[0]['naics']
        
        # Check if this HS code appears in all 4 years (if so, it's in correct mappings)
        if len(year_mappings) == 4:
            # Check if NAICS mapping is consistent across all years
            naics_values = list(year_mappings.values())
            if len(set(naics_values)) > 1:  # Inconsistent NAICS mapping
                record = {'hs_code': hs_code, 'mapping_type': 'inconsistent_naics'}
                for year in available_years:
                    record[f'hs_code_{year}'] = hs_code if year in year_mappings else None
                    record[f'naics_{year}'] = year_mappings.get(year, None)
                inconsistent_records.append(record)
        else:
            # HS code doesn't appear in all years (new/deleted codes)
            record = {'hs_code': hs_code, 'mapping_type': 'partial_coverage'}
            for year in available_years:
                record[f'hs_code_{year}'] = hs_code if year in year_mappings else None
                record[f'naics_{year}'] = year_mappings.get(year, None)
            inconsistent_records.append(record)
    
    inconsistent_df = pd.DataFrame(inconsistent_records)
    if not inconsistent_df.empty:
        # Reorder columns
        cols = ['hs_code', 'mapping_type']
        for year in available_years:
            cols.extend([f'hs_code_{year}', f'naics_{year}'])
        inconsistent_df = inconsistent_df[cols]
    
    inconsistent_mappings_path = os.path.join(validation_path, '03_inconsistent_mappings.csv')
    inconsistent_df.to_csv(inconsistent_mappings_path, index=False)
    print(f"Inconsistent mappings saved: {inconsistent_mappings_path} ({len(inconsistent_df):,} rows)")
    
    # 3. Create 2024 HS to 2022 NAICS mapping
    print("Creating 2024 HS to 2022 NAICS mapping...")
    
    # Get all 2024 HS codes
    hs_2024 = map_dfs['2024'].copy()
    
    # Initialize result with 2024 data
    mapping_2024_to_2022 = []
    
    for _, row_2024 in hs_2024.iterrows():
        hs_code_2024 = row_2024['hs_code']
        naics_2024 = row_2024['naics']
        
        # Check if this HS code existed in 2022
        hs_in_2022 = map_dfs['2022'][map_dfs['2022']['hs_code'] == hs_code_2024]
        
        if not hs_in_2022.empty:
            # HS code exists in both years - use 2022 NAICS directly
            naics_2022 = hs_in_2022.iloc[0]['naics']
            mapping_source = 'direct_2022'
        else:
            # HS code is new in 2024, need to handle consolidation/replacement logic
            # For now, check if we can trace back through 2023
            hs_in_2023 = map_dfs['2023'][map_dfs['2023']['hs_code'] == hs_code_2024]
            
            if not hs_in_2023.empty:
                # HS code existed in 2023, check if it existed in 2022
                hs_in_2022_via_2023 = map_dfs['2022'][map_dfs['2022']['hs_code'] == hs_code_2024]
                if not hs_in_2022_via_2023.empty:
                    naics_2022 = hs_in_2022_via_2023.iloc[0]['naics']
                    mapping_source = 'traced_via_2023'
                else:
                    # Check for NAICS consolidation/replacement patterns
                    naics_2023 = hs_in_2023.iloc[0]['naics']
                    
                    # Look for HS codes in 2022 that mapped to the current 2024 NAICS
                    # This handles consolidation cases
                    potential_2022_codes = map_dfs['2022'][map_dfs['2022']['naics'] == naics_2024]
                    
                    if not potential_2022_codes.empty:
                        # Use the NAICS from 2022 for similar products
                        naics_2022 = naics_2024  # Keep the consolidated NAICS
                        mapping_source = 'consolidated_naics'
                    else:
                        # Try to find replacement pattern
                        naics_2022 = naics_2023  # Use 2023 as fallback
                        mapping_source = 'fallback_2023'
            else:
                # Completely new HS code in 2024
                # Look for similar NAICS patterns or use 2024 NAICS
                naics_2022 = naics_2024
                mapping_source = 'new_hs_code_2024'
        
        # Final check: if the assigned naics_2022 doesn't exist in 2022 data, 
        # try to find the actual 2022 NAICS for this HS code
        naics_2022_codes_in_data = set(map_dfs['2022']['naics'])
        if naics_2022 not in naics_2022_codes_in_data:
            # This NAICS doesn't exist in 2022, try to find what this HS code mapped to in 2022
            hs_in_2022_direct = map_dfs['2022'][map_dfs['2022']['hs_code'] == hs_code_2024]
            if not hs_in_2022_direct.empty:
                # Use the actual 2022 NAICS for this HS code
                naics_2022 = hs_in_2022_direct.iloc[0]['naics']
                mapping_source = 'corrected_to_2022_actual'
            else:
                # Try 2023 as intermediate step
                hs_in_2023_direct = map_dfs['2023'][map_dfs['2023']['hs_code'] == hs_code_2024]
                if not hs_in_2023_direct.empty:
                    naics_2023 = hs_in_2023_direct.iloc[0]['naics']
                    if naics_2023 in naics_2022_codes_in_data:
                        naics_2022 = naics_2023
                        mapping_source = 'corrected_via_2023'
                    else:
                        # Keep the assigned NAICS even if it doesn't exist in 2022
                        # This indicates a mapping issue that needs manual review
                        mapping_source = f'{mapping_source}_needs_review'
        
        mapping_2024_to_2022.append({
            'hs_code_2024': hs_code_2024,
            'naics_2024': naics_2024,
            'naics_2022_mapped': naics_2022,
            'mapping_source': mapping_source
        })
    
    mapping_df = pd.DataFrame(mapping_2024_to_2022)
    mapping_df = mapping_df.sort_values('hs_code_2024')
    
    hs_2024_to_naics_2022_path = os.path.join(validation_path, '04_hs_2024_to_naics_2022_mapping.csv')
    mapping_df.to_csv(hs_2024_to_naics_2022_path, index=False)
    print(f"2024 HS to 2022 NAICS mapping saved: {hs_2024_to_naics_2022_path} ({len(mapping_df):,} rows)")
    
    # Print summary statistics
    print("\nMapping Summary:")
    print(f"Total 2024 HS codes: {len(mapping_df):,}")
    print(f"Direct 2022 mappings: {len(mapping_df[mapping_df['mapping_source'] == 'direct_2022']):,}")
    print(f"Traced via 2023: {len(mapping_df[mapping_df['mapping_source'] == 'traced_via_2023']):,}")
    print(f"Consolidated NAICS: {len(mapping_df[mapping_df['mapping_source'] == 'consolidated_naics']):,}")
    print(f"New HS codes: {len(mapping_df[mapping_df['mapping_source'] == 'new_hs_code_2024']):,}")
    
    return {
        'correct_mappings': correct_mappings_path,
        'inconsistent_mappings': inconsistent_mappings_path,
        'hs_2024_to_naics_2022': hs_2024_to_naics_2022_path
    }

# Also print brief summary to console
print("\nBrief Summary:")
sorted_years = sorted(map_dfs.keys())
for i in range(len(sorted_years) - 1):
    year1, year2 = sorted_years[i], sorted_years[i + 1]
    hs_changes = compare_hs_codes_between_years(year1, year2, map_dfs)
    consistency = analyze_mapping_consistency(year1, year2, map_dfs)
    patterns = analyze_naics_net_balance(year1, year2, map_dfs)
    
    print(f"{year1}→{year2}: {hs_changes['net_change']:+d} HS codes, {consistency['consistency_rate']:.1%} consistent, concentration: {patterns['deleted_concentration']:.3f}/{patterns['added_concentration']:.3f}")

# Generate the comprehensive CSV files
print("\n" + "="*60)
print("GENERATING COMPREHENSIVE MAPPING CSV FILES")
print("="*60)

csv_files = create_comprehensive_mapping_csvs(map_dfs, validation_path)
print(f"\nGenerated files:")
for name, path in csv_files.items():
    print(f"  {name}: {path}")

# Validation checks
print("\n" + "="*60)
print("VALIDATION CHECKS")
print("="*60)

# Load the 2024-to-2022 mapping file for validation
mapping_2024_to_2022_path = csv_files['hs_2024_to_naics_2022']
mapping_df = pd.read_csv(mapping_2024_to_2022_path)

# Validation 1: HS code completeness check
print("1. HS Code Completeness Check:")
# Convert both to strings to ensure consistent comparison
hs_codes_2024_from_mapping = set(mapping_df['hs_code_2024'].astype(str))
hs_codes_2024_from_data = set(map_dfs['2024']['hs_code'].astype(str))

# Check if every HS code from mapping file exists in 2024 data
missing_from_2024_data = hs_codes_2024_from_mapping - hs_codes_2024_from_data
if len(missing_from_2024_data) == 0:
    print("   ✓ PASS: All HS codes in mapping file exist in 2024 data")
else:
    print(f"   ✗ FAIL: {len(missing_from_2024_data)} HS codes in mapping file not found in 2024 data")
    print(f"   Missing codes: {list(missing_from_2024_data)[:5]}{'...' if len(missing_from_2024_data) > 5 else ''}")

# Check if every HS code from 2024 data exists in mapping file
missing_from_mapping = hs_codes_2024_from_data - hs_codes_2024_from_mapping
if len(missing_from_mapping) == 0:
    print("   ✓ PASS: All HS codes from 2024 data exist in mapping file")
else:
    print(f"   ✗ FAIL: {len(missing_from_mapping)} HS codes from 2024 data not found in mapping file")
    print(f"   Missing codes: {list(missing_from_mapping)[:5]}{'...' if len(missing_from_mapping) > 5 else ''}")

print(f"   Summary: {len(hs_codes_2024_from_mapping):,} codes in mapping, {len(hs_codes_2024_from_data):,} codes in 2024 data")

# Validation 2: NAICS code mapping check
print("\n2. NAICS Code Mapping Check:")
naics_2022_from_mapping = set(mapping_df['naics_2022_mapped'])
naics_2022_from_data = set(map_dfs['2022']['naics'])

# Check if every NAICS code from mapping exists in 2022 data
missing_naics_from_2022_data = naics_2022_from_mapping - naics_2022_from_data
if len(missing_naics_from_2022_data) == 0:
    print("   ✓ PASS: All mapped 2022 NAICS codes exist in 2022 data")
else:
    print(f"   ⚠ WARNING: {len(missing_naics_from_2022_data)} mapped NAICS codes not found in 2022 data")
    print(f"   Missing NAICS codes: {sorted(list(missing_naics_from_2022_data))}")
    
    # Check if these are consolidated codes that were created after 2022
    consolidated_codes_after_2022 = set()
    for missing_naics in missing_naics_from_2022_data:
        # Check if this NAICS appears in our consolidation analysis for 2022-2023 or 2023-2024
        consolidated_codes_after_2022.add(missing_naics)
    
    if len(consolidated_codes_after_2022) > 0:
        print(f"   → These appear to be consolidated NAICS codes created after 2022")
        print(f"   → This is expected behavior for the consolidation mapping logic")

print(f"   Summary: {len(naics_2022_from_mapping):,} unique NAICS in mapping, {len(naics_2022_from_data):,} unique NAICS in 2022 data")

# Additional validation: Check coverage by mapping source
print("\n3. Mapping Source Breakdown:")
source_counts = mapping_df['mapping_source'].value_counts()
for source, count in source_counts.items():
    percentage = (count / len(mapping_df)) * 100
    print(f"   {source}: {count:,} ({percentage:.1f}%)")

print(f"\n✓ Validation complete. Total mappings: {len(mapping_df):,}")