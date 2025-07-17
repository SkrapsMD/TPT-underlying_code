import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy import linalg
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
custom_colors = {
    'blue1': '#3581b4',
    'green1': '#7fac1c',
    'yellow1': '#f3bb00',
    'blue2': '#56bfd6',
    'orange1': '#ca590c',
    'teal1': '#53c49f',
    'pink1': '#d34682',
    'purple1': '#4a3e8e',
    'maroon1': '#580d10',
    'blue3': '#006278',
    'green2': '#385100',
    'gray1': '#414141',
}
CORNFLOWERBLUE   = '\033[38;2;100;149;237m'
INDIANRED = '\033[38;2;205;92;92m'
RESET     = '\033[0m'
# Set the color cycle for matplotlib to use your custom palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[custom_colors['blue1'], custom_colors['green1'], custom_colors['yellow1'],
                                                    custom_colors['blue2'], custom_colors['orange1'], custom_colors['teal1'],
                                                    custom_colors['pink1'], custom_colors['purple1'], custom_colors['maroon1'],
                                                    custom_colors['blue3'], custom_colors['green2'], custom_colors['gray1']])

"""
Data gets saved to the data/working/Components for Calculations directory under:
--------------------------------------------------
|-- BEA
|    | -- 71
|    |     |-- 2017
|    |     |-- 2019
|    |     |-- 2022
|-- TiVA
|    | -- 71
|    |     |-- 2017
|    |     |-- 2019
|    |     |-- 2022
|    | -- 138
|    |     |-- 2017
|    |     |-- 2019
|    |     |-- 2022

Note, for the import shares omega_m star, and omega_d star, there are two categories, 336111 and 3342 where the time imputation fails, so we just use the shares from 2017. 
-------------------------------------------------
They will always come with the name column included (U.Summary Code of Summary Code).
m - Number of Industries
n - Number of Commodities
-------------------------------------------------

g - Total Industry Output (Use Table) [1 x m]  *

q - Total Commodity Output (Use Table) [n x 1]  *

c - Total Compensation of Employees (Use Table) [1 x m]

u - Total Inputs from Use Table [1 x m] -- includes Other and Used (0)

U - Intermediate Inputs from Use Table [n x m] -- includes Other and Used  (0)

V - Output from the Make Table [m x n] -- includes Other and Used  (0)

M - Input matrix from the Import Use Table [n x m] -- includes Other and Used (0)

--------------------------------------------------
* - can be switched out for the Make table. 
"""
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)  # Go up one level to Calculations/
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
########################################################################################################################################
# Read in the data as its respective matricies and in the correct formats.... 
########################################################################################################################################
year = 2023



def read_data_as_linalg(year, name, type = 'TiVA', agg = '138', shape = 'column'):
    name_text = name
    name = pd.read_csv(os.path.join(calculations_dir, type, agg, f'{year}' , f'{name}.csv'), index_col=0)
    width = name.shape[1] 
    height = name.shape[0]
    if shape == 'column': 
        if width > height:
            name = name.T
            name = name.values
            print(f'{RESET}The object {INDIANRED} {name_text} {RESET} was ROW, transposed to COLUMN. Original shape: [{height}, {width}] -> {name.shape} {RESET}')
        elif width == height: 
            print(f"{INDIANRED} !!!!! {RESET}The object {INDIANRED} {name_text}{RESET} is Square, can NOT become column. {name.shape} {INDIANRED} !!!!! {RESET}")
        else: 
            name = name.values
            print(f"{RESET}The object {INDIANRED} {name_text}{RESET}  is already a column vector. {name.shape} {RESET}")
    elif shape == 'row':
        if width < height: 
            name = name.T
            name = name.values
            print(f'{RESET}The object {INDIANRED} {name_text} {RESET} was COLUMN, transposed to ROW. Original shape: [{height}, {width}] -> {name.shape} {RESET}')
        elif width == height: 
            print(f"{INDIANRED} !!!!! {RESET}The object {INDIANRED} {name_text}{RESET} is Square, can NOT become ROW. {name.shape} {INDIANRED} !!!!! {RESET}")
        else: 
            name = name.values
            print(f"{RESET}The object {INDIANRED} {name_text}{RESET}  is already a row vector. {name.shape} {RESET}")
    elif shape == 'square':
        if width == height: 
            name = name.values
            print(f"{RESET}The object {INDIANRED} {name_text}{RESET}  is already a square matrix. {name.shape} {RESET}")
        else: 
            print(f"{INDIANRED} !!!!! {RESET}The object {INDIANRED} {name_text}{RESET}  is NOT a square matrix. {name.shape} {INDIANRED} !!!!! {RESET}")
    return name


# Read in the data as its respective matricies and in the correct formats for the incoming linear algebra calculations.
data = {'q':'column', 'g':'column', 'c':'column', 'u':'column' , 'U':'square', 'V':'square', 'M':'square', 'w_d_star':'column', 'w_m_star':'column', 'C':'square', 'impGroups/M_CAN':'square', 'impGroups/M_CHN':'square', 'impGroups/M_EUR':'square','impGroups/M_JAP':'square','impGroups/M_MEX':'square','impGroups/M_RoW':'square','impGroups/M_RoAsia':'square', 'impGroups/w_m_star_can':'column', 'impGroups/w_m_star_chn':'column', 'impGroups/w_m_star_eur':'column', 'impGroups/w_m_star_jap':'column', 'impGroups/w_m_star_mex':'column', 'impGroups/w_m_star_RoAsia':'column', 'impGroups/w_m_star_RoW':'column'}
linalg_objs = {}
for name, shape in data.items():
    if 'impGroups/' in name:
        short_name = name.split('/')[-1]  # Extract the file name after 'impGroups/'
        linalg_objs[short_name] = read_data_as_linalg(year, name, type='TiVA', agg='138', shape=shape)
        globals()[short_name] = linalg_objs[short_name]
    else:
        linalg_objs[name] = read_data_as_linalg(year, name, type='TiVA', agg='138', shape=shape)
        globals()[name] = linalg_objs[name]

num =140# Number of Industries/commodities should be the number of categories + 2 

# create the identity matrix and the summation vector
I = np.eye(num)
summation = np.ones((num, 1))

nipa_summation = np.ones((C.shape[1],1))

# Create the Domestic Use Input Matrix (U_d = U - M) 
U_d = U - M

U_CAN_d = U-M_CAN

# Creation of the different markup assumption vectors: 
CDM = g
CPM = c+u 

# Create and invert diagonal matrices for the column vectors in a loop
matrices = {'g': g, 'q': q, 'c': c, 'u': u, 'CDM':CDM, 'CPM': CPM, 'w_d_star': w_d_star, 'w_m_star': w_m_star, 'w_m_star_can': w_m_star_can, 'w_m_star_chn': w_m_star_chn, 'w_m_star_eur': w_m_star_eur, 'w_m_star_jap': w_m_star_jap, 'w_m_star_mex': w_m_star_mex, 'w_m_star_RoAsia': w_m_star_RoAsia, 'w_m_star_RoW': w_m_star_RoW}
diagonal_matrices = {}
inverted_matrices = {}

for name, vector in matrices.items():
    diagonal_matrices[name] = np.diag(vector.flatten())
    print(name)
    inverted_matrices[f"{name}_inv"] = np.linalg.pinv(diagonal_matrices[name])

g_mat, q_mat, c_mat, u_mat, CDM_mat, CPM_mat, w_d_star_mat, w_m_star_mat = diagonal_matrices['g'], diagonal_matrices['q'], diagonal_matrices['c'], diagonal_matrices['u'], diagonal_matrices['CDM'], diagonal_matrices['CPM'], diagonal_matrices['w_d_star'], diagonal_matrices['w_m_star']
w_m_star_can_mat, w_m_star_chn_mat, w_m_star_eur_mat, w_m_star_jap_mat, w_m_star_mex_mat, w_m_star_RoAsia_mat, w_m_star_RoW_mat = diagonal_matrices['w_m_star_can'], diagonal_matrices['w_m_star_chn'], diagonal_matrices['w_m_star_eur'], diagonal_matrices['w_m_star_jap'], diagonal_matrices['w_m_star_mex'], diagonal_matrices['w_m_star_RoAsia'], diagonal_matrices['w_m_star_RoW']
g_inv, q_inv, c_inv, u_inv, CDM_inv ,CPM_inv = inverted_matrices['g_inv'], inverted_matrices['q_inv'], inverted_matrices['c_inv'], inverted_matrices['u_inv'], inverted_matrices['CDM_inv'], inverted_matrices['CPM_inv']


# Create the B matrix by normalizing the U with the markup assumptions (CDM -> Constant Dollar Markup, CPM -> Constant Percent Markup) -- Board paper uses constand dollar.... 
B = np.dot(U, CDM_inv)
B_d = np.dot(U_d, CDM_inv)
B_m = np.dot(M, CDM_inv)

B_CAN_m = np.dot(M_CAN, CDM_inv) # Constant Dollar Markup for China
B_CHN_m = np.dot(M_CHN, CDM_inv) # Constant Dollar Markup for China
B_EUR_m = np.dot(M_EUR, CDM_inv) # Constant Dollar Markup for Europe
B_JAP_m = np.dot(M_JAP, CDM_inv) # Constant Dollar Markup for Japan
B_MEX_m = np.dot(M_MEX, CDM_inv) # Constant Dollar Markup for Mexico
B_RoAsia_m = np.dot(M_RoAsia, CDM_inv) # Constant Dollar Markup Rest of Asia Pacific
B_RoW_m = np.dot(M_RoW, CDM_inv) # Constant Dollar Markup for Rest of World


B_cpm = np.dot(U, CPM_inv) # Constant Percent Markup
B_d_cpm = np.dot(U_d, CPM_inv) # Constant Percent Markup Domestic Use Inputs
B_m_cpm = np.dot(M, CPM_inv) # Constant Percent Markup Imports


# Create the market share matrix (D) which is D = Vq^{-1} -- This does not depend on markup assumptions. 
D = np.dot(V, q_inv)

# Now produce the different total requirements tables: 
#1.) Commodity by Commodity Total Requirements Table (I - BD)^-1
Comm_TRT = linalg.pinv(I-np.dot(B,D))
#2.) Industry by Industry Total Requirements Table (I-DB)^-1
Ind_TRT = linalg.pinv(I-np.dot(D,B))
#3.) Industry by Commodity Total Requirements Table D(I-BD)^-1
IndCom_TRT = np.dot(D, Comm_TRT)

# 4.) Comm X Comm Domestic TRT
D_Comm_TRT = linalg.pinv(I-np.dot(B_d,D))
# 5.) Ind X Ind Domestic TRT
D_Ind_TRT = linalg.pinv(I-np.dot(D,B_d))
# 6.) Ind X Comm Domestic TRT
D_IndCom_TRT= np.dot(D, D_Comm_TRT)

# Import Total Requirements Table B_m (I-B_d, D) ^ {-1}  As defined by the Boston Fed Paper... I think I have some qualms with this one? 
BD = np.dot(B,D)
B_mD = np.dot(B_m, D) 

B_CAN_mD = np.dot(B_CAN_m, D) # Constant Dollar Markup for Canada
B_CHN_mD = np.dot(B_CHN_m, D) # Constant Dollar Markup for China 
B_EUR_mD = np.dot(B_EUR_m, D) # Constant Dollar Markup for Europe
B_JAP_mD = np.dot(B_JAP_m, D)
B_MEX_mD = np.dot(B_MEX_m, D)
B_RoAsia_mD = np.dot(B_RoAsia_m, D)
B_RoW_mD = np.dot(B_RoW_m, D)


B_dD = np.dot(B_d, D)
Imp_Comm_TRT = linalg.inv(I-np.dot(B_d, D))


pd.DataFrame(Imp_Comm_TRT).to_csv(os.path.join(validations_dir,'04_main_calculations','LeontieffInverse.csv'))

### Now we can try and test our approximation of the boston fed paper. Their complete methodology is as follows: 
"""
P =  tau_d [ w_m + B_mD (I-BdD)^-1 w_d] 

If you want to use the producer prices, or you can do: 

P = tau_m [ w_d_star + B_dD (I-BdD)^-1 w_m_star]C 

where C is the map from the BEA U.Sum level to the NIPA line item and can map to the purchaser prices there

They try to replicate another paper in the end of theirs with a 14% increase in the tariff, which led to 3.6% increase in prices. I want to try that. Tau is the weighted change in tariffs. 
"""

tau_t = np.full((num, 1), 0.0704).T  # China @ 10% if it is 400 billion in trade (equally distributed across products) then this is the tariff rate (0.07024)

#P= np.dot(tau_t, (w_m_mat + np.dot(B_mD, Imp_Comm_TRT @ w_d_mat)))
P_star = np.dot(np.dot(tau_t, (w_m_star_mat + np.dot(B_mD, Imp_Comm_TRT @ w_d_star_mat))), C)
P_star_direct = np.dot(np.dot(tau_t, w_m_star_mat), C)
p_star_indirect = np.dot(np.dot(tau_t, (np.dot(B_mD, Imp_Comm_TRT @ w_d_star_mat))), C)



output = np.dot(P_star, nipa_summation)
direct = np.dot(P_star_direct, nipa_summation)
indirect = np.dot(p_star_indirect, nipa_summation)
print(f"The total impact on inflation would be:{INDIANRED} {output[0][0]*100:.2f}% {RESET} increase in prices at {CORNFLOWERBLUE}PURCHASER PRICES and CONSTANT DOLLAR MARKUP{RESET}")
print(f"The direct impact would be {INDIANRED} {direct[0][0]*100:.2f}% {RESET} and the indirect impact would be {INDIANRED} {indirect[0][0]*100:.2f}% {RESET} increase in prices at {CORNFLOWERBLUE}PURCHASER PRICES and CONSTANT DOLLAR MARKUP{RESET}")












#### This is the correct method of approximating... Notice, the final approximations are all missing the Identity Matrix. This simply needs to be added back in....  
# This is kind of like if we missed out on the direct inputs and only have the indirect inputs...  (i.e. we don't have the inputs from the industry into itself.) ### READ THIS AND CHECK THE .allclose specifications.
max_layers = 12
approximation = np.zeros_like(Imp_Comm_TRT)

# For tracking the convergence visualization
p_indirect_by_iteration = []
all_p_indirect_values = []  # Will store all values for each iteration
layer_contributions = []  # Will store the contribution of each layer

exact_p_indirect = np.dot(np.dot(tau_t, (np.dot(B_mD, Imp_Comm_TRT @ w_d_star_mat))), C)
exact_p_indirect_value = np.dot(exact_p_indirect, nipa_summation)[0][0] * 100  # Convert to percentage

# Get the exact p_indirect vector for all categories
exact_p_indirect_values = exact_p_indirect.flatten() * 100  # Convert to percentage

# Initialize previous values for calculating contributions
previous_p_indirect = np.zeros_like(exact_p_indirect_values)

for i in range(0, max_layers + 1):
    layer = np.linalg.matrix_power(B_dD, i)
    approximation += layer
    
    # Calculate P_indirect using current approximation
    current_approx_with_I = approximation.copy()  # Add identity later if needed
    current_p_indirect = np.dot(np.dot(tau_t, (np.dot(B_mD, current_approx_with_I @ w_d_star_mat))), C)
    current_p_indirect_value = np.dot(current_p_indirect, nipa_summation)[0][0] * 100  # Convert to percentage
    
    # Store individual values for each category at this iteration
    current_p_indirect_values = current_p_indirect.flatten() * 100  # Convert to percentage
    all_p_indirect_values.append(current_p_indirect_values)
    
    # Calculate contribution of just this layer (change from previous iteration)
    layer_contribution = current_p_indirect_values - previous_p_indirect
    layer_contributions.append(layer_contribution)
    previous_p_indirect = current_p_indirect_values.copy()
    
    # Store for summary visualization
    p_indirect_by_iteration.append({
        'iteration': i,
        'p_indirect': current_p_indirect_value,
        'exact_value': exact_p_indirect_value,
        'error': abs(current_p_indirect_value - exact_p_indirect_value)
    })
    
    if i == max_layers:
        approximation += 0    
    if np.allclose(approximation, Imp_Comm_TRT, 0.1):
        print(f"{CORNFLOWERBLUE}Approximation is close to the exact value after {i} layers.{RESET}")
        break

if not np.allclose(approximation, Imp_Comm_TRT, 0.1):
    print(f"{INDIANRED}Approximation did not converge to the exact value within {max_layers} layers.{RESET}")

# Create DataFrame for visualization
convergence_df = pd.DataFrame(p_indirect_by_iteration)
print(convergence_df)

# Create visualization of convergence
plt.figure(figsize=(12, 7))
plt.plot(convergence_df['iteration'], convergence_df['p_indirect'], marker='o', linestyle='-', color=custom_colors['blue1'], label='Approximation')
plt.axhline(y=exact_p_indirect_value, color=custom_colors['orange1'], linestyle='--', label=f'Exact Value ({exact_p_indirect_value:.4f}%)')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('P_indirect (%)', fontsize=14)
plt.title('Convergence of P_indirect with Von Neumann Series Approximation', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'von_neumann_convergence.png'), dpi=300)
plt.close()

# Create visualization of error
plt.figure(figsize=(12, 7))
plt.plot(convergence_df['iteration'], convergence_df['error'], marker='o', linestyle='-', color=custom_colors['pink1'])
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Absolute Error in P_indirect (%)', fontsize=14)
plt.title('Error in P_indirect Approximation by Iteration', fontsize=16)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to better visualize error reduction
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'von_neumann_error.png'), dpi=300)
plt.close()

# Create visualization of the full convergence for all categories
# Convert to numpy array for easier manipulation
all_p_indirect_values_array = np.array(all_p_indirect_values)
n_iterations = all_p_indirect_values_array.shape[0]
n_categories = all_p_indirect_values_array.shape[1]

# Select a subset of categories to visualize (e.g., top 10 by final value)
top_indices = np.argsort(np.abs(all_p_indirect_values_array[-1]))[-10:]

# Create a figure showing convergence for top categories
plt.figure(figsize=(15, 10))

# Plot lines for each of the selected categories
for idx in top_indices:
    category_values = all_p_indirect_values_array[:, idx]
    plt.plot(range(n_iterations), category_values, marker='o', linestyle='-', 
             label=f'Category {idx}')

# Add the exact values as horizontal lines
for idx in top_indices:
    plt.axhline(y=exact_p_indirect_values[idx], linestyle='--', alpha=0.5, 
                color=plt.gca().lines[-1].get_color())

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('P_indirect by Category (%)', fontsize=14)
plt.title('Convergence of Top 10 P_indirect Categories', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'von_neumann_convergence_by_category.png'), dpi=300)
plt.close()

# Create a heatmap visualization showing all categories convergence
plt.figure(figsize=(15, 10))

# Calculate percent error from exact values for each category at each iteration
percent_errors = np.zeros_like(all_p_indirect_values_array)
for i in range(n_iterations):
    percent_errors[i] = 100 * np.abs(all_p_indirect_values_array[i] - exact_p_indirect_values) / (np.abs(exact_p_indirect_values) + 1e-10)  # Add small value to avoid div by zero

# Use log scale for better visualization
log_errors = np.log10(percent_errors + 1e-10)  # Add small value to avoid log(0)

# Create heatmap
im = plt.imshow(log_errors.T, aspect='auto', cmap='viridis_r')
plt.colorbar(im, label='Log10(Percent Error)')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Category Index', fontsize=14)
plt.title('Convergence Heatmap of P_indirect by Category and Iteration', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'von_neumann_convergence_heatmap.png'), dpi=300)
plt.close()

# Convert layer contributions to numpy array
layer_contributions_array = np.array(layer_contributions)

# Create a stacked area chart for all contributions
# Select top categories by absolute total contribution for better visualization
top_indices = np.argsort(np.sum(np.abs(layer_contributions_array), axis=0))[-10:]

# Create the layer contribution visualization
plt.figure(figsize=(15, 10))

# Create stacked bar chart for each iteration
x = np.arange(n_iterations)
bottom = np.zeros(n_iterations)

for idx in top_indices:
    plt.bar(x, layer_contributions_array[:, idx], bottom=bottom, 
            label=f'Category {idx}', alpha=0.7)
    bottom += layer_contributions_array[:, idx]

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Layer Contribution (%)', fontsize=14)
plt.title('Contribution of Each Layer to P_indirect for Top 10 Categories', fontsize=16)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'von_neumann_layer_contributions_stacked.png'), dpi=300)
plt.close()

# Create a heatmap of layer contributions
plt.figure(figsize=(15, 10))
im = plt.imshow(layer_contributions_array.T, aspect='auto', cmap='RdBu_r', 
                norm=plt.cm.colors.SymLogNorm(linthresh=0.01, linscale=1.0))
plt.colorbar(im, label='Layer Contribution (%)')
plt.xlabel('Iteration (Layer)', fontsize=14)
plt.ylabel('Category Index', fontsize=14)
plt.title('Contribution of Each Layer to P_indirect by Category', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'von_neumann_layer_contributions_heatmap.png'), dpi=300)
plt.close()

# Create an area plot showing ALL categories' contributions stacked by iteration
plt.figure(figsize=(20, 12))

# Get total contribution for each layer (sum across all categories)
total_layer_contrib = np.sum(layer_contributions_array, axis=1)

# Create a plot showing the waterfall of all contributions
# First, sort categories by their total absolute contribution
total_category_contrib = np.sum(np.abs(layer_contributions_array), axis=0)
sorted_indices = np.argsort(total_category_contrib)

# Create cumulative waterfall plot
plt.figure(figsize=(20, 10))

# Initialize
x_labels = [f"Layer {i}" for i in range(n_iterations)]
cumulative_values = np.zeros(n_iterations)

# Plot each category as a separate line showing cumulative effect
for idx in sorted_indices:
    # Get this category's contributions across all layers
    category_contrib = layer_contributions_array[:, idx]
    
    # Add to cumulative total
    new_cumulative = cumulative_values + category_contrib
    
    # Plot as a filled area between current cumulative and new cumulative
    plt.fill_between(range(n_iterations), cumulative_values, new_cumulative, alpha=0.7)
    
    # Update cumulative values
    cumulative_values = new_cumulative

# Add line for total
plt.plot(range(n_iterations), cumulative_values, 'k-', linewidth=2, label='Total')

plt.xlabel('Layer', fontsize=14)
plt.ylabel('Cumulative Contribution to P_indirect (%)', fontsize=14)
plt.title('Cumulative Contribution of All Categories by Layer', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xticks(range(n_iterations), x_labels, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'von_neumann_all_categories_cumulative.png'), dpi=300)
plt.close()

# Create a visualization showing the total contribution of each layer
plt.figure(figsize=(15, 8))
plt.bar(range(n_iterations), total_layer_contrib, color=custom_colors['blue1'])
plt.xlabel('Layer', fontsize=14)
plt.ylabel('Total Layer Contribution (%)', fontsize=14)
plt.title('Total Contribution of Each Layer to P_indirect (All Categories)', fontsize=16)
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(range(n_iterations), [f"Layer {i}" for i in range(n_iterations)])
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'von_neumann_total_layer_contributions.png'), dpi=300)
plt.close()

# Create a line chart showing distribution of contributions by layer for the top categories
plt.figure(figsize=(15, 10))
for idx in top_indices:
    plt.plot(x, layer_contributions_array[:, idx], marker='o', label=f'Category {idx}')

plt.xlabel('Iteration (Layer)', fontsize=14)
plt.ylabel('Layer Contribution (%)', fontsize=14)
plt.title('Layer Contribution to P_indirect by Iteration for Top Categories', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'von_neumann_layer_contributions_line.png'), dpi=300)
plt.close()

pd.DataFrame(approximation).to_csv('approx.csv')

print(f"Final Approximation Error: {np.max(np.abs(approximation+I - Imp_Comm_TRT))}")


direct = np.dot(w_m_star_mat, C)
direct_CAN = np.dot(w_m_star_can_mat, C).tolist()
direct_CHN = np.dot(w_m_star_chn_mat, C).tolist()
direct_EUR = np.dot(w_m_star_eur_mat, C).tolist()
direct_JAP = np.dot(w_m_star_jap_mat, C).tolist()
direct_MEX = np.dot(w_m_star_mex_mat, C).tolist()
direct_RoAsia = np.dot(w_m_star_RoAsia_mat, C).tolist()
direct_RoW = np.dot(w_m_star_RoW_mat, C).tolist()
print(np.dot(np.dot(tau_t, direct), nipa_summation)*100)


indirect = np.dot(np.dot(B_mD, Imp_Comm_TRT @ w_d_star_mat), C)
print(np.dot(np.dot(tau_t, indirect), nipa_summation)*100)
indirect_CAN = np.dot(np.dot(B_CAN_mD, Imp_Comm_TRT @ w_d_star_mat), C).tolist()
indirect_CHN = np.dot(np.dot(B_CHN_mD, Imp_Comm_TRT @ w_d_star_mat), C).tolist()
indirect_EUR = np.dot(np.dot(B_EUR_mD, Imp_Comm_TRT @ w_d_star_mat), C).tolist()
indirect_JAP = np.dot(np.dot(B_JAP_mD, Imp_Comm_TRT @ w_d_star_mat), C).tolist()
indirect_MEX = np.dot(np.dot(B_MEX_mD, Imp_Comm_TRT @ w_d_star_mat), C).tolist()
indirect_RoAsia = np.dot(np.dot(B_RoAsia_mD, Imp_Comm_TRT @ w_d_star_mat), C).tolist()
indirect_RoW = np.dot(np.dot(B_RoW_mD, Imp_Comm_TRT @ w_d_star_mat), C).tolist()

total = np.dot(w_m_star_mat + np.dot(B_mD, Imp_Comm_TRT @ w_d_star_mat),C)
print(np.dot(np.dot(tau_t, total), nipa_summation)*100)

# save these to the final data directoy: 
direct_df = pd.DataFrame(direct, index=C.index, columns = C.columns)
indirect_df = pd.DataFrame(indirect, index=C.index, columns = C.columns)
total_df = pd.DataFrame(total, index=C.index, columns = C.columns)
direct_df.to_csv(os.path.join(final_data_dir, 'csv/direct.csv'))
indirect_df.to_csv(os.path.join(final_data_dir, 'csv/indirect.csv'))
total_df.to_csv(os.path.join(final_data_dir, 'csv/total.csv'))

direct_can_df = pd.DataFrame(indirect_CAN, index = C.index, columns = C.columns)
direct_can_df.to_csv(os.path.join(final_data_dir, 'csv/direct_CAN.csv'))





N = direct.shape[0]
E = direct.shape[1]

direct_list = direct.tolist()
indirect_list = indirect.tolist()
direct_matrix_json = {
    'rows':N,
    'columns':E,
    'data':direct_list,
    'data_CAN': direct_CAN,
    'data_CHN': direct_CHN,
    'data_EUR': direct_EUR,
    'data_JAP': direct_JAP,
    'data_MEX': direct_MEX,
    'data_RoAsia': direct_RoAsia,
    'data_RoW': direct_RoW
}
indirect_matrix_json = {
    'rows':N,
    'columns':E,
    'data':indirect_list,
    'data_CAN': indirect_CAN,
    'data_CHN': indirect_CHN,
    'data_EUR': indirect_EUR,
    'data_JAP': indirect_JAP,
    'data_MEX': indirect_MEX,
    'data_RoAsia': indirect_RoAsia,
    'data_RoW': indirect_RoW
}
with open(os.path.join(final_data_dir, 'direct_matrix_2023.json'), 'w') as f:
    import json
    json.dump(direct_matrix_json, f)

with open(os.path.join(final_data_dir, 'indirect_matrix_2023.json'), 'w') as f:
    import json
    json.dump(indirect_matrix_json, f)

