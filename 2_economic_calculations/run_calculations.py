import os
import subprocess
import sys
from datetime import datetime

"""
Economic Calculations Pipeline Runner

This script runs the complete economic calculations pipeline that computes
tariff price impacts using trade weights from the data construction pipeline.

The pipeline processes trade data through multiple calculation stages:
1. PCE data loading and processing
2. Input-Output data loading  
3. Input/output validation
4. Main tariff impact calculations
5. Various analytical validations and comparisons

All outputs are saved to data/final/ and data/working/ folders.
"""

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'      # Success
    RED = '\033[91m'        # Error/Failed  
    BLUE = '\033[94m'       # General notices
    YELLOW = '\033[93m'     # Warnings
    BOLD = '\033[1m'
    END = '\033[0m'

def colored_print(message, color=Colors.BLUE):
    """Print message with color formatting"""
    print(f"{color}{message}{Colors.END}")

def run_script(script_path, description, required=True):
    """Run a Python script and handle errors."""
    script_name = os.path.basename(script_path)
    
    print(f"\n{'='*60}")
    colored_print(f"RUNNING: {script_name}", Colors.BLUE)
    colored_print(f"DESCRIPTION: {description}", Colors.BLUE)
    colored_print(f"REQUIRED: {'Yes' if required else 'No (Optional)'}", Colors.YELLOW if not required else Colors.BLUE)
    print(f"{'='*60}")
    
    try:
        # Run the script and capture output
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            colored_print(f"SUCCESS: {script_name} completed successfully", Colors.GREEN)
            if result.stdout:
                colored_print("OUTPUT:", Colors.BLUE)
                print(result.stdout)
        else:
            colored_print(f"ERROR: {script_name} failed with return code {result.returncode}", Colors.RED)
            if result.stderr:
                colored_print("ERROR OUTPUT:", Colors.RED)
                print(result.stderr)
            if result.stdout:
                colored_print("STDOUT:", Colors.BLUE)
                print(result.stdout)
            
            if required:
                return False
            else:
                colored_print(f"CONTINUING: {script_name} failed but is optional", Colors.YELLOW)
                
    except Exception as e:
        colored_print(f"EXCEPTION: Failed to run {script_name}: {str(e)}", Colors.RED)
        if required:
            return False
        else:
            colored_print(f"CONTINUING: {script_name} failed but is optional", Colors.YELLOW)
    
    return True

def main():
    """Run the complete economic calculations pipeline."""
    
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    colored_print("STARTING ECONOMIC CALCULATIONS PIPELINE", Colors.BLUE)
    colored_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.BLUE)
    colored_print(f"Working directory: {os.getcwd()}", Colors.BLUE)
    
    # Core calculation steps (required)
    core_steps = [
        ("code/core/01_read_pce_data.py", 
         "Loads and processes Personal Consumption Expenditure (PCE) data", True),
        
        ("code/core/02_read_io_data.py", 
         "Loads and processes Input-Output (IO) table data for inter-industry relationships", True),
        
        ("code/core/03_validate_inputs_outputs.py", 
         "Validates data consistency between inputs and outputs", True),
        
        ("code/core/04_main_tariff_calculations.py", 
         "Computes main tariff impact calculations including direct and indirect effects", True),
    ]
    
    # Analysis steps (optional - for validation and additional insights)
    analysis_steps = [
        ("code/analysis/01_bea_calculations.py", 
         "Alternative calculation method using BEA framework", False),
        
        ("code/analysis/01_nipa_calculations.py", 
         "Alternative calculation method using NIPA framework", False),
        
        ("code/analysis/02_bea_country_ranking_analysis.py", 
         "Ranks countries by economic impact magnitude (BEA method)", False),
        
        ("code/analysis/02_nipa_country_ranking_analysis.py", 
         "Ranks countries by economic impact magnitude (NIPA method)", False),
        
        ("code/analysis/03_bea_product_ranking_analysis.py", 
         "Ranks products/sectors by economic impact (BEA method)", False),
        
        ("code/analysis/03_nipa_product_ranking_analysis.py", 
         "Ranks products/sectors by economic impact (NIPA method)", False),
        
        ("code/analysis/04_import_effect_comparison.py", 
         "Compares import rankings with economic effect rankings", False),
        
        ("code/analysis/05_cosine_similarity_analysis.py", 
         "Analyzes similarity between import patterns and economic effects", False),
        
        ("code/analysis/06_within_country_analysis.py", 
         "Analyzes effects within individual countries", False),
        
        ("code/analysis/07_pce_flattener.py", 
         "Creates flattened PCE data for analysis", False),
        
        ("code/analysis/08_weighted_correlation_analysis.py", 
         "Computes weighted correlations and Spearman rank statistics", False),
    ]
    
    # Track success/failure
    successful_steps = []
    failed_steps = []
    
    # Run core calculation steps
    colored_print("RUNNING CORE CALCULATIONS (REQUIRED)", Colors.BOLD)
    for script_path, description, required in core_steps:
        success = run_script(script_path, description, required)
        
        if success:
            successful_steps.append(os.path.basename(script_path))
        else:
            failed_steps.append(os.path.basename(script_path))
            colored_print(f"Core pipeline stopped at {os.path.basename(script_path)} due to error", Colors.RED)
            break
    
    # Run analysis steps if core succeeded
    if len(failed_steps) == 0:
        colored_print("RUNNING ANALYSIS SCRIPTS (OPTIONAL)", Colors.BOLD)
        for script_path, description, required in analysis_steps:
            success = run_script(script_path, description, required)
            
            if success:
                successful_steps.append(os.path.basename(script_path))
            else:
                failed_steps.append(os.path.basename(script_path))
                # Continue with other analysis scripts even if one fails
    
    # Final summary
    print(f"\n{'='*60}")
    colored_print("PIPELINE SUMMARY", Colors.BLUE)
    print(f"{'='*60}")
    colored_print(f"Successful steps: {len(successful_steps)}", Colors.GREEN)
    for step in successful_steps:
        colored_print(f"   - {step}", Colors.GREEN)
    
    if failed_steps:
        colored_print(f"Failed steps: {len(failed_steps)}", Colors.RED)
        for step in failed_steps:
            colored_print(f"   - {step}", Colors.RED)
    
    colored_print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.BLUE)
    
    core_success = len([s for s in failed_steps if s in [os.path.basename(step[0]) for step in core_steps]]) == 0
    
    if core_success:
        colored_print("CORE CALCULATIONS COMPLETED SUCCESSFULLY!", Colors.GREEN)
        colored_print("Check the following directories for outputs:", Colors.BLUE)
        colored_print("   - data/final/: Final economic impact matrices and calculations", Colors.BLUE)
        colored_print("   - data/working/: Intermediate calculation files", Colors.BLUE)
        if len(successful_steps) > len(core_steps):
            colored_print("   - Analysis outputs: Various ranking and validation results", Colors.BLUE)
        return True
    else:
        colored_print("CORE CALCULATIONS FAILED - Check error messages above", Colors.RED)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)