#!/usr/bin/env python3
"""
Master Run Script for TPT Underlying Code Replication Package

This script runs the complete pipeline for Trade Price Pulse (TPT) analysis:
1. Data Construction: HS-to-BEA mapping and trade weights
2. Economic Calculations: Tariff impact analysis

Run this script from the repository root directory.
"""

import os
import subprocess
import sys
from datetime import datetime

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'      # Success
    RED = '\033[91m'        # Error/Failed  
    BLUE = '\033[94m'       # General notices
    YELLOW = '\033[93m'     # Warnings
    BOLD = '\033[1m'
    CYAN = '\033[96m'       # Headers
    END = '\033[0m'

def colored_print(message, color=Colors.BLUE):
    """Print message with color formatting"""
    print(f"{color}{message}{Colors.END}")

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    colored_print(f"{title}", Colors.CYAN + Colors.BOLD)
    print(f"{'='*80}")

def run_component(component_dir, script_name, description):
    """Run a component pipeline script."""
    print_header(f"RUNNING: {description}")
    
    # Change to component directory
    original_dir = os.getcwd()
    component_path = os.path.join(original_dir, component_dir)
    
    if not os.path.exists(component_path):
        colored_print(f"ERROR: Component directory {component_dir} not found", Colors.RED)
        return False
    
    script_path = os.path.join(component_path, script_name)
    if not os.path.exists(script_path):
        colored_print(f"ERROR: Script {script_name} not found in {component_dir}", Colors.RED)
        return False
    
    os.chdir(component_path)
    colored_print(f"Working directory: {os.getcwd()}", Colors.BLUE)
    colored_print(f"Running: {script_name}", Colors.BLUE)
    
    try:
        # Run the script and capture output
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            colored_print(f"SUCCESS: {description} completed successfully", Colors.GREEN)
            # Show last few lines of output for progress indication
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 10:
                    colored_print("Last 10 lines of output:", Colors.BLUE)
                    for line in lines[-10:]:
                        print(f"  {line}")
                else:
                    print(result.stdout)
        else:
            colored_print(f"ERROR: {description} failed with return code {result.returncode}", Colors.RED)
            if result.stderr:
                colored_print("ERROR OUTPUT:", Colors.RED)
                print(result.stderr)
            if result.stdout:
                colored_print("STDOUT:", Colors.BLUE)
                print(result.stdout)
            return False
            
    except Exception as e:
        colored_print(f"EXCEPTION: Failed to run {description}: {str(e)}", Colors.RED)
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    return True

def main():
    """Run the complete TPT analysis pipeline."""
    
    print_header("TRADE PRICE PULSE (TPT) - COMPLETE REPLICATION PIPELINE")
    colored_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.BLUE)
    colored_print(f"Repository root: {os.getcwd()}", Colors.BLUE)
    
    # Verify we're in the right directory
    if not os.path.exists("1_data_construction") or not os.path.exists("2_economic_calculations"):
        colored_print("ERROR: This script must be run from the repository root directory", Colors.RED)
        colored_print("Expected directories: 1_data_construction/, 2_economic_calculations/", Colors.RED)
        return False
    
    # Pipeline components
    components = [
        {
            "dir": "1_data_construction",
            "script": "run_pipeline.py", 
            "description": "Data Construction Pipeline - HS-to-BEA mapping and trade weights",
            "required": True
        },
        {
            "dir": "2_economic_calculations", 
            "script": "run_calculations.py",
            "description": "Economic Calculations Pipeline - Tariff impact analysis", 
            "required": True
        }
    ]
    
    # Track success/failure
    successful_components = []
    failed_components = []
    
    # Run each component
    for component in components:
        success = run_component(
            component["dir"], 
            component["script"], 
            component["description"]
        )
        
        if success:
            successful_components.append(component["description"])
        else:
            failed_components.append(component["description"])
            if component["required"]:
                colored_print(f"Pipeline stopped due to failure in required component", Colors.RED)
                break
    
    # Final summary
    print_header("PIPELINE SUMMARY")
    colored_print(f"Successful components: {len(successful_components)}", Colors.GREEN)
    for component in successful_components:
        colored_print(f"   ✓ {component}", Colors.GREEN)
    
    if failed_components:
        colored_print(f"Failed components: {len(failed_components)}", Colors.RED)
        for component in failed_components:
            colored_print(f"   ✗ {component}", Colors.RED)
    
    colored_print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.BLUE)
    
    if len(successful_components) == len(components):
        print_header("PIPELINE COMPLETED SUCCESSFULLY!")
        colored_print("Final outputs are available in:", Colors.GREEN)
        colored_print("   📊 3_final_outputs/trade_weights/ - Trade weights and mappings", Colors.GREEN)
        colored_print("   📈 3_final_outputs/import_calculations/ - Economic impact results", Colors.GREEN) 
        colored_print("   📋 3_final_outputs/validation_reports/ - Quality validation", Colors.GREEN)
        colored_print("   📁 3_final_outputs/figures/ - Key visualizations", Colors.GREEN)
        colored_print("\n🎉 Ready for analysis and publication!", Colors.GREEN + Colors.BOLD)
        return True
    else:
        colored_print(f"PIPELINE INCOMPLETE: {len(failed_components)} component(s) failed", Colors.RED)
        colored_print("Check the error messages above for troubleshooting guidance", Colors.YELLOW)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)