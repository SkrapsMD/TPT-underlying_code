import os
import json
import sys
from typing import Dict, List, Optional, Any

class DataPathManager:
    """
    A centralized path management system for the Underlying Data Construction pipeline.
    
    This class provides easy access to all data paths and files used in the pipeline,
    making it simple for users to replicate the analysis by updating the JSON configuration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the DataPathManager with configuration from JSON file.
        
        Args:
            config_path: Path to the JSON configuration file. If None, uses default location.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_paths.json")
        
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {self.config_path}")
    
    def get_base_path(self, path_key: str) -> str:
        """
        Get a base path from the configuration.
        
        Args:
            path_key: Key for the base path (e.g., 'project_root', 'raw_data', 'working_data')
            
        Returns:
            Full path as string
        """
        if path_key not in self.config['base_paths']:
            raise KeyError(f"Base path '{path_key}' not found in configuration")
        return self.config['base_paths'][path_key]
    
    def get_raw_data_path(self, data_source: str, file_key: Optional[str] = None) -> str:
        """
        Get path to raw data files.
        
        Args:
            data_source: Data source key (e.g., 'hs10', 'hs4')
            file_key: Specific file key within the data source
            
        Returns:
            Full path as string
        """
        if data_source not in self.config['raw_data_sources']:
            raise KeyError(f"Raw data source '{data_source}' not found in configuration")
        
        base_path = self.get_base_path('underlying_data_root')
        source_config = self.config['raw_data_sources'][data_source]
        
        if file_key is None:
            return os.path.join(base_path, source_config['base_path'])
        
        if 'files' in source_config and file_key in source_config['files']:
            return os.path.join(base_path, source_config['base_path'], source_config['files'][file_key])
        
        raise KeyError(f"File key '{file_key}' not found in data source '{data_source}'")
    
    def get_hs4_regional_files(self, region: str, trade_type: str) -> List[str]:
        """
        Get list of HS4 regional trade files.
        
        Args:
            region: Region name (e.g., 'Asia', 'Europe')
            trade_type: 'exports' or 'imports'
            
        Returns:
            List of full file paths
        """
        if 'hs4' not in self.config['raw_data_sources']:
            raise KeyError("HS4 configuration not found")
        
        hs4_config = self.config['raw_data_sources']['hs4']
        
        if region not in hs4_config['regions']:
            raise KeyError(f"Region '{region}' not found in HS4 configuration")
        
        if trade_type not in hs4_config['regions'][region]:
            raise KeyError(f"Trade type '{trade_type}' not found for region '{region}'")
        
        base_path = self.get_base_path('underlying_data_root')
        region_base = os.path.join(base_path, hs4_config['base_path'], region)
        
        files = hs4_config['regions'][region][trade_type]
        return [os.path.join(region_base, filename) for filename in files]
    
    def get_working_data_path(self, output_type: str, file_key: Optional[str] = None) -> str:
        """
        Get path to working data files.
        
        Args:
            output_type: Working data type (e.g., 'schott_concordances')
            file_key: Specific file key within the output type
            
        Returns:
            Full path as string
        """
        if output_type not in self.config['working_data_outputs']:
            raise KeyError(f"Working data output type '{output_type}' not found in configuration")
        
        base_path = self.get_base_path('underlying_data_root')
        output_config = self.config['working_data_outputs'][output_type]
        
        if file_key is None:
            return os.path.join(base_path, output_config['base_path'])
        
        if 'files' in output_config and file_key in output_config['files']:
            return os.path.join(base_path, output_config['base_path'], output_config['files'][file_key])
        
        raise KeyError(f"File key '{file_key}' not found in working data output type '{output_type}'")
    
    def get_final_data_path(self, filename: Optional[str] = None) -> str:
        """
        Get path to final data directory or specific file.
        
        Args:
            filename: Optional filename to append to the path
            
        Returns:
            Full path as string
        """
        base_path = self.get_base_path('underlying_data_root')
        final_path = os.path.join(base_path, self.config['final_data_outputs']['base_path'])
        
        if filename:
            return os.path.join(final_path, filename)
        return final_path
    
    def get_validation_path(self, subdirectory: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        Get path to validation directory or specific file.
        
        Args:
            subdirectory: Optional subdirectory within validations
            filename: Optional filename to append to the path
            
        Returns:
            Full path as string
        """
        base_path = self.get_base_path('underlying_data_root')
        validation_path = os.path.join(base_path, self.config['validation_outputs']['base_path'])
        
        if subdirectory:
            if 'subdirectories' in self.config['validation_outputs'] and subdirectory in self.config['validation_outputs']['subdirectories']:
                validation_path = os.path.join(validation_path, self.config['validation_outputs']['subdirectories'][subdirectory])
            else:
                validation_path = os.path.join(validation_path, subdirectory)
        
        if filename:
            return os.path.join(validation_path, filename)
        return validation_path
    
    def ensure_directory_exists(self, path: str) -> None:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Directory path to create
        """
        os.makedirs(path, exist_ok=True)
    
    def list_available_regions(self) -> List[str]:
        """Get list of available regions for HS4 data."""
        return list(self.config['raw_data_sources']['hs4']['regions'].keys())
    
    def list_available_data_sources(self) -> List[str]:
        """Get list of available raw data sources."""
        return list(self.config['raw_data_sources'].keys())
    
    def print_configuration_summary(self) -> None:
        """Print a summary of the current configuration."""
        print("=== Data Path Configuration Summary ===")
        print(f"Configuration file: {self.config_path}")
        print(f"Project root: {self.get_base_path('project_root')}")
        print(f"Underlying data root: {self.get_base_path('underlying_data_root')}")
        print()
        
        print("Available raw data sources:")
        for source in self.list_available_data_sources():
            print(f"  - {source}")
        print()
        
        print("Available HS4 regions:")
        for region in self.list_available_regions():
            print(f"  - {region}")
        print()


# Global instance for easy access across modules
_path_manager = None

def get_path_manager() -> DataPathManager:
    """
    Get the global DataPathManager instance.
    
    Returns:
        DataPathManager instance
    """
    global _path_manager
    if _path_manager is None:
        _path_manager = DataPathManager()
    return _path_manager

def get_data_path(path_type: str, *args, **kwargs) -> str:
    """
    Convenience function to get data paths.
    
    Args:
        path_type: Type of path ('base', 'raw', 'working', 'final', 'validation')
        *args: Arguments to pass to the specific path method
        **kwargs: Keyword arguments to pass to the specific path method
        
    Returns:
        Full path as string
    """
    pm = get_path_manager()
    
    if path_type == 'base':
        return pm.get_base_path(*args, **kwargs)
    elif path_type == 'raw':
        return pm.get_raw_data_path(*args, **kwargs)
    elif path_type == 'working':
        return pm.get_working_data_path(*args, **kwargs)
    elif path_type == 'final':
        return pm.get_final_data_path(*args, **kwargs)
    elif path_type == 'validation':
        return pm.get_validation_path(*args, **kwargs)
    else:
        raise ValueError(f"Unknown path type: {path_type}")


if __name__ == "__main__":
    pm = get_path_manager()
    pm.print_configuration_summary()
    
    print("=== Example Usage ===")
    print(f"HS10 exports file: {pm.get_raw_data_path('hs10', 'exports')}")
    print(f"Asia HS4 export files: {pm.get_hs4_regional_files('Asia', 'exports')}")
    print(f"Schott concordances directory: {pm.get_working_data_path('schott_concordances')}")
    print(f"Final data directory: {pm.get_final_data_path()}")