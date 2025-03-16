import os
from typing import List, Dict
import yaml
from pyprojroot import here


def create_directory(directory_path: str) -> None:
    """
    Create a directory if it does not exist.

    Parameters:
        directory_path (str): The path of the directory to be created.

    Example:
    ```python
    create_directory("/path/to/new/directory")
    ```

    """
    if not os.path.exists(here(directory_path)):
        os.makedirs(here(directory_path))


def load_config(config_path: str = "configs/custom_config.yml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    with open(here(config_path)) as f:
        return yaml.safe_load(f)


def get_available_collections() -> List[str]:
    """
    Get list of available document collections.
    
    Returns:
        List[str]: List of collection names
    """
    config = load_config()
    return [col['name'] for col in config['vector_db_config']['collections']]


def get_available_databases() -> Dict[str, List[str]]:
    """
    Get list of available databases by type.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping database types to list of database names
    """
    config = load_config()
    databases = {}
    
    for db_type in ['sqlite', 'postgres', 'mysql']:
        if db_type in config['sql_db_config']:
            databases[db_type] = [
                db['name'] for db in config['sql_db_config'][db_type]['databases']
            ]
    
    return databases
