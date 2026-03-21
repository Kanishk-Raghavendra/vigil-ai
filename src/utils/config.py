"""Configuration system for VIGIL."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Load and manage VIGIL configuration."""
    
    @staticmethod
    def load(config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
