import os
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate an active learning configuration YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
        
    # Validate required sections
    required_sections = ["detector", "dataset", "active_learning", "training"]
    for s in required_sections:
        if s not in cfg:
            raise ValueError(f"Missing required section in config: {s}")
            
    # Validate critical paths
    for key in ["architecture_yaml", "checkpoint_path"]:
        p = cfg["detector"].get(key)
        if p and not Path(p).exists():
            raise FileNotFoundError(f"Detector path does not exist: {p}")
            
    return cfg
