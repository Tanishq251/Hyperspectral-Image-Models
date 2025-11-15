import yaml
import json
from pathlib import Path


class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        path = Path(self.config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def __getitem__(self, key):
        return self.get(key)
    
    def to_dict(self):
        return self.config


def load_config(config_path="config.yaml"):
    return Config(config_path)
