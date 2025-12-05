"""Model registry with decorator pattern - auto-discovers models"""

import os
import importlib

_model_registry = {}
_model_configs = {}
_models_loaded = False


def register_model(name, expects_4d=False, **default_cfg):
    """Decorator to register a model with optional config"""
    def decorator(cls):
        _model_registry[name] = cls
        config = default_cfg or {}
        config['expects_4d'] = expects_4d
        _model_configs[name] = config
        return cls
    return decorator


def _load_models():
    """Auto-discover and load all models from this directory and subdirectories"""
    global _models_loaded
    if _models_loaded:
        return
    
    models_dir = os.path.dirname(__file__)
    skip_files = {'__init__.py', 'registry.py', '__pycache__'}
    
    # Load models from root directory
    for filename in os.listdir(models_dir):
        if filename in skip_files or not filename.endswith('.py'):
            continue
        try:
            importlib.import_module(f'models.{filename[:-3]}')
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
    
    # Load models from subdirectories (e.g., Ours)
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path) and not item.startswith('_') and item != '__pycache__':
            try:
                importlib.import_module(f'models.{item}')
            except Exception as e:
                print(f"Warning: Could not load models.{item}: {e}")
    
    _models_loaded = True


def create_model(model_name, **kwargs):
    """Create a model by registered name with optional config override.
    
    Standard parameters (handled by each model's factory function):
    - num_classes: Number of output classes
    - bands: Number of spectral bands
    - patch_size: Spatial patch size
    """
    _load_models()
    
    if model_name not in _model_registry:
        available = list(_model_registry.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    return_config = kwargs.pop('return_config', False)

    # Merge default config with provided kwargs
    if model_name in _model_configs:
        cfg = _model_configs[model_name].copy()
        cfg.update(kwargs)
        model_args = {k: v for k, v in cfg.items() if k != 'expects_4d'}
        model = _model_registry[model_name](**model_args)
        if return_config:
            return cfg, model
        return model
    
    model = _model_registry[model_name](**kwargs)
    if return_config:
        return kwargs, model
    return model


def list_models():
    """List all registered models"""
    _load_models()
    return list(_model_registry.keys())


def get_model_config(model_name):
    """Get the config for a model"""
    _load_models()
    return _model_configs.get(model_name, {})
