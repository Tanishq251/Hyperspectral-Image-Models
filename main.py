"""
Main entry point for HSI classification pipeline
"""

import sys
import warnings

# Suppress deprecation warnings from third-party libraries
warnings.filterwarnings('ignore', category=UserWarning, module='pyramid.path')
warnings.filterwarnings('ignore', category=FutureWarning, module='timm.models.layers')

from config.config_loader import load_config
from utils.experiment_runner import run_single_experiment
from utils.data_loader import DatasetLoader
from models import list_models


def get_datasets_to_run(cfg):
    """Determine which datasets to run based on config"""
    run_all_datasets = cfg.get('dataset.run_all_datasets', False)
    
    if run_all_datasets:
        # Load all dataset names from dataset.yaml
        import yaml
        from pathlib import Path
        dataset_yaml_path = Path("config/dataset.yaml")
        if dataset_yaml_path.exists():
            with open(dataset_yaml_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            datasets_to_run = list(dataset_config.get('datasets', {}).keys())
        else:
            raise FileNotFoundError("dataset.yaml not found. Cannot run all datasets.")
    else:
        datasets_to_run = cfg.get('dataset.names', [])
        
        if isinstance(datasets_to_run, str):
            datasets_to_run = [datasets_to_run]
    
    if not datasets_to_run:
        raise ValueError("No datasets specified in config. Please provide dataset.names or set run_all_datasets: True")
    
    print(f"Datasets to run ({len(datasets_to_run)}): {', '.join(datasets_to_run)}")
    print("(Datasets will be automatically downloaded to cache if not present)")
    
    return datasets_to_run


def get_models_to_run(cfg):
    """Determine which models to run based on config"""
    run_all_models = cfg.get('model.run_all_models', False)
    
    if run_all_models:
        models_to_run = list_models()
    else:
        model_name_cfg = cfg.get('model.name', '')
        if isinstance(model_name_cfg, list):
            models_to_run = model_name_cfg
        else:
            models_to_run = [m.strip() for m in model_name_cfg.split(',')]
        models_to_run = [m for m in models_to_run if m]
    
    return models_to_run


def run_map_arrangement_only(cfg):
    """Run map arrangement without training"""
    print("\n" + "="*60)
    print("Map Arrangement Only Mode")
    print("="*60 + "\n")
    
    from utils.map_arranger_integration import arrange_maps_after_training
    
    dataset_name = cfg.get('map_arrangement.dataset_dir', 'Utopia')
    models_cfg = cfg.get('map_arrangement.models', [])
    
    if isinstance(models_cfg, dict):
        models = list(models_cfg.keys())
    else:
        models = models_cfg if isinstance(models_cfg, list) else []
    
    if not models:
        print("Error: No models specified in map_arrangement config")
        return
    
    # force_run=True to bypass the enabled check for --arrange-only mode
    arrange_maps_after_training(cfg, dataset_name, models, force_run=True)


def main(config_path="config/config.yaml", arrange_only=False):
    """Main entry point for training pipeline"""
    
    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)
    
    if arrange_only:
        run_map_arrangement_only(cfg)
        return
    
    # Get configuration
    num_runs = cfg.get('training.num_runs', 1)
    datasets_to_run = get_datasets_to_run(cfg)
    models_to_run = get_models_to_run(cfg)
    
    print(f"Number of runs per model: {num_runs}")
    print("-" * 60)
    
    # Run experiments
    total_experiments = len(datasets_to_run) * len(models_to_run) * num_runs
    current_experiment = 0
    
    for dataset_name in datasets_to_run:
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*60}\n")
        
        for model_name in models_to_run:
            print(f"\n{'='*60}")
            print(f"Starting experiments for model: {model_name}")
            print(f"{'='*60}\n")
            
            for run_num in range(1, num_runs + 1):
                current_experiment += 1
                print(f"\n--- Experiment {current_experiment}/{total_experiments}: {dataset_name} | {model_name} | Run {run_num}/{num_runs} ---\n")
                
                try:
                    run_single_experiment(
                        cfg, 
                        model_name=model_name, 
                        run_number=None,
                        dataset_name=dataset_name
                    )
                    print(f"\n✓ Completed: {dataset_name} | {model_name} | Run {run_num}/{num_runs}\n")
                except Exception as e:
                    print(f"\n✗ Failed: {dataset_name} | {model_name} | Run {run_num}/{num_runs}")
                    print(f"Error: {e}\n")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Total experiments run: {current_experiment}/{total_experiments}")
    print(f"{'='*60}")
    
    # Optionally arrange maps after training
    if cfg.get('map_arrangement.enabled', False):
        from utils.map_arranger_integration import arrange_maps_after_training
        for dataset_name in datasets_to_run:
            arrange_maps_after_training(cfg, dataset_name, models_to_run)


def list_available_models():
    """List all available models"""
    from models import list_models, get_model_config
    
    models = list_models()
    print(f"\n{'='*60}")
    print(f"Available Models ({len(models)})")
    print(f"{'='*60}")
    print(f"{'Model Name':<25} {'4D Input':<10} {'Notes'}")
    print(f"{'-'*60}")
    for name in sorted(models):
        cfg = get_model_config(name)
        expects_4d = 'Yes' if cfg.get('expects_4d', False) else 'No'
        notes = ''
        if name == 'GSCViT':
            notes = 'Requires 8x8 patch'
        print(f"{name:<25} {expects_4d:<10} {notes}")
    print(f"{'='*60}\n")


def list_available_datasets():
    """List all available datasets"""
    from utils.data_loader import DatasetLoader
    
    loader = DatasetLoader(use_cache=True)
    datasets = loader.list_available_datasets()
    print(f"\n{'='*60}")
    print(f"Available Datasets ({len(datasets)})")
    print(f"{'='*60}")
    for i, name in enumerate(sorted(datasets), 1):
        print(f"  {i:2d}. {name}")
    print(f"{'='*60}")
    print(f"Datasets are auto-downloaded from HuggingFace Hub")
    print(f"Cache location: ~/.cache/huggingface/")
    print(f"{'='*60}\n")


def print_help():
    """Print help message"""
    print("""
Hyperspectral Image Classification Framework
=============================================

Usage:
  python main.py [config_file] [options]

Options:
  --list-models      List all available models
  --list-datasets    List all available datasets
  --arrange-only     Run map arrangement only (no training)
  --help, -h         Show this help message

Examples:
  python main.py                          # Run with default config
  python main.py config/my_config.yaml    # Run with custom config
  python main.py --list-models            # List available models
  python main.py --list-datasets          # List available datasets
  python main.py --arrange-only           # Arrange maps only
""")


if __name__ == "__main__":
    config_file = "config/config.yaml"
    arrange_only = False
    
    # Parse arguments
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        print_help()
        sys.exit(0)
    
    if '--list-models' in args:
        list_available_models()
        sys.exit(0)
    
    if '--list-datasets' in args:
        list_available_datasets()
        sys.exit(0)
    
    for arg in args:
        if arg == "--arrange-only":
            arrange_only = True
        elif not arg.startswith("--"):
            config_file = arg
    
    main(config_file, arrange_only=arrange_only)
