"""
Main entry point for HSI classification pipeline
"""

import os
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
        model_name_cfg = cfg.get('model.name', [])
        if isinstance(model_name_cfg, list):
            models_to_run = model_name_cfg
        elif isinstance(model_name_cfg, str):
            models_to_run = [m.strip() for m in model_name_cfg.split(',')]
        else:
            models_to_run = []
        models_to_run = [m for m in models_to_run if m]
    
    # Apply exclude list
    exclude_models = cfg.get('model.exclude', [])
    if exclude_models:
        if isinstance(exclude_models, str):
            exclude_models = [m.strip() for m in exclude_models.split(',')]
        models_to_run = [m for m in models_to_run if m not in exclude_models]
        print(f"Excluded models: {', '.join(exclude_models)}")
    
    if not models_to_run:
        raise ValueError("No models specified in config. Please provide model.name or set run_all_models: True")
    
    print(f"Models to run ({len(models_to_run)}): {', '.join(models_to_run)}")
    
    return models_to_run


def run_map_arrangement_only(cfg, save_sep_folder=False):
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
    arrange_maps_after_training(cfg, dataset_name, models, force_run=True, save_sep_folder=save_sep_folder)


def main(config_path="config/config.yaml", arrange_only=False, save_sep_folder=False):
    """Main entry point for training pipeline
    
    Args:
        config_path: Path to config file
        arrange_only: If True, only run map arrangement
        save_sep_folder: If True, save individual maps to dataset_maps folder
    """
    
    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)
    
    if arrange_only:
        run_map_arrangement_only(cfg, save_sep_folder=save_sep_folder)
        return
    
    # Get results directory from config
    results_dir = cfg.get('results.directory', 'results')
    
    # Get configuration
    num_runs = cfg.get('training.num_runs', 1)
    datasets_to_run = get_datasets_to_run(cfg)
    models_to_run = get_models_to_run(cfg)
    
    # Normal execution mode
    print(f"Number of runs per model: {num_runs}")
    print(f"Results directory: {results_dir}")
    print("-" * 60)
    
    # Run experiments
    # Get patch sizes and training samples to loop through
    patch_sizes = cfg.get('dataset.patch_size', 11)
    if not isinstance(patch_sizes, list):
        patch_sizes = [patch_sizes]
    
    split_samples_list = cfg.get('data_split.split_samples', [30, 10])
    # Handle different formats: [30, 10] or [[30, 10], [20, 7]]
    if split_samples_list and isinstance(split_samples_list[0], list):
        # Already list of lists
        pass
    else:
        # Single config, wrap in list
        split_samples_list = [split_samples_list]
    
    total_experiments = len(datasets_to_run) * len(models_to_run) * len(patch_sizes) * len(split_samples_list) * num_runs
    current_experiment = 0
    
    for dataset_name in datasets_to_run:
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*60}\n")
        
        for patch_size in patch_sizes:
            for split_samples in split_samples_list:
                # Update config with current patch_size and split_samples
                cfg.config['dataset']['patch_size'] = patch_size
                cfg.config['data_split']['split_samples'] = split_samples
                
                train_samples = split_samples[0] if split_samples else 'default'
                
                print(f"\n{'*'*60}")
                print(f"* Patch Size: {patch_size}x{patch_size} | Training Samples: {split_samples}")
                print(f"{'*'*60}\n")
                
                for model_name in models_to_run:
                    print(f"\n{'='*60}")
                    print(f"Starting experiments for model: {model_name}")
                    print(f"{'='*60}\n")
                    
                    # Get the next available run number for this model (normal directory structure)
                    model_dir = f"{results_dir}/{dataset_name}/{model_name}"
                    existing_runs = []
                    if os.path.exists(model_dir):
                        for item in os.listdir(model_dir):
                            if os.path.isdir(os.path.join(model_dir, item)) and item.startswith('run_'):
                                try:
                                    existing_runs.append(int(item.split('_')[1]))
                                except (ValueError, IndexError):
                                    pass
                    
                    # Skip runs already completed for this model (resume-safe):
                    # if num_runs=3 and run_1..3 already exist, do nothing here
                    # instead of appending run_4..6 with out-of-spec seeds.
                    if len(existing_runs) >= num_runs:
                        total_experiments -= num_runs
                        print(f"Skipping {model_name}: {len(existing_runs)} run(s) already completed "
                              f"(>= num_runs={num_runs}).")
                        continue

                    start_run = max(existing_runs) + 1 if existing_runs else 1

                    for run_num in range(start_run, start_run + num_runs - len(existing_runs)):
                        current_experiment += 1
                        
                        print(f"\n--- Experiment {current_experiment}/{total_experiments}: {dataset_name} | {model_name} | Patch {patch_size} | Train {train_samples} | Run {run_num}/{start_run + num_runs - 1} ---\n")
                        
                        try:
                            run_single_experiment(
                                cfg, 
                                model_name=model_name, 
                                run_number=run_num,
                                dataset_name=dataset_name
                            )
                            
                            print(f"\nCompleted: {dataset_name} | {model_name} | Patch {patch_size} | Train {train_samples} | Run {run_num}\n")
                        except Exception as e:
                            print(f"\nFailed: {dataset_name} | {model_name} | Patch {patch_size} | Train {train_samples} | Run {run_num}")
                            print(f"Error: {e}\n")
                            import traceback
                            traceback.print_exc()
                            continue
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Total: {total_experiments} | Completed: {current_experiment}")
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


def run_score_arrangement(cfg):
    """Arrange scores and generate LaTeX tables"""
    from utils.score_arranger import arrange_scores, print_scores_summary, get_available_datasets, get_available_models
    
    # Get results directory from config
    results_dir = cfg.get('results.directory', 'results')
    
    # Check if we should process all datasets
    arrange_all_datasets = cfg.get('score_arrangement.all_datasets', False)
    
    if arrange_all_datasets:
        # Get all available datasets from results directory
        dataset_names = get_available_datasets(results_dir=results_dir)
        if not dataset_names:
            print(f"No datasets found in {results_dir} directory")
            return
        print(f"Processing all available datasets: {', '.join(dataset_names)}")
    else:
        # Use datasets from config
        dataset_names = cfg.get('dataset.names', [])
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
    
    if not dataset_names:
        print("No datasets specified. Set 'dataset.names' in config or use 'score_arrangement.all_datasets: True'")
        return
    
    # Check if we should process all models or specific models
    arrange_all_models = cfg.get('score_arrangement.all_models', True)
    specific_models = cfg.get('score_arrangement.models', [])
    
    if isinstance(specific_models, str):
        specific_models = [specific_models]
    
    use_mean = cfg.get('score_arrangement.use_mean', True)
    
    for dataset_name in dataset_names:
        # Determine which models to process
        if arrange_all_models:
            models_to_process = None  # None = all models
            print(f"\nProcessing all models for dataset: {dataset_name}")
        elif specific_models:
            models_to_process = specific_models
            print(f"\nProcessing specific models for dataset: {dataset_name}: {', '.join(models_to_process)}")
        else:
            models_to_process = None
            print(f"\nProcessing all models for dataset: {dataset_name}")
        
        # Print console summary
        print_scores_summary(dataset_name, models=models_to_process, results_dir=results_dir)
        # Generate LaTeX tables
        arrange_scores(dataset_name, models=models_to_process, use_mean=use_mean, results_dir=results_dir)


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
  --arrange-scores   Generate LaTeX tables from results
  --help, -h         Show this help message

Examples:
  python main.py                          # Run with default config
  python main.py config/my_config.yaml    # Run with custom config
  python main.py --list-models            # List available models
  python main.py --list-datasets          # List available datasets
  python main.py --arrange-only           # Arrange maps only
  python main.py --arrange-scores         # Generate LaTeX tables
""")


if __name__ == "__main__":
    config_file = "config/config.yaml"
    arrange_only = False
    save_sep_folder = False
    arrange_scores = False
    
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
        elif arg == "--in_sep_folder":
            save_sep_folder = True
        elif arg == "--arrange-scores":
            arrange_scores = True
        elif not arg.startswith("--"):
            config_file = arg
    
    # Handle arrange-scores mode
    if arrange_scores:
        cfg = load_config(config_file)
        run_score_arrangement(cfg)
        sys.exit(0)
    
    main(config_file, arrange_only=arrange_only, save_sep_folder=save_sep_folder)
