import torch
from utils.experiment import setup_experiment
from utils.trainer import train_model
from utils.metrics import post_training_analysis
from models.DBCTnet import DBCTNet
from models.M3DRecNet import HSIVit
from config.config_loader import load_config


def print_model_summary(model, input_shape, device='cuda'):
    """Print model summary using torchinfo"""
    try:
        from torchinfo import summary
        
        print("\n" + "="*60)
        print("Model Summary")
        print("="*60)
        
        summary(
            model,
            input_size=input_shape,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=10,
            device=device
        )
        
        print("="*60 + "\n")
    except ImportError:
        print("Warning: torchinfo not installed. Install with: pip install torchinfo")


def run_single_experiment(cfg, model_name=None, run_number=None, dataset_name=None):
    """Run a single experiment with specified model, run number, and dataset"""
    
    # Override model name if provided
    if model_name:
        cfg.config['model']['name'] = model_name
    
    # Setup experiment: dataset, dataloaders, run directory
    exp = setup_experiment(cfg, run_number=run_number, dataset_name=dataset_name)
    
    # Device configuration
    use_cuda = cfg.get('device.use_cuda', True)
    device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'
    
    # Model initialization
    if exp['model_name'] == "DBCTNet":
        model = DBCTNet(
            channels=16,
            patch=exp['patch_size'],
            bands=exp['bands'],
            num_class=exp['num_classes'],
            fc_dim=16,
            heads=2,
            drop=0.1
        )
    elif exp['model_name'] == "3DRecNet":
        model = HSIVit(
            in_chans=1,
            num_classes=exp['num_classes'],
            depths=[3, 3, 9, 3],
            dims=[32, 64, 128, 256],
            drop_path_rate=0.05,
            layer_scale_init_value=1e-6
        )
    else:
        raise ValueError(f"Unknown model: {exp['model_name']}. Supported: DBCTNet, 3DRecNet")
    
    # Print model summary if requested
    print_summary = cfg.get('model.print_summary', False)
    summary_only = cfg.get('model.summary_only', False)
    
    if print_summary:
        sample_input, _ = exp['dataset'][0]
        input_shape = (exp['batch_size'],) + tuple(sample_input.shape)
        print_model_summary(model, input_shape, device)
    
    # If summary_only mode, skip training
    if summary_only:
        print("\n" + "="*60)
        print("Summary-only mode enabled. Skipping training.")
        print("="*60)
        return
    
    # Training parameters
    num_epochs = cfg.get('training.num_epochs', 50)
    learning_rate = cfg.get('training.learning_rate', 0.001)
    patience = cfg.get('training.patience', 10)
    checkpoint_interval = cfg.get('training.checkpoint_interval', 10)
    
    # Train model
    trained_model, predictions, targets, best_epoch, training_time = train_model(
        model=model,
        train_loader=exp['train_loader'],
        val_loader=exp['val_loader'],
        test_loader=exp['test_loader'],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        patience=patience,
        run_dir=exp['run_dir'],
        checkpoint_interval=checkpoint_interval
    )
    
    # Visualization parameters
    cmap = cfg.get('visualization.cmap', 'tab20')
    show_colorbar = cfg.get('visualization.show_colorbar', False)
    dpi = cfg.get('visualization.dpi', 300)
    
    # Post-training analysis: metrics, classification map, CSV
    post_training_analysis(
        trained_model=trained_model,
        predictions=predictions,
        targets=targets,
        best_epoch=best_epoch,
        training_time=training_time,
        dataset=exp['dataset'],
        device=device,
        run_dir=exp['run_dir'],
        dataset_name=exp['dataset_name'],
        model_name=exp['model_name'],
        run_number=exp['run_number'],
        num_epochs=num_epochs,
        patch_size=exp['patch_size'],
        batch_size=exp['batch_size'],
        split_ratios=exp['split_ratios'],
        split_samples_count=exp['split_samples_count'],
        train_idx=exp['train_idx'],
        val_idx=exp['val_idx'],
        test_idx=exp['test_idx'],
        num_classes=exp['num_classes'],
        cmap=cmap,
        show_colorbar=show_colorbar,
        dpi=dpi
    )


def main(config_path="config/config.yaml"):
    """Main entry point for training"""
    
    cfg = load_config(config_path)
    
    # Get configuration for multiple runs, models, and datasets
    num_runs = cfg.get('training.num_runs', 1)
    run_all_models = cfg.get('model.run_all_models', False)
    use_all_datasets = cfg.get('dataset.use_all', False)
    
    # Determine which datasets to run
    if use_all_datasets:
        from utils.data_loader import DatasetLoader
        datasets_folder = cfg.get('dataset.datasets_folder', 'datasets')
        loader = DatasetLoader(datasets_folder)
        datasets_to_run = list(loader.available_datasets.keys())
        print(f"Running on ALL datasets: {datasets_to_run}")
    else:
        datasets_to_run = cfg.get('dataset.names', ['WHU-Hi-HanChuan'])
        if isinstance(datasets_to_run, str):
            datasets_to_run = [datasets_to_run]
        print(f"Running on specified datasets: {datasets_to_run}")
    
    # Determine which models to run
    if run_all_models:
        models_to_run = ["DBCTNet", "3DRecNet"]
        print(f"Running all models: {models_to_run}")
    else:
        models_to_run = [cfg.get('model.name', 'DBCTNet')]
        print(f"Running single model: {models_to_run[0]}")
    
    print(f"Number of runs per model: {num_runs}")
    print("-" * 60)
    
    # Run experiments for each dataset, model, and run
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
                        run_number=run_num,
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


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    main(config_file)
