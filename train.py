import torch
from HSI import setup_experiment, train_model, post_training_analysis
from DBCTnet import DBCTNet
from M3DRecNet import HSIVit
from config_loader import load_config


def run_single_experiment(cfg, model_name=None, run_number=None):
    """Run a single experiment with specified model and run number"""
    # Override model name if provided
    if model_name:
        cfg.config['model']['name'] = model_name
    
    # Setup experiment: dataset, dataloaders, run directory
    exp = setup_experiment(cfg, run_number=run_number)
    
    # Model hyperparameters - modify these directly to experiment
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

# you can add more models here as needed

    else:
        raise ValueError(f"Unknown model: {exp['model_name']}. Supported models: DBCTNet, 3DRecNet")
    
    # Training parameters
    num_epochs = cfg.get('training.num_epochs', 50)
    learning_rate = cfg.get('training.learning_rate', 0.001)
    patience = cfg.get('training.patience', 10)
    checkpoint_interval = cfg.get('training.checkpoint_interval', 10)
    use_cuda = cfg.get('device.use_cuda', True)
    device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'
    
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
    
    # Post-training analysis: metrics, classification map, Excel
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


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    
    # Get configuration for multiple runs and models
    num_runs = cfg.get('training.num_runs', 1)
    run_all_models = cfg.get('model.run_all_models', False)
    
    # Determine which models to run
    if run_all_models:
        models_to_run = ["DBCTNet", "3DRecNet", "GGSM"]
        print(f"Running all models: {models_to_run}")
    else:
        models_to_run = [cfg.get('model.name', '3DRecNet')]
        print(f"Running single model: {models_to_run[0]}")
    
    print(f"Number of runs per model: {num_runs}")
    print("-" * 60)
    
    # Run experiments for each model and each run
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Starting experiments for model: {model_name}")
        print(f"{'='*60}\n")
        
        for run_num in range(1, num_runs + 1):
            print(f"\n--- Run {run_num}/{num_runs} for {model_name} ---\n")
            run_single_experiment(cfg, model_name=model_name, run_number=run_num)
            print(f"\n--- Completed Run {run_num}/{num_runs} for {model_name} ---\n")
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_file)
