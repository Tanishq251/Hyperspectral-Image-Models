"""Experiment runner - handles single experiment execution"""

import torch
from utils.experiment import setup_experiment
from utils.trainer import train_model, print_model_summary
from utils.metrics import post_training_analysis
from models import create_model, InputShapeWrapper


def run_single_experiment(cfg, model_name=None, run_number=None, dataset_name=None):
    """Run a single experiment with specified model, run number, and dataset"""
    
    # Setup experiment: dataset, dataloaders, run directory
    exp = setup_experiment(cfg, run_number=run_number, dataset_name=dataset_name, model_name=model_name)
    
    # Device configuration
    use_cuda = cfg.get('device.use_cuda', True)
    device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'
    
    # Model initialization - pass dataset-specific parameters
    # Registry handles mapping to model-specific parameter names
    model_kwargs = {
        'num_classes': exp['num_classes'],
        'bands': exp['bands'],
        'patch_size': exp['patch_size'],
    }
    
    model_cfg, model = create_model(
        model_name,
        return_config=True,
        **model_kwargs
    )
    
    # Automatically wrap model if it expects a 4D input
    if model_cfg.get('expects_4d', False):
        print(f"Info: Model '{model_name}' expects 4D input. Applying InputShapeWrapper.")
        model = InputShapeWrapper(model)
    
    # Print model summary if requested
    print_summary = cfg.get('model.print_summary', False)
    summary_only = cfg.get('model.summary_only', False)
    
    if print_summary:
        sample_input, _ = exp['dataset'][0]
        input_shape = (exp['batch_size'],) + tuple(sample_input.shape)
        summary_depth = cfg.get('model.summary_depth', 4)
        print_model_summary(model, input_shape, device, depth=summary_depth)
    
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
    optimizer_name = cfg.get('training.optimizer', 'adam')
    optimizer_params = cfg.get('training.optimizer_params', {})
    
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
        checkpoint_interval=checkpoint_interval,
        optimizer_name=optimizer_name,
        optimizer_params=optimizer_params
    )
    
    # Visualization parameters
    cmap = cfg.get('visualization.cmap', 'tab20')
    show_colorbar = cfg.get('visualization.show_colorbar', False)
    dpi = cfg.get('visualization.dpi', 300)
    block_background = cfg.get('visualization.block_background', True)
    
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
        dpi=dpi,
        block_background=block_background
    )
