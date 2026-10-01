"""Experiment runner - handles single experiment execution"""

import random
import numpy as np
import torch
from utils.experiment import setup_experiment
from utils.trainer import train_model, print_model_summary
from utils.metrics import post_training_analysis
from models import create_model, InputShapeWrapper


def set_global_seed(seed: int = 42):
    """Seed everything for full reproducibility.

    Covers: Python random, NumPy, PyTorch CPU, PyTorch CUDA (all GPUs),
    cuDNN deterministic mode, and DataLoader worker seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # disables auto-tuner (speed vs repro)
    print(f"[Seed] Global seed set to {seed}")


def resolve_run_seed(cfg, run_number):
    """Resolve the seed for a specific run so that each run is genuinely different.

    Priority:
      1. data_split.seeds: [s1, s2, ...]  → use the seed listed for this run
         (run_number 1 → seeds[0], run 2 → seeds[1], ...). If the list is
         shorter than the run number, fall back to the increasing-order rule.
      2. Otherwise → base_seed + (run_number - 1), i.e. seeds increase per run
         (base, base+1, base+2, ...) so 5 runs give 5 different splits/inits
         and a meaningful mean ± std.

    Note: if you WANT identical runs, set data_split.seeds to a single repeated
    value (e.g. [42, 42, 42, 42, 42]).
    """
    base_seed = cfg.get('data_split.random_state', 42)
    run_idx = (run_number - 1) if run_number else 0

    seeds = cfg.get('data_split.seeds', None)
    if seeds:
        if isinstance(seeds, int):
            seeds = [seeds]
        if run_idx < len(seeds):
            return int(seeds[run_idx])
        # Not enough seeds listed → continue the increasing sequence from base
        print(f"[Seed] Only {len(seeds)} seed(s) listed for run {run_number}; "
              f"falling back to increasing-order seed.")

    return int(base_seed) + run_idx


def run_single_experiment(cfg, model_name=None, run_number=None, dataset_name=None):
    """Run a single experiment with specified model, run number, and dataset"""

    # ── Reproducibility: seed everything before any random op ──────────────
    # Each run gets its own seed so num_runs > 1 yields genuinely different
    # weight inits AND data splits (real mean ± std). See resolve_run_seed.
    seed = resolve_run_seed(cfg, run_number)
    print(f"[Seed] Run {run_number} using seed {seed}")
    set_global_seed(seed)

    # Setup experiment: dataset, dataloaders, run directory
    # Pass the resolved per-run seed so the train/val/test split also varies.
    exp = setup_experiment(cfg, run_number=run_number, dataset_name=dataset_name,
                           model_name=model_name, seed=seed)
    
    # Device configuration
    use_cuda = cfg.get('device.use_cuda', True)
    cuda_device = cfg.get('device.cuda_device', 0)
    if torch.cuda.is_available() and use_cuda:
        device = f'cuda:{cuda_device}'
    else:
        device = 'cpu'
    
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
        optimizer_params=optimizer_params,
        num_classes=exp['num_classes'],
    )
    
    # Visualization parameters
    vis_mode = cfg.get('visualization.mode', 'labeled_only')
    cmap = cfg.get('visualization.cmap', 'tab20')
    use_spy_colors = cfg.get('visualization.use_spy_colors', True)
    show_colorbar = cfg.get('visualization.show_colorbar', False)
    dpi = cfg.get('visualization.dpi', 300)
    block_background = cfg.get('visualization.block_background', True)
    results_dir = cfg.get('results.directory', 'results')
    generate_maps = cfg.get('visualization.generate_maps', True)
    prune_checkpoints = cfg.get('results.keep_best_worst_checkpoints_only', True)
    pruning_metric = cfg.get('results.checkpoint_pruning_metric', 'OA')

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
        block_background=block_background,
        vis_mode=vis_mode,
        use_spy_colors=use_spy_colors,
        results_dir=results_dir,
        generate_maps=generate_maps,
        prune_checkpoints=prune_checkpoints,
        pruning_metric=pruning_metric,
    )
