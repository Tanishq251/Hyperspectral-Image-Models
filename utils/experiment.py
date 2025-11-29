import os
import json
import torch
from torch.utils.data import DataLoader, Subset
from utils.data_loader import DatasetLoader, HyperspectralDataset
from utils.data_split import split_data, split_samples, print_class_stats


def get_next_run_number(results_dir):
    """Get the next available run number"""
    if not os.path.exists(results_dir):
        return 1
    
    existing_runs = [d for d in os.listdir(results_dir) 
                    if d.startswith('run_') and os.path.isdir(os.path.join(results_dir, d))]
    if not existing_runs:
        return 1
    
    run_numbers = []
    for run in existing_runs:
        try:
            num = int(run.split('_')[1])
            run_numbers.append(num)
        except:
            continue
    
    return max(run_numbers) + 1 if run_numbers else 1


def create_run_directory(dataset_name, model_name, run_number=None):
    """Create directory structure for experiment run"""
    results_base = "results"
    dataset_dir = os.path.join(results_base, dataset_name)
    model_dir = os.path.join(dataset_dir, model_name)
    
    if run_number is None:
        run_number = get_next_run_number(model_dir)
    
    run_dir = os.path.join(model_dir, f"run_{run_number}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir, run_number


def save_config(run_dir, config):
    """Save experiment configuration"""
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved: {config_path}")


def setup_experiment(cfg, run_number=None, dataset_name=None):
    """Setup complete experiment: dataset, dataloaders, and run directory"""
    
    # Get configuration
    datasets_folder = cfg.get('dataset.datasets_folder', 'datasets')
    
    # Use provided dataset_name or get from config
    if dataset_name is None:
        dataset_name = cfg.get('dataset.names', ['WHU-Hi-HanChuan'])[0]
    
    model_name = cfg.get('model.name', 'DBCTNet')
    
    # Load dataset using smart loader
    loader = DatasetLoader(datasets_folder)
    data, gt = loader.load_dataset(dataset_name)
    
    # Get preprocessing parameters
    patch_size = cfg.get('dataset.patch_size', 9)
    batch_size = cfg.get('training.batch_size', 32)
    stride = cfg.get('dataset.stride', 1)
    
    split_method = cfg.get('data_split.method', 'ratio')
    if split_method == 'ratio':
        split_ratios = tuple(cfg.get('data_split.split_ratios', [0.2, 0.1, 0.7]))
        split_samples_count = None
    else:
        split_ratios = None
        split_samples_count = tuple(cfg.get('data_split.split_samples', [30, 10]))
    
    dim_reduction_method = cfg.get('preprocessing.dim_reduction_method')
    num_pca_bands = cfg.get('preprocessing.num_pca_bands', 20)
    maxpool_kernel = cfg.get('preprocessing.maxpool_kernel', 2)
    use_channel_dim = cfg.get('preprocessing.use_channel_dim', True)
    band_indices = cfg.get('preprocessing.band_indices')
    
    # Create run directory
    if run_number is not None:
        run_dir, actual_run_number = create_run_directory(dataset_name, model_name, run_number=run_number)
    else:
        run_dir, actual_run_number = create_run_directory(dataset_name, model_name)
    
    print(f"\n{'='*60}")
    print(f"Run Directory: {run_dir}")
    print(f"Run Number: {actual_run_number}")
    print(f"{'='*60}\n")
    
    run_config = cfg.to_dict()
    run_config['run_number'] = actual_run_number
    save_config(run_dir, run_config)
    
    # Create dataset
    verbose_dataset = cfg.get('dataset.verbose', True)
    dataset = HyperspectralDataset(
        data=data,
        gt=gt,
        patch_size=patch_size,
        stride=stride,
        dim_reduction_method=dim_reduction_method,
        num_pca_bands=num_pca_bands,
        maxpool_kernel=maxpool_kernel,
        use_channel_dim=use_channel_dim,
        band_indices=band_indices,
        verbose=verbose_dataset
    )
    
    # Split dataset
    labels = [label for _, _, label in dataset.positions]
    num_classes = len(set(labels))
    
    if split_samples_count is not None:
        split_result = split_samples(*split_samples_count, labels=labels, random_state=42)
    else:
        split_result = split_data(*split_ratios, labels=labels, random_state=42)
    
    if len(split_result) == 2:
        train_idx, test_idx = split_result
        val_idx = None
    else:
        train_idx, val_idx, test_idx = split_result
    
    # Create dataloaders
    num_workers = cfg.get('training.num_workers', 4)
    
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    if val_idx is not None:
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        val_loader = None
    
    # Print class statistics if enabled
    print_stats = cfg.get('data_split.print_stats', True)
    if print_stats:
        print_class_stats(labels, train_idx, test_idx, val_idx, num_classes)
    
    # Get bands from sample
    sample_input, _ = dataset[0]
    bands = sample_input.shape[1]
    
    return {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'run_dir': run_dir,
        'run_number': actual_run_number,
        'dataset': dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'num_classes': num_classes,
        'bands': bands,
        'patch_size': patch_size,
        'batch_size': batch_size,
        'split_ratios': split_ratios,
        'split_samples_count': split_samples_count
    }
