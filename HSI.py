import os
import numpy as np
import rasterio
import scipy.io
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import json
import time
from sklearn.metrics import cohen_kappa_score, accuracy_score


class KeyMapper:
    def __init__(self, data_key='data', gt_key='gt'):
        self.data_key = data_key
        self.gt_key = gt_key
        self.data_key_variations = [
            'data', 'Data', 'DATA', 'image', 'Image', 'IMAGE', 'hsi', 'HSI', 'hyperspectral', 'Hyperspectral', 'cube', 'Cube'
        ]
        self.gt_key_variations = [
            'gt', 'GT', 'Gt', 'label', 'Label', 'LABEL', 'labels', 'Labels', 'LABELS',
            'ground_truth', 'groundtruth', 'GroundTruth', 'mask', 'Mask', 'MASK'
        ]
    
    def find_key(self, mat_dict, key_variations):
        valid_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
        for key in key_variations:
            if key in valid_keys:
                return key
        if valid_keys:
            return valid_keys[0]
        return None
    
    def get_mapped_dict(self, mat_dict):
        data_actual = self.find_key(mat_dict, self.data_key_variations)
        gt_actual = self.find_key(mat_dict, self.gt_key_variations)
        if data_actual == self.data_key and gt_actual == self.gt_key:
            return mat_dict, True
        new_dict = {}
        if data_actual is not None:
            new_dict[self.data_key] = mat_dict[data_actual]
        if gt_actual is not None:
            new_dict[self.gt_key] = mat_dict[gt_actual]
        return new_dict, False


class ChangeKeys:
    def __init__(self, folder_path):
        self.folder_path = folder_path
    
    def to(self, data_key='data', gt_key='gt', save=False, overwrite=True):
        key_mapper = KeyMapper(data_key=data_key, gt_key=gt_key)
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.mat'):
                file_path = os.path.join(self.folder_path, filename)
                mat_dict = scipy.io.loadmat(file_path)
                mapped_dict, no_change = key_mapper.get_mapped_dict(mat_dict)
                if no_change:
                    continue
                if save:
                    out_path = file_path if overwrite else f"{file_path[:-4]}_standard.mat"
                    scipy.io.savemat(out_path, mapped_dict)


class HyperspectralDataset(Dataset):
    def __init__(self, dataset_folder, patch_size=11, stride=1, 
                 dim_reduction_method=None, num_pca_bands=None,
                 maxpool_kernel=2, use_channel_dim=True, band_indices=None, band_axis='channels_first'):
        self.dataset_folder = dataset_folder
        self.patch_size = patch_size
        self.stride = stride
        self.dim_reduction_method = dim_reduction_method
        self.num_pca_bands = num_pca_bands
        self.maxpool_kernel = maxpool_kernel
        self.use_channel_dim = use_channel_dim
        self.band_indices = band_indices
        self.band_axis = band_axis
        
        self.data_file, self.gt_file = self.find_data_files()
        self.data, self.gt = self.load_data()
        self.positions = self.create_patch_positions()
        self.targets = [label for _, _, label in self.positions]
        self.print_dataset_info()

    def find_data_files(self):
        data_file, gt_file = None, None
        for file in os.listdir(self.dataset_folder):
            if file.endswith('_data.mat') or file.endswith('_data.tif'):
                data_file = file
            elif file.endswith('_gt.mat') or file.endswith('_gt.tif'):
                gt_file = file
        if not data_file or not gt_file:
            raise FileNotFoundError("Required _data and _gt files not found.")
        return data_file, gt_file

    def load_data(self):
        data_path = os.path.join(self.dataset_folder, self.data_file)
        gt_path = os.path.join(self.dataset_folder, self.gt_file)
        
        if data_path.endswith('.mat'):
            data_dict = scipy.io.loadmat(data_path)
            gt_dict = scipy.io.loadmat(gt_path)
            data = data_dict['data']
            gt = gt_dict['gt']
            
            # FIX: Better handling of data orientation
            if data.ndim == 3:
                # Assume smallest dimension is channels
                if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                    pass  # Already (bands, H, W)
                elif data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
                    data = np.transpose(data, (2, 0, 1))  # (H, W, bands) -> (bands, H, W)
                else:
                    print(f"Warning: Ambiguous data shape {data.shape}, assuming channels-first")
        else:
            with rasterio.open(data_path) as src:
                data = src.read()  # Already (bands, H, W)
            with rasterio.open(gt_path) as src:
                gt = src.read(1)
        
        # FIX: Apply PCA if requested
        if self.dim_reduction_method == "pca" and self.num_pca_bands and self.num_pca_bands < data.shape[0]:
            bands, h, w = data.shape
            flat = data.reshape(bands, -1).T
            pca = PCA(n_components=self.num_pca_bands)
            reduced = pca.fit_transform(flat)
            data = reduced.T.reshape(self.num_pca_bands, h, w)
        
        return data, gt

    def create_patch_positions(self):
        bands, height, width = self.data.shape
        half = self.patch_size // 2
        positions = []
        for i in range(0, height - self.patch_size + 1, self.stride):
            for j in range(0, width - self.patch_size + 1, self.stride):
                label = self.gt[i + half, j + half]
                if label == 0:
                    continue
                # FIX: Store original label (1-indexed) for correct printing
                positions.append((i, j, int(label)))
        return positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        i, j, label = self.positions[idx]
        patch = self.data[:, i:i+self.patch_size, j:j+self.patch_size]
        
        if self.band_indices is not None:
            patch = patch[self.band_indices, :, :]
        
        patch = normalize_hyperspectral_data(patch)
        
        # FIX: Apply maxpool before axis transformation if using channels_first
        if self.dim_reduction_method == "maxpool" and self.band_axis == "channels_first":
            # Maxpool expects (C, H, W), patch is currently (bands, H, W)
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)  # Add batch dim
            patch_tensor = torch.nn.functional.max_pool2d(patch_tensor, kernel_size=self.maxpool_kernel)
            patch = patch_tensor.squeeze(0).numpy()  # Remove batch dim
        
        # Now handle axis transformation
        if self.band_axis == 'channels_first':
            pass  # Already in correct format
        elif self.band_axis == 'channels_last':
            patch = np.transpose(patch, (1, 2, 0))
        
        # Add channel dimension if requested
        if self.use_channel_dim:
            patch = patch.reshape((1,) + patch.shape)
        
        t_patch = torch.from_numpy(patch).float()
        
        # FIX: Return 0-indexed label for training
        return t_patch, label - 1

    def print_dataset_info(self):
        unique_classes = np.unique(self.gt)
        nonzero_classes = unique_classes[unique_classes != 0]
        
        print("\n=== Dataset Summary ===")
        print(f"Patch size: {self.patch_size}x{self.patch_size} | Stride: {self.stride}")
        print(f"Reduction: {self.dim_reduction_method}", end="")
        if self.dim_reduction_method == "pca":
            print(f" (bands={self.num_pca_bands})", end="")
        elif self.dim_reduction_method == "maxpool":
            print(f" (kernel={self.maxpool_kernel})", end="")
        print(f"\nClasses: {len(nonzero_classes)} | Total samples: {len(self.positions)}")
        print("="*23 + "\n")


def normalize_hyperspectral_data(data):
    bands, h, w = data.shape
    flat = data.reshape(bands, -1)
    norm = np.zeros_like(flat, dtype=np.float32)
    for i in range(bands):
        band = flat[i]
        mn, mx = band.min(), band.max()
        if mx > mn:
            norm[i] = (band - mn) / (mx - mn)
        else:
            norm[i] = 0
    return norm.reshape(bands, h, w)


def get_next_run_number(results_dir):
    if not os.path.exists(results_dir):
        return 1
    
    existing_runs = [d for d in os.listdir(results_dir) if d.startswith('run_') and os.path.isdir(os.path.join(results_dir, d))]
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
    """
    Create a run directory for the experiment.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        run_number: Optional specific run number to use (for multiple runs)
    
    Returns:
        run_dir: Path to the run directory
        run_number: The run number used
    """
    results_base = "results"
    dataset_dir = os.path.join(results_base, dataset_name)
    model_dir = os.path.join(dataset_dir, model_name)
    
    if run_number is None:
        run_number = get_next_run_number(model_dir)
    
    run_dir = os.path.join(model_dir, f"run_{run_number}")
    
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir, run_number


def save_config(run_dir, config):
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved: {config_path}")


def save_checkpoint(model, run_dir, epoch):
    old_checkpoints = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    for old_ckpt in old_checkpoints:
        os.remove(os.path.join(run_dir, old_ckpt))
    
    checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✓ Checkpoint saved: epoch_{epoch}.pth")


def save_model(model, run_dir, name='best_model', verbose=True):
    model_path = os.path.join(run_dir, f'{name}.pth')
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f"✓ Model saved: {name}.pth")


def load_model(model, model_path, device='cuda'):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from: {model_path}")
        return True
    else:
        print(f"Model file not found: {model_path}")
        return False


def update_results_excel(dataset_name, model_name, run_number, results_dict):
    results_base = "results"
    dataset_dir = os.path.join(results_base, dataset_name)
    model_dir = os.path.join(dataset_dir, model_name)
    excel_path = os.path.join(model_dir, 'results_summary.xlsx')
    
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:
        df = pd.DataFrame()
    
    new_row = pd.DataFrame([results_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_excel(excel_path, index=False)
    print(f"Results updated: {excel_path}")


def setup_experiment(cfg, run_number=None):
    """
    Setup complete experiment: dataset, dataloaders, and run directory.
    Returns all necessary components for training.
    
    Args:
        cfg: Configuration object
        run_number: Optional run number to use (for multiple runs)
    """
    folder = cfg.get('dataset.folder')
    dataset_name = folder.split('/')[-1]
    model_name = cfg.get('model.name', 'DBCTNet')
    
    # Standardize keys
    keys = ChangeKeys(folder)
    keys.to("data", "gt")
    
    # Get config parameters
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
    band_axis = cfg.get('preprocessing.band_axis', 'channels_first')
    band_indices = cfg.get('preprocessing.band_indices')
    
    # Create run directory (use provided run_number if available)
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
    
    # Prepare dataset
    split_result = prepare_dataset(
        folder, patch_size, stride=stride,
        split_ratios=split_ratios if split_samples_count is None else (0.8, 0.2),
        split_samples_count=split_samples_count,
        dim_reduction_method=dim_reduction_method,
        num_pca_bands=num_pca_bands,
        maxpool_kernel=maxpool_kernel,
        use_channel_dim=use_channel_dim,
        band_indices=band_indices,
        band_axis=band_axis
    )
    
    num_workers = cfg.get('training.num_workers', 4)
    
    # Get dataloaders
    if len(split_result) == 4:
        train_idx, test_idx, num_classes, dataset = split_result
        val_idx = None
        train_loader, test_loader = get_dataloaders(dataset, train_idx, test_idx, batch_size=batch_size, num_workers=num_workers)
        val_loader = None
    else:
        train_idx, val_idx, test_idx, num_classes, dataset = split_result
        train_loader, val_loader, test_loader = get_dataloaders(dataset, train_idx, test_idx, val_idx, batch_size=batch_size, num_workers=num_workers)
    
    # Print class statistics
    labels = [label for _, _, label in dataset.positions]
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


def post_training_analysis(trained_model, predictions, targets, best_epoch, training_time,
                           dataset, device, run_dir, dataset_name, model_name, run_number,
                           num_epochs, patch_size, batch_size, split_ratios, split_samples_count,
                           train_idx, val_idx, test_idx, num_classes, cmap='tab20', 
                           show_colorbar=False, dpi=300):
    """
    Perform all post-training analysis: calculate metrics, generate classification map, 
    and update Excel results.
    """
    # Calculate metrics
    oa = accuracy_score(targets, predictions) * 100
    kappa = cohen_kappa_score(targets, predictions)
    
    per_class_acc = []
    for class_id in range(num_classes):
        mask = np.array(targets) == class_id
        if mask.sum() > 0:
            class_acc = accuracy_score(np.array(targets)[mask], np.array(predictions)[mask]) * 100
            per_class_acc.append(class_acc)
    aa = np.mean(per_class_acc)
    
    generate_classification_map(
        model=trained_model,
        dataset=dataset,
        device=device,
        run_dir=run_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        run_number=run_number,
        cmap=cmap,
        show_colorbar=show_colorbar,
        dpi=dpi
    )
    
    # Prepare Excel results
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    results_dict = {
        'Run': run_number,
        'Epochs': num_epochs,
        'Best_Epoch': best_epoch,
        'Patch_Size': patch_size,
        'Batch_Size': batch_size,
        'Split_Config': str(split_ratios) if split_samples_count is None else str(split_samples_count),
        'Train_Samples': len(train_idx),
        'Val_Samples': len(val_idx) if val_idx else 0,
        'Test_Samples': len(test_idx),
        'OA': round(oa, 2),
        'AA': round(aa, 2),
        'Kappa': round(kappa, 4),
        'Training_Time': time_str
    }
    
    # Add class-wise accuracies
    for i, class_acc in enumerate(per_class_acc, start=1):
        results_dict[f'Class_{i}_Acc'] = round(class_acc, 2)
    
    update_results_excel(dataset_name, model_name, run_number, results_dict)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Results saved in: {run_dir}")
    print("="*60)


def split_data(*args, labels, random_state=42):
    indices = list(range(len(labels)))
    
    if len(args) == 2:
        train_ratio, test_ratio = args
        if not np.isclose(train_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0. Got {train_ratio + test_ratio}")
        
        train_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=random_state, stratify=labels
        )
        print(f"Split: Train={train_ratio*100:.0f}% | Test={test_ratio*100:.0f}%")
        return train_idx, test_idx
    
    elif len(args) == 3:
        train_ratio, val_ratio, test_ratio = args
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0. Got {train_ratio + val_ratio + test_ratio}")
        
        train_idx, temp_idx = train_test_split(
            indices, test_size=(val_ratio + test_ratio), random_state=random_state, stratify=labels
        )
        
        temp_labels = [labels[i] for i in temp_idx]
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1 - val_size_adjusted), random_state=random_state, stratify=temp_labels
        )
        print(f"Split: Train={train_ratio*100:.0f}% | Val={val_ratio*100:.0f}% | Test={test_ratio*100:.0f}%")
        return train_idx, val_idx, test_idx
    
    else:
        raise ValueError(f"Expected 2 or 3 split ratios, got {len(args)}")


def split_samples(*args, labels, random_state=42):
    np.random.seed(random_state)
    
    # Group indices by class
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Shuffle indices within each class
    for label in class_indices:
        np.random.shuffle(class_indices[label])
    
    if len(args) == 1:
        # Two-way split: train_samples, rest
        train_samples = args[0]
        train_idx, test_idx = [], []
        
        min_samples = min(len(indices) for indices in class_indices.values())
        if train_samples > min_samples:
            raise ValueError(f"Requested {train_samples} samples but smallest class has only {min_samples} samples")
        
        for label, indices in class_indices.items():
            train_idx.extend(indices[:train_samples])
            test_idx.extend(indices[train_samples:])
        
        print(f"Split: Train={train_samples} samples/class | Test=remaining samples")
        print(f"Total: Train={len(train_idx)} | Test={len(test_idx)}")
        return train_idx, test_idx
    
    elif len(args) == 2:
        # Three-way split: train_samples, val_samples, rest
        train_samples, val_samples = args
        train_idx, val_idx, test_idx = [], [], []
        
        min_samples = min(len(indices) for indices in class_indices.values())
        if train_samples + val_samples > min_samples:
            raise ValueError(f"Requested {train_samples + val_samples} samples but smallest class has only {min_samples} samples")
        
        for label, indices in class_indices.items():
            train_idx.extend(indices[:train_samples])
            val_idx.extend(indices[train_samples:train_samples + val_samples])
            test_idx.extend(indices[train_samples + val_samples:])
        
        print(f"Split: Train={train_samples} samples/class | Val={val_samples} samples/class | Test=remaining samples")
        print(f"Total: Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")
        return train_idx, val_idx, test_idx
    
    else:
        raise ValueError(f"Expected 1 or 2 sample counts, got {len(args)}")


def prepare_dataset(dataset_folder, patch_size=11, stride=1, split_ratios=(0.8, 0.2), 
                    split_samples_count=None, random_state=42,
                    dim_reduction_method=None, num_pca_bands=None, maxpool_kernel=2,
                    use_channel_dim=True, band_indices=None, band_axis='channels_first'):
    dataset = HyperspectralDataset(
        dataset_folder, patch_size, stride, 
        dim_reduction_method=dim_reduction_method, 
        num_pca_bands=num_pca_bands, 
        maxpool_kernel=maxpool_kernel,
        use_channel_dim=use_channel_dim,
        band_indices=band_indices,
        band_axis=band_axis
    )
    
    # Use original labels (1-indexed) for stratification
    labels = [label for _, _, label in dataset.positions]
    num_classes = len(set(labels))
    
    # Choose splitting method
    if split_samples_count is not None:
        split_result = split_samples(*split_samples_count, labels=labels, random_state=random_state)
    else:
        split_result = split_data(*split_ratios, labels=labels, random_state=random_state)
    
    if len(split_result) == 2:
        train_idx, test_idx = split_result
        return train_idx, test_idx, num_classes, dataset
    else:
        train_idx, val_idx, test_idx = split_result
        return train_idx, val_idx, test_idx, num_classes, dataset


def get_dataloaders(dataset, train_idx, test_idx, val_idx=None, batch_size=32, num_workers=4):
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
        return train_loader, val_loader, test_loader
    
    return train_loader, test_loader


def print_class_stats(labels, train_idx, test_idx, val_idx=None, num_classes=None):
    if num_classes is None:
        num_classes = len(set(labels))
    
    def get_distribution(indices):
        subset = [labels[i] for i in indices]
        ctr = Counter(subset)
        return ctr
    
    train_counter = get_distribution(train_idx)
    test_counter = get_distribution(test_idx)
    
    # Get only classes that actually exist in the dataset
    all_classes = sorted(set(labels))
    
    total_train = sum(train_counter.values())
    total_test = sum(test_counter.values())
    
    if val_idx is not None:
        val_counter = get_distribution(val_idx)
        total_val = sum(val_counter.values())
        total_all = total_train + total_val + total_test
        
        header = f"{'Class':^7} | {'Train':^10} | {'Val':^10} | {'Test':^10} | {'Total':^10}"
        print("\n" + "="*len(header))
        print(header)
        print("-"*len(header))
        
        # FIX: Labels are already 1-indexed from positions, so print directly
        for class_id in all_classes:
            train_count = train_counter.get(class_id, 0)
            val_count = val_counter.get(class_id, 0)
            test_count = test_counter.get(class_id, 0)
            total_count = train_count + val_count + test_count
            print(f"{class_id:^7} | {train_count:^10} | {val_count:^10} | {test_count:^10} | {total_count:^10}")
        
        print("-"*len(header))
        print(f"{'TOTAL':^7} | {total_train:^10} | {total_val:^10} | {total_test:^10} | {total_all:^10}")
        print("="*len(header) + "\n")
    else:
        total_all = total_train + total_test
        
        header = f"{'Class':^7} | {'Train':^10} | {'Test':^10} | {'Total':^10}"
        print("\n" + "="*len(header))
        print(header)
        print("-"*len(header))
        
        # FIX: Labels are already 1-indexed from positions, so print directly
        for class_id in all_classes:
            train_count = train_counter.get(class_id, 0)
            test_count = test_counter.get(class_id, 0)
            total_count = train_count + test_count
            print(f"{class_id:^7} | {train_count:^10} | {test_count:^10} | {total_count:^10}")
        
        print("-"*len(header))
        print(f"{'TOTAL':^7} | {total_train:^10} | {total_test:^10} | {total_all:^10}")
        print("="*len(header) + "\n")


def train_model(model, train_loader, val_loader, test_loader, num_epochs=50, 
                learning_rate=0.001, device='cuda', patience=10, run_dir=None, checkpoint_interval=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if val_loader is not None:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    best_epoch = 0
    
    training_log = []
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Training on: {device}")
    print(f"Using validation: {'Yes' if val_loader is not None else 'No'}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # ============ TRAINING ============
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # ============ VALIDATION (if provided) ============
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            with torch.no_grad():
                for inputs, targets in val_pbar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    predicted = outputs.argmax(dim=1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100. * val_correct / val_total:.2f}%'
                    })
            
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                if run_dir:
                    save_model(model, run_dir, 'best_model', verbose=False)
            else:
                epochs_without_improvement += 1
            
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            training_log.append(log_entry)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"No improve: {epochs_without_improvement}/{patience}")
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                print("Restoring best model...")
                model.load_state_dict(best_model_state)
                break
        else:
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                if run_dir:
                    save_model(model, run_dir, 'best_model', verbose=False)
            
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc
            }
            training_log.append(log_entry)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
        
        if run_dir and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, run_dir, epoch + 1)
    
    training_time = time.time() - start_time
    
    if run_dir:
        save_model(model, run_dir, 'final_model', verbose=True)
    
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}\n")
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    test_pbar = tqdm(test_loader, desc="Testing", leave=True)
    with torch.no_grad():
        for inputs, targets in test_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            test_pbar.set_postfix({
                'acc': f'{100. * test_correct / test_total:.2f}%'
            })
    
    test_acc = 100. * test_correct / test_total
    
    # Calculate class-wise accuracies
    num_classes = len(set(all_targets))
    class_wise_acc = []
    for class_id in range(num_classes):
        mask = np.array(all_targets) == class_id
        if mask.sum() > 0:
            class_acc = accuracy_score(np.array(all_targets)[mask], np.array(all_preds)[mask]) * 100
            class_wise_acc.append((class_id, class_acc, mask.sum()))
        else:
            class_wise_acc.append((class_id, 0.0, 0))
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"\n{'='*60}")
    print("Class-wise Test Accuracies:")
    print(f"{'='*60}")
    for class_id, acc, count in class_wise_acc:
        print(f"Class {class_id+1}: {acc:.2f}% ({count} samples)")
    print(f"{'='*60}\n")
    
    if run_dir:
        log_path = os.path.join(run_dir, 'training_log.txt')
        with open(log_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Training Log\n")
            f.write("="*60 + "\n\n")
            for entry in training_log:
                f.write(str(entry) + '\n')
            f.write("\n" + "="*60 + "\n")
            f.write("Test Results\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Test Accuracy: {test_acc:.2f}%\n\n")
            f.write("Class-wise Test Accuracies:\n")
            f.write("-"*60 + "\n")
            for class_id, acc, count in class_wise_acc:
                f.write(f"Class {class_id+1}: {acc:.2f}% \n")
            f.write("="*60 + "\n")
        
        print(f"\n{'='*60}")
        print("Models & Logs Saved:")
        print(f"{'='*60}")
        print(f"✓ best_model.pth (epoch {best_epoch})")
        print(f"✓ final_model.pth")
        
        checkpoints = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch_')]
        if checkpoints:
            print(f"✓ {checkpoints[0]}")
        print(f"✓ training_log.txt (with class-wise accuracies)")
        print(f"{'='*60}\n")
    
    return model, all_preds, all_targets, best_epoch, training_time


def generate_classification_map(model, dataset, device='cuda', run_dir=None, dataset_name=None, model_name=None, run_number=None, cmap='tab20', show_colorbar=False, dpi=300):
    print("\n" + "="*60)
    print("Generating Classification Map...")
    print("="*60)
    
    model.eval()
    bands, height, width = dataset.data.shape
    half_patch = dataset.patch_size // 2
    
    # Initialize prediction map
    prediction_map = np.zeros((height, width), dtype=np.int32)
    
    # Pad data for edge handling
    padded_data = np.pad(
        dataset.data, 
        ((0, 0), (half_patch, half_patch), (half_patch, half_patch)), 
        mode='reflect'
    )
    
    batch_size = 64
    batch_patches = []
    batch_positions = []
    
    # Create progress bar for all pixels
    total_pixels = height * width
    pbar = tqdm(total=total_pixels, desc="Classifying pixels", unit="px")
    
    for i in range(height):
        for j in range(width):
            # Extract patch
            patch = padded_data[:, i:i+dataset.patch_size, j:j+dataset.patch_size]
            
            # Normalize and prepare patch
            normalized_patch = normalize_hyperspectral_data(patch)
            
            # Match dataset format
            if dataset.use_channel_dim:
                normalized_patch = normalized_patch.reshape((1,) + normalized_patch.shape)
            
            batch_patches.append(normalized_patch)
            batch_positions.append((i, j))
            
            # Process batch
            if len(batch_patches) == batch_size or (i == height - 1 and j == width - 1):
                if batch_patches:
                    batch_count = len(batch_patches)  # Store count before clearing
                    
                    patches_tensor = torch.tensor(
                        np.array(batch_patches), 
                        dtype=torch.float32
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(patches_tensor)
                        preds = outputs.argmax(dim=1).cpu().numpy()
                    
                    # Assign predictions
                    for idx, (y, x) in enumerate(batch_positions):
                        prediction_map[y, x] = preds[idx] + 1
                    
                    # Update progress bar with the batch count
                    pbar.update(batch_count)
                    
                    batch_patches = []
                    batch_positions = []
    
    pbar.close()
    
    # Save and visualize
    if run_dir:
        output_path = os.path.join(run_dir, 'classification_map.png')
    else:
        output_path = f"classification_map_{dataset.dataset_folder.split('/')[-1]}.png"
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(prediction_map, cmap=cmap)
    
    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Class')
    
    ax.axis('off')  # Remove axes
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✓ Classification map saved: classification_map.png")
    
    return prediction_map



