import os
import numpy as np
import scipy.io
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import torch


class DatasetLoader:
    """Smart dataset loader that auto-detects .mat file keys and loads data"""
    
    def __init__(self, datasets_folder):
        self.datasets_folder = datasets_folder
        self.available_datasets = self._scan_datasets()
    
    def _scan_datasets(self):
        """Scan the datasets folder for available datasets"""
        if not os.path.exists(self.datasets_folder):
            raise FileNotFoundError(f"Datasets folder not found: {self.datasets_folder}")
        
        datasets = {}
        for item in os.listdir(self.datasets_folder):
            item_path = os.path.join(self.datasets_folder, item)
            if os.path.isdir(item_path):
                # Check if it contains .mat files
                mat_files = [f for f in os.listdir(item_path) if f.endswith('.mat')]
                if mat_files:
                    datasets[item] = item_path
        
        return datasets
    
    def load_dataset(self, dataset_name):
        """Load dataset by name from the datasets folder"""
        if dataset_name not in self.available_datasets:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available: {list(self.available_datasets.keys())}"
            )
        
        dataset_path = self.available_datasets[dataset_name]
        return self._load_mat_files(dataset_path)
    
    def _load_mat_files(self, dataset_path):
        """Load data and gt from .mat files with auto key detection"""
        mat_files = [f for f in os.listdir(dataset_path) if f.endswith('.mat')]
        
        if len(mat_files) < 1:
            raise ValueError(f"No .mat files found in {dataset_path}")
        
        data, gt = None, None
        
        for mat_file in mat_files:
            file_path = os.path.join(dataset_path, mat_file)
            mat_dict = scipy.io.loadmat(file_path)
            
            # Auto-detect keys
            data_key = self._find_data_key(mat_dict)
            gt_key = self._find_gt_key(mat_dict)
            
            # Prioritize loading data if not already loaded
            if data_key and data is None:
                candidate_data = mat_dict[data_key]
                # Data should be 3D (bands, H, W) or (H, W, bands)
                if candidate_data.ndim == 3:
                    data = candidate_data
                    print(f"✓ Loaded data from '{mat_file}' using key '{data_key}' - shape: {data.shape}")
            
            # Prioritize loading GT if not already loaded
            if gt_key and gt is None:
                candidate_gt = mat_dict[gt_key]
                # GT should be 2D (H, W)
                if candidate_gt.ndim == 2:
                    gt = candidate_gt
                    print(f"✓ Loaded GT from '{mat_file}' using key '{gt_key}' - shape: {gt.shape}")
                elif candidate_gt.ndim == 3:
                    # If 3D, squeeze to 2D
                    gt = np.squeeze(candidate_gt)
                    if gt.ndim == 2:
                        print(f"✓ Loaded GT from '{mat_file}' using key '{gt_key}' - shape: {gt.shape} (squeezed from 3D)")
        
        if data is None or gt is None:
            raise ValueError(f"Could not find both data and GT in {dataset_path}. Found: data={data is not None}, gt={gt is not None}")
        
        return data, gt
    
    def _find_data_key(self, mat_dict):
        """Auto-detect data key from common variations"""
        data_variations = [
            'data', 'Data', 'DATA', 
            'image', 'Image', 'IMAGE',
            'hsi', 'HSI', 
            'hyperspectral', 'Hyperspectral',
            'cube', 'Cube'
        ]
        
        valid_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
        
        for key in data_variations:
            if key in valid_keys:
                return key
        
        return None
    
    def _find_gt_key(self, mat_dict):
        """Auto-detect ground truth key from common variations"""
        gt_variations = [
            'gt', 'GT', 'Gt',
            'label', 'Label', 'LABEL',
            'labels', 'Labels', 'LABELS',
            'ground_truth', 'groundtruth', 'GroundTruth',
            'mask', 'Mask', 'MASK'
        ]
        
        valid_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
        
        for key in gt_variations:
            if key in valid_keys:
                return key
        
        return None


def normalize_hyperspectral_data(data):
    """Normalize hyperspectral data band-wise"""
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


class HyperspectralDataset(Dataset):
    """PyTorch Dataset for hyperspectral image classification"""
    
    def __init__(self, data, gt, patch_size=11, stride=1,
                 dim_reduction_method=None, num_pca_bands=None,
                 maxpool_kernel=2, use_channel_dim=True, band_indices=None,
                 verbose=True):
        
        self.patch_size = patch_size
        self.stride = stride
        self.dim_reduction_method = dim_reduction_method
        self.num_pca_bands = num_pca_bands
        self.maxpool_kernel = maxpool_kernel
        self.use_channel_dim = use_channel_dim
        self.band_indices = band_indices
        
        # Process data orientation
        self.data = self._process_data_orientation(data)
        
        # Ensure GT is 2D
        self.gt = self._process_gt_orientation(gt)
        
        # Apply dimensionality reduction if requested
        if self.dim_reduction_method == "pca" and self.num_pca_bands:
            self.data = self._apply_pca(self.data)
        
        # Create patch positions
        self.positions = self._create_patch_positions()
        self.targets = [label for _, _, label in self.positions]
        
        self._print_dataset_info(verbose=verbose)
    
    def _process_data_orientation(self, data):
        """Ensure data is in (bands, H, W) format"""
        if data.ndim != 3:
            raise ValueError(f"Expected 3D data, got shape {data.shape}")
        
        # Assume smallest dimension is channels
        if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
            return data  # Already (bands, H, W)
        elif data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
            return np.transpose(data, (2, 0, 1))  # (H, W, bands) -> (bands, H, W)
        else:
            print(f"Warning: Ambiguous data shape {data.shape}, assuming channels-first")
            return data
    
    def _process_gt_orientation(self, gt):
        """Ensure GT is 2D (H, W)"""
        if gt.ndim == 2:
            return gt
        elif gt.ndim == 3:
            # Squeeze out singleton dimensions
            gt_squeezed = np.squeeze(gt)
            if gt_squeezed.ndim == 2:
                print(f"GT squeezed from {gt.shape} to {gt_squeezed.shape}")
                return gt_squeezed
            else:
                raise ValueError(f"Cannot convert GT shape {gt.shape} to 2D")
        else:
            raise ValueError(f"Expected 2D or 3D GT, got shape {gt.shape}")
    
    def _apply_pca(self, data):
        """Apply PCA for dimensionality reduction"""
        if self.num_pca_bands >= data.shape[0]:
            return data
        
        bands, h, w = data.shape
        flat = data.reshape(bands, -1).T
        pca = PCA(n_components=self.num_pca_bands)
        reduced = pca.fit_transform(flat)
        return reduced.T.reshape(self.num_pca_bands, h, w)
    
    def _create_patch_positions(self):
        """Create list of valid patch positions (excluding background)"""
        bands, height, width = self.data.shape
        half = self.patch_size // 2
        positions = []
        
        for i in range(0, height - self.patch_size + 1, self.stride):
            for j in range(0, width - self.patch_size + 1, self.stride):
                label = self.gt[i + half, j + half]
                
                # Handle both scalar and array labels
                if np.isscalar(label):
                    label_val = label
                else:
                    label_val = label.item() if hasattr(label, 'item') else label
                
                if label_val == 0:
                    continue
                positions.append((i, j, int(label_val)))
        
        return positions
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        i, j, label = self.positions[idx]
        patch = self.data[:, i:i+self.patch_size, j:j+self.patch_size]
        
        # Apply band selection if specified
        if self.band_indices is not None:
            patch = patch[self.band_indices, :, :]
        
        # Normalize
        patch = normalize_hyperspectral_data(patch)
        
        # Apply maxpool if requested
        if self.dim_reduction_method == "maxpool":
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
            patch_tensor = torch.nn.functional.max_pool2d(patch_tensor, kernel_size=self.maxpool_kernel)
            patch = patch_tensor.squeeze(0).numpy()
        
        # Add channel dimension if requested
        if self.use_channel_dim:
            patch = patch.reshape((1,) + patch.shape)
        
        t_patch = torch.from_numpy(patch).float()
        
        # Return 0-indexed label for training
        return t_patch, label - 1
    
    def _print_dataset_info(self, verbose=True):
        """Print dataset summary"""
        if not verbose:
            return
            
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
