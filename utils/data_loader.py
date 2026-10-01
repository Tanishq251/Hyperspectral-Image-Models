import os
import numpy as np
import scipy.io
import h5py
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import torch
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files

class DatasetLoader:
    """Load HSI datasets from HuggingFace Hub (auto-downloads to cache)"""
    
    REPO_ID = "Tanishq165/HSI_Datasets"
    REPO_TYPE = "dataset"

    # Dataset-specific band slices to drop non-HSI channels
    BAND_SLICES = {
        'Houston18': slice(0, 48),  # Drop extra LiDAR/DSM + mask bands
    }

    def __init__(self, use_cache=True):
        """
        Initialize dataset loader
        
        Args:
            use_cache: If True, download datasets to HuggingFace cache (~/.cache/huggingface)
                      If False, use local 'dataset' folder
        """
        self.use_cache = use_cache
    
    def list_available_datasets(self):
        """List all available datasets in the HF repo"""
        try:
            files = list_repo_files(repo_id=self.REPO_ID, repo_type=self.REPO_TYPE)
            
            # Extract dataset names (directories)
            datasets = set()
            for file_path in files:
                parts = file_path.split('/')
                if len(parts) > 1 and parts[0] != '.':
                    datasets.add(parts[0])
            
            return sorted(list(datasets))
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return []
    
    def download_dataset(self, dataset_name, verbose=True):
        """
        Download a specific dataset to cache
        
        Args:
            dataset_name: Name of the dataset
            verbose: Print download progress
            
        Returns:
            str: Path to the downloaded dataset folder
        """
        try:
            if verbose:
                print(f"Downloading {dataset_name} to cache...")
            
            # Download entire dataset folder to HF cache
            cache_path = snapshot_download(
                repo_id=self.REPO_ID,
                repo_type=self.REPO_TYPE,
                allow_patterns=f"{dataset_name}/*",
                cache_dir=None,  # Use default HF cache
            )
            dataset_path = os.path.join(cache_path, dataset_name)
            
            if verbose:
                print(f"{dataset_name} cached at: {dataset_path}")
            return dataset_path
        
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return None
    
    def load_dataset(self, dataset_name):
        """
        Load a dataset (downloads to cache if not present)
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            tuple: (data, gt) numpy arrays
        """
        # Get dataset path (downloads to cache if needed)
        dataset_path = self.download_dataset(dataset_name, verbose=True)

        if dataset_path is None:
            raise ValueError(f"Failed to download dataset '{dataset_name}'")

        data, gt = self._load_mat_files(dataset_path)

        # Drop extra non-HSI channels (e.g., LiDAR/DSM concatenated to HSI cube)
        if dataset_name in self.BAND_SLICES:
            sl = self.BAND_SLICES[dataset_name]
            # Assumes bands are the last dimension after _load_mat_files
            if data.ndim == 3:
                data = data[..., sl]
                print(f"  Sliced bands to {data.shape[-1]} for {dataset_name}")

        return data, gt
    
    def _load_mat_files(self, dataset_path):
        mat_files = [f for f in os.listdir(dataset_path) if f.endswith('.mat')]
        
        if len(mat_files) < 1:
            raise ValueError(f"No .mat files found in {dataset_path}")
        
        data, gt = None, None
        
        for mat_file in mat_files:
            file_path = os.path.join(dataset_path, mat_file)
            mat_dict = self._load_mat_file(file_path)
            valid_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
            
            # Check filename to determine if it's data or gt file
            if mat_file.endswith('_gt.mat') or mat_file.endswith('gt.mat'):
                # This is a gt file
                if gt is None and valid_keys:
                    gt = mat_dict[valid_keys[0]]
                    if gt.ndim == 3:
                        gt = np.squeeze(gt)
            elif mat_file.endswith('_data.mat') or mat_file.endswith('data.mat'):
                # This is a data file
                if data is None and valid_keys:
                    data = mat_dict[valid_keys[0]]
        
        if data is None or gt is None:
            raise ValueError(f"Could not find both data and GT in {dataset_path}")
        
        return data, gt
    
    def _load_mat_file(self, file_path):
        """Load .mat file, handling both v7.3 (HDF5) and older formats"""
        try:
            # Try scipy first (works for v7.0 and earlier)
            return scipy.io.loadmat(file_path)
        except NotImplementedError:
            # Fall back to h5py for v7.3 files
            mat_dict = {}
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    mat_dict[key] = np.array(f[key])
            return mat_dict
    
    def _find_data_key(self, mat_dict):
        valid_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
        # Try common data key names first
        for key in ['data', 'Data', 'DATA']:
            if key in valid_keys:
                return key
        # Then try any key that doesn't contain 'gt'
        for key in valid_keys:
            if 'gt' not in key.lower():
                return key
        print(f"Could not find data key. Available keys: {valid_keys}")
        return None
    
    def _find_gt_key(self, mat_dict):
        valid_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
        # Try common gt key names first
        for key in ['gt', 'GT', 'Gt']:
            if key in valid_keys:
                return key
        # Then try any key that contains 'gt'
        for key in valid_keys:
            if 'gt' in key.lower():
                return key
        print(f"Could not find gt key. Available keys: {valid_keys}")
        return None

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

def test_mat_keys(dataset_path):
    """Test .mat files for 'data' and 'gt' keys, print all keys if not found"""
    mat_files = [f for f in os.listdir(dataset_path) if f.endswith('.mat')]
    loader = DatasetLoader()
    
    for mat_file in mat_files:
        file_path = os.path.join(dataset_path, mat_file)
        mat_dict = loader._load_mat_file(file_path)
        valid_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
        
        print(f"\n--- {mat_file} ---")
        print(f"All keys: {valid_keys}")
        
        if 'data' in valid_keys:
            print(f"'data' key found - shape: {mat_dict['data'].shape}")
        else:
            print(f"'data' key NOT found")
        
        if 'gt' in valid_keys:
            print(f"'gt' key found - shape: {mat_dict['gt'].shape}")
        else:
            print(f"'gt' key NOT found")

class HyperspectralDataset(Dataset):
    def __init__(self, data, gt, patch_size=11, stride=1,
                 dim_reduction_method=None, num_pca_bands=None,
                 maxpool_kernel=2, use_channel_dim=True, band_indices=None,
                 verbose=True, pca_random_state=42):

        self.patch_size = patch_size
        self.stride = stride
        self.dim_reduction_method = dim_reduction_method
        self.num_pca_bands = num_pca_bands
        self.maxpool_kernel = maxpool_kernel
        self.use_channel_dim = use_channel_dim
        self.band_indices = band_indices
        self.pca_random_state = pca_random_state

        # First process GT to ensure it's 2D
        self.gt = self._process_gt(gt)
        
        # Then process data to match GT dimensions
        self.data = self._process_data_to_match_gt(data, self.gt, verbose)

        # Normalize data
        self.data = normalize_hyperspectral_data(self.data)

        # Apply PCA if requested
        if self.dim_reduction_method == "pca" and self.num_pca_bands:
            self.data = self._apply_pca(self.data)

        # Create patch positions
        self.positions = self._create_patch_positions()
        self.targets = [label for _, _, label in self.positions]

        # Create label mapping for non-contiguous labels
        unique_labels = sorted(set(self.targets))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        if verbose:
            self._print_dataset_info()
    
    def _process_gt(self, gt):
        """Process GT to ensure it's 2D"""
        if gt.ndim == 2:
            return gt
        elif gt.ndim == 3:
            gt_2d = np.squeeze(gt)
            if gt_2d.ndim != 2:
                raise ValueError(f"Cannot convert GT shape {gt.shape} to 2D")
            return gt_2d
        else:
            raise ValueError(f"Expected 2D or 3D GT, got shape {gt.shape}")
    
    def _process_data_to_match_gt(self, data, gt, verbose):
        """
        Process data to match GT dimensions.
        Tries to orient data as (bands, H, W) where (H, W) matches GT shape.
        """
        if data.ndim != 3:
            raise ValueError(f"Expected 3D data, got shape {data.shape}")
        
        gt_h, gt_w = gt.shape   
        
        if verbose:
            print(f"Original data shape: {data.shape}")
            print(f"GT shape: {gt.shape}")
        
        # Try all 6 possible permutations to find one that matches GT
        permutations = [
            (0, 1, 2),  # (bands, H, W) or (D1, D2, D3)
            (0, 2, 1),  # (bands, W, H) or (D1, D3, D2)
            (1, 0, 2),  # (H, bands, W) or (D2, D1, D3)
            (1, 2, 0),  # (H, W, bands) or (D2, D3, D1)
            (2, 0, 1),  # (W, bands, H) or (D3, D1, D2)
            (2, 1, 0),  # (W, H, bands) or (D3, D2, D1)
        ]
        
        best_match = None
        best_perm = None
        
        for perm in permutations:
            test_data = np.transpose(data, perm)
            test_h, test_w = test_data.shape[1], test_data.shape[2]
            
            # Check if spatial dimensions match GT
            if test_h == gt_h and test_w == gt_w:
                # Perfect match!
                if verbose:
                    print(f"Found exact match with permutation {perm}")
                    print(f"  Data shape after transpose: {test_data.shape}")
                return test_data
            
            # Check if this permutation has bands as smallest dimension
            # This is a good heuristic for HSI data
            if test_data.shape[0] < test_data.shape[1] and test_data.shape[0] < test_data.shape[2]:
                if best_match is None:
                    best_match = test_data
                    best_perm = perm
        
        # If no exact match, use the best heuristic match
        if best_match is not None:
            if verbose:
                print(f"Warning: No exact spatial match found. Using permutation {best_perm}")
                print(f"  Data shape: {best_match.shape}, GT shape: {gt.shape}")
                print(f"  Note: Spatial dimensions don't match. Using minimum overlap.")
            return best_match
        
        # Last resort: assume data is already in correct format
        if verbose:
            print(f"Warning: Could not determine correct orientation. Using original shape.")
            print(f"  Data shape: {data.shape}, GT shape: {gt.shape}")
        return data
    
    def _apply_pca(self, data):
        if self.num_pca_bands >= data.shape[0]:
            return data
        
        bands, h, w = data.shape
        flat = data.reshape(bands, -1).T
        pca = PCA(n_components=self.num_pca_bands, random_state=self.pca_random_state)
        reduced = pca.fit_transform(flat)
        return reduced.T.reshape(self.num_pca_bands, h, w)
    
    def _create_patch_positions(self):
        bands, height, width = self.data.shape
        gt_height, gt_width = self.gt.shape       
        half = self.patch_size // 2
        positions = []
        
        # Use the minimum of data and gt dimensions
        max_height = min(height, gt_height)
        max_width = min(width, gt_width)
        
        for i in range(0, max_height - self.patch_size + 1, self.stride):
            for j in range(0, max_width - self.patch_size + 1, self.stride):
                label = self.gt[i + half, j + half]
                
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

        if self.band_indices is not None:
            patch = patch[self.band_indices, :, :]

        if self.dim_reduction_method == "maxpool":
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
            patch_tensor = torch.nn.functional.max_pool2d(patch_tensor, kernel_size=self.maxpool_kernel)
            patch = patch_tensor.squeeze(0).numpy()

        if self.use_channel_dim:
            patch = patch.reshape((1,) + patch.shape)

        t_patch = torch.from_numpy(patch).float()

        # Map original label to contiguous index
        label_idx = self.label_to_idx[label]

        return t_patch, label_idx
    
    def _print_dataset_info(self):
        unique_classes = np.unique(self.gt)
        nonzero_classes = unique_classes[unique_classes != 0]
        
        print("\n=== Dataset Summary ===")
        print(f"Data shape: {self.data.shape}")
        print(f"GT shape: {self.gt.shape}")
        print(f"Patch size: {self.patch_size}x{self.patch_size} | Stride: {self.stride}")
        print(f"Reduction: {self.dim_reduction_method}", end="")
        if self.dim_reduction_method == "pca":
            print(f" (bands={self.num_pca_bands})", end="")
        elif self.dim_reduction_method == "maxpool":
            print(f" (kernel={self.maxpool_kernel})", end="")
        print(f"\nClasses: {len(nonzero_classes)} | Total samples: {len(self.positions)}")
        print("="*23 + "\n")