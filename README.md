# HSI_Pipeline

A modular and configuration-driven pipeline for **Hyperspectral Image (HSI) Classification**. This pipeline supports multiple deep learning models, flexible data splitting, preprocessing options, and automated result tracking.

---

## Project Structure

```
HSI_Pipeline/
├── config.yaml              # Main configuration file
├── config_loader.py         # Configuration parser
├── HSI.py                   # Core pipeline (dataset, training, analysis)
├── train.py                 # Main training script
├── DBCTnet.py              # DBCTNet model implementation
├── M3DRecNet.py            # 3DRecNet (HSIVit) model implementation
├── datasets_folder/     # Dataset directory (you create this)
│   └── <dataset_name>/
│       ├── <name>_data.mat  # Hyperspectral data file
│       └── <name>_gt.mat    # Ground truth labels file
└── results/                # Auto-generated results directory
    └── <dataset_name>/
        └── <model_name>/
            ├── run_1/
            │   ├── config.json
            │   ├── best_model.pth
            │   ├── final_model.pth
            │   ├── checkpoint_epoch_*.pth
            │   ├── training_log.txt
            │   └── classification_map.png
            ├── run_2/
            └── results_summary.xlsx  # Aggregated metrics
```

---

## Dataset Preparation

### Required Format

Your dataset must follow this naming convention:

```
dataset_folder/
└── <YourDatasetName>/
    ├── <name>_data.mat    # Hyperspectral image data
    └── <name>_gt.mat      # Ground truth labels
```

### Data File Structure

#### 1. `<name>_data.mat`
- **Variable name**: `data` (or variations: `Data`, `IMAGE`, `hsi`, `cube`, etc.)
- **Shape**: `(bands, height, width)` or `(height, width, bands)`
  - The code auto-detects orientation
- **Type**: Float/Integer array

#### 2. `<name>_gt.mat`
- **Variable name**: `gt` (or variations: `GT`, `label`, `mask`, etc.)
- **Shape**: `(height, width)`
- **Values**: Integer class labels (1, 2, 3, ..., N)
  - **0** represents unlabeled/background pixels

### Example Dataset Setup

```bash
# Create dataset directory
mkdir -p Matlab_data_format/WHU-Hi-HanChuan

# Place your files
Matlab_data_format/WHU-Hi-HanChuan/
├── WHU_data.mat
└── WHU_gt.mat
```

### Supported Data Formats

- **.mat files** (MATLAB format) - Automatically loads `data` and `gt` keys
- **.tif files** (GeoTIFF) - Loads raster data with rasterio

---

## Configuration

Edit `config.yaml` to customize your experiment:

### Dataset Settings
```yaml
dataset:
  folder: "Matlab_data_format/WHU-Hi-HanChuan"  # Path to your dataset
  patch_size: 11                                  # Spatial patch size (e.g., 11x11)
  stride: 1                                       # Sampling stride
```

### Data Splitting
```yaml
data_split:
  method: "ratio"                    # "ratio" or "samples"
  split_ratios: [0.3, 0.1, 0.6]     # [train, val, test] ratios
  split_samples: null                # Alternative: [30, 10] samples per class
  random_state: 42                   # Random seed for reproducibility
```

### Preprocessing
```yaml
preprocessing:
  dim_reduction_method: "pca"        # "pca", "maxpool", or null
  num_pca_bands: 30                  # Number of PCA components (if pca)
  maxpool_kernel: 2                  # MaxPool kernel size (if maxpool)
  use_channel_dim: true              # Add channel dimension
  band_axis: "channels_first"        # "channels_first" or "channels_last"
  band_indices: null                 # Specify band indices (optional)
```

### Model Selection
```yaml
model:
  name: "DBCTNet"                    # "DBCTNet" or "3DRecNet"
  run_all_models: false              # Set true to benchmark all models
```

### Training Parameters
```yaml
training:
  num_epochs: 100                    # Maximum epochs
  num_runs: 3                        # Number of independent runs
  batch_size: 64
  learning_rate: 0.002
  patience: 100                      # Early stopping patience
  checkpoint_interval: 10            # Save checkpoint every N epochs
  num_workers: 4                     # DataLoader workers
```

---

## Usage

### 1. Basic Training

```bash
python train.py
```

This runs the experiment with default `config.yaml` settings.

### 2. Custom Configuration

```bash
python train.py my_experiment_config.yaml
```

### 3. Multiple Runs for Statistical Analysis

Set `num_runs: 5` in `config.yaml` to run 5 independent experiments.

### 4. Model Comparison

Set `run_all_models: true` in `config.yaml` to automatically train and compare all available models.

---

## Output Structure

After training, results are saved in:

```
results/
└── <dataset_name>/              # e.g., WHU-Hi-HanChuan
    └── <model_name>/            # e.g., DBCTNet
        ├── run_1/
        │   ├── config.json                  # Experiment configuration
        │   ├── best_model.pth              # Best model checkpoint
        │   ├── final_model.pth             # Final epoch model
        │   ├── checkpoint_epoch_10.pth     # Periodic checkpoint
        │   ├── training_log.txt            # Detailed training log
        │   └── classification_map.png      # Full-scene prediction
        ├── run_2/
        ├── run_3/
        └── results_summary.xlsx           # Aggregated metrics across runs
```

### Results Summary (Excel)

The `results_summary.xlsx` file contains:

| Column | Description |
|--------|-------------|
| `Run` | Run number |
| `Epochs` | Total epochs |
| `Best_Epoch` | Epoch with best validation performance |
| `OA` | Overall Accuracy (%) |
| `AA` | Average Accuracy (%) |
| `Kappa` | Cohen's Kappa coefficient |
| `Class_1_Acc`, `Class_2_Acc`, ... | Per-class accuracies |
| `Training_Time` | Total training time (HH:MM:SS) |

---

## Adding New Models

To add a custom model:

1. **Create model file**: `MyModel.py`
2. **Implement your model** with standard PyTorch nn.Module
3. **Edit `train.py`**:

```python
from MyModel import MyModel

# In run_single_experiment function:
elif exp['model_name'] == "MyModel":
    model = MyModel(
        num_classes=exp['num_classes'],
        input_bands=exp['bands'],
        # ... your parameters
    )
```

4. **Update config.yaml**:
```yaml
model:
  name: "MyModel"
```

---

## Workflow Summary

1. **Prepare dataset** in `Matlab_data_format/<dataset>/` with `_data.mat` and `_gt.mat` files
2. **Configure** `config.yaml` (dataset path, model, hyperparameters)
3. **Run training**: `python train.py`
4. **Check results** in `results/<dataset>/<model>/`
5. **Analyze** metrics in `results_summary.xlsx`
6. **View** classification maps (`.png` files)

---

## Troubleshooting

### Issue: "Required _data and _gt files not found"
- Ensure filenames end with `_data.mat` and `_gt.mat`
- Check `config.yaml` has correct dataset folder path

### Issue: "ValueError: Ratios must sum to 1.0"
- Verify `split_ratios` in config sum to 1.0 (e.g., [0.3, 0.1, 0.6])

### Issue: CUDA out of memory
- Reduce `batch_size` in config
- Reduce `patch_size`
- Use `dim_reduction_method: "pca"` with fewer bands

---

## Requirements

```
python >= 3.8
torch >= 1.10
torchvision
numpy
scipy
rasterio
scikit-learn
matplotlib
pandas
tqdm
openpyxl
pyyaml
```

Install dependencies:
```bash
pip install torch torchvision numpy scipy rasterio scikit-learn matplotlib pandas tqdm openpyxl pyyaml
```

---

## Contact

For issues or questions, open an issue on [GitHub](https://github.com/Tanishq251/HSI_Pipeline/issues).
