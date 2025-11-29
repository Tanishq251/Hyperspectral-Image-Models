# HSI_Pipeline

A modular and configuration-driven pipeline for **Hyperspectral Image (HSI) Classification**. This pipeline supports multiple deep learning models, flexible data splitting, preprocessing options, and automated result tracking.

## Features

✨ **Key Capabilities:**
- **Multi-Model Support**: DBCTNet and 3DRecNet (HSIVit) models
- **Flexible Configuration**: YAML-based or inline dictionary configuration
- **Multiple Data Splitting Strategies**: Ratio-based or sample-based splits
- **Advanced Preprocessing**: PCA dimensionality reduction, MaxPooling, channel dimension handling
- **Comprehensive Metrics**: OA, AA, Kappa, per-class accuracies
- **Automated Checkpointing**: Save model at regular intervals and best epochs
- **Result Tracking**: Organized results with training logs and classification maps
- **Multi-Run Experiments**: Statistical analysis with multiple independent runs
- **Model Summarization**: Print model architecture and parameter counts
- **Visualization**: Classification maps with customizable colormaps and DPI

## Project Structure

```
HSI_Pipeline/
├── main.py                      # Entry point with inline config support
├── config/
│   ├── config.yaml              # YAML configuration file
│   └── config_loader.py         # Configuration parser
├── models/
│   ├── DBCTnet.py               # DBCTNet model implementation
│   ├── M3DRecNet.py             # 3DRecNet (HSIVit) model implementation
│   └── __init__.py
├── utils/
│   ├── data_loader.py           # Dataset loading and preprocessing
│   ├── data_split.py            # Data splitting strategies
│   ├── experiment.py            # Experiment setup and initialization
│   ├── trainer.py               # Training loop and model management
│   ├── metrics.py               # Metrics calculation and post-training analysis
│   ├── visualization.py         # Result visualization and classification maps
│   └── __init__.py
├── datasets_folder/             # Dataset directory (you create this)
│   └── <dataset_name>/
│       ├── <name>_data.mat      # Hyperspectral image data
│       └── <name>_gt.mat        # Ground truth labels
└── results/                     # Auto-generated results directory
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
            ├── run_3/
            └── results_summary.xlsx
```

## Dataset Preparation

### Required Format

Your dataset must follow this naming convention:

```
dataset_folder/
└── <dataset_name>/
    ├── <dataset_name>_data.mat  # Hyperspectral image data
    └── <dataset_name>_gt.mat    # Ground truth labels
```

### Data File Structure

#### 1. `<name>_data.mat`

- **Variable name**: `data` (or variations: `Data`, `IMAGE`, `hsi`, `cube`, etc.)
- **Shape**: `(bands, height, width)` or `(height, width, bands)` - auto-detected
- **Type**: Float/Integer array
- **Example**: `(224, 224, 144)` for 224x224 pixels with 144 spectral bands

#### 2. `<name>_gt.mat`

- **Variable name**: `gt` (or variations: `GT`, `label`, `mask`, etc.)
- **Shape**: `(height, width)`
- **Values**: Integer class labels (1, 2, 3, ..., N)
  - **0** represents unlabeled/background pixels

### Example Dataset Setup

```bash
# Create dataset directory
mkdir -p datasets/WHU-Hi-HanChuan

# Place your files
datasets/WHU-Hi-HanChuan/
├── WHU-Hi-HanChuan_data.mat
└── WHU-Hi-HanChuan_gt.mat
```

### Supported Data Formats

- **.mat files** (MATLAB format) - Automatically loads `data` and `gt` keys
- **.tif files** (GeoTIFF) - Loads raster data with rasterio

## Configuration

Edit `config/config.yaml` or use `INLINE_CONFIG` in `main.py` to customize your experiment.

### Main Configuration Sections

#### Dataset Settings

```yaml
dataset:
  datasets_folder: "datasets"      # Path to dataset folder
  use_all: false                   # Run on all datasets in folder
  names: ['WHU-Hi-HanChuan']       # Dataset names to use
  patch_size: 11                   # Spatial patch size (e.g., 11x11)
  stride: 1                        # Sampling stride
  verbose: true                    # Print debug info
```

#### Data Splitting

```yaml
data_split:
  method: "ratio"                  # "ratio" or "samples"
  split_ratios: [0.3, 0.1, 0.6]   # [train, val, test] ratios
  split_samples: null              # Alternative: [30, 10] samples per class
  random_state: 42                 # Random seed for reproducibility
  print_stats: true                # Print split statistics
```

#### Preprocessing

```yaml
preprocessing:
  dim_reduction_method: "pca"      # "pca", "maxpool", or null
  num_pca_bands: 30                # Number of PCA components
  maxpool_kernel: 2                # MaxPool kernel size
  use_channel_dim: true            # Add channel dimension
  band_axis: "channels_first"      # "channels_first" or "channels_last"
  band_indices: null               # Specify band indices (optional)
```

#### Model Selection

```yaml
model:
  name: "DBCTNet"                  # "DBCTNet" or "3DRecNet"
  run_all_models: false            # Set true to benchmark all models
  print_summary: true              # Print model architecture
  summary_only: false              # Skip training (summary only)
  summary_depth: 10                # Depth of model summary
```

#### Training Parameters

```yaml
training:
  num_epochs: 100                  # Maximum epochs
  num_runs: 3                      # Number of independent runs
  batch_size: 64                   # Training batch size
  learning_rate: 0.002             # Learning rate
  patience: 100                    # Early stopping patience
  checkpoint_interval: 10          # Save checkpoint every N epochs
  num_workers: 4                   # DataLoader workers
```

#### Device & Visualization

```yaml
device:
  use_cuda: true                   # Use CUDA if available

visualization:
  cmap: "jet"                      # Colormap for visualization
  show_colorbar: false             # Show colorbar in maps
  dpi: 600                         # DPI for saved images
```

## Usage

### 1. Basic Training

```bash
python main.py
```

This runs the experiment with default `config/config.yaml` settings.

### 2. Custom Configuration File

```bash
python main.py config/my_experiment.yaml
```

Load a custom YAML configuration file.

### 3. Inline Configuration

Edit `main.py` and set:

```python
USE_INLINE_CONFIG = True
```

Then modify `INLINE_CONFIG` dictionary directly in the file for quick experimentation.

### 4. Multiple Runs for Statistical Analysis

Set `num_runs: 5` in config to run 5 independent experiments with different random seeds.

### 5. Model Comparison

Set `run_all_models: true` in config to automatically train and compare all available models (DBCTNet, 3DRecNet).

### 6. Summary-Only Mode

Set `summary_only: true` and `print_summary: true` to view model architecture without training.

## Output Structure

After training, results are organized in:

```
results/
└── <dataset_name>/
    └── <model_name>/
        ├── run_1/
        │   ├── config.json              # Experiment configuration
        │   ├── best_model.pth           # Best model checkpoint
        │   ├── final_model.pth          # Final epoch model
        │   ├── checkpoint_epoch_10.pth  # Periodic checkpoints
        │   ├── training_log.txt         # Detailed training logs
        │   └── classification_map.png   # Full-scene prediction
        ├── run_2/
        ├── run_3/
        └── results_summary.xlsx         # Aggregated metrics
```

### Results Summary (Excel)

The `results_summary.xlsx` file contains aggregated metrics across all runs:

| Column | Description |
|--------|-------------|
| `Run` | Run number |
| `Epochs` | Total epochs trained |
| `Best_Epoch` | Epoch with best validation performance |
| `OA` | Overall Accuracy (%) |
| `AA` | Average Accuracy (%) |
| `Kappa` | Cohen's Kappa coefficient |
| `Class_1_Acc`, `Class_2_Acc`, ... | Per-class accuracies (%) |
| `Training_Time` | Total training time (HH:MM:SS) |

### Training Log

Detailed `training_log.txt` includes:
- Epoch-by-epoch metrics (loss, OA, AA, Kappa)
- Best epoch information
- Model configuration
- Dataset statistics

## Adding New Models

To add a custom model:

1. **Create model file**: `models/MyModel.py`

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes, input_bands, **kwargs):
        super(MyModel, self).__init__()
        # Your model architecture
        pass
    
    def forward(self, x):
        # Forward pass
        return x
```

2. **Edit `main.py`** in `run_single_experiment()` function:

```python
from models.MyModel import MyModel

# In model initialization section:
if exp['model_name'] == "MyModel":
    model = MyModel(
        num_classes=exp['num_classes'],
        input_bands=exp['bands'],
        # ... your parameters
    )
```

3. **Update `config.yaml`**:

```yaml
model:
  name: "MyModel"
```

## Workflow Summary

1. **Prepare dataset** in `datasets/<dataset_name>/` with `<name>_data.mat` and `<name>_gt.mat` files
2. **Configure** `config/config.yaml` (or use inline config in `main.py`)
   - Set dataset path, model name, hyperparameters
3. **Run training**: `python main.py`
4. **Check results** in `results/<dataset_name>/<model_name>/`
5. **Analyze metrics** in `results_summary.xlsx`
6. **View** classification maps (`.png` files in each run folder)

## Troubleshooting

### Issue: "Required _data and _gt files not found"

- Ensure filenames end with `_data.mat` and `_gt.mat`
- Check `config.yaml` has correct `datasets_folder` path
- Verify dataset directory structure matches expected format

### Issue: "ValueError: Ratios must sum to 1.0"

- Verify `split_ratios` in config sum to 1.0 (e.g., [0.3, 0.1, 0.6])
- If using `split_samples`, ensure values are valid for your dataset size

### Issue: CUDA out of memory

- Reduce `batch_size` in config
- Reduce `patch_size` in config
- Use `dim_reduction_method: "pca"` with fewer bands
- Reduce `num_pca_bands` or use `maxpool` instead

### Issue: "Unknown model" error

- Check model name in config matches implemented models (DBCTNet, 3DRecNet)
- Verify model is imported in `main.py`
- For custom models, ensure they're added to `run_single_experiment()` function

### Issue: Poor classification accuracy

- Check data preprocessing and normalization settings
- Verify dataset split is balanced across classes
- Try different `patch_size` values
- Increase `num_epochs` or adjust `learning_rate`
- Check `patience` value for early stopping

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
torchinfo  # Optional: for model summary
```

### Installation

```bash
pip install torch torchvision numpy scipy rasterio scikit-learn matplotlib pandas tqdm openpyxl pyyaml

# Optional (for model summary feature)
pip install torchinfo
```

## Models

### DBCTNet
A dual-branch convolutional transformer network combining spatial and spectral features for HSI classification.

### 3DRecNet (HSIVit)
A 3D recurrent vision transformer model with hierarchical feature extraction for hyperspectral image analysis.

## License

MIT License

## Contact

For issues, questions, or suggestions, please open an issue on [GitHub](https://github.com/Tanishq251/HSI_Pipeline/issues).
