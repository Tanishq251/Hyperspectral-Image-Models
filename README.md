# HSI_Pipeline

A comprehensive deep learning framework for Hyperspectral Image (HSI) classification with 21 state-of-the-art models and 23 benchmark datasets. Features automated experiment management, flexible YAML configuration, and advanced visualization tools.

## 🌟 Key Features

- **21 State-of-the-Art Models** - Transformers, Mamba-based, CNNs, and hybrid architectures
- **23 Benchmark Datasets** - Auto-download from Hugging Face Hub
- **Single YAML Configuration** - Control everything from one config file
- **Automated Experiments** - Run multiple models × datasets × runs automatically
- **Smart Visualization** - Classification maps with customizable colormaps
- **Map Arrangement Tool** - Create publication-ready figure grids
- **Comprehensive Metrics** - OA, AA, Kappa, per-class accuracy with CSV export

---

## 🤗 Hugging Face Dataset Hub

> **All datasets are hosted on Hugging Face and automatically downloaded to cache!**
>
> � **[HSI_Pipeline Datasets](https://huggingface.co/datasets/Tanishq165/HSI_Datasets)**

```bash
# List all available datasets
python main.py --list-datasets

# List all available models
python main.py --list-models
```

---

## 🚀 Quick Start

### Installation

```bash
# Core dependencies
pip install torch torchvision numpy scipy scikit-learn pandas
pip install pyyaml huggingface_hub matplotlib pillow tqdm

# For Mamba-based models
pip install mamba-ssm causal-conv1d

# For model summaries (optional)
pip install torchinfo timm einops
```

### Basic Usage

```bash
# Run with default config
python main.py

# Run with custom config
python main.py config/my_config.yaml

# List available models/datasets
python main.py --list-models
python main.py --list-datasets

# Run map arrangement only
python main.py --arrange-only

# Show help
python main.py --help
```

---

## 📁 Project Structure

```
HSI_Pipeline/
├── config/
│   ├── config.yaml          # Main configuration (single file for everything)
│   ├── config_loader.py     # Config loading utility
│   └── dataset.yaml         # Dataset metadata (24 datasets)
├── models/                  # 21 model implementations
│   ├── __init__.py          # Exports: create_model, list_models, InputShapeWrapper
│   ├── registry.py          # Model registry with @register_model decorator
│   ├── Ours/                # Custom models (AMMT)
│   ├── MambaHSI.py          # Mamba-based models
│   ├── SSFTTnet.py          # Transformer models
│   └── ...                  # More models
├── utils/
│   ├── data_loader.py       # DatasetLoader, HyperspectralDataset
│   ├── data_split.py        # split_data, split_samples
│   ├── experiment.py        # setup_experiment
│   ├── experiment_runner.py # run_single_experiment
│   ├── trainer.py           # train_model with early stopping
│   ├── metrics.py           # calculate_metrics, post_training_analysis
│   ├── visualization.py     # generate_classification_map
│   ├── map_arranger.py      # MapArranger class
│   ├── map_arranger_integration.py
│   └── optimizers.py        # create_optimizer (adam, adamw, sgd, etc.)
├── results/                 # Output directory (auto-created)
├── main.py                  # Main entry point
└── README.md
```

---

## ⚙️ Complete Configuration Reference

All settings are in `config/config.yaml`:

```yaml
# ============================================================
# DATASET
# ============================================================
dataset:
  names: ["Botswana"] # Dataset(s) to run (auto-downloaded)
  run_all_datasets: False # True = run all 24 datasets
  patch_size: 11 # Spatial patch size (e.g., 11×11)
  stride: 1 # Stride for patch extraction
  verbose: True # Print dataset info

# ============================================================
# DATA SPLIT
# ============================================================
data_split:
  method: "ratio" # "ratio" or "samples"
  split_ratios: [0.1, 0.1, 0.8] # [train, val, test] percentages
  # split_samples: [30, 10]  # Alternative: fixed samples per class
  random_state: 42 # Random seed
  print_stats: True # Print split statistics

# ============================================================
# PREPROCESSING
# ============================================================
preprocessing:
  dim_reduction_method: "pca" # "pca", "maxpool", or null
  num_pca_bands: 30 # Number of PCA components
  maxpool_kernel: 2 # Kernel size for maxpool (if used)
  use_channel_dim: True # Add channel dimension
  band_indices: null # Specific bands to use (null = all)

# ============================================================
# MODEL
# ============================================================
model:
  name: ["MambaHSI"] # Model(s) to run
  run_all_models: False # True = run all 21 models
  print_summary: False # Print model architecture
  summary_only: False # Only print summary, skip training
  summary_depth: 4 # Depth of model summary

# ============================================================
# TRAINING
# ============================================================
training:
  num_epochs: 100 # Training epochs
  num_runs: 3 # Runs per model (for averaging)
  batch_size: 32 # Batch size
  learning_rate: 0.001 # Learning rate
  optimizer: "adam" # adam, adamw, sgd, rmsprop, adagrad, adadelta
  optimizer_params: {} # Extra params (weight_decay, momentum, etc.)
  patience: 10 # Early stopping patience
  checkpoint_interval: 10 # Save checkpoint every N epochs
  num_workers: 8 # DataLoader workers

# ============================================================
# DEVICE
# ============================================================
device:
  use_cuda: True # Use GPU if available

# ============================================================
# VISUALIZATION
# ============================================================
visualization:
  cmap: "jet" # Colormap: jet, viridis, tab20, etc.
  block_background: True # Black background for unlabeled pixels
  show_colorbar: False # Show colorbar on map
  dpi: 300 # Image resolution

# ============================================================
# MAP ARRANGEMENT (for figure grids)
# ============================================================
map_arrangement:
  enabled: False # Auto-run after training
  base_output_dir: "results"
  dataset_dir: "Botswana"

  # Model Selection
  all_models: False # True = auto-discover all models
  models: ["MambaHSI", "SSMamba", "AMMT"]
  map_type: ["best", "best", "best"] # best, worst, or threshold
  metric: ["OA", "OA", "OA"] # OA, AA, Kappa

  # Visualization
  visualization:
    cmap: "jet"
    block_background: True
    include_gt: False # Include ground truth map
    regenerate_maps: False # Force regenerate from checkpoints

  # Layout
  layout:
    rows: 2
    cols: 2
    gap: 20 # Column gap (%)
    row_gap: 30 # Row gap (%)
    orientation: "horizontal"

  # Labels
  labels:
    fontsize: 16
    position: "top" # top, bottom, left, right
    alignment: "center"

  # Output
  output:
    dpi: 300
    path: null # Custom output path
```

---

## 🧠 Available Models (21)

AMMT, MambaHSI, MambaHSI_Plus, SSMamba, S2Mamba, SSFTTNet, SpectralFormer, MorphFormer, MFT, GAHT, MASSFormer, 3DConvSST, DBCTNet, GTCFN, FAHM, GSCViT, HybridSN, 3DRecNet, SACNet, S3ANet, MCTGCL

```bash
# List all models with details
python main.py --list-models
```

---

## 📊 Available Datasets (24)

| Dataset          | Size      | Bands | Classes |
| ---------------- | --------- | ----- | ------- |
| Indian_Pines     | 200×145   | 145   | 16      |
| Pavia University | 610×340   | 103   | 9       |
| Pavia Center     | 1096×715  | 102   | 9       |
| Botswana         | 1476×256  | 145   | 14      |
| KSC              | 512×614   | 176   | 13      |
| Salinas          | 512×217   | 204   | 16      |
| Houston13        | 954×210   | 48    | 7       |
| Houston18        | 349×1905  | 144   | 15      |
| Trento           | 166×600   | 63    | 6       |
| Berlin           | 1723×476  | 244   | 8       |
| Augsburg         | 332×485   | 180   | 7       |
| WHU-Hi-LongKou   | 550×400   | 270   | 9       |
| WHU-Hi-HanChuan  | 1217×303  | 274   | 16      |
| WHU-Hi-HongHu    | 940×475   | 270   | 22      |
| Dioni            | 250×1376  | 176   | 12      |
| Loukia           | 249×945   | 176   | 14      |
| Muufl            | 325×220   | 64    | 11      |
| Utopia           | 478×595   | 432   | 9       |
| Holden           | 595×440   | 418   | 6       |
| NiliFossae       | 478×593   | 425   | 9       |
| Qingyun          | 880×1360  | 176   | 6       |
| Pingan           | 1230×1000 | 176   | 10      |
| Tangdaowan       | 1740×860  | 176   | 18      |

---

## 📂 Output Structure

```
results/
└── Botswana/
    └── MambaHSI/
        ├── results_summary.csv          # All runs summary
        └── run_1/
            ├── config.yaml              # Run configuration
            ├── best_model.pth           # Best checkpoint (by val accuracy)
            ├── final_model.pth          # Final epoch checkpoint
            ├── checkpoint_epoch_90.pth  # Periodic checkpoint
            ├── classification_map_jet.png   # Map with colormap name
            ├── classification_map.png       # Generic map (backward compat)
            └── training.log             # Training log
```

---

## 🎯 Example Workflows

### Quick Test (Single Model, Single Dataset)

```yaml
dataset:
  names: ["Indian_Pines"]
model:
  name: ["MambaHSI"]
training:
  num_epochs: 10
  num_runs: 1
```

### Benchmark (Multiple Models × Datasets)

```yaml
dataset:
  names: ["Botswana", "Indian_Pines", "Houston13"]
model:
  name: ["AMMT", "MambaHSI", "SSMamba", "SSFTTNet"]
training:
  num_epochs: 100
  num_runs: 5
```

### Run All Models on One Dataset

```yaml
dataset:
  names: ["Botswana"]
model:
  run_all_models: True
training:
  num_runs: 3
```

### Run All Datasets with One Model

```yaml
dataset:
  run_all_datasets: True
model:
  name: ["MambaHSI"]
training:
  num_runs: 3
```

---

## 🎨 Map Arrangement

Create publication-ready figure grids:

```bash
# Run map arrangement only
python main.py --arrange-only
```

### Smart Colormap Handling

- **Existing maps**: If `classification_map_{cmap}.png` exists, it's reused
- **New colormap**: If different colormap requested, regenerates from checkpoint
- **Backward compatible**: Also saves generic `classification_map.png`

---

## 🔧 Adding New Models

1. Create model file in `models/`:

```python
# models/my_model.py
import torch.nn as nn
from .registry import register_model

class MyModel(nn.Module):
    def __init__(self, num_classes, bands, patch_size, **kwargs):
        super().__init__()
        # ... model architecture

    def forward(self, x):
        # ... forward pass
        return x

@register_model('MyModel', expects_4d=True)  # Set expects_4d if model needs 4D input
def my_model(pretrained=False, **kwargs):
    return MyModel(**kwargs)
```

2. Model is auto-discovered and available immediately!

---

## 📈 Data Flow

```
main.py
  ├── load_config() → Config object
  ├── get_datasets_to_run() → dataset names
  ├── get_models_to_run() → model names
  └── run_single_experiment()
        ├── setup_experiment()
        │     ├── DatasetLoader.load_dataset() → HuggingFace download
        │     ├── HyperspectralDataset() → patches + preprocessing
        │     ├── split_data() / split_samples()
        │     └── DataLoader creation
        ├── create_model() + InputShapeWrapper (if 4D)
        ├── train_model()
        │     ├── create_optimizer()
        │     └── training loop + early stopping
        └── post_training_analysis()
              ├── calculate_metrics() → OA, AA, Kappa
              ├── generate_classification_map()
              └── update_results_csv()
```

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for dataset hosting
- [Mamba-SSM](https://github.com/state-spaces/mamba) authors
- All original model authors

---

**Happy Experimenting! 🚀**
