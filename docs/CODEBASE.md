# 🧰 Codebase Guide

[← Back to README](../README.md)

---

## 📂 Repository Structure

```
├── main.py
├── run_all_experiments.py
├── arrange_maps.py
├── check_results.py
├── model_info.py
├── config/
│   ├── config.yaml
│   ├── example_config.yaml
│   ├── dataset.yaml
│   └── config_loader.py
├── models/
│   ├── registry.py
│   ├── y2017/ … y2026/ # published baselines, grouped by publication year
│   └── helpers/        # shared building blocks used by multiple models
├── utils/
└── {results.directory}/   # created on first run, name set in config.yaml
```

### Top-level scripts

- `main.py`: The main research workflow. Use it for regular experiments, dataset/model listing, map arrangement, and score arrangement.
- `run_all_experiments.py`: Extended runner for sweep-style experiments, especially useful for ablation studies.
- `arrange_maps.py`: Standalone figure-generation script for classification maps and GT map export.
- `check_results.py`: Result inspector and recovery tool when you want to audit completed runs or rebuild missing summary CSVs.
- `model_info.py`: Utility for parameter-count and FLOP comparison across models. Prints a params/FLOPs table for every model under a fixed `(1, 1, 30, 11, 11)` probe input at 16 classes, and writes the same table as `model_complexity.tex` for papers. Both outputs are generated locally and are not tracked in the repository.

---

---

## 🧰 Core Utilities, Explained for Researchers

The `utils/` folder is where most of the experiment logic lives. Instead of treating it as a flat list of helpers, it is easier to think of it in groups.

### Data and dataset handling

- `utils/data_loader.py`: Downloads datasets from HuggingFace, loads the hyperspectral cube and GT map, and prepares dataset objects for training. The main classes here are `DatasetLoader` and `HyperspectralDataset`.
- `utils/data_split.py`: Handles train/val/test creation. This is the file to read if your work depends on how labeled samples are allocated, especially for disjoint splitting.

### Experiment setup and execution

- `utils/experiment.py`: Builds the run directory, saves the active config, loads data, and prepares loaders before training starts.
- `utils/experiment_runner.py`: Runs one complete experiment from seed setup to post-training analysis. The main entry point here is `run_single_experiment(...)`.
- `utils/trainer.py`: Contains the actual training loop, validation logic, checkpointing, and summary printing.
- `utils/optimizers.py`: Central place for optimizer selection and optimizer defaults.

### Metrics and result storage

- `utils/metrics.py`: Computes OA, AA, Kappa, records results, and runs post-training result handling.
- `utils/results_io.py`: Shared result-loading layer used by the reporting and map-arrangement utilities.
- `utils/score_arranger.py`: Generates LaTeX tables from saved experimental results. If your goal is paper tables, this is one of the main files to understand.

### Visualization and figure generation

- `utils/visualization.py`: Generates single classification maps during or after training.
- `utils/map_arranger.py`: Builds arranged comparison figures across models or runs.
- `utils/map_arranger_integration.py`: Connects arranged figure generation to the main training pipeline.
- `utils/disjoint_visualizer.py`: Useful when you want to visually inspect the train/val/test regions created by disjoint splitting.
- `utils/colormap_helpers.py`: Shared colormap logic used by the visualization code.

### Dataset and model inspection

- `utils/viewer.py`: Unified inspection module for dataset metadata and registered models. It exposes `DatasetViewer` and `ModelViewer`.
- `utils/dataset_viewer.py`: Compatibility wrapper around the dataset-viewing functionality now centered in `utils/viewer.py`.
- `utils/model_viewer.py`: Compatibility wrapper around the model-viewing functionality now centered in `utils/viewer.py`.

### Package exports

- `utils/__init__.py`: Re-exports the most commonly used utilities so they can be imported more conveniently.

---

## 🔍 When to Look at Which File

If you are trying to answer a specific research question, these are the best starting points.

- “How are datasets loaded and normalized?”
  Read `utils/data_loader.py`.

- “How exactly are train/val/test splits created?”
  Read `utils/data_split.py`.

- “Where does one experiment actually start?”
  Read `main.py`, then `utils/experiment_runner.py`, then `utils/experiment.py`.

- “How does training, early stopping, and checkpointing work?”
  Read `utils/trainer.py`.

- “Where are OA, AA, and Kappa computed and saved?”
  Read `utils/metrics.py`.

- “How are classification maps generated?”
  Read `utils/visualization.py`.

- “How are arranged figures built for papers?”
  Read `utils/map_arranger.py`.

- “How are LaTeX tables produced?”
  Read `utils/score_arranger.py`.

---

---

## 📋 Common Workflows

### Standard training

```bash
python main.py
python main.py config/config.yaml
python main.py config/example_config.yaml
```

### Explore available resources

```bash
python main.py --list-models
python main.py --list-datasets
```

### Run post-training figure and table generation

```bash
python main.py --arrange-only
python main.py --arrange-only --in_sep_folder
python main.py --arrange-scores
```

### Run larger sweeps

```bash
python run_all_experiments.py config/config.yaml
```

### Generate arranged maps directly

```bash
python arrange_maps.py
python arrange_maps.py   --dataset Botswana   --models MambaHSI GAHT SpectralFormer   --rows 1 --cols 3   --metric OA --selection best   --colormap jet   --include-gt   --dpi 300
```

### Inspect and repair results

```bash
python check_results.py
python check_results.py -r ./results
python check_results.py --recover
python check_results.py --dry-run
```

### Compare model size and FLOPs

```bash
python model_info.py
python model_info.py MambaHSI GAHT SpectralFormer
python model_info.py --latex
```

---

---

## 📁 Output Structure

A standard run is saved under:

```text
{results.directory}/{dataset}/{model}/run_{N}/
```

A typical run folder contains:

- the best checkpoint
- the final checkpoint
- periodic checkpoints
- the config used for that run
- the training log
- the generated classification map

At the model level, results are aggregated into `results_summary.csv`, which is then used by map arrangement and score arrangement.

---

---

## 🗂️ File-by-File Summary

For quick navigation, here is the compact file-by-file summary.

### Config

- `config/config.yaml`: main experiment config
- `config/example_config.yaml`: second runnable template / sweep starting point
- `config/dataset.yaml`: dataset metadata registry (24 datasets)
- `config/config_loader.py`: config loading and dot-notation access

### Utils

- `utils/data_loader.py`: dataset download, loading, normalization, dataset objects
- `utils/data_split.py`: standard and disjoint data splitting
- `utils/trainer.py`: training loop and checkpointing
- `utils/metrics.py`: metric computation and result recording
- `utils/experiment.py`: run setup and dataloader preparation
- `utils/experiment_runner.py`: one full experiment execution path
- `utils/optimizers.py`: optimizer selection
- `utils/results_io.py`: result loading helpers
- `utils/visualization.py`: single classification-map generation
- `utils/map_arranger.py`: arranged map figure generation
- `utils/map_arranger_integration.py`: map arrangement hook for training scripts
- `utils/score_arranger.py`: LaTeX score-table generation
- `utils/disjoint_visualizer.py`: disjoint split visualization
- `utils/colormap_helpers.py`: shared colormap utilities
- `utils/viewer.py`: dataset and model inspection utilities
- `utils/dataset_viewer.py`: compatibility wrapper for dataset viewer
- `utils/model_viewer.py`: compatibility wrapper for model viewer
- `utils/__init__.py`: convenience re-exports

---
