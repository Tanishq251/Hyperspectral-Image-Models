# ⚙️ Configuration & Experiment Design

[← Back to README](../README.md)

---

The repository is driven by config files in `config/`. In practice, you usually only need to think in terms of a few research decisions: which datasets to run, which models to compare, how to split the data, how to preprocess the cube, and whether you want maps and tables afterward.

### `config/config.yaml`

This is the main config and the best place to start. It controls the full experiment pipeline: dataset selection, splitting, preprocessing, model choice, training setup, map generation, arranged-map settings, and score-table settings.

### `config/example_config.yaml`

A second runnable template with a different dataset/model/split combination. Useful as a reference for a concrete alternative setup instead of editing the main config from scratch, and as a starting point for `run_all_experiments.py` sweeps (copy it and vary the settings you want to ablate).

### `config/dataset.yaml`

This is the dataset registry used by the project. It stores dataset names, dimensions, class counts, and class names, and it powers dataset listing plus dataset-aware visualization.

### `config/config_loader.py`

This is the small config access layer used by the training scripts. It exposes `Config` and `load_config(...)`, and lets the rest of the code read nested settings cleanly with dot notation.

---

## 🎛️ The Settings That Matter Most

Instead of reading the config as a long schema, it helps to think of it in terms of research questions.

### Which datasets am I running?

Use the `dataset` section.

- `names` chooses the datasets explicitly
- `run_all_datasets` uses every dataset listed in `config/dataset.yaml`
- `patch_size` controls the spatial neighborhood seen by the model
- `stride` controls how patches are extracted

If you are doing a sweep study, `run_all_experiments.py` also supports patch-size lists.

### How is the train/val/test split defined?

Use the `data_split` section.

There are two main ways to run experiments:

- standard random splitting
- spatially disjoint splitting

If you want the common random experimental setup, choose `method: "ratio"` or `method: "samples"`.

With `method: "samples"`, `data_split.split_samples_overrides` sets a different budget for named datasets, for example `Indian_Pines: [10, 5]`, whose smallest classes have only 20 pixels. Each run's saved `config.yaml` records the budget that was actually used.

If you want a stricter spatial generalization setting, enable `data_split.disjoint: True`. That switches the pipeline to the disjoint split utilities in `utils/data_split.py`.

A practical way to think about the choices:

- `method: "ratio"` is useful when you want percentage-based splits
- `method: "samples"` is useful when you want fixed labeled samples per class
- `disjoint: True` is useful when you want to reduce spatial leakage between training and testing regions

How `disjoint: True` works with `method: "ratio"`:

- Each class's connected components (`scipy.ndimage.label`) are assigned to train / val / test at a fixed **50 / 30 / 20**: a single component is cut along its longer axis, several components are assigned whole (greedily, largest first) with a correction that caps train near its target.
- A sample goes to the split whose region contains its centre pixel. Every class is kept in all three splits (a repair step moves samples across regions for classes with fewer than 5 samples in a split).
- There is no gap between regions, so patches centred near a region border still share pixels with the neighbouring region.

The realised split for every dataset and class (original pixels, train/val/test counts and percentages, border overlap) is listed in [DISJOINT_SPLIT_CLASSES.md](DISJOINT_SPLIT_CLASSES.md).

### How is the spectral cube prepared?

Use the `preprocessing` section.

This is where you decide whether to:

- keep all spectral bands
- reduce dimensionality with PCA
- use max-pooling-based reduction
- restrict the input to specific spectral bands

For many studies, the biggest decision here is whether you want full-band input or PCA-based input.

### Which models are being compared?

Use the `model` section.

- `name` chooses one or more models explicitly
- `run_all_models` runs the whole registered model set
- `exclude` is useful when you want “all models except a few”
- `print_summary` and `summary_only` help when inspecting architecture shape without launching a full run

### How do I control training behavior?

Use the `training` section.

This is where you control the main experimental budget:

- number of epochs
- number of repeated runs
- batch size
- optimizer choice
- optimizer-specific settings
- early stopping patience

If your goal is a paper-quality comparison, `num_runs` is one of the most important settings because the codebase is built to summarize repeated runs rather than a single seed.

### How are maps generated?

Use the `visualization` section.

This controls whether a classification map is generated after training and how it is rendered. This is the training-time visualization path implemented in `utils/visualization.py`.

### How are arranged comparison figures generated?

Use the `map_arrangement` section.

This controls post-training figure assembly: which models to include, whether to include the ground-truth map, whether to regenerate maps, how the grid should look, and whether to add a class-name legend. This path is implemented in `utils/map_arranger.py` and wired into the main workflow by `utils/map_arranger_integration.py`.

### How are LaTeX tables generated?

Use the `score_arrangement` section.

This controls whether all datasets or only selected ones should be processed, whether all models should be included, and whether the tables should summarize repeated runs or focus on best-run values.

---

---

## 🔬 Reproducibility

Every run is self-describing: the exact config used is written into the run folder alongside the checkpoints and log, so any result can be traced back to the settings that produced it.

- **Seeding.** `data_split.seeds` takes an explicit list, one seed per run (`seeds: [1, 2, 3]` → three runs on three fixed splits). Leave it empty and seeds auto-increment from `data_split.random_state`. The same seed list gives the same splits across models, which is what makes a model-vs-model table fair.
- **Repeated runs.** `training.num_runs` controls repeats; `--arrange-scores` reports mean ± std over them rather than a single seed.
- **Protocol.** Fix `dataset.patch_size`, `preprocessing.dim_reduction_method`/`num_pca_bands`, and `data_split.split_samples` across every model in a comparison. Report them in your paper — they move results more than most architectural differences.
- **Complexity.** `python model_info.py` reports parameters and FLOPs for every model under a fixed `(1, 1, 30, 11, 11)` probe input at 16 classes, so the numbers are comparable across models. Regenerate it locally rather than relying on a stored table.

A reproducible comparison, end to end:

```bash
python main.py config/your_config.yaml      # train: N models × M datasets × K seeds
python main.py config/your_config.yaml --arrange-scores   # LaTeX table, mean ± std
python main.py config/your_config.yaml --arrange-only     # qualitative map figure
python model_info.py --latex                              # params/FLOPs table
```

---

---

## 📝 Notes for Paper-Style Experiments

A few settings matter much more than others in actual research use:

- `training.num_runs`: important for stable comparison across repeated runs
- `data_split.method` and `data_split.disjoint`: important for defining the experimental protocol
- `dataset.patch_size`: important for spatial-context studies
- `preprocessing.dim_reduction_method` and `preprocessing.num_pca_bands`: important for PCA-vs-full-spectrum studies
- `visualization.generate_maps`: useful when qualitative comparisons matter
- `map_arrangement` and `score_arrangement`: useful when turning finished runs into paper figures and tables

If you are extending the repository, the cleanest path is usually:

1. add or modify a model in `models/`
2. select it through the config
3. run `main.py` for standard experiments or `run_all_experiments.py` for sweeps
4. use the built-in map and score utilities to prepare visual and tabular comparisons

---
