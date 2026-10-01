"""Unified viewer module for datasets and models.

Consolidates the identical patterns from dataset_viewer.py and model_viewer.py
into one file.  All original public APIs are preserved.
"""

import sys
import yaml
from pathlib import Path
from tabulate import tabulate


# ──────────────────────────────────────────────
# DatasetViewer (moved from dataset_viewer.py)
# ──────────────────────────────────────────────

class DatasetViewer:
    """View and display dataset information from config in CLI"""

    def __init__(self, config_path='config/dataset.yaml'):
        self.config_path = config_path
        self.datasets = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('datasets', {})
        except FileNotFoundError:
            print(f"Error: Config file not found at {self.config_path}")
            return {}

    def list_datasets(self):
        if not self.datasets:
            print("No datasets found")
            return

        dataset_names = list(self.datasets.keys())
        print(f"\n{'='*60}")
        print(f"Available Datasets ({len(dataset_names)})")
        print(f"{'='*60}")
        for i, name in enumerate(dataset_names, 1):
            print(f"  {i:2d}. {name}")
        print(f"{'='*60}\n")

    def print_summary_table(self):
        if not self.datasets:
            print("No datasets found")
            return

        table_data = []
        for name, info in self.datasets.items():
            table_data.append([
                name,
                info.get('height', 'N/A'),
                info.get('width', 'N/A'),
                info.get('bands', 'N/A'),
                info.get('num_classes', 'N/A'),
                info.get('labeled_samples', 'N/A'),
            ])

        headers = ['Dataset', 'Height', 'Width', 'Bands', 'Classes', 'Labeled Samples']
        print(f"\n{'='*100}")
        print("Dataset Summary")
        print(f"{'='*100}")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print(f"{'='*100}\n")

    def print_dataset_info(self, dataset_name):
        if dataset_name not in self.datasets:
            print(f"Error: Dataset '{dataset_name}' not found")
            self.list_datasets()
            return

        info = self.datasets[dataset_name]

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        print(f"Spatial Dimensions: {info.get('height', 'N/A')} x {info.get('width', 'N/A')}")
        print(f"Number of Bands: {info.get('bands', 'N/A')}")
        print(f"Number of Classes: {info.get('num_classes', 'N/A')}")
        print(f"Total Samples: {info.get('total_samples', 'N/A'):,}")
        print(f"Labeled Samples: {info.get('labeled_samples', 'N/A'):,}")

        class_names = info.get('class_names', [])
        class_samples = info.get('class_samples', {})

        if class_names:
            print(f"\n{'Class Details':^70}")
            print(f"{'-'*70}")

            table_data = []
            for i, class_name in enumerate(class_names, 1):
                samples = class_samples.get(f'Class_{i}', 0)
                table_data.append([i, class_name, f"{samples:,}"])

            headers = ['#', 'Class Name', 'Samples']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))

        print(f"{'='*70}\n")

    def print_all_datasets(self):
        if not self.datasets:
            print("No datasets found")
            return
        for dataset_name in self.datasets.keys():
            self.print_dataset_info(dataset_name)


# ──────────────────────────────────────────────
# ModelViewer (moved from model_viewer.py)
# ──────────────────────────────────────────────

class ModelViewer:
    """View and display available models"""

    def __init__(self):
        # Add parent directory to path so we can import models
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models import list_models, get_model_config
        self._list_models = list_models
        self._get_model_config = get_model_config
        self.models = list_models()

    def list_models(self):
        print(f"\n{'='*60}")
        print(f"Available Models ({len(self.models)})")
        print(f"{'='*60}")
        for i, name in enumerate(self.models, 1):
            print(f"  {i:2d}. {name}")
        print(f"{'='*60}\n")

    def print_summary_table(self):
        table_data = []
        for name in self.models:
            cfg = self._get_model_config(name)
            expects_4d = cfg.get('expects_4d', False)
            table_data.append([name, 'Yes' if expects_4d else 'No'])

        headers = ['Model Name', 'Expects 4D Input']
        print(f"\n{'='*70}")
        print("Model Summary")
        print(f"{'='*70}")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print(f"{'='*70}\n")

    def print_model_info(self, model_name):
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found")
            self.list_models()
            return

        cfg = self._get_model_config(model_name)

        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        print(f"Expects 4D Input: {cfg.get('expects_4d', False)}")

        if cfg:
            print(f"\nConfiguration:")
            for key, value in cfg.items():
                if key != 'expects_4d':
                    print(f"  {key}: {value}")

        print(f"{'='*70}\n")

    def print_all_models(self):
        for model_name in self.models:
            self.print_model_info(model_name)


# ──────────────────────────────────────────────
# Convenience functions — Dataset
# ──────────────────────────────────────────────

def get_dataset_list(config_path='config/dataset.yaml'):
    """Get list of all available datasets"""
    viewer = DatasetViewer(config_path)
    return list(viewer.datasets.keys())


def get_dataset_info(dataset_name, config_path='config/dataset.yaml'):
    """Get info dict for a specific dataset"""
    viewer = DatasetViewer(config_path)
    return viewer.datasets.get(dataset_name)


def print_dataset_list(config_path='config/dataset.yaml'):
    DatasetViewer(config_path).list_datasets()


def print_dataset_summary(config_path='config/dataset.yaml'):
    DatasetViewer(config_path).print_summary_table()


def print_dataset_details(dataset_name, config_path='config/dataset.yaml'):
    DatasetViewer(config_path).print_dataset_info(dataset_name)


def print_all_datasets(config_path='config/dataset.yaml'):
    DatasetViewer(config_path).print_all_datasets()


def print_datasets(*dataset_names, config_path='config/dataset.yaml'):
    viewer = DatasetViewer(config_path)
    for name in dataset_names:
        viewer.print_dataset_info(name)


# ──────────────────────────────────────────────
# Convenience functions — Model
# ──────────────────────────────────────────────

def get_model_list():
    """Get list of all available models"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models import list_models
    return list_models()


def print_model_list():
    ModelViewer().list_models()


def print_model_summary():
    ModelViewer().print_summary_table()


def print_model_details(model_name):
    ModelViewer().print_model_info(model_name)


def print_all_models():
    ModelViewer().print_all_models()


def print_models(*model_names):
    viewer = ModelViewer()
    for name in model_names:
        viewer.print_model_info(name)
