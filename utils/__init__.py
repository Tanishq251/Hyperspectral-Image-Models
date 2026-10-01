# Utils package for hyperspectral image classification
#
# Public API — import the most commonly used names for convenience.

from utils.data_loader import DatasetLoader, HyperspectralDataset
from utils.data_split import split_data, split_samples, split_disjoint_data, print_class_stats
from utils.experiment import setup_experiment, create_run_directory, save_config
from utils.experiment_runner import run_single_experiment, set_global_seed
from utils.metrics import calculate_metrics, post_training_analysis, update_results_csv
from utils.trainer import train_model, save_checkpoint, save_model, print_model_summary
from utils.visualization import generate_classification_map
from utils.disjoint_visualizer import plot_disjoint_maps
from utils.optimizers import create_optimizer, list_optimizers
from utils.results_io import (
    find_summary_csv,
    load_model_results,
    get_available_datasets,
    get_available_models,
    get_best_run,
    get_mean_std,
)
from utils.colormap_helpers import get_colormap, get_spy_cmap, get_spy_colormap_dict, SPY_COLORS

__all__ = [
    # Data
    'DatasetLoader', 'HyperspectralDataset',
    'split_data', 'split_samples', 'split_disjoint_data', 'print_class_stats',
    # Experiment
    'setup_experiment', 'create_run_directory', 'save_config',
    'run_single_experiment', 'set_global_seed',
    # Training
    'train_model', 'save_checkpoint', 'save_model', 'print_model_summary',
    # Metrics / Results
    'calculate_metrics', 'post_training_analysis', 'update_results_csv',
    'find_summary_csv', 'load_model_results',
    'get_available_datasets', 'get_available_models',
    'get_best_run', 'get_mean_std',
    # Visualization
    'generate_classification_map', 'get_colormap',
    'get_spy_cmap', 'get_spy_colormap_dict', 'SPY_COLORS',
    'plot_disjoint_maps',
    # Optimizers
    'create_optimizer', 'list_optimizers',
]
