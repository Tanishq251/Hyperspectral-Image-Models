"""Shared helpers for locating and reading experiment results (CSV files).

Used by both map_arranger.py and score_arranger.py to avoid duplication.
"""

import os
import pandas as pd
import numpy as np


def find_summary_csv(results_dir, dataset_name, model_name):
    """Locate the results_summary.csv for a given model.

    Args:
        results_dir: Base results directory (e.g. 'results')
        dataset_name: Name of the dataset
        model_name: Name of the model

    Returns:
        str or None: Path to the CSV file, or None if not found.
    """
    csv_path = os.path.join(results_dir, dataset_name, model_name, 'results_summary.csv')
    return csv_path if os.path.exists(csv_path) else None


def load_model_results(results_dir, dataset_name, model_name):
    """Load results CSV for a model into a DataFrame.

    Returns:
        pd.DataFrame or None
    """
    csv_path = find_summary_csv(results_dir, dataset_name, model_name)
    if csv_path is not None:
        return pd.read_csv(csv_path)
    return None


def get_available_datasets(results_dir="results"):
    """Get list of datasets that have results."""
    if not os.path.exists(results_dir):
        return []
    return sorted(
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    )


def get_available_models(results_dir, dataset_name):
    """Get list of models that have results for a dataset."""
    dataset_path = os.path.join(results_dir, dataset_name)
    if not os.path.exists(dataset_path):
        return []

    models = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            csv_path = os.path.join(item_path, 'results_summary.csv')
            if os.path.exists(csv_path):
                models.append(item)
    return sorted(models)


def get_best_run(df, metric='OA'):
    """Get the best run (row) based on a metric."""
    if df is None or df.empty:
        return None
    return df.loc[df[metric].idxmax()]


def get_mean_std(df, columns):
    """Get mean ± std for specified columns.

    Returns:
        dict: {column_name: (mean, std), ...}
    """
    results = {}
    for col in columns:
        if col in df.columns:
            results[col] = (df[col].mean(), df[col].std())
    return results
