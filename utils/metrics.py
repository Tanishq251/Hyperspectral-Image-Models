import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score


def calculate_metrics(predictions, targets, num_classes):
    """Calculate OA, AA, Kappa, and per-class accuracies"""
    
    # Overall Accuracy
    oa = accuracy_score(targets, predictions) * 100
    
    # Kappa
    kappa = cohen_kappa_score(targets, predictions)
    
    # Per-class accuracy. A class with no test samples gets NaN in its own slot
    # (instead of being skipped), so Class_i_Acc always refers to class i.
    per_class_acc = []
    for class_id in range(num_classes):
        mask = np.array(targets) == class_id
        if mask.sum() > 0:
            class_acc = accuracy_score(np.array(targets)[mask], np.array(predictions)[mask]) * 100
        else:
            class_acc = float('nan')
        per_class_acc.append(class_acc)
    
    # Average Accuracy over the classes present in the test set
    aa = np.nanmean(per_class_acc)
    
    return oa, aa, kappa, per_class_acc


def update_results_csv(dataset_name, model_name, run_number, results_dict, results_dir="results"):
    """Update or create CSV file with experiment results"""
    
    results_base = results_dir
    dataset_dir = os.path.join(results_base, dataset_name)
    model_dir = os.path.join(dataset_dir, model_name)
    csv_path = os.path.join(model_dir, 'results_summary.csv')
    
    # Ensure directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame()
        
        new_row = pd.DataFrame([results_dict])
        df = pd.concat([df, new_row], ignore_index=True)
        
        df.to_csv(csv_path, index=False)
        print(f"Results saved: {csv_path}")
    except Exception as e:
        print(f"Error saving results CSV: {e}")
        print(f"  Path: {csv_path}")
        print(f"  Results dict: {results_dict}")


def prune_model_checkpoints(dataset_name, model_name, results_dir="results", metric="OA"):
    """Keep only best and worst run checkpoints for a model based on a metric.

    Policy:
    - Keep only `best_model.pth` in the best and worst runs.
    - Remove `best_model.pth`, `final_model.pth`, and epoch checkpoints from all other runs.
    - Preserve configs, logs, CSV summaries, and generated maps.
    """
    model_dir = os.path.join(results_dir, dataset_name, model_name)
    csv_path = os.path.join(model_dir, 'results_summary.csv')

    if not os.path.exists(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Warning: Could not read results summary for checkpoint pruning: {e}")
        return

    if df.empty or metric not in df.columns or 'Run' not in df.columns:
        return

    best_idx = df[metric].idxmax()
    worst_idx = df[metric].idxmin()
    keep_runs = {
        int(df.loc[best_idx, 'Run']),
        int(df.loc[worst_idx, 'Run']),
    }

    removed_files = 0
    kept_runs_label = ", ".join(f"run_{run}" for run in sorted(keep_runs))

    for item in os.listdir(model_dir):
        run_dir = os.path.join(model_dir, item)
        if not (os.path.isdir(run_dir) and item.startswith('run_')):
            continue

        try:
            run_number = int(item.split('_')[1])
        except (IndexError, ValueError):
            continue

        removable = []
        for filename in os.listdir(run_dir):
            is_epoch_checkpoint = filename.startswith('checkpoint_epoch_') and filename.endswith('.pth')
            is_model_checkpoint = filename in {'best_model.pth', 'final_model.pth'}
            if is_epoch_checkpoint or is_model_checkpoint:
                removable.append(filename)

        for filename in removable:
            file_path = os.path.join(run_dir, filename)
            if run_number in keep_runs and filename == 'best_model.pth':
                continue
            try:
                os.remove(file_path)
                removed_files += 1
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Warning: Failed to remove checkpoint {file_path}: {e}")

    print(f"Checkpoint pruning complete for {dataset_name}/{model_name} using {metric}")
    print(f"  Kept checkpoint runs: {kept_runs_label}")
    print(f"  Removed checkpoint files: {removed_files}")


def post_training_analysis(trained_model, predictions, targets, best_epoch, training_time,
                           dataset, device, run_dir, dataset_name, model_name, run_number,
                           num_epochs, patch_size, batch_size, split_ratios, split_samples_count,
                           train_idx, val_idx, test_idx, num_classes, cmap='tab20',
                           show_colorbar=False, dpi=300, block_background=True,
                           vis_mode='labeled_only', use_spy_colors=True, results_dir="results",
                           generate_maps=True, prune_checkpoints=False,
                           pruning_metric="OA"):
    """Perform complete post-training analysis"""

    from utils.visualization import generate_classification_map

    # Calculate metrics
    oa, aa, kappa, per_class_acc = calculate_metrics(predictions, targets, num_classes)

    # Generate classification map (optional)
    if generate_maps:
        generate_classification_map(
            model=trained_model,
            dataset=dataset,
            device=device,
            run_dir=run_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            run_number=run_number,
            cmap=cmap,
            show_colorbar=show_colorbar,
            dpi=dpi,
            block_background=block_background,
            mode=vis_mode,
            use_spy_colors=use_spy_colors
        )
    else:
        print("⊘ Skipping classification map generation (disabled in config)")
    
    # Prepare results dictionary
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    results_dict = {
        'Run': run_number,
        'Epochs': num_epochs,
        'Best_Epoch': best_epoch,
        'Patch_Size': patch_size,
        'Batch_Size': batch_size,
        'Split_Config': str(split_ratios) if split_samples_count is None else str(split_samples_count),
        'Train_Samples': len(train_idx),
        'Val_Samples': len(val_idx) if val_idx else 0,
        'Test_Samples': len(test_idx),
        'OA': round(oa, 2),
        'AA': round(aa, 2),
        'Kappa': round(kappa, 4),
        'Training_Time': time_str
    }
    
    # Add class-wise accuracies
    for i, class_acc in enumerate(per_class_acc, start=1):
        results_dict[f'Class_{i}_Acc'] = round(class_acc, 2)
    
    update_results_csv(dataset_name, model_name, run_number, results_dict, results_dir=results_dir)

    if prune_checkpoints:
        prune_model_checkpoints(
            dataset_name=dataset_name,
            model_name=model_name,
            results_dir=results_dir,
            metric=pruning_metric,
        )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Results saved in: {run_dir}")
    print("="*60)
