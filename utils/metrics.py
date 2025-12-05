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
    
    # Per-class accuracy
    per_class_acc = []
    for class_id in range(num_classes):
        mask = np.array(targets) == class_id
        if mask.sum() > 0:
            class_acc = accuracy_score(np.array(targets)[mask], np.array(predictions)[mask]) * 100
            per_class_acc.append(class_acc)
    
    # Average Accuracy
    aa = np.mean(per_class_acc)
    
    return oa, aa, kappa, per_class_acc


def update_results_csv(dataset_name, model_name, run_number, results_dict):
    """Update or create CSV file with experiment results"""
    
    results_base = "results"
    dataset_dir = os.path.join(results_base, dataset_name)
    model_dir = os.path.join(dataset_dir, model_name)
    csv_path = os.path.join(model_dir, 'results_summary.csv')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()
    
    new_row = pd.DataFrame([results_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    print(f"Results updated: {csv_path}")


def post_training_analysis(trained_model, predictions, targets, best_epoch, training_time,
                           dataset, device, run_dir, dataset_name, model_name, run_number,
                           num_epochs, patch_size, batch_size, split_ratios, split_samples_count,
                           train_idx, val_idx, test_idx, num_classes, cmap='tab20',
                           show_colorbar=False, dpi=300, block_background=True):
    """Perform complete post-training analysis"""
    
    from utils.visualization import generate_classification_map
    
    # Calculate metrics
    oa, aa, kappa, per_class_acc = calculate_metrics(predictions, targets, num_classes)
    
    # Generate classification map
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
        block_background=block_background
    )
    
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
    
    update_results_csv(dataset_name, model_name, run_number, results_dict)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Results saved in: {run_dir}")
    print("="*60)
