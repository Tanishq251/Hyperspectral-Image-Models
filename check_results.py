import os
import re
from pathlib import Path
import pandas as pd


def parse_training_log(log_path):
    """
    Parse a training.log file and extract results.
    
    Returns:
        dict: Results dictionary with OA, AA, Kappa, class accuracies, etc.
              Returns None if parsing fails.
    """
    if not os.path.exists(log_path):
        return None
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        results = {}
        
        # Extract OA
        oa_match = re.search(r'Overall Test Accuracy \(OA\):\s*([\d.]+)%', content)
        if oa_match:
            results['OA'] = float(oa_match.group(1))
        
        # Extract AA
        aa_match = re.search(r'Average Test Accuracy \(AA\):\s*([\d.]+)%', content)
        if aa_match:
            results['AA'] = float(aa_match.group(1))
        
        # Extract Kappa
        kappa_match = re.search(r'Kappa:\s*([\d.]+)', content)
        if kappa_match:
            results['Kappa'] = float(kappa_match.group(1))
        
        # Extract class-wise accuracies
        # trainer.py writes "Class 1: xx.xx%" (1-indexed), so class_num from regex
        # is already 1-indexed — matches metrics.py's Class_1_Acc, Class_2_Acc...
        class_matches = re.findall(r'Class (\d+):\s*([\d.]+)%', content)
        for class_num, acc in class_matches:
            results[f'Class_{class_num}_Acc'] = float(acc)  # class_num is 1-indexed
        
        # Extract best epoch if available
        best_epoch_match = re.search(r'Best Model from Epoch (\d+)', content)
        if best_epoch_match:
            results['Best_Epoch'] = int(best_epoch_match.group(1))
        
        # Check if we got the essential metrics
        if 'OA' in results and 'AA' in results and 'Kappa' in results:
            return results
        else:
            return None
            
    except Exception as e:
        print(f"  Error parsing {log_path}: {e}")
        return None


def create_csv_from_training_log(run_dir, dataset_name, model_name, run_number, results_dir='./results'):
    """
    Create results_summary.csv entry from training.log if CSV doesn't exist.
    
    Args:
        run_dir: Path to the run directory (e.g., results/Dataset/Model/run_1)
        dataset_name: Name of the dataset
        model_name: Name of the model
        run_number: Run number
        results_dir: Base results directory
    
    Returns:
        bool: True if CSV was created/updated, False otherwise
    """
    log_path = os.path.join(run_dir, 'training.log')
    
    if not os.path.exists(log_path):
        return False
    
    results = parse_training_log(log_path)
    if results is None:
        return False
    
    # Add run metadata
    results['Run'] = run_number
    
    # Try to get config info if available
    config_path = os.path.join(run_dir, 'config.yaml')
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            results['Epochs'] = config.get('training', {}).get('num_epochs', 'N/A')
            results['Patch_Size'] = config.get('dataset', {}).get('patch_size', 'N/A')
            results['Batch_Size'] = config.get('training', {}).get('batch_size', 'N/A')
        except Exception:
            pass
    
    # Create/update CSV
    model_dir = os.path.join(results_dir, dataset_name, model_name)
    csv_path = os.path.join(model_dir, 'results_summary.csv')
    
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Check if this run already exists
            if 'Run' in df.columns and run_number in df['Run'].values:
                return False  # Already exists
        else:
            df = pd.DataFrame()
        
        new_row = pd.DataFrame([results])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)
        return True
        
    except Exception as e:
        print(f"  Error creating CSV: {e}")
        return False


def recover_missing_csvs(results_dir='./results', dry_run=False):
    """
    Scan all run directories and create results_summary.csv from training.log
    for any runs that are missing CSV entries.
    
    Args:
        results_dir: Path to results directory
        dry_run: If True, only report what would be done without making changes
    
    Returns:
        tuple: (recovered_count, failed_count, already_exists_count)
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return 0, 0, 0
    
    recovered = 0
    failed = 0
    already_exists = 0
    
    print("\n" + "=" * 80)
    print("RECOVERING MISSING CSV FILES FROM TRAINING LOGS")
    print("=" * 80)
    
    if dry_run:
        print("(DRY RUN - No changes will be made)\n")
    else:
        print()
    
    # Iterate through datasets
    for dataset_folder in sorted(results_path.iterdir()):
        if not dataset_folder.is_dir():
            continue
        
        dataset_name = dataset_folder.name
        
        # Iterate through models
        for model_folder in sorted(dataset_folder.iterdir()):
            if not model_folder.is_dir():
                continue
            
            model_name = model_folder.name
            csv_path = model_folder / 'results_summary.csv'
            
            # Get existing runs in CSV
            existing_runs = set()
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if 'Run' in df.columns:
                        existing_runs = set(df['Run'].values)
                except Exception:
                    pass
            
            # Check each run directory
            for run_folder in sorted(model_folder.iterdir()):
                if not run_folder.is_dir() or not run_folder.name.startswith('run_'):
                    continue
                
                try:
                    run_number = int(run_folder.name.split('_')[1])
                except:
                    continue
                
                log_path = run_folder / 'training.log'
                
                if not log_path.exists():
                    continue
                
                if run_number in existing_runs:
                    already_exists += 1
                    continue
                
                # Try to recover
                results = parse_training_log(str(log_path))
                
                if results is None:
                    print(f"{dataset_name}/{model_name}/run_{run_number} - Could not parse training.log")
                    failed += 1
                    continue
                
                if dry_run:
                    print(f"→ Would recover: {dataset_name}/{model_name}/run_{run_number} (OA: {results.get('OA', 'N/A')}%)")
                    recovered += 1
                else:
                    success = create_csv_from_training_log(
                        str(run_folder), dataset_name, model_name, run_number, results_dir
                    )
                    if success:
                        print(f"Recovered: {dataset_name}/{model_name}/run_{run_number} (OA: {results.get('OA', 'N/A')}%)")
                        recovered += 1
                    else:
                        print(f"Failed: {dataset_name}/{model_name}/run_{run_number}")
                        failed += 1
    
    print("\n" + "-" * 80)
    print(f"Summary: Recovered={recovered}, Failed={failed}, Already Existed={already_exists}")
    print("=" * 80 + "\n")
    
    return recovered, failed, already_exists


def check_results(results_dir='./results'):
    """
    Check which datasets and models have results_summary.csv files
    and show how many runs are completed in each

    Args:
        results_dir: Path to results directory
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return None

    # First pass: discover all datasets and models
    all_datasets = set()
    all_models = set()

    for dataset_folder in sorted(results_path.iterdir()):
        if not dataset_folder.is_dir():
            continue
        all_datasets.add(dataset_folder.name)

        for model_folder in sorted(dataset_folder.iterdir()):
            if model_folder.is_dir():
                all_models.add(model_folder.name)

    all_datasets = sorted(all_datasets)
    all_models = sorted(all_models)

    # Print discovery results
    print("\n" + "=" * 100)
    print("DISCOVERY - Available Datasets and Models")
    print("=" * 100)
    print(f"\nTotal Datasets Found: {len(all_datasets)}")
    print(f"Datasets: {', '.join(all_datasets)}")
    print(f"\nTotal Models Found: {len(all_models)}")
    print(f"Models: {', '.join(all_models)}")
    print(f"\nTotal Expected Combinations: {len(all_datasets)} × {len(all_models)} = {len(all_datasets) * len(all_models)}")
    print("=" * 100)

    # Collect all data
    results_data = []

    # Iterate through datasets
    for dataset_folder in sorted(results_path.iterdir()):
        if not dataset_folder.is_dir():
            continue

        dataset_name = dataset_folder.name

        # Iterate through models in each dataset
        for model_folder in sorted(dataset_folder.iterdir()):
            if not model_folder.is_dir():
                continue

            model_name = model_folder.name

            # Check for results_summary.csv
            summary_file = model_folder / 'results_summary.csv'

            if summary_file.exists():
                try:
                    # Read CSV and count runs
                    df = pd.read_csv(summary_file)
                    num_runs = len(df)
                    status = 'yes'
                except Exception:
                    num_runs = 'Error'
                    status = 'no'
            else:
                num_runs = 0
                status = 'no'

            results_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Status': status,
                'Runs': num_runs
            })

    if not results_data:
        print("No results found!")
        return None

    # Create DataFrame
    df = pd.DataFrame(results_data)

    # Print detailed table
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY TABLE - All Datasets and Models")
    print("=" * 100)
    print(f"\n{'Dataset':<20} {'Model':<30} {'Status':<10} {'Runs':<10}")
    print("-" * 100)

    for _, row in df.iterrows():
        print(f"{row['Dataset']:<20} {row['Model']:<30} {row['Status']:<10} {row['Runs']:<10}")

    print("=" * 100)

    # Statistics
    print("\n" + "=" * 100)
    print("STATISTICS")
    print("=" * 100)

    total_combinations = len(df)
    with_summary = len(df[df['Status'] == 'yes'])
    without_summary = total_combinations - with_summary
    total_runs = df[df['Runs'] != 'Error']['Runs'].sum() if with_summary > 0 else 0

    print(f"\nTotal Dataset-Model Combinations: {total_combinations}")
    print(f"With results_summary.csv: {with_summary} ({with_summary/total_combinations*100:.1f}%)")
    print(f"Without results_summary.csv: {without_summary} ({without_summary/total_combinations*100:.1f}%)")
    print(f"Total Runs Completed: {total_runs}")

    # Per dataset summary
    print("\n" + "-" * 100)
    print("PER DATASET SUMMARY")
    print("-" * 100)
    for dataset in sorted(df['Dataset'].unique()):
        dataset_df = df[df['Dataset'] == dataset]
        count_with = len(dataset_df[dataset_df['Status'] == 'yes'])
        total_models = len(dataset_df)
        total_runs_dataset = dataset_df[dataset_df['Runs'] != 'Error']['Runs'].sum()
        print(f"{dataset:<20} Models: {count_with}/{total_models}  |  Total Runs: {total_runs_dataset}")

    # Per model summary
    print("\n" + "-" * 100)
    print("PER MODEL SUMMARY")
    print("-" * 100)
    for model in sorted(df['Model'].unique()):
        model_df = df[df['Model'] == model]
        count_with = len(model_df[model_df['Status'] == 'yes'])
        total_datasets = len(model_df)
        total_runs_model = model_df[model_df['Runs'] != 'Error']['Runs'].sum()
        print(f"{model:<30} Datasets: {count_with}/{total_datasets}  |  Total Runs: {total_runs_model}")

    # Matrix view
    print("\n" + "=" * 100)
    print("MATRIX VIEW - Dataset × Model (Runs Count)")
    print("=" * 100)

    # Create pivot table
    pivot_data = {}
    for _, row in df.iterrows():
        if row['Dataset'] not in pivot_data:
            pivot_data[row['Dataset']] = {}
        pivot_data[row['Dataset']][row['Model']] = row['Runs']

    # Print matrix header
    print(f"\n{'Dataset':<20}", end='')
    for model in all_models:
        print(f"{model:<15}", end='')
    print()
    print("-" * (20 + 15 * len(all_models)))

    # Print matrix rows
    for dataset in all_datasets:
        print(f"{dataset:<20}", end='')
        for model in all_models:
            if dataset in pivot_data and model in pivot_data[dataset]:
                runs = pivot_data[dataset][model]
                if runs == 0:
                    cell = "-"
                elif runs == 'Error':
                    cell = "Error"
                else:
                    cell = f"{runs} runs"
                print(f"{cell:<15}", end='')
            else:
                print(f"{'N/A':<15}", end='')
        print()

    print("=" * 100 + "\n")
    print("Legend: n runs = has results_summary.csv with n runs | - = no results_summary.csv | N/A = folder doesn't exist")
    print("=" * 100 + "\n")

    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check and recover experiment results')
    parser.add_argument('--results-dir', '-r', default='./RESULTS', help='Results directory path')
    parser.add_argument('--recover', action='store_true', help='Recover missing CSVs from training logs')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be recovered without making changes')
    
    args = parser.parse_args()
    
    # First, try to recover missing CSVs if requested
    if args.recover or args.dry_run:
        recover_missing_csvs(args.results_dir, dry_run=args.dry_run)
    
    # Then show the results summary
    df = check_results(args.results_dir)
