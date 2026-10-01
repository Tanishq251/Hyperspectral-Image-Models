"""Score Arranger - Generate LaTeX tables from experiment results"""

import os

import numpy as np

from utils.results_io import (
    get_available_datasets,
    get_available_models,
    load_model_results,
    get_best_run,
    get_mean_std,
)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

_TEX_SPECIAL = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def _tex(text):
    """Escape LaTeX special characters (e.g. the '_' in 'HSIC_FM', 'Indian_Pines')."""
    return ''.join(_TEX_SPECIAL.get(ch, ch) for ch in str(text))


def _collect_model_data(results_dir, dataset_name, models, class_cols=None, use_mean=True):
    """Collect per-model statistics used by both main and class-wise table generators.

    Args:
        results_dir: Base results directory
        dataset_name: Dataset name
        models: List of model names
        class_cols: List of class accuracy column names (optional, for class-wise tables)
        use_mean: Whether to compute mean±std (True) or use best run only (False)

    Returns:
        dict: {model_name: {'table': {...}, 'best': {...}, 'runs': int}}
        Each inner dict maps 'oa', 'aa', 'kappa' and class columns to (value, std).
        'table' holds mean±std over runs (or the best-OA run when use_mean=False or
        there is a single run); 'best' always holds the single best-OA run.
        std is None whenever the value comes from one run. Kappa is scaled x100.
    """
    data = {}
    for model_name in models:
        df = load_model_results(results_dir, dataset_name, model_name)
        if df is None:
            continue

        # Every value of the best run comes from that same run (no mixing across runs)
        best_run = get_best_run(df, 'OA')
        best = {
            'oa': (best_run['OA'], None),
            'aa': (best_run['AA'], None),
            'kappa': (best_run['Kappa'] * 100, None),
        }
        for col in class_cols or []:
            if col in df.columns:
                best[col] = (best_run[col], None)

        if use_mean and len(df) > 1:
            stats = get_mean_std(df, ['OA', 'AA', 'Kappa'] + (class_cols or []))
            table = {
                'oa': stats['OA'],
                'aa': stats['AA'],
                'kappa': (stats['Kappa'][0] * 100, stats['Kappa'][1] * 100),
            }
            for col in class_cols or []:
                if col in stats:
                    table[col] = stats[col]
        else:
            table = best

        data[model_name] = {'table': table, 'best': best, 'runs': len(df)}

    return data


def _format_metric(value_std):
    """Return a formatted cell like '95.32 $\\pm$ 0.21', or '95.32' for a single run."""
    val, std = value_std
    if val is None or np.isnan(val):
        return "-"  # class had no test samples
    if std is None or np.isnan(std):
        return f"{val:.2f}"
    return f"{val:.2f} $\\pm$ {std:.2f}"


def _variant(arrange_best):
    """(data key, label suffix, caption suffix) for the original / arranged tables."""
    if arrange_best:
        return 'best', '_arranged', ' (Best Run by OA)'
    return 'table', '', ''


# ──────────────────────────────────────────────
# Public API — LaTeX generators
# ──────────────────────────────────────────────

def generate_latex_main_table(results_dir, dataset_name, models=None, use_mean=True, arrange_best=False):
    """Generate LaTeX table for OA, AA, Kappa."""
    if models is None:
        models = get_available_models(results_dir, dataset_name)
    if not models:
        print(f"No models found for dataset: {dataset_name}")
        return None

    model_data = _collect_model_data(results_dir, dataset_name, models, use_mean=use_mean)
    key, suffix, caption_extra = _variant(arrange_best)
    sorted_models = sorted(model_data.keys(), key=lambda x: model_data[x][key]['oa'][0])

    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Classification Results on {_tex(dataset_name)} Dataset{caption_extra}}}",
        f"\\label{{tab:{dataset_name.lower()}_results{suffix}}}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Model & OA (\\%) & AA (\\%) & Kappa ($\\times$100) \\\\",
        "\\midrule",
    ]
    for model_name in sorted_models:
        d = model_data[model_name][key]
        latex.append(f"{_tex(model_name)} & {_format_metric(d['oa'])} & {_format_metric(d['aa'])} & {_format_metric(d['kappa'])} \\\\")

    latex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(latex)


def generate_latex_classwise_table(results_dir, dataset_name, models=None, use_mean=True, arrange_best=False):
    """Generate LaTeX table for class-wise accuracies."""
    if models is None:
        models = get_available_models(results_dir, dataset_name)
    if not models:
        print(f"No models found for dataset: {dataset_name}")
        return None

    # Discover class columns from a sample CSV
    sample_df = None
    for model_name in models:
        sample_df = load_model_results(results_dir, dataset_name, model_name)
        if sample_df is not None:
            break
    if sample_df is None:
        return None

    class_cols = sorted(
        [col for col in sample_df.columns if col.startswith('Class_') and col.endswith('_Acc')],
        key=lambda x: int(x.split('_')[1])
    )
    if not class_cols:
        print("No class-wise accuracy columns found")
        return None

    data = _collect_model_data(results_dir, dataset_name, models, class_cols=class_cols, use_mean=use_mean)
    key, suffix, caption_extra = _variant(arrange_best)
    sorted_models = sorted(data.keys(), key=lambda x: data[x][key]['oa'][0])

    col_format = "l" + "c" * len(sorted_models)
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Class-wise Accuracy on {_tex(dataset_name)} Dataset{caption_extra}}}",
        f"\\label{{tab:{dataset_name.lower()}_classwise{suffix}}}",
        f"\\begin{{tabular}}{{{col_format}}}",
        "\\toprule",
        "Class & " + " & ".join(_tex(m) for m in sorted_models) + " \\\\",
        "\\midrule",
    ]

    # Class rows
    for i, col in enumerate(class_cols, start=1):
        row_data = [f"C{i}"]
        for model_name in sorted_models:
            d = data[model_name][key]
            row_data.append(_format_metric(d[col]) if col in d else "-")
        latex.append(" & ".join(row_data) + " \\\\")

    latex.append("\\midrule")

    # OA, AA, Kappa rows
    for metric, label in [('oa', 'OA'), ('aa', 'AA'), ('kappa', 'Kappa ($\\times$100)')]:
        row_data = [label] + [_format_metric(data[m][key][metric]) for m in sorted_models]
        latex.append(" & ".join(row_data) + " \\\\")

    latex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(latex)


# ──────────────────────────────────────────────
# Public API — orchestration
# ──────────────────────────────────────────────

def arrange_scores(dataset_name, models=None, output_dir=None, use_mean=True, results_dir="results"):
    """Generate and save all LaTeX table variants (original + arranged)."""
    print(f"\n{'='*60}")
    print(f"Arranging Scores for: {dataset_name}")
    print(f"{'='*60}\n")

    if models is None:
        models = get_available_models(results_dir, dataset_name)
    if not models:
        print(f"No models found for dataset: {dataset_name}")
        return

    print(f"Models found: {', '.join(models)}")
    print(f"Using: {'Mean ± Std' if use_mean else 'Best Run'}")

    if output_dir is None:
        output_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Generate all four table variants
    tables = {
        'latex_main_table.tex':               generate_latex_main_table(results_dir, dataset_name, models, use_mean, arrange_best=False),
        'latex_main_table_arranged.tex':      generate_latex_main_table(results_dir, dataset_name, models, use_mean, arrange_best=True),
        'latex_classwise_table.tex':          generate_latex_classwise_table(results_dir, dataset_name, models, use_mean, arrange_best=False),
        'latex_classwise_table_arranged.tex': generate_latex_classwise_table(results_dir, dataset_name, models, use_mean, arrange_best=True),
    }

    for filename, content in tables.items():
        if content:
            path = os.path.join(output_dir, filename)
            with open(path, 'w') as f:
                f.write(content)
            label = filename.replace('.tex', '').replace('_', ' ').title()
            print(f"\n{label} saved: {path}")
            print(f"\n--- {label} ---")
            print(content)

    # Combined files
    for suffix, arranged in [('', False), ('_arranged', True)]:
        main_t = tables[f'latex_main_table{suffix}.tex']
        class_t = tables[f'latex_classwise_table{suffix}.tex']
        combined_path = os.path.join(output_dir, f'latex_tables{suffix}.tex')
        with open(combined_path, 'w') as f:
            kind = ' (Best Run by OA)' if arranged else ' (Original)'
            f.write(f"% LaTeX Tables for {dataset_name} Dataset{kind}\n")
            f.write(f"% Generated automatically\n\n")
            if main_t:
                f.write("% Main Results Table\n")
                f.write(main_t)
                f.write("\n\n")
            if class_t:
                f.write("% Class-wise Accuracy Table\n")
                f.write(class_t)
        print(f"\nCombined tables{suffix} saved: {combined_path}")

    print(f"\n{'='*60}")
    print("Score arrangement complete!")
    print(f"{'='*60}\n")


def print_scores_summary(dataset_name, models=None, results_dir="results"):
    """Print a quick console-friendly summary of scores."""
    if models is None:
        models = get_available_models(results_dir, dataset_name)
    if not models:
        print(f"No models found for dataset: {dataset_name}")
        return

    print(f"\n{'='*70}")
    print(f"Results Summary: {dataset_name}")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'OA (%)':<15} {'AA (%)':<15} {'Kappa':<15} {'Runs':<5}")
    print(f"{'-'*70}")

    for model_name in models:
        df = load_model_results(results_dir, dataset_name, model_name)
        if df is None:
            continue

        num_runs = len(df)
        if num_runs > 1:
            oa_str = f"{df['OA'].mean():.2f}±{df['OA'].std():.2f}"
            aa_str = f"{df['AA'].mean():.2f}±{df['AA'].std():.2f}"
            kappa_str = f"{df['Kappa'].mean()*100:.2f}±{df['Kappa'].std()*100:.2f}"
        else:
            oa_str = f"{df['OA'].iloc[0]:.2f}"
            aa_str = f"{df['AA'].iloc[0]:.2f}"
            kappa_str = f"{df['Kappa'].iloc[0]*100:.2f}"

        print(f"{model_name:<20} {oa_str:<15} {aa_str:<15} {kappa_str:<15} {num_runs:<5}")

    print(f"{'='*70}\n")
