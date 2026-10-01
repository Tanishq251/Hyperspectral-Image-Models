"""
Standalone script to arrange classification maps from multiple models/runs.

Usage examples
--------------
# Just the classification-map grid:
  python arrange_maps.py

# Standalone GT map as PDF (default):
  python arrange_maps.py --gt-map --skip-grid --dataset Indian_Pines

# GT map as PNG with legend:
  python arrange_maps.py --gt-map --gt-legend --gt_format png --dataset Indian_Pines

# Multiple datasets in one command:
  python arrange_maps.py --gt-map --gt-legend --datasets Indian_Pines Botswana WHU-Hi-HanChuan
"""

import argparse
from utils.map_arranger import create_arranged_maps
from config.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(
        description='Arrange classification maps from multiple models into a grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── existing flags ──────────────────────────────────────────────────
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--base-output-dir', type=str, default=None,
                        help='Base output directory (overrides config)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset name (overrides config)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        metavar='DATASET',
                        help='One or more dataset names to process in a loop '
                             '(e.g. --datasets Indian_Pines Botswana WHU-Hi-HanChuan). '
                             'Overrides --dataset and config.')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Model names to include (space-separated)')
    parser.add_argument('--rows', type=int, default=2,
                        help='Number of rows in grid')
    parser.add_argument('--cols', type=int, default=2,
                        help='Number of columns in grid')
    parser.add_argument('--gap', type=int, default=20,
                        help='Pixel gap between maps')
    parser.add_argument('--fontsize', type=int, default=26,
                        help='Font size for labels')
    parser.add_argument('--label-position', type=str,
                        choices=['top', 'bottom', 'left', 'right'], default='top',
                        help='Position of labels')
    parser.add_argument('--label-alignment', type=str,
                        choices=['left', 'center', 'right'], default='center',
                        help='Alignment of labels')
    parser.add_argument('--metric', type=str, choices=['OA', 'AA', 'Kappa'],
                        default='OA', help='Metric to use for selection')
    parser.add_argument('--selection', type=str,
                        choices=['best', 'worst', 'all'], default='best',
                        help='Selection type: best, worst, or all runs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for arranged maps (optional)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output image')
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerate maps from checkpoints')
    parser.add_argument('--colormap', type=str, default=None,
                        help='Colormap (e.g. jet, viridis, tab20)')
    parser.add_argument('--include-gt', action='store_true',
                        help='Include GT map inside the arranged grid')
    spy_group = parser.add_mutually_exclusive_group()
    spy_group.add_argument('--spy-colors', action='store_true', default=False,
                        help='Use spectral spy_colors palette (same as used in '
                             'full-image classification maps). '
                             'Mutually exclusive with --colormap.')
    spy_group.add_argument('--no-spy-colors', action='store_true', default=False,
                        help='Explicitly disable spy_colors (use matplotlib colormap instead)')

    # ── NEW: standalone GT map flags ────────────────────────────────────
    parser.add_argument('--gt-map', action='store_true',
                        help='Also save the ground-truth map as a separate file')
    parser.add_argument('--gt-legend', action='store_true',
                        help='Add class-name legend to the standalone GT map')
    parser.add_argument('--gt-format', type=str, default='pdf',
                        choices=['pdf', 'png'],
                        help="Format for the standalone GT map (default: pdf)")
    parser.add_argument('--gt-output', type=str, default=None,
                        help='Custom output path for the standalone GT map')
    parser.add_argument('--with-grid', action='store_true',
                        help='Also run the arranged classification grid '
                             'when --gt-map is specified')
    parser.add_argument('--skip-grid', action='store_true',
                        help='Force-skip the arranged grid (legacy; '
                             '--gt-map already skips it by default)')
    parser.add_argument('--gt-orientation', type=str, default='horizontal',
                        choices=['horizontal', 'vertical'],
                        help='Orientation for the standalone GT map (default: horizontal)')
    parser.add_argument('--add-table', action='store_true',
                        help='Transform the GT legend into a detailed table showing '
                             'Train, Val, Test, and Total sample counts per class. '
                             'Requires --gt-legend to be passed as well.')
    parser.add_argument('--only-colors', action='store_true',
                        help='Extract class names and their corresponding colors, '
                             'printing them as LaTeX \definecolor commands.')

    args = parser.parse_args()

    # ── Resolve dataset list ─────────────────────────────────────────────
    # Priority: --datasets > --dataset > config
    if args.datasets:
        dataset_list = args.datasets
    elif args.dataset:
        dataset_list = [args.dataset]
    else:
        dataset_list = None  # resolved per-dataset from config below

    # ── load base config ─────────────────────────────────────────────────
    cfg     = load_config(args.config)
    map_cfg = cfg.get('map_arrangement', {})

    if args.base_output_dir:
        map_cfg['base_output_dir'] = args.base_output_dir
    if args.models:
        map_cfg['models']    = args.models
        map_cfg['map_type']  = [args.selection] * len(args.models)
        map_cfg['metric']    = [args.metric]    * len(args.models)

    if 'layout' not in map_cfg:
        map_cfg['layout'] = {}
    map_cfg['layout']['rows'] = args.rows
    map_cfg['layout']['cols'] = args.cols
    map_cfg['layout']['gap']  = args.gap

    if 'labels' not in map_cfg:
        map_cfg['labels'] = {}
    map_cfg['labels']['fontsize']   = args.fontsize
    map_cfg['labels']['position']   = args.label_position
    map_cfg['labels']['alignment']  = args.label_alignment

    if 'selection' not in map_cfg:
        map_cfg['selection'] = {}
    map_cfg['selection']['metric'] = args.metric
    map_cfg['selection']['type']   = args.selection

    if 'output' not in map_cfg:
        map_cfg['output'] = {}
    map_cfg['output']['dpi'] = args.dpi
    if args.output:
        map_cfg['output']['path'] = args.output

    # If no datasets from CLI, fall back to config
    if dataset_list is None:
        dataset_list = [map_cfg.get('dataset_dir', 'Utopia')]

    models = map_cfg.get('models', [])
    if not isinstance(models, list):
        models = [models]

    # ── resolve whether to run the grid ─────────────────────────────────
    # --gt-map alone  → grid OFF by default (user just wants GT maps)
    # no --gt-map     → grid ON  (original behavior)
    run_grid = not (args.skip_grid or (args.gt_map and not args.with_grid))

    print("\n" + "="*60)
    print("Classification Map Arranger")
    print("="*60)
    print(f"Datasets : {', '.join(dataset_list)}")
    print(f"Models   : {', '.join(models) if models else '(all discovered)'}")
    if run_grid:
        print(f"Grid     : {args.rows}x{args.cols}")
        print(f"Selection: {args.selection} by {args.metric}")
    else:
        print(f"Grid     : skipped")
    if args.gt_map:
        print(f"GT map   : yes  (format={args.gt_format.upper()}"
              f"{'  legend=yes' if args.gt_legend else ''})")
    color_mode = 'spy_colors' if args.spy_colors else (args.colormap or 'default')
    print(f"Colormode : {color_mode}")
    print("="*60)

    # ── build arranger and loop over datasets ────────────────────────────
    from utils.map_arranger import MapArranger

    for dataset_name in dataset_list:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'─'*60}")

        # Clone map_cfg for this dataset so we don't mutate the shared dict
        ds_cfg = dict(map_cfg)
        ds_cfg['dataset_dir'] = dataset_name

        # ── Colormap mode: spy_colors vs matplotlib ────────────────────────
        # --spy-colors sets use_spy_colors=True, which activates the spy palette
        # in any vis_mode. --colormap always wins (explicit cmap disables spy).
        if args.spy_colors and not args.colormap:
            ds_cfg['use_spy_colors'] = True
        elif args.no_spy_colors:
            ds_cfg['use_spy_colors'] = False
        # else: leave whatever is in config (use_spy_colors default = True)

        arranger = MapArranger(config=ds_cfg)

        # ── 0. color information for LaTeX ──────────────────────────────────
        if args.only_colors:
            class_names = arranger._get_class_names()
            num_classes = len(class_names)
            colors = arranger._get_class_colors(num_classes, cmap_name=args.colormap)
            
            if args.colormap:
                cmap_label = args.colormap
            elif arranger._should_use_spy(args.colormap):
                cmap_label = 'spy_colors'
            else:
                cmap_label = 'tab20'
            print(f"\n% Colors for {dataset_name} (Colormap: {cmap_label})")
            for i, (name, color) in enumerate(zip(class_names, colors)):
                # Clean name: prefix with dataset, only alphanumeric for LaTeX command
                clean_ds = "".join(x for x in dataset_name if x.isalnum())
                clean_name = "".join(x for x in name if x.isalnum())
                label = f"{clean_ds}{clean_name}"
                print(f"\\definecolor{{{label}}}{{rgb}}{{{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}}}")
            
            # If ONLY --only-colors is specified (and not --gt-map or others that imply more output),
            # we can decide to continue or skip. For now, let's keep it flexible.
            if not run_grid and not args.gt_map:
                continue

        # ── 1. arranged classification grid ──────────────────────────────
        if run_grid:
            grid_path = arranger.arrange_maps(
                output_path     = ds_cfg.get('output', {}).get('path') if len(dataset_list) == 1 else None,
                regenerate_maps = args.regenerate,
                cmap            = args.colormap,
                include_gt      = args.include_gt,
            )
            print(f"\n  Grid saved → {grid_path}")
        else:
            print("  (Grid skipped — --skip-grid)")

        # ── 2. standalone GT map ─────────────────────────────────────────
        if args.gt_map:
            # Standalone GT uses a different cmap logic
            gt_path = arranger.save_gt_map(
                output_path    = args.gt_output if len(dataset_list) == 1 else None,
                cmap           = args.colormap,
                include_legend = args.gt_legend,
                fmt            = args.gt_format,
                dpi            = args.dpi,
                orientation    = args.gt_orientation,
                add_table      = args.add_table,
            )
            if gt_path:
                print(f"  GT map saved → {gt_path}")
            else:
                print(f"  GT map failed for '{dataset_name}'")


if __name__ == '__main__':
    main()
