"""
Standalone script to arrange classification maps from multiple models/runs
"""

import argparse
import json
from utils.map_arranger import create_arranged_maps
from config.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(
        description='Arrange classification maps from multiple models into a grid'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file (loads map_arrangement settings)'
    )
    parser.add_argument(
        '--base-output-dir',
        type=str,
        default=None,
        help='Base output directory containing results (overrides config)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset name (e.g., Utopia) (overrides config)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Model names to include (space-separated) (overrides config)'
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=2,
        help='Number of rows in grid'
    )
    parser.add_argument(
        '--cols',
        type=int,
        default=2,
        help='Number of columns in grid'
    )
    parser.add_argument(
        '--gap',
        type=int,
        default=20,
        help='Pixel gap between maps'
    )
    parser.add_argument(
        '--fontsize',
        type=int,
        default=12,
        help='Font size for labels'
    )
    parser.add_argument(
        '--label-position',
        type=str,
        choices=['top', 'bottom', 'left', 'right'],
        default='top',
        help='Position of labels'
    )
    parser.add_argument(
        '--label-alignment',
        type=str,
        choices=['left', 'center', 'right'],
        default='center',
        help='Alignment of labels'
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=['OA', 'AA', 'Kappa'],
        default='OA',
        help='Metric to use for selection'
    )
    parser.add_argument(
        '--selection',
        type=str,
        choices=['best', 'worst', 'all'],
        default='best',
        help='Selection type: best, worst, or all runs'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for arranged maps (optional)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output image'
    )
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help='Regenerate maps from checkpoints instead of loading PNGs'
    )
    parser.add_argument(
        '--colormap',
        type=str,
        default=None,
        help='Colormap to use for regenerated maps (e.g., jet, viridis, tab20, hot, cool)'
    )
    parser.add_argument(
        '--include-gt',
        action='store_true',
        help='Include ground truth map alongside classification maps'
    )
    
    args = parser.parse_args()
    
    # Load config from yaml
    cfg = load_config(args.config)
    map_cfg = cfg.get('map_arrangement', {})
    
    # Override with command-line arguments
    if args.base_output_dir:
        map_cfg['base_output_dir'] = args.base_output_dir
    if args.dataset:
        map_cfg['dataset_dir'] = args.dataset
    if args.models:
        # Set models as list
        map_cfg['models'] = args.models
        # Set map_type and metric as parallel lists
        map_cfg['map_type'] = [args.selection] * len(args.models)
        map_cfg['metric'] = [args.metric] * len(args.models)
    
    # Build layout config
    if not 'layout' in map_cfg:
        map_cfg['layout'] = {}
    map_cfg['layout']['rows'] = args.rows
    map_cfg['layout']['cols'] = args.cols
    map_cfg['layout']['gap'] = args.gap
    
    # Build labels config
    if not 'labels' in map_cfg:
        map_cfg['labels'] = {}
    map_cfg['labels']['fontsize'] = args.fontsize
    map_cfg['labels']['position'] = args.label_position
    map_cfg['labels']['alignment'] = args.label_alignment
    
    # Build selection config
    if not 'selection' in map_cfg:
        map_cfg['selection'] = {}
    map_cfg['selection']['metric'] = args.metric
    map_cfg['selection']['type'] = args.selection
    
    # Build output config
    if not 'output' in map_cfg:
        map_cfg['output'] = {}
    map_cfg['output']['dpi'] = args.dpi
    if args.output:
        map_cfg['output']['path'] = args.output
    
    dataset_name = map_cfg.get('dataset_dir', 'Utopia')
    models = map_cfg.get('models', [])
    if not isinstance(models, list):
        models = [models]
    
    print("\n" + "="*60)
    print("Classification Map Arranger")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Models: {', '.join(models)}")
    print(f"Grid: {args.rows}x{args.cols}")
    print(f"Selection: {args.selection} by {args.metric}")
    print("="*60 + "\n")
    
    # Create arranger and arrange maps
    from utils.map_arranger import MapArranger
    arranger = MapArranger(config=map_cfg)
    output_path = arranger.arrange_maps(
        output_path=map_cfg.get('output', {}).get('path'),
        regenerate_maps=args.regenerate,
        cmap=args.colormap,
        include_gt=args.include_gt
    )
    
    print(f"\n✓ Done! Maps saved to: {output_path}")


if __name__ == '__main__':
    main()
