"""
Integration helper for map arrangement with main training pipeline
"""

from utils.map_arranger import MapArranger


def arrange_maps_after_training(cfg, dataset_name=None, model_names=None, force_run=False, save_sep_folder=False):
    """
    Arrange maps after training completes (called from main.py if enabled)
    
    Args:
        cfg: Configuration dict from config.yaml
        dataset_name: Optional override for dataset name
        model_names: Optional override for model names
        force_run: If True, run even if enabled=False (for --arrange-only mode)
        save_sep_folder: If True, save individual maps to dataset_maps folder
    """
    
    # Check if map arrangement is enabled (skip check if force_run)
    if not force_run and not cfg.get('map_arrangement.enabled', False):
        return None
    
    print("\n" + "="*60)
    print("Generating Arranged Classification Maps...")
    print("="*60 + "\n")
    
    # Get config from yaml - use nested structure
    map_cfg = cfg.get('map_arrangement', {})

    # Respect the configured CUDA device for arrange-only runs.
    use_cuda = cfg.get('device.use_cuda', True)
    cuda_device = cfg.get('device.cuda_device', 0)
    map_cfg['device'] = f'cuda:{cuda_device}' if use_cuda else 'cpu'
    
    # Override with parameters if provided
    if dataset_name:
        map_cfg['dataset_dir'] = dataset_name
    if model_names:
        map_cfg['models'] = model_names
    
    try:
        arranger = MapArranger(config=map_cfg)
        
        # Get visualization settings from correct location
        vis_cfg = map_cfg.get('visualization', {})
        regenerate = vis_cfg.get('regenerate_maps', False)
        cmap = vis_cfg.get('cmap', None)
        include_gt = vis_cfg.get('include_gt', False)
        
        output_path = arranger.arrange_maps(
            output_path=map_cfg.get('output', {}).get('path'),
            regenerate_maps=regenerate,
            cmap=cmap,
            include_gt=include_gt,
            save_sep_folder=save_sep_folder
        )
        
        if output_path:
            print(f"\nMap arrangement complete: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"\nError arranging maps: {e}\n")
        import traceback
        traceback.print_exc()
        return None
