"""
Classification Map Arranger - Combines multiple classification maps into a single figure
with configurable layout, spacing, labels, and colormaps.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from pathlib import Path

# Shared helpers
from utils.data_loader import DatasetLoader, HyperspectralDataset
from utils.visualization import generate_classification_map
from utils.results_io import find_summary_csv as _find_summary_csv
from utils.colormap_helpers import get_colormap as _get_colormap, SPY_COLORS
from matplotlib.colors import ListedColormap

import math

# ──────────────────────────────────────────────
# Publication presets — sensible defaults so you
# don't have to fiddle with 20 config options.
# Use via:  map_arrangement.preset: "paper"
# ──────────────────────────────────────────────
# The figure is built at its printed size (figure_width, in inches) and font
# sizes are derived from that width, so text stays readable in the paper.
PRESETS = {
    'paper': {                      # double-column journal width
        'figure_width': 7.16,
        'base_label_pt': 9,
        'label_position': 'bottom',
        'label_alignment': 'center',
        'gap': 4,
        'row_gap': 4,
        'dpi': 600,
    },
    'poster': {
        'figure_width': 20,
        'base_label_pt': 24,
        'label_position': 'bottom',
        'label_alignment': 'center',
        'gap': 5,
        'row_gap': 5,
        'dpi': 300,
    },
    'presentation': {               # 16:9 slide width
        'figure_width': 13.33,
        'base_label_pt': 18,
        'label_position': 'bottom',
        'label_alignment': 'center',
        'gap': 5,
        'row_gap': 5,
        'dpi': 200,
    },
}

# Auto font size: base_label_pt at the preset's reference width, growing with the
# square root of the figure width (9 pt at 7.16", ~10.4 pt at a 9.5" landscape
# page), so printed text stays close to normal figure-text size at any width.
DEFAULT_BASE_LABEL_PT, DEFAULT_REF_WIDTH = 9.0, 7.16
LEGEND_TO_LABEL = 0.88
LAYOUT_PAD_IN = 0.02                   # constrained-layout padding around each axes
MIN_LABEL_PT, MIN_LEGEND_PT = 7.5, 7   # floor for single-column (3.5") figures
MAX_RASTER_DPI = 1200                  # cap when raising dpi to keep every data pixel
HARD_MIN_LABEL_PT = 6                  # labels are shrunk to fit their map, never below this


def _text_width_pt(text, size, weight='bold'):
    """Rendered width of the widest line of *text* in points."""
    from matplotlib.textpath import TextToPath
    from matplotlib.font_manager import FontProperties
    prop = FontProperties(size=size, weight=weight)
    return max(TextToPath().get_text_width_height_descent(line, prop, ismath=False)[0]
               for line in str(text).split('\n'))


def _wrap_two_lines(name, size):
    """Split a model name onto two lines at its most balanced natural break:
    a space, '_' or '-' when present (e.g. 'Ground\nTruth', 'MambaHSI_\nPlus'),
    otherwise a lower->Upper case change (e.g. 'FuzzySpectral\nMamba')."""
    if '\n' in name:
        return name
    # (index where line 2 starts, index where line 1 ends)
    breaks = [(i + 1, i) for i, ch in enumerate(name[:-1]) if ch == ' ' and i > 0]
    breaks += [(i + 1, i + 1) for i, ch in enumerate(name[:-1]) if ch in '_-' and i > 0]
    if not breaks:
        breaks = [(i, i) for i in range(1, len(name)) if name[i].isupper() and name[i - 1].islower()]
    if not breaks:
        return name
    start, end = min(breaks, key=lambda b: max(_text_width_pt(name[:b[1]], size), _text_width_pt(name[b[0]:], size)))
    return name[:end] + '\n' + name[start:]


class MapArranger:
    """Arrange and combine classification maps from multiple models/runs"""
    
    def __init__(self, base_output_dir=None, dataset_dir=None, model_names=None, config=None):
        """
        Initialize the map arranger.
        
        Args:
            base_output_dir: Base results directory (e.g., 'results')
            dataset_dir: Dataset name (e.g., 'Utopia')
            model_names: List of model names to include
            config: Dict with full visualization config (can include all settings)
        """
        # If config is provided with all settings, use it
        if config and 'base_output_dir' in config:
            self.base_output_dir = config.get('base_output_dir', 'results')
            self.dataset_dir = config.get('dataset_dir', 'Utopia')
            
            # Check if all_models is True - auto-discover all models in results directory
            all_models = config.get('all_models', False)
            if all_models:
                self.model_names = self._discover_all_models()
                print(f"Auto-discovered {len(self.model_names)} models: {self.model_names}")
            else:
                # Handle models as list
                self.model_names = config.get('models', [])
                if not isinstance(self.model_names, list):
                    self.model_names = [self.model_names]
            
            # Get parallel lists for map_type and metric
            map_types = config.get('map_type', [])
            metrics = config.get('metric', [])
            
            # Ensure lists are same length as models. A scalar or a one-element
            # list applies to every model (e.g. map_type: ["worst"]).
            if not isinstance(map_types, list):
                map_types = [map_types] * len(self.model_names)
            elif len(map_types) == 1:
                map_types = map_types * len(self.model_names)
            if not isinstance(metrics, list):
                metrics = [metrics] * len(self.model_names)
            elif len(metrics) == 1:
                metrics = metrics * len(self.model_names)
            for name, lst in (('map_type', map_types), ('metric', metrics)):
                if 1 < len(lst) < len(self.model_names):
                    print(f"Warning: {name} has {len(lst)} entries for {len(self.model_names)} models; "
                          f"remaining models use the default")

            # Pad with defaults if needed
            while len(map_types) < len(self.model_names):
                map_types.append('best')
            while len(metrics) < len(self.model_names):
                metrics.append('OA')
            
            # Create model config dict from parallel lists
            self.model_configs = {}
            for i, model_name in enumerate(self.model_names):
                self.model_configs[model_name] = {
                    'type': map_types[i] if i < len(map_types) else 'best',
                    'metric': metrics[i] if i < len(metrics) else 'OA'
                }
        else:
            # Otherwise use individual parameters
            self.base_output_dir = base_output_dir or 'results'
            self.dataset_dir = dataset_dir or 'Utopia'
            self.model_names = model_names if isinstance(model_names, list) else [model_names]
            self.model_configs = {}
        
        # ── Apply preset first (if provided) ──
        preset_name = (config or {}).get('preset', None)
        preset_defaults = PRESETS.get(preset_name, {}) if preset_name else {}

        # Default config — bigger font, publication-ready out of the box
        self.config = {
            'rows': None,           # None → auto-calculated from model count
            'cols': None,
            'orientation': 'horizontal',
            'gap': 10,
            'row_gap': 15,
            'fontsize': 'auto',     # 'auto' = scaled to figure_width, or a number in pt
            'show_metric': False,   # append e.g. '(OA=95.12%)' under the model name
            'label_position': 'bottom',
            'label_alignment': 'center',
            'label_offset': 0.02,
            'selection_metric': 'OA',
            'selection_type': 'best',
            'dpi': 300,
            'figure_width': 7.16,   # printed figure width in inches
            'max_figure_height': 9.0,  # maps shrink to keep the figure within this
            # Visualization defaults
            'cmap': None,
            'block_background': True,
            'include_gt': False,
            'regenerate_maps': False,
            'vis_mode': 'labeled_only',
            'use_spy_colors': True,
            'include_legend': False,
            'legend_cols': 6,
            'legend_fontsize': None,  # None = auto (0.9 x label fontsize)
        }

        # Reference width for auto font scaling = the preset's own width
        self._preset_width = preset_defaults.get('figure_width')

        # Override with preset values
        if preset_defaults:
            self.config.update(preset_defaults)
            print(f"Using '{preset_name}' preset (figure_width={self.config['figure_width']}in, dpi={self.config['dpi']})")
        
        # Update with provided config
        if config:
            # Handle nested config structure
            if 'layout' in config:
                self.config.update(config['layout'])
            if 'labels' in config:
                self.config['fontsize'] = config['labels'].get('fontsize', self.config['fontsize'])
                self.config['show_metric'] = config['labels'].get('show_metric', self.config['show_metric'])
                self.config['label_position'] = config['labels'].get('position', self.config['label_position'])
                self.config['label_alignment'] = config['labels'].get('alignment', self.config['label_alignment'])
                self.config['label_offset'] = config['labels'].get('offset', self.config.get('label_offset', 0.02))
            if 'selection' in config:
                self.config['selection_metric'] = config['selection'].get('metric', self.config['selection_metric'])
                self.config['selection_type'] = config['selection'].get('type', self.config['selection_type'])
            if 'output' in config:
                self.config['dpi'] = config['output'].get('dpi', self.config['dpi'])
            # Handle visualization config
            if 'visualization' in config:
                vis_cfg = config['visualization']
                self.config['cmap'] = vis_cfg.get('cmap', self.config['cmap'])
                self.config['block_background'] = vis_cfg.get('block_background', self.config['block_background'])
                self.config['include_gt'] = vis_cfg.get('include_gt', self.config['include_gt'])
                self.config['regenerate_maps'] = vis_cfg.get('regenerate_maps', self.config['regenerate_maps'])
                self.config['vis_mode'] = vis_cfg.get('mode', self.config['vis_mode'])
                self.config['use_spy_colors'] = vis_cfg.get('use_spy_colors', self.config['use_spy_colors'])
                self.config['include_legend'] = vis_cfg.get('include_legend', self.config['include_legend'])
                self.config['legend_cols'] = vis_cfg.get('legend_cols', self.config['legend_cols'])
                self.config['legend_fontsize'] = vis_cfg.get('legend_fontsize', self.config['legend_fontsize'])
            
            # Also update with flat config for backward compatibility
            self.config.update({k: v for k, v in config.items() 
                              if k not in ['layout', 'labels', 'selection', 'output', 'visualization',
                                         'base_output_dir', 'dataset_dir', 'models', 'encoder', 'all_models']})
    
    def _discover_all_models(self):
        """Auto-discover all model directories in the results folder for the dataset."""
        dataset_path = os.path.join(self.base_output_dir, self.dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path not found: {dataset_path}")
            return []
        
        models = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            # Check if it's a directory and has run folders or results_summary.csv
            if os.path.isdir(item_path):
                has_runs = any(d.startswith('run_') for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d)))
                has_summary = os.path.exists(os.path.join(item_path, 'results_summary.csv'))
                if has_runs or has_summary:
                    models.append(item)
        
        return sorted(models)
    
    def find_summary_csv(self, model_name):
        """Find the summary CSV for a model"""
        return _find_summary_csv(self.base_output_dir, self.dataset_dir, model_name)
    
    def select_best_run(self, model_name, metric='OA', selection='best'):
        """
        Select run based on metric from summary.csv
        
        Args:
            model_name: Name of the model
            metric: Metric to use ('OA', 'AA', 'Kappa')
            selection: 'best', 'worst', or threshold value (e.g., 89.3)
        
        Returns:
            Tuple of (run_number, run_dir, metric_value)
        """
        csv_path = self.find_summary_csv(model_name)
        
        if not csv_path:
            print(f"Warning: No summary.csv found for {model_name}")
            return None, None, None
        
        df = pd.read_csv(csv_path)
        
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in {csv_path}")
            print(f"Available metrics: {df.columns.tolist()}")
            return None, None, None
        
        # Handle threshold value (numeric)
        try:
            threshold = float(selection)
            # Find run closest to threshold
            df['diff'] = abs(df[metric] - threshold)
            best_idx = df['diff'].idxmin()
        except (ValueError, TypeError):
            # Handle 'best' or 'worst'
            if selection == 'best':
                best_idx = df[metric].idxmax()
            else:  # worst
                best_idx = df[metric].idxmin()
        
        best_row = df.iloc[best_idx]
        run_number = int(best_row['Run'])
        metric_value = best_row[metric]
        
        # Construct run directory
        run_dir = os.path.join(
            self.base_output_dir, 
            self.dataset_dir, 
            model_name, 
            f'run_{run_number}'
        )
        
        return run_number, run_dir, metric_value
    
    def _find_checkpoint(self, run_dir):
        """Find best available checkpoint in run directory."""
        candidates = [
            os.path.join(run_dir, 'best_model.pth'),
            os.path.join(run_dir, 'final_model.pth'),
        ]
        # Add epoch checkpoints (latest first)
        if os.path.exists(run_dir):
            epoch_ckpts = sorted([f for f in os.listdir(run_dir) 
                                  if f.startswith('checkpoint_epoch_') and f.endswith('.pth')], reverse=True)
            candidates.extend([os.path.join(run_dir, ckpt) for ckpt in epoch_ckpts])
        
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None
    
    def _load_checkpoint(self, model, checkpoint_path, device, dataset=None):
        """
        Load model weights from checkpoint.

        Args:
            model: Model instance
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            dataset: Optional dataset to get sample input for lazy initialization
        """
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # For models with lazy initialization (like HybridSN), do a dummy forward pass first
        # This initializes any dynamically created layers
        if hasattr(model, '_initialized') and not model._initialized and dataset is not None:
            try:
                sample_input, _ = dataset[0]
                sample_input = sample_input.unsqueeze(0).to(device)
                with torch.no_grad():
                    _ = model(sample_input)
                print("  Initialized model with dummy forward pass")
            except Exception as e:
                print(f"  Warning: Could not initialize model with forward pass: {e}")

        # Try strict loading first
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # If strict loading fails, try non-strict loading
            print(f"  Warning: Strict checkpoint loading failed. Attempting flexible loading...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"    Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"    Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

        model.eval()
        return model
    
    def load_classification_map(self, run_dir, regenerate=False, model=None, dataset=None, device='cuda', cmap=None):
        """
        Load classification map image from run directory
        
        Args:
            run_dir: Run directory path
            regenerate: If True, regenerate map from checkpoint instead of loading PNG
            model: Model instance (required if regenerate=True)
            dataset: Dataset instance (required if regenerate=True)
            device: Device to use for inference
            cmap: Colormap to use for regeneration
        
        Returns:
            PIL Image or None
        """
        # Determine visualization mode
        vis_mode = self.config.get('vis_mode', 'labeled_only')
        # spy_colors is requested either via cmap='spy_colors' alias or use_spy_colors=True
        use_spy_colors = self._should_use_spy(cmap)

        # Determine map paths based on mode. spy maps are written with a
        # '_spy' suffix by generate_classification_map(); a real matplotlib
        # cmap is written with a '_{cmap}' suffix.
        if vis_mode == 'full_image':
            original_map_path = os.path.join(run_dir, 'classification_map_full.png')
            if use_spy_colors:
                cmap_specific_path = os.path.join(run_dir, 'classification_map_full_spy.png')
            elif cmap:
                cmap_specific_path = os.path.join(run_dir, f'classification_map_full_{cmap}.png')
            else:
                cmap_specific_path = None
        else:
            # Labeled-only mode (default)
            original_map_path = os.path.join(run_dir, 'classification_map.png')
            if use_spy_colors:
                cmap_specific_path = os.path.join(run_dir, 'classification_map_spy.png')
            elif cmap:
                cmap_specific_path = os.path.join(run_dir, f'classification_map_{cmap}.png')
            else:
                cmap_specific_path = None

        # Determine final map path for saving/loading
        map_path = cmap_specific_path if cmap_specific_path else original_map_path

        # Check if the requested map exists
        map_exists = os.path.exists(map_path)

        # If not regenerating AND map exists, try to load it
        if not regenerate and map_exists:
            print(f"Loading existing {vis_mode} map: {os.path.basename(map_path)}")
            return Image.open(map_path)

        # Auto-regenerate if:
        # 1. User explicitly requested regeneration (regenerate=True), OR
        # 2. The requested map doesn't exist (switching modes or first time)
        needs_regeneration = regenerate or not map_exists

        if needs_regeneration and model is not None and dataset is not None:
            if not map_exists:
                print(f"  → Map not found for mode '{vis_mode}': {os.path.basename(map_path)}")
            print(f"  → Regenerating {vis_mode} map for {os.path.basename(run_dir)} (cmap={cmap})...")
            
            # Find best checkpoint
            best_checkpoint = self._find_checkpoint(run_dir)
            if not best_checkpoint:
                print(f"Warning: No checkpoint found in {run_dir}")
                return Image.open(original_map_path) if os.path.exists(original_map_path) else None
            
            print(f"Using checkpoint: {os.path.basename(best_checkpoint)}")
            
            # Load checkpoint and generate map
            try:
                model = self._load_checkpoint(model, best_checkpoint, device, dataset=dataset)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return Image.open(original_map_path) if os.path.exists(original_map_path) else None
            
            # Generate map using imported generate_classification_map.
            # If spy_colors was requested (via alias or use_spy_colors), generate
            # the spy version and skip the matplotlib-cmap render (the 'spy_colors'
            # alias is NOT a real matplotlib colormap).
            real_cmap = None if use_spy_colors else cmap
            try:
                generate_classification_map(
                    model=model,
                    dataset=dataset,
                    device=device,
                    run_dir=run_dir,
                    cmap=real_cmap or 'tab20',
                    show_colorbar=False,
                    dpi=self.config.get('dpi', 300),
                    block_background=self.config.get('block_background', True),
                    mode=vis_mode,
                    use_spy_colors=use_spy_colors,
                    use_cmap=not use_spy_colors,
                )
                
                # generate_classification_map now saves as classification_map_{cmap}.png directly
                # Load the newly generated map
                if os.path.exists(map_path):
                    print(f"Generated map with colormap '{cmap}': {map_path}")
                    return Image.open(map_path)
                    
            except Exception as e:
                print(f"Error generating map: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Fallback: try to load existing map (original or colormap-specific)
        if os.path.exists(map_path):
            return Image.open(map_path)
        
        original_map = os.path.join(run_dir, 'classification_map.png')
        if os.path.exists(original_map):
            return Image.open(original_map)
        
        print(f"Warning: Classification map not found at {map_path}")
        return None
    
    def load_colormap_from_config(self, run_dir):
        """Load colormap from training config if available"""
        config_path = os.path.join(run_dir, 'config.yaml')
        
        if not os.path.exists(config_path):
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('visualization', {}).get('cmap', None)
        except:
            return None
    
    def get_colormap(self, run_dir, cmap_name=None):
        """
        Get colormap - either from config or use default
        
        Args:
            run_dir: Run directory
            cmap_name: Optional colormap name to override
        
        Returns:
            matplotlib colormap
        """
        if cmap_name:
            try:
                return plt.get_cmap(cmap_name)
            except:
                pass
        
        # Try to load from config
        config_cmap = self.load_colormap_from_config(run_dir)
        if config_cmap:
            try:
                return plt.get_cmap(config_cmap)
            except:
                pass
        
        # Default to tab20
        return plt.get_cmap('tab20')
    
    def load_run_config(self, run_dir):
        """Load config.yaml from run directory"""
        config_path = os.path.join(run_dir, 'config.yaml')
        
        if not os.path.exists(config_path):
            return None
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return None
    
    def load_dataset_for_run(self, run_dir, dataset_name):
        """
        Load dataset for a specific run using the saved config.
        Uses DatasetLoader and HyperspectralDataset from utils.data_loader.
        """
        try:
            run_cfg = self.load_run_config(run_dir) or {}
            
            # Load raw data using cached loader
            loader = DatasetLoader(use_cache=True)
            data, gt = loader.load_dataset(dataset_name)
            
            # Get preprocessing params from saved config
            dataset_cfg = run_cfg.get('dataset', {})
            preproc_cfg = run_cfg.get('preprocessing', {})
            
            return HyperspectralDataset(
                data=data,
                gt=gt,
                patch_size=dataset_cfg.get('patch_size', 11),
                stride=dataset_cfg.get('stride', 1),
                dim_reduction_method=preproc_cfg.get('dim_reduction_method', 'pca'),
                num_pca_bands=preproc_cfg.get('num_pca_bands', 30),
                use_channel_dim=preproc_cfg.get('use_channel_dim', True),
                verbose=False
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_ground_truth_map(self, dataset_name, cmap=None):
        """Load and visualize ground truth map using DatasetLoader."""
        try:
            loader = DatasetLoader(use_cache=True)
            _, gt = loader.load_dataset(dataset_name)
            
            # Handle gt orientation
            if gt.ndim == 3:
                gt = np.squeeze(gt)
            
            # Render with the same routine, figure size and dpi as the prediction
            # maps so the GT panel has the same colours and resolution.
            from utils.visualization import _save_with_matplotlib
            import tempfile
            fd, temp_path = tempfile.mkstemp(suffix='_gt_map.png')
            os.close(fd)
            try:
                _save_with_matplotlib(gt, temp_path, self._get_colormap_instance(cmap),
                                      show_colorbar=False, dpi=self.config.get('dpi', 300),
                                      block_background=self.config.get('block_background', True))
                with Image.open(temp_path) as im:
                    return im.copy()
            finally:
                os.remove(temp_path)
        except Exception as e:
            print(f"Error loading ground truth map: {e}")
            return None
    
    def _get_colormap_instance(self, cmap_name):
        """Get matplotlib colormap instance from name.

        Treats 'spy_colors'/'spy' as an alias for the spy palette rather than
        a matplotlib colormap name.
        """
        if self._should_use_spy(cmap_name):
            from utils.colormap_helpers import get_spy_cmap
            return get_spy_cmap()
        return _get_colormap(cmap_name)
    
    def _load_model_for_regeneration(self, model_name, run_dir):
        """Load model and dataset for map regeneration using models registry."""
        try:
            from models import create_model, InputShapeWrapper
            
            run_cfg = self.load_run_config(run_dir) or {}
            dataset = self.load_dataset_for_run(run_dir, self.dataset_dir)
            
            if not run_cfg:
                return None, dataset
            
            # Get model params from config or dataset
            num_classes = run_cfg.get('num_classes')
            if num_classes is None and dataset is not None:
                num_classes = max(len(np.unique(dataset.gt)) - 1, 1)
            num_classes = num_classes or 9
            
            preproc = run_cfg.get('preprocessing', {})
            dataset_cfg = run_cfg.get('dataset', {})
            bands = run_cfg.get('bands') or preproc.get('num_pca_bands', 30)
            patch_size = dataset_cfg.get('patch_size', 11)
            
            print(f"Creating model: num_classes={num_classes}, bands={bands}, patch_size={patch_size}")
            
            model_cfg, model = create_model(
                model_name, 
                num_classes=num_classes, 
                bands=bands,
                patch_size=patch_size,
                return_config=True
            )
            
            # Wrap model if it expects 4D input
            if model_cfg.get('expects_4d', False):
                model = InputShapeWrapper(model)
            
            return model, dataset
        except Exception as e:
            print(f"Warning: Could not load model/dataset: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _check_regeneration_needed(self, run_dir, model_name, cmap, force_regenerate=False):
        """
        Check if map regeneration is needed based on colormap and visualization mode.
        Returns True if regeneration is needed, False if existing map can be used.

        Logic:
        - If force_regenerate is True -> regenerate
        - Check for mode-specific map file based on vis_mode config
        - If file exists -> skip regeneration
        - Otherwise -> regenerate
        """
        if force_regenerate:
            print(f"⚡ {model_name}: Force regenerate requested")
            return True

        # Determine visualization mode and expected filename
        vis_mode = self.config.get('vis_mode', 'labeled_only')
        use_spy_colors = self._should_use_spy(cmap)

        if vis_mode == 'full_image':
            # Full image mode
            if use_spy_colors:
                expected_path = os.path.join(run_dir, 'classification_map_full_spy.png')
            elif cmap:
                expected_path = os.path.join(run_dir, f'classification_map_full_{cmap}.png')
            else:
                expected_path = os.path.join(run_dir, 'classification_map_full.png')
        else:
            # Labeled-only mode
            if use_spy_colors:
                expected_path = os.path.join(run_dir, 'classification_map_spy.png')
            elif cmap:
                expected_path = os.path.join(run_dir, f'classification_map_{cmap}.png')
            else:
                expected_path = os.path.join(run_dir, 'classification_map.png')

        # Check if the expected map exists
        if os.path.exists(expected_path):
            print(f"{model_name}: Using existing {vis_mode} map: {os.path.basename(expected_path)}")
            return False

        # Need to regenerate - mode-specific file doesn't exist
        print(f"🔄 {model_name}: Need to generate {vis_mode} map: {os.path.basename(expected_path)}")
        return True
    
    def _load_gt_array(self):
        """Ground-truth label array for the current dataset (cached)."""
        if getattr(self, '_gt_cache', None) is None:
            _, gt = DatasetLoader(use_cache=True).load_dataset(self.dataset_dir)
            self._gt_cache = np.squeeze(gt).astype(int)
        return self._gt_cache

    def _load_label_map(self, run_dir):
        """Raw per-pixel prediction labels saved by generate_classification_map, or None."""
        path = os.path.join(run_dir, 'prediction_map.npy')
        return np.load(path) if os.path.exists(path) else None

    def _map_shape(self, map_data):
        """(height, width) of a map in data pixels."""
        if map_data.get('labels') is not None:
            return map_data['labels'].shape[:2]
        w, h = map_data['image'].size
        return h, w

    def _draw_panel(self, ax, map_data, cmap, orientation):
        """Draw one map. Label arrays are drawn pixel-exact with the same colour
        rules as visualization._save_with_matplotlib; PNGs are drawn unsmoothed."""
        labels = map_data.get('labels')
        if labels is None:
            img = map_data['image']
            if orientation == 'vertical':
                img = img.rotate(-90, expand=True)
            ax.imshow(np.asarray(img.convert('RGB')), interpolation='none')
            return

        arr = labels.astype(int).copy()
        if map_data.get('is_gt') or self.config.get('vis_mode', 'labeled_only') != 'full_image':
            gt = self._load_gt_array()
            if gt.shape == arr.shape:
                arr[gt == 0] = 0
        if orientation == 'vertical':
            arr = np.rot90(arr, k=-1)

        cm = self._get_colormap_instance(cmap)
        if isinstance(cm, ListedColormap):
            vmin, vmax = 0, cm.N - 1
        else:
            vmin, vmax = None, None
        if self.config.get('block_background', True):
            cm.set_bad(color='black')
            ax.imshow(np.ma.masked_where(arr == 0, arr), cmap=cm, vmin=vmin, vmax=vmax, interpolation='none')
        else:
            if vmax is None:
                vmin, vmax = 0, (arr.max() if arr.max() > 0 else None)
            ax.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax, interpolation='none')

    def _collect_single_run(self, model_name, run_dir, run_number, metric_value, metric, cmap, device, regenerate_maps):
        """Helper to collect a single run's map data."""
        labels = None if regenerate_maps else self._load_label_map(run_dir)
        img = None
        if labels is None:
            should_regenerate = self._check_regeneration_needed(run_dir, model_name, cmap, regenerate_maps)

            model, dataset = None, None
            if should_regenerate:
                model, dataset = self._load_model_for_regeneration(model_name, run_dir)
                should_regenerate = model is not None

            img = self.load_classification_map(run_dir, regenerate=should_regenerate, model=model, dataset=dataset, device=device, cmap=cmap)
            labels = self._load_label_map(run_dir)  # written by a regeneration
        else:
            print(f"{model_name}: drawing from prediction_map.npy")
        if img or labels is not None:
            # Create label with model name and metric value. Kappa is stored as
            # 0-1; show it x100 without '%' to match the LaTeX tables.
            if metric_value is None or not self.config.get('show_metric', False):
                label = model_name
            elif metric == 'Kappa':
                label = f"{model_name}\n(Kappa={metric_value * 100:.2f})"
            else:
                label = f"{model_name}\n({metric}={metric_value:.2f}%)"
            return {
                'model_name': model_name,
                'display_label': label,
                'run_number': run_number,
                'run_dir': run_dir,
                'image': img,
                'labels': labels,
                'metric_value': metric_value,
                'metric_name': metric,
                'is_gt': False
            }
        return None
    
    def collect_maps(self, regenerate_maps=False, cmap=None, include_gt=False):
        """Collect classification maps for all models based on per-model selection criteria."""
        maps_data = []
        device = self.config.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Add ground truth if requested
        if include_gt:
            try:
                gt_labels, gt_image = self._load_gt_array(), None
            except Exception as e:
                print(f"Could not load GT labels ({e}); using rendered GT image")
                gt_labels, gt_image = None, self.load_ground_truth_map(self.dataset_dir, cmap=cmap)
            if gt_labels is not None or gt_image:
                maps_data.append({
                    'model_name': 'Ground Truth', 'run_number': None, 'run_dir': None,
                    'image': gt_image, 'labels': gt_labels,
                    'metric_value': None, 'metric_name': None, 'is_gt': True
                })
        
        if regenerate_maps or cmap:
            print(f"Will regenerate maps if needed with device={device}")
        
        for model_name in self.model_names:
            model_cfg = self.model_configs.get(model_name, {})
            metric = model_cfg.get('metric', 'OA')
            selection = model_cfg.get('type', 'best')
            
            if selection == 'all':
                csv_path = self.find_summary_csv(model_name)
                if csv_path:
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        run_number = int(row['Run'])
                        run_dir = os.path.join(self.base_output_dir, self.dataset_dir, model_name, f'run_{run_number}')
                        map_data = self._collect_single_run(model_name, run_dir, run_number, row.get(metric, 0), metric, cmap, device, regenerate_maps)
                        if map_data:
                            maps_data.append(map_data)
            else:
                run_number, run_dir, metric_value = self.select_best_run(model_name, metric=metric, selection=selection)
                if run_dir:
                    map_data = self._collect_single_run(model_name, run_dir, run_number, metric_value, metric, cmap, device, regenerate_maps)
                    if map_data:
                        maps_data.append(map_data)
        
        return maps_data
    
    def _get_class_names(self):
        """Load class names for the current dataset from dataset.yaml."""
        try:
            config_path = os.path.join('config', 'dataset.yaml')
            with open(config_path, 'r') as f:
                ds_config = yaml.safe_load(f)
            ds_info = ds_config.get('datasets', {}).get(self.dataset_dir, {})
            return ds_info.get('class_names', [])
        except Exception:
            return []

    def _should_use_spy(self, cmap_name=None):
        """Determine if spy_colors should be used for coloring.

        Spy colours are chosen when EITHER trigger is set:
        1. cmap is explicitly named 'spy_colors'/'spy' (alias, not a matplotlib cmap)
        2. use_spy_colors: True in config (works in any vis_mode)

        An explicit *real* matplotlib cmap (e.g. 'jet') always wins and
        disables spy colours.
        """
        if cmap_name is not None:
            # 'spy_colors'/'spy' is an alias meaning "use the spy palette",
            # not a matplotlib colormap name.
            return str(cmap_name).lower() in ('spy_colors', 'spy')
        return bool(self.config.get('use_spy_colors', True))

    def _get_class_colors(self, num_classes, cmap_name=None):
        """Get RGB colors for each class (1-indexed, matching the maps).

        Uses the same priority as the actual map files:
        - spy_colors only when no explicit cmap AND use_spy_colors + full_image
        - matplotlib colormap otherwise
        """
        # Mirror visualization._save_with_matplotlib: a ListedColormap is drawn
        # with vmin=0, vmax=N-1, so class i gets palette entry i (clipped to the
        # last entry); a continuous cmap is auto-scaled over classes 1..C when the
        # background is masked, or over 0..C when it is drawn (block_background=False).
        if self._should_use_spy(cmap_name):
            palette = SPY_COLORS / 255.0
            return [list(palette[min(i, len(palette) - 1)]) for i in range(1, num_classes + 1)]

        cm = _get_colormap(cmap_name)
        if isinstance(cm, ListedColormap):
            palette = np.asarray(cm.colors)
            return [list(palette[min(i, len(palette) - 1)][:3]) for i in range(1, num_classes + 1)]
        if self.config.get('block_background', True):
            denom = max(num_classes - 1, 1)
            return [list(cm((i - 1) / denom)[:3]) for i in range(1, num_classes + 1)]
        return [list(cm(i / max(num_classes, 1))[:3]) for i in range(1, num_classes + 1)]

    def _num_classes(self):
        """Number of classes for the current dataset (from dataset.yaml)."""
        names = self._get_class_names()
        if names:
            return len(names)
        try:
            with open(os.path.join('config', 'dataset.yaml'), 'r') as f:
                ds = yaml.safe_load(f).get('datasets', {}).get(self.dataset_dir, {})
            return int(ds.get('num_classes', 0))
        except Exception:
            return 0

    def _resolve_cmap(self, cmap):
        """Switch to spy_colors when a discrete cmap has too few colours for the dataset.

        A ListedColormap with N entries can only show classes 1..N-1 distinctly;
        higher classes are clipped onto the last colour (e.g. tab20 merges
        classes 19 and 20 on Houston18).
        """
        if self._should_use_spy(cmap):
            return cmap
        n = self._num_classes()
        cm = _get_colormap(cmap)  # None -> tab20, same as the renderer
        if n and isinstance(cm, ListedColormap) and n > cm.N - 1:
            print(f"Warning: colormap '{cmap or 'tab20'}' has {cm.N} colours, too few for {n} classes "
                  f"(classes {cm.N - 1}..{n} would share a colour). Using spy_colors instead.")
            return 'spy_colors'
        return cmap

    @staticmethod
    def _legend_height(n_classes, legend_cols, legend_fs):
        """Height in inches of a legend with n_classes entries in legend_cols columns."""
        return math.ceil(n_classes / legend_cols) * legend_fs * 1.45 / 72 + 0.15

    def _font_sizes(self, figure_width=None):
        """(label, legend) font sizes in pt. 'auto' scales with the printed figure width."""
        width = figure_width or self.config.get('figure_width', 7.16)
        fs = self.config.get('fontsize', 'auto')
        if fs in (None, 'auto'):
            base = float(self.config.get('base_label_pt', DEFAULT_BASE_LABEL_PT))
            ref = float(self._preset_width or DEFAULT_REF_WIDTH)
            label_fs = max(base * math.sqrt(width / ref), MIN_LABEL_PT)
        else:
            label_fs = float(fs)
        legend_fs = self.config.get('legend_fontsize')
        legend_fs = max(LEGEND_TO_LABEL * label_fs, MIN_LEGEND_PT) if legend_fs in (None, 'auto') else float(legend_fs)
        return label_fs, legend_fs

    def _draw_legend(self, target, cmap_name=None):
        """Draw a class-color legend below the figure.

        Reads class names from dataset.yaml, gets corresponding colors,
        and adds a row of (color-patch + label) entries laid out in N columns.
        """
        class_names = self._get_class_names()
        if not class_names:
            print("Warning: No class names found in dataset.yaml — skipping legend")
            return

        num_classes = len(class_names)
        colors = self._get_class_colors(num_classes, cmap_name)

        legend_cols = getattr(self, '_legend_cols', None) or self.config.get('legend_cols', 6)
        legend_fontsize = getattr(self, '_legend_fs', None) or self._font_sizes()[1]

        # Build legend handles
        from matplotlib.patches import Patch
        handles = []
        for i, name in enumerate(class_names):
            color = colors[i] if i < len(colors) else [0.5, 0.5, 0.5]
            handles.append(Patch(facecolor=color, edgecolor='black', linewidth=0.5, label=name))

        # target is the figure or a dedicated (axis-off) legend axes
        legend = target.legend(
            handles=handles,
            loc='center',
            ncol=legend_cols,
            fontsize=legend_fontsize,
            frameon=True,
            fancybox=True,
            shadow=False,
            borderpad=0.8,
            columnspacing=1.2,
            handletextpad=0.5,
            handlelength=1.5,
        )
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_edgecolor('gray')

        print(f"Legend added: {num_classes} classes in {legend_cols} columns")

    def _export_full_map_image(self, model_name, run_dir, regenerate_maps=False, cmap=None, device=None):
        """Load or regenerate a full-image map for separate export."""
        if run_dir is None:
            return None

        device = device or self.config.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        original_mode = self.config.get('vis_mode', 'labeled_only')

        try:
            self.config['vis_mode'] = 'full_image'
            should_regenerate = self._check_regeneration_needed(run_dir, model_name, cmap, regenerate_maps)
            model, dataset = None, None
            if should_regenerate:
                model, dataset = self._load_model_for_regeneration(model_name, run_dir)
                should_regenerate = model is not None and dataset is not None

            return self.load_classification_map(
                run_dir,
                regenerate=should_regenerate,
                model=model,
                dataset=dataset,
                device=device,
                cmap=cmap,
            )
        finally:
            self.config['vis_mode'] = original_mode

    def _save_separate_maps(self, maps_data, regenerate_maps=False, cmap=None):
        """Save per-model full maps and ground truth into a separate dataset folder."""
        output_dir = os.path.join(self.base_output_dir, f'{self.dataset_dir}_maps')
        os.makedirs(output_dir, exist_ok=True)

        device = self.config.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')

        for map_data in maps_data:
            if map_data.get('is_gt'):
                continue

            model_name = map_data['model_name']
            export_name = f"{model_name}_Full_map.png"
            export_path = os.path.join(output_dir, export_name)

            full_img = self._export_full_map_image(
                model_name=model_name,
                run_dir=map_data.get('run_dir'),
                regenerate_maps=regenerate_maps,
                cmap=cmap,
                device=device,
            )

            image_to_save = full_img or map_data.get('image')
            if image_to_save is None:
                print(f"Warning: Could not export map for {model_name}")
                continue

            image_to_save.save(export_path)
            print(f"Saved separate full map: {export_path}")

        gt_path = os.path.join(output_dir, f'{self.dataset_dir}_GT_map.png')
        self.save_gt_map(
            output_path=gt_path,
            cmap=cmap,
            include_legend=False,
            fmt='png',
            orientation=self.config.get('orientation', 'horizontal'),
        )

    def arrange_maps(self, output_path=None, regenerate_maps=None, cmap=None, include_gt=None, save_sep_folder=False):
        """
        Arrange collected maps into a grid layout
        
        Args:
            output_path: Path to save the arranged figure
            regenerate_maps: If True, regenerate maps from checkpoints (None = use config value)
            cmap: Colormap to use for regeneration (None = use config value)
            include_gt: If True, include ground truth map alongside predictions (None = use config value)
            save_sep_folder: If True, save individual maps to dataset_maps/model_full_map.png
        
        Returns:
            Path to saved figure
        """
        # Use config values if not explicitly provided
        if regenerate_maps is None:
            regenerate_maps = self.config.get('regenerate_maps', False)
        if cmap is None:
            cmap = self.config.get('cmap')
        if include_gt is None:
            include_gt = self.config.get('include_gt', False)
        cmap = self._resolve_cmap(cmap)

        maps_data = self.collect_maps(regenerate_maps=regenerate_maps, cmap=cmap, include_gt=include_gt)

        if not maps_data:
            print("Error: No maps collected")
            return None

        n_maps = len(maps_data)
        orientation = self.config.get('orientation', 'horizontal')
        label_pos = self.config['label_position']
        label_align = self.config['label_alignment']
        dpi = self.config.get('dpi', 300)
        col_gap = self.config.get('gap', 4) / 100
        row_gap = self.config.get('row_gap', 4) / 100

        # Map shape (from the label array when available, else the PNG)
        data_h, data_w = self._map_shape(maps_data[0])
        if orientation == 'vertical':
            data_h, data_w = data_w, data_h
        aspect_ratio = data_w / data_h

        # ── Printed figure width; fonts follow that width ──
        fig_width = float(self.config.get('figure_width', 7.16))
        max_height = float(self.config.get('max_figure_height', 9.0))
        fontsize, legend_fs = self._font_sizes(fig_width)
        self._legend_fs = legend_fs

        label_lines = max(str(m.get('display_label', m['model_name'])).count('\n') + 1 for m in maps_data)
        label_in = label_lines * fontsize * 1.3 / 72 + 0.06
        side_labels = label_pos in ('left', 'right')
        label_h = 0 if side_labels else label_in

        include_legend = self.config.get('include_legend', False) and bool(self._get_class_names())
        legend_in = 0.0
        if include_legend:
            names = self._get_class_names()
            # Fit as many legend columns as the width allows (up to legend_cols)
            # measured name width + patch, gaps and frame (legend kwargs in _draw_legend)
            col_w = (max(_text_width_pt(n, legend_fs, weight='normal') for n in names)
                     + (1.5 + 0.5 + 1.2) * legend_fs) / 72
            legend_cols = max(1, min(self.config.get('legend_cols', 6), int(fig_width // col_w)))
            self._legend_cols = legend_cols
            legend_in = self._legend_height(len(names), legend_cols, legend_fs)

        labels_text = [m.get('display_label', m['model_name']) for m in maps_data]
        widest_pt = max(_text_width_pt(t, fontsize) for t in labels_text)
        # Widest label if long names were wrapped onto two lines
        widest_wrapped_pt = max(_text_width_pt(_wrap_two_lines(t, fontsize), fontsize) for t in labels_text)

        def label_font_for(w, h, widest=None):
            """Label size that fits within a w x h (in) map's width (its height for
            left/right labels), so names never touch a neighbour or the figure edge."""
            avail_pt = (h if side_labels else w) * 72 * 0.97
            return fontsize * min(1.0, avail_pt / (widest or widest_pt))

        def panel_size(r, c):
            """Largest map panel (w, h) in inches for an r x c grid within the page limits."""
            pad = 2 * LAYOUT_PAD_IN
            w = (fig_width - (c * label_in if side_labels else 0) - c * pad - pad) / (c + (c - 1) * col_gap)
            h = w / aspect_ratio
            avail_h = max_height - r * label_h - legend_in - 0.1 - (r + int(include_legend)) * pad
            if r * h + (r - 1) * row_gap * h > avail_h:
                h = avail_h / (r + (r - 1) * row_gap)
                w = h * aspect_ratio
            return w, h

        # ── Grid: pick the rows x cols that gives the biggest maps for this shape ──
        rows, cols = self.config['rows'], self.config['cols']
        if rows is None or cols is None:
            def evaluate(c):
                r = math.ceil(n_maps / c)
                w, h = panel_size(r, c)
                fig_h = (r * (h + label_h) + (r - 1) * row_gap * h + legend_in + 0.1
                         + (r + int(include_legend)) * 2 * LAYOUT_PAD_IN)
                fill = n_maps * w * h / (fig_width * fig_h)
                # Avoid grids whose names would need tiny text to fit under their maps
                readable = label_font_for(w, h, widest_wrapped_pt) >= MIN_LABEL_PT
                # Big maps, but also a figure mostly made of maps
                return dict(c=c, area=w * h, fig_h=fig_h, readable=readable,
                            score=w * h * fill ** 2 * (1.0 if readable else 1e-3))
            options = [evaluate(c) for c in range(1, n_maps + 1)]
            best = max(options, key=lambda o: o['score'])
            # If the best grid only fits by hitting max_figure_height (maps shrunk
            # to fit the page), prefer a compact grid whose maps are still at least
            # 40% as large, e.g. 5 wide Trento maps as 3x2 instead of a page-tall
            # single column. Tall scenes (Salinas) that fit keep their big maps.
            compact = [o for o in options if o['readable'] and o['fig_h'] <= 0.75 * max_height
                       and o['area'] >= 0.4 * best['area']]
            if compact and best['fig_h'] >= max_height - 0.01:
                best = max(compact, key=lambda o: o['score'])
            cols = best['c']
            rows = math.ceil(n_maps / cols)
            print(f"Auto-grid: {rows}×{cols} for {n_maps} maps (aspect {aspect_ratio:.2f})")
        elif rows * cols < n_maps:
            new_rows = math.ceil(n_maps / cols)
            print(f"Warning: {rows}×{cols} grid holds {rows * cols} maps but {n_maps} were collected; "
                  f"using {new_rows}×{cols} so no model is dropped")
            rows = new_rows

        panel_w, panel_h = panel_size(rows, cols)
        # Shrink the label font so the longest model name fits its map
        fitted = label_font_for(panel_w, panel_h)
        if fitted < MIN_LABEL_PT and not side_labels:
            # Too long for one line at a readable size: wrap the names that don't fit
            avail_pt = panel_w * 72 * 0.97
            for m in maps_data:
                text = m.get('display_label', m['model_name'])
                if _text_width_pt(text, MIN_LABEL_PT) > avail_pt:
                    m['display_label'] = _wrap_two_lines(text, MIN_LABEL_PT)
            labels_text = [m.get('display_label', m['model_name']) for m in maps_data]
            widest_pt = max(_text_width_pt(t, fontsize) for t in labels_text)
            label_lines = max(t.count('\n') + 1 for t in labels_text)
            fitted = label_font_for(panel_w, panel_h)
        if fitted < fontsize or label_lines * fontsize * 1.3 / 72 + 0.06 != label_in:
            if fitted < HARD_MIN_LABEL_PT:
                print(f"Warning: {rows}×{cols} maps are too small for the model names even at "
                      f"{HARD_MIN_LABEL_PT} pt; use fewer columns or a wider figure_width")
            fontsize = max(fitted, HARD_MIN_LABEL_PT)
            label_in = label_lines * fontsize * 1.3 / 72 + 0.06
            label_h = 0 if side_labels else label_in
            panel_w, panel_h = panel_size(rows, cols)
        fig_height = (rows * (panel_h + label_h) + (rows - 1) * row_gap * panel_h + legend_in + 0.1
                      + (rows + int(include_legend)) * 2 * LAYOUT_PAD_IN)
        # Raster output: raise dpi so each data pixel gets at least one output pixel
        dpi = int(min(max(dpi, math.ceil(data_w / panel_w)), max(dpi, MAX_RASTER_DPI)))
        print(f"Figure {fig_width:.2f}x{fig_height:.2f} in | map {panel_w:.2f}x{panel_h:.2f} in | "
              f"labels {fontsize:.1f} pt | legend {legend_fs:.1f} pt | dpi {dpi}")

        # Constrained layout reserves room for the panel labels, so multi-line
        # labels no longer run into the next row of maps. The legend gets its
        # own grid row so it cannot overlap the labels of the last row.
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, layout='constrained')
        fig.get_layout_engine().set(wspace=col_gap, hspace=row_gap, w_pad=LAYOUT_PAD_IN, h_pad=LAYOUT_PAD_IN)
        map_row_in = panel_h + label_h
        height_ratios = [map_row_in] * rows + ([legend_in] if include_legend else [])
        gs = fig.add_gridspec(rows + int(include_legend), cols, height_ratios=height_ratios)
        axes_flat = [fig.add_subplot(gs[idx // cols, idx % cols]) for idx in range(n_maps)]

        # Plot each map
        for idx, map_data in enumerate(maps_data):
            ax = axes_flat[idx]
            
            self._draw_panel(ax, map_data, cmap, orientation)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add model name label
            label = map_data.get('display_label', map_data['model_name'])
            
            # Axis labels/titles (not free text) so the layout engine sizes the gaps
            if label_pos == 'top':
                ax.set_title(label, fontsize=fontsize, fontweight='bold', pad=3)
            elif label_pos == 'bottom':
                ax.set_xlabel(label, fontsize=fontsize, fontweight='bold', labelpad=3)
            elif label_pos == 'left':
                ax.set_ylabel(label, fontsize=fontsize, fontweight='bold', labelpad=3)
            elif label_pos == 'right':
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(label, fontsize=fontsize, fontweight='bold', labelpad=3, rotation=270, va='bottom')
        
 
        # ── Legend: add class-name legend below the maps ──
        if include_legend:
            legend_ax = fig.add_subplot(gs[rows, :])
            legend_ax.axis('off')
            self._draw_legend(legend_ax, cmap)
            # Safety net: if the drawn legend is still wider than the figure,
            # use one column fewer (and a taller legend row) until it fits.
            while self._legend_cols > 1:
                fig.draw_without_rendering()
                leg_w = legend_ax.get_legend().get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
                if leg_w <= fig_width - 2 * LAYOUT_PAD_IN:
                    break
                self._legend_cols -= 1
                legend_ax.get_legend().remove()
                self._draw_legend(legend_ax, cmap)
                new_legend_in = self._legend_height(len(self._get_class_names()), self._legend_cols, legend_fs)
                height_ratios[-1] = new_legend_in
                gs.set_height_ratios(height_ratios)
                fig.set_figheight(min(fig.get_figheight() + new_legend_in - legend_in, max_height))
                legend_in = new_legend_in

        # Fit the figure height to what the layout engine actually drew (the
        # estimate above can be off by a few tenths of an inch for very tall or
        # wide maps): no empty band, nothing cut off, never above max height.
        # The legend row is also grown if the drawn legend is taller than its row.
        for _ in range(8):
            fig.draw_without_rendering()
            renderer = fig.canvas.get_renderer()
            if include_legend:
                leg_h = legend_ax.get_legend().get_window_extent(renderer).height / fig.dpi
                row_h = legend_ax.get_window_extent(renderer).height / fig.dpi
                if leg_h > row_h + 0.01 and fig.get_figheight() < max_height:
                    height_ratios[-1] *= 1.02 * leg_h / max(row_h, 1e-3)
                    gs.set_height_ratios(height_ratios)
                    fig.set_figheight(min(fig.get_figheight() + leg_h - row_h, max_height))
                    continue
            content_h = fig.get_tightbbox(renderer).height
            target_h = min(content_h + 2 * LAYOUT_PAD_IN, max_height)
            if abs(target_h - fig.get_figheight()) < 0.01:
                break
            fig.set_figheight(target_h)

        # Centre a short last row (odd map counts, e.g. 7 maps in 2 columns):
        # freeze the regular-grid layout, then slide that row's maps sideways.
        last_row_n = n_maps - (rows - 1) * cols
        if rows > 1 and 0 < last_row_n < cols:
            fig.draw_without_rendering()
            boxes = [ax.get_position(original=True) for ax in fig.axes]
            fig.set_layout_engine('none')
            for ax, box in zip(fig.axes, boxes):
                ax.set_position(box)
            pitch = axes_flat[1].get_position().x0 - axes_flat[0].get_position().x0
            shift = (cols - last_row_n) * pitch / 2
            for ax in axes_flat[(rows - 1) * cols:]:
                box = ax.get_position()
                ax.set_position([box.x0 + shift, box.y0, box.width, box.height])

        # Save figure - include colormap name if specified
        if output_path is None:
            cmap_suffix = f'_{cmap}' if cmap else ''
            output_path = os.path.join(
                self.base_output_dir,
                self.dataset_dir,
                f'arranged_maps_{self.config["selection_type"]}{cmap_suffix}.png'
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Saved at exactly figure_width (no 'tight' cropping), so the fonts come
        # out at their intended point size when placed at that width in the paper.
        fig.savefig(output_path, dpi=dpi)
        # PDF keeps text as vectors and embeds each map at its native
        # resolution (interpolation='none'), so it stays sharp at any zoom.
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        if pdf_path != output_path:
            fig.savefig(pdf_path)
        plt.close(fig)
        
        print(f"Arranged maps saved: {output_path} (+ {os.path.basename(pdf_path)})")
        
        # If save_sep_folder flag is set, also save individual full maps and GT
        if save_sep_folder:
            self._save_separate_maps(maps_data, regenerate_maps=regenerate_maps, cmap=cmap)
        
        return output_path


    def save_gt_map(self, output_path=None, cmap=None, include_legend=False,
                    fmt='pdf', dpi=None, figsize=None, orientation='horizontal',
                    add_table=False):
        """Save ground-truth map as a standalone figure file.

        Layout (when include_legend=True):
            ┌──────────────────────┬─────────────────┐
            │                      │  Class legend   │
            │     GT Heat-map      │  ─────────────  │
            │      (left)          │  ■ Class 1      │
            │                      │  ■ Class 2 …    │
            └──────────────────────┴─────────────────┘

        Args:
            output_path (str|None): Full path.  None → auto-name in results dir.
            cmap (str|None)       : Colourmap override (None → spy_colors / config).
            include_legend (bool) : Draw class-name legend to the right of the map.
            fmt (str)             : 'pdf' or 'png'.
            dpi (int|None)        : DPI for PNG.  None → config value.
            figsize (tuple|None)  : (w, h) in inches.  None → auto.
            orientation (str)     : 'horizontal' or 'vertical'. Rotates GT map.

        Returns:
            str: Saved file path, or None on failure.
        """
        fmt = fmt.lower().strip('.')
        if fmt not in ('pdf', 'png'):
            print(f"Warning: Unknown format '{fmt}', defaulting to 'pdf'")
            fmt = 'pdf'

        dataset_name   = self.dataset_dir
        dpi            = dpi or self.config.get('dpi', 300)
        # GT map: do NOT inherit cmap from arranger config.
        # Default is spy_colors (handled below); only use explicit user-passed cmap.
        effective_cmap = cmap   # None → spy_colors; str → that matplotlib cmap

        # ── load raw GT ──────────────────────────────────────────────────
        try:
            from utils.data_loader import DatasetLoader
            loader = DatasetLoader(use_cache=True)
            _, gt = loader.load_dataset(dataset_name)
        except Exception as e:
            print(f"Could not load GT for '{dataset_name}': {e}")
            return None

        if gt.ndim == 3:
            gt = np.squeeze(gt)

        # ── apply orientation ────────────────────────────────────────────
        if orientation == 'vertical':
            # np.rot90 with k=-1 rotates 90 degrees clockwise
            # matches img.rotate(-90) logic from the arranged maps grid
            gt = np.rot90(gt, k=-1)

        num_classes = int(gt.max())

        # ── colours ──────────────────────────────────────────────────────
        # Rule: no explicit --colormap  → use spy_colors directly (same as
        #       the full-image classification maps, regardless of vis_mode).
        #       explicit --colormap     → use that matplotlib colormap.
        if effective_cmap is None:
            # Shared SPY_COLORS palette (same one the prediction maps use)
            colors = self._get_class_colors(num_classes, 'spy_colors')
            print(f"GT map: using spy_colors palette ({num_classes} classes)")
        else:
            # Explicit colormap name given — use it
            colors = self._get_class_colors(num_classes, effective_cmap)
            print(f"GT map: using colormap '{effective_cmap}' ({num_classes} classes)")

        # Background=0 → black, classes 1…N → palette
        import matplotlib.colors as mcolors
        palette  = [[0, 0, 0]] + list(colors)
        cmap_obj = mcolors.ListedColormap(palette[:num_classes + 1])
        bounds   = np.arange(-0.5, num_classes + 1.5)
        norm     = mcolors.BoundaryNorm(bounds, cmap_obj.N)

        # ── splitting stats for table (if requested) ─────────────────────
        train_counts, val_counts, test_counts, total_counts = {}, {}, {}, {}
        has_val = False

        class_names = self._get_class_names()


        if add_table and class_names:
            try:
                valid_mask = gt > 0
                labels = gt[valid_mask].astype(int).flatten()

                from collections import Counter
                total_counter = Counter(labels)
                for c_id in range(1, num_classes + 1):
                    total_counts[c_id] = total_counter.get(c_id, 0)

                ds_cfg = self.config.get('data_split') or {}
                split_method = ds_cfg.get('method', 'ratio')
                
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()): # suppress split print statements
                    if split_method == 'ratio':
                        from utils.data_split import split_data
                        split_args = ds_cfg.get('split_ratios', [0.2, 0.1, 0.7])
                        split_res = split_data(*split_args, labels=labels, random_state=ds_cfg.get('random_state', 42))
                    else:
                        from utils.data_split import split_samples
                        split_args = ds_cfg.get('split_samples', [30, 10])
                        split_res = split_samples(*split_args, labels=labels, random_state=ds_cfg.get('random_state', 42))

                train_counter = Counter([labels[i] for i in split_res[0]])
                if len(split_res) == 3:
                    val_counter = Counter([labels[i] for i in split_res[1]])
                    test_counter = Counter([labels[i] for i in split_res[2]])
                    has_val = True
                else:
                    val_counter = Counter()
                    test_counter = Counter([labels[i] for i in split_res[1]])
                    has_val = False
                
                for c_id in range(1, num_classes + 1):
                    train_counts[c_id] = train_counter.get(c_id, 0)
                    val_counts[c_id] = val_counter.get(c_id, 0)
                    test_counts[c_id] = test_counter.get(c_id, 0)

            except Exception as e:
                print(f"Warning: Could not generate split numbers for table: {e}")
                add_table = False


        # ── typography ───────────────────────────────────────────────────
        fs = self.config.get('fontsize', 20)
        title_fs   = max(fs if isinstance(fs, (int, float)) else 20, 20)
        legend_fs  = self.config.get('legend_fontsize') or max(title_fs - 4, 16)
        patch_size = max(legend_fs * 1.1, 16)   # colour-patch height in pts

        # ── class names ──────────────────────────────────────────────────
        # (Already extracted above)


        # ── figure sizing ────────────────────────────────────────────────
        gt_h, gt_w  = gt.shape
        map_aspect  = gt_w / gt_h
        map_h_in    = 8.0                         # fixed map height (in)
        map_w_in    = map_h_in * map_aspect

        if include_legend and class_names:
            # User request: Map takes 40%, legend 60%, and square figure ratio
            # Using 10x10 as a standard square publication size
            total_w = 10.0
            total_h = 10.0
        else:
            total_w = map_w_in
            total_h = map_h_in

        if figsize is None:
            figsize = (total_w, total_h)

        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')

        if include_legend and class_names:
            from matplotlib.gridspec import GridSpec
            # Map occupies 40%, legend occupies 60%
            gs = GridSpec(
                1, 2, figure=fig,
                width_ratios=[0.4, 0.6],
                wspace=0.0,  # Zero gap between subplots
                left=0.0, right=1.0, top=1.0, bottom=0.0
            )
            ax_map = fig.add_subplot(gs[0, 0])
            ax_leg = fig.add_subplot(gs[0, 1])
            ax_leg.set_facecolor('white')
            ax_leg.set_xlim(0, 1)
            ax_leg.set_ylim(0, 1)
            ax_leg.axis('off')
            
            # Sync ax_leg height to the map's actual data height after drawing
            # This ensures "Legend" is aligned with the top of the map
            # and "Total" is aligned with the bottom of the map.
            ax_map.set_adjustable('box') # Keeps aspect ratio by shrinking the axes
        else:
            ax_map = fig.add_axes([0.0, 0.0, 1.0, 1.0])

        # ── draw map ─────────────────────────────────────────────────────
        ax_map.imshow(gt, cmap=cmap_obj, norm=norm, interpolation='nearest')
        ax_map.axis('off')                      # no title, no borders, no axes

        # ── draw RIGHT-SIDE legend ───────────────────────────────────────
        if include_legend and class_names:
            n = len(class_names)
            # if add_table, add 2 rows (header + total)
            total_rows = n + 2 if add_table else n
            # Evenly distribute rows across the full axes height
            row_gap = 1.0 / total_rows

            # Force premium Serif font (Times New Roman-like)
            font_props = {'family': 'serif', 'fontweight': 'bold'}
            if add_table:
                # Optimized spacing for 6" panel - zero-gap patch start
                col_x = {'patch': 0.0, 'name': 0.10, 'train': 0.62, 'val': 0.72, 'test': 0.83, 'total': 0.94}
                if not has_val:
                    col_x = {'patch': 0.0, 'name': 0.10, 'train': 0.65, 'test': 0.82, 'total': 0.94}

                # Header row
                y_hdr = 1.0 - 0.5 * row_gap
                header_fs = legend_fs + 1
                ax_leg.text(col_x['patch'], y_hdr, "Legend", ha='left', va='center', fontsize=header_fs, color='#1a1a1a', transform=ax_leg.transAxes, **font_props)
                ax_leg.text(col_x['name'], y_hdr, "Class Name", ha='left', va='center', fontsize=header_fs, color='#1a1a1a', transform=ax_leg.transAxes, **font_props)
                ax_leg.text(col_x['train'], y_hdr, "Train", ha='center', va='center', fontsize=header_fs, color='#1a1a1a', transform=ax_leg.transAxes, **font_props)
                if has_val:
                    ax_leg.text(col_x['val'], y_hdr, "Val", ha='center', va='center', fontsize=header_fs, color='#1a1a1a', transform=ax_leg.transAxes, **font_props)
                ax_leg.text(col_x['test'], y_hdr, "Test", ha='center', va='center', fontsize=header_fs, color='#1a1a1a', transform=ax_leg.transAxes, **font_props)
                ax_leg.text(col_x['total'], y_hdr, "Total", ha='center', va='center', fontsize=header_fs, color='#1a1a1a', transform=ax_leg.transAxes, **font_props)

                # Line under header (LaTeX style horizontal line)
                ax_leg.plot([0.02, 0.98], [1.0 - row_gap, 1.0 - row_gap], color='#000000', linewidth=1.5, transform=ax_leg.transAxes)
            else:
                col_x = {'patch': 0.05, 'name': 0.25}

            from matplotlib.patches import Rectangle
            for i, name in enumerate(class_names):
                color = colors[i] if i < len(colors) else [0.5, 0.5, 0.5]
                c_id = i + 1
                
                # y position
                y = 1.0 - (i + 1.5 if add_table else i + 0.5) * row_gap

                # Square colour patch
                patch = Rectangle(
                    (col_x['patch'], y - row_gap * 0.32),
                    width=0.06 if add_table else 0.14, height=row_gap * 0.64,
                    facecolor=color,
                    edgecolor='#000000',
                    linewidth=0.8,
                    transform=ax_leg.transAxes,
                    clip_on=False
                )
                ax_leg.add_patch(patch)

                # Class name (Bold for LaTeX quality) - No Wrap
                ax_leg.text(
                    col_x['name'], y,
                    name,
                    ha='left', va='center',
                    fontsize=legend_fs,
                    color='#111111',
                    transform=ax_leg.transAxes,
                    **font_props
                )
                
                # Table values
                if add_table:
                    ax_leg.text(col_x['train'], y, str(train_counts.get(c_id, 0)), ha='center', va='center', fontsize=legend_fs, transform=ax_leg.transAxes, **font_props)
                    if has_val:
                        ax_leg.text(col_x['val'], y, str(val_counts.get(c_id, 0)), ha='center', va='center', fontsize=legend_fs, transform=ax_leg.transAxes, **font_props)
                    ax_leg.text(col_x['test'], y, str(test_counts.get(c_id, 0)), ha='center', va='center', fontsize=legend_fs, transform=ax_leg.transAxes, **font_props)
                    ax_leg.text(col_x['total'], y, str(total_counts.get(c_id, 0)), ha='center', va='center', fontsize=legend_fs, transform=ax_leg.transAxes, **font_props)

            # Footer / Total row
            if add_table:
                # Line above total
                y_tot = 1.0 - (n + 1.5) * row_gap
                ax_leg.plot([0.02, 0.98], [y_tot + row_gap * 0.5, y_tot + row_gap * 0.5], color='#000000', linewidth=1.5, transform=ax_leg.transAxes)
                
                ax_leg.text(col_x['name'], y_tot, "TOTAL", ha='left', va='center', fontsize=header_fs, transform=ax_leg.transAxes, **font_props)
                ax_leg.text(col_x['train'], y_tot, str(sum(train_counts.values())), ha='center', va='center', fontsize=header_fs, transform=ax_leg.transAxes, **font_props)
                if has_val:
                    ax_leg.text(col_x['val'], y_tot, str(sum(val_counts.values())), ha='center', va='center', fontsize=header_fs, transform=ax_leg.transAxes, **font_props)
                ax_leg.text(col_x['test'], y_tot, str(sum(test_counts.values())), ha='center', va='center', fontsize=header_fs, transform=ax_leg.transAxes, **font_props)
                ax_leg.text(col_x['total'], y_tot, str(sum(total_counts.values())), ha='center', va='center', fontsize=header_fs, transform=ax_leg.transAxes, **font_props)


            print(f"GT legend: {n} classes (right-side panel)")

        elif include_legend and not class_names:
            print("Warning: No class names in dataset.yaml — GT legend skipped")


        # ── alignment normalization ──────────────────────────────────────
        if include_legend and class_names:
            # Shift ax_leg to match ax_map's data height exactly
            fig.canvas.draw()
            pos = ax_map.get_position(original=False)
            # Use exact pos.x1 to eliminate the gap
            ax_leg.set_position([pos.x1, pos.y0, 1.0 - pos.x1, pos.height])

        # ── save ─────────────────────────────────────────────────────────
        if output_path is None:
            legend_tag = '_legend' if (include_legend and class_names) else ''
            cmap_tag   = f'_{effective_cmap}' if effective_cmap else ''
            filename   = f'{dataset_name}_gt_map{legend_tag}{cmap_tag}.{fmt}'
            output_path = os.path.join(self.base_output_dir, dataset_name, filename)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Remove extra whitespace in PDF edges (pad_inches=0.01)
        save_kwargs = dict(bbox_inches='tight', pad_inches=0.01, facecolor='white')
        if fmt == 'png':
            save_kwargs['dpi'] = dpi

        fig.savefig(output_path, format=fmt, **save_kwargs)
        plt.close(fig)

        print(f"GT map saved → {output_path}")
        return output_path



def create_arranged_maps(base_output_dir, dataset_name, model_names, config=None, output_path=None):
    """
    Convenience function to create arranged maps
    
    Args:
        base_output_dir: Base results directory
        dataset_name: Dataset name
        model_names: List of model names
        config: Visualization config dict
        output_path: Optional output path
    
    Returns:
        Path to saved figure
    """
    arranger = MapArranger(base_output_dir, dataset_name, model_names, config)
    return arranger.arrange_maps(output_path)
