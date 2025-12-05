"""
Classification Map Arranger - Combines multiple classification maps into a single figure
with configurable layout, spacing, labels, and colormaps.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import torch
from pathlib import Path

# Reuse existing utilities
from utils.data_loader import DatasetLoader, HyperspectralDataset
from utils.visualization import generate_classification_map


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
                print(f"✓ Auto-discovered {len(self.model_names)} models: {self.model_names}")
            else:
                # Handle models as list
                self.model_names = config.get('models', [])
                if not isinstance(self.model_names, list):
                    self.model_names = [self.model_names]
            
            # Get parallel lists for map_type and metric
            map_types = config.get('map_type', [])
            metrics = config.get('metric', [])
            
            # Ensure lists are same length as models
            if not isinstance(map_types, list):
                map_types = [map_types] * len(self.model_names)
            if not isinstance(metrics, list):
                metrics = [metrics] * len(self.model_names)
            
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
        
        # Default config
        self.config = {
            'rows': 2,
            'cols': 2,
            'orientation': 'horizontal',
            'gap': 20,
            'fontsize': 12,
            'label_position': 'top',
            'label_alignment': 'center',
            'label_offset': 0.02,  # Distance between label and image (0.02 = 2% of image size)
            'selection_metric': 'OA',
            'selection_type': 'best',
            'dpi': 300,
            'figsize': (16, 12),
            # Visualization defaults
            'cmap': None,
            'block_background': True,
            'include_gt': False,
            'regenerate_maps': False,
        }
        
        # Update with provided config
        if config:
            # Handle nested config structure
            if 'layout' in config:
                self.config.update(config['layout'])
            if 'labels' in config:
                self.config['fontsize'] = config['labels'].get('fontsize', self.config['fontsize'])
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
        model_path = os.path.join(self.base_output_dir, self.dataset_dir, model_name)
        
        # Try both possible filenames
        csv_path1 = os.path.join(model_path, 'results_summary.csv')
        csv_path2 = os.path.join(model_path, 'results_summary.csv')
        
        if os.path.exists(csv_path1):
            return csv_path1
        if os.path.exists(csv_path2):
            return csv_path2
        
        return None
    
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
    
    def _load_checkpoint(self, model, checkpoint_path, device):
        """Load model weights from checkpoint."""
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
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
        # Determine map paths
        original_map_path = os.path.join(run_dir, 'classification_map.png')
        cmap_specific_path = os.path.join(run_dir, f'classification_map_{cmap}.png') if cmap else None
        
        # If not regenerating, try to load existing maps
        if not regenerate:
            # Priority 1: Load colormap-specific file if it exists
            if cmap_specific_path and os.path.exists(cmap_specific_path):
                return Image.open(cmap_specific_path)
            
            # Priority 2: Load original map ONLY if no specific cmap requested
            # Don't trust config - always regenerate if specific cmap requested but file doesn't exist
            if not cmap and os.path.exists(original_map_path):
                return Image.open(original_map_path)
        
        # Determine final map path for saving/loading
        map_path = cmap_specific_path if cmap else original_map_path
        
        # Check if regeneration is needed
        needs_regeneration = regenerate
        if needs_regeneration and model is not None and dataset is not None:
            print(f"Regenerating classification map for {run_dir} with colormap={cmap}...")
            
            # Find best checkpoint
            best_checkpoint = self._find_checkpoint(run_dir)
            if not best_checkpoint:
                print(f"Warning: No checkpoint found in {run_dir}")
                return Image.open(original_map_path) if os.path.exists(original_map_path) else None
            
            print(f"Using checkpoint: {os.path.basename(best_checkpoint)}")
            
            # Load checkpoint and generate map
            try:
                model = self._load_checkpoint(model, best_checkpoint, device)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return Image.open(original_map_path) if os.path.exists(original_map_path) else None
            
            # Generate map using imported generate_classification_map
            try:
                generate_classification_map(
                    model=model,
                    dataset=dataset,
                    device=device,
                    run_dir=run_dir,
                    cmap=cmap or 'tab20',
                    show_colorbar=False,
                    dpi=self.config.get('dpi', 300),
                    block_background=self.config.get('block_background', True)
                )
                
                # generate_classification_map now saves as classification_map_{cmap}.png directly
                # Load the newly generated map
                if os.path.exists(map_path):
                    print(f"✓ Generated map with colormap '{cmap}': {map_path}")
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
            
            # Create GT visualization
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            current_cmap = self._get_colormap_instance(cmap)
            
            # Mask background (0s) and set to black
            masked_gt = np.ma.masked_where(gt == 0, gt)
            current_cmap.set_bad(color='black')
            
            ax.imshow(masked_gt, cmap=current_cmap, interpolation='nearest')
            ax.axis('off')
            
            # Save to temporary file (use tempfile for cross-platform)
            import tempfile
            temp_path = os.path.join(tempfile.gettempdir(), 'gt_map_temp.png')
            fig.savefig(temp_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            return Image.open(temp_path)
        except Exception as e:
            print(f"Error loading ground truth map: {e}")
            return None
    
    def _get_colormap_instance(self, cmap_name):
        """Get matplotlib colormap instance from name."""
        if cmap_name == 'tab20':
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            return ListedColormap(colors)
        return plt.get_cmap(cmap_name or 'tab20').copy()
    
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
        Check if map regeneration is needed based on colormap.
        Returns True if regeneration is needed, False if existing map can be used.
        
        Logic:
        - If colormap-specific file exists (classification_map_{cmap}.png) -> use it, no regeneration
        - If no cmap specified -> use original map, no regeneration
        - Otherwise -> regenerate with requested colormap
        """
        if force_regenerate:
            print(f"⚡ {model_name}: Force regenerate requested")
            return True
        
        if not cmap:
            # No specific colormap requested, use original map
            return False
            
        cmap_specific_path = os.path.join(run_dir, f'classification_map_{cmap}.png')
        
        # ONLY skip if colormap-specific file exists
        # Don't trust config - the actual image might have been generated with different colormap
        if os.path.exists(cmap_specific_path):
            print(f"✓ {model_name}: Skipping - found existing map: classification_map_{cmap}.png")
            return False
        
        # Need to regenerate - colormap-specific file doesn't exist
        print(f"🔄 {model_name}: Regenerating - classification_map_{cmap}.png not found")
        return True
    
    def _collect_single_run(self, model_name, run_dir, run_number, metric_value, metric, cmap, device, regenerate_maps):
        """Helper to collect a single run's map data."""
        should_regenerate = self._check_regeneration_needed(run_dir, model_name, cmap, regenerate_maps)
        
        model, dataset = None, None
        if should_regenerate:
            model, dataset = self._load_model_for_regeneration(model_name, run_dir)
            should_regenerate = model is not None
        
        img = self.load_classification_map(run_dir, regenerate=should_regenerate, model=model, dataset=dataset, device=device, cmap=cmap)
        if img:
            return {
                'model_name': model_name,
                'run_number': run_number,
                'run_dir': run_dir,
                'image': img,
                'metric_value': metric_value,
                'metric_name': metric,
                'is_gt': False
            }
        return None
    
    def collect_maps(self, regenerate_maps=False, cmap=None, include_gt=False):
        """Collect classification maps for all models based on per-model selection criteria."""
        maps_data = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Add ground truth if requested
        if include_gt:
            gt_image = self.load_ground_truth_map(self.dataset_dir, cmap=cmap)
            if gt_image:
                maps_data.append({
                    'model_name': 'Ground Truth', 'run_number': None, 'run_dir': None,
                    'image': gt_image, 'metric_value': None, 'metric_name': None, 'is_gt': True
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
    
    def arrange_maps(self, output_path=None, regenerate_maps=None, cmap=None, include_gt=None):
        """
        Arrange collected maps into a grid layout
        
        Args:
            output_path: Path to save the arranged figure
            regenerate_maps: If True, regenerate maps from checkpoints (None = use config value)
            cmap: Colormap to use for regeneration (None = use config value)
            include_gt: If True, include ground truth map alongside predictions (None = use config value)
        
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
        
        maps_data = self.collect_maps(regenerate_maps=regenerate_maps, cmap=cmap, include_gt=include_gt)
        
        if not maps_data:
            print("Error: No maps collected")
            return None
        
        rows = self.config['rows']
        cols = self.config['cols']
        orientation = self.config.get('orientation', 'horizontal')
        fontsize = self.config['fontsize']
        label_pos = self.config['label_position']
        label_align = self.config['label_alignment']
        dpi = self.config.get('dpi', 300)
        
        # Get image dimensions from first map
        first_img = maps_data[0]['image']
        img_width, img_height = first_img.size
        
        # If vertical orientation, swap dimensions for layout calculation
        if orientation == 'vertical':
            layout_width = img_height
            layout_height = img_width
        else:
            layout_width = img_width
            layout_height = img_height
        
        # Calculate aspect ratio of each image
        aspect_ratio = layout_width / layout_height
        
        # Calculate figure size based on number of images and their aspect ratio
        # Base size per image in inches
        base_size = 4  # inches per image
        fig_width = cols * base_size * aspect_ratio
        fig_height = rows * base_size
        
        # Get gap settings from config (as fraction of subplot size)
        # wspace = width space between columns, hspace = height space between rows
        col_gap = self.config.get('gap', 20) / 100  # Convert to fraction (20 -> 0.2)
        row_gap = self.config.get('row_gap', 30) / 100  # Convert to fraction (30 -> 0.3)
        
        # Create figure with gridspec for controlled spacing
        fig, axes = plt.subplots(
            rows, cols, 
            figsize=(fig_width, fig_height),
            dpi=dpi,
            squeeze=False,  # Always return 2D array
            gridspec_kw={'wspace': col_gap, 'hspace': row_gap}
        )
        
        # Flatten axes for easy iteration
        axes_flat = axes.flatten()
        
        # Plot each map
        for idx, map_data in enumerate(maps_data[:rows * cols]):
            ax = axes_flat[idx]
            
            # Rotate image based on orientation
            img = map_data['image']
            if orientation == 'vertical':
                img = img.rotate(-90, expand=True)
            
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add model name label
            label = map_data['model_name']
            
            if label_pos == 'top':
                ax.set_title(label, fontsize=fontsize, fontweight='bold', pad=10)
            elif label_pos == 'bottom':
                ax.text(0.5, -0.05, label, transform=ax.transAxes,
                       fontsize=fontsize, fontweight='bold', ha='center', va='top')
            elif label_pos == 'left':
                ax.text(-0.05, 0.5, label, transform=ax.transAxes,
                       fontsize=fontsize, fontweight='bold', ha='right', va='center', rotation=90)
            elif label_pos == 'right':
                ax.text(1.05, 0.5, label, transform=ax.transAxes,
                       fontsize=fontsize, fontweight='bold', ha='left', va='center', rotation=270)
        
        # Hide any unused subplots
        for idx in range(len(maps_data), rows * cols):
            axes_flat[idx].axis('off')
        
        # Save figure - include colormap name if specified
        if output_path is None:
            cmap_suffix = f'_{cmap}' if cmap else ''
            output_path = os.path.join(
                self.base_output_dir,
                self.dataset_dir,
                f'arranged_maps_{self.config["selection_type"]}{cmap_suffix}.png'
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Arranged maps saved: {output_path}")
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
