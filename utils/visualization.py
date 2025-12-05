import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

def generate_classification_map(model, dataset, device='cuda', run_dir=None, 
                                dataset_name=None, model_name=None, run_number=None,
                                cmap='tab20', show_colorbar=False, dpi=300, block_background=True):
    
    print("\n" + "="*60)
    print("Generating Classification Map...")
    print("="*60)
    
    model.eval()
    bands, height, width = dataset.data.shape
    half_patch = dataset.patch_size // 2
    
    # Initialize prediction map with 0s (0 will be our background class)
    prediction_map = np.zeros((height, width), dtype=np.int32)
    
    # Pad data for edge handling
    # dataset.data is already globally normalized via data_loader
    padded_data = np.pad(
        dataset.data,
        ((0, 0), (half_patch, half_patch), (half_patch, half_patch)),
        mode='reflect'
    )
    
    batch_size = 64
    batch_patches = []
    batch_positions = []
    
    total_pixels = height * width
    pbar = tqdm(total=total_pixels, desc="Classifying pixels", unit="px")
    
    for i in range(height):
        for j in range(width):
            patch = padded_data[:, i:i+dataset.patch_size, j:j+dataset.patch_size]
            
            if dataset.use_channel_dim:
                patch = patch.reshape((1,) + patch.shape)
            
            batch_patches.append(patch)
            batch_positions.append((i, j))
            
            if len(batch_patches) == batch_size or (i == height - 1 and j == width - 1):
                if batch_patches:
                    batch_count = len(batch_patches)
                    
                    patches_tensor = torch.tensor(
                        np.array(batch_patches),
                        dtype=torch.float32
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(patches_tensor)
                        preds = outputs.argmax(dim=1).cpu().numpy()
                    
                    # Store predictions (class 1, 2, 3...)
                    # We add 1 because model outputs 0-indexed classes
                    for idx, (y, x) in enumerate(batch_positions):
                        prediction_map[y, x] = preds[idx] + 1
                    
                    pbar.update(batch_count)
                    
                    batch_patches = []
                    batch_positions = []
    
    pbar.close()
    
    # 1. Enforce Background from Ground Truth (if available)
    if hasattr(dataset, 'gt'):
        # Force pixels that are 0 in Ground Truth to be 0 in Prediction
        mask = dataset.gt == 0
        prediction_map[mask] = 0
    
    # Save with colormap name in filename for easy identification
    if run_dir:
        output_path = os.path.join(run_dir, f'classification_map_{cmap}.png')
        # Also save as generic name for backward compatibility
        generic_path = os.path.join(run_dir, 'classification_map.png')
    else:
        output_path = f"classification_map_{dataset_name}_{cmap}.png"
        generic_path = None
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 3. Handle Colormaps & Background Color
    if cmap == 'tab20':
        # Create discrete colormap
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        current_cmap = ListedColormap(colors)
    else:
        # Get any other colormap (jet, viridis, etc.)
        current_cmap = plt.get_cmap(cmap).copy()
    
    # 2. Handle background pixels based on block_background flag
    if block_background:
        # Mask 0s and set background to black
        masked_prediction = np.ma.masked_where(prediction_map == 0, prediction_map)
        current_cmap.set_bad(color='black')
        im = ax.imshow(masked_prediction, cmap=current_cmap, interpolation='nearest')
    else:
        # Show all pixels including 0s with colormap colors
        # Use the actual class values with proper vmin/vmax
        max_class = prediction_map.max()
        if max_class > 0:
            # Map class values directly to colormap
            im = ax.imshow(prediction_map, cmap=current_cmap, interpolation='nearest', 
                          vmin=0, vmax=max_class)
        else:
            im = ax.imshow(prediction_map, cmap=current_cmap, interpolation='nearest')
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, label='Class Label', fraction=0.046, pad=0.04)
        cbar.set_label('Class Label', rotation=270, labelpad=20)
    
    ax.axis('off')
    
    # Save colormap-specific file
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    # Also save generic file for backward compatibility
    if generic_path:
        plt.savefig(generic_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    plt.close()
    
    print(f"✓ Classification map saved: {output_path}")
    if generic_path:
        print(f"✓ Generic map saved: {generic_path}")
    print(f"  Map shape: {prediction_map.shape}")
    print(f"  Unique classes: {np.unique(prediction_map)}")
    
    return prediction_map