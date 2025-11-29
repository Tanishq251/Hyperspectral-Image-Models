import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.data_loader import normalize_hyperspectral_data


def generate_classification_map(model, dataset, device='cuda', run_dir=None, 
                                dataset_name=None, model_name=None, run_number=None,
                                cmap='tab20', show_colorbar=False, dpi=300):
    """Generate and save classification map for the entire image"""
    
    print("\n" + "="*60)
    print("Generating Classification Map...")
    print("="*60)
    
    model.eval()
    bands, height, width = dataset.data.shape
    half_patch = dataset.patch_size // 2
    
    # Initialize prediction map
    prediction_map = np.zeros((height, width), dtype=np.int32)
    
    # Pad data for edge handling
    padded_data = np.pad(
        dataset.data,
        ((0, 0), (half_patch, half_patch), (half_patch, half_patch)),
        mode='reflect'
    )
    
    batch_size = 64
    batch_patches = []
    batch_positions = []
    
    # Create progress bar for all pixels
    total_pixels = height * width
    pbar = tqdm(total=total_pixels, desc="Classifying pixels", unit="px")
    
    for i in range(height):
        for j in range(width):
            # Extract patch
            patch = padded_data[:, i:i+dataset.patch_size, j:j+dataset.patch_size]
            
            # Normalize and prepare patch
            normalized_patch = normalize_hyperspectral_data(patch)
            
            # Match dataset format
            if dataset.use_channel_dim:
                normalized_patch = normalized_patch.reshape((1,) + normalized_patch.shape)
            
            batch_patches.append(normalized_patch)
            batch_positions.append((i, j))
            
            # Process batch
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
                    
                    # Assign predictions
                    for idx, (y, x) in enumerate(batch_positions):
                        prediction_map[y, x] = preds[idx] + 1
                    
                    pbar.update(batch_count)
                    
                    batch_patches = []
                    batch_positions = []
    
    pbar.close()
    
    # Save and visualize
    if run_dir:
        output_path = os.path.join(run_dir, 'classification_map.png')
    else:
        output_path = f"classification_map_{dataset_name}.png"
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(prediction_map, cmap=cmap)
    
    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Class')
    
    ax.axis('off')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✓ Classification map saved: classification_map.png")
    
    return prediction_map
