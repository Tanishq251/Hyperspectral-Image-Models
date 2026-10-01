import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.colormap_helpers import get_colormap as _get_colormap, get_spy_cmap as _get_spy_cmap


def generate_classification_map(
    model,
    dataset,
    device='cuda',
    run_dir=None,
    dataset_name=None,
    model_name=None,
    run_number=None,
    cmap='tab20',
    show_colorbar=False,
    dpi=300,
    block_background=True,
    mode='labeled_only',
    use_spy_colors=True,   # Save a spy-palette version of the map
    use_cmap=True,          # Save a matplotlib-cmap version of the map
    mask_background_in_full_map=False
):
    """
    Generate pixel-wise classification map for HSI datasets.

    use_spy_colors: Save the map using the SPY palette (spy_colors).
    use_cmap:       Save the map using the named matplotlib cmap.
    Both can be True simultaneously — two separate images are saved.

    mask_background_in_full_map:
        - False → show model predictions everywhere (default, paper-standard)
        - True  → force GT background pixels to 0 even in full map
    """

    print("\n" + "=" * 60)
    print("Generating Classification Map...")
    print("=" * 60)

    # --------------------------------------------------
    # Model setup
    # --------------------------------------------------
    model = model.to(device)
    model.eval()

    bands, height, width = dataset.data.shape
    gt_height, gt_width = dataset.gt.shape
    half_patch = dataset.patch_size // 2

    print(f"  Data shape: ({bands}, {height}, {width})")
    print(f"  GT shape:   ({gt_height}, {gt_width})")

    prediction_map = np.zeros((gt_height, gt_width), dtype=np.int32)

    map_height = min(height, gt_height)
    map_width = min(width, gt_width)

    if height != gt_height or width != gt_width:
        print(f"  Warning: Data dims ({height},{width}) != GT dims ({gt_height},{gt_width})")
        print(f"  Using overlap region: ({map_height},{map_width})")

    # --------------------------------------------------
    # Padding
    # --------------------------------------------------
    padded_data = np.pad(
        dataset.data,
        ((0, 0), (half_patch, half_patch), (half_patch, half_patch)),
        mode='reflect'
    )

    batch_size = getattr(dataset, 'inference_batch_size', 64)
    batch_patches = []
    batch_positions = []

    total_pixels = map_height * map_width
    pbar = tqdm(total=total_pixels, desc="Classifying pixels", unit="px")

    # --------------------------------------------------
    # Sliding-window inference
    # --------------------------------------------------
    for i in range(map_height):
        for j in range(map_width):

            patch = padded_data[:, i:i + dataset.patch_size, j:j + dataset.patch_size]

            if dataset.use_channel_dim:
                patch = patch.reshape((1,) + patch.shape)

            batch_patches.append(patch)
            batch_positions.append((i, j))

            is_last = (i == map_height - 1 and j == map_width - 1)

            if len(batch_patches) == batch_size or is_last:

                if len(batch_patches) == 0:
                    continue

                patches_tensor = torch.from_numpy(
                    np.stack(batch_patches)
                ).float().to(device)

                with torch.no_grad():
                    outputs = model(patches_tensor)

                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    elif isinstance(outputs, dict):
                        outputs = outputs.get('logits', outputs[list(outputs.keys())[0]])

                    preds = outputs.argmax(dim=1).cpu().numpy()

                for idx, (y, x) in enumerate(batch_positions):
                    # Map the model's contiguous 0-based prediction back to the
                    # ORIGINAL GT label value, so class colours match the spy
                    # palette by true label (color index == original label).
                    # Using `preds + 1` breaks for datasets with non-contiguous
                    # GT labels, colouring by contiguous rank instead of label.
                    prediction_map[y, x] = dataset.idx_to_label[preds[idx]]

                pbar.update(len(batch_patches))
                batch_patches.clear()
                batch_positions.clear()

    pbar.close()

    # Keep the raw per-pixel labels so figures can be redrawn pixel-exact
    # (map_arranger renders from this instead of re-scaling a PNG screenshot).
    if run_dir:
        np.save(os.path.join(run_dir, 'prediction_map.npy'), prediction_map.astype(np.int16))

    # ==================================================
    # Visualization
    # ==================================================
    if mode == 'full_image':
        print("  Visualization mode: Full Image")

        vis_prediction = prediction_map.copy()

        if mask_background_in_full_map and hasattr(dataset, 'gt'):
            print("  Applying GT background mask in full map")
            vis_prediction[dataset.gt == 0] = 0

        if use_spy_colors:
            spy_out = (
                os.path.join(run_dir, 'classification_map_full_spy.png')
                if run_dir else f"classification_map_{dataset_name}_full_spy.png"
            )
            _save_with_matplotlib(vis_prediction, spy_out, _get_spy_cmap(),
                                  show_colorbar, dpi, block_background=False)
            print(f"Full spy-colors map saved: {spy_out}")

            try:
                import spectral as spy
                from spectral import spy_colors as _spy_colors_arr

                spy.save_rgb(spy_out, vis_prediction, colors=_spy_colors_arr)
                print(f"  (overwritten with native spectral render)")

                if hasattr(dataset, 'gt'):
                    gt_spy = (
                        os.path.join(run_dir, 'ground_truth_spy.png')
                        if run_dir else f"ground_truth_{dataset_name}_spy.png"
                    )
                    spy.save_rgb(gt_spy, dataset.gt, colors=_spy_colors_arr)
                    print(f"Ground truth (spy) saved: {gt_spy}")
            except ImportError:
                pass

        if use_cmap:
            cmap_out = (
                os.path.join(run_dir, f'classification_map_full_{cmap}.png')
                if run_dir else f"classification_map_{dataset_name}_full_{cmap}.png"
            )
            _save_with_matplotlib(vis_prediction, cmap_out, cmap,
                                  show_colorbar, dpi, block_background=False)
            print(f"Full cmap ({cmap}) map saved: {cmap_out}")

    else:
        print("  Visualization mode: Labeled Regions Only")

        labeled = prediction_map.copy()
        if hasattr(dataset, 'gt'):
            labeled[dataset.gt == 0] = 0

        if use_spy_colors:
            spy_out = (
                os.path.join(run_dir, 'classification_map_spy.png')
                if run_dir else f"classification_map_{dataset_name}_spy.png"
            )
            _save_with_matplotlib(labeled, spy_out, _get_spy_cmap(),
                                  show_colorbar, dpi, block_background)
            print(f"Spy-colors map saved: {spy_out}")

        if use_cmap:
            cmap_out = (
                os.path.join(run_dir, f'classification_map_{cmap}.png')
                if run_dir else f"classification_map_{dataset_name}_{cmap}.png"
            )
            generic_path = (
                os.path.join(run_dir, 'classification_map.png')
                if run_dir else None
            )
            _save_with_matplotlib(labeled, cmap_out, cmap,
                                  show_colorbar, dpi, block_background, generic_path)
            print(f"Cmap ({cmap}) map saved: {cmap_out}")

    print(f"  Map shape: {prediction_map.shape}")
    print(f"  Unique classes: {np.unique(prediction_map)}")

    return prediction_map


# ======================================================
# Matplotlib helper
# ======================================================
def _save_with_matplotlib(
    prediction_map,
    output_path,
    cmap,
    show_colorbar,
    dpi,
    block_background,
    generic_path=None
):
    fig, ax = plt.subplots(figsize=(12, 10))
    current_cmap = cmap if not isinstance(cmap, str) else _get_colormap(cmap)

    # For a discrete ListedColormap (e.g. spy_colors, tab20) each class value
    # must map to the palette entry at that same index. Pin vmin/vmax to the
    # palette size so class i → colour i; otherwise matplotlib auto-scales the
    # class range across the whole palette and colours come out wrong.
    from matplotlib.colors import ListedColormap
    if isinstance(current_cmap, ListedColormap):
        cmap_vmin, cmap_vmax = 0, current_cmap.N - 1
    else:
        cmap_vmin, cmap_vmax = None, None

    if block_background:
        masked = np.ma.masked_where(prediction_map == 0, prediction_map)
        current_cmap.set_bad(color='black')
        im = ax.imshow(masked, cmap=current_cmap, interpolation='nearest',
                       vmin=cmap_vmin, vmax=cmap_vmax)
    else:
        if cmap_vmax is not None:
            vmin, vmax = cmap_vmin, cmap_vmax
        else:
            _vmax = prediction_map.max()
            vmin, vmax = 0, (_vmax if _vmax > 0 else None)
        im = ax.imshow(
            prediction_map,
            cmap=current_cmap,
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax
        )

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Class Label', rotation=270, labelpad=20)

    ax.axis('off')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)

    if generic_path:
        plt.savefig(generic_path, dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.close()
