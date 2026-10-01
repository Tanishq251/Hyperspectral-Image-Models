"""
Disjoint split map visualization utility.

Generates and saves colored maps showing the spatially disjoint
train / val / test regions for any dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils.data_loader import DatasetLoader
from utils.data_split import _generate_disjoint_masks


def _build_gt_colormap(gt, cmap='jet', use_spy_colors=True):
    """
    Build a colormap for the GT label map.

    Priority:
        1. use_spy_colors=True  → spectral.spy_colors (if installed)
        2. cmap string          → matplotlib colormap (e.g. 'jet', 'tab20')

    Returns:
        render_fn(label_map) → RGBA image (H, W, 4)
    """
    unique_classes = sorted([c for c in np.unique(gt) if c != 0])
    max_label = int(gt.max())

    # ── Try spy_colors first ────────────────────────────────
    if use_spy_colors:
        try:
            from spectral import spy_colors
            # spy_colors is (N, 3) uint8 array
            spy_arr = np.array(spy_colors, dtype=np.float64) / 255.0
            n_spy = spy_arr.shape[0]

            def _render_spy(label_map):
                h, w = label_map.shape
                rgb = np.zeros((h, w, 3), dtype=np.float64)
                for cls in unique_classes:
                    mask = (label_map == cls)
                    color_idx = cls % n_spy
                    rgb[mask] = spy_arr[color_idx]
                return rgb

            print(f"  Colormap: spy_colors ({n_spy} entries)")
            return _render_spy
        except ImportError:
            print("  Warning: spectral not installed, falling back to matplotlib cmap")

    # ── Matplotlib colormap fallback ────────────────────────
    n_cls = len(unique_classes)
    mpl_cmap = plt.get_cmap(cmap)
    colors = mpl_cmap(np.linspace(0, 1, max(n_cls, 2)))

    cls_to_color = {}
    for idx, cls_id in enumerate(unique_classes):
        cls_to_color[cls_id] = colors[idx % len(colors)][:3]

    def _render_mpl(label_map):
        h, w = label_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.float64)
        for cls_id, color in cls_to_color.items():
            mask = (label_map == cls_id)
            rgb[mask] = color
        return rgb

    print(f"  Colormap: {cmap} (matplotlib)")
    return _render_mpl


def plot_disjoint_maps(dataset_name, save_dir=None, dpi=300, show=False,
                       cmap='jet', use_spy_colors=True):
    """
    Load a dataset, compute its disjoint split masks, and save
    a side-by-side figure (GT | Split | Train | Val | Test).

    Args:
        dataset_name   : str   – Name recognized by DatasetLoader (e.g. "NiliFossae")
        save_dir       : str   – Directory to write the PNG into.
                                 Defaults to "disjoint_maps/<dataset_name>/".
        dpi            : int   – Output resolution (default 300).
        show           : bool  – If True, also call plt.show().
        cmap           : str   – Matplotlib colormap name (default "jet").
                                 Used when use_spy_colors=False or spectral not installed.
        use_spy_colors : bool  – If True (default), use spectral.spy_colors for
                                 class coloring. Falls back to `cmap` if spectral
                                 is not installed.

    Returns:
        str – path to the saved image.
    """
    # ── 1. Load data ─────────────────────────────────────────
    loader = DatasetLoader(use_cache=True)
    data, gt = loader.load_dataset(dataset_name)

    # Ensure GT is 2-D
    if gt.ndim == 3:
        gt = np.squeeze(gt)

    print(f"\n{'='*60}")
    print(f"Disjoint Map Visualization: {dataset_name}")
    print(f"{'='*60}")
    print(f"  GT shape : {gt.shape}")
    num_classes = len([c for c in np.unique(gt) if c != 0])
    total_labeled = int(np.sum(gt != 0))
    print(f"  Classes  : {num_classes}  |  Labeled pixels: {total_labeled}")

    # ── 2. Generate disjoint masks ───────────────────────────
    train_mask, val_mask, test_mask = _generate_disjoint_masks(gt)

    train_px = int(train_mask.sum())
    val_px   = int(val_mask.sum())
    test_px  = int(test_mask.sum())
    total_px = train_px + val_px + test_px

    print(f"\n  Split pixels:")
    print(f"    Train : {train_px:>8}  ({100*train_px/total_px:.1f}%)")
    print(f"    Val   : {val_px:>8}  ({100*val_px/total_px:.1f}%)")
    print(f"    Test  : {test_px:>8}  ({100*test_px/total_px:.1f}%)")
    print(f"    Total : {total_px:>8}")

    # Check overlap
    overlap = int(np.sum(train_mask & test_mask) +
                  np.sum(train_mask & val_mask) +
                  np.sum(val_mask & test_mask))
    if overlap > 0:
        print(f"  ❌ WARNING: {overlap} overlapping pixels!")
    else:
        print(f"  ✅ Zero spatial overlap between all splits")

    # ── 3. Build color renderer ──────────────────────────────
    render_fn = _build_gt_colormap(gt, cmap=cmap, use_spy_colors=use_spy_colors)

    # Render GT and per-split maps
    gt_rgb    = render_fn(gt)
    train_gt  = np.where(train_mask, gt, 0)
    val_gt    = np.where(val_mask,   gt, 0)
    test_gt   = np.where(test_mask,  gt, 0)
    train_rgb = render_fn(train_gt)
    val_rgb   = render_fn(val_gt)
    test_rgb  = render_fn(test_gt)

    # Split overview map: train=blue, val=orange, test=green, bg=black
    split_rgb = np.zeros((*gt.shape, 3), dtype=np.float64)
    split_rgb[train_mask] = [0.2, 0.4, 0.9]   # blue
    split_rgb[val_mask]   = [1.0, 0.6, 0.1]   # orange
    split_rgb[test_mask]  = [0.2, 0.8, 0.3]   # green

    # ── 4. Plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    titles = [
        f"Ground Truth\n({total_labeled} px)",
        f"Disjoint Split\n(Train/Val/Test)",
        f"Train Region\n({train_px} px – {100*train_px/total_px:.0f}%)",
        f"Val Region\n({val_px} px – {100*val_px/total_px:.0f}%)",
        f"Test Region\n({test_px} px – {100*test_px/total_px:.0f}%)",
    ]
    images = [gt_rgb, split_rgb, train_rgb, val_rgb, test_rgb]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Legend for split map
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.2, 0.4, 0.9], label='Train'),
        Patch(facecolor=[1.0, 0.6, 0.1], label='Val'),
        Patch(facecolor=[0.2, 0.8, 0.3], label='Test'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower center',
                   fontsize=9, ncol=3, framealpha=0.8)

    fig.suptitle(f"Spatially Disjoint Split – {dataset_name}",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # ── 5. Save ──────────────────────────────────────────────
    if save_dir is None:
        save_dir = os.path.join("disjoint_maps", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, "disjoint_split_map.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    print(f"\n  Saved combined map: {out_path}")

    # Save individual maps without headings/axes (with black background)
    indiv_names = ["gt.png", "split_map.png", "train_gt.png", "val_gt.png", "test_gt.png"]
    for img, fname in zip(images, indiv_names):
        fpath = os.path.join(save_dir, fname)
        plt.imsave(fpath, img)
        print(f"  Saved individual map: {fpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path
