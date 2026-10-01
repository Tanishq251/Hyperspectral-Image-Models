"""Shared colormap helpers used by visualization.py and map_arranger.py."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    # Prefer the real spectral (spy) package's palette when installed, since
    # it's the canonical source. The fallback below is an exact copy of its
    # 39 entries (checked against spectral 0.24).
    import spectral as _spy
    SPY_COLORS = _spy.spy_colors.astype(np.uint8)
except ImportError:
    # Local fallback so callers don't need the spectral package installed.
    SPY_COLORS = np.array([
        [0,   0,   0],
        [255, 0,   0],
        [0,   255, 0],
        [0,   0,   255],
        [255, 255, 0],
        [255, 0,   255],
        [0,   255, 255],
        [200, 100, 0],
        [0,   200, 100],
        [100, 0,   200],
        [200, 0,   100],
        [100, 200, 0],
        [0,   100, 200],
        [150, 75,  75],
        [75,  150, 75],
        [75,  75,  150],
        [255, 100, 100],
        [100, 255, 100],
        [100, 100, 255],
        [255, 150, 75],
        [75,  255, 150],
        [150, 75,  255],
        [50,  50,  50],
        [100, 100, 100],
        [150, 150, 150],
        [200, 200, 200],
        [250, 250, 250],
        [100, 0,   0],
        [200, 0,   0],
        [0,   100, 0],
        [0,   200, 0],
        [0,   0,   100],
        [0,   0,   200],
        [100, 100, 0],
        [200, 200, 0],
        [100, 0,   100],
        [200, 0,   200],
        [0,   100, 100],
        [0,   200, 200],
    ], dtype=np.uint8)


def get_spy_cmap():
    """Return a ListedColormap built from the SPY_COLORS palette."""
    return ListedColormap(SPY_COLORS / 255.0)


def get_spy_colormap_dict():
    """Return a rasterio-compatible colormap dict from SPY_COLORS."""
    return {idx: tuple(color.tolist()) for idx, color in enumerate(SPY_COLORS)}


def get_colormap(name=None):
    """Return a matplotlib colormap instance.

    For 'tab20' (the default), builds a proper ListedColormap.
    For anything else, returns ``plt.get_cmap(name).copy()``.

    Args:
        name: Colormap name string, or None for 'tab20'.

    Returns:
        matplotlib colormap instance
    """
    if name is None or name == 'tab20':
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        return ListedColormap(colors)
    return plt.get_cmap(name).copy()
