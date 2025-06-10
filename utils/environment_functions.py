import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def world_to_map_coords_short(g_x: float, g_y: float, start_x: float, start_y: float, map_resolution: float = 0.05):
    map_size = 100
    origin_x = start_x - (map_size // 2) * map_resolution  # start_x - 2.5 meters
    origin_y = start_y - (map_size // 2) * map_resolution  # start_y - 2.5 meters
    
    mx = int((g_x - origin_x) / map_resolution)
    my = int((g_y - origin_y) / map_resolution)
    
    return max(0, min(mx, 99)), max(0, min(my, 99))


def visualize_band_probs(band_map, band_probs,
                         paths_pos, sp,
                         title, save_path):
    """
    band_map   (100,100) ints   -  band-id each cell, -1 outside corridor
    band_probs (10,)            -  policy probabilities for the 10 bands
    """
    # ------------- tidy inputs ----------------------------------- #
    if isinstance(band_map, torch.Tensor):
        band_map = band_map.cpu().numpy()
    if band_map.ndim == 3 and band_map.shape[0] == 1:
        band_map = band_map[0]
    band_map = np.nan_to_num(band_map, nan=-1).astype(int)
    band_map = np.clip(band_map, -1, 9)

    # ------------- build DISCRETE palette ------------------------ #
    # colour for each band = plasma(prob)  ;  invalid = transparent
    base_cmap = plt.cm.plasma
    cmap_list = [(0,0,0,0)]                        # colour for -1 (transparent)
    for p in band_probs:
        cmap_list.append(base_cmap(p))             # colour for band 0..9
    cmap = mcolors.ListedColormap(cmap_list)

    # discrete norm so 0..9 get the exact entries we set above
    boundaries = np.arange(-1.5, 10.5, 1)          # [-1.5, -0.5, 0.5, …, 9.5]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    # ------------- plot map -------------------------------------- #
    plt.figure(figsize=(6,6))
    plt.title(title)

    mask = np.ma.masked_where(band_map == -1, band_map)
    plt.imshow(mask, origin="lower", cmap=cmap, norm=norm)
    plt.grid(False)

    # ------------- overlay paths & start ------------------------- #
    for path_xy in paths_pos:
        if not path_xy: continue
        xs, ys = zip(*path_xy)
        mxs, mys = zip(*[world_to_map_coords_short(x, y, sp["x"], sp["y"])
                         for x, y in zip(xs, ys)])
        plt.plot(mxs, mys, lw=1.5, c="black")

    spx, spy = world_to_map_coords_short(sp["x"], sp["y"], sp["x"], sp["y"])
    plt.scatter(spy, spx, marker="o", s=60, c="lime",
                edgecolors="k", label="start")
    plt.legend(loc="upper right", fontsize=7)

    # ------------- matching colour-bar --------------------------- #
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])                                # dummy
    cbar = plt.colorbar(sm, ticks=np.arange(0,10))
    cbar.set_label("band id  ( colour = prob )")
    # write the actual probabilities next to each tick
    cbar.ax.set_yticklabels([f"{i}  ({band_probs[i]:.2f})" for i in range(10)])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


