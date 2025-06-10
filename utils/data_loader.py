import os
import json
import re
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
import matplotlib.colors as mcolors


class SubgoalDataset(Dataset):
    def __init__(self, data_list, device='cuda', max_path_len=32, map_size=100, map_res=0.05, safe_dist=0.3,
                 bounds_width=1.0):
        self.device = device
        self.max_path_len = max_path_len
        self.map_size = map_size
        self.map_res = map_res
        self.safe_dist = int(safe_dist / map_res)
        self.bounds_width = bounds_width
        self.scenarios = []

        # 1 start point + 1 occupancy map + (1 goal point + 3 global paths) => one scenario
        for item in data_list:
            occupancy_map = item['occupancy_map']
            start_point = item['start_point']
            goal_points = item['goal_points']  # up to 10
            global_paths = item['global_paths']  # also up to 10 elements, each is [ pathDict1, pathDict2, pathDict3 ]
            path_dist_map = item['path_dist_map']

            # Make sure consistent lengths
            # goal_points[i] corresponds to global_paths[i], which is the list of 3 path dicts for that goal
            num_goals = min(len(goal_points), len(global_paths))

            for g_idx in range(num_goals):
                goal_point = goal_points[g_idx]
                # Expect exactly 3 paths in global_paths[g_idx]
                path_dicts = global_paths[g_idx]  # should be a list of 3 path dictionaries
                path_dist = path_dist_map[g_idx]

                # If the data might have fewer than 3 paths, skip or handle accordingly
                if len(path_dicts) < 3:
                    continue

                # Save a single scenario
                scenario = {
                    'occupancy_map': occupancy_map,
                    'start_point': start_point,
                    'goal_point': goal_point,
                    'path_dicts': path_dicts,  # exactly 3 path dictionaries
                    'path_dist_map': path_dist
                }
                self.scenarios.append(scenario)

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]

        occupancy_map = scenario['occupancy_map']
        start_point = scenario['start_point']
        goal_point = scenario['goal_point']
        path_dicts = scenario['path_dicts']  # list of 3 path dictionaries
        path_dist_map = scenario['path_dist_map']                                 

        # convert to tensor once, reuse everywhere                              
        path_dist_tensor = torch.tensor(path_dist_map.clone().detach().cpu().numpy(), device=self.device)  

        # 1) --- build 3‑channel (valid, path, goal) tensor -----------------
        # channel‑0  (free=1, obstacle=0)  
        valid_mask, _ = self.generate_valid_mask(
            orig_data=torch.tensor(occupancy_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            path_dist=path_dist_tensor,
            path_dicts=path_dicts,
            start_point=start_point,
            occupancy_map=occupancy_map
        )                                # bool  (100,100)
        ch0 = valid_mask.float()
        
        # channel‑1  draw the 3 global paths
        path_dicts = preprocess_paths(path_dicts)
        ch1 = torch.zeros_like(ch0)
        for pD in path_dicts:
            for pt in pD['path']:
                wx, wy = pt['position']                       # world metres
                mx, my = world_to_map_coords(wx, wy,
                                            start_point['x'], start_point['y'])
                if 0 <= mx < 100 and 0 <= my < 100:
                    ch1[mx, my] = 1.0 

        # channel‑2  goal one‑hot
        ch2 = torch.zeros_like(ch0)
        gx_map, gy_map = world_to_map_coords(
            goal_point['x'], goal_point['y'],
            start_point['x'], start_point['y'])
        if 0 <= gx_map < 100 and 0 <= gy_map < 100:
            ch2[gx_map, gy_map] = 1.0 # keep the path points only in the map

        # channel-3 band_idx_map (0..9 inside corridor, -1 elsewhere)
        N_BANDS = 10
        band_idx_map = torch.full((100, 100), -1, dtype=torch.long)   # default = -1

        masked_dist2d            = path_dist_map.clone()       # ⚡ keep shape (100,100)
        masked_dist2d[~valid_mask] = float('nan')              # ⚡ NaN out invalid cells
        max_d      = masked_dist2d[valid_mask].max().item()
        band_width = max_d / N_BANDS

        for b in range(N_BANDS):
            lower, upper = b * band_width, (b + 1) * band_width
            mask_b = (masked_dist2d >= lower) & (masked_dist2d < upper) & valid_mask
            band_idx_map[mask_b] = b

        band_mean_dist = torch.zeros(N_BANDS, dtype=torch.float32)
        for b in range(N_BANDS):
            dists = masked_dist2d[band_idx_map == b]           # still (100,100)->1-D
            dists = dists[~torch.isnan(dists)]                 # drop NaNs
            band_mean_dist[b] = dists.mean() if dists.numel() else 0.0
            # print("band_width =", band_width.item() if torch.is_tensor(band_width) else band_width)
            # for b, mu in enumerate(band_mean_dist):
                # expected = (b + 0.5) * band_width
                # print(f"Band {b:2d}:  mean={mu:6.3f}  |  expected≈{expected:6.3f}")

        # turn band_idx_map into normalised channel‑3
        # band_idx_map = band_idx_map.T  
        band_idx_int = band_idx_map.clone()       
        band_norm = band_idx_map.clone().float()
        band_norm[band_norm == -1] = N_BANDS
        band_norm = torch.clamp(band_norm / (N_BANDS - 1), 0.0, 1.0)
        ch3 = band_norm  

        # concat with the 4‑channel mask
        # print(f"ch0 shape: {ch0.shape}, ch1 shape: {ch1.shape}, ch2: {ch2.shape}, ch3: {ch3.shape}")
        map_4ch =  torch.stack([ch0, ch1, ch2, ch3.squeeze(0)], dim=0)   # (4,100,100)
        map_4ch = map_4ch.unsqueeze(0)

        # 2) --- convert every path point to map indices --------------------
        max_len   = self.max_path_len
        path_tensors = []
        for pD in path_dicts:
            positions = [world_to_map_coords(pt['position'][0],
                                            pt['position'][1],
                                            start_point['x'],
                                            start_point['y'])
                        for pt in pD['path']]
            positions = np.array(positions, dtype=np.float32)      # (N,2) in map coords

            padded = np.zeros((max_len, 2), dtype=np.float32)
            actual = min(len(positions), max_len)
            padded[:actual] = positions[:actual]

            path_tensor = torch.tensor(padded, dtype=torch.float32, device=self.device)         # (T,2)
            path_tensors.append(path_tensor.unsqueeze(0))          # (1,T,2)

        path_tensor_3 = torch.cat(path_tensors, dim=0)             # (3,T,2)

        # 3) --- other scalars ---------------------------------------------
        # ---- Occupancy map as a (1,1,H,W) tensor ----
        occ_tensor = torch.tensor(occupancy_map, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # ---- Start + Goal ----
        O_t_xy = torch.tensor(
            [start_point['x'], start_point['y']], dtype=torch.float32, device=self.device).unsqueeze(0)
        X_goal_xy = torch.tensor([goal_point['x'], goal_point['y']], dtype=torch.float32, device=self.device).unsqueeze(
            0)
        X_goal_array = np.array([goal_point['x'], goal_point['y']], dtype=np.float32)
        O_t_array = np.array([start_point['x'],start_point['y']], dtype=np.float32)
        # --- compute raw distance (metres) ---------------------------------
        distance = np.linalg.norm(O_t_array - X_goal_array)    # 0‥25 m in your data
        # --- scale to [0,1]  (global min-max) ------------------------------
        d_hat = np.clip(distance / 25.0, 0.0, 1.0)             # 0 = goal reached, 1 = 25 m away
        Ot_to_goal_tensor = torch.tensor(d_hat,
                                        dtype=torch.float32,
                                        device=self.device).view(1, 1)

        valid_mask_tensor = valid_mask.clone().detach().bool() 
        valid_mask_tensor = valid_mask_tensor.unsqueeze(0)

        return {
            'occ_tensor': occ_tensor,
            'valid_mask_3ch': map_4ch.to(self.device),   # (3,100,100)
            'path_tensor': path_tensor_3,
            'O_t_xy': O_t_xy,
            'X_goal_xy': X_goal_xy,
            'Ot_to_goal': Ot_to_goal_tensor,
            'path_dist_map': path_dist_tensor,  # Add path distance map to return
            'valid_mask': valid_mask_tensor.to(self.device),
            'band_idx_map': band_idx_int,          
            'band_mean_dist': band_mean_dist,
            # Original data for reward or other calculations
            'occupancy_map': occupancy_map,
            'start_point': start_point,
            'goal_point': goal_point,
            'paths_positions': [
                [pt['position'] for pt in pD['path']] for pD in path_dicts
            ]
        }

    def visualize_bands(self, sample, title="bands‑debug"):
        """
        sample : the dict you already return from __getitem__
                (needs band_idx_map, paths_positions, start_point, goal_point)
        """
        band_map   = sample["band_idx_map"].cpu().numpy()        # (100,100) int
        paths_pos  = sample["paths_positions"]                   # list of list[(x,y)]
        sp         = sample["start_point"]
        gp         = sample["goal_point"]
        dist_map   = sample["path_dist_map"]

        # ------------------------------------------------------------------
        # build a ListedColormap → 10 colours + black for -1
        # ------------------------------------------------------------------
        band_colors = plt.cm.get_cmap("viridis", 10)             # 10 discrete
        cmap_list = [(0,0,0)] + [band_colors(i) for i in range(10)] + [(1,1,1,1)]  # extra dummy color
        cmap        = mcolors.ListedColormap(cmap_list)
        norm        = mcolors.BoundaryNorm(np.arange(-1,12), cmap.N)

        # ------------------------------------------------------------------
        plt.figure(figsize=(6,6))
        plt.title(title)
        plt.imshow(band_map, origin="lower", cmap=cmap, norm=norm)
        plt.grid(False)

        # overlay each global path
        for path_xy in paths_pos:
            if len(path_xy)==0: continue
            xs, ys = zip(*path_xy)
            map_coords = [world_to_map_coords(x, y, sp["x"], sp["y"]) for x, y in zip(xs, ys)]
            mxs, mys = zip(*map_coords)  # unzip into X and Y for plotting
            plt.plot(mxs, mys, lw=1.5, c="white")


        spx, spy = world_to_map_coords(sp["x"], sp["y"],sp["x"], sp["y"])
        plt.scatter(spy, spx, marker="o", s=60, c="lime", edgecolors="k", label="start")
        # plt.scatter(gp["x"], gp["y"], marker="*", s=100, c="red" , edgecolors="k", label="goal")
        plt.legend(loc="upper right", fontsize=8)

        # colour‑bar with nice ticks
        cbar = plt.colorbar(ticks=list(range(0,10)))
        cbar.set_label("band id")

        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------------
        dist_map = dist_map.detach().cpu().numpy()
        vmax = np.percentile(dist_map, 95)            # robust colour‑scale
        plt.figure(figsize=(6,6))
        plt.title(title)
        plt.imshow(dist_map, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
        plt.grid(False)

        # overlay each global path
        for path_xy in paths_pos:
            if len(path_xy)==0: continue
            xs, ys = zip(*path_xy)
            map_coords = [world_to_map_coords(x, y, sp["x"], sp["y"]) for x, y in zip(xs, ys)]
            mxs, mys = zip(*map_coords)  # unzip into X and Y for plotting
            plt.plot(mxs, mys, lw=1.5, c="white")


        spx, spy = world_to_map_coords(sp["x"], sp["y"],sp["x"], sp["y"])
        plt.scatter(spy, spx, marker="o", s=60, c="lime", edgecolors="k", label="start")
        # plt.scatter(gp["x"], gp["y"], marker="*", s=100, c="red" , edgecolors="k", label="goal")
        plt.legend(loc="upper right", fontsize=8)

        # colour‑bar with nice ticks
        cbar = plt.colorbar(ticks=list(range(0,10)))
        cbar.set_label("band id")

        plt.tight_layout()
        plt.show()

    
    def generate_valid_mask(self, orig_data, path_dist, path_dicts, start_point, occupancy_map):
        """
        orig_data: [H, W] 原始占用网格 (0=free, 1=障碍)
        path_dist: [H, W] 到全局路径的距离（米）
        start_xy: (x, y) 地图起点实际坐标
        Return: [H, W] 二进制掩膜 (True表示有效候选点)
        """
        # 克隆数据以避免污染原始数据
        safe_mask = orig_data.clone().float()
        # print(f"safe_mask.shape = {safe_mask.shape}")
        _, _, h, w = safe_mask.shape
        # print(f"h = {h}, w = {w}")

        # Rule 1: 基于动态边界的掩膜 ==============================
        O_t_map = (self.map_size // 2, self.map_size // 2)  # 栅格中心对应机器人当前位置

        # 转换边界限制到像素单位
        bounds_pixels = int((self.map_size // 2 * self.map_res - self.bounds_width) / self.map_res)
        lower = O_t_map[0] - bounds_pixels
        upper = O_t_map[0] + bounds_pixels

        # 创建边界限制掩膜
        boundary_mask = torch.ones(h, w, dtype=bool)
        boundary_mask[lower:upper, lower:upper] = False  # 移除中心区域

        # debug_boundary_mask = boundary_mask.cpu().numpy()
        # plt.figure(figsize=(10, 10))
        # plt.imshow(debug_boundary_mask, origin='lower')
        # plt.title("Bundary Mask")
        # plt.show()
        # cv2.imwrite("debug_bundary_mask.png", debug_boundary_mask * 255)

        # Rule 2: 障碍物扩展 ================================
        # 使用形态学膨胀扩展障碍物区域 (numpy实现更高效)
        if self.safe_dist > 0:
            orig_np = orig_data.squeeze().cpu().numpy().astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.safe_dist + 1, 2 * self.safe_dist + 1))
            dilated = cv2.dilate(orig_np, kernel)
            safe_mask = torch.from_numpy(dilated).to(orig_data.device)

            # debug_safe_mask = safe_mask.cpu().numpy()
            # cv2.imwrite("debug_safe_mask.png", debug_safe_mask * 255)

        # Rule 3: 全局路径邻近区域 ==========================
        # max_path_dist_pixels = 1.5 / self.map_res  # 允许的最大距离（cm）
        # near_path_mask = (path_dist < max_path_dist_pixels)  # 直接使用路径距离图
        # print("Path distance range:", path_dist.min().item(), "to", path_dist.max().item())
        radius = 25
        path_mask = draw_path_lines(path_dicts, occupancy_map, start_point)  # 将路径点连接为线段
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))  # 扩展半径
        near_path_mask = cv2.dilate(path_mask, kernel)  # 膨胀生成邻近区域
        # cv2.imwrite("debug_path_mask.png", near_path_mask * 255)

        # 转换设备
        device = orig_data.device
        # print(f"device = {device}")

        boundary_mask = boundary_mask.to(device)
        safe_mask = safe_mask.to(device)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        near_path_mask = torch.from_numpy(near_path_mask).to(device)

        # 合并所有规则 =================================
        valid_mask = (
                boundary_mask &
                (safe_mask == 0) &
                (near_path_mask == 1)
        )

        # debug_valid_mask  = valid_mask.cpu().numpy()
        # cv2.imwrite("debug_valid_mask.png", debug_valid_mask * 255)

        return valid_mask, near_path_mask

    def visualize_mask(self, mask, near_path_mask):
        plt.figure(figsize=(10, 10))
        plt.imshow(mask.cpu().numpy(), origin='lower')
        plt.scatter(50, 50, c='red', s=50)
        plt.title("Valid Subgoal Mask")
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(near_path_mask.cpu().numpy(), origin='lower')
        plt.title("Near Path Mask")
        plt.show()


def world_to_map_coords(g_x: float, g_y: float, start_x: float, start_y: float, map_resolution: float = 0.05):
    map_size = 100
    center_x = map_size // 2
    center_y = map_size // 2

    # Convert world coordinates to map indices relative to the center
    mx = int(center_x + (g_x - start_x) / map_resolution)
    my = int(center_y + (g_y - start_y) / map_resolution)

    return mx, my


def preprocess_paths(paths, target_num=3):
    """将路径数量固定为 target_num，不足则复制最后一条路径，超出则截断"""
    if len(paths) < target_num:
        # 复制最后一条路径直到达到目标数量
        last_path = paths[-1]
        paths += [last_path] * (target_num - len(paths))
        print(f"len(paths) < 3, len(path) = {len(paths)}")
    elif len(paths) > target_num:
        # 截断前 target_num 条路径
        paths = paths[:target_num]
        print(f"len(paths) > 3, len(path) = {len(paths)}")
    return paths


def draw_path_lines(global_paths, occupancy_map, start_point, map_resolution=0.05):
    # 创建二值化路径掩膜
    path_mask = np.zeros_like(occupancy_map, dtype=np.uint8)
    start_x, start_y = start_point['x'], start_point['y']

    # 遍历所有路径组（每个目标对应3条路径）
    for path_group in global_paths:
        if 'path' not in path_group:
            continue
        path_points = []
        # print("[DEBUG] Type of path_group['path']:", type(path_group['path']))  # 重点检查是否是字符串
        # print("[DEBUG] Sample path data:", path_group['path'][:2])  # 打印前两个元素
        # 转换所有路径点到地图坐标
        for pt in path_group['path']:
            x_world, y_world = pt['position'][0], pt['position'][1]
            x_map, y_map = world_to_map_coords(
                x_world, y_world,
                start_x, start_y,
                map_resolution
            )
            if 0 <= x_map < 100 and 0 <= y_map < 100:
                path_points.append((x_map, y_map))
                path_mask[y_map, x_map] = 1  # 标记点
        # 在相邻点之间绘制线段
        for i in range(len(path_points) - 1):
            cv2.line(
                path_mask,
                path_points[i],
                path_points[i + 1],
                1,  # 线段值为1
                thickness=1  # 线宽
            )
    
    # 在生成 path_mask 后立即翻转
    # path_mask = cv2.flip(path_mask, 0)  

    # cv2.imwrite("debug_flipped.png", path_mask * 255)
    # path_mask_visual = cv2.cvtColor((path_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # 转为彩色图像
    # path_mask_visual = cv2.flip(path_mask_visual, 0)  # 垂直翻转
    # path_mask_visual = draw_coordinate_axes(path_mask_visual, map_size=100, map_resolution=0.05)
    # cv2.imwrite("debug_path_line_with_axes.png", path_mask_visual)

    return path_mask


def show_dist_map(dist_map: torch.Tensor,
                  global_paths: list,
                  start_point: dict,
                  save_as: str = "dist_map_vis.png"):
    """
    dist_map      : torch.Tensor (100,100) – output of compute_path_dist_map_fast
    global_paths  : same structure you feed into compute_path_dist_map_fast
    start_point   : {'x','y'}
    """

    dm_np = dist_map.detach().cpu().numpy()  # (100,100)
    vmax = np.percentile(dm_np, 95)            # robust colour‑scale

    plt.figure(figsize=(6,6))
    plt.imshow(dm_np, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
    plt.colorbar(label="pixel distance to path")

    # -------- overlay the three paths (optional) -----------------
    for path_dict in global_paths:
        # for path_dict in path_group:
            # print(f"path_dist shape: {path_dict.shape}")
        xs = []
        ys = []
        # print(f"Type of path_dict['path']: {type(path_dict['path'])}, content: {path_dict['path']}")
        for pt in path_dict['path']:
            mx, my = world_to_map_coords(pt['position'][0],
                                            pt['position'][1],
                                            start_point['x'],
                                            start_point['y'])
            if 0 <= mx < 100 and 0 <= my < 100:
                xs.append(mx)
                ys.append(my)
        if xs and ys:
            # 先绘制连线
            plt.plot(xs, ys, color='red', linewidth=1)
            # 再绘制散点
            plt.scatter(xs, ys, color='red', s=4)

    plt.title("distance‑to‑path map (0 = on a path)")
    plt.tight_layout()
    plt.savefig(save_as, dpi=150)
    plt.show()
    plt.close()


def compute_path_dist_map_fast(occupancy_map: np.ndarray,
                               global_paths   : list,
                               start_point    : dict,
                               map_resolution : float = 0.05,
                               device         : str   = "cpu"
                               ) -> torch.Tensor:

    # ---------- 1) build binary mask (0 = path, 1 = free) ----------
    # start with all 1 (free space)
    path_mask = np.ones((100, 100), dtype=np.uint8)

    sx, sy = start_point['x'], start_point['y']

    for path_dict in global_paths:          # 10 goals -> 3 paths each
        # for path_dict in path_group:
        if 'path' not in path_dict:
            print("'path' not in path_dict")
            continue
        # convert every waypoint to map indices
        pts = []
        # print(f"Type of path_dict['path']: {type(path_dict['path'])}, content: {path_dict['path']}")
        for pt in path_dict['path']:
            wx, wy = pt['position']
            mx, my = world_to_map_coords(wx, wy, sx, sy, map_resolution)
            if 0 <= mx < 100 and 0 <= my < 100:
                pts.append((mx, my))

        # draw segments
        for i in range(len(pts) - 1):
            cv2.line(path_mask,
                        pts[i], pts[i+1],
                        color=0,          # 0 = foreground for distanceTransform
                        thickness=1)

    # ---------- 2) distance transform (pixels) ---------------------
    # path pixels are 0, free space 1 – perfect for cv2.distanceTransform
    dist_pix = cv2.distanceTransform(path_mask, cv2.DIST_L2, maskSize=5)

    # ---------- 3) convert to metres & to torch --------------------
    dist_m = dist_pix.astype(np.float32) * map_resolution
    return torch.tensor(dist_m, dtype=torch.float32, device=device)


def load_start_point_data(env_path):
    data = []

    # Use a predictable sorting so data_start_1.json comes before data_start_2.json, etc.
    for start_file in sorted(os.listdir(env_path)):
        # Only process JSON files named like data_start_XX.json
        if start_file.startswith("data_start_") and start_file.endswith(".json"):
            # Grab the number from the file name
            match = re.search(r'data_start_(\d+)\.json', start_file)
            if not match:
                continue
            file_count = int(match.group(1))
            # print(f"We are processing {start_file} which has index: {file_count}")

            # Build the full path and load
            start_path = os.path.join(env_path, start_file)
            with open(start_path, 'r') as f:
                data_point = json.load(f)

            pointcloud_key = f"pointcloud{file_count}"
            start_point_key = f"start_point{file_count}"

            occupancy_map = np.array(data_point[pointcloud_key]['grid_map'])
            start_point = data_point[start_point_key]

            goal_points = []
            global_paths = []
            path_dist_map = []

            # For each of the 10 possible goals
            for goal_index in range(1, 11):
                goal_key = f"goal_points_{file_count}_{goal_index}"
                if goal_key in data_point:
                    # This JSON has that goal
                    goal_points.append(data_point[goal_key])

                    # For each of the (3) possible paths
                    paths_for_this_goal = []
                    for path_index in range(1, 4):
                        path_key = f"path_{file_count}_{goal_index}_{path_index}"
                        if path_key in data_point:
                            paths_for_this_goal.append(data_point[path_key])
                    global_paths.append(paths_for_this_goal)
                    path_dist_map_for_this_goal = compute_path_dist_map_fast(
                        occupancy_map,
                        paths_for_this_goal,
                        start_point,
                        map_resolution=0.05
                    )
                    path_dist_map.append(path_dist_map_for_this_goal)

                    # show_dist_map(path_dist_map_for_this_goal, paths_for_this_goal, start_point, save_as=f"dist_map_.png")
                    
                # print(f"Goal {goal_index} has {len(paths_for_this_goal)} paths and {len(path_dist_map)} path_dist_maps")
                # print(f"Global_paths after processing: {len(global_paths)} entries")

            data.append({
                'occupancy_map': occupancy_map,
                'start_point': start_point,
                'goal_points': goal_points,
                'global_paths': global_paths,
                'path_dist_map': path_dist_map
            })

    return data


def load_data_for_environment(env_path):
    """
    This calls existing loader, returning the data for one env folder.
    """
    if not Path(env_path).exists():
        return []
    data_list = load_start_point_data(env_path)
    return data_list


def custom_collate_fn(batch):
    """
    Custom collate function to prepare batched data for DataLoader.
    Args:
        batch (list): List of data samples from SubgoalDataset.__getitem__.
    Returns:
        dict: Batched data with keys matching SubgoalDataset.__getitem__.
    """
    # Organize batch into a dictionary with keys from the dataset
    batched_data = {}

    # Collect tensor data and stack them
    batched_data['occ_tensor'] = torch.cat([item['occ_tensor'] for item in batch], dim=0)
    batched_data['path_tensor'] = torch.stack([item['path_tensor'] for item in batch], dim=0)
    batched_data['O_t_xy'] = torch.cat([item['O_t_xy'] for item in batch], dim=0)
    batched_data['X_goal_xy'] = torch.cat([item['X_goal_xy'] for item in batch], dim=0)
    batched_data['Ot_to_goal'] = torch.cat([item['Ot_to_goal'] for item in batch], dim=0)
    batched_data['path_dist_map'] = torch.stack([s['path_dist_map'] for s in batch], dim=0)
    batched_data['valid_mask'] = torch.cat([item['valid_mask'] for item in batch], dim=0)
    batched_data['valid_mask_3ch'] = torch.cat([item['valid_mask_3ch'] for item in batch], dim=0)
    batched_data['band_idx_map'] = torch.cat([item['band_idx_map'] for item in batch], dim=0)
    batched_data['band_mean_dist'] = torch.cat([item['band_mean_dist'] for item in batch], dim=0)

    # Collect lists of raw data without stacking
    batched_data['occupancy_map'] = [item['occupancy_map'] for item in batch]
    batched_data['start_point'] = [item['start_point'] for item in batch]
    batched_data['goal_point'] = [item['goal_point'] for item in batch]
    batched_data['paths_positions'] = [item['paths_positions'] for item in batch]
    batched_data['valid_mask'] = [item['valid_mask'] for item in batch]

    return batched_data

