# FPV 到 BEV 地图转换完整指南

## 概述

在 `Semantic-MapNet111` 项目中，有多种方法可以将第一人称视角(First-Person View, FPV)转换为鸟瞰图视角(Bird's Eye View, BEV)。

---

## 方法 1: 使用 Depth 深度图投影（推荐用于实时观测）

### 原理
从相机拍摄的深度图中提取 3D 点云，然后将其投影到地面平面，生成占据地图。

### 关键步骤

#### 1.1 深度图 → 相机坐标系 3D 点云

```python
import numpy as np

def depth_to_camera_3d(depth, camera_intrinsics):
    """
    将深度图转换为相机坐标系下的3D点
    
    Args:
        depth: (H, W) 深度图，单位米
        camera_intrinsics: 相机内参 (fx, fy, cx, cy)
    
    Returns:
        points_3d: (H, W, 3) 相机坐标系下的3D点，格式 (x, y, z)
    """
    H, W = depth.shape
    fx, fy, cx, cy = camera_intrinsics
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 转换为相机坐标系（cv坐标系：x右, y下, z前）
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    
    points_3d = np.stack([x_cam, y_cam, z_cam], axis=-1)
    return points_3d
```

#### 1.2 相机坐标系 → 世界坐标系

考虑相机高度和倾角：

```python
def camera_to_world_3d(points_cam, camera_height=1.38, camera_pitch=0.174):
    """
    将相机坐标系的点转换到世界坐标系
    
    Args:
        points_cam: (H, W, 3) 相机坐标系的点
        camera_height: 相机离地面的高度（米）
        camera_pitch: 相机下倾角度（弧度），向下为正
    
    Returns:
        points_world: (H, W, 3) 世界坐标系的点，格式 (x_world, z_world, height)
    """
    # 相机坐标系：x右, y下, z前
    # 世界坐标系：x右, z前, height上
    
    x_world = points_cam[..., 0]  # 左右方向保持不变
    
    # 处理垂直方向（考虑相机倾角）
    y_cam = points_cam[..., 1]
    z_cam = points_cam[..., 2]
    
    cos_pitch = np.cos(camera_pitch)
    sin_pitch = np.sin(camera_pitch)
    
    # 绕X轴旋转（相机向下倾斜）
    z_world = y_cam * sin_pitch + z_cam * cos_pitch
    height = camera_height + (y_cam * cos_pitch - z_cam * sin_pitch)
    
    return np.stack([x_world, z_world, height], axis=-1)
```

#### 1.3 3D 点 → 占据地图（2D俯视图）

```python
def project_depth_to_occupancy(depth, camera_intrinsics, map_size=256, 
                               map_resolution=0.025, camera_height=1.38, 
                               camera_pitch=0.174):
    """
    将深度图投影为占据地图
    
    Args:
        depth: (H, W) 深度图
        camera_intrinsics: (fx, fy, cx, cy) 相机内参
        map_size: 生成地图的大小（map_size x map_size）
        map_resolution: 地图分辨率（m/pixel）
        camera_height: 相机高度（米）
        camera_pitch: 相机倾角（弧度）
    
    Returns:
        occupancy_map: (map_size, map_size) 二进制占据地图
            1 = 可通行（地面）
            0 = 障碍物或未观测
    """
    # Step 1: Depth → 3D点
    points_cam = depth_to_camera_3d(depth, camera_intrinsics)
    
    # Step 2: 相机坐标 → 世界坐标
    points_world = camera_to_world_3d(points_cam, camera_height, camera_pitch)
    
    # Step 3: 初始化地图
    occupancy_map = np.zeros((map_size, map_size), dtype=np.uint8)
    
    # Step 4: 投影到地面平面
    height_threshold = 0.3  # 高度差≤0.3m认为是地面
    
    x_world = points_world[..., 0]
    z_world = points_world[..., 1]
    height = points_world[..., 2]
    
    # 映射到地图坐标
    x_map = ((x_world / map_resolution) + map_size // 2).astype(int)
    z_map = (z_world / map_resolution).astype(int)
    
    # 掩码：有效深度且地面
    valid_mask = (z_world > 0.1) & (z_world < map_size * map_resolution)
    valid_mask &= (x_map >= 0) & (x_map < map_size)
    valid_mask &= (z_map >= 0) & (z_map < map_size)
    
    # 判断是否为地面（高度接近0）
    ground_mask = valid_mask & (np.abs(height) < height_threshold)
    
    occupancy_map[z_map[ground_mask], x_map[ground_mask]] = 1
    
    return occupancy_map
```

---

## 方法 2: 使用 NavMesh 生成真值地图（用于 Ground Truth）

### 原理
直接使用场景的导航网格(Navigation Mesh)生成权威的可通行区域地图。这是本项目生成GT数据的方式。

### 实现

```python
import habitat_sim

def get_topdown_view_from_navmesh(navmesh_path, resolution=0.02, 
                                   slice_height=None, map_crop_info=None):
    """
    使用 habitat_sim PathFinder 生成 topdown 俯视图
    
    Args:
        navmesh_path: 导航网格文件路径
        resolution: 生成地图的分辨率（米/像素）
        slice_height: 切片高度（从地面往上）
        map_crop_info: 坐标对齐信息 {
            'map_world_shift': 地图世界坐标原点,
            'dim': (map_width, map_height),
            'y_min_value': 地面高度
        }
    
    Returns:
        topdown_map: (H, W) 二进制地图，True=可通行，False=障碍
    """
    # 创建PathFinder对象
    pathfinder = habitat_sim.PathFinder()
    
    # 加载导航网格
    if not pathfinder.load_nav_mesh(navmesh_path):
        raise RuntimeError(f"Failed to load navmesh: {navmesh_path}")
    
    # 获取默认参数
    if slice_height is None:
        bounds_min, bounds_max = pathfinder.get_bounds()
        slice_height = bounds_min[1] + 0.1  # 地面上方 10cm
    
    # 生成俯视图
    topdown = pathfinder.get_topdown_view(resolution, slice_height)
    topdown_array = np.array(topdown, dtype=bool)
    
    # 如果需要坐标对齐
    if map_crop_info is not None:
        topdown_array = align_topdown_to_map(
            topdown_array, pathfinder, map_crop_info, resolution
        )
    
    return topdown_array


def align_topdown_to_map(topdown, pathfinder, map_info, resolution=0.02):
    """
    将 topdown 视图与语义地图坐标对齐
    """
    map_width = int(map_info['dim'][0])
    map_height = int(map_info['dim'][2])
    map_shift = np.array(map_info['map_world_shift'], dtype=np.float32)
    
    bounds_min, _ = pathfinder.get_bounds()
    
    # 计算偏移量
    src_col_in_target = int(np.round((bounds_min[0] - map_shift[0]) / resolution))
    src_row_in_target = int(np.round((bounds_min[2] - map_shift[2]) / resolution))
    
    # 初始化对齐后的地图
    aligned = np.zeros((map_height, map_width), dtype=bool)
    
    # 复制有效区域
    dst_row0 = max(0, src_row_in_target)
    dst_col0 = max(0, src_col_in_target)
    dst_row1 = min(map_height, src_row_in_target + topdown.shape[0])
    dst_col1 = min(map_width, src_col_in_target + topdown.shape[1])
    
    if dst_row0 < dst_row1 and dst_col0 < dst_col1:
        src_row0 = max(0, -src_row_in_target)
        src_col0 = max(0, -src_col_in_target)
        src_row1 = src_row0 + (dst_row1 - dst_row0)
        src_col1 = src_col0 + (dst_col1 - dst_col0)
        
        aligned[dst_row0:dst_row1, dst_col0:dst_col1] = \
            topdown[src_row0:src_row1, src_col0:src_col1]
    
    return aligned
```

---

## 方法 3: 使用高度直方图（用于 ObjectNav 观测数据）

### 原理
从累积的高度观察中提取地面高度，再结合语义信息过滤障碍物。

### 实现

```python
def build_freespace_from_height_histogram(height_map, semmap=None, observed_map=None):
    """
    从高度分布直方图构建可通行地图
    
    Args:
        height_map: (H, W) 累积高度映射
        semmap: (H, W) 语义地图，0=地面，1-12=物体
        observed_map: (H, W) 观测掩码
    
    Returns:
        nav_map: (H, W) 可通行地图
    """
    h_map = height_map.copy()
    
    # 规范化高度
    h_map[h_map > 0] = h_map[h_map > 0] - h_map[h_map > 0].min() + 1
    
    # 统计高度直方图
    n, bin_edges = np.histogram(h_map[h_map > 0], bins=200)
    
    # 找到主峰值（地面高度）
    floor_height = bin_edges[np.argmax(n)]
    
    # 地面范围：±0.1m
    nav_map = (h_map > floor_height - 0.1) & (h_map < floor_height + 0.1)
    
    # 形态学闭运算（填充小孔洞）
    from scipy.ndimage import binary_closing
    nav_map = binary_closing(nav_map.astype(int), structure=np.ones((5, 5))).astype(bool)
    
    # 应用掩码
    if observed_map is not None:
        nav_map = nav_map & observed_map
    
    if semmap is not None:
        nav_map = nav_map & (semmap == 0)  # 只保留地面区域
    
    return nav_map
```

---

## 完整流程对比

| 方法 | 输入 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| **Depth投影** | RGB-D深度图 | 实时、考虑观测 | 易受深度噪声影响 | 在线机器人导航 |
| **NavMesh** | 场景导航网格 | 精确、权威真值 | 需要预处理场景 | GT标注、离线规划 |
| **高度直方图** | 累积高度观察 | 稳健、融合多帧 | 需要足够观测 | ObjectNav数据集 |

---

## 代码使用示例

### 例 1: 从 RGB-D 生成实时地图

```python
import cv2
from PIL import Image

# 读取 RGB-D 数据
depth_raw = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)
depth = depth_raw.astype(np.float32) / 1000.0  # mm -> m

# 相机内参（示例）
fx = fy = 554.3
cx, cy = 320, 240

# 生成地图
occupancy_map = project_depth_to_occupancy(
    depth,
    camera_intrinsics=(fx, fy, cx, cy),
    map_size=256,
    map_resolution=0.025,
    camera_height=1.38,
    camera_pitch=0.174
)

# 可视化
import matplotlib.pyplot as plt
plt.imshow(occupancy_map, cmap='gray')
plt.title('BEV Occupancy Map')
plt.savefig('bev_map.png')
```

### 例 2: 从 NavMesh 生成 GT 地图

```python
import json

# 读取元数据
with open('semmap_GT_info.json') as f:
    gt_info = json.load(f)

env_name = '1LXtFkjw3qL_0'
navmesh_path = f'/data/mp3d/{env_name.rsplit("_", 1)[0]}/{env_name.rsplit("_", 1)[0]}.navmesh'

# 生成地图
topdown_map = get_topdown_view_from_navmesh(
    navmesh_path,
    resolution=0.02,
    map_crop_info=gt_info[env_name]
)

# 保存
Image.fromarray((topdown_map * 255).astype(np.uint8)).save('gt_freespace.png')
```

### 例 3: 从高度直方图生成 ObjectNav 地图

```python
import h5py

# 读取 ObjectNav 数据
with h5py.File('data/ObjectNav/semmap/scene_id.h5') as f:
    height_map = np.array(f['height_map'])
    semmap = np.array(f['semmap'])
    observed_map = np.array(f['observed_map'])

# 生成地图
freespace_map = build_freespace_from_height_histogram(
    height_map,
    semmap=semmap,
    observed_map=observed_map
)

# 保存
Image.fromarray((freespace_map * 255).astype(np.uint8)).save('freespace.png')
```

---

## 关键参数

### 相机参数
- **相机高度 (camera_height)**: 通常 1.38m（成人眼睛高度）
- **相机倾角 (camera_pitch)**: 通常 10° ≈ 0.174 rad（向下看）
- **视场角 (hfov)**: 110° （深度相机典型值）

### 地图参数
- **分辨率 (resolution)**: 0.02m/pixel （5cm 精度）或 0.025m/pixel
- **地图大小 (map_size)**: 256×256 或 512×512
- **地面高度阈值**: ±0.3m （可通行的高度范围）

### 形态学参数
- **闭运算核 (closing_kernel)**: 5×5 （ObjectNav）或 3×3 （NavMesh）
- 作用：填充小孔洞、连接断裂的区域

---

## 调试和可视化

```python
def visualize_bev_generation_pipeline(depth_or_height_map, output_path='bev_debug.png'):
    """
    生成调试可视化图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 子图1: 输入（深度或高度）
    im = axes[0, 0].imshow(depth_or_height_map, cmap='jet')
    axes[0, 0].set_title('Input (Depth or Height)')
    plt.colorbar(im, ax=axes[0, 0])
    
    # 子图2: 3D点云侧视图
    # ... 点云处理代码
    
    # 子图3: 占据地图
    # ... 地图生成代码
    
    # 子图4: 放大细节
    # ... 细节可视化
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Debug visualization saved to: {output_path}")
```

---

## 常见问题

**Q: 生成的地图全是黑色？**
- A: 检查深度范围、相机内参、高度阈值设置

**Q: NavMesh 加载失败？**
- A: 确保 navmesh 文件存在，使用 `habitat_sim.PathFinder.is_loaded` 检查

**Q: 地图坐标不对齐？**
- A: 验证 `map_world_shift` 和分辨率设置，查看偏移计算公式

---

## 相关文件

- 实现: [build_freespace_maps.py](ObjectNav/build_freespace_maps.py)
- 深度投影: [map-cmp-vla/verify_cmp_with_groundtruth.py](../map-cmp-vla/verify_cmp_with_groundtruth.py)
- 元数据: `data/semmap_GT_info.json`
- 数据格式: `data/semmap/*.h5` (GT) / `data/ObjectNav/semmap/*.h5` (ObjectNav)

