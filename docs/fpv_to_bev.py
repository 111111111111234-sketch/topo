#!/usr/bin/env python3
"""
FPV 到 BEV 地图转换工具库
包含三种方法：深度投影、NavMesh、高度直方图

使用示例：
    from fpv_to_bev import project_depth_to_occupancy, get_topdown_from_navmesh
    
    # 方法1：从深度图
    bev_map = project_depth_to_occupancy(depth_img, camera_intrinsics)
    
    # 方法2：从NavMesh
    bev_map = get_topdown_from_navmesh(navmesh_path)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Union
import os


# ============================================================================
# 方法 1: 深度图投影 (Depth Projection)
# ============================================================================

def depth_to_camera_3d(
    depth: np.ndarray,
    camera_intrinsics: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    将深度图转换为相机坐标系下的 3D 点云
    
    Args:
        depth: (H, W) 深度图，单位米
        camera_intrinsics: (fx, fy, cx, cy) 相机内参
        
    Returns:
        points_3d: (H, W, 3) 相机坐标系下的 3D 点，格式 (x, y, z)
                   其中 x=左右, y=上下, z=前后(深度)
    """
    H, W = depth.shape
    fx, fy, cx, cy = camera_intrinsics
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 从像素坐标转换为相机坐标
    # OpenCV 坐标系：x向右, y向下, z向前
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    
    points_3d = np.stack([x_cam, y_cam, z_cam], axis=-1)
    return points_3d


def camera_to_world_3d(
    points_cam: np.ndarray,
    camera_height: float = 1.38,
    camera_pitch: float = 0.174
) -> np.ndarray:
    """
    将相机坐标系的点转换到世界坐标系
    
    Args:
        points_cam: (H, W, 3) 相机坐标系的点，格式 (x, y, z)
        camera_height: 相机离地面的高度（米），默认 1.38m
        camera_pitch: 相机的下倾角度（弧度），默认 0.174 rad ≈ 10°
        
    Returns:
        points_world: (H, W, 3) 世界坐标系的点，格式 (x_world, z_world, height)
    """
    # 提取坐标分量
    x_cam = points_cam[..., 0]  # 左右方向保持不变
    y_cam = points_cam[..., 1]  # 相机坐标的y（垂直，向下为正）
    z_cam = points_cam[..., 2]  # 相机坐标的z（深度，向前为正）
    
    # 计算旋转参数（相机绕X轴向下倾斜）
    cos_pitch = np.cos(camera_pitch)
    sin_pitch = np.sin(camera_pitch)
    
    # 世界坐标系：x向右, z向前, height向上
    # 通过旋转矩阵将相机坐标转换到世界坐标
    x_world = x_cam  # 左右方向不变
    z_world = y_cam * sin_pitch + z_cam * cos_pitch  # 前后方向
    height = camera_height + (y_cam * cos_pitch - z_cam * sin_pitch)  # 高度
    
    return np.stack([x_world, z_world, height], axis=-1)


def project_depth_to_occupancy(
    depth: np.ndarray,
    camera_intrinsics: Tuple[float, float, float, float],
    map_size: int = 256,
    map_resolution: float = 0.025,
    camera_height: float = 1.38,
    camera_pitch: float = 0.174,
    ground_threshold: float = 0.3,
    valid_depth_range: Tuple[float, float] = (0.1, 10.0)
) -> np.ndarray:
    """
    将深度图投影为占据地图（俯视图）
    
    Args:
        depth: (H, W) 深度图，单位米
        camera_intrinsics: (fx, fy, cx, cy) 相机内参
        map_size: 生成地图的尺寸（map_size x map_size）
        map_resolution: 地图分辨率（米/像素）
        camera_height: 相机高度（米）
        camera_pitch: 相机倾角（弧度）
        ground_threshold: 地面高度阈值（米），±threshold范围内认为是可通行地面
        valid_depth_range: (min, max) 有效深度范围
        
    Returns:
        occupancy_map: (map_size, map_size) 二进制占据地图
            1 = 可通行（地面）
            0 = 障碍物或未观测
    """
    # Step 1: 深度图 → 相机坐标系 3D 点
    points_cam = depth_to_camera_3d(depth, camera_intrinsics)
    
    # Step 2: 相机坐标 → 世界坐标
    points_world = camera_to_world_3d(points_cam, camera_height, camera_pitch)
    
    # Step 3: 初始化地图
    occupancy_map = np.zeros((map_size, map_size), dtype=np.uint8)
    
    # Step 4: 提取坐标分量
    x_world = points_world[..., 0]
    z_world = points_world[..., 1]
    height = points_world[..., 2]
    
    # Step 5: 映射到地图坐标
    # 地图坐标系：原点在机器人位置，x向右，z向前
    x_map = ((x_world / map_resolution) + map_size // 2).astype(int)
    z_map = (z_world / map_resolution).astype(int)
    
    # Step 6: 构建有效性掩码
    valid_depth = (depth >= valid_depth_range[0]) & (depth <= valid_depth_range[1])
    valid_z = (z_world > 0.1) & (z_world < map_size * map_resolution)
    valid_x = (x_map >= 0) & (x_map < map_size)
    valid_z_map = (z_map >= 0) & (z_map < map_size)
    
    valid_mask = valid_depth & valid_z & valid_x & valid_z_map
    
    # Step 7: 判断地面（高度接近0）
    ground_mask = valid_mask & (np.abs(height) < ground_threshold)
    
    # Step 8: 填充占据地图
    occupancy_map[z_map[ground_mask], x_map[ground_mask]] = 1
    
    return occupancy_map


# ============================================================================
# 方法 2: NavMesh 投影 (NavMesh Projection)
# ============================================================================

def get_topdown_from_navmesh(
    navmesh_path: str,
    resolution: float = 0.02,
    slice_height: Optional[float] = None,
    map_crop_info: Optional[Dict] = None
) -> np.ndarray:
    """
    使用 habitat_sim 从导航网格生成俯视图
    
    Args:
        navmesh_path: 导航网格文件路径 (.navmesh)
        resolution: 生成地图的分辨率（米/像素），默认 0.02m
        slice_height: 切片高度（从地面往上，米）。如果为None，自动使用 y_min + 0.1m
        map_crop_info: 坐标对齐信息字典，包含：
            - 'map_world_shift': 地图世界坐标原点 [x, y, z]
            - 'dim': 地图尺寸 (width, height, depth)
            - 'y_min_value': 地面高度
            如果为None，返回原始的 topdown 视图
        
    Returns:
        topdown_map: (H, W) 二进制地图，True=可通行，False=障碍
        
    Raises:
        ImportError: 如果 habitat_sim 未安装
        FileNotFoundError: 如果 navmesh 文件不存在
        RuntimeError: 如果 navmesh 加载失败
    """
    try:
        import habitat_sim
    except ImportError:
        raise ImportError("habitat_sim not installed. Install with: pip install habitat-sim")
    
    if not os.path.exists(navmesh_path):
        raise FileNotFoundError(f"NavMesh file not found: {navmesh_path}")
    
    # 创建 PathFinder 对象
    pathfinder = habitat_sim.PathFinder()
    
    # 加载导航网格
    if not pathfinder.load_nav_mesh(navmesh_path):
        raise RuntimeError(f"Failed to load navmesh: {navmesh_path}")
    
    if not pathfinder.is_loaded:
        raise RuntimeError(f"NavMesh not properly loaded: {navmesh_path}")
    
    # 确定切片高度
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
    
    return topdown_array, pathfinder


def align_topdown_to_map(
    topdown: np.ndarray,
    pathfinder,
    map_info: Dict,
    resolution: float = 0.02
) -> np.ndarray:
    """
    将 topdown 视图与语义地图坐标对齐
    
    Args:
        topdown: (H, W) topdown 视图
        pathfinder: habitat_sim PathFinder 对象
        map_info: 地图信息字典，包含 'dim', 'map_world_shift'
        resolution: 分辨率（米/像素）
        
    Returns:
        aligned: (H, W) 对齐后的地图
    """
    map_width = int(map_info['dim'][0])
    map_height = int(map_info['dim'][2])
    map_shift = np.array(map_info['map_world_shift'], dtype=np.float32)
    
    # 获取 NavMesh 边界
    bounds_min, bounds_max = pathfinder.get_bounds()
    
    # 计算偏移量（topdown 在目标地图中的位置）
    src_col_in_target = int(np.round((bounds_min[0] - map_shift[0]) / resolution))
    src_row_in_target = int(np.round((bounds_min[2] - map_shift[2]) / resolution))
    
    # 初始化对齐后的地图
    aligned = np.zeros((map_height, map_width), dtype=bool)
    
    # 计算有效的复制区域
    dst_row0 = max(0, src_row_in_target)
    dst_col0 = max(0, src_col_in_target)
    dst_row1 = min(map_height, src_row_in_target + topdown.shape[0])
    dst_col1 = min(map_width, src_col_in_target + topdown.shape[1])
    
    # 如果没有有效的重叠区域，返回空地图
    if dst_row0 >= dst_row1 or dst_col0 >= dst_col1:
        return aligned
    
    # 计算源区域
    src_row0 = max(0, -src_row_in_target)
    src_col0 = max(0, -src_col_in_target)
    src_row1 = src_row0 + (dst_row1 - dst_row0)
    src_col1 = src_col0 + (dst_col1 - dst_col0)
    
    # 复制
    aligned[dst_row0:dst_row1, dst_col0:dst_col1] = \
        topdown[src_row0:src_row1, src_col0:src_col1]
    
    # 应用轻微的形态学闭运算平滑
    from scipy.ndimage import binary_closing
    aligned = binary_closing(aligned.astype(np.uint8), 
                             structure=np.ones((3, 3))).astype(bool)
    
    return aligned


# ============================================================================
# 方法 3: 高度直方图 (Height Histogram)
# ============================================================================

def build_freespace_from_height_histogram(
    height_map: np.ndarray,
    semmap: Optional[np.ndarray] = None,
    observed_map: Optional[np.ndarray] = None,
    floor_tolerance: float = 0.1,
    closing_kernel_size: int = 5
) -> np.ndarray:
    """
    从高度分布直方图构建可通行地图
    
    Args:
        height_map: (H, W) 累积高度映射
        semmap: (H, W) 语义地图，0=地面，1-12=物体（可选）
        observed_map: (H, W) 观测掩码，1=已观测（可选）
        floor_tolerance: 地面高度容差（米），默认 ±0.1m
        closing_kernel_size: 闭运算核大小（像素）
        
    Returns:
        nav_map: (H, W) 可通行地图，True=可通行，False=障碍
    """
    from scipy.ndimage import binary_closing
    
    h_map = height_map.copy()
    
    # 规范化高度（将最小正高度设为1）
    h_map[h_map > 0] = h_map[h_map > 0] - h_map[h_map > 0].min() + 1
    
    # 统计高度直方图
    n, bin_edges = np.histogram(h_map[h_map > 0], bins=200)
    
    # 找到主峰值（地面高度）
    floor_height = bin_edges[np.argmax(n)]
    
    # 地面范围：floor_height ± floor_tolerance
    nav_map = (h_map > floor_height - floor_tolerance) & \
              (h_map < floor_height + floor_tolerance)
    
    # 形态学闭运算（填充小孔洞）
    kernel = np.ones((closing_kernel_size, closing_kernel_size))
    nav_map = binary_closing(nav_map.astype(int), structure=kernel).astype(bool)
    
    # 应用观测掩码
    if observed_map is not None:
        nav_map = nav_map & observed_map.astype(bool)
    
    # 应用语义掩码（只保留地面和未标记的区域）
    if semmap is not None:
        nav_map = nav_map & (semmap == 0)
    
    return nav_map


# ============================================================================
# 辅助函数
# ============================================================================

def get_default_camera_intrinsics(image_width: int, image_height: int,
                                  hfov: float = 110.0) -> Tuple[float, float, float, float]:
    """
    根据图像尺寸和视场角计算相机内参（假设主点在图像中心）
    
    Args:
        image_width: 图像宽度
        image_height: 图像高度
        hfov: 水平视场角（度数），默认 110°
        
    Returns:
        (fx, fy, cx, cy) 相机内参
    """
    cx = image_width / 2.0
    cy = image_height / 2.0
    fx = fy = image_width / (2.0 * np.tan(np.deg2rad(hfov) / 2.0))
    return (fx, fy, cx, cy)


def filter_moving_average(bev_map: np.ndarray, 
                          prev_maps: list,
                          alpha: float = 0.3) -> np.ndarray:
    """
    应用指数移动平均平滑地图序列
    
    Args:
        bev_map: 当前帧的 BEV 地图
        prev_maps: 之前的地图列表（最多保留3-5帧）
        alpha: 平滑因子（0-1），越小越平滑
        
    Returns:
        filtered_map: 平滑后的地图
    """
    if not prev_maps:
        return bev_map
    
    filtered = bev_map.astype(np.float32) * alpha
    for prev_map in prev_maps:
        filtered += prev_map.astype(np.float32) * (1 - alpha) / len(prev_maps)
    
    return (filtered > 0.5).astype(np.uint8)


# ============================================================================
# 可视化函数
# ============================================================================

def visualize_bev_map(bev_map: np.ndarray, title: str = "BEV Map", 
                      save_path: Optional[str] = None):
    """
    可视化 BEV 地图
    
    Args:
        bev_map: (H, W) BEV 地图
        title: 图表标题
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bev_map, cmap='gray', origin='upper')
    ax.set_title(title)
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Z (Forward)')
    
    # 标记机器人位置
    h, w = bev_map.shape
    ax.plot(w // 2, h // 2, 'r*', markersize=15, label='Robot')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved BEV map to: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    print("FPV to BEV Conversion Toolkit")
    print("=" * 50)
    print("\n方法 1: 深度图投影")
    print("  from fpv_to_bev import project_depth_to_occupancy")
    print("\n方法 2: NavMesh 投影")
    print("  from fpv_to_bev import get_topdown_from_navmesh")
    print("\n方法 3: 高度直方图")
    print("  from fpv_to_bev import build_freespace_from_height_histogram")
    print("\n查看源代码获取详细文档和示例")
