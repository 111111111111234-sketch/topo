# FPV 到 BEV 转换 - 快速参考卡片

## 一页纸总结

### 三种方法速览

| 特性 | 深度投影 | NavMesh | 高度直方图 |
|------|--------|---------|----------|
| **公式** | `depth → 3D点 → 地面检测` | `navmesh → topdown视图` | `直方图峰值 → 地面` |
| **精度** | 中等 | 高（GT真值） | 中等 |
| **速度** | 快 | 中 | 快 |
| **输入** | RGB-D图 | .navmesh文件 | 累积高度 |
| **用途** | 在线导航 | 离线GT | ObjectNav数据 |

---

## 快速代码片段

### 方法1: 深度投影
```python
from fpv_to_bev import project_depth_to_occupancy

# 读取深度图
depth = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH).astype(float) / 1000.0

# 生成BEV地图
bev = project_depth_to_occupancy(
    depth, 
    camera_intrinsics=(fx, fy, cx, cy),
    map_size=256,
    map_resolution=0.025
)
```

### 方法2: NavMesh
```python
from fpv_to_bev import get_topdown_from_navmesh

# 加载NavMesh生成地图
bev, pathfinder = get_topdown_from_navmesh(
    'path/to/scene.navmesh',
    resolution=0.02,
    map_crop_info=metadata  # 可选
)
```

### 方法3: 高度直方图
```python
from fpv_to_bev import build_freespace_from_height_histogram

# 从高度直方图生成地图
bev = build_freespace_from_height_histogram(
    height_map,
    semmap=semmap,
    observed_map=observed_map
)
```

---

## 关键参数速查

### 相机参数
- **相机高度**: 1.38m (成人眼睛)
- **相机倾角**: 0.174 rad ≈ 10° (向下)
- **焦距计算**: `f = width / (2 * tan(hfov/2))` 其中 hfov=110°

### 地图参数
- **分辨率**: 0.02-0.025 m/pixel
- **地图大小**: 256×256 或 512×512 像素
- **可通行高度**: ±0.3m (相对地面)

### 形态学处理
- **闭运算核**: 3×3 (NavMesh) 或 5×5 (ObjectNav)
- **用途**: 填充孔洞，连接断裂

---

## 坐标系速记

```
相机坐标系 (OpenCV)        世界坐标系
  y↓ (下)                  height↑
  |                        |
  └─→ x(右)               └─→ z(前)
   \                        \
    ↘ z(前)                 ↘ x(右)

转换公式:
  x_world = x_cam
  z_world = y_cam * sin(pitch) + z_cam * cos(pitch)
  height = camera_height + y_cam * cos(pitch) - z_cam * sin(pitch)
```

---

## 检查清单

### 问题排查
- [ ] 深度图全黑？→ 检查深度范围、相机内参
- [ ] 地图有噪点？→ 增加形态学核、平滑深度
- [ ] NavMesh失败？→ 检查路径、habitat_sim版本
- [ ] 坐标不对齐？→ 打印偏移量、验证计算

### 优化建议
- [ ] 使用中值滤波平滑深度
- [ ] 应用指数移动平均融合多帧
- [ ] 调整地图分辨率平衡精度和速度
- [ ] GPU加速关键操作

---

## 相关文件

### 实现代码
- [fpv_to_bev.py](/workspace/tangyx7@xiaopeng.com/fpv_to_bev.py) - 核心库
- [build_freespace_maps.py](Semantic-MapNet111/ObjectNav/build_freespace_maps.py) - 完整示例

### 文档
- [FPV_TO_BEV_GUIDE.md](/workspace/tangyx7@xiaopeng.com/FPV_TO_BEV_GUIDE.md) - 详细指南
- [FPV_to_BEV_Complete_Guide.ipynb](/workspace/tangyx7@xiaopeng.com/FPV_to_BEV_Complete_Guide.ipynb) - Jupyter笔记本

### 数据文件
- `Semantic-MapNet111/data/semmap/*.h5` - GT语义地图
- `Semantic-MapNet111/data/ObjectNav/semmap/*.h5` - ObjectNav数据
- `Semantic-MapNet111/data/mp3d/*/\*.navmesh` - 导航网格
- `Semantic-MapNet111/data/semmap_GT_info.json` - 元数据

---

## 性能基准

在标准硬件上的执行时间：

```
输入: 640×480 深度图，输出: 256×256 BEV地图

深度投影法:        ~5-10 ms  (CPU)  / <1 ms (GPU)
NavMesh投影:      ~1-2 ms   (首次加载) / <1 ms (缓存)
高度直方图法:     ~3-5 ms   (CPU)
```

---

## 常见陷阱

❌ **错误做法**
```python
# 1. 忽略坐标系差异
z_world = depth  # 错误！

# 2. 不处理无效深度
occupied[depth.astype(int), ...] = 1  # 会indexing错误

# 3. 形态学操作过度
closing(closing(closing(map)))  # 会丧失细节

# 4. 忽视舍入误差
offset = int((bounds_min[0] - shift[0]) / res)  # 可能差1像素
```

✅ **正确做法**
```python
# 1. 使用变换矩阵
points_world = camera_to_world_3d(points_cam, h, pitch)

# 2. 有效性检查
valid = (depth > 0.1) & (depth < 10.0)
map[z_map[valid], x_map[valid]] = 1

# 3. 单次闭运算
map = binary_closing(map, kernel)

# 4. 精确舍入
offset = int(np.round(...))
```

---

## 扩展主题

### 动态场景处理
- 使用光流(optical flow)检测移动物体
- 利用Kalman滤波器平滑轨迹
- 融合多传感器(LiDAR + RGB-D)

### 大规模应用
- 构建全局地图(SLAM)
- 路径规划与导航
- 自动驾驶场景理解

### 算法改进
- 考虑地面斜率
- 处理动态高度(楼梯、坡道)
- 语义分割细化分类

---

## 参考资源

### 相关论文
- Perspective Transformer Nets (PTN) - NIPS 2016
- Monodepth: Single Image Depth Estimation
- Mono3D++: Monocular 3D Object Detection

### 开源项目
- habitat-sim: 高保真模拟器
- MonoDepth: 单目深度估计
- YOLOv8-Seg: 实时分割

### 深度学习框架
- PyTorch3D - 3D深度学习
- OpenCV - 经典视觉算法
- scikit-image - 图像处理工具包

