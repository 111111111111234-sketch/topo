# 📋 Semantic-MapNet111 代码文件完整清单

## 按模块组织的所有文件

### 🎯 核心投影库 (3个文件)

#### 1. **projector/core.py** [276行]
```python
# 关键函数
_transform3D(xyzhe)              # L6   pose(x,y,z,heading,elev) → 4×4变换矩阵

class ProjectorUtils:
  compute_intrinsic_matrix()      # L67  vfov → 相机内参K(fx,fy,cx,cy)
  compute_scaling_params()        # L79  预计算缩放参数
  point_cloud(depth)              # L116 深度图(H,W) → 相机3D点(H,W,3)
  transform_camera_to_world(xyz1, T) # L151 (x,y,z)_cam → (x,y,z)_world
  pixel_to_world_mapping(depth, T) # L177 完整流程: 像素 → 3D → 世界坐标
  discretize_point_cloud(xyz, h)  # 后续 3D点(x,y,z) → 2D栅格(u,v)
  get_topdown_coordinate_range()  # 计算BEV坐标范围
```

**关键参数:**
- vfov: 垂直视场角(弧度)
- gridcellsize: 栅格大小(0.02m)
- output_height/width: BEV地图尺寸
- world_shift_origin: 世界坐标原点偏移

#### 2. **projector/projector.py** [108行]
```python
class Projector(ProjectorUtils):
  __init__()                      # L11  初始化投影器参数
  forward(depth, T, obs_per_map=1, return_heights=False)  # L66 主投影函数
```

**功能流程:**
```
depth(N,1,H,W) + T(N,4,4)
  ↓
pixel_to_world_mapping()  # 像素 → 世界坐标
  ↓
过滤(depth==0, z_clip)
  ↓
discretize_point_cloud()  # 3D → 2D栅格
  ↓
输出: mask(N,H_out,W_out) + projection_indices_2D
```

#### 3. **projector/point_cloud.py** [100+行]
```python
class PointCloud(ProjectorUtils):
  forward(depth, T, obs_per_map=1)  # L56 保留RGB/语义特征的投影
```

**与Projector的区别:**
- Projector: 仅输出mask(0/1)
- PointCloud: 保留特征值(RGB或语义)

#### 4. **projector/__init__.py**
```python
from projector.core import _transform3D, ProjectorUtils
from projector.projector import Projector
from projector.point_cloud import PointCloud
```

---

### 🎨 地面真值生成 (2个文件)

#### 5. **compute_GT_topdown_semantic_maps/build_semmap_from_egoGT.py** [306行]
```python
# 配置
env = '17DRP5sb8fy_0'          # 场景ID
resolution = 0.02              # BEV分辨率
vfov = 67.5 * np.pi / 180.0   # 垂直视场角
z_clip = 0.50                  # 天花板过滤高度
features_spatial_dimensions = (480, 640)  # egocentric分辨率

# 核心流程
from projector import _transform3D, PointCloud
# 1. 加载Habitat环境
# 2. 迭代egocentric观察
# 3. PointCloud投影到BEV
# 4. 融合多帧结果
```

**输入:** Egocentric RGB-D + GT语义标签
**输出:** BEV语义地图

#### 6. **compute_GT_topdown_semantic_maps/build_semmap_from_obj_point_cloud.py**
```python
# 论文使用的方法
# 输入: 场景的3D对象点云 + 语义标签
# 输出: 完整、无偏的BEV语义地图
```

---

### 📊 数据预处理 (4个文件)

#### 7. **precompute_training_inputs/build_data.py** [223行] ⭐ 重要
```python
# 全局设置
output_dir = 'data/training/smnet_training_data/'
resolution = 0.02
default_ego_dim = (480, 640)

# 核心流程
from projector import _transform3D
from projector.point_cloud import PointCloud
from semseg.rednet import RedNet

for scene in scenes:
  # 1. 加载Habitat场景
  # 2. 获取轨迹关键帧
  # 3. 读取RGB-D-Semantic
  # 4. RedNet语义分割
  # 5. PointCloud投影
  # 6. 保存到h5:
  #    - 'rgb': egocentric RGB(480,640,3)
  #    - 'depth': 深度图(480,640)
  #    - 'semantic': GT语义(480,640)
  #    - 'target': BEV语义地图(512,512,13)
```

**输入:** MP3D场景集合
**输出:** h5训练数据(egocentric + BEV目标)

#### 8. **precompute_training_inputs/build_projindices.py**
```python
def get_projections_indices(file):
  # 预计算投影索引
  # 加速训练时的投影操作
  # 输出: projection_indices_2D
```

#### 9. **precompute_training_inputs/build_crops.py**
```python
# 空间记忆动态裁剪
# 根据agent移动裁剪BEV特征
```

#### 10. **precompute_test_inputs/build_test_data.py**
```python
# 同build_data.py，用于测试集
# 处理逻辑完全相同
```

---

### 🧠 模型网络 (4个文件)

#### 11. **SMNet/model.py** [较长] ⭐ 核心网络
```python
# 网络结构
Encoder: egocentric特征编码
  RGB(480,640) + Depth(480,640) → 特征向量

Projection Layer: 投影到BEV
  egocentric特征 → BEV特征(512,512,C)

Spatial Memory Aggregator: 多帧融合 ⭐关键
  Frame_t + Frame_{t-1} → 累积特征

SemmapDecoder: (L168)
  BEV特征(512,512,C) → 语义地图(512,512,13)
```

**完整管道:**
```
Egocentric(t) → Encode → Project → Aggregate Memory 
  ← Memory(t-1)
  → Memory(t) → Decode → Topdown Semantic(t)
```

#### 12. **SMNet/model_test.py**
```python
# 推理时的模型结构
class SemmapDecoder(nn.Module):  # L188
  # 同训练时的解码器
  # 仅用于推理(无梯度)
```

#### 13. **SMNet/loader.py**
```python
# 数据加载器
# 读取h5文件 → DataLoader
# 输出: (egocentric, BEV_target)
```

#### 14. **SMNet/loss.py**
```python
class SemmapLoss(nn.Module):  # L5
  # 语义地图损失函数
  # 通常: cross_entropy(pred, target)
```

#### 15. **SMNet/smnet_utils.py**
```python
# 网络工具函数
```

---

### 📷 语义分割 (1个文件)

#### 16. **semseg/rednet.py** [较长]
```python
class RedNet(nn.Module):
  forward_downsample(rgb, depth)  # L155 RGB-D下采样
  forward(rgb, depth)              # L223 完整前向
```

**用途:** Egocentric RGB-D → 像素级语义标签(13类)

**输入:** RGB(480,640,3) + Depth(480,640,1)
**输出:** 语义标签(480,640,13) - logits

---

### 🗺️ 导航应用 (3个文件)

#### 17. **ObjectNav/build_freespace_maps.py** ⭐⭐⭐ 当前项目最常用
```python
def process_gt_h5(env_name, pathfinder_cache, semmap_info, navmesh_root):
  # GT数据处理 - 使用NavMesh
  # 1. 加载NavMesh文件
  # 2. pathfinder.get_topdown_view(resolution=0.02)
  # 3. 坐标对齐(map_world_shift)
  # 输出: 二进制freespace地图(1=可走, 0=障碍)

def process_objnav_h5(path, semmap_info):
  # ObjectNav数据处理 - 使用高度直方图
  # 1. 读取h5: height_map, semmap, observed_map
  # 2. build_freespace_from_height_histogram()
  # 3. 形态学操作(闭运算)
  # 输出: 二进制freespace地图

# 主程序
if __name__ == '__main__':
  # 扫描所有h5文件
  # 分别处理GT和ObjectNav
  # 保存PNG地图
```

**处理规模:** 89个GT场景 + 22个ObjectNav场景

#### 18. **ObjectNav/run_astar_planning.py**
```python
# 使用freespace地图进行A*路径规划
# 输入: freespace地图 + start + goal
# 输出: 路径序列
```

#### 19. **ObjectNav/astar.py**
```python
# A*算法核心实现
```

---

### 🛠️ 工具程序 (4个文件)

#### 20. **utils/habitat_utils.py**
```python
class HabitatUtils:
  # Habitat环境交互工具
  initialize_scene()   # 初始化模拟器
  get_agent_pose()     # 获取相机位姿(x,y,z,heading,elevation)
  get_observations()   # 读取RGB-D-Semantic观察
  get_semantic_mask()  # 获取对象语义标签
```

#### 21. **utils/semantic_utils.py**
```python
def color_label(label_id):
  # 语义标签ID → RGB颜色
  # 用于可视化

# 其他语义处理函数
```

#### 22. **utils/crop_memories.py**
```python
# 空间记忆动态裁剪
# 随agent移动裁剪BEV特征
```

#### 23. **utils/__init__.py**
```python
from utils.habitat_utils import HabitatUtils
from utils.semantic_utils import color_label
```

---

### 🎬 主程序和脚本 (5个文件)

#### 24. **train.py**
```python
# 主训练脚本
# 使用 SMNet/loader.py 加载数据
# 使用 SMNet/model.py 定义网络
# 使用 SMNet/loss.py 计算损失
```

#### 25. **test.py**
```python
# 主测试脚本
# 加载预训练模型
# 在测试集上评估
```

#### 26. **demo.py**
```python
# 演示脚本
# 实时展示FPV→BEV转换
```

---

### 📈 评估指标 (5个文件)

#### 27-31. **metric/ 和 eval/**
```python
metric/metric.py          # 通用指标
metric/iou.py            # Intersection over Union
metric/acc.py            # 准确率
metric/confusionmatrix.py # 混淆矩阵

eval/eval.py             # 主评估脚本
eval/eval_bfscore.py     # BFS评分
eval/bfscore.py          # BFS距离计算
```

---

## 🔀 数据流向总结

```
场景 → HabitatUtils加载 
  ↓
多帧egocentric观察(RGB-D)
  ↓
RedNet语义分割 (semseg/rednet.py)
  ↓
Projector投影 (projector/{core,projector}.py)
  深度 + 位姿 → pixel_to_world_mapping() → BEV投影
  ↓
SMNet空间聚合 (SMNet/model.py)
  多帧BEV特征融合 → Spatial Memory
  ↓
SemmapDecoder解码 (SMNet/model.py:L168)
  ↓
输出: BEV语义地图(512,512,13)
  ↓
应用:
  1. ObjectNav/freespace_map (自由空间)
  2. ObjectNav/路径规划 (A*)
```

---

## 📊 文件统计

| 类别 | 文件数 | 行数(估) | 复杂度 |
|------|--------|---------|--------|
| 投影库 | 4 | 500+ | ⭐⭐⭐ |
| 地面真值 | 2 | 600+ | ⭐⭐⭐ |
| 数据预处理 | 4 | 500+ | ⭐⭐ |
| 模型网络 | 5 | 2000+ | ⭐⭐⭐ |
| 语义分割 | 1 | 1000+ | ⭐⭐⭐ |
| 导航应用 | 3 | 300+ | ⭐⭐ |
| 工具 | 4 | 500+ | ⭐ |
| 主程序 | 3 | 500+ | ⭐⭐ |
| 评估 | 5 | 500+ | ⭐⭐ |
| **总计** | **31** | **6500+** | |

---

## 🎯 快速索引

**如何找到特定功能:**

| 需求 | 文件 | 函数 |
|------|------|------|
| 深度→3D点 | projector/core.py | point_cloud() |
| 3D点→世界坐标 | projector/core.py | transform_camera_to_world() |
| 完整投影流程 | projector/projector.py | Projector.forward() |
| 数据生成 | precompute_training_inputs/build_data.py | 主循环 |
| 语义分割 | semseg/rednet.py | RedNet.forward() |
| 多帧融合 | SMNet/model.py | Spatial Memory Aggregator |
| 语义解码 | SMNet/model.py | SemmapDecoder |
| 自由空间 | ObjectNav/build_freespace_maps.py | process_*_h5() |
| 路径规划 | ObjectNav/run_astar_planning.py | A*搜索 |

