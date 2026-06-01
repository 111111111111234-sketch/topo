# Semantic-MapNet111 FPV→BEV 相关代码文件完整指南

## 📋 文件结构总览

```
Semantic-MapNet111/
├── 核心转换模块
│   ├── projector/                    ⭐ FPV→BEV 投影核心库
│   └── compute_GT_topdown_semantic_maps/  地面真值生成
│
├── 数据处理预处理
│   ├── precompute_training_inputs/   训练数据预处理
│   └── precompute_test_inputs/       测试数据预处理
│
├── 主要网络和模型
│   ├── SMNet/                        语义映射网络
│   └── semseg/                       语义分割模块
│
└── 导航和应用
    └── ObjectNav/                    导航应用（包括freespace生成）
```

---

## 🎯 关键代码文件详解

### 1️⃣ **投影核心模块 (projector/)**

#### [projector/core.py](Semantic-MapNet111/projector/core.py)
**核心坐标变换工具库**

| 类/函数 | 行号 | 功能 |
|--------|------|------|
| `_transform3D()` | L6 | **生成4×4位姿变换矩阵** - 从(x,y,z,heading,elevation)到SE(3)矩阵 |
| `ProjectorUtils` | L36 | **投影工具基类**，包含: |
| `.compute_intrinsic_matrix()` | L67 | 计算相机内参矩阵K (fx, fy, cx, cy) |
| `.compute_scaling_params()` | L79 | 预计算缩放参数 |
| `.point_cloud()` | L116 | **深度图→相机坐标系3D点云** (像素→(x,y,z)) |
| `.transform_camera_to_world()` | L151 | **相机坐标→世界坐标** (应用变换矩阵T) |
| `.pixel_to_world_mapping()` | L177 | **完整流程**: 像素→3D点→世界坐标 |
| `.discretize_point_cloud()` | 后续 | 3D点→2D栅格（离散化） |

**关键概念:**
- 内部参数K: 相机焦距和光心
- 外参T: camera_to_world变换矩阵 (包含位置和姿态)
- 输出: 世界坐标系下的(x,y,z)点云

#### [projector/projector.py](Semantic-MapNet111/projector/projector.py)
**投影器主类 - 处理各种特征投影**

| 类/函数 | 行号 | 功能 |
|--------|------|------|
| `Projector` | L7 | **继承ProjectorUtils的主投影类** |
| `.__init__()` | L11 | 初始化: vfov, 图像尺寸, 输出地图尺寸, 分辨率, 世界坐标偏移 |
| `.forward()` | L66 | **完整投影流程**: |
| | | 1. 深度图(H,W) + 位姿T → 世界坐标点云 |
| | | 2. 过滤异常点(depth==0, z_clip) |
| | | 3. 离散化为2D栅格坐标 |
| | | 输出: 占据地图 + 投影索引 |

**核心参数:**
- `vfov`: 垂直视场角(rad)
- `gridcellsize`: BEV地图分辨率(m/pixel)，通常0.02m
- `z_clip_threshold`: 过滤天花板(通常0.5m)
- `world_shift_origin`: 地图世界坐标原点偏移

#### [projector/point_cloud.py](Semantic-MapNet111/projector/point_cloud.py)
**特征投影器 - 保留RGB/语义特征**

| 函数 | 行号 | 功能 |
|------|------|------|
| `PointCloud` | 类 | 继承ProjectorUtils |
| `.forward()` | L56 | **投影RGB/语义特征到BEV** |
| | | - 保留特征值(而不仅仅mask) |
| | | - 支持特征融合和插值 |
| | | - 输出: 特征地图(H,W,C) |

#### [projector/__init__.py](Semantic-MapNet111/projector/__init__.py)
**模块初始化和导入**

```python
from projector.core import _transform3D, ProjectorUtils
from projector.projector import Projector
from projector.point_cloud import PointCloud
```

---

### 2️⃣ **地面真值生成 (compute_GT_topdown_semantic_maps/)**

#### [compute_GT_topdown_semantic_maps/build_semmap_from_egoGT.py](Semantic-MapNet111/compute_GT_topdown_semantic_maps/build_semmap_from_egoGT.py)
**从egocentric GT投影生成BEV语义地图**

| 部分 | 行号 | 内容 |
|------|------|------|
| 说明 | L1-8 | 注释：投影egocentric GT语义标签生成topdown地图 |
| 设置 | L34-44 | 分辨率(0.02m)、相机参数(vfov=67.5°)、z_clip=0.5m |
| 流程 | L后续 | 1. 加载场景和语义GT |
| | | 2. 迭代多帧egocentric观察 |
| | | 3. 用projector投影到BEV |
| | | 4. 融合多帧结果 |

**应用场景:** 构建GT语义地图用于监督学习

#### [compute_GT_topdown_semantic_maps/build_semmap_from_obj_point_cloud.py](Semantic-MapNet111/compute_GT_topdown_semantic_maps/build_semmap_from_obj_point_cloud.py)
**从对象点云生成BEV语义地图 (论文使用方法)**

| 特点 | 说明 |
|------|------|
| 输入 | 场景的3D对象点云 + 语义标签 |
| 处理 | 直接投影3D点云到地面 |
| 优点 | 比egocentric投影更准确和完整 |
| 用途 | 生成无偏的GT语义地图 |

---

### 3️⃣ **数据预处理 (precompute_training_inputs/)**

#### [precompute_training_inputs/build_data.py](Semantic-MapNet111/precompute_training_inputs/build_data.py)
**构建训练数据 - 大规模FPV→BEV转换**

| 部分 | 行号 | 功能 |
|------|------|------|
| 导入 | L1-23 | 导入projector, RedNet(语义分割), HabitatUtils |
| 设置 | L25-33 | 分辨率0.02m, egocentric 480×640, 输出目录 |
| 主流程 | 后续 | 1. 遍历所有训练场景 |
| | | 2. 加载egocentric RGB-D和GT语义 |
| | | 3. 使用Projector投影到BEV |
| | | 4. 保存为h5文件(输入+目标) |

**处理规模:** MP3D数据集多个场景，生成BEV训练数据

#### [precompute_training_inputs/build_projindices.py](Semantic-MapNet111/precompute_training_inputs/build_projindices.py)
**预计算投影索引**

| 函数 | 功能 |
|------|------|
| `get_projections_indices()` | 预计算像素→BEV的映射关系，加速训练 |

#### [precompute_test_inputs/build_test_data.py](Semantic-MapNet111/precompute_test_inputs/build_test_data.py)
**构建测试数据 - 同上，用于测试集**

---

### 4️⃣ **模型网络 (SMNet/)**

#### [SMNet/model.py](Semantic-MapNet111/SMNet/model.py)
**语义映射网络主体**

| 模块 | 行号 | 功能 |
|------|------|------|
| Encoder | | 编码egocentric特征 |
| Spatial Memory Aggregator | | **多帧融合关键** - 累积BEV特征 |
| SemmapDecoder | L168 | 从聚合的BEV特征解码语义地图 |

**核心思路:** 
```
Egocentric Frame → Encode → Project to BEV → Accumulate 
  → Aggregate Memory → Decode → Topdown Semantic Map
```

#### [SMNet/model_test.py](Semantic-MapNet111/SMNet/model_test.py)
**测试时模型 (推理)**

| 内容 | 功能 |
|------|------|
| `SemmapDecoder` | L188 - 解码器子网络 |

---

### 5️⃣ **语义分割 (semseg/)**

#### [semseg/rednet.py](Semantic-MapNet111/semseg/rednet.py)
**RedNet - egocentric语义分割网络**

| 函数 | 行号 | 功能 |
|------|------|------|
| `forward_downsample()` | L155 | RGB-D下采样处理 |
| `forward()` | L223 | **完整前向**: RGB-D → 语义标签(H,W,C) |

**用途:** 
- 输入: Egocentric RGB-D
- 输出: 像素级语义标签(13类)
- 用于: 投影前的语义编码

---

### 6️⃣ **导航应用 (ObjectNav/)**

#### [ObjectNav/build_freespace_maps.py](Semantic-MapNet111/ObjectNav/build_freespace_maps.py)
**✅ 当前项目最常用脚本**

| 部分 | 功能 |
|------|------|
| `process_gt_h5()` | GT地面真值处理 - NavMesh投影 |
| `process_objnav_h5()` | ObjectNav处理 - 高度直方图 |
| 主程序 | 批量处理111个场景(22+89) |

**详见:** [之前提供的详细说明]

#### [ObjectNav/run_astar_planning.py](Semantic-MapNet111/ObjectNav/run_astar_planning.py)
**A*路径规划应用 - 使用freespace地图**

---

### 7️⃣ **工具和实用程序 (utils/)**

#### [utils/habitat_utils.py](Semantic-MapNet111/utils/habitat_utils.py)
**Habitat环境工具**

| 功能 | 说明 |
|------|------|
| 场景加载 | 初始化Habitat模拟器 |
| 位姿获取 | 获取agent的相机位姿 |
| 传感器读取 | 读取RGB-D-Semantic观察 |

#### [utils/semantic_utils.py](Semantic-MapNet111/utils/semantic_utils.py)
**语义标签工具**

| 函数 | 功能 |
|------|------|
| `color_label()` | 标签→RGB颜色映射 |
| 其他 | 标签处理和可视化 |

#### [utils/crop_memories.py](Semantic-MapNet111/utils/crop_memories.py)
**空间记忆裁剪**

| 功能 | 说明 |
|------|------|
| 动态裁剪 | 根据agent位置裁剪BEV特征图 |

---

## 🔄 数据流向图

```
┌─────────────────────────────────────────────────────────────┐
│                 Semantic Mapping Pipeline                    │
└─────────────────────────────────────────────────────────────┘

[多帧Egocentric RGB-D观察]
         ↓
[RedNet - 语义分割]  (semseg/rednet.py)
         ↓
[projector.core] 
  1. point_cloud()          深度→3D点
  2. transform_camera_to_world()  相机→世界坐标
  3. discretize_point_cloud()     3D→2D
         ↓
[Projector.forward()] (projector/projector.py)
  - 过滤异常点
  - 投影到BEV
         ↓
[融合多帧BEV特征]
  ↓ (使用SMNet/model.py的Spatial Memory)
  
[Decode] (SMNet/SemmapDecoder)
         ↓
[输出: Topdown语义地图]
         ↓
[应用]:
  - ObjectNav/freespace地图
  - ObjectNav/路径规划
```

---

## 💡 使用示例对应关系

| 场景 | 相关文件 | 函数调用链 |
|------|--------|---------|
| **实时机器人导航** | projector/{core,projector}.py | depth → ProjectorUtils.point_cloud() → Projector.forward() → BEV |
| **离线训练数据生成** | precompute_training_inputs/build_data.py | 批量迭代 → Projector → h5存储 |
| **GT标注生成** | compute_GT_topdown_semantic_maps/*.py | 点云 → 投影 → 语义地图 |
| **模型训练** | SMNet/model.py | BEV特征 → 空间聚合 → 解码 |
| **自由空间生成** | ObjectNav/build_freespace_maps.py | h5/navmesh → freespace地图 |

---

## 📊 代码复杂度和依赖关系

```
高层应用
  ↑
  ├─ SMNet (训练网络)
  ├─ ObjectNav (导航应用)
  └─ precompute_*_inputs (数据预处理)
       ↑
中间层
  ├─ projector/ (核心投影库) ⭐ 最重要
  └─ semseg/ (语义分割)
       ↑
低层基础
  ├─ utils/ (工具)
  └─ habitat-sim/lab (模拟器)
```

**最核心的模块:** `projector/core.py` + `projector/projector.py`

---

## 🎓 学习路径建议

### 快速入门 (30分钟)
1. 阅读本文档的"数据流向图"
2. 查看 `projector/core.py` 的 `point_cloud()` 和 `transform_camera_to_world()`

### 深入理解 (2小时)
1. 详细阅读 `projector/core.py` 全文 (276行)
2. 理解 `Projector.forward()` 的完整流程
3. 跟踪 `precompute_training_inputs/build_data.py` 的调用

### 实践应用 (1天)
1. 修改 `ObjectNav/build_freespace_maps.py` 适应自己的场景
2. 或扩展 `projector/` 支持新的特征投影
3. 集成到自己的机器人系统

---

## 📌 关键参数速查表

| 参数 | 默认值 | 说明 | 文件 |
|------|--------|------|------|
| `vfov` | 67.5° | 垂直视场角 | projector/core.py |
| `resolution` | 0.02 m | BEV地图分辨率 | precompute_training_inputs |
| `gridcellsize` | 0.02 m | 栅格大小 | projector/projector.py |
| `z_clip_threshold` | 0.5 m | 高度过滤阈值 | projector/projector.py |
| `world_shift_origin` | (x,y,z) | BEV原点偏移 | projector/core.py |
| 图像分辨率 | 480×640 | egocentric输入 | build_data.py |

