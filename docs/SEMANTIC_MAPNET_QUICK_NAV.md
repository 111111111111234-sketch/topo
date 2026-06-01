# 📂 Semantic-MapNet111 代码文件快速导航

## ⚡ 最常用的4个文件

```
1. 核心投影库 (必读)
   └─ Semantic-MapNet111/projector/core.py (276行)
      函数: _transform3D(), ProjectorUtils
      功能: 深度→3D点→世界坐标变换

2. 投影器主类 (常用)
   └─ Semantic-MapNet111/projector/projector.py (108行)
      类: Projector
      功能: 完整的FPV→BEV投影流程

3. 训练数据生成 (实践)
   └─ Semantic-MapNet111/precompute_training_inputs/build_data.py (223行)
      函数: 主程序循环
      功能: 批量转换RGB-D→BEV

4. 导航应用 (当前项目)
   └─ Semantic-MapNet111/ObjectNav/build_freespace_maps.py
      函数: process_gt_h5(), process_objnav_h5()
      功能: 生成自由空间地图
```

---

## 📁 完整代码文件树 (按模块)

### 🔴 **核心转换 (projector/)**
```
projector/
├─ core.py           [276行] ⭐⭐⭐ 核心坐标变换
│  ├─ _transform3D() - pose→4×4变换矩阵
│  └─ ProjectorUtils - 投影工具基类
│     ├─ compute_intrinsic_matrix()
│     ├─ point_cloud()          [L116]
│     ├─ transform_camera_to_world() [L151]
│     └─ pixel_to_world_mapping() [L177]
│
├─ projector.py      [108行] ⭐⭐⭐ 投影器主类
│  └─ Projector.forward()       [L66] 完整流程
│
├─ point_cloud.py    特征投影
│  └─ PointCloud.forward()      [L56]
│
└─ __init__.py       导出接口
```

### 🟢 **地面真值生成 (compute_GT_topdown_semantic_maps/)**
```
compute_GT_topdown_semantic_maps/
├─ build_semmap_from_egoGT.py      [306行]
│  └─ 投影egocentric GT→BEV语义地图
│
└─ build_semmap_from_obj_point_cloud.py
   └─ 投影对象点云→BEV语义地图 (论文方法)
```

### 🔵 **数据预处理 (precompute_training_inputs/)**
```
precompute_training_inputs/
├─ build_data.py          [223行] ⭐⭐ 大规模FPV→BEV转换
│  └─ 迭代场景→RGB-D→投影→h5存储
│
├─ build_projindices.py   预计算投影索引
│  └─ get_projections_indices()
│
└─ build_crops.py         空间记忆裁剪

precompute_test_inputs/
└─ build_test_data.py     测试数据生成
```

### 🟡 **模型网络 (SMNet/)**
```
SMNet/
├─ model.py          [训练时模型]
│  └─ SemmapDecoder  [L168] 语义解码器
│
└─ model_test.py     [推理时模型]
   └─ SemmapDecoder  [L188]

SMNet/loader.py      数据加载器
SMNet/loss.py        损失函数
```

### 🟠 **语义分割 (semseg/)**
```
semseg/
└─ rednet.py         [Egocentric语义分割网络]
   ├─ forward_downsample() [L155]
   └─ forward()            [L223] RGB-D→语义标签
```

### 🟣 **导航应用 (ObjectNav/)**
```
ObjectNav/
├─ build_freespace_maps.py    [⭐ 当前项目最常用]
│  ├─ process_gt_h5()
│  ├─ process_objnav_h5()
│  └─ main程序
│
├─ run_astar_planning.py      路径规划
└─ astar.py                   A*算法实现
```

### ⚫ **工具程序 (utils/)**
```
utils/
├─ habitat_utils.py    [Habitat环境接口]
│  └─ HabitatUtils - 场景加载、位姿、传感器
│
├─ semantic_utils.py   [语义标签工具]
│  └─ color_label() - 标签→RGB映射
│
├─ crop_memories.py    [空间记忆处理]
│  └─ 动态裁剪BEV特征图
│
└─ __init__.py         [工具导出]
```

### 🔳 **其他模块**
```
train.py              主训练脚本
test.py               主测试脚本
demo.py               演示脚本
eval/                 评估指标
  ├─ eval.py
  ├─ eval_bfscore.py
  └─ bfscore.py
metric/               评估指标
  ├─ metric.py
  ├─ iou.py
  ├─ acc.py
  └─ confusionmatrix.py
```

---

## 🎯 按用途快速查找

### 我要**理解坐标变换**
```python
→ projector/core.py
  - _transform3D() [L6]      pose→SE(3)
  - point_cloud() [L116]     depth→3D点
  - transform_camera_to_world() [L151]  应用变换
```

### 我要**调整投影参数**
```python
→ projector/projector.py [L11-62]
  - vfov: 垂直视场角
  - gridcellsize: BEV分辨率
  - z_clip_threshold: 高度过滤
  - world_shift_origin: 坐标原点
```

### 我要**查看训练数据生成**
```python
→ precompute_training_inputs/build_data.py
  1. 加载场景 (HabitatUtils)
  2. 遍历关键帧
  3. 获取RGB-D-Semantic
  4. 投影到BEV (Projector)
  5. 保存为h5
```

### 我要**处理ObjectNav数据**
```python
→ ObjectNav/build_freespace_maps.py
  - process_gt_h5(): NavMesh→freespace
  - process_objnav_h5(): height_map→freespace
```

### 我要**看网络结构**
```python
→ SMNet/model.py [训练时]
  或 SMNet/model_test.py [推理时]
  
核心: Encoder → BEV投影 → 空间聚合 → Decoder
```

### 我要**获取egocentric语义**
```python
→ semseg/rednet.py
  - forward() [L223]: RGB-D→语义标签(13类)
```

---

## 🔗 代码依赖关系

```
高层应用
├─ train.py / test.py / demo.py
│  ↓
├─ SMNet/model.py  (网络主体)
│  ↓
├─ projector/projector.py  (投影器)
│  ↓
└─ projector/core.py  (坐标变换)
   ↓
└─ Habitat环境 + PyTorch
```

**最小依赖链:** 
```
RGB-D + Pose 
  → projector/core.py (point_cloud + transform)
  → Projector.forward()
  → BEV地图
```

---

## 📊 代码行数统计

| 模块 | 行数 | 复杂度 | 重要性 |
|------|------|--------|--------|
| projector/core.py | 276 | ⭐⭐⭐ | ⭐⭐⭐ |
| projector/projector.py | 108 | ⭐⭐ | ⭐⭐⭐ |
| precompute_training_inputs/build_data.py | 223 | ⭐⭐ | ⭐⭐ |
| compute_GT_topdown_semantic_maps/build_semmap_from_egoGT.py | 306 | ⭐⭐⭐ | ⭐ |
| SMNet/model.py | 较长 | ⭐⭐⭐ | ⭐⭐⭐ |
| semseg/rednet.py | 较长 | ⭐⭐⭐ | ⭐⭐ |

---

## 🚀 5分钟快速开始

**如果你只有5分钟，读这些:**

1. **projector/core.py** [L116-151]
   ```python
   # 深度图 → 3D点 → 世界坐标
   xyz_camera = self.point_cloud(depth)
   xyz_world = self.transform_camera_to_world(xyz_camera, T)
   ```

2. **projector/projector.py** [L66-85]
   ```python
   # 投影到BEV
   def forward(self, depth, T):
       point_cloud = self.pixel_to_world_mapping(depth, T)
       projection_indices_2D = self.discretize_point_cloud(point_cloud)
   ```

3. **ObjectNav/build_freespace_maps.py** [核心思路]
   - GT数据用NavMesh
   - ObjectNav用高度直方图

完成！你已理解核心思想。

