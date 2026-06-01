# 🎯 Semantic-MapNet111 快速参考卡

## 你问的问题
> "Semantic-MapNet111 项目里相关的代码文件是哪些？"

## 答案 (一句话)
**31 个代码文件**，分为**投影库、数据处理、神经网络、导航应用**四大类。

---

## ⚡ 最重要的 3 个文件

```
1. projector/core.py                    [276行] ⭐⭐⭐
   └─ 坐标变换: 深度→3D→世界坐标
   
2. projector/projector.py               [108行] ⭐⭐⭐
   └─ 完整投影: FPV→BEV的整个流程
   
3. ObjectNav/build_freespace_maps.py    [应用层] ⭐⭐⭐
   └─ 自由空间: 实际项目中最常用
```

---

## 📂 4 大模块分布

```
🔴 投影库 (4个文件)
   projector/{core.py, projector.py, point_cloud.py, __init__.py}
   
🟢 数据处理 (6个文件)
   precompute_training_inputs/ + precompute_test_inputs/
   
🔵 神经网络 (5个文件)
   SMNet/{model.py, model_test.py, loader.py, loss.py, utils.py}
   
🟡 语义分割 (1个文件)
   semseg/rednet.py
   
🟣 导航应用 (3个文件)
   ObjectNav/{build_freespace_maps.py, run_astar_planning.py, astar.py}
   
⚫ 工具和主程序 (6个文件)
   utils/ + train.py + test.py + demo.py + eval/ + metric/
```

---

## 🔄 数据流向 (核心管道)

```
RGB-D序列 
  ↓ [semseg/rednet.py]
语义分割
  ↓ [projector/core.py]
坐标变换 (深度→3D→世界)
  ↓ [projector/projector.py]
投影到BEV (2D栅格)
  ↓ [SMNet/model.py]
多帧融合 (空间记忆)
  ↓ [SMNet/SemmapDecoder]
BEV语义地图 (512×512×13)
  ↓
应用 (自由空间/路径规划)
```

---

## 📚 6 份文档对应关系

| # | 文档 | 大小 | 用途 |
|---|------|------|------|
| 🔍 | [00_START_HERE.md](00_START_HERE.md) | 12K | 📍 **总索引** (从这里开始) |
| ⚡ | [SEMANTIC_MAPNET_QUICK_NAV.md](SEMANTIC_MAPNET_QUICK_NAV.md) | 8K | 快速浏览(5分钟) |
| 📚 | [SEMANTIC_MAPNET_CODE_FILES_GUIDE.md](SEMANTIC_MAPNET_CODE_FILES_GUIDE.md) | 12K | 深入学习(详细指南) |
| 📋 | [SEMANTIC_MAPNET_ALL_FILES.md](SEMANTIC_MAPNET_ALL_FILES.md) | 12K | 完整清单(快速查阅) |
| 🔄 | [SEMANTIC_MAPNET_FLOW_DIAGRAM.md](SEMANTIC_MAPNET_FLOW_DIAGRAM.md) | 12K | 执行流程(跟踪代码) |
| 📖 | [README_CODE_DOCUMENTATION.md](README_CODE_DOCUMENTATION.md) | 12K | 文档导航(使用指南) |

---

## 🎯 如何使用这些文档

```
初学者 (5-30分钟)
  1. 打开 QUICK_NAV (5分钟概览)
  2. 打开 CODE_FILES_GUIDE (30分钟深入)
  3. 完成！已掌握核心

工程师 (修改代码)
  1. 打开 ALL_FILES (查找对应文件)
  2. 打开 FLOW_DIAGRAM (理解影响)
  3. 修改代码

研究者 (深度研究)
  1. 顺序读所有文档
  2. 仔细研究源代码
  3. 实验和扩展
```

---

## 🔑 关键参数一览

```python
# 相机
camera_height = 1.38 m      # 眼睛高度
vfov = 67.5°                # 垂直视场角
input_resolution = 480×640  # egocentric分辨率

# BEV地图
map_resolution = 0.02 m/pixel
map_size = 512×512 pixels   # = 10.24×10.24 m
z_clip = 0.5 m              # 天花板过滤

# 形态学
ObjectNav: kernel = 5×5 (闭运算)
NavMesh:   kernel = 3×3 (闭运算)
```

---

## 📍 快速定位代码

| 我要找... | 查看文件 | 位置 |
|---------|--------|------|
| 深度→3D点 | projector/core.py | point_cloud() [L116] |
| 应用位姿 | projector/core.py | transform_camera_to_world() [L151] |
| 完整投影 | projector/projector.py | Projector.forward() [L66] |
| 语义分割 | semseg/rednet.py | RedNet.forward() [L223] |
| 多帧融合 | SMNet/model.py | Spatial Memory Aggregator |
| 语义解码 | SMNet/model.py | SemmapDecoder [L168] |
| 自由空间 | ObjectNav/build_freespace_maps.py | process_*_h5() |
| 加载场景 | utils/habitat_utils.py | HabitatUtils 类 |

---

## 🚀 3 步快速开始

```
Step 1 (5分钟)
  打开: SEMANTIC_MAPNET_QUICK_NAV.md
  学习: 4个最重要文件、依赖关系

Step 2 (30分钟)
  打开: SEMANTIC_MAPNET_CODE_FILES_GUIDE.md
  学习: 投影库详解、参数含义

Step 3 (30分钟)
  打开: SEMANTIC_MAPNET_FLOW_DIAGRAM.md
  学习: 执行流程、函数调用链

>>> 1小时掌握项目核心！
```

---

## 💡 核心概念速记

**坐标变换 (Transform)**
```
pose = (x, y, z, heading, elevation)
  ↓ _transform3D()
T = 4×4 变换矩阵 (rotation + translation)
```

**投影流程 (Projection)**
```
深度像素 (u, v, depth)
  ↓ point_cloud() [相机内参]
相机3D点 (X_cam, Y_cam, Z_cam)
  ↓ transform_camera_to_world() [位姿T]
世界3D点 (X_world, Y_world, Z_world)
  ↓ discretize_point_cloud() [分辨率]
2D栅格 (grid_x, grid_y)
```

**多帧融合 (Aggregation)**
```
Frame_t egocentric
  ↓ 投影到BEV
BEV特征_t
  ↓ 与Memory融合
聚合特征
  ↓ 解码
Topdown Semantic_t
```

---

## 📊 代码统计

| 类别 | 文件数 | 重要性 |
|------|--------|--------|
| 投影库 | 4 | ⭐⭐⭐ 必读 |
| 数据处理 | 6 | ⭐⭐ 常用 |
| 神经网络 | 5 | ⭐⭐⭐ 创新 |
| 其他 | 16 | ⭐ 参考 |
| **总计** | **31** | |

---

## ✨ 你现在拥有

✅ 完整的代码地图
✅ 6份详细文档 (2000+行)
✅ 快速查阅索引
✅ 执行流程图示
✅ 学习路线建议
✅ 参数速查表

**开始阅读 → [00_START_HERE.md](00_START_HERE.md)**

---

## 🎓 推荐阅读顺序

```
1️⃣ 00_START_HERE.md           (总索引)
   ↓
2️⃣ SEMANTIC_MAPNET_QUICK_NAV.md (快速浏览)
   ↓
3️⃣ SEMANTIC_MAPNET_CODE_FILES_GUIDE.md (深入学习)
   或
   SEMANTIC_MAPNET_FLOW_DIAGRAM.md (理解流程)
   ↓
4️⃣ 查看源代码 (验证理解)
   ↓
5️⃣ SEMANTIC_MAPNET_ALL_FILES.md (快速参考)
```

---

**祝你使用愉快！🚀**

有任何问题，查阅对应文档或搜索关键字即可找到答案。

