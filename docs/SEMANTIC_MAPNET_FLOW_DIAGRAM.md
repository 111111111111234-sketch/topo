# 📌 Semantic-MapNet 图表 ↔ 代码对应关系

## 根据论文Figure 1理解代码

```
┌─────────────────────────────────────────────────────────────────┐
│ Figure 1: Semantic Mapping Pipeline (论文图表)                  │
└─────────────────────────────────────────────────────────────────┘

(a) Agent Trajectory           →  代码: utils/habitat_utils.py
    (机器人运动轨迹)                HabitatUtils.get_agent_pose()
                                  获取位姿序列 [x,y,z,heading,elev]

      ↓

(b) Egocentric Observations    →  代码: semseg/rednet.py
    (多帧RGB-D第一人称视图)         RedNet.forward(rgb, depth)
                                  输入: 480×640 RGB + Depth
                                  输出: 语义标签(13类)

      ↓

(c) Spatial Memory             →  代码: projector/ + SMNet/model.py
    (3D体素网格)                    
    ├─ Depth投影:               projector/core.py
    │  ├─ depth_to_camera_3d()        深度 → 相机3D点
    │  ├─ transform_camera_to_world()  相机坐标 → 世界坐标
    │  └─ discretize_point_cloud()    3D点 → 2D栅格
    │
    ├─ 多帧融合:                SMNet/model.py
    │  └─ Spatial Memory Aggregator   累积特征到体素网格
    │
    └─ 参数:
       - 分辨率: 0.02m/pixel
       - 相机高度: 1.38m (HabitatUtils)
       - 视场角: vfov=67.5° (projector/core.py)

      ↓

(d) Top-down Segmentation     →  代码: SMNet/model.py
    (俯瞰图语义分割)                SemmapDecoder
                                  输入: 累积的BEV特征
                                  输出: 512×512×13 语义地图

```

---

## 完整代码执行流程

### 📌 **训练阶段** (training pipeline)

```
1. 数据生成
   ├─ precompute_training_inputs/build_data.py [主脚本]
   │  ├─ HabitatUtils.initialize_scene()  [utils/habitat_utils.py]
   │  │  └─ 加载MP3D场景
   │  │
   │  ├─ 迭代关键帧
   │  │  ├─ HabitatUtils.get_agent_pose()  获取位姿(x,y,z,heading,elev)
   │  │  │
   │  │  ├─ HabitatUtils.get_observations()  读取RGB-D
   │  │  │
   │  │  ├─ RedNet.forward(rgb, depth)  [semseg/rednet.py]
   │  │  │  输出: 语义标签(480,640,13)
   │  │  │
   │  │  ├─ PointCloud.forward(depth, T)  [projector/point_cloud.py]
   │  │  │  ├─ pixel_to_world_mapping(depth, T)
   │  │  │  │  ├─ point_cloud(depth)  深度→相机3D
   │  │  │  │  └─ transform_camera_to_world()  应用位姿
   │  │  │  │
   │  │  │  └─ discretize_point_cloud()  3D→2D栅格
   │  │  │
   │  │  └─ 保存到h5:
   │  │     ├─ 'rgb': (480,640,3)
   │  │     ├─ 'depth': (480,640)
   │  │     ├─ 'semantic': (480,640,13) [GT]
   │  │     └─ 'target': (512,512,13) [BEV GT]
   │
   └─ 输出: data/training/smnet_training_data/*.h5

2. 网络训练
   ├─ train.py
   │  ├─ SMNet/loader.py  加载h5数据
   │  │  └─ egocentric (rgb, depth, semantic) + BEV target
   │  │
   │  ├─ SMNet/model.py   定义网络
   │  │  ├─ Encoder: RGB-D → egocentric特征
   │  │  ├─ Projection: egocentric→BEV特征投影
   │  │  ├─ Spatial Memory: 多帧融合
   │  │  └─ SemmapDecoder: BEV特征→语义地图
   │  │
   │  ├─ SMNet/loss.py    计算损失
   │  │  └─ CrossEntropyLoss(pred, target)
   │  │
   │  └─ 输出: smnet_mp3d_best_model.pkl
```

### 📌 **测试/推理阶段** (inference pipeline)

```
1. 初始化模型
   ├─ test.py
   │  ├─ 加载预训练权重 (smnet_mp3d_best_model.pkl)
   │  ├─ SMNet/model_test.py  推理模型
   │  │
   │  └─ 初始化场景 (HabitatUtils)

2. 逐帧推理
   ├─ 读取egocentric RGB-D
   │  └─ HabitatUtils.get_observations()
   │
   ├─ 语义分割
   │  └─ RedNet.forward(rgb, depth)  → 语义标签
   │
   ├─ 投影到BEV
   │  ├─ PointCloud.forward(depth, T)
   │  │  └─ pixel_to_world_mapping()  完整投影
   │  │
   │  └─ 输出: BEV特征
   │
   ├─ 网络前向
   │  ├─ SMNet forward  (encoder + projection + aggregation)
   │  └─ SemmapDecoder  → BEV语义地图(512,512,13)
   │
   └─ 可视化或保存

3. 评估
   ├─ eval/eval.py
   │  ├─ 计算mIoU, mAcc  [metric/iou.py, metric/acc.py]
   │  ├─ 混淆矩阵  [metric/confusionmatrix.py]
   │  └─ BFS距离  [eval/bfscore.py]
```

### 📌 **应用阶段** (ObjectNav/freespace)

```
ObjectNav/build_freespace_maps.py  [主脚本]
  │
  ├─ GT数据处理
  │  └─ process_gt_h5(env_name, pathfinder_cache, semmap_info)
  │     ├─ 加载NavMesh: /data/mp3d/{house}/{house}.navmesh
  │     ├─ habitat_sim.PathFinder.load_nav_mesh()
  │     ├─ pathfinder.get_topdown_view(resolution=0.02, height=y_min_value+0.1)
  │     │  输出: 二进制可走性地图
  │     ├─ 坐标对齐 (map_world_shift元数据)
  │     └─ 输出: /data/freespace_map/{scene_id}.png
  │
  ├─ ObjectNav数据处理
  │  └─ process_objnav_h5(path, semmap_info)
  │     ├─ 读取h5: height_map, semmap, observed_map
  │     ├─ build_freespace_from_height_histogram()  [fpv_to_bev.py]
  │     │  ├─ 直方图峰值 → 地面高度
  │     │  ├─ 高度阈值过滤
  │     │  └─ 形态学闭运算 (5×5或3×3核)
  │     └─ 输出: /data/ObjectNav/freespace_map/{scene_id}.png
  │
  └─ 路径规划
     ├─ run_astar_planning.py
     └─ astar.py  使用freespace地图进行搜索
```

---

## 🔗 关键函数调用链

### 链1: 深度图转BEV完整流程
```
depth(H,W) + T(4,4) + intrinsics
  ↓
projector/core.py:ProjectorUtils.compute_intrinsic_matrix()
  ↓ K = (fx, fy, cx, cy)
projector/core.py:ProjectorUtils.point_cloud(depth)
  ↓ xyz_camera = (H, W, 3)
projector/core.py:ProjectorUtils.transform_camera_to_world(xyz_camera, T)
  ↓ xyz_world = (H, W, 3)
projector/core.py:ProjectorUtils.discretize_point_cloud(xyz_world, camera_height)
  ↓ u_indices, v_indices = (H, W)
projector/projector.py:Projector.forward(depth, T)
  ↓ 输出: mask(H_out, W_out) + projection_indices_2D
```

### 链2: 训练数据生成
```
for each scene:
  for each frame:
    HabitatUtils.get_agent_pose()  → T
    HabitatUtils.get_observations()  → rgb, depth, sem_gt
    ↓
    RedNet.forward(rgb, depth)  → sem_pred
    ↓
    PointCloud.forward(depth, T)  → BEV特征投影
    ↓
    save h5 {egocentric_inputs, BEV_target}
```

### 链3: 网络训练
```
loader.py(h5数据)
  ↓
SMNet.Encoder(egocentric)
  ↓
projector投影到BEV  [这里再次使用投影]
  ↓
SMNet.Spatial Memory Aggregator(多帧融合)
  ↓
SMNet.SemmapDecoder(BEV特征→语义地图)
  ↓
loss.py(CrossEntropyLoss)
  ↓
反向传播更新权重
```

### 链4: 自由空间生成
```
方式1 (GT):
  NavMesh文件
    ↓
  habitat_sim.PathFinder
    ↓
  get_topdown_view()
    ↓
  freespace_map

方式2 (ObjectNav):
  h5文件 {height_map, semmap, observed_map}
    ↓
  fpv_to_bev.build_freespace_from_height_histogram()
    ↓
  freespace_map
```

---

## 📊 模块关系图

```
输入数据
  ├─ RGB-D序列  →  semseg/rednet.py (语义分割)
  ├─ 位姿序列   →  utils/habitat_utils.py
  └─ 场景文件   →  Habitat模拟器

投影层 (核心)
  └─ projector/
     ├─ core.py  (坐标变换)
     ├─ projector.py  (占据投影)
     └─ point_cloud.py  (特征投影)

网络层
  └─ SMNet/
     ├─ model.py  (Encoder + Spatial Memory + Decoder)
     ├─ loss.py
     └─ loader.py

应用层
  ├─ train.py / test.py  (模型训练推理)
  ├─ ObjectNav/build_freespace_maps.py  (应用)
  └─ ObjectNav/run_astar_planning.py  (导航)

评估层
  └─ metric/ + eval/  (性能评估)
```

---

## 🎯 查找代码的方法

| 我想了解... | 查看文件 | 具体位置 |
|------------|--------|--------|
| 深度图怎样变成3D点 | projector/core.py | point_cloud() [L116] |
| 3D点怎样变成BEV坐标 | projector/core.py | discretize_point_cloud() |
| 位姿怎样表示 | projector/core.py | _transform3D() [L6] |
| egocentric怎样投影到BEV | projector/projector.py | forward() [L66] |
| 多帧怎样融合 | SMNet/model.py | Spatial Memory Aggregator |
| BEV特征怎样生成语义地图 | SMNet/model.py | SemmapDecoder [L168] |
| 自由空间怎样生成 | ObjectNav/build_freespace_maps.py | process_*_h5() |
| 怎样从场景读数据 | utils/habitat_utils.py | HabitatUtils类 |
| egocentric语义怎样获得 | semseg/rednet.py | forward() [L223] |

---

## 📝 核心参数默认值

```python
# 相机参数
camera_height = 1.38  # meters
camera_tilt = 0.174  # radians (10 degrees)
vfov = 67.5  # degrees 或 1.178 radians
egocentric_resolution = (480, 640)  # H, W

# BEV参数
map_resolution = 0.02  # meters per pixel
topdown_map_size = (512, 512)  # H, W
gridcellsize = 0.02  # same as map_resolution
z_clip_threshold = 0.50  # meters (ceiling filter)

# 形态学参数
ObjectNav: kernel_size = 5×5  (闭运算)
NavMesh: kernel_size = 3×3    (闭运算)

# 语义类别
num_classes = 13  # MP3D数据集
```

这样就完整了！你现在有了完整的代码文件对应关系。
