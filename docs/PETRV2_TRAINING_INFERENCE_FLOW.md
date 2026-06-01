# PETRV2 BEVSeg 训练推理流程完全指南

## 📋 总览

```
数据集 → 数据管道 → 模型 → 损失计算 / 推理 → 结果
(Dataset)  (Transform)  (Model)  (Loss/Predict)  (Output)
```

---

## 🎯 Part I: 训练流程（Training）

### 1️⃣ **初始化阶段 - Runner 构建**

#### 代码示例：

```python
from mmengine.config import Config
from mmengine.runner import Runner

# 加载配置
cfg = Config.fromfile('projects/PETRV2/configs/petrv2_bevseg.py')
cfg.work_dir = 'work_dirs/petrv2_bevseg_train'

# 构造 Runner（自动构建数据加载、模型、优化器、调度器）
runner = Runner.from_cfg(cfg)
```

#### Runner 做了什么：

```
Config (petrv2_bevseg.py)
    ├─ model 字段 → 调用 build_detector()
    │   └─ Petr3D_seg(img_backbone=VoVNetCP, pts_bbox_head=PETRHeadlane_cvt200_h3_f3, ...)
    │       ↓
    │       runner.model
    │
    ├─ train_dataloader 字段 → 调用 build_dataloader()
    │   └─ DataLoader(NuScenesMapDataset, train_pipeline, batch_size=1, num_workers=2)
    │       ↓
    │       runner.train_dataloader
    │
    ├─ optim_wrapper 字段 → 调用 build_optim_wrapper()
    │   └─ OptimWrapper(optimizer=AdamW(lr=2e-4), clip_grad=35)
    │       ↓
    │       runner.optim_wrapper
    │
    └─ train_cfg 字段
        └─ max_epochs=24, val_interval=24
            ↓
            runner.train_cfg
```

---

### 2️⃣ **数据加载阶段 - Pipeline**

#### 关键 Transform 链（train_pipeline）：

```python
train_pipeline = [
    # Step 1: 加载多视图图像（6摄像头）
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, backend_args=None),
    
    # Step 2: 加载地图数据并二值化
    dict(type='LoadMapsFromFiles_flattenf200f3', map_one_is_bg=False),
    
    # Step 3: 加载多帧数据（当前帧 + 1 帧历史 = 12 张图）
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', 
         sweeps_num=1,              # 加载 1 帧历史
         to_float32=True,
         pad_empty_sweeps=True,     # 如果无历史，复制当前帧
         test_mode=False,           # 训练时随机选择历史帧
         sweep_range=[3, 27]),      # 从 3-27 ms 的历史帧中选择
    
    # Step 4: 加载 3D 标注（边界框、标签）
    dict(type='LoadAnnotations3D', 
         with_bbox_3d=True, 
         with_label_3d=True, 
         with_attr_label=False),
    
    # Step 5: 过滤点云范围外的物体
    dict(type='ObjectRangeFilter', 
         point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    
    # Step 6: 过滤不在 class_names 中的物体
    dict(type='ObjectNameFilter', 
         classes=['car', 'truck', ..., 'traffic_cone']),
    
    # Step 7: 图像归一化（ImageNet 标准）
    dict(type='NormalizeMultiviewImage', 
         mean=[103.530, 116.280, 123.675],
         std=[57.375, 57.120, 58.395], 
         to_rgb=False),            # 输入是 BGR 格式
    
    # Step 8: 图像填充（对齐到 32 的倍数）
    dict(type='PadMultiViewImage', size_divisor=32),
    
    # Step 9: ⭐ 打包成模型输入格式
    dict(type='Pack3DDetAndSegInputs', 
         keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'maps', 'gt_map'])
]
```

#### 数据流详解：

```
样本来自 NuScenesMapDataset：
{
    'img': [6, H, W, 3],              # 6 摄像头当前帧
    'img_path': [...],
    'timestamp': float,               # lidar 时间戳
    'img_timestamp': [6],             # 各摄像头时间戳
    'lidar2cam': [[4,4], ...],        # 6 个转换矩阵
    'cam2img': [[3,3], ...],          # 6 个内参矩阵
    'sweeps': [sweep0, sweep1, ...],  # 历史帧数据
    'gt_bboxes_3d': [...],            # 检测 GT
    'gt_labels_3d': [...],
    'map_filename': '/path/to/xxx.npz'  # ⭐ token 匹配得到
}

↓ (transform 链处理)

DataSample (送入模型)：
{
    'inputs': {
        'imgs': Tensor[B=1, N=12, C=3, H, W]  # 1 batch × 12 张图 (6*2)
    },
    'metainfo': {
        'img_shape': (H, W),
        'img_norm_cfg': {'mean': [...], 'std': [...]},
        'timestamp': [6, 6],           # 12 个时间戳
        'lidar2cam': [[4,4], ...],     # 12 个矩阵
        'cam2img': [[3,3], ...],
        'lidar2img': [[4,4], ...],     # 新增（在 petr3d_seg.add_lidar2img 中补全）
        ...
    },
    'gt_instances_3d': {
        'bboxes_3d': Tensor[M, 7],     # M 个 GT 框
        'labels_3d': Tensor[M]
    },
    'gt_map_seg': Tensor[3, 40000],    # ⭐ 分割 GT（展平）
    'gt_map': Tensor[3, 200, 200]      # 原始地图（用于可视化）
}
```

---

### 3️⃣ **模型前向与损失计算 - loss()**

#### 调用路径：

```
Runner.train_loop()
    ↓
for batch in dataloader:
    ↓
model.loss(inputs=batch['inputs'], data_samples=batch['data_samples'])
    ↓ (进入 Petr3D_seg.loss())
```

#### Petr3D_seg.loss() 详解：

```python
def loss(self, inputs=None, data_samples=None, **kwargs):
    """
    输入：
      inputs: {'imgs': [B=1, N=12, C=3, H, W]}
      data_samples: List[DataSample]
    """
    
    # Step 1: 提取输入和标注
    img = inputs['imgs']                                    # [B, N=12, C=3, H, W]
    batch_img_metas = [ds.metainfo for ds in data_samples]  # 元信息
    
    batch_gt_instances_3d = [ds.gt_instances_3d for ds in data_samples]
    gt_bboxes_3d = [gt.bboxes_3d for gt in batch_gt_instances_3d]  # 检测框
    gt_labels_3d = [gt.labels_3d for gt in batch_gt_instances_3d]  # 检测标签
    
    # ⭐ Step 2: 提取分割 GT（关键！）
    maps = [getattr(ds, 'gt_map_seg', None) for ds in data_samples]  # [B, 3, 40000]
    
    # Step 3: 补全 lidar2img 矩阵（v1.4 兼容性）
    batch_img_metas = self.add_lidar2img(img, batch_img_metas)
    
    # Step 4: 图像特征提取 (CNN)
    img_feats = self.extract_feat(img=img, img_metas=batch_img_metas)
    #           ├─ img_backbone: VoVNetCP(input: [B*N, C=3, H, W])
    #           │   └─ 输出: List of multi-level features
    #           ├─ img_neck: CPFPN
    #           │   └─ 输出: [B, N=12, C=256, H'=25, W'=25], [B, N, 256, 50, 50]
    #           └─ 返回: img_feats_reshaped List[[B, N, C, H', W'], ...]
    
    # Step 5: 调用 3D 检测头的前向和损失
    losses = self.forward_pts_train(
        img_feats,          # [B, N, 256, 25, 25], [B, N, 256, 50, 50]
        gt_bboxes_3d,       # 检测框 GT
        gt_labels_3d,       # 检测标签 GT
        maps,               # [B, 3, 40000] ⭐ 分割 GT
        batch_img_metas
    )
    #   ↓ (内部调用)
    # outs = self.pts_bbox_head(img_feats, img_metas)
    # losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs, maps)
    
    # Step 6: 返回损失字典
    return losses  # Dict: {'loss_dri': tensor, 'loss_lan': tensor, ...}
```

---

### 4️⃣ **分割头的损失计算 - PETRHead_seg.loss()**

#### PETRHeadlane_cvt200_h3_f3 的前向与损失：

```python
# 在 pts_bbox_head.loss() 中：

def loss(self, gt_bboxes_3d, gt_labels_3d, outs, maps):
    """
    outs 来自 forward(img_feats, img_metas)，包含：
      {
          'all_lane_preds': List[6] of [B, 3, 200, 200]  # 6层decoder每层输出
          'all_cls_preds': List[6] of [B, M]             # (检测关闭，为空)
          'all_bbox_preds': List[6] of [B, M, 7]         # (检测关闭，为空)
      }
    
    maps: [B, 3, 40000]
      ├─ Channel 0: 驾驶区域 (Drivable)
      ├─ Channel 1: 车道线 (Lane)
      └─ Channel 2: 车辆 (Vehicle)
    """
    
    losses = {}
    
    # 📍 分割损失（多层监督）
    if self.enable_seg:
        # 展开 all_lane_preds: [6, B, 3, 200, 200] → [6, B, 3, 40000]
        lane_preds = [pred.flatten(2) for pred in outs['all_lane_preds']]  # 空间维展平
        
        # 准备分割 GT 列表
        gt_lane_list = [
            maps[:, 0, :],  # 驾驶区域 [B, 40000]
            maps[:, 1, :],  # 车道线 [B, 40000]
            maps[:, 2, :]   # 车辆 [B, 40000]
        ]
        
        # 计算分割损失（调用 loss_lane_single）
        loss_lane = self.loss_lane_single(lane_preds, gt_lane_list, labels=None)
        losses['loss_lane'] = loss_lane
    
    return losses
```

#### loss_lane_single 详解：

```python
def loss_lane_single(self, lane_preds, gt_lane_list, labels):
    """
    lane_preds: List[6] of [B, 3, 40000]  (6层decoder每层的预测)
    gt_lane_list: List[3]
      ├─ [0] [B, 40000] drivable
      ├─ [1] [B, 40000] lane
      └─ [2] [B, 40000] vehicle
    """
    
    loss = torch.tensor(0.0, device=lane_preds[0].device)
    
    # 对 6 层 decoder 的每一层，都计算损失
    for layer_idx in range(len(lane_preds)):  # 6 layers
        
        pred_layer = lane_preds[layer_idx]  # [B, 3, 40000]
        
        # 驾驶区域损失（Focal Loss）
        loss_dri = self.loss_dri(
            pred_layer[:, 0, :],      # 预测 logits [B, 40000]
            gt_lane_list[0],          # GT [B, 40000]
            labels
        )  # output: scalar
        
        # 车道线损失
        loss_lan = self.loss_lan(
            pred_layer[:, 1, :],
            gt_lane_list[1],
            labels
        )
        
        # 车辆损失
        loss_veh = self.loss_veh(
            pred_layer[:, 2, :],
            gt_lane_list[2],
            labels
        )
        
        # 累加三个通道的损失
        loss = loss + loss_dri + loss_lan + loss_veh
        
        # loss_dri / loss_lan / loss_veh 的权重：
        #   loss_dri.weight = 2.0  (驾驶区域)
        #   loss_lan.weight = 4.0  (车道线，难度最高)
        #   loss_veh.weight = 8.0  (车辆，难度最高)
    
    # 总损失 = 6层 × (loss_dri + loss_lan + loss_veh)
    #        = 6 × (2.0 + 4.0 + 8.0) = 6 × 14.0 = 84.0 权重单位
    
    return loss / (len(lane_preds) * 3)  # 平均化
```

#### 损失函数配置（FocalLoss）：

```python
# 来自 configs/petrv2_bevseg.py

loss_dri = dict(
    type='mmdet.FocalLoss',
    use_sigmoid=True,    # 应用 sigmoid，输出 [0, 1]
    gamma=2.0,          # 困难样本权重指数
    alpha=0.5,          # 正样本权重
    loss_weight=2.0     # 通道权重
)

loss_lan = dict(
    type='mmdet.FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.5,
    loss_weight=4.0     # 车道线权重最高，难度最大
)

loss_veh = dict(
    type='mmdet.FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.5,
    loss_weight=8.0     # 车辆权重第二高
)
```

**FocalLoss 公式：**
$$
\text{FL}(p_t) = -\alpha (1-p_t)^{\gamma} \log(p_t)
$$

- $p_t$: 模型预测的概率（0到1）
- $\gamma=2.0$: 困难样本权重（当 $p_t$ 很小时，$(1-p_t)^2$ 很大）
- $\alpha=0.5$: 正样本权重

---

### 5️⃣ **反向传播与优化**

#### 优化器配置：

```python
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=2e-4,              # 学习率
        weight_decay=0.01     # L2 正则化
    ),
    paramwise_cfg=dict(
        custom_keys={'img_backbone': dict(lr_mult=0.1)}  # 骨干网络学习率降低 10 倍
    ),
    clip_grad=dict(max_norm=35, norm_type=2)  # 梯度裁剪
)
```

#### 学习率调度：

```python
param_scheduler = [
    # 前 500 步：线性预热，从 1/3 到 1
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        begin=0,
        end=500,
        by_epoch=False
    ),
    # 之后：余弦退火，24 个 epoch 内衰减到 0
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        by_epoch=True
    )
]
```

#### Runner 优化流程：

```
for epoch in range(24):
    for batch_idx, batch in enumerate(train_dataloader):
        
        # 前向
        losses = model.loss(inputs=batch['inputs'], data_samples=batch['data_samples'])
        #        返回 Dict: {'loss_dri': t1, 'loss_lan': t2, 'loss_veh': t3}
        
        # 计算总损失
        loss = sum(losses.values())
        
        # 反向
        optim_wrapper.backward(loss)
        
        # 梯度裁剪
        optim_wrapper.clip_grad_norm(max_norm=35)
        
        # 优化器更新
        optim_wrapper.step()
        optim_wrapper.zero_grad()
        
        # 学习率调度
        param_scheduler.step()
    
    # 验证
    if epoch == 23:
        model.eval()
        # 运行验证集
```

---

## 🔍 Part II: 推理流程（Inference）

### 1️⃣ **推理初始化**

#### 代码示例：

```python
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile('projects/PETRV2/configs/petrv2_bevseg.py')
cfg.work_dir = 'work_dirs/petrv2_bevseg_vis'
cfg.load_from = 'work_dirs/petrv2_bevseg_sanity/iter_20.pth'  # 加载训练好的权重

# 构造 Runner 并加载权重
runner = Runner.from_cfg(cfg)
model = runner.model.cuda().eval()  # 设置为评估模式

# 获取测试数据加载器
test_dataloader = runner.test_dataloader
```

---

### 2️⃣ **测试数据管道**

#### test_pipeline 与 train_pipeline 的区别：

```python
test_pipeline = [
    # 基本相同的前 8 步
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, backend_args=None),
    dict(type='LoadMapsFromFiles_flattenf200f3', map_one_is_bg=False),
    
    # ⭐ 区别 1: test_mode=True（选择固定的历史帧）
    dict(type='LoadMultiViewImageFromMultiSweepsFiles',
         sweeps_num=1,
         to_float32=True,
         pad_empty_sweeps=True,
         test_mode=True,           # 固定选择中间帧 (vs. 训练时随机)
         sweep_range=[3, 27]),
    
    # 注意：没有 LoadAnnotations3D（推理时无 GT）
    
    dict(type='NormalizeMultiviewImage', ...),
    dict(type='PadMultiViewImage', size_divisor=32),
    
    # ⭐ 区别 2: 推理时只打包 img 和 maps，不打包 GT
    dict(type='Pack3DDetAndSegInputs', keys=['img', 'maps', 'gt_map'])
]
```

---

### 3️⃣ **模型推理 - predict()**

#### 调用路径：

```
for batch in test_dataloader:
    ↓
model.predict(inputs=batch['inputs'], data_samples=batch['data_samples'])
    ↓ (进入 Petr3D_seg.predict())
```

#### Petr3D_seg.predict() 详解：

```python
def predict(self, inputs=None, data_samples=None, **kwargs):
    """
    输入：
      inputs: {'imgs': [B=1, N=12, C=3, H, W]}
      data_samples: List[DataSample]（无 gt_instances_3d）
    """
    
    # Step 1: 提取输入
    img = inputs['imgs']
    batch_img_metas = [ds.metainfo for ds in data_samples]
    
    # Step 2: 获取 GT map（推理时可能为 None，用于计算 IOU）
    maps = [getattr(ds, 'gt_map_seg', None) for ds in data_samples]
    
    # Step 3: 补全 lidar2img 矩阵
    batch_img_metas = self.add_lidar2img(img, batch_img_metas)
    
    # Step 4: 调用 simple_test（单样本推理）
    results_list_3d = self.simple_test(batch_img_metas, maps, img, **kwargs)
    #                 ↓
    #                 simple_test_pts() → 返回 bbox_results, ret_ious
    
    # Step 5: 将结果打包回 data_sample
    for i, data_sample in enumerate(data_samples):
        pred = results_list_3d[i]['pts_bbox']
        
        # 创建预测实例
        pred_instances_3d = InstanceData()
        pred_instances_3d.bboxes_3d = pred['boxes_3d']  # [N_pred, 7]
        pred_instances_3d.scores_3d = pred['scores_3d']  # [N_pred]
        pred_instances_3d.labels_3d = pred['labels_3d']  # [N_pred]
        
        data_sample.pred_instances_3d = pred_instances_3d
        data_sample.pred_instances = InstanceData()  # 空 2D 检测
    
    # Step 6: 返回包含预测的 data_sample 列表
    return data_samples
```

---

### 4️⃣ **单样本推理详解 - simple_test_pts()**

#### 流程：

```python
def simple_test_pts(self, x, img_metas, gt_map, rescale=False):
    """
    输入：
      x: img_feats [B, N, 256, 25, 25], [B, N, 256, 50, 50] ...
      img_metas: List[Dict] 元信息
      gt_map: [B, 3, 40000] 或 None
    """
    
    # Step 1: 通过分割头前向
    outs = self.pts_bbox_head(x, img_metas)
    # outs = {
    #     'all_lane_preds': List[6] of [B, 3, 200, 200]
    # }
    
    # Step 2: 获取 3D 检测框（检测关闭，返回空）
    bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
    bbox_results = []
    for bboxes, scores, labels in bbox_list:
        bbox_results.append(
            dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
        )
    
    # ⭐ Step 3: 分割后处理 + IOU 计算
    with torch.no_grad():
        # 取最后一层 decoder 的输出
        lane_preds = outs['all_lane_preds'][5].squeeze(0)  # [B, 3, 200, 200] → [3, 200, 200]
        
        # 应用 sigmoid，转为概率
        f_lane = lane_preds.sigmoid()  # [0, 1]
        
        # ⭐ 硬阈值：0.43
        f_lane[f_lane >= 0.43] = 1
        f_lane[f_lane < 0.43] = 0
        # 结果：[3, 200, 200]，二值化的分割掩码
        
        # 展平用于 IOU 计算
        f_lane = f_lane.view(3, -1)  # [3, 40000]
        
        # 计算 IOU（仅当有 GT 时）
        ret_ious = None
        if gt_map is not None and gt_map[0] is not None:
            curr_gt_map = gt_map[0].view(3, -1)  # [3, 40000]
            inter, union = IOU(f_lane, curr_gt_map)
            ret_ious = [inter, union]  # ([inter_dri, inter_lan, inter_veh], [...])
    
    return bbox_results, ret_ious
```

#### 硬阈值 0.43 的含义：

```
sigmoid 输出范围：[0, 1]
阈值 0.43：当模型置信度 ≥ 0.43 时，将像素分类为前景

原始 PETR 经验：
  - 0.43 是在训练集上验证的最优阈值
  - 不同的地图类型可能需要不同的阈值
```

**示例代码（寻找最优阈值）：**

```python
# 来自上下文的终端输出
model.eval()
batch = next(iter(runner.test_dataloader))
with torch.no_grad():
    data = model.data_preprocessor(batch, False)
    imgs = data['inputs']['imgs']
    dss = data['data_samples']
    metas = [ds.metainfo for ds in dss]
    metas = model.add_lidar2img(imgs, metas)
    feats = model.extract_feat(img=imgs, img_metas=metas)
    outs = model.pts_bbox_head(feats, metas)
    prob = outs['all_lane_preds'][-1].sigmoid().view(-1, 3, 200, 200)[0]
    
    print('min max mean', float(prob.min()), float(prob.max()), float(prob.mean()))
    
    # 测试多个阈值
    for t in [0.43, 0.48, 0.5, 0.52, 0.55]:
        m = (prob >= t).float().mean(dim=(1, 2))  # 计算覆盖率
        print('thr', t, 'cov', m.tolist())

# 输出示例：
# min max mean 0.0 0.988 0.316
# thr 0.43 cov [0.35, 0.47, 0.18]  # [drivable, lane, vehicle]
# thr 0.48 cov [0.31, 0.43, 0.15]
# thr 0.5  cov [0.29, 0.41, 0.14]
# thr 0.52 cov [0.27, 0.38, 0.12]
# thr 0.55 cov [0.25, 0.35, 0.10]
```

---

### 5️⃣ **输出格式**

#### 推理返回的 DataSample：

```python
data_sample = {
    'pred_instances_3d': {
        'bboxes_3d': Tensor[N_pred, 7],  # [x, y, z, w, l, h, θ]
        'scores_3d': Tensor[N_pred],
        'labels_3d': Tensor[N_pred]
    },
    'pred_instances': InstanceData(),    # 空 2D 预测
    'metainfo': {...},
    'gt_map': Tensor[3, 200, 200],       # 可选，用于可视化
}
```

#### IOU 指标（ret_ious）：

```python
# 返回值 ret_ious = [inter, union]
# inter: [inter_dri, inter_lan, inter_veh]  交集像素数
# union: [union_dri, union_lan, union_veh]  并集像素数

# IOU 计算
iou_dri = inter[0] / union[0]
iou_lan = inter[1] / union[1]
iou_veh = inter[2] / union[2]
mIOU = (iou_dri + iou_lan + iou_veh) / 3
```

---

## 📊 完整数据流图表

### 训练数据流：

```
┌─────────────────────────────────────────────────────────────────┐
│ Dataset: NuScenesMapDataset                                      │
│ - Sample token → map_filename (token matching)                  │
│ - 检测注解 (mmdet3d_nuscenes_30f_infos_train.pkl)              │
│ - 地图注解 (HDmaps-final_infos_train.pkl)                      │
└──────────────┬──────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────────┐
│ Pipeline (train_pipeline)                                        │
│ 1. LoadMultiViewImageFromFiles           → [6, H, W, 3]        │
│ 2. LoadMapsFromFiles_flattenf200f3       → [3, 40000]          │
│ 3. LoadMultiViewImageFromMultiSweepsFiles → [12, H, W, 3]      │
│ 4. LoadAnnotations3D                     → GT 框和标签          │
│ 5. ObjectRangeFilter, ObjectNameFilter   → 过滤                 │
│ 6. NormalizeMultiviewImage               → ImageNet 标准化     │
│ 7. PadMultiViewImage                     → size_divisor=32     │
│ 8. Pack3DDetAndSegInputs                 → DataSample          │
└──────────────┬──────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────────┐
│ DataSample {                                                     │
│   inputs: {'imgs': [B=1, N=12, C=3, H, W]},                     │
│   metainfo: {img_shape, lidar2img, timestamp, ...},             │
│   gt_instances_3d: {bboxes_3d, labels_3d},                      │
│   gt_map_seg: [3, 40000] ← ⭐ 分割 GT                           │
│ }                                                                │
└──────────────┬──────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────────┐
│ Model Forward: Petr3D_seg.loss()                                 │
│ ├─ extract_feat()        → img_feats [B, N, 256, H', W']       │
│ │  ├─ img_backbone (VoVNetCP): [B*N, 3, H, W] → features      │
│ │  └─ img_neck (CPFPN): multi-level fusion                     │
│ │                                                               │
│ ├─ forward_pts_train()   → outs                                 │
│ │  └─ pts_bbox_head()    → {all_lane_preds: [6, B, 3, 200, 200]}
│ │                                                               │
│ └─ pts_bbox_head.loss()  → losses Dict                          │
│    ├─ loss_lane_single() × 6 layers (multi-scale supervision)   │
│    │  ├─ loss_dri (weight=2.0)  Σ(layers) FocalLoss           │
│    │  ├─ loss_lan (weight=4.0)  Σ(layers) FocalLoss           │
│    │  └─ loss_veh (weight=8.0)  Σ(layers) FocalLoss           │
│    └─ 返回: {'loss_lane': scalar}                              │
└──────────────┬──────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────────┐
│ Optimization                                                     │
│ ├─ optim_wrapper.backward(loss)  ← 反向传播                     │
│ ├─ clip_grad_norm(max_norm=35)   ← 梯度裁剪                     │
│ ├─ optim_wrapper.step()          ← 更新参数                     │
│ └─ param_scheduler.step()        ← 学习率调度                   │
└──────────────────────────────────────────────────────────────────┘
```

### 推理数据流：

```
┌──────────────────────────────────────────────────────────────────┐
│ Dataset: NuScenesMapDataset (test_mode=True)                     │
└──────────────┬──────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────────┐
│ Pipeline (test_pipeline)                                         │
│ - 同训练，但 test_mode=True，不包含 LoadAnnotations3D            │
└──────────────┬──────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────────┐
│ DataSample {                                                     │
│   inputs: {'imgs': [B=1, N=12, C=3, H, W]},                     │
│   metainfo: {...},                                              │
│   gt_map: [3, 200, 200] (可选，用于评估)                        │
│ }                                                                │
└──────────────┬──────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────────┐
│ Model Forward: Petr3D_seg.predict()                              │
│ ├─ extract_feat()     → img_feats                                │
│ ├─ simple_test()      → bbox_list, ret_iou                      │
│ │  └─ simple_test_pts()                                         │
│ │     ├─ pts_bbox_head.forward() → all_lane_preds[6]          │
│ │     ├─ sigmoid()              → [0, 1]                       │
│ │     ├─ threshold @ 0.43       → binary mask [3, 40000]       │
│ │     └─ IOU(pred, gt)          → [inter, union]               │
│ └─ 返回: data_sample with pred_instances_3d                     │
└──────────────────────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────────┐
│ Output DataSample {                                              │
│   pred_instances_3d: {                                           │
│     bboxes_3d: [N_pred, 7],                                      │
│     scores_3d: [N_pred],                                         │
│     labels_3d: [N_pred]                                          │
│   },                                                             │
│   metainfo: {...}                                               │
│ }                                                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔧 关键细节总结

| 组件 | 训练 | 推理 |
|------|------|------|
| **数据加载** | sweeps_num=1, test_mode=False (随机) | sweeps_num=1, test_mode=True (固定) |
| **标注** | 包含 GT 框、标签、地图 | 仅包含地图（可选） |
| **前向** | loss() 返回损失 Dict | predict() 返回 DataSample |
| **分割输出** | all_lane_preds[6 layers] × 3 channels | 仅用 [5]（最后一层）+ sigmoid + threshold |
| **IOU 计算** | 训练时不计算 | 推理时计算（gt_map 可用时） |
| **模型状态** | model.train() | model.eval() |
| **梯度** | 需要梯度 | no_grad() 上下文 |

---

## 💡 常见问题

### Q1: 为什么分割预测有 6 层输出？
**A:** 多层监督（Multi-scale Supervision）。Transformer decoder 有 6 层，每层都产生分割预测，可以：
- 提供更多的监督信号到浅层
- 学习多尺度的特征表示
- 改善梯度流

### Q2: 为什么车道线（lane）的权重最高（4.0）？
**A:** 车道线是最重要的分割任务：
- loss_dri: 2.0 (驾驶区域，相对容易，范围大)
- loss_lan: 4.0 (车道线，细线，难度大)
- loss_veh: 8.0 (车辆，稀疏，极难)

等等，文档中是 loss_lan=4.0, loss_veh=8.0，车辆最高，这是因为：
- 车辆物体稀疏，需要更多权重来学习
- 车道线虽然细，但覆盖面积大

### Q3: 为什么用 0.43 作为阈值？
**A:** 经验值。原始 PETR 在训练集上验证的最优阈值。可以通过 sweep_thresholds 动态调整以优化特定指标。

### Q4: 为什么需要 add_lidar2img？
**A:** v1.4 data_list 格式中 lidar2img 可能缺失，但分割头需要投影矩阵将 3D 点投影到图像平面。从 lidar2cam + cam2img 可以重建。

### Q5: gt_map_seg 和 gt_map 的区别？
**A:**
- **gt_map_seg** [3, 40000]: 展平的分割标签，用于 loss 计算
- **gt_map** [3, 200, 200]: 原始地图，用于可视化和 IOU 计算

