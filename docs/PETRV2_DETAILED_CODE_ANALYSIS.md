# 分割实现代码文件详解

## 📁 目录结构
```
petrv2/
├── __init__.py                          # 模块初始化
├── models/
│   ├── __init__.py
│   ├── petrv2_seg.py                    # ⭐ 检测器主类 (202行)
│   └── petrv2_head_seg.py               # ⭐ 分割头实现 (1178行)
├── transforms/
│   ├── __init__.py
│   └── transforms.py                    # ⭐ 数据管道 (263行)
└── datasets/
    ├── __init__.py
    └── nuscenes_map_dataset.py          # ⭐ 数据集加载 (...)
```

---

## 📄 文件 1: petrv2_seg.py (检测器主类)

### 目的
连接图像特征提取、分割头、损失计算的一体化检测器。负责：
- ✅ 多视图图像特征提取
- ✅ 分割 loss 计算（MMEngine 接口）
- ✅ 推理 forward pass
- ✅ lidar2img 矩阵处理

### 核心组件

#### 1️⃣ **IOU 计算工具**
```python
def IOU(intputs, targets, eps=1e-6):
    """计算交并比（IoU）"""
    intputs = intputs.bool()           # [3, 40000] → bool
    targets = targets.bool()           # [3, 40000] → bool
    inter = (intputs & targets).sum(-1)  # 逐通道交集 → [3]
    union = (intputs | targets).sum(-1)  # 逐通道并集 → [3]
    return inter.cpu(), union.cpu()    # 返回 CPU 张量
```

**说明：** 用于分割准确度评估，对比预测掩码与 GT 掩码

---

#### 2️⃣ **Petr3D_seg 类**

```python
@MODELS.register_module()
class Petr3D_seg(MVXTwoStageDetector):
    """分割专用检测器（继承 MMDetection3D v1.4 基类）"""
    
    def __init__(self, use_grid_mask=False, ...):
        super().__init__(...)
        self.grid_mask = GridMask(...)  # 随机网格掩码增强
        self.use_grid_mask = use_grid_mask
```

**继承关系：**
```
Petr3D_seg
    ↓
MVXTwoStageDetector (mmdet3d v1.4)
    ↓
BaseDetector (MMEngine)
```

---

### 核心方法详解

#### 📍 **extract_img_feat() - 图像特征提取**

```python
def extract_img_feat(self, img, img_metas):
    """提取多视图图像特征"""
    
    # Step 1: 处理输入形状
    if isinstance(img, list):
        img = torch.stack(img, dim=0)  # List → Tensor [B, N, C, H, W]
    
    B = img.size(0)
    
    # Step 2: 更新 metadata（输入分辨率）
    input_shape = img.shape[-2:]       # 获取 H, W
    for img_meta in img_metas:
        img_meta.update(input_shape=input_shape)
    
    # Step 3: 如果是 5D (batch × view × channel × H × W)，展平为 4D
    if img.dim() == 5:
        if img.size(0) == 1 and img.size(1) != 1:
            img.squeeze_()              # [1, N, C, H, W] → [N, C, H, W]
        else:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)  # [B, N, C, H, W] → [B*N, C, H, W]
    
    # Step 4: 应用 GridMask 增强（如果在 CUDA）
    if self.use_grid_mask and img.is_cuda:
        img = self.grid_mask(img)      # 随机删除像素块
    
    # Step 5: 通过骨干网络
    img_feats = self.img_backbone(img)  # [B*N, C, H, W] → 多级特征
    if isinstance(img_feats, dict):
        img_feats = list(img_feats.values())  # Dict → List
    
    # Step 6: 通过颈部网络
    if self.with_img_neck:
        img_feats = self.img_neck(img_feats)
    
    # Step 7: Reshape 回 5D (batch × view × channel × H × W)
    img_feats_reshaped = []
    for img_feat in img_feats:
        BN, C, H, W = img_feat.size()
        img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
    
    return img_feats_reshaped
```

**输入输出：**
- 📥 Input: img [B, N, 3, H, W] 或 List[ndarray]
- 📤 Output: img_feats_reshaped List[[B, N, C, H', W'], ...]

**关键点：**
- ✅ GridMask：随机删除 50% 像素块的正方形区域，增强泛化性
- ✅ 维度变换：5D → 4D → 5D（方便骨干网络处理）

---

#### 📍 **loss() - 训练损失计算（MMEngine 接口）**

```python
def loss(self, inputs=None, data_samples=None, **kwargs):
    """MMEngine 标准损失接口"""
    
    img = inputs['imgs']
    batch_img_metas = [ds.metainfo for ds in data_samples]
    
    # 提取 GT 框
    batch_gt_instances_3d = [ds.gt_instances_3d for ds in data_samples]
    gt_bboxes_3d = [gt.bboxes_3d for gt in batch_gt_instances_3d]
    gt_labels_3d = [gt.labels_3d for gt in batch_gt_instances_3d]
    
    # ⭐ 关键：从 data_samples 中提取分割 GT
    maps = [getattr(ds, 'gt_map_seg', None) for ds in data_samples]
    
    # 处理 lidar2img 矩阵（v1.4 兼容性）
    batch_img_metas = self.add_lidar2img(img, batch_img_metas)
    
    # 提取特征
    img_feats = self.extract_feat(img=img, img_metas=batch_img_metas)
    
    # 调用分割头的损失函数
    losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                        gt_labels_3d, maps, batch_img_metas)
    return losses_pts
```

**数据流：**
```
data_samples
    ↓ (ds.gt_map_seg)
gt_map_seg [B, 3, 40000] → maps List
    ↓
forward_pts_train() → losses Dict
```

**注意点：**
- ⚠️ `gt_map_seg` 必须由 `Pack3DDetAndSegInputs` transform 提前打包
- ⚠️ 如果缺失，返回 None，分割头需处理

---

#### 📍 **predict() - 推理（MMEngine 接口）**

```python
def predict(self, inputs=None, data_samples=None, **kwargs):
    """MMEngine 标准推理接口"""
    
    img = inputs['imgs']
    batch_img_metas = [ds.metainfo for ds in data_samples]
    
    # 获取 GT map（用于 IOU 计算，测试集可能无）
    maps = [getattr(ds, 'gt_map_seg', None) for ds in data_samples]
    
    # 处理 lidar2img
    batch_img_metas = self.add_lidar2img(img, batch_img_metas)
    
    # 调用 simple_test 推理
    results_list_3d = self.simple_test(batch_img_metas, maps, img, **kwargs)
    
    # 将结果打包回 data_samples
    for i, data_sample in enumerate(data_samples):
        pred = results_list_3d[i]['pts_bbox']
        
        # 创建预测实例
        pred_instances_3d = InstanceData()
        pred_instances_3d.bboxes_3d = pred['boxes_3d']
        pred_instances_3d.scores_3d = pred['scores_3d']
        pred_instances_3d.labels_3d = pred['labels_3d']
        
        data_sample.pred_instances_3d = pred_instances_3d
        data_sample.pred_instances = InstanceData()  # 空的 2D 检测
    
    return data_samples
```

**输出格式：**
```
data_sample.pred_instances_3d
    ├─ bboxes_3d [N, 7] (x, y, z, w, l, h, θ)
    ├─ scores_3d [N]
    └─ labels_3d [N]
```

---

#### 📍 **simple_test_pts() - 推理+IOU计算**

```python
def simple_test_pts(self, x, img_metas, gt_map, rescale=False):
    """推理单阶段"""
    
    # Step 1: 通过分割头
    outs = self.pts_bbox_head(x, img_metas)
    
    # Step 2: 获取检测框
    bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
    bbox_results = []
    for bboxes, scores, labels in bbox_list:
        bbox_results.append(
            dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels))
    
    # Step 3: ⭐ 分割后处理 + IOU 计算
    with torch.no_grad():
        lane_preds = outs['all_lane_preds'][5].squeeze(0)  # [B, 3, 40000]
        f_lane = lane_preds.sigmoid()                       # sigmoid → [0,1]
        
        # 硬阈值：0.43
        f_lane[f_lane >= 0.43] = 1
        f_lane[f_lane < 0.43] = 0                           # [B, 3, 40000]
        
        f_lane = f_lane.view(3, -1)                         # 移除 batch 维
        
        # 计算 IOU（测试用）
        ret_ious = None
        if gt_map is not None and gt_map[0] is not None:
            curr_gt_map = gt_map[0].view(3, -1)  # [3, 40000]
            inter, union = IOU(f_lane, curr_gt_map)
            ret_ious = [inter, union]
    
    return bbox_results, ret_ious
```

**核心细节：**
- ✅ `outs['all_lane_preds'][5]`：取最后一层 decoder 输出
- ✅ 硬阈值 0.43：来自原始 PETR 训练经验
- ✅ IOU 计算：仅在有 GT 时进行（评估模式）

---

#### 📍 **add_lidar2img() - 矩阵兼容性处理**

```python
def add_lidar2img(self, img, batch_input_metas):
    """处理 v1.4 中 lidar2img 缺失的问题"""
    
    for meta in batch_input_metas:
        num_views = len(meta.get('cam2img', []))
        
        # 检查 lidar2img 是否存在且长度正确
        if 'lidar2img' in meta:
            lidar2img = meta.get('lidar2img', [])
            if isinstance(lidar2img, (list, tuple)) and len(lidar2img) == num_views:
                continue  # 已存在且正确，跳过
        
        # 需要重建 lidar2img 矩阵
        lidar2img_rts = []
        for i in range(num_views):
            # 从 extrinsics (lidar→cam) + intrinsics (cam 内参) 重建
            lidar2cam_rt = torch.tensor(meta['lidar2cam'][i]).double()  # [4, 4]
            intrinsic = torch.tensor(meta['cam2img'][i]).double()       # [3, 3]
            
            # 构造齐次投影矩阵 [3, 4]
            viewpad = torch.eye(4).double()
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            
            # lidar2img = intrinsics @ lidar2cam
            lidar2img_rt = (viewpad @ lidar2cam_rt)  # [4, 4] @ [4, 4] = [4, 4]
            lidar2img_rts.append(lidar2img_rt)
        
        meta['lidar2img'] = lidar2img_rts
        
        # 处理 img_shape（v1.4 兼容性）
        if 'img_shape' in meta:
            img_shape = meta['img_shape'][:3] if isinstance(meta['img_shape'], tuple) else meta['img_shape']
            meta['img_shape'] = [img_shape] * len(img[0])
    
    return batch_input_metas
```

**数学推导：**
```
lidar 坐标 X_lidar
    ↓ (lidar2cam_rt)
camera 坐标 X_cam = R_l2c @ X_lidar + t_l2c
    ↓ (cam2img / intrinsic)
像素坐标 u = K @ X_cam / Z
    ↓ (一步完成)
u = K @ R_l2c @ X_lidar
```

**为什么需要？**
- ❌ 原始 PETR 有 lidar2img，但 v1.4 data_list 格式中可能缺失
- ✅ 从 lidar2cam + cam2img 可以重建，确保兼容性

---

### 整体流程图

```
训练：
input ['imgs', 'gt_instances_3d', 'gt_map_seg']
    ↓
loss() → add_lidar2img() → extract_feat() 
    ↓
forward_pts_train() → pts_bbox_head.loss()
    ↓
losses {'loss_dri', 'loss_lan', 'loss_veh', 'loss_cls', ...}

推理：
input ['imgs']
    ↓
predict() → add_lidar2img() → simple_test()
    ↓
simple_test_pts() → {boxes_3d, scores_3d, labels_3d, ret_iou}
    ↓
pred_instances_3d
```

---

---

## 📄 文件 2: petrv2_head_seg.py (分割头，1178行)

### 目的
实现 3D 检测 + 分割的 Transformer 解码器头。核心：
- ✅ 多头 Transformer decoder
- ✅ 3 分支分割头（驾驶/车道/车辆）
- ✅ 多层监督（6 层 decoder 每层都有输出）
- ✅ 权重兼容性处理

### 核心组件

#### 1️⃣ **位置编码函数**

```python
def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    """3D 位置 → 正弦位置编码"""
    # pos: [*, 3]  (x, y, z)
    # 输出: [*, 128*3] = [*, 384]
    
    scale = 2 * math.pi
    pos = pos * scale
    
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    
    # 对每个维度应用正弦/余弦编码
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    
    # sin(x)@偶数, cos(x)@奇数
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    
    # 级联 [pos_y (128), pos_x (128), pos_z (128)]
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)  # [*, 384]
    return posemb
```

**对比：**
- `pos2posemb3d`: 用于 3D 检测（融合 x, y, z）
- `pos2posemb2d`: 用于 2D 分割（仅 x, y）

---

#### 2️⃣ **DecoderBlock - 分割解码块**

```python
class DecoderBlock(torch.nn.Module):
    """上采样 + 卷积 + 残差连接"""
    
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()
        
        dim = out_channels // factor  # 通常 factor=2
        
        # 上采样分支
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
        )
        
        # 残差跳跃连接
        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        # x: [B, C_in, H, W]
        # skip: [B, skip_dim, H, W]
        
        x = self.conv(x)          # [B, C_in, H, W] → [B, C_out, 2H, 2W]
        
        if self.up is not None:
            up = self.up(skip)              # [B, C_skip, H, W] → [B, C_out, H, W]
            up = F.interpolate(up, x.shape[-2:])  # 插值到输出大小
            x = x + up  # 残差加
        
        return self.relu(x)
```

**流程：**
```
Input: x [B, 256, 25, 25]
    ↓
Upsample(×2) → [B, 256, 50, 50]
    ↓
Conv2d(256→128→256) → [B, 256, 50, 50]
    ↓ (如果 residual=True)
Skip 投影 + 插值 → [B, 256, 50, 50]
    ↓ (相加)
Output: [B, 256, 50, 50]
```

---

#### 3️⃣ **Decoder - 多层堆叠**

```python
class Decoder(nn.Module):
    """多个 DecoderBlock 的堆叠"""
    
    def __init__(self, dim, blocks, out_dim, residual=True, factor=2):
        # dim: 输入通道 (256)
        # blocks: [256, 256, 128]  → 3 层
        # out_dim: 1 (最终输出通道)
        
        super().__init__()
        
        layers = []
        channels = dim
        
        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)
            channels = out_channels
        
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels  # 128
        
        # 最后的 logits 层
        self.to_logits = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, out_dim, 1)  # 输出 1 通道
        )
    
    def forward(self, x):
        # x: [B, 256, 25, 25]
        y = x
        for layer in self.layers:
            y = layer(y, x)  # 逐层上采样
        # y: [B, 128, 200, 200]
        
        y = self.to_logits(y)  # → [B, 1, 200, 200]
        return y
```

**尺寸变化：**
```
[B, 256, 25, 25]
    ↓ Block1 (256→256)
[B, 256, 50, 50]
    ↓ Block2 (256→256)
[B, 256, 100, 100]
    ↓ Block3 (256→128)
[B, 128, 200, 200]
    ↓ to_logits
[B, 1, 200, 200]  (logits)
```

---

#### 4️⃣ **PETRHead_seg 类（核心分割头）**

```python
@MODELS.register_module()
class PETRHead_seg(AnchorFreeHead):
    """分割 + 检测 Transformer 头"""
    
    def __init__(self, 
                 num_classes=10,      # 3D 检测类别
                 in_channels=256,
                 num_query=900,       # 3D 检测查询
                 num_lane=625,        # 分割查询 (25×25)
                 blocks=[256,256,128],
                 transformer=None,    # 3D 检测 decoder
                 transformer_lane=None,  # 分割 decoder
                 loss_dri=...,        # 驾驶损失
                 loss_lan=...,        # 车道损失
                 loss_veh=...,        # 车辆损失
                 **kwargs):
        
        super().__init__(num_classes, in_channels, **kwargs)
        
        # 三个独立的分割分支
        lane_branch_dri = Decoder(self.embed_dims, blocks, 1)
        lane_branch_lan = Decoder(self.embed_dims, blocks, 1)
        lane_branch_vie = Decoder(self.embed_dims, blocks, 1)
        
        # 为每个 decoder 层复制分支（6 层）
        self.lane_branches_dri = nn.ModuleList(
            [lane_branch_dri for _ in range(self.num_pred)])  # num_pred=6
        self.lane_branches_lan = nn.ModuleList(
            [lane_branch_lan for _ in range(self.num_pred)])
        self.lane_branches_vie = nn.ModuleList(
            [lane_branch_vie for _ in range(self.num_pred)])
        
        # 损失函数
        self.loss_dri = build_loss(loss_dri)  # FocalLoss
        self.loss_lan = build_loss(loss_lan)
        self.loss_veh = build_loss(loss_veh)
        
        # 分割用 transformer decoder
        self.transformer_lane = build_transformer(transformer_lane)
```

**架构：**
```
特征 [B, N, 256, H, W] (多视图)
    ↓
Transformer Lane Decoder (6 层)
    ↓ (每层输出)
Feat Maps [B, 256, 25, 25]
    ↓ (并行3个分支)
┌───────────────────────────┐
│ lane_branches_dri[i]      │  → [B, 1, 200, 200]
│ lane_branches_lan[i]      │  → [B, 1, 200, 200]
│ lane_branches_vie[i]      │  → [B, 1, 200, 200]
└───────────────────────────┘
    ↓ (堆叠)
all_lane_preds[i] = [B, 3, 200, 200]  ← (第 i 层输出)
```

---

#### 5️⃣ **Forward 方法（多层监督）**

伪代码（简化）：
```python
def forward(self, mlvl_feats, img_metas):
    # mlvl_feats: [L, B, N, C, H, W] (多层多视图特征)
    
    # 检测 transformer decoder
    det_outputs = self.transformer(...)  # 6 层输出
    
    # 分割 transformer decoder
    seg_outputs = self.transformer_lane(...)  # 6 层输出
    
    all_lane_preds = []
    
    for i in range(6):  # 每一 decoder 层
        feat_maps = seg_outputs[i]  # [B, 256, 25, 25]
        
        # 三个分支并行处理
        lane_pred_dri = self.lane_branches_dri[i](feat_maps)  # [B, 1, 200, 200]
        lane_pred_lan = self.lane_branches_lan[i](feat_maps)
        lane_pred_vie = self.lane_branches_vie[i](feat_maps)
        
        # 通道堆叠：(驾驶, 车道, 车辆)
        lane_pred = torch.cat(
            [lane_pred_dri, lane_pred_lan, lane_pred_vie], 
            dim=1
        )  # [B, 3, 200, 200]
        
        all_lane_preds.append(lane_pred)
    
    return {'all_lane_preds': all_lane_preds}  # List[6] of [B, 3, 200, 200]
```

---

#### 6️⃣ **loss_lane_single() 损失计算**

```python
def loss_lane_single(self, lane_preds, gt_lane_list, labels):
    """
    Args:
        lane_preds: [6, B, 3, 40000] (6 层decoder每层的预测)
        gt_lane_list: List[3]，每个元素 [40000] (驾驶/车道/车辆)
        labels: [B] (样本权重)
    """
    loss = 0
    
    # 遍历 6 层，每层都有损失
    for i in range(len(lane_preds)):
        # 驾驶通道：lane_preds[i][:, 0, :]
        loss_dri = self.loss_dri(
            lane_preds[i][:, 0, :],      # 预测 logits [B, 40000]
            gt_lane_list[0],              # GT [40000]
            labels
        )
        
        # 车道通道
        loss_lan = self.loss_lan(
            lane_preds[i][:, 1, :],
            gt_lane_list[1],
            labels
        )
        
        # 车辆通道
        loss_veh = self.loss_veh(
            lane_preds[i][:, 2, :],
            gt_lane_list[2],
            labels
        )
        
        loss = loss + loss_dri + loss_lan + loss_veh
    
    return loss
```

**多层监督的好处：**
- ✅ 早期层有直接的梯度流
- ✅ 浅层特征也学到分割信息
- ✅ 多尺度的监督信号

---

### 完整流程图

```
Multi-view Images [B, N, 3, H, W]
    ↓
Image Backbone + Neck
    ↓
Multi-level Features [B, N, 256, 25, 25], [B, N, 256, 50, 50], ...
    ↓
Transformer Lane Decoder (6 layers)
    ├─ Layer 0: Feature [B, 256, 25, 25] → (dri/lan/vie branches) → Pred [B, 3, 200, 200]
    ├─ Layer 1: Feature [B, 256, 25, 25] → (dri/lan/vie branches) → Pred [B, 3, 200, 200]
    ├─ ...
    └─ Layer 5: Feature [B, 256, 25, 25] → (dri/lan/vie branches) → Pred [B, 3, 200, 200]
    ↓
all_lane_preds = List[6] of [B, 3, 200, 200]
    ↓ (GT)
GT: [3, 40000] (dri/lan/vie, 扁平)
    ↓
loss_lane_single()
    ├─ Layer 0 loss: FocalLoss(pred[0], gt) per channel
    ├─ Layer 1 loss: FocalLoss(pred[1], gt) per channel
    └─ ...
    ↓
Total Loss = Σ(6层的分割loss + 3通道)
```

---

---

## 📄 文件 3: transforms.py (数据管道)

### 目的
实现数据加载、预处理、分割 label 打包的所有 Transform。

#### 1️⃣ **LoadMultiViewImageFromMultiSweepsFiles - 多帧多视图加载**

```python
@TRANSFORMS.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(BaseTransform):
    """加载多帧、多视图图像"""
    
    def __init__(self,
                 sweeps_num=1,           # 额外扫描帧数
                 to_float32=False,
                 pad_empty_sweeps=True,  # 如果无扫描数据，重复当前帧
                 sweep_range=(3, 27),    # 扫描帧选择范围 (ms)
                 test_mode=True,         # 测试模式下选择中间帧
                 sensors=(...)):         # 相机列表 (6 个)
        self.sweeps_num = sweeps_num
        # ... 保存其他参数 ...
    
    def transform(self, results: dict) -> dict:
        """多帧加载逻辑"""
        
        # Step 1: 初始化列表
        sweep_imgs_list = list(results['img'])
        base_lidar2img = list(results.get('lidar2img', []))
        
        # Step 2: 计算相对时间戳 (lidar_ts - cam_ts)
        lidar_timestamp = float(results.get('timestamp', 0.0))
        img_timestamp = results.get('img_timestamp')
        timestamp_list = [lidar_timestamp - t for t in img_timestamp]
        
        # Step 3: 处理 sweep (历史帧)
        sweeps = results.get('sweeps', None)
        
        if sweeps is None or len(sweeps) == 0:
            # 无 sweep 数据，填充当前帧
            if self.pad_empty_sweeps:
                for _ in range(self.sweeps_num):
                    sweep_imgs_list.extend(results['img'])
                    mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                    timestamp_list.extend([t + mean_time for t in timestamp_list[:len(results['img'])]])
                    base_lidar2img.extend(base_lidar2img)
        else:
            # 选择 sweep 帧
            # ...实现逻辑...
        
        # Step 4: 整合结果
        results['img'] = sweep_imgs_list
        results['timestamp'] = timestamp_list
        results['lidar2img'] = base_lidar2img
        
        return results
```

---

#### 2️⃣ **LoadMapsFromFiles_flattenf200f3 - 地图加载**

```python
@TRANSFORMS.register_module()
class LoadMapsFromFiles_flattenf200f3(BaseTransform):
    """加载并处理分割地图"""
    
    def transform(self, results: dict) -> dict:
        # Step 1: 加载 npz 文件
        map_filename = results['map_filename']
        maps_data = np.load(map_filename)
        map_mask = maps_data['arr_0'].astype(np.float32)  # [200, 200, 3]
        
        # Step 2: 转置为通道优先
        gt_map = map_mask.transpose((2, 0, 1))  # [3, 200, 200]
        results['gt_map'] = gt_map
        
        # Step 3: 二值化 + 扁平
        maps = gt_map.reshape(3, 200 * 200)  # [3, 40000]
        maps[maps >= 0.5] = 1
        maps[maps < 0.5] = 0
        
        results['gt_map_seg'] = maps  # ⭐ 分割专用键
        
        return results
```

---

#### 3️⃣ **Pack3DDetAndSegInputs - 最终打包**

```python
@TRANSFORMS.register_module()
class Pack3DDetAndSegInputs(Pack3DDetInputs):
    """打包检测 + 分割输入"""
    
    def pack_single_results(self, results: dict) -> dict:
        # 调用父类打包（处理标准的检测数据）
        packed = super().pack_single_results(results)
        data_sample = packed['data_samples']
        
        # ⭐ 关键：添加分割数据到 data_sample
        if 'maps' in self.full_keys and 'maps' in results:
            maps = results['maps']  # [3, 40000]
            if not isinstance(maps, torch.Tensor):
                maps = torch.as_tensor(maps, dtype=torch.float32)
            data_sample.gt_map_seg = maps  # ← 模型 loss() 会读取这个
        
        if 'gt_map' in self.full_keys and 'gt_map' in results:
            gt_map = results['gt_map']  # [3, 200, 200]
            if not isinstance(gt_map, torch.Tensor):
                gt_map = torch.as_tensor(gt_map, dtype=torch.float32)
            data_sample.gt_map = gt_map  # ← 用于可视化
        
        return packed
```

---

---

## 📄 文件 4 & 5: 数据集和配置

**nuscenes_map_dataset.py:**
- Token 匹配：sample_token → map_filename
- 关联检测和分割注解

**configs/petrv2_bevseg.py:**
- 损失权重：loss_lan > loss_veh > loss_dri
- 管道链：LoadMultiSweeps → LoadMaps → Pack

---

## 🎓 核心数据流总结

```
训练数据流：
1. Dataset: 检测数据 + 地图文件路径 (token 匹配)
   ↓
2. LoadMultiViewImageFromMultiSweepsFiles: 扩展到 30 帧
   ↓
3. LoadMapsFromFiles_flattenf200f3: 加载 + 二值化 → [3, 40000]
   ↓
4. Pack3DDetAndSegInputs: DataSample.gt_map_seg
   ↓
5. Petr3D_seg.loss(): 提取 gt_map_seg
   ↓
6. PETRHead_seg.loss_lane_single(): 6 层 × 3 通道 FocalLoss
   ↓
7. 反向传播

推理数据流：
1. Dataset: 仅检测数据（无 gt_map_seg）
   ↓
2-4. 同上（gt_map_seg = None）
   ↓
5. Petr3D_seg.predict():
   ↓
6. simple_test_pts(): sigmoid → threshold 0.43
   ↓
7. 输出 pred_instances_3d
```

---

所有文件详解完毕！😊