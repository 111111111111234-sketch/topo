# PETRV2 Patch Mapping 方法文档（v27 技术详版）

简版见同目录下的 [PETRV2_PATCH_METHOD.md](./PETRV2_PATCH_METHOD.md)。

---

## 一、主要实现方法

基于 PETR 框架 **Backbone → Neck → Decoder → Head** 实现前向单目 RGB **PV-to-BEV 端到端 Mapping**。

前视 RGB 图像及 **1 帧历史** 为输入，经多尺度特征提取、3D 位置编码（3DPE）、Pose-aware Temporal Fusion 和深层 Patch Decoder，预测局部 topdown **semantic patch**（10 类）与 **freespace patch**（2 类）；再通过 **global stitching**（soft voting）构建场景级地图并服务后续导航评估。

### 模型总览

| 组件 | 实现 | 关键参数 |
|------|------|---------|
| Backbone | VoVNetCP (V-99-eSE) | `frozen_stages=-1`（全部可训练），backbone LR ×0.1 |
| Neck | CPFPN | `in_channels=[768, 1024]` → `out_channels=256`，2 个输出尺度 |
| Head | PETRHead_patch | `embed_dims=256`，多尺度融合 + 3DPE + temporal gate + deep decoder |
| 整体 | Petr3D_patch | 继承 MVXTwoStageDetector，仅使用图像分支 |

### Patch 定义

| 参数 | 值 | 含义 |
|------|---|------|
| `patch_size` | (135, 150) | 输出 patch 像素尺寸 (H×W) |
| `patch_width_m` | 3.0 m | 物理宽度 |
| `patch_z_near` | 0.3 m | 最近距离 |
| `patch_z_far` | 3.0 m | 最远距离 |

配置文件：`mmdet3d_petr/projects/PETRV2/configs/petrv2_mp3d_patch_v27.py`。

---

## 二、模型结构细节

### 1. 10 类 Semantic + 2 类 Freespace

**语义 remap**：`_semantic_remap_v22`（长度 14 的映射表），将 MPCAT40 原始前 14 类映射为 10 类：

| ID | 类名 | 像素占比 | 类权重（CE） |
|----|------|---------|--------|
| 0 | Floor | ~51% | 1.0 |
| 1 | storage | ~2.7% | 2.0 |
| 2 | bed | ~1.1% | 2.0 |
| 3 | sofa | ~1.5% | 2.0 |
| 4 | chair | ~2.4% | 2.0 |
| 5 | surface | ~4.2% | 1.5 |
| 6 | sink | ~0.25% | 2.5 |
| 7 | plant | ~0.6% | 2.5 |
| 8 | cushion | ~0.5% | 2.5 |
| 9 | other | ~36% | 1.0 |

**Freespace**：2 类二分类（occupied=0 / free=1），**BCE + Dice**，BCE 像素类权重 `[3.0, 1.0]`（非 free 放大，缓解 free 过多）。

### 2. Mask 设置

#### GT 有效区域（数据：`GenerateTopdownPatchFromPose`）

- **`warp_boundary`**：位姿下全局图投影到 patch 的覆盖区（常为梯形）。
- **`annotation_mask`**：H5 中 `mask > 0` 裁到 patch。

**语义 valid** = `warp_boundary` **∧** `annotation_mask`；无效处语义 GT = `255`。

**Freespace valid（v27）**：`freespace_use_valid_mask=False` → 仅 **`warp_boundary`**，不与 annotation 求交。

#### Loss mask（训练：`PETRHead_patch._semantic_loss`）

```
loss_mask = (gt_patch_semantic != 255) & gt_patch_semantic_valid_mask
```

Freespace 同理：`!= 255` 且 `gt_patch_freespace_valid_mask`。

#### Observed mask（全局融合）

`DiffGlobalMapFusion` / `vis_global_stitch.py`：各帧 `max(softmax(probs))` 作为置信度累加到 `count_map`；**`observed = count > 0.01`**（数值稳定阈值）。

#### GridMask

`Petr3D_patch.extract_img_feat`：`prob=0.7, ratio=0.5, mode=1`，仅训练 + CUDA；不改 GT 与 mask。

### 3. 多尺度特征融合

取 FPN level 0 与 level 1；level 1 双线性插值到 level 0 尺寸；`2×256` 通道拼接 → `1×1 Conv + BN + ReLU` → `256`。在 **2D 空间** 上完成，非 temporal 维拼接。

### 4. 3DPE（`use_camera_3dpe=True`）

| 参数 | 值 |
|------|---|
| `depth_num` | 64 |
| `depth_start` | 0.3 m |
| `position_range` | `[-5, -3, 0, 5, 5, 5]`（最远约 5 m） |

流程：每像素在 D 个深度上生成 3D 坐标 → `cam→lidar` → 归一化到 `[0,1]` → inverse sigmoid → 展平 `3×D=192` 通道 → `position_encoder`（两层 `1×1 Conv` + ReLU）→ `embed_dims=256`；**`x = x + pe_3d`**。

### 5. Pose Warp + Temporal Fusion

**Pose warp**：相对位姿 `[dx, dz, dyaw]` → `affine_grid + grid_sample` 在特征图平面对齐历史帧；`pose_warp_resolution=[0.2, 0.2]`（米/像素尺度）。

**Temporal gate**：每帧 GAP 得 `(B,T,C)`；可选 `pose_encoder(3→256)` 加到 token；`Linear→ReLU→Linear` 得标量再 **softmax(T)**；**`fused = Σ_t w_t · feat_t`**。

### 6. Deep Decoder + 双头（`use_deep_decoder=True`）

**`shared_decoder`**：`ResBlock×2 → ChannelAttention → ResBlock(256→128)`，`mid_channels=128`。

分裂后 **两路独立权重**：

- **语义**：`_UpsampleBlock(×2)` ×2 → `interpolate` 到 (135,150) → `ChannelAttention` → `ResBlock + Conv1×1` → 10 类。
- **Freespace**：同构 → 2 类。

`_UpsampleBlock`：`ConvTranspose2d(stride=2) + BN + ReLU + ResBlock`。

**深度监督**：v27 `deep_supervision=False`；若开启则有 `aux_head` 与 `deep_supervision_weight=0.4`。

---

## 三、损失函数

### 语义

总乘子 `loss_semantic_weight=2.0`：

```
L_sem = 2.0 × (0.5×CE + 1.0×Focal + 2.0×Lovász)
```

- CE：类权重见上表，`ignore_index=255`。
- Focal：`gamma=2.0`，在 CE map 上构造。
- Lovász：`classes='present'`。

### Freespace

总乘子 `loss_freespace_weight=1.0`：

```
L_free = 1.0 × (1.0×BCE + 2.0×Dice)
```

二分类 logits：`logits[:,1] - logits[:,0]` 作 BCE 输入。

### 其他

- **一致性**：`consistency_weight=0.0`（v27 未启用）。
- **深度监督**：`deep_supervision=False`。

实现：`petrv2/models/petrv2_head_patch.py`。

---

## 四、数据增强

| 阶段 | 内容 |
|------|------|
| Pipeline | `PhotoMetricDistortionMultiViewImage`；`RandomFlipEgoPatch` flip_ratio=0.5（图+GT+cam2img）；`RandomRotateScaleEgoPatch` prob=0.6，±20°，scale 0.75–1.25，**仅 GT patch** |
| 前向 | GridMask 见上 |

---

## 五、训练策略

| 项 | 配置 |
|----|------|
| 优化器 | AdamW，`lr=0.0012`，`weight_decay=0.01` |
| Backbone | `lr_mult=0.1` |
| AMP | `AmpOptimWrapper`，`dtype=bfloat16` |
| 梯度裁剪 | `max_norm=35`，`norm_type=2` |
| LR | 前 500 iter `LinearLR`（start_factor=1/3）→ `CosineAnnealingLR`（T_max=36 epoch，`eta_min=1e-5`） |
| Batch | 每 GPU 64；`max_epochs=36`，`val_interval=4` |
| DDP | `static_graph=True` |

---

## 六、全局地图融合与后处理

### Soft voting

1. `softmax(logits)` → `probs`；`conf = max(probs)`。
2. 水平 flip 与 patch 坐标约定对齐（与 stitch 脚本一致）。
3. `grid_sample` 将 `conf * probs` 与 `conf` 累加到全局 `score` / `count`。
4. `mean = score / count`；解码 `argmax(mean)`。

### 后处理（`vis_global_stitch.py` 等）

- **rare_boost**：非主导类 logit 偏置。
- **logit_adjustment**：Menon 式 `log(p) - τ·log(π)` 等。
- **confidence_threshold**：低置信标 255。

### `eval_mask_mode`

- **intersection**：评 **GT 有效 ∩ 已观测**。
- **observed**：评全部 GT 有效，未覆盖惩罚更强。

可微融合参考：`petrv2/models/global_map_fusion.py`（`observed = count.squeeze(0) > 0.01`）。

---

## 七、导航评估

- 由 pred freespace 建二值可导航图；**GT / pred 分别 A\***（8 邻域，对角代价 √2）。
- **GT / pred** 均按 **`agent_radius_m`** 腐蚀后再规划（`eval_nav_planning.py`）。
- 指标：**Success Rate、SPL、Collision Rate、Path Ratio**。
- **`safe_nav`**（`eval_mapping_then_navigation.py`）：`free_prob` 阈值、最小观测次数、小连通域剔除等。

---

## 八、监督数据与清洗

全局 GT → 按位姿裁 patch，与 stitch 几何一致。

清洗：`clean_mp3d_patch_infos.py` → `mp3d_patch_*_clean_v2.json`；可调 **`mask_radius_px`、`min_sem_ratio`、`min_free_ratio`** 等，剔除相机邻域标注过少或 freespace 异常样本。

---

## 相关代码路径（速查）

| 主题 | 路径 |
|------|------|
| v27 配置 | `projects/PETRV2/configs/petrv2_mp3d_patch_v27.py` |
| Patch 与 mask | `petrv2/transforms/mp3d_transforms.py` |
| Head / loss / decoder | `petrv2/models/petrv2_head_patch.py` |
| GridMask 入口 | `petrv2/models/petrv2_patch.py` |
| 可微全局融合 | `petrv2/models/global_map_fusion.py` |
| Stitch / 后处理 | `tools/vis_global_stitch.py` |
| 映射后导航 | `tools/eval_mapping_then_navigation.py` |
| A* 与指标 | `tools/eval_nav_planning.py` |
| 清洗标注列表 | `tools/clean_mp3d_patch_infos.py` |
