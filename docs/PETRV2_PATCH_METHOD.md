# PETRV2 Patch Mapping 方法简述（v27）

技术详版（表格、损失权重、训练与评估细节）：[PETRV2_PATCH_METHOD_DETAIL.md](./PETRV2_PATCH_METHOD_DETAIL.md)

## 一、主要实现方法

基于 PETR 的基本框架：**Backbone → Neck → Decoder → Head**，实现前向单目 RGB **PV-to-BEV 端到端 Mapping**。

前视 RGB 与 **1 帧历史** 为输入，经多尺度特征、**3DPE**、**pose-aware temporal fusion** 与 **深层 patch decoder**，预测局部 topdown **semantic patch**（10 类）与 **freespace patch**（2 类）；再经 **global stitching**（soft voting，按位姿单应拼接）得到场景级地图，供后续导航评估。

实现上：Backbone 为 **VoVNetCP (V-99-eSE)**，Neck 为 **CPFPN**（两尺度 → 256 维），Patch 输出约 **135×150** 像素，对应宽约 **3 m**、深度约 **0.3–3.0 m** 的俯视区域。

---

## 二、具体优化措施

### 1. 10 类 semantic、2 类 freespace

将原始 MP3D / MPCAT40 语义空间压到 **10 类**（Floor、storage、bed、sofa、chair、surface、sink、plant、cushion、other）。

- **作用**：减轻稀疏与长尾，patch 更可学，也更贴近导航关心的物体与结构。
- **原因**：原始类多、不少与导航弱相关；压缩后监督更稳，全局拼接后的语义也更好解释。

Freespace 为 **occupied / free** 二分类，训练上用 **BCE + Dice**（并配合类权重缓解 free 过多）；语义侧为 **CE + Focal + Lovász** 等组合（v27 配置见 `petrv2_mp3d_patch_v27.py`）。

### 2. Mask 设置

- **GT valid（语义）**：投影覆盖（`warp_boundary`）与 H5 标注有效区（`annotation_mask`，mask>0）的 **交集**；无效处语义 GT 为 **255**。
- **GT valid（freespace，v27）**：仅按投影覆盖（`freespace_use_valid_mask=False`），**不要求**与语义 annotation 一致。
- **Loss mask**：在 valid 上再排除 `ignore_index`（如 255），只在参与监督的像素上算 loss。
- **Observed mask**：全局融合时对各帧置信度加权累加得到 `count_map`，**count>0**（实现里常用小阈值如 0.01）表示该格点曾被观测。
- **GridMask**：仅作用于训练时 RGB，**不改** GT 与 valid mask。

### 3. Multi-scale feature fusion

高分辨率层偏细节与边界，低分辨率层偏上下文与布局；将两尺度对齐后 **拼接再 1×1 卷积压回同一通道**，再进 decoder，便于同时利用细节与上下文。

### 4. 3DPE

在前视特征上显式注入 **相机几何下的 3D 位置编码**（多深度 bin + 内外参），使映射显式依赖 **RGB + geometry → topdown patch**，增强几何先验。

### 5. Pose warp、temporal fusion

由相对位姿 **[dx, dz, dyaw]** 在特征平面上 **warp** 历史帧对齐当前帧，再经 **temporal gate**（对帧 token 加权 softmax）融合，减轻时序错位。

### 6. Semantic / freespace patch head

**共享 trunk（shared decoder）** 后 **分裂为两路独立头**：语义与可通行共用底层表征、各走独立上采样与预测，使表征同时服务语义与导航相关的可通行结构。

### 7. Global map

多帧 **soft vote**（按 max-softmax 置信加权、归一化）融合，兼顾去噪与覆盖；**observed_mask** 标出「被看过」区域。流程上：**先学好局部 patch → 已知位姿显式几何拼接 → 再在拼接图上做导航类评估**。全局图大、覆盖不完全时，仍依赖局部预测质量而非端到端背整张 global memory。

### 8. 监督数据

**GT 全局图 → 按位姿几何裁剪 → 局部 patch 监督**，与测试时 stitch 的几何定义一致。训练集可使用 **`clean_mp3d_patch_infos.py`** 等清洗得到 `*_clean_v2.json`，按相机邻域过滤标注过少或 freespace 异常的样本。

---

## 三、后续方向

- **Patch → global stitch → navigation** 闭环，用于规划与指标迭代。
- **数据侧**：接入或融合更高质量 / 多源数据集，继续用清洗与采样策略控分布。
- **3D 感知**：在现有 3DPE 与俯视监督基础上，探索更强的 **3D 结构或深度一致性**（如多视图几何、显式高度/占用等），与当前 BEV patch 范式衔接。
