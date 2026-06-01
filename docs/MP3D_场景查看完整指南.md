# 🎯 MP3D 场景查看完整指南

## 问题：为什么 glTF Viewer 看不了？

### 常见原因：
1. **扩展没装好** - 版本冲突或不兼容
2. **文件太大** - 21MB GLB 文件加载超时
3. **VSCode 版本老** - 需要 1.50+ 版本
4. **WebGL 不支持** - 某些系统/GPU 驱动问题

---

## ✅ 推荐方案（4 种）

### 方案 1️⃣: 在线预览（最简单）⭐

**优点：** 无需安装，即时查看  
**缺点：** 需要网络，上传可能慢

#### 选项 A - Three.js Editor（推荐）
```
网址: https://threejs.org/editor/
步骤:
  1. 打开网站
  2. File → Import → 选择 17DRP5sb8fy.glb
  3. 即时 3D 预览
操作:
  - 左键拖拽：旋转
  - 滚轮：缩放
  - 右键拖拽：平移
```

#### 选项 B - Babylon.js Sandbox
```
网址: https://sandbox.babylonjs.com
步骤: 拖拽 GLB 文件到网页
```

#### 选项 C - Khronos glTF Viewer（专业）
```
网址: https://github.khronos.org/glTF-Sample-Viewer-Release/
特点: 显示详细的 glTF 信息和材质
```

---

### 方案 2️⃣: 本地 Python 脚本（推荐用于数据分析）

#### 安装依赖
```bash
pip install trimesh matplotlib
```

#### 运行查看脚本
```bash
python /workspace/tangyx7@xiaopeng.com/view_glb_scene.py
```

#### 脚本功能
- ✓ 显示模型的详细信息（顶点数、面数等）
- ✓ 显示空间范围（边界框）
- ✓ 显示物理属性（体积、表面积、中心）
- ✓ 交互式 3D 预览（支持鼠标旋转、缩放）

#### 输出示例
```
📊 GLB 文件分析: 17DRP5sb8fy.glb
==================================================
📐 几何信息:
   ├─ 顶点数: 1,234,567
   ├─ 面数: 617,283
   └─ 边数: 1,851,549

🎯 空间范围:
   ├─ X: [-10.50, 45.23]
   ├─ Y: [0.00, 3.50]
   └─ Z: [-20.10, 25.67]

⚙️  物理属性:
   ├─ 体积: 1234.56 m³
   ├─ 表面积: 5678.90 m²
   └─ 中心: (5.23, 1.75, 2.45)
```

---

### 方案 3️⃣: VSCode 扩展（如果非要用）

#### 推荐扩展列表
| 扩展名 | 出版商 | 市场ID |
|-------|-------|---------|
| glTF Tools | khronosgroup | khronosgroup.gltf-tools |
| Babylon.js | babylon | babylonjs.babylonjs-viewer |
| Three.js | WXY | wx.threejs-editor |

#### 安装步骤
1. `Ctrl+Shift+X` 打开扩展市场
2. 搜索扩展名
3. 点击 Install
4. 重载 VSCode (Ctrl+Shift+P → Reload Window)

#### 使用方式
1. 右键 `.glb` 文件
2. 选择 "Open Preview" 或 "View"
3. 预览窗口打开

**注意：** 如果还是看不了，可能是版本兼容问题，改用其他方案

---

### 方案 4️⃣: 命令行工具

#### 使用 gltf-transform（需要 Node.js）
```bash
npm install -g @gltf-transform/cli
cd /workspace/tangyx7@xiaopeng.com/Semantic-MapNet111/data/mp3d/17DRP5sb8fy/
gltf-transform inspect 17DRP5sb8fy.glb
```

输出示例：
```
glTF Inspector
├─ Asset Version: 2.0
├─ Generator: Blender
├─ Meshes: 45
├─ Nodes: 200
├─ Materials: 120
├─ Textures: 256
└─ Animations: 12
```

---

## 📂 文件目录结构

```
mp3d/17DRP5sb8fy/
├─ 17DRP5sb8fy.glb          (✓ 3D 模型 - 21MB)
├─ 17DRP5sb8fy.house        (房间拓扑信息)
├─ 17DRP5sb8fy.navmesh      (导航网格)
└─ 17DRP5sb8fy_semantic.ply (语义点云)
```

---

## 🔗 数据流向

```
MP3D 原始数据
    ↓
GLB 文件 (3D 模型)
    ├─→ 在线工具查看 (Three.js, Babylon)
    ├─→ Python 脚本分析 (trimesh)
    ├─→ VSCode 扩展预览
    └─→ 命令行工具 (gltf-transform)
    ↓
House 文件 (房间拓扑)
    └─→ 文本编辑器打开
    ↓
PLY 文件 (语义点云)
    └─→ Python 脚本可视化 (Open3D, etc.)
```

---

## 🎯 快速决策树

```
要查看 MP3D 场景吗?
    ├─ 不想安装任何东西?
    │   └─→ 使用在线工具 (Three.js Editor)
    │
    ├─ 想做数据分析?
    │   └─→ 运行 Python 脚本
    │
    ├─ 想要最佳性能?
    │   └─→ Babylon.js Sandbox
    │
    ├─ 需要详细信息?
    │   └─→ 命令行 gltf-transform
    │
    └─ 坚持用 VSCode 扩展?
        ├─ 更新 VSCode 到最新版
        ├─ 重新安装扩展
        └─ 还不行 → 回到上面的方案
```

---

## 📝 常见问题

### Q: 为什么文件那么大（21MB）？
A: MP3D 是真实房间的高保真 3D 重建，包含大量顶点、纹理等细节。

### Q: 能不能压缩文件？
A: 可以用 gltf-transform 压缩：
```bash
gltf-transform compress model.glb model-compressed.glb
```

### Q: 怎样提取点云？
A: 使用 trimesh 提取顶点：
```python
import trimesh
mesh = trimesh.load('17DRP5sb8fy.glb')
import numpy as np
points = mesh.vertices  # 获取顶点坐标
```

### Q: 支持其他 3D 格式吗？
A: 是的，所有在线工具都支持：
- .glb / .gltf (glTF 格式)
- .obj (Wavefront)
- .ply (点云)
- .stl (固体几何)
- .fbx (Autodesk)

---

## 总结

| 方案 | 难度 | 速度 | 功能 | 推荐度 |
|------|------|------|------|--------|
| 在线工具 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Python 脚本 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| VSCode 扩展 | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ |
| 命令行工具 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

**最终推荐：** 使用 Three.js Editor 在线工具或 Python 脚本！

