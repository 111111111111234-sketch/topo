#!/usr/bin/env python3
"""
FPV 到 BEV 地图转换完整资源指南
=====================================

这个目录包含了在 Semantic-MapNet111 项目中进行第一人称视角(FPV)到
鸟瞰图(BEV/Topdown)地图转换的所有资源。

📂 目录结构:
============

核心库和代码:
  ✅ fpv_to_bev.py                     (15KB) - 生产级库实现
  ✅ generate_fpv_to_bev_visualizations.py - 可视化生成工具

完整文档和指南:
  ✅ FPV_TO_BEV_GUIDE.md               (13KB) - 详细技术指南
  ✅ FPV_TO_BEV_QUICK_REFERENCE.md     (5KB)  - 一页纸快速参考
  ✅ FPV_TO_BEV_RESOURCES_INDEX.md     (11KB) - 资源总索引 ⭐ 从这里开始

可视化资源:
  ✅ fpv_to_bev_method_comparison.png  (93KB) - 三种方法对比图
  ✅ fpv_to_bev_parameter_sensitivity.png - 参数敏感性分析
  ✅ fpv_to_bev_algorithm_flowchart.png (97KB) - 算法流程图

教学和演示:
  ✅ FPV_to_BEV_Complete_Guide.ipynb   (31KB) - 可运行的Jupyter笔记本


🚀 快速开始 (3步):
===================

第1步: 查看概览
  阅读本文件的下面部分 (5分钟)

第2步: 了解三种方法
  查看 fpv_to_bev_method_comparison.png (5分钟)
  或阅读 FPV_TO_BEV_QUICK_REFERENCE.md (10分钟)

第3步: 运行代码示例
  $ jupyter notebook FPV_to_BEV_Complete_Guide.ipynb
  或导入库: from fpv_to_bev import project_depth_to_occupancy


📖 详细指南 (按推荐阅读顺序):
=============================

初级 (入门, 30分钟):
  1. 本文件 (README)
  2. FPV_TO_BEV_QUICK_REFERENCE.md
  3. fpv_to_bev_method_comparison.png

中级 (应用, 2小时):
  4. FPV_to_BEV_Complete_Guide.ipynb (运行和修改代码)
  5. FPV_TO_BEV_GUIDE.md (对应方法部分)
  6. fpv_to_bev_parameter_sensitivity.png (优化参数)

高级 (深度, 3小时+):
  7. FPV_TO_BEV_GUIDE.md (完整阅读)
  8. fpv_to_bev.py (源码分析)
  9. 修改代码适应特定应用场景


🎯 三种方法概览:
=================

┌─ 方法1: 深度投影 ✅ 最灵活
│  ├─ 输入: RGB-D深度图
│  ├─ 输出: BEV占据地图  
│  ├─ 速度: ~5-10ms
│  ├─ 精度: 中等
│  └─ 最佳用途: 实时机器人导航
│
├─ 方法2: NavMesh投影 ✅ 最精确
│  ├─ 输入: NavMesh文件
│  ├─ 输出: 权威GT地图
│  ├─ 速度: ~1-2ms
│  ├─ 精度: 高
│  └─ 最佳用途: GT标注、离线规划
│
└─ 方法3: 高度直方图 ✅ 最稳健
   ├─ 输入: 累积高度映射
   ├─ 输出: BEV地图
   ├─ 速度: ~3-5ms
   ├─ 精度: 中等
   └─ 最佳用途: 数据集处理、多帧融合


💻 代码使用示例:
=================

# 1. 方法1 - 深度投影 (最简单)
from fpv_to_bev import project_depth_to_occupancy

bev_map = project_depth_to_occupancy(
    depth_image,
    camera_intrinsics=(fx, fy, cx, cy),  # 相机内参
    map_size=256,
    map_resolution=0.025
)

# 2. 方法2 - NavMesh投影
from fpv_to_bev import get_topdown_from_navmesh

bev_map, pathfinder = get_topdown_from_navmesh(
    'path/to/scene.navmesh'
)

# 3. 方法3 - 高度直方图
from fpv_to_bev import build_freespace_from_height_histogram

bev_map = build_freespace_from_height_histogram(
    height_map,
    semmap=semantic_map,
    observed_map=observed_map
)


📊 关键参数参考:
=================

相机参数:
  • 相机高度: 1.38m (成人眼睛高度)
  • 相机倾角: 0.174 rad ≈ 10°
  • 视场角(hfov): 110° (深度相机)

地图参数:
  • 分辨率: 0.02-0.025 m/pixel
  • 地图大小: 256×256 或 512×512
  • 地面高度范围: ±0.3m

形态学参数:
  • 闭运算核: 5×5 (ObjectNav) 或 3×3 (NavMesh)


🔧 安装和设置:
================

基础安装:
  pip install numpy scipy opencv-python pillow matplotlib

可选依赖:
  # 用于 NavMesh 方法
  pip install habitat-sim
  
  # 用于 GPU 加速
  pip install torch

验证安装:
  python -c "from fpv_to_bev import *; print('✅ 安装成功')"


❓ 常见问题:
=============

Q: 选择哪种方法?
A: 根据输入数据类型:
   - 有RGB-D相机     → 方法1 (深度投影)
   - 有NavMesh文件   → 方法2 (NavMesh投影)  
   - 有历史高度数据  → 方法3 (高度直方图)

Q: 地图全是黑色怎么办?
A: 检查:
   1. 深度图范围 (应该是 0.1-10m)
   2. 相机内参是否正确
   3. 地面高度阈值 (尝试改为 0.5m)
   4. 参考Jupyter笔记本的调试部分

Q: 如何加快处理速度?
A: 可以尝试:
   1. 降低地图分辨率 (0.025 → 0.05)
   2. 减小地图大小 (512 → 256)
   3. 使用GPU加速
   4. 参考 FPV_TO_BEV_GUIDE.md 的优化部分

更多问题见: FPV_TO_BEV_QUICK_REFERENCE.md 或 FPV_TO_BEV_GUIDE.md


📚 相关资源:
=============

官方项目:
  • Semantic-MapNet111 - 原始项目
  • habitat-sim - 模拟器和NavMesh加载
  • mmdetection3d - 3D目标检测

参考论文:
  • Perspective Transformer Nets (NIPS 2016)
  • Monodepth - 单目深度估计
  • habitat-lab - 模拟环境


🎓 学习建议:
==============

如果你是完全初学者:
  1. 先阅读 FPV_TO_BEV_QUICK_REFERENCE.md
  2. 看 fpv_to_bev_method_comparison.png
  3. 运行 FPV_to_BEV_Complete_Guide.ipynb 的前3个cell

如果你有3D视觉经验:
  1. 直接查看 fpv_to_bev_algorithm_flowchart.png
  2. 阅读 FPV_TO_BEV_GUIDE.md 的"坐标系"部分
  3. 浏览 fpv_to_bev.py 源码

如果你需要快速集成:
  1. 查看本文件的"代码使用示例"部分
  2. 复制相关函数到你的项目
  3. 根据 FPV_TO_BEV_QUICK_REFERENCE.md 调整参数


✨ 资源统计:
==============

📝 文档:          29 KB (3份)
💻 代码:          31 KB (2份)  
📊 图表:         322 KB (3份)
📓 笔记本:        31 KB (1份)
─────────────────────────
总计:            413 KB (9份资源)


💡 使用建议:
=============

• 把 fpv_to_bev.py 复制到你的项目中使用
• 参考 Jupyter 笔记本进行快速实验
• 根据 FPV_TO_BEV_QUICK_REFERENCE.md 快速查阅参数
• 遇到问题时查阅详细指南

"""

if __name__ == '__main__':
    # 打印上面的文档字符串
    print(__doc__)
    
    # 可选: 验证依赖
    print("\n✅ 检查依赖...\n")
    deps = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'scipy': 'SciPy',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✓ {name:15} 已安装")
        except ImportError:
            print(f"✗ {name:15} 未安装")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  缺少依赖: {', '.join(missing)}")
        print("   运行: pip install " + " ".join(missing).lower().replace(" ", "-"))
    else:
        print("\n✅ 所有基础依赖已安装!")
    
    # 检查可选依赖
    print("\n📦 检查可选依赖...\n")
    optional = {'habitat_sim': 'Habitat-Sim (用于NavMesh方法)', 'torch': 'PyTorch (用于GPU加速)'}
    
    for module, desc in optional.items():
        try:
            __import__(module)
            print(f"✓ {desc}")
        except ImportError:
            print(f"  {desc} - 可选")
