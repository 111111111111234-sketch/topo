#!/usr/bin/env python3
"""
查看和分析 MP3D GLB 场景文件的脚本
使用 trimesh 库进行 3D 模型分析和可视化
"""

import os
import sys
import trimesh
import numpy as np
from pathlib import Path

def analyze_glb(glb_path):
    """分析并显示 GLB 文件信息"""
    
    if not os.path.exists(glb_path):
        print(f"❌ 文件不存在: {glb_path}")
        return
    
    print("\n" + "=" * 70)
    print(f"📊 GLB 文件分析: {os.path.basename(glb_path)}")
    print("=" * 70)
    
    try:
        # 加载模型
        mesh = trimesh.load(glb_path)
        
        # 基本信息
        print(f"\n📐 几何信息:")
        print(f"   ├─ 顶点数: {len(mesh.vertices):,}")
        print(f"   ├─ 面数: {len(mesh.faces):,}")
        print(f"   ├─ 边数: {len(mesh.edges_unique):,}")
        
        # 边界框
        bounds = mesh.bounds
        print(f"\n🎯 空间范围:")
        print(f"   ├─ X: [{bounds[0][0]:.2f}, {bounds[1][0]:.2f}]")
        print(f"   ├─ Y: [{bounds[0][1]:.2f}, {bounds[1][1]:.2f}]")
        print(f"   └─ Z: [{bounds[0][2]:.2f}, {bounds[1][2]:.2f}]")
        
        # 物理属性
        print(f"\n⚙️  物理属性:")
        print(f"   ├─ 体积: {mesh.volume:.2f} m³")
        print(f"   ├─ 表面积: {mesh.area:.2f} m²")
        print(f"   ├─ 中心: ({mesh.center[0]:.2f}, {mesh.center[1]:.2f}, {mesh.center[2]:.2f})")
        
        # 网格统计
        if isinstance(mesh, trimesh.Scene):
            print(f"\n🔗 场景信息:")
            print(f"   ├─ 网格数: {len(mesh.geometry)}")
            for i, (name, geom) in enumerate(mesh.geometry.items()):
                print(f"   ├─ [{i}] {name}: {len(geom.vertices)} 顶点, {len(geom.faces)} 面")
        else:
            print(f"\n✓ 单一网格模型")
        
        # 材质信息
        if hasattr(mesh, 'visual'):
            print(f"\n🎨 材质信息:")
            if hasattr(mesh.visual, 'material'):
                print(f"   └─ 有材质定义")
            else:
                print(f"   └─ 无材质定义")
        
        print(f"\n✓ 分析完成！")
        print("=" * 70 + "\n")
        
        return mesh
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return None


def interactive_viewer():
    """交互式查看器"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    print("\n" + "=" * 70)
    print("🎮 交互式 3D 预览")
    print("=" * 70)
    print("鼠标操作：")
    print("  ├─ 左键拖拽：旋转视角")
    print("  ├─ 滚轮：缩放")
    print("  └─ 右键拖拽：平移\n")
    
    glb_path = '/workspace/tangyx7@xiaopeng.com/Semantic-MapNet111/data/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb'
    
    if not os.path.exists(glb_path):
        print(f"❌ 找不到文件: {glb_path}")
        return
    
    try:
        mesh = trimesh.load(glb_path)
        
        # 绘制 3D 模型
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制顶点点云
        vertices = mesh.vertices
        print(f"正在绘制 {len(vertices):,} 个顶点...")
        
        # 采样显示（大模型的情况）
        if len(vertices) > 100000:
            sample_idx = np.random.choice(len(vertices), 100000, replace=False)
            vertices_sample = vertices[sample_idx]
        else:
            vertices_sample = vertices
        
        ax.scatter(vertices_sample[:, 0], vertices_sample[:, 1], vertices_sample[:, 2],
                  s=0.5, c='lightblue', alpha=0.6, marker='.')
        
        # 绘制三角面（低分辨率）
        if len(mesh.faces) < 50000:
            # 绘制边框
            for face in mesh.faces[:10000]:
                vertices_face = mesh.vertices[face]
                # 闭合三角形
                vertices_face = np.vstack([vertices_face, vertices_face[0]])
                ax.plot(vertices_face[:, 0], vertices_face[:, 1], vertices_face[:, 2],
                       'gray', linewidth=0.2, alpha=0.3)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'MP3D 场景: {os.path.basename(glb_path)}')
        
        plt.show()
        print("✓ 预览完成")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎯 MP3D GLB 场景查看工具")
    print("=" * 70)
    
    # 分析 GLB 文件
    glb_path = '/workspace/tangyx7@xiaopeng.com/Semantic-MapNet111/data/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb'
    mesh = analyze_glb(glb_path)
    
    # 问用户是否要进行交互式预览
    if mesh:
        try:
            response = input("\n是否进行 3D 交互式预览? (y/n): ").strip().lower()
            if response == 'y':
                interactive_viewer()
        except KeyboardInterrupt:
            print("\n已取消")
