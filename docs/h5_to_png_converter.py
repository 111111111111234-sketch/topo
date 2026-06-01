#!/usr/bin/env python3
"""
将 H5 文件转换为 PNG 图像
支持将语义地图、实例地图等数据可视化为图像
"""

import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from pathlib import Path
from tqdm import tqdm

def h5_to_png(h5_file, output_dir, save_individual=True, save_combined=False):
    """
    将 H5 文件转换为 PNG 图像
    
    参数:
        h5_file: H5 文件路径
        output_dir: 输出目录
        save_individual: 是否保存各个数据为单独的 PNG
        save_combined: 是否保存组合可视化
    """
    
    h5_name = Path(h5_file).stem  # 获取文件名（不含扩展名）
    
    with h5py.File(h5_file, 'r') as f:
        # 创建输出子目录
        item_output_dir = os.path.join(output_dir, h5_name)
        os.makedirs(item_output_dir, exist_ok=True)
        
        data = {}
        for key in f.keys():
            data[key] = f[key][:]
        
        if save_individual:
            # 保存 map_semantic
            if 'map_semantic' in data:
                semantic_map = data['map_semantic']
                # 归一化到 0-255
                if semantic_map.max() > 0:
                    semantic_normalized = ((semantic_map - semantic_map.min()) / 
                                         (semantic_map.max() - semantic_map.min()) * 255).astype(np.uint8)
                else:
                    semantic_normalized = semantic_map.astype(np.uint8)
                
                img = Image.fromarray(semantic_normalized, mode='L')
                semantic_path = os.path.join(item_output_dir, 'semantic_map.png')
                img.save(semantic_path)
                print(f"  ✓ 已保存: {semantic_path}")
            
            # 保存 map_instance
            if 'map_instance' in data:
                instance_map = data['map_instance']
                # 处理负值，将其转换为 0
                instance_map_processed = np.where(instance_map >= 0, instance_map, 0)
                if instance_map_processed.max() > 0:
                    instance_normalized = ((instance_map_processed - instance_map_processed.min()) / 
                                         (instance_map_processed.max() - instance_map_processed.min()) * 255).astype(np.uint8)
                else:
                    instance_normalized = instance_map_processed.astype(np.uint8)
                
                img = Image.fromarray(instance_normalized, mode='L')
                instance_path = os.path.join(item_output_dir, 'instance_map.png')
                img.save(instance_path)
                print(f"  ✓ 已保存: {instance_path}")
            
            # 保存 map_z (Z 坐标)
            if 'map_z' in data:
                z_map = data['map_z']
                if z_map.max() > z_map.min():
                    z_normalized = ((z_map - z_map.min()) / 
                                  (z_map.max() - z_map.min()) * 255).astype(np.uint8)
                else:
                    z_normalized = z_map.astype(np.uint8)
                
                img = Image.fromarray(z_normalized, mode='L')
                z_path = os.path.join(item_output_dir, 'z_map.png')
                img.save(z_path)
                print(f"  ✓ 已保存: {z_path}")
            
            # 保存 mask
            if 'mask' in data:
                mask = data['mask'].astype(np.uint8) * 255
                img = Image.fromarray(mask, mode='L')
                mask_path = os.path.join(item_output_dir, 'mask.png')
                img.save(mask_path)
                print(f"  ✓ 已保存: {mask_path}")
        
        if save_combined:
            # 创建组合可视化
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            if 'map_semantic' in data:
                axes[0, 0].imshow(data['map_semantic'], cmap='viridis')
                axes[0, 0].set_title('Semantic Map')
                axes[0, 0].axis('off')
            
            if 'map_instance' in data:
                axes[0, 1].imshow(data['map_instance'], cmap='tab20')
                axes[0, 1].set_title('Instance Map')
                axes[0, 1].axis('off')
            
            if 'map_z' in data:
                axes[1, 0].imshow(data['map_z'], cmap='coolwarm')
                axes[1, 0].set_title('Z Map')
                axes[1, 0].axis('off')
            
            if 'mask' in data:
                axes[1, 1].imshow(data['mask'], cmap='gray')
                axes[1, 1].set_title('Mask')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            combined_path = os.path.join(item_output_dir, 'combined_visualization.png')
            plt.savefig(combined_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 已保存: {combined_path}")

def batch_convert(input_dir, output_dir, save_individual=True, save_combined=False):
    """
    批量转换目录中的所有 H5 文件
    
    参数:
        input_dir: 包含 H5 文件的目录
        output_dir: 输出目录
        save_individual: 是否保存各个数据为单独的 PNG
        save_combined: 是否保存组合可视化
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有 H5 文件
    h5_files = list(Path(input_dir).glob('*.h5'))
    
    if not h5_files:
        print(f"未找到 H5 文件在 {input_dir}")
        return
    
    print(f"找到 {len(h5_files)} 个 H5 文件")
    print(f"开始转换...")
    
    for h5_file in tqdm(h5_files, desc="转换进度"):
        try:
            h5_to_png(str(h5_file), output_dir, save_individual, save_combined)
        except Exception as e:
            print(f"  ✗ 错误: {h5_file} - {str(e)}")

if __name__ == '__main__':
    # 配置参数
    input_dir = '/workspace/tangyx7@xiaopeng.com/Semantic-MapNet111/data/semmap'
    output_dir = '/workspace/tangyx7@xiaopeng.com/Semantic-MapNet111/data/semmap_png'
    
    # 执行转换
    batch_convert(input_dir, output_dir, save_individual=True, save_combined=True)
    
    print(f"\n✓ 转换完成！输出目录: {output_dir}")
