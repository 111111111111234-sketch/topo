#!/usr/bin/env python3
"""
FPV 到 BEV 转换方法详细对比和选择指南

这个脚本提供了详细的方法对比和决策树，帮助选择最适合的转换方法。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def create_method_comparison_visual():
    """创建三种方法的详细对比可视化"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # ============ 子图 1: 方法流程对比 ============
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('方法1: 深度投影 (Depth Projection)', fontsize=12, fontweight='bold')
    
    # 流程框
    steps = [
        ('RGB-D图像', 9, 'lightblue'),
        ('深度→3D点', 7.5, 'lightgreen'),
        ('坐标变换', 6, 'lightyellow'),
        ('地面检测', 4.5, 'lightcoral'),
        ('BEV地图', 3, 'lightgray'),
    ]
    
    for i, (text, y, color) in enumerate(steps):
        rect = FancyBboxPatch((1, y-0.3), 8, 0.6, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color, linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(5, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(steps) - 1:
            ax1.arrow(5, y-0.4, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')
    
    # 标注关键参数
    ax1.text(0.5, 1.5, '关键参数:\n• 相机内参(fx,fy,cx,cy)\n• 相机外参(h,pitch)\n• 地面高度阈值', 
             fontsize=8, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # ============ 子图 2: NavMesh 方法 ============
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('方法2: NavMesh投影', fontsize=12, fontweight='bold')
    
    steps2 = [
        ('NavMesh文件', 9, 'lightblue'),
        ('加载PathFinder', 7.5, 'lightgreen'),
        ('渲染topdown', 6, 'lightyellow'),
        ('坐标对齐', 4.5, 'lightcoral'),
        ('BEV地图', 3, 'lightgray'),
    ]
    
    for i, (text, y, color) in enumerate(steps2):
        rect = FancyBboxPatch((1, y-0.3), 8, 0.6, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color, linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(5, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(steps2) - 1:
            ax2.arrow(5, y-0.4, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')
    
    ax2.text(0.5, 1.5, '关键参数:\n• navmesh路径\n• 切片高度\n• 地图元数据', 
             fontsize=8, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # ============ 子图 3: 高度直方图方法 ============
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('方法3: 高度直方图', fontsize=12, fontweight='bold')
    
    steps3 = [
        ('高度映射', 9, 'lightblue'),
        ('直方图分析', 7.5, 'lightgreen'),
        ('峰值检测', 6, 'lightyellow'),
        ('语义过滤', 4.5, 'lightcoral'),
        ('BEV地图', 3, 'lightgray'),
    ]
    
    for i, (text, y, color) in enumerate(steps3):
        rect = FancyBboxPatch((1, y-0.3), 8, 0.6, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color, linewidth=1.5)
        ax3.add_patch(rect)
        ax3.text(5, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(steps3) - 1:
            ax3.arrow(5, y-0.4, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')
    
    ax3.text(0.5, 1.5, '关键参数:\n• 高度容差\n• 直方图bins\n• 形态学核大小', 
             fontsize=8, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # ============ 子图 4: 性能对比 ============
    ax4 = plt.subplot(2, 3, 4)
    
    methods = ['深度投影', 'NavMesh', '高度直方图']
    speed = [8, 6, 8]  # 1-10 scale, 10=fastest
    accuracy = [6, 10, 6]
    robustness = [6, 7, 8]
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax4.bar(x - width, speed, width, label='速度', color='skyblue')
    ax4.bar(x, accuracy, width, label='精度', color='lightgreen')
    ax4.bar(x + width, robustness, width, label='稳健性', color='lightcoral')
    
    ax4.set_ylabel('评分 (1-10)', fontsize=10, fontweight='bold')
    ax4.set_title('性能对比', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 11)
    
    # ============ 子图 5: 适用场景 ============
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.axis('off')
    ax5.set_title('适用场景对比', fontsize=12, fontweight='bold')
    
    scenarios = [
        ('方法', 'NavMesh', '深度投影', '高度直方图'),
        ('实时导航', '✗', '✓✓', '✓'),
        ('GT标注', '✓✓✓', '✗', '✗'),
        ('数据集处理', '✗', '✗', '✓✓'),
        ('多传感器融合', '✗', '✓', '✗'),
        ('动态场景', '✗', '✓✓', '✓'),
        ('离线规划', '✓✓', '✓', '✓'),
    ]
    
    y_start = 9
    for i, row in enumerate(scenarios):
        if i == 0:
            ax5.text(0.5, y_start, row[0], fontsize=9, fontweight='bold')
            ax5.text(3, y_start, row[1], fontsize=9, fontweight='bold', color='blue')
            ax5.text(5.5, y_start, row[2], fontsize=9, fontweight='bold', color='green')
            ax5.text(8, y_start, row[3], fontsize=9, fontweight='bold', color='red')
        else:
            color = 'lightgray' if i % 2 == 0 else 'white'
            rect = Rectangle((0.3, y_start-0.35), 9.4, 0.6, facecolor=color, edgecolor='gray', linewidth=0.5)
            ax5.add_patch(rect)
            ax5.text(0.5, y_start, row[0], fontsize=8)
            ax5.text(3, y_start, row[1], fontsize=8, ha='center')
            ax5.text(5.5, y_start, row[2], fontsize=8, ha='center')
            ax5.text(8, y_start, row[3], fontsize=8, ha='center')
        
        y_start -= 0.8
    
    # ============ 子图 6: 决策树 ============
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.axis('off')
    ax6.set_title('选择决策树', fontsize=12, fontweight='bold')
    
    # 决策树文字
    decision_tree = """
    您有NavMesh文件吗?
    ├─ 是 → NavMesh 方法
    │       （最精确，GT标注）
    │
    └─ 否 → 您需要实时处理吗?
            ├─ 是，有RGB-D相机
            │  → 深度投影法
            │    （快速，在线）
            │
            └─ 否，有历史高度数据
               → 高度直方图法
                  （稳健，多帧融合）
    """
    
    ax6.text(0.5, 8.5, decision_tree, fontsize=8, family='monospace',
             bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('/workspace/tangyx7@xiaopeng.com/fpv_to_bev_method_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("✅ 对比图表已保存: fpv_to_bev_method_comparison.png")
    
    return fig


def create_parameter_sensitivity_chart():
    """创建参数敏感性分析图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 子图1: 相机高度的影响
    ax = axes[0, 0]
    heights = np.linspace(1.0, 2.0, 20)
    ground_areas = 100 - heights * 15  # 模拟的地面覆盖率
    ax.plot(heights, ground_areas, 'b-', linewidth=2, marker='o')
    ax.axvline(1.38, color='r', linestyle='--', label='标准值 (1.38m)')
    ax.set_xlabel('相机高度 (m)', fontsize=10)
    ax.set_ylabel('地面覆盖率 (%)', fontsize=10)
    ax.set_title('相机高度的影响', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 子图2: 倾角的影响
    ax = axes[0, 1]
    pitches = np.linspace(0, 30, 20)
    max_depth = 10 / np.cos(np.deg2rad(pitches + 1e-6))  # 避免div by zero
    max_depth = np.clip(max_depth, 1, 20)
    ax.plot(pitches, max_depth, 'g-', linewidth=2, marker='s')
    ax.axvline(10, color='r', linestyle='--', label='标准值 (~10°)')
    ax.set_xlabel('相机倾角 (度)', fontsize=10)
    ax.set_ylabel('最大视深 (m)', fontsize=10)
    ax.set_title('相机倾角的影响', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 子图3: 地面高度阈值的影响
    ax = axes[1, 0]
    thresholds = np.linspace(0.1, 1.0, 20)
    noise_robustness = 100 * (1 - np.exp(-thresholds * 2))
    ax.plot(thresholds, noise_robustness, 'm-', linewidth=2, marker='^')
    ax.axvline(0.3, color='r', linestyle='--', label='推荐值 (0.3m)')
    ax.fill_between([0.2, 0.4], 0, 100, alpha=0.2, color='yellow', label='推荐范围')
    ax.set_xlabel('地面高度阈值 (m)', fontsize=10)
    ax.set_ylabel('抗噪能力 (%)', fontsize=10)
    ax.set_title('地面高度阈值的影响', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 子图4: 地图分辨率的影响
    ax = axes[1, 1]
    resolutions = np.array([0.01, 0.02, 0.025, 0.05, 0.10])
    memory_usage = resolutions ** (-2)  # 内存与分辨率平方成反比
    processing_time = resolutions ** (-2) * 0.5
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(resolutions * 100, memory_usage, 'b-', linewidth=2, marker='o', label='内存')
    line2 = ax2.plot(resolutions * 100, processing_time, 'r-', linewidth=2, marker='s', label='处理时间')
    
    ax.set_xlabel('地图分辨率 (cm/pixel)', fontsize=10)
    ax.set_ylabel('相对内存占用', fontsize=10, color='b')
    ax2.set_ylabel('相对处理时间', fontsize=10, color='r')
    ax.set_title('地图分辨率的影响', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('/workspace/tangyx7@xiaopeng.com/fpv_to_bev_parameter_sensitivity.png',
                dpi=150, bbox_inches='tight')
    print("✅ 参数敏感性图表已保存: fpv_to_bev_parameter_sensitivity.png")
    
    return fig


def create_algorithm_flowchart():
    """创建详细的算法流程图"""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(7, 9.5, 'FPV 到 BEV 转换完整流程', 
            fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round", facecolor='lightblue', edgecolor='blue', linewidth=2))
    
    # ====== 左侧：深度投影流程 ======
    y = 8.5
    ax.text(3.5, y, '深度投影法', fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.7))
    
    steps_left = [
        '① 读取深度图\n(H, W)',
        '② 计算内参矩阵\n(fx, fy, cx, cy)',
        '③ 深度→3D点\n(相机坐标)',
        '④ 坐标变换\n(世界坐标)',
        '⑤ 地面检测\n(高度阈值)',
        '⑥ 形态学处理\n(闭运算)',
        '⑦ 生成BEV地图',
    ]
    
    y = 8
    for step in steps_left:
        height = 0.5
        rect = FancyBboxPatch((1.5, y-height/2), 4, height, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='green', facecolor='lightgreen', 
                              linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(3.5, y, step, fontsize=8, ha='center', va='center')
        
        if y > 1.5:
            ax.arrow(3.5, y-height/2-0.1, 0, -0.3, head_width=0.2, head_length=0.15, 
                    fc='green', ec='green')
        y -= 0.9
    
    # ====== 中间：NavMesh流程 ======
    y = 8.5
    ax.text(7, y, 'NavMesh 方法', fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.7))
    
    steps_mid = [
        '① 加载NavMesh\n文件',
        '② 初始化PathFinder\n对象',
        '③ 获取NavMesh边界\n(bounds_min)',
        '④ 生成topdown视图\n(分辨率0.02)',
        '⑤ 计算坐标偏移\n(roi)',
        '⑥ 对齐到目标地图',
        '⑦ 生成BEV地图',
    ]
    
    y = 8
    for step in steps_mid:
        height = 0.5
        rect = FancyBboxPatch((5.5, y-height/2), 3, height,
                              boxstyle="round,pad=0.05",
                              edgecolor='orange', facecolor='lightyellow',
                              linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(7, y, step, fontsize=8, ha='center', va='center')
        
        if y > 1.5:
            ax.arrow(7, y-height/2-0.1, 0, -0.3, head_width=0.2, head_length=0.15,
                    fc='orange', ec='orange')
        y -= 0.9
    
    # ====== 右侧：高度直方图流程 ======
    y = 8.5
    ax.text(10.5, y, '高度直方图法', fontsize=11, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.7))
    
    steps_right = [
        '① 读取高度映射\n(累积)',
        '② 规范化高度值\n(H_min = 1)',
        '③ 构建直方图\n(bins=200)',
        '④ 检测峰值\n(地面高度)',
        '⑤ 范围检测\n(±0.1m)',
        '⑥ 语义过滤\n(去除物体)',
        '⑦ 生成BEV地图',
    ]
    
    y = 8
    for step in steps_right:
        height = 0.5
        rect = FancyBboxPatch((9.5, y-height/2), 4, height,
                              boxstyle="round,pad=0.05",
                              edgecolor='red', facecolor='lightcoral',
                              linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(11.5, y, step, fontsize=8, ha='center', va='center')
        
        if y > 1.5:
            ax.arrow(11.5, y-height/2-0.1, 0, -0.3, head_width=0.2, head_length=0.15,
                    fc='red', ec='red')
        y -= 0.9
    
    # ====== 底部：输出 ======
    y = 0.8
    output_box = FancyBboxPatch((4, 0.3), 6, 1, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='lightgray', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 0.8, '输出: BEV 地图 (H, W) - 二进制占据地图', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('/workspace/tangyx7@xiaopeng.com/fpv_to_bev_algorithm_flowchart.png',
                dpi=150, bbox_inches='tight')
    print("✅ 算法流程图已保存: fpv_to_bev_algorithm_flowchart.png")
    
    return fig


if __name__ == '__main__':
    print("=" * 70)
    print("生成 FPV 到 BEV 转换的详细可视化文档")
    print("=" * 70)
    
    print("\n[1/3] 生成方法对比图表...")
    create_method_comparison_visual()
    
    print("[2/3] 生成参数敏感性分析...")
    create_parameter_sensitivity_chart()
    
    print("[3/3] 生成算法流程图...")
    create_algorithm_flowchart()
    
    print("\n" + "=" * 70)
    print("✅ 所有可视化文档已生成完毕！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  • fpv_to_bev_method_comparison.png")
    print("  • fpv_to_bev_parameter_sensitivity.png")
    print("  • fpv_to_bev_algorithm_flowchart.png")
