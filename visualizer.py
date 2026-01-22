from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from kmeans_decrement import Region
import matplotlib

# 尝试设置常见中文字体优先级，若系统存在这些字体则用于绘图，
# 这样可以避免中文字符缺失导致的方块或乱码问题。
preferred_cn_fonts = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['font.sans-serif'] = preferred_cn_fonts + matplotlib.rcParams.get('font.sans-serif', [])
matplotlib.rcParams['font.family'] = 'sans-serif'
# 允许负号正常显示（避免被误替换为方块）
matplotlib.rcParams['axes.unicode_minus'] = False


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.L = config.L
        self.marker_size = config.marker_size
        self.alpha = config.alpha
        self.dpi = config.figure_dpi
        self.figure_size = config.figure_size
    
    def plot_comprehensive_results(self, regions: List[Region], platform_path: List[int],
                                  timeline: List[Dict], metrics: Dict, output_file: str):
        fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_regions_and_points(ax1, regions)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_detailed_platform_path(ax2, regions, platform_path)
        
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_timeline_gantt(ax3, timeline)
        
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"可视化结果已保存至 {output_file}")
    
    def _plot_regions_and_points(self, ax, regions: List[Region]):
        for i, region in enumerate(regions):
            cx, cy = region.center
            
            square = patches.Rectangle(
                (cx - self.L/2, cy - self.L/2), self.L, self.L,
                fill=False, edgecolor='blue', alpha=0.3, linewidth=1.5
            )
            ax.add_patch(square)
            
            if region.points:
                xs = [p.x for p in region.points]
                ys = [p.y for p in region.points]
                ax.scatter(xs, ys, s=self.marker_size, alpha=self.alpha, color='red')
            
            ax.scatter(cx, cy, marker='x', color='green', s=50, linewidth=2)
            ax.text(cx, cy, f'R{i}', fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        ax.set_title('加工点分布与区域划分', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_detailed_platform_path(self, ax, regions: List[Region], platform_path: List[int]):
        # 绘制区域方格（较淡）
        for region in regions:
            cx, cy = region.center
            square = patches.Rectangle(
                (cx - self.L/2, cy - self.L/2), self.L, self.L,
                fill=False, edgecolor='#6FA8DC', alpha=0.12, linewidth=0.8
            )
            ax.add_patch(square)

        # 如果没有路径则直接返回
        if not platform_path:
            return

        # 提取路径中心坐标并绘制连线，优化线条样式
        path_coords = [regions[idx].center for idx in platform_path]
        xs = [p[0] for p in path_coords]
        ys = [p[1] for p in path_coords]

        ax.plot(xs, ys, linestyle='-', color='#FF6B6B', linewidth=2.0, alpha=0.9, zorder=5)
        ax.scatter(xs, ys, s=120, c='#FFD27F', edgecolors='#D35400', linewidth=1.5, zorder=10)

        # 绘制箭头表示移动方向：在每段中点位置放一个小箭头
        for i in range(1, len(path_coords)):
            start = path_coords[i - 1]
            end = path_coords[i]
            self._draw_arrow(ax, start, end, idx=i, color='#D35400')

        # 绘制序号（有白色圆背景并有边框以便在任意背景上清晰可见）
        for i, (x, y) in enumerate(path_coords):
            ax.text(x, y, str(i + 1), fontsize=10, ha='center', va='center', zorder=11,
                    bbox=dict(boxstyle='circle,pad=0.25', facecolor='white', edgecolor='#333333', linewidth=1))

        # 标注起点/终点图例
        ax.scatter(xs[0], ys[0], s=160, c='#7DCEA0', edgecolors='#145A32', linewidth=1.5, marker='o', zorder=12, label='Start')
        ax.scatter(xs[-1], ys[-1], s=160, c='#F1948A', edgecolors='#7A1F1F', linewidth=1.5, marker='o', zorder=12, label='End')

        # 视觉调整
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        ax.set_title('平台运动路径', fontsize=14, fontweight='bold', pad=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.6))
        ax.grid(True, alpha=0.25)

        # 增加少量边距，使箭头和标注不被裁剪
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_margin = max(5.0, (x_max - x_min) * 0.05)
        y_margin = max(5.0, (y_max - y_min) * 0.05)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # 图例位置
        ax.legend(loc='upper right')
    
    def _draw_arrow(self, ax, start: Tuple[float, float], end: Tuple[float, float], idx: int, color: str = '#D35400'):
        # 使用 FancyArrowPatch 绘制风格化箭头（短线段上显示）
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy

        # 计算箭头属性，mutation_scale 跟随 L 缩放，但对很短的段做最小值保护
        dist = (dx**2 + dy**2) ** 0.5
        mut_scale = max(self.L * 0.8, 8)

        arrow = FancyArrowPatch(posA=(sx, sy), posB=(ex, ey),
                               arrowstyle='-|>', mutation_scale=mut_scale,
                               linewidth=1.6, color=color, alpha=0.9)
        ax.add_patch(arrow)
    
    def _plot_timeline_gantt(self, ax, timeline: List[Dict]):
        for i, segment in enumerate(timeline):
         # 为避免与 matplotlib.bar/ barh 的参数冲突，使用不同的 y 偏移绘制堆叠条形
         ax.barh(i - 0.2, segment['move_time'], left=segment['platform_start'],
             height=0.4, color='steelblue', alpha=0.7, label='平台移动' if i == 0 else "")

         ax.barh(i + 0.2, segment['scan_time'], left=segment['laser_start'],
             height=0.4, color='tomato', alpha=0.7, label='激光扫描' if i == 0 else "")

         if segment['wait_time'] > 0.01:
          ax.barh(i - 0.4, segment['wait_time'], 
              left=min(segment['platform_start'], segment['laser_start']),
              height=0.8, color='yellow', alpha=0.4, label='等待时间' if i == 0 else "")
        
        ax.set_xlabel('时间', fontsize=10)
        ax.set_ylabel('区域编号', fontsize=10)
        ax.set_title('协同加工时间线', fontsize=12, fontweight='bold')
        
        ytick_labels = [f'R{s["region"]}' for s in timeline]
        ax.set_yticks(range(len(timeline)))
        ax.set_yticklabels(ytick_labels)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    def plot_metrics(self, metrics: Dict, output_file: str):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        text = self._format_metrics_text(metrics)
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='brown', linewidth=2))
        
        ax.axis('off')
        ax.set_title('关键性能指标', fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"指标图表已保存至 {output_file}")
    
    def _format_metrics_text(self, metrics: Dict) -> str:
        return f"""
{'='*50}
        关键性能指标
{'='*50}

【基本信息】
  总加工点数:     {metrics['total_points']:8}
  区域数量:       {metrics['num_regions']:8}
  覆盖率:         {metrics['coverage']*100:8.2f}%

【路径信息】
  平台路径长度:   {metrics['platform_distance']:8.2f} mm
  平均每段移动:   {metrics['avg_move_distance']:8.2f} mm

【时间信息】
  平台移动时间:   {metrics['platform_time']:8.2f} s
  激光扫描时间:   {metrics['laser_time']:8.2f} s
  等待时间:       {metrics['wait_time']:8.2f} s
  总加工时间:     {metrics['total_time']:8.2f} s
  平均每点时间:   {metrics['avg_time_per_point']:8.4f} s

【效率指标】
  时间利用率:     {metrics['time_utilization']:8.2f}%
  扫描时间占比:   {metrics['scan_ratio']:8.2f}%
  移动时间占比:   {metrics['move_ratio']:8.2f}%
{'='*50}
"""

    def save_metrics_table(self, metrics: Dict, output_file: str):
        """Save key performance metrics as a simple CSV table.

        The CSV will contain two columns: metric, value. Values are formatted for readability.
        """
        import csv

        # Ensure parent directory exists
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write CSV
        with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metric', 'value'])

            # Write metrics in a stable order
            ordered_keys = [
                'total_points', 'num_regions', 'coverage', 'platform_distance',
                'platform_time', 'laser_time', 'wait_time', 'total_time',
                'time_utilization', 'region_balance', 'avg_move_distance', 'avg_time_per_point',
                'scan_ratio', 'move_ratio'
            ]

            for k in ordered_keys:
                if k in metrics:
                    val = metrics[k]
                    # format percentages and floats nicely
                    if isinstance(val, float):
                        if 'ratio' in k or k == 'coverage' or k == 'time_utilization':
                            writer.writerow([k, f"{val:.4f}"])
                        else:
                            writer.writerow([k, f"{val:.4f}"])
                    else:
                        writer.writerow([k, str(val)])

            # write any remaining keys
            for k, v in metrics.items():
                if k not in ordered_keys:
                    writer.writerow([k, str(v)])

        print(f"指标表已保存至 {output_file}")
    
