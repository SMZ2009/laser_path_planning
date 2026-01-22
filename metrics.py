from typing import List, Dict, Tuple
import numpy as np
from kmeans_decrement import Region


class Metrics:
    @staticmethod
    def calculate_all(points: List, regions: List[Region], platform_path: List[int],
                     timeline: List[Dict], v1: float, v2: float, L: float) -> Dict:
        total_points = len(points)
        num_regions = len(regions)
        
        coverage = Metrics._calculate_coverage(points, regions)
        
        platform_distance = Metrics._calculate_platform_distance(regions, platform_path)
        
        platform_time = sum(s['move_time'] for s in timeline)
        laser_time = max((s['laser_end'] - s['laser_start']) for s in timeline) if timeline else 0
        wait_time = sum(s['wait_time'] for s in timeline)
        total_time = max(s['laser_end'] for s in timeline) if timeline else 0
        
        avg_time_per_point = total_time / total_points if total_points > 0 else 0
        avg_move_distance = platform_distance / len(platform_path) if platform_path else 0
        
        time_utilization = (laser_time / total_time * 100) if total_time > 0 else 0
        scan_ratio = (laser_time / total_time * 100) if total_time > 0 else 0
        move_ratio = (platform_time / total_time * 100) if total_time > 0 else 0
        
        points_per_region = [len(r.points) for r in regions]
        region_balance = np.std(points_per_region) / np.mean(points_per_region) if points_per_region else 0
        
        return {
            'total_points': total_points,
            'num_regions': num_regions,
            'coverage': coverage,
            'platform_distance': platform_distance,
            'avg_move_distance': avg_move_distance,
            'platform_time': platform_time,
            'laser_time': laser_time,
            'wait_time': wait_time,
            'total_time': total_time,
            'avg_time_per_point': avg_time_per_point,
            'time_utilization': time_utilization,
            'scan_ratio': scan_ratio,
            'move_ratio': move_ratio,
            'region_balance': region_balance,
            'avg_points_per_region': np.mean(points_per_region) if points_per_region else 0,
            'max_points_per_region': max(points_per_region) if points_per_region else 0,
            'min_points_per_region': min(points_per_region) if points_per_region else 0,
        }
    
    @staticmethod
    def _calculate_coverage(points: List, regions: List[Region]) -> float:
        total_points = len(points)
        if total_points == 0:
            return 1.0
        
        covered_points = sum(len(r.points) - len(r.uncovered_points) for r in regions)
        return covered_points / total_points
    
    @staticmethod
    def _calculate_platform_distance(regions: List[Region], platform_path: List[int]) -> float:
        if len(platform_path) <= 1:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(platform_path) - 1):
            r1 = regions[platform_path[i]]
            r2 = regions[platform_path[i + 1]]
            dist = ((r1.center[0] - r2.center[0])**2 + (r1.center[1] - r2.center[1])**2)**0.5
            total_distance += dist
        
        return total_distance
    
    @staticmethod
    def compare_solutions(metrics1: Dict, metrics2: Dict, labels: Tuple[str, str] = None) -> str:
        if labels is None:
            labels = ('方案1', '方案2')
        
        comparison = f"\n{'='*60}\n"
        comparison += f"{'='*20} 方案对比 {'='*20}\n"
        comparison += f"{'='*60}\n"
        comparison += f"{'指标':<25} {labels[0]:<15} {labels[1]:<15} {'改进':<10}\n"
        comparison += f"{'-'*60}\n"
        
        for key in ['total_time', 'platform_distance', 'coverage', 'time_utilization']:
            if key in metrics1 and key in metrics2:
                val1 = metrics1[key]
                val2 = metrics2[key]
                
                if key == 'coverage' or key == 'time_utilization':
                    improvement = (val2 - val1) / val1 * 100 if val1 > 0 else 0
                    improvement_str = f"+{improvement:+.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
                else:
                    improvement = (val1 - val2) / val1 * 100 if val1 > 0 else 0
                    improvement_str = f"+{improvement:+.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
                
                comparison += f"{Metrics._get_label(key):<25} {val1:<15.2f} {val2:<15.2f} {improvement_str:<10}\n"
        
        comparison += f"{'='*60}\n"
        
        return comparison
    
    @staticmethod
    def _get_label(key: str) -> str:
        labels = {
            'total_time': '总加工时间 (s)',
            'platform_distance': '平台路径长度 (mm)',
            'coverage': '覆盖率',
            'time_utilization': '时间利用率 (%)',
            'avg_time_per_point': '平均每点时间 (s)',
            'region_balance': '区域均衡度'
        }
        return labels.get(key, key)
