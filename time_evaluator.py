from typing import List, Tuple, Dict
import numpy as np
from kmeans_decrement import Region


class TimeEvaluator:
    def __init__(self, regions: List[Region], v1: float, v2: float, L: float):
        self.regions = regions
        self.v1 = v1
        self.v2 = v2
        self.L = L
        self.region_scan_times = self._precompute_scan_times()
    
    def _precompute_scan_times(self) -> List[float]:
        scan_times = []
        
        for region in self.regions:
            if len(region.points) <= 1:
                scan_times.append(0.0)
            else:
                scan_distance = self._calculate_region_scan_distance(region)
                scan_time = scan_distance / self.v2
                scan_times.append(scan_time)
        
        return scan_times
    
    def _calculate_region_scan_distance(self, region: Region) -> float:
        if len(region.points) <= 1:
            return 0.0
        
        points = region.points
        total_distance = 0.0
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            total_distance += np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        return total_distance
    
    def evaluate(self, platform_path: List[int]) -> float:
        if not platform_path:
            return 0.0
        
        total_time = 0.0
        platform_pos = self.regions[platform_path[0]].center
        
        for i, region_idx in enumerate(platform_path):
            region = self.regions[region_idx]
            
            move_time = self._distance(platform_pos, region.center) / self.v1
            total_time += move_time
            
            scan_time = self.region_scan_times[region_idx]
            total_time += scan_time
            
            platform_pos = region.center
        
        return total_time
    
    def evaluate_with_timeline(self, platform_path: List[int]) -> Tuple[List[Dict], float]:
        if not platform_path:
            return [], 0.0
        
        timeline = []
        platform_time = 0.0
        laser_time = 0.0
        
        for i, region_idx in enumerate(platform_path):
            region = self.regions[region_idx]
            
            if i == 0:
                move_time = 0.0
            else:
                prev_region = self.regions[platform_path[i - 1]]
                move_time = self._distance(prev_region.center, region.center) / self.v1
            
            platform_time += move_time
            
            scan_time = self.region_scan_times[region_idx]
            
            laser_start = max(platform_time, laser_time)
            laser_end = laser_start + scan_time
            laser_time = laser_end
            
            timeline.append({
                'region': region_idx,
                'platform_start': platform_time - move_time,
                'platform_end': platform_time,
                'laser_start': laser_start,
                'laser_end': laser_end,
                'move_time': move_time,
                'scan_time': scan_time,
                'wait_time': max(0, platform_time - laser_start)
            })
        
        total_time = max(platform_time, laser_time)
        
        return timeline, total_time
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def get_region_info(self) -> List[Dict]:
        region_info = []
        
        for i, region in enumerate(self.regions):
            info = {
                'region_id': i,
                'center': region.center,
                'num_points': len(region.points),
                'scan_time': self.region_scan_times[i],
                'coverage': region.coverage,
                'num_uncovered': len(region.uncovered_points)
            }
            region_info.append(info)
        
        return region_info
