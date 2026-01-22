from typing import List, Dict
from kmeans_decrement import Region


class Coordinator:
    def __init__(self, regions: List[Region]):
        self.regions = regions
    
    def generate_detailed_schedule(self, platform_path: List[int], 
                                  region_scan_orders: Dict[int, List[int]] = None) -> Dict:
        if region_scan_orders is None:
            region_scan_orders = {}
        if region_scan_orders is None:
            region_scan_orders = {}
        schedule = {
            'platform_path': platform_path,
            'platform_movements': [],
            'laser_scans': [],
            'timeline': []
        }
        
        current_position = None
        
        for i, region_idx in enumerate(platform_path):
            region = self.regions[region_idx]
            
            if current_position is None:
                move_distance = 0.0
            else:
                prev_region = self.regions[platform_path[i - 1]]
                move_distance = self._calculate_distance(
                    prev_region.center, region.center
                )
            
            schedule['platform_movements'].append({
                'from_region': platform_path[i - 1] if i > 0 else None,
                'to_region': region_idx,
                'distance': move_distance,
                'center': region.center,
                'num_points': len(region.points)
            })
            
            scan_order = region_scan_orders.get(region_idx, list(range(len(region.points))))
            schedule['laser_scans'].append({
                'region': region_idx,
                'scan_order': scan_order,
                'num_points': len(region.points),
                'scan_distance': self._calculate_scan_distance(region, scan_order)
            })
            
            current_position = region.center
        
        return schedule
    
    def _calculate_distance(self, p1: tuple, p2: tuple) -> float:
        from time_evaluator import TimeEvaluator
        evaluator = TimeEvaluator(self.regions, 1.0, 1.0, 1.0)
        return evaluator._distance(p1, p2)
    
    def _calculate_scan_distance(self, region: Region, scan_order: List[int]) -> float:
        if len(region.points) <= 1:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(scan_order) - 1):
            p1 = region.points[scan_order[i]]
            p2 = region.points[scan_order[i + 1]]
            total_distance += ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        
        return total_distance
    
    def optimize_region_scan_order(self, region_idx: int) -> List[int]:
        region = self.regions[region_idx]
        n = len(region.points)
        
        if n <= 1:
            return list(range(n))
        
        return self._nearest_neighbor_tsp(region)
    
    def _nearest_neighbor_tsp(self, region: Region) -> List[int]:
        n = len(region.points)
        visited = [False] * n
        order = [0]
        visited[0] = True
        
        for _ in range(n - 1):
            current = order[-1]
            nearest = -1
            min_dist = float('inf')
            
            for j in range(n):
                if not visited[j]:
                    p1 = region.points[current]
                    p2 = region.points[j]
                    dist = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest = j
            
            order.append(nearest)
            visited[nearest] = True
        
        return order
    
    def generate_optimized_schedule(self, platform_path: List[int]) -> Dict:
        region_scan_orders = {}
        
        for region_idx in platform_path:
            region_scan_orders[region_idx] = self.optimize_region_scan_order(region_idx)
        
        return self.generate_detailed_schedule(platform_path, region_scan_orders)
