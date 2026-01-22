from typing import List, Tuple
import numpy as np
from svg_parser import Point
from adaptive_grid import Bounds


class Region:
    def __init__(self, center: Tuple[float, float], points: List[Point], L: float):
        self.center = center
        self.points = points
        self.L = L
        self._update_bounds()
    
    def _update_bounds(self):
        if not self.points:
            self.bounds = Bounds(self.center[0] - self.L/2, self.center[0] + self.L/2,
                                self.center[1] - self.L/2, self.center[1] + self.L/2)
            self.uncovered_points = []
            return
        
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        self.bounds = Bounds(min(xs), max(xs), min(ys), max(ys))
        
        self.uncovered_points = []
        for p in self.points:
            if not self._is_in_square(p):
                self.uncovered_points.append(p)
    
    def _is_in_square(self, point: Point) -> bool:
        min_x = self.center[0] - self.L / 2
        max_x = self.center[0] + self.L / 2
        min_y = self.center[1] - self.L / 2
        max_y = self.center[1] + self.L / 2
        
        return min_x <= point.x <= max_x and min_y <= point.y <= max_y
    
    @property
    def coverage(self) -> float:
        if not self.points:
            return 1.0
        return (len(self.points) - len(self.uncovered_points)) / len(self.points)


class KmeansDecrement:
    def __init__(self, L: float, target_coverage: float = 0.999,
                 max_iterations: int = 100, convergence_tol: float = 1e-6,
                 convergence_threshold: float = 0.01):
        self.L = L
        self.target_coverage = target_coverage
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.convergence_threshold = convergence_threshold
    
    def decrement_cluster(self, points: List[Point], initial_centers: List[Tuple[float, float]], 
                          initial_K: int) -> Tuple[List[Region], int]:
        best_K = initial_K
        best_regions: List[Region] = []
        best_coverage = 0.0
        # 保存最近一次计算的 regions 以便在没有达到目标覆盖率时作为回退
        last_regions: List[Region] = []
        last_coverage = 0.0
        
        current_K = initial_K
        current_centers = initial_centers.copy()
        
        while current_K >= 1:
            regions = self._constrained_kmeans(points, current_centers, current_K)
            total_coverage = self._calculate_total_coverage(regions)
            last_regions = regions
            last_coverage = total_coverage
            
            print(f"K={current_K}, 覆盖率: {total_coverage:.4f}")
            
            if total_coverage >= self.target_coverage:
                best_K = current_K
                best_regions = regions
                best_coverage = total_coverage
                current_K -= 1
                current_centers = self._reduce_centers(current_centers, regions)
            else:
                break
        
        print(f"最终选择K={best_K}, 覆盖率: {best_coverage:.4f}")
        # 如果没有任何 K 满足 target_coverage，则回退到最近一次计算得到的 regions
        if not best_regions and last_regions:
            best_regions = last_regions
            best_coverage = last_coverage
        return best_regions, best_K
    
    def _constrained_kmeans(self, points: List[Point], centers: List[Tuple[float, float]], 
                           K: int) -> List[Region]:
        current_centers = centers[:K]
        prev_centers = None
        iteration = 0
        
        while iteration < self.max_iterations:
            prev_centers = current_centers.copy()
            
            assignments = self._assign_points(points, current_centers)
            
            current_centers = self._update_centers(assignments, current_centers, points)
            
            if self._has_converged(current_centers, prev_centers):
                break
            
            iteration += 1
        
        regions = []
        for i, center in enumerate(current_centers):
            region_points = [p for p, idx in zip(points, self._assign_points(points, current_centers)) 
                           if self._find_closest_center(p, current_centers) == i]
            region = Region(center, region_points, self.L)
            regions.append(region)
        
        return regions
    
    def _assign_points(self, points: List[Point], centers: List[Tuple[float, float]]) -> List[int]:
        assignments = []
        for point in points:
            assignments.append(self._find_closest_center(point, centers))
        return assignments
    
    def _find_closest_center(self, point: Point, centers: List[Tuple[float, float]]) -> int:
        min_dist = float('inf')
        closest_idx = 0
        
        for i, center in enumerate(centers):
            dist = np.sqrt((point.x - center[0])**2 + (point.y - center[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx
    
    def _update_centers(self, assignments: List[int], centers: List[Tuple[float, float]], points: List[Point]) -> List[Tuple[float, float]]:
        new_centers = []
        
        K = len(centers)
        for cluster_idx in range(K):
            cluster_points = [points[i] for i, a in enumerate(assignments) if a == cluster_idx]
            
            if len(cluster_points) == 0:
                new_centers.append(centers[cluster_idx])
            else:
                avg_x = np.mean([p.x for p in cluster_points])
                avg_y = np.mean([p.y for p in cluster_points])
                new_centers.append((avg_x, avg_y))
        
        return new_centers
    
    def _has_converged(self, current_centers: List[Tuple[float, float]], 
                      prev_centers: List[Tuple[float, float]]) -> bool:
        total_movement = 0.0
        
        for curr, prev in zip(current_centers, prev_centers):
            movement = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            total_movement += movement
        
        return total_movement < self.convergence_threshold
    
    def _calculate_total_coverage(self, regions: List[Region]) -> float:
        total_points = sum(len(r.points) for r in regions)
        if total_points == 0:
            return 1.0
        
        covered_points = 0
        for region in regions:
            covered_points += len(region.points) - len(region.uncovered_points)
        
        return covered_points / total_points
    
    def _reduce_centers(self, centers: List[Tuple[float, float]], regions: List[Region]) -> List[Tuple[float, float]]:
        if len(centers) <= 1:
            return centers
        
        region_loads = [len(r.points) for r in regions]
        region_importance = [(i, load) for i, load in enumerate(region_loads)]
        
        region_importance.sort(key=lambda x: x[1], reverse=True)
        
        new_centers = [centers[i] for i, _ in region_importance[:-1]]
        
        return new_centers
