from typing import List, Tuple
import numpy as np
from svg_parser import Point


class Bounds:
    def __init__(self, min_x: float, max_x: float, min_y: float, max_y: float):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
    
    @property
    def width(self):
        return self.max_x - self.min_x
    
    @property
    def height(self):
        return self.max_y - self.min_y


class AdaptiveGrid:
    def __init__(self, safety_factor: float = 0.9):
        self.safety_factor = safety_factor
    
    def generate(self, points: List[Point], L: float) -> Tuple[List[Tuple[float, float]], int]:
        if not points:
            return [], 0
        
        bounds = self._calculate_bounds(points)
        grid_size = L * self.safety_factor
        
        centers, num_centers = self._generate_centers(bounds, grid_size)
        
        return centers, num_centers
    
    def _calculate_bounds(self, points: List[Point]) -> Bounds:
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        
        return Bounds(min(xs), max(xs), min(ys), max(ys))
    
    def _generate_centers(self, bounds: Bounds, grid_size: float) -> Tuple[List[Tuple[float, float]], int]:
        centers = []
        
        num_cols = int(np.ceil(bounds.width / grid_size))
        num_rows = int(np.ceil(bounds.height / grid_size))
        
        for i in range(num_rows):
            for j in range(num_cols):
                x = bounds.min_x + j * grid_size + grid_size / 2
                y = bounds.min_y + i * grid_size + grid_size / 2
                centers.append((x, y))
        
        return centers, len(centers)
    
    def assign_points_to_grid(self, points: List[Point], centers: List[Tuple[float, float]], 
                              grid_size: float) -> dict:
        assignments = {i: [] for i in range(len(centers))}
        
        for point in points:
            closest_center_idx = self._find_closest_center(point, centers)
            assignments[closest_center_idx].append(point)
        
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
