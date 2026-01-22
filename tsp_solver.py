from typing import List, Tuple
import numpy as np
import random


class TSPSolver:
    def __init__(self, use_3opt: bool = False, max_iterations: int = 1000):
        self.use_3opt = use_3opt
        self.max_iterations = max_iterations
    
    def solve(self, centers: List[Tuple[float, float]]) -> List[int]:
        if not centers:
            return []
        
        if len(centers) == 1:
            return [0]
        
        initial_path = self._nearest_neighbor(centers)
        
        if self.use_3opt:
            optimized_path = self._three_opt(initial_path, centers)
        else:
            optimized_path = self._two_opt(initial_path, centers)
        
        return optimized_path
    
    def _nearest_neighbor(self, centers: List[Tuple[float, float]]) -> List[int]:
        n = len(centers)
        visited = [False] * n
        path = [random.randint(0, n - 1)]
        visited[path[0]] = True
        
        for _ in range(n - 1):
            current = path[-1]
            nearest = -1
            min_dist = float('inf')
            
            for j in range(n):
                if not visited[j]:
                    dist = self._distance(centers[current], centers[j])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = j
            
            path.append(nearest)
            visited[nearest] = True
        
        return path
    
    def _two_opt(self, path: List[int], centers: List[Tuple[float, float]]) -> List[int]:
        n = len(path)
        current_path = path.copy()
        current_cost = self._calculate_path_cost(current_path, centers)
        
        iteration = 0
        improved = True
        
        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1
            
            for i in range(n - 1):
                for j in range(i + 2, n):
                    new_path = self._two_opt_swap(current_path, i, j)
                    new_cost = self._calculate_path_cost(new_path, centers)
                    
                    if new_cost < current_cost:
                        current_path = new_path
                        current_cost = new_cost
                        improved = True
        
        return current_path
    
    def _two_opt_swap(self, path: List[int], i: int, j: int) -> List[int]:
        new_path = path[:i + 1]
        new_path.extend(path[j:i:-1])
        new_path.extend(path[j + 1:])
        return new_path
    
    def _three_opt(self, path: List[int], centers: List[Tuple[float, float]]) -> List[int]:
        n = len(path)
        current_path = self._two_opt(path, centers)
        current_cost = self._calculate_path_cost(current_path, centers)
        
        improved = True
        iteration = 0
        
        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1
            
            for i in range(n - 3):
                for j in range(i + 1, n - 2):
                    for k in range(j + 1, n - 1):
                        for opt in range(1, 7):
                            new_path = self._three_opt_reconnect(current_path, i, j, k, opt)
                            new_cost = self._calculate_path_cost(new_path, centers)
                            
                            if new_cost < current_cost:
                                current_path = new_path
                                current_cost = new_cost
                                improved = True
        
        return current_path
    
    def _three_opt_reconnect(self, path: List[int], i: int, j: int, k: int, 
                           opt: int) -> List[int]:
        a, b, c, d = path[i], path[i + 1], path[j], path[j + 1]
        e, f = path[k], path[(k + 1) % len(path)]
        
        if opt == 1:
            new_path = (path[:i + 1] + path[j:k + 1] + path[i + 1:j + 1] + path[k + 1:])
        elif opt == 2:
            new_path = (path[:i + 1] + path[j + 1:k + 1][::-1] + path[i + 1:j + 1] + path[k + 1:])
        elif opt == 3:
            new_path = (path[:i + 1] + path[j:k + 1] + path[j:i:-1] + path[k + 1:])
        elif opt == 4:
            new_path = (path[:i + 1] + path[k:j:-1] + path[i + 1:j + 1] + path[k + 1:])
        elif opt == 5:
            new_path = (path[:i + 1] + path[j:k + 1] + path[j + 1:i:-1] + path[k + 1:])
        elif opt == 6:
            new_path = (path[:i + 1] + path[k:j:-1] + path[j:i:-1] + path[k + 1:])
        else:
            new_path = path
        
        return new_path
    
    def _calculate_path_cost(self, path: List[int], centers: List[Tuple[float, float]]) -> float:
        total = 0.0
        
        for i in range(len(path) - 1):
            total += self._distance(centers[path[i]], centers[path[i + 1]])
        
        return total
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
