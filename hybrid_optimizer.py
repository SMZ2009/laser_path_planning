from typing import List, Callable
import numpy as np
import random
import math


class HybridOptimizer:
    def __init__(self, ga_config: dict, sa_config: dict):
        self.ga_config = ga_config
        self.sa_config = sa_config
    
    def optimize(self, initial_path: List[int], evaluator: Callable[[List[int]], float]) -> List[int]:
        problem_scale = len(initial_path)

        # 边界保护：当路径为空或仅包含一个节点时，直接返回
        if problem_scale == 0:
            print("警告: 初始路径为空，跳过优化")
            return initial_path
        if problem_scale == 1:
            print("警告: 初始路径仅包含1个节点，优化无意义，直接返回")
            return initial_path

        if problem_scale < 10:
            print("小规模问题: 采用SA优先策略")
            strategy = 'sa_first'
        elif problem_scale < 50:
            print("中等规模问题: 采用GA优先策略")
            strategy = 'ga_first'
        else:
            print("大规模问题: 采用并行混合策略")
            strategy = 'parallel_hybrid'
        
        if strategy == 'sa_first':
            sa_path = self._simulated_annealing(initial_path, evaluator)
            final_path = self._genetic_algorithm(sa_path, evaluator)
        elif strategy == 'ga_first':
            ga_path = self._genetic_algorithm(initial_path, evaluator)
            final_path = self._simulated_annealing(ga_path, evaluator)
        else:
            final_path = self._parallel_hybrid(initial_path, evaluator)
        
        final_path = self._local_search(final_path, evaluator)
        
        return final_path
    
    def _genetic_algorithm(self, initial_path: List[int], evaluator: Callable[[List[int]], float]) -> List[int]:
        population = self._initialize_population(initial_path, self.ga_config['ga_population_size'])
        
        best_path = initial_path.copy()
        best_cost = evaluator(best_path)
        
        for generation in range(self.ga_config['ga_generations']):
            fitness = [1 / (evaluator(p) + 1e-10) for p in population]
            
            selected = self._selection(population, fitness, self.ga_config['ga_tournament_size'])
            offspring = self._crossover(selected, self.ga_config['ga_crossover_rate'])
            mutated = self._mutation(offspring, self.ga_config['ga_mutation_rate'])
            
            population = mutated
            
            current_best = min(population, key=evaluator)
            current_cost = evaluator(current_best)
            
            if current_cost < best_cost:
                best_path = current_best.copy()
                best_cost = current_cost
            
            if generation % 10 == 0:
                print(f"GA迭代 {generation}: 当前最优成本 = {best_cost:.2f}")
        
        return best_path
    
    def _simulated_annealing(self, initial_path: List[int], evaluator: Callable[[List[int]], float]) -> List[int]:
        current_path = initial_path.copy()
        current_cost = evaluator(current_path)
        
        best_path = current_path.copy()
        best_cost = current_cost
        
        temp = self.sa_config['sa_initial_temp']
        
        iteration = 0
        while temp > self.sa_config['sa_final_temp']:
            for _ in range(self.sa_config['sa_iterations_per_temp']):
                neighbor = self._perturb(current_path)
                neighbor_cost = evaluator(neighbor)
                
                delta = neighbor_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_path = neighbor
                    current_cost = neighbor_cost
                    
                    if current_cost < best_cost:
                        best_path = current_path.copy()
                        best_cost = current_cost
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"SA温度 {temp:.2f}: 当前最优成本 = {best_cost:.2f}")
            
            temp *= self.sa_config['sa_cooling_rate']
        
        return best_path
    
    def _parallel_hybrid(self, initial_path: List[int], evaluator: Callable[[List[int]], float]) -> List[int]:
        ga_path = self._genetic_algorithm(initial_path, evaluator)
        sa_path = self._simulated_annealing(initial_path, evaluator)
        
        costs = [evaluator(ga_path), evaluator(sa_path)]
        if costs[0] < costs[1]:
            return ga_path
        else:
            return sa_path
    
    def _local_search(self, path: List[int], evaluator: Callable[[List[int]], float]) -> List[int]:
        improved = True
        current_path = path.copy()
        current_cost = evaluator(current_path)
        
        while improved:
            improved = False
            n = len(current_path)
            
            for i in range(n - 1):
                for j in range(i + 2, n):
                    new_path = current_path[:i + 1] + current_path[i + 1:j + 1][::-1] + current_path[j + 1:]
                    new_cost = evaluator(new_path)
                    
                    if new_cost < current_cost:
                        current_path = new_path
                        current_cost = new_cost
                        improved = True
        
        return current_path
    
    def _initialize_population(self, initial_path: List[int], size: int) -> List[List[int]]:
        population = [initial_path.copy()]
        n = len(initial_path)
        
        for _ in range(size - 1):
            individual = initial_path.copy()
            random.shuffle(individual)
            population.append(individual)
        
        return population
    
    def _selection(self, population: List[List[int]], fitness: List[float], tournament_size: int) -> List[List[int]]:
        selected = []
        
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, parents: List[List[int]], crossover_rate: float) -> List[List[int]]:
        offspring = []
        n = len(parents)
        
        for i in range(0, n, 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % n]
            
            if random.random() < crossover_rate:
                child1, child2 = self._order_crossover(parent1, parent2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            offspring.append(child1)
            offspring.append(child2)
        
        return offspring[:n]
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> tuple:
        n = len(parent1)
        start = random.randint(0, n - 1)
        end = random.randint(start, n - 1)
        
        child1 = [-1] * n
        child2 = [-1] * n
        
        child1[start:end + 1] = parent1[start:end + 1]
        child2[start:end + 1] = parent2[start:end + 1]
        
        self._fill_remaining(child1, parent2, start, end)
        self._fill_remaining(child2, parent1, start, end)
        
        return child1, child2
    
    def _fill_remaining(self, child: List[int], parent: List[int], start: int, end: int):
        n = len(child)
        child_set = set(x for x in child if x != -1)
        
        parent_ptr = (end + 1) % n
        child_ptr = (end + 1) % n
        
        while child_ptr != start:
            if parent[parent_ptr] not in child_set:
                child[child_ptr] = parent[parent_ptr]
                child_ptr = (child_ptr + 1) % n
            parent_ptr = (parent_ptr + 1) % n
    
    def _mutation(self, population: List[List[int]], mutation_rate: float) -> List[List[int]]:
        mutated = []
        
        for individual in population:
            if random.random() < mutation_rate:
                mutated_individual = self._perturb(individual)
                mutated.append(mutated_individual)
            else:
                mutated.append(individual.copy())
        
        return mutated
    
    def _perturb(self, path: List[int]) -> List[int]:
        perturb_type = random.choice(['swap', 'reverse', 'insert'])
        new_path = path.copy()
        n = len(new_path)
        # 如果节点不足以进行扰动，直接返回拷贝
        if n < 2:
            return new_path

        if perturb_type == 'swap':
            i, j = random.sample(range(n), 2)
            new_path[i], new_path[j] = new_path[j], new_path[i]
        
        elif perturb_type == 'reverse':
            i, j = sorted(random.sample(range(n), 2))
            new_path[i:j + 1] = new_path[i:j + 1][::-1]
        
        elif perturb_type == 'insert':
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            item = new_path.pop(i)
            new_path.insert(j, item)
        
        return new_path
