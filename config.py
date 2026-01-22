import yaml
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Config:
    v1: float
    v2: float
    L: float
    input_svg: str
    output_image: str
    output_json: str
    bezier_samples_per_curve: int
    min_curve_length_sample: int
    grid_safety_factor: float
    target_coverage: float
    max_kmeans_iterations: int
    kmeans_convergence_tol: float
    kmeans_convergence_threshold: float
    ga_population_size: int
    ga_generations: int
    ga_crossover_rate: float
    ga_mutation_rate: float
    ga_tournament_size: int
    sa_initial_temp: float
    sa_cooling_rate: float
    sa_final_temp: float
    sa_iterations_per_temp: int
    use_3opt: bool
    tsp_max_iterations: int
    figure_dpi: int
    figure_size: Tuple[int, int]
    marker_size: float
    alpha: float


def load_config(config_path: str) -> Config:
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(
        v1=config_dict['v1'],
        v2=config_dict['v2'],
        L=config_dict['L'],
        input_svg=config_dict['input_svg'],
        output_image=config_dict['output_image'],
        output_json=config_dict['output_json'],
        bezier_samples_per_curve=config_dict['bezier_samples_per_curve'],
        min_curve_length_sample=config_dict['min_curve_length_sample'],
        grid_safety_factor=config_dict['grid_safety_factor'],
        target_coverage=config_dict['target_coverage'],
        max_kmeans_iterations=config_dict['max_kmeans_iterations'],
        kmeans_convergence_tol=config_dict['kmeans_convergence_tol'],
        kmeans_convergence_threshold=config_dict['kmeans_convergence_threshold'],
        ga_population_size=config_dict['ga_population_size'],
        ga_generations=config_dict['ga_generations'],
        ga_crossover_rate=config_dict['ga_crossover_rate'],
        ga_mutation_rate=config_dict['ga_mutation_rate'],
        ga_tournament_size=config_dict['ga_tournament_size'],
        sa_initial_temp=config_dict['sa_initial_temp'],
        sa_cooling_rate=config_dict['sa_cooling_rate'],
        sa_final_temp=config_dict['sa_final_temp'],
        sa_iterations_per_temp=config_dict['sa_iterations_per_temp'],
        use_3opt=config_dict['use_3opt'],
        tsp_max_iterations=config_dict['tsp_max_iterations'],
        figure_dpi=config_dict['figure_dpi'],
        figure_size=tuple(config_dict['figure_size']),
        marker_size=config_dict['marker_size'],
        alpha=config_dict['alpha']
    )
