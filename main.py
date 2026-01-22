import yaml
import json
from config import Config, load_config
from svg_parser import SVGParser
from adaptive_grid import AdaptiveGrid
from kmeans_decrement import KmeansDecrement
from tsp_solver import TSPSolver
from hybrid_optimizer import HybridOptimizer
from time_evaluator import TimeEvaluator
from coordinator import Coordinator
from visualizer import Visualizer
from metrics import Metrics
import os


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    print("="*60)
    print("  大尺度激光加工系统协同路径规划")
    print("  自适应网格法引导的K递减K-means聚类 + 混合优化算法")
    print("="*60)
    
    config = load_config('config.yaml')
    print(f"\n系统参数:")
    print(f"  平台速度 v1: {config.v1} mm/s")
    print(f"  振镜速度 v2: {config.v2} mm/s")
    print(f"  覆盖区域边长 L: {config.L} mm")
    
    if not os.path.exists(config.input_svg):
        print(f"\n输入文件不存在: {config.input_svg}")
        print("正在生成测试SVG文件...")
        svg_parser = SVGParser()
        test_svg_path = os.path.join('data', 'test_pattern.svg')
        os.makedirs('data', exist_ok=True)
        svg_parser.generate_test_svg(test_svg_path)
        config.input_svg = test_svg_path
        print(f"测试SVG文件已生成: {test_svg_path}")
    
    print_section("1. SVG解析")
    svg_parser = SVGParser(config.bezier_samples_per_curve, config.min_curve_length_sample)
    points = svg_parser.parse(config.input_svg)
    print(f"解析完成，共 {len(points)} 个加工点")
    
    if len(points) == 0:
        print("错误: 未找到任何加工点！")
        return
    
    print_section("2. 自适应网格法")
    adaptive_grid = AdaptiveGrid(config.grid_safety_factor)
    initial_centers, K0 = adaptive_grid.generate(points, config.L)
    print(f"自适应网格生成初始K值: {K0}")
    
    print_section("3. K递减K-means聚类")
    kmeans = KmeansDecrement(
        L=config.L,
        target_coverage=config.target_coverage,
        max_iterations=config.max_kmeans_iterations,
        convergence_tol=config.kmeans_convergence_tol,
        convergence_threshold=config.kmeans_convergence_threshold
    )
    regions, final_K = kmeans.decrement_cluster(points, initial_centers, K0)
    print(f"最终区域数: {final_K}")
    
    print_section("4. TSP初始路径求解")
    centers = [r.center for r in regions]
    tsp_solver = TSPSolver(use_3opt=config.use_3opt, max_iterations=config.tsp_max_iterations)
    initial_path = tsp_solver.solve(centers)
    print(f"TSP初始路径长度: {sum(((centers[initial_path[i]][0] - centers[initial_path[i+1]][0])**2 + (centers[initial_path[i]][1] - centers[initial_path[i+1]][1])**2)**0.5 for i in range(len(initial_path)-1)):.2f} mm")
    print(f"初始路径: {initial_path}")
    
    print_section("5. 时间评估模型")
    time_evaluator = TimeEvaluator(regions, config.v1, config.v2, config.L)
    initial_time = time_evaluator.evaluate(initial_path)
    print(f"初始路径总耗时: {initial_time:.2f} s")
    
    print_section("6. 混合优化 (SA + GA + 局部搜索)")
    ga_config = {
        'ga_population_size': config.ga_population_size,
        'ga_generations': config.ga_generations,
        'ga_crossover_rate': config.ga_crossover_rate,
        'ga_mutation_rate': config.ga_mutation_rate,
        'ga_tournament_size': config.ga_tournament_size
    }
    sa_config = {
        'sa_initial_temp': config.sa_initial_temp,
        'sa_cooling_rate': config.sa_cooling_rate,
        'sa_final_temp': config.sa_final_temp,
        'sa_iterations_per_temp': config.sa_iterations_per_temp
    }
    
    hybrid_optimizer = HybridOptimizer(ga_config, sa_config)
    optimized_path = hybrid_optimizer.optimize(initial_path, time_evaluator.evaluate)
    optimized_time = time_evaluator.evaluate(optimized_path)
    
    improvement = (initial_time - optimized_time) / initial_time * 100
    print(f"优化后路径总耗时: {optimized_time:.2f} s")
    print(f"时间改进: {improvement:.2f}%")
    print(f"优化后路径: {optimized_path}")
    
    print_section("7. 生成时间线")
    timeline, total_time = time_evaluator.evaluate_with_timeline(optimized_path)
    print(f"总加工时间: {total_time:.2f} s")
    
    print_section("8. 协同调度")
    coordinator = Coordinator(regions)
    detailed_schedule = coordinator.generate_optimized_schedule(optimized_path)
    
    print_section("9. 计算指标")
    metrics = Metrics.calculate_all(points, regions, optimized_path, timeline, 
                                   config.v1, config.v2, config.L)
    
    print("\n关键指标:")
    print(f"  总加工点数: {metrics['total_points']}")
    print(f"  区域数量: {metrics['num_regions']}")
    print(f"  覆盖率: {metrics['coverage']*100:.2f}%")
    print(f"  平台路径长度: {metrics['platform_distance']:.2f} mm")
    print(f"  平台移动时间: {metrics['platform_time']:.2f} s")
    print(f"  激光扫描时间: {metrics['laser_time']:.2f} s")
    print(f"  等待时间: {metrics['wait_time']:.2f} s")
    print(f"  总加工时间: {metrics['total_time']:.2f} s")
    print(f"  时间利用率: {metrics['time_utilization']:.2f}%")
    print(f"  区域均衡度: {metrics['region_balance']:.2f}")
    
    print_section("10. 可视化")
    os.makedirs('results', exist_ok=True)
    
    visualizer = Visualizer(config)
    visualizer.plot_comprehensive_results(regions, optimized_path, timeline, 
                                        metrics, config.output_image)
    
    # 将指标以表格形式保存为 CSV（不再生成指标图片）
    metrics_output = config.output_image.replace('.png', '_metrics.csv')
    visualizer.save_metrics_table(metrics, metrics_output)
    
    print_section("11. 保存结果")
    results = {
        'config': {
            'v1': config.v1,
            'v2': config.v2,
            'L': config.L
        },
        'metrics': metrics,
        'platform_path': optimized_path,
        'timeline': timeline,
        'region_info': time_evaluator.get_region_info()
    }
    
    with open(config.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至 {config.output_json}")
    
    print("\n" + "="*60)
    print("  路径规划完成！")
    print("="*60)


if __name__ == '__main__':
    main()
