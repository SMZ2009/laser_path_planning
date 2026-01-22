from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import numpy as np
import os

if TYPE_CHECKING:
    try:
        from svgpathtools import Path
    except ImportError:
        Path = None
else:
    try:
        from svgpathtools import parse_path, Path, svg2paths2
    except ImportError:
        print("警告: svgpathtools未安装, SVG解析功能将不可用")
        Path = None
        parse_path = None
        svg2paths2 = None


@dataclass
class Point:
    x: float
    y: float
    
    def __array__(self):
        return np.array([self.x, self.y])
    
    def to_tuple(self):
        return (self.x, self.y)


class SVGParser:
    def __init__(self, bezier_samples: int = 20, min_length_samples: int = 10):
        self.bezier_samples = bezier_samples
        self.min_length_samples = min_length_samples
    
    def parse(self, svg_path: str) -> List[Point]:
        if parse_path is None:
            raise ImportError("svgpathtools库未安装，请运行: pip install svgpathtools")
        
        points = []
        # 如果传入的是svg文件路径，使用 svg2paths2 读取所有 path 元素
        if os.path.exists(svg_path) and svg_path.lower().endswith('.svg'):
            if svg2paths2 is None:
                raise ImportError("svgpathtools缺少 svg2paths2: 请安装兼容版本的 svgpathtools")

            paths, attributes, svg_att = svg2paths2(svg_path)
            for p in paths:
                segment_points = self._extract_points_from_path(p)
                points.extend(segment_points)
        else:
            # 假设传入的是单个 path 数据字符串
            path_obj = parse_path(svg_path)
            if not isinstance(path_obj, list):
                path_obj = [path_obj]
            for path in path_obj:
                segment_points = self._extract_points_from_path(path)
                points.extend(segment_points)
        
        return points
    
    def _extract_points_from_path(self, path) -> List[Point]:
        points = []
        
        for segment in path:
            if hasattr(segment, 'length'):
                seg_length = segment.length()
                
                if seg_length == 0:
                    continue
                
                adaptive_samples = self._calculate_adaptive_samples(seg_length)
                
                if hasattr(segment, 'point'):
                    for t in np.linspace(0, 1, adaptive_samples):
                        p = segment.point(t)
                        points.append(Point(float(p.real), float(p.imag)))
                elif hasattr(segment, 'start') and hasattr(segment, 'end'):
                    points.append(Point(float(segment.start.real), float(segment.start.imag)))
                    points.append(Point(float(segment.end.real), float(segment.end.imag)))
        
        return self._deduplicate_points(points)
    
    def _calculate_adaptive_samples(self, length: float) -> int:
        if length < 5:
            return 2
        elif length < 20:
            return max(self.min_length_samples, int(length / 2))
        else:
            return min(self.bezier_samples, max(self.min_length_samples, int(length)))
    
    def _deduplicate_points(self, points: List[Point], tolerance: float = 0.01) -> List[Point]:
        if not points:
            return points
        
        unique_points = [points[0]]
        
        for point in points[1:]:
            is_duplicate = False
            for existing in unique_points:
                if abs(point.x - existing.x) < tolerance and abs(point.y - existing.y) < tolerance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_points.append(point)
        
        return unique_points
    
    def generate_test_svg(self, output_path: str, patterns: list = None):
        import xml.etree.ElementTree as ET
        
        if patterns is None:
            patterns = [
                ('circle', {'cx': '100', 'cy': '100', 'r': '30'}),
                ('rect', {'x': '200', 'y': '50', 'width': '60', 'height': '40'}),
                ('circle', {'cx': '300', 'cy': '150', 'r': '20'}),
                ('rect', {'x': '50', 'y': '200', 'width': '80', 'height': '60'}),
                ('circle', {'cx': '200', 'cy': '250', 'r': '25'}),
                ('circle', {'cx': '350', 'cy': '250', 'r': '15'}),
            ]
        
        svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg", 
                        width="400", height="400")
        
        for tag, attrs in patterns:
            elem = ET.SubElement(svg, tag)
            for key, value in attrs.items():
                elem.set(key, value)
        
        tree = ET.ElementTree(svg)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"测试SVG文件已生成: {output_path}")
