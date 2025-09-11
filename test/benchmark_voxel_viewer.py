#!/usr/bin/env python3
"""Benchmark script for voxel_viewer performance optimization."""

import time
import numpy as np
import open3d as o3d
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import sys
sys.path.append('/home/ryo/image_compressor_ws/install/voxel_viewer/lib/python3.10/site-packages')
from voxel_viewer.marker_to_open3d import MarkerToOpen3D


class VoxelViewerBenchmark:
    """Benchmark for voxel viewer operations."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.converter = MarkerToOpen3D()
        self.timings = {}
        
    def create_test_marker_array(self, num_points=100000):
        """Create a test MarkerArray with specified number of points."""
        marker_array = MarkerArray()
        marker = Marker()
        marker.type = Marker.CUBE_LIST
        marker.scale.x = 0.09
        marker.scale.y = 0.09
        marker.scale.z = 0.09
        
        # Create random points
        np.random.seed(42)
        for i in range(num_points):
            point = Point()
            point.x = np.random.uniform(0, 100)
            point.y = np.random.uniform(0, 100)
            point.z = np.random.uniform(0, 10)
            marker.points.append(point)
            
            color = ColorRGBA()
            color.r = 0.5
            color.g = 0.5
            color.b = 0.5
            color.a = 1.0
            marker.colors.append(color)
        
        marker_array.markers.append(marker)
        return marker_array
    
    def benchmark_marker_to_pointcloud(self, marker_array):
        """Benchmark MarkerArray to PointCloud conversion."""
        start = time.perf_counter()
        pcd, scale = self.converter.marker_array_to_pointcloud(marker_array)
        end = time.perf_counter()
        
        self.timings['marker_to_pointcloud'] = (end - start) * 1000
        return pcd, scale
    
    def benchmark_voxel_rounding(self, points, voxel_size):
        """Benchmark voxel rounding operation."""
        start = time.perf_counter()
        
        # Original method
        rounded_points = np.round(points / voxel_size) * voxel_size
        unique_points = np.unique(rounded_points, axis=0)
        
        end = time.perf_counter()
        self.timings['voxel_rounding'] = (end - start) * 1000
        return unique_points
    
    def benchmark_point_comparison(self, points1, points2, voxel_size):
        """Benchmark point cloud comparison."""
        start = time.perf_counter()
        
        # Round both point sets
        rounded1 = np.round(points1 / voxel_size) * voxel_size
        rounded2 = np.round(points2 / voxel_size) * voxel_size
        
        # Convert to sets for comparison
        set1 = set(map(tuple, rounded1))
        set2 = set(map(tuple, rounded2))
        
        # Find matches and mismatches
        matches = set1.intersection(set2)
        only1 = set1 - set2
        only2 = set2 - set1
        
        end = time.perf_counter()
        self.timings['point_comparison'] = (end - start) * 1000
        return len(matches), len(only1), len(only2)
    
    def benchmark_voxel_grid_creation(self, pcd, voxel_size):
        """Benchmark VoxelGrid creation from PointCloud."""
        start = time.perf_counter()
        
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=voxel_size * 0.9
        )
        
        end = time.perf_counter()
        self.timings['voxel_grid_creation'] = (end - start) * 1000
        return voxel_grid
    
    def run_benchmark(self):
        """Run complete benchmark suite."""
        print("=" * 60)
        print("VOXEL VIEWER PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        # Test with different data sizes
        test_sizes = [10000, 50000, 100000, 500000]
        
        for size in test_sizes:
            print(f"\n--- Testing with {size} points ---")
            
            # Create test data
            marker_array1 = self.create_test_marker_array(size)
            marker_array2 = self.create_test_marker_array(int(size * 0.8))  # 80% overlap
            
            # 1. MarkerArray to PointCloud conversion
            pcd1, scale1 = self.benchmark_marker_to_pointcloud(marker_array1)
            pcd2, scale2 = self.benchmark_marker_to_pointcloud(marker_array2)
            
            points1 = np.asarray(pcd1.points)
            points2 = np.asarray(pcd2.points)
            
            # 2. Voxel rounding
            unique1 = self.benchmark_voxel_rounding(points1, scale1)
            
            # 3. Point comparison
            matches, only1, only2 = self.benchmark_point_comparison(
                points1, points2, (scale1 + scale2) / 2
            )
            
            # 4. VoxelGrid creation
            voxel_grid = self.benchmark_voxel_grid_creation(pcd1, scale1)
            
            # Print results
            print(f"MarkerArray → PointCloud: {self.timings['marker_to_pointcloud']:.2f} ms")
            print(f"Voxel rounding: {self.timings['voxel_rounding']:.2f} ms")
            print(f"Point comparison: {self.timings['point_comparison']:.2f} ms")
            print(f"VoxelGrid creation: {self.timings['voxel_grid_creation']:.2f} ms")
            print(f"Total: {sum(self.timings.values()):.2f} ms")
            print(f"Comparison results: {matches} matches, {only1} + {only2} mismatches")
        
        print("\n" + "=" * 60)
        print("BOTTLENECK ANALYSIS")
        print("=" * 60)
        
        # Identify bottlenecks
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        total_time = sum(self.timings.values())
        
        for operation, timing in sorted_timings:
            percentage = (timing / total_time) * 100
            print(f"{operation}: {timing:.2f} ms ({percentage:.1f}%)")
        
        return self.timings


if __name__ == '__main__':
    benchmark = VoxelViewerBenchmark()
    benchmark.run_benchmark()