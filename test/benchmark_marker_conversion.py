#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import time
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxel_viewer.fast_marker_to_open3d import FastMarkerToOpen3D

def create_test_marker_array(num_points=100000):
    """Create a test MarkerArray with CUBE_LIST markers"""
    marker_array = MarkerArray()
    points_per_marker = 10000
    num_markers = num_points // points_per_marker
    
    for i in range(num_markers):
        marker = Marker()
        marker.type = Marker.CUBE_LIST
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.5
        
        # Add random points
        for j in range(points_per_marker):
            point = Point()
            point.x = np.random.uniform(-10, 10)
            point.y = np.random.uniform(-10, 10)
            point.z = np.random.uniform(-10, 10)
            marker.points.append(point)
        
        marker_array.markers.append(marker)
    
    return marker_array

def marker_to_pointcloud_original(marker_array):
    """Original implementation from voxel_viewer.py"""
    points = []
    scale = 0.05  # Default scale
    
    for marker in marker_array.markers:
        if marker.type == Marker.CUBE_LIST:
            scale = marker.scale.x
            for point in marker.points:
                points.append([point.x, point.y, point.z])
    
    if points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        return pcd, scale
    return None, scale

def benchmark_method(method, marker_array, name, iterations=10):
    """Benchmark a single method"""
    times = []
    
    # Warmup
    for _ in range(2):
        result = method(marker_array)
    
    # Actual benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        result = method(marker_array)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time

def main():
    # Test with different sizes
    sizes = [10000, 50000, 100000, 500000]
    
    print("=" * 80)
    print("MarkerArray to PointCloud Conversion Benchmark")
    print("=" * 80)
    
    converter = FastMarkerToOpen3D()
    
    for size in sizes:
        print(f"\nTesting with {size:,} points:")
        print("-" * 40)
        
        marker_array = create_test_marker_array(size)
        
        # Benchmark original method
        orig_avg, orig_std = benchmark_method(marker_to_pointcloud_original, marker_array, "Original")
        print(f"{'Original':20s}: {orig_avg:8.2f} ± {orig_std:6.2f} ms")
        
        # Benchmark optimized method
        def optimized_wrapper(ma):
            pcd, scale, time_ms = converter.marker_array_to_pointcloud_optimized(ma)
            return pcd, scale
        
        opt_avg, opt_std = benchmark_method(optimized_wrapper, marker_array, "Optimized")
        print(f"{'Optimized':20s}: {opt_avg:8.2f} ± {opt_std:6.2f} ms")
        
        # Calculate speedup
        speedup = orig_avg / opt_avg
        time_saved = orig_avg - opt_avg
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Time saved: {time_saved:.2f} ms ({(time_saved/orig_avg)*100:.1f}% reduction)")
    
    # Test individual marker conversion methods
    print("\n" + "=" * 80)
    print("Individual Marker Conversion Methods (100k points)")
    print("=" * 80)
    
    marker = Marker()
    marker.type = Marker.CUBE_LIST
    marker.scale.x = 0.05
    marker.color.r = 0.5
    marker.color.g = 0.5
    marker.color.b = 1.0
    marker.color.a = 0.5
    
    for i in range(100000):
        point = Point()
        point.x = np.random.uniform(-10, 10)
        point.y = np.random.uniform(-10, 10)
        point.z = np.random.uniform(-10, 10)
        marker.points.append(point)
    
    # Warmup
    converter.marker_to_pointcloud_original(marker)
    converter.marker_to_pointcloud_vectorized(marker)
    converter.marker_to_pointcloud_memmap(marker)
    
    # Benchmark
    results = []
    methods = [
        ("Original", converter.marker_to_pointcloud_original),
        ("Vectorized", converter.marker_to_pointcloud_vectorized),
        ("Chunked", converter.marker_to_pointcloud_memmap)
    ]
    
    for name, method in methods:
        times = []
        for _ in range(5):
            _, _, time_ms = method(marker)
            times.append(time_ms)
        avg_time = np.mean(times)
        results.append((name, avg_time))
        print(f"{name:15s}: {avg_time:8.2f} ms")
    
    # Show speedup
    baseline = results[0][1]
    print("\nSpeedup vs Original:")
    for name, avg_time in results:
        speedup = baseline / avg_time
        print(f"{name:15s}: {speedup:6.2f}x")
    
    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("-" * 80)
    
    best_time = min(r[1] for r in results)
    best_method = [r[0] for r in results if r[1] == best_time][0]
    
    print(f"Best single marker method: {best_method}")
    print(f"For MarkerArray: Use marker_array_to_pointcloud_optimized")
    print(f"Expected speedup: {baseline/best_time:.2f}x for single markers")
    print(f"Expected speedup: {orig_avg/opt_avg:.2f}x for MarkerArrays")

if __name__ == "__main__":
    main()