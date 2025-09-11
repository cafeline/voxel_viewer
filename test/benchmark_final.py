#!/usr/bin/env python3
"""Final benchmark comparing original vs optimized implementation."""

import numpy as np
import open3d as o3d
import time
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxel_viewer.voxel_viewer import VoxelViewerNode
from voxel_viewer.optimized_voxel_viewer import OptimizedVoxelViewerNode
from voxel_viewer.fast_marker_to_open3d import FastMarkerToOpen3D

def create_test_marker_arrays(num_points=100000):
    """Create test MarkerArrays for occupied and pattern."""
    occupied_array = MarkerArray()
    pattern_array = MarkerArray()
    
    points_per_marker = 10000
    num_markers = num_points // points_per_marker
    
    # Create occupied markers
    for i in range(num_markers):
        marker = Marker()
        marker.type = Marker.CUBE_LIST
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        for j in range(points_per_marker):
            point = Point()
            point.x = np.random.uniform(-10, 10)
            point.y = np.random.uniform(-10, 10)
            point.z = np.random.uniform(-10, 10)
            marker.points.append(point)
        
        occupied_array.markers.append(marker)
    
    # Create pattern markers (70% overlap with occupied)
    for i in range(num_markers):
        marker = Marker()
        marker.type = Marker.CUBE_LIST
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        for j in range(points_per_marker):
            point = Point()
            if j < points_per_marker * 0.7:
                # Use same points as occupied for overlap
                orig_point = occupied_array.markers[i].points[j]
                point.x = orig_point.x
                point.y = orig_point.y
                point.z = orig_point.z
            else:
                # Different points
                point.x = np.random.uniform(-10, 10)
                point.y = np.random.uniform(-10, 10)
                point.z = np.random.uniform(-10, 10)
            marker.points.append(point)
        
        pattern_array.markers.append(marker)
    
    return occupied_array, pattern_array

def benchmark_original_implementation(occupied_array, pattern_array):
    """Benchmark original implementation."""
    from voxel_viewer.marker_to_open3d import MarkerToOpen3D
    
    converter = MarkerToOpen3D()
    
    # Measure marker to pointcloud conversion
    start = time.perf_counter()
    occupied_pcd, occupied_scale = converter.marker_array_to_pointcloud(occupied_array)
    pattern_pcd, pattern_scale = converter.marker_array_to_pointcloud(pattern_array)
    marker_time = (time.perf_counter() - start) * 1000
    
    occupied_points = np.asarray(occupied_pcd.points)
    pattern_points = np.asarray(pattern_pcd.points)
    
    # Measure voxel rounding
    start = time.perf_counter()
    voxel_size = (occupied_scale + pattern_scale) / 2.0
    
    def round_to_voxel(points, voxel_size):
        """Original round to voxel implementation."""
        return np.round(points / voxel_size) * voxel_size
    
    occupied_rounded = round_to_voxel(occupied_points, voxel_size)
    pattern_rounded = round_to_voxel(pattern_points, voxel_size)
    rounding_time = (time.perf_counter() - start) * 1000
    
    # Measure comparison
    start = time.perf_counter()
    occupied_set = set(map(tuple, occupied_rounded))
    pattern_set = set(map(tuple, pattern_rounded))
    
    matches = occupied_set.intersection(pattern_set)
    occupied_only = occupied_set - pattern_set
    pattern_only = pattern_set - occupied_set
    comparison_time = (time.perf_counter() - start) * 1000
    
    # Measure voxel grid creation
    start = time.perf_counter()
    all_points = []
    all_colors = []
    
    green_color = [0.0, 0.6, 0.0]
    red_color = [0.6, 0.0, 0.0]
    
    for point in matches:
        all_points.append(point)
        all_colors.append(green_color)
    
    for point in occupied_only:
        all_points.append(point)
        all_colors.append(red_color)
    
    for point in pattern_only:
        all_points.append(point)
        all_colors.append(red_color)
    
    if all_points:
        comparison_pcd = o3d.geometry.PointCloud()
        comparison_pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
        comparison_pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            comparison_pcd,
            voxel_size=voxel_size * 0.9
        )
    
    voxel_grid_time = (time.perf_counter() - start) * 1000
    
    total_time = marker_time + rounding_time + comparison_time + voxel_grid_time
    
    return {
        'marker_to_pointcloud': marker_time,
        'voxel_rounding': rounding_time,
        'point_comparison': comparison_time,
        'voxel_grid_creation': voxel_grid_time,
        'total': total_time,
        'matches': len(matches),
        'mismatches': len(occupied_only) + len(pattern_only)
    }

def benchmark_optimized_implementation(occupied_array, pattern_array):
    """Benchmark optimized implementation."""
    converter = FastMarkerToOpen3D()
    
    # Measure optimized marker to pointcloud conversion
    start = time.perf_counter()
    occupied_pcd, occupied_scale, _ = converter.marker_array_to_pointcloud_optimized(occupied_array)
    pattern_pcd, pattern_scale, _ = converter.marker_array_to_pointcloud_optimized(pattern_array)
    marker_time = (time.perf_counter() - start) * 1000
    
    occupied_points = np.asarray(occupied_pcd.points)
    pattern_points = np.asarray(pattern_pcd.points)
    
    # Measure optimized voxel rounding
    start = time.perf_counter()
    voxel_size = (occupied_scale + pattern_scale) / 2.0
    
    def round_to_voxel_fast(points, voxel_size):
        """Optimized in-place voxel rounding."""
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)
        else:
            points = points.astype(np.float32, copy=True)
        
        points /= voxel_size
        np.round(points, out=points)
        points *= voxel_size
        
        return points
    
    occupied_rounded = round_to_voxel_fast(occupied_points, voxel_size)
    pattern_rounded = round_to_voxel_fast(pattern_points, voxel_size)
    rounding_time = (time.perf_counter() - start) * 1000
    
    # Measure comparison (same as original, already optimal)
    start = time.perf_counter()
    occupied_set = set(map(tuple, occupied_rounded))
    pattern_set = set(map(tuple, pattern_rounded))
    
    matches = occupied_set.intersection(pattern_set)
    occupied_only = occupied_set - pattern_set
    pattern_only = pattern_set - occupied_set
    comparison_time = (time.perf_counter() - start) * 1000
    
    # Measure voxel grid creation
    start = time.perf_counter()
    all_points = []
    all_colors = []
    
    green_color = [0.0, 0.6, 0.0]
    red_color = [0.6, 0.0, 0.0]
    
    for point in matches:
        all_points.append(point)
        all_colors.append(green_color)
    
    for point in occupied_only:
        all_points.append(point)
        all_colors.append(red_color)
    
    for point in pattern_only:
        all_points.append(point)
        all_colors.append(red_color)
    
    if all_points:
        comparison_pcd = o3d.geometry.PointCloud()
        comparison_pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
        comparison_pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            comparison_pcd,
            voxel_size=voxel_size * 0.9
        )
    
    voxel_grid_time = (time.perf_counter() - start) * 1000
    
    total_time = marker_time + rounding_time + comparison_time + voxel_grid_time
    
    return {
        'marker_to_pointcloud': marker_time,
        'voxel_rounding': rounding_time,
        'point_comparison': comparison_time,
        'voxel_grid_creation': voxel_grid_time,
        'total': total_time,
        'matches': len(matches),
        'mismatches': len(occupied_only) + len(pattern_only)
    }

def main():
    """Run final benchmark."""
    print("=" * 80)
    print("FINAL PERFORMANCE BENCHMARK: Original vs Optimized")
    print("=" * 80)
    
    sizes = [10000, 50000, 100000, 500000]
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {size:,} points per dataset")
        print(f"{'='*60}")
        
        # Create test data
        occupied_array, pattern_array = create_test_marker_arrays(size)
        
        # Warmup
        benchmark_original_implementation(occupied_array, pattern_array)
        benchmark_optimized_implementation(occupied_array, pattern_array)
        
        # Benchmark original
        print("\n--- Original Implementation ---")
        orig_results = []
        for i in range(3):
            result = benchmark_original_implementation(occupied_array, pattern_array)
            orig_results.append(result)
            if i == 0:  # Print first result
                print(f"  marker_to_pointcloud: {result['marker_to_pointcloud']:8.2f} ms")
                print(f"  voxel_rounding:       {result['voxel_rounding']:8.2f} ms")
                print(f"  point_comparison:     {result['point_comparison']:8.2f} ms")
                print(f"  voxel_grid_creation:  {result['voxel_grid_creation']:8.2f} ms")
                print(f"  TOTAL:                {result['total']:8.2f} ms")
        
        # Average original results
        orig_avg = {
            key: np.mean([r[key] for r in orig_results])
            for key in orig_results[0].keys()
        }
        
        # Benchmark optimized
        print("\n--- Optimized Implementation ---")
        opt_results = []
        for i in range(3):
            result = benchmark_optimized_implementation(occupied_array, pattern_array)
            opt_results.append(result)
            if i == 0:  # Print first result
                print(f"  marker_to_pointcloud: {result['marker_to_pointcloud']:8.2f} ms")
                print(f"  voxel_rounding:       {result['voxel_rounding']:8.2f} ms")
                print(f"  point_comparison:     {result['point_comparison']:8.2f} ms")
                print(f"  voxel_grid_creation:  {result['voxel_grid_creation']:8.2f} ms")
                print(f"  TOTAL:                {result['total']:8.2f} ms")
        
        # Average optimized results
        opt_avg = {
            key: np.mean([r[key] for r in opt_results])
            for key in opt_results[0].keys()
        }
        
        # Calculate speedup
        print("\n--- Speedup Analysis ---")
        operations = ['marker_to_pointcloud', 'voxel_rounding', 'point_comparison', 
                     'voxel_grid_creation', 'total']
        
        for op in operations:
            speedup = orig_avg[op] / opt_avg[op] if opt_avg[op] > 0 else float('inf')
            time_saved = orig_avg[op] - opt_avg[op]
            print(f"  {op:20s}: {speedup:6.2f}x speedup ({time_saved:8.2f} ms saved)")
        
        # Verify correctness
        print(f"\n--- Correctness Check ---")
        print(f"  Matches:    Original={orig_avg['matches']:.0f}, Optimized={opt_avg['matches']:.0f}")
        print(f"  Mismatches: Original={orig_avg['mismatches']:.0f}, Optimized={opt_avg['mismatches']:.0f}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("PERFORMANCE IMPROVEMENT SUMMARY")
    print("=" * 80)
    
    # Test with largest dataset for final numbers
    occupied_array, pattern_array = create_test_marker_arrays(500000)
    
    orig_times = []
    opt_times = []
    
    for _ in range(5):
        orig_result = benchmark_original_implementation(occupied_array, pattern_array)
        opt_result = benchmark_optimized_implementation(occupied_array, pattern_array)
        orig_times.append(orig_result['total'])
        opt_times.append(opt_result['total'])
    
    orig_avg_total = np.mean(orig_times)
    opt_avg_total = np.mean(opt_times)
    
    final_speedup = orig_avg_total / opt_avg_total
    total_time_saved = orig_avg_total - opt_avg_total
    
    print(f"\nFor 500,000 points per dataset (1,000,000 total):")
    print(f"  Original implementation: {orig_avg_total:8.2f} ms")
    print(f"  Optimized implementation: {opt_avg_total:8.2f} ms")
    print(f"  \n  🚀 TOTAL SPEEDUP: {final_speedup:.2f}x")
    print(f"  ⏱️  TIME SAVED: {total_time_saved:.2f} ms ({(total_time_saved/orig_avg_total)*100:.1f}% reduction)")
    
    print("\n--- Key Optimizations Applied ---")
    print("  1. marker_to_pointcloud: Pre-allocation and vectorized extraction")
    print("  2. voxel_rounding: In-place operations with float32")
    print("  3. point_comparison: Original set-based approach (already optimal)")
    print("  4. Overall: Reduced memory allocations and improved cache usage")

if __name__ == "__main__":
    main()