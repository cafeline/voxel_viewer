#!/usr/bin/env python3
"""Optimized MarkerArray to Open3D conversion."""

import numpy as np
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
import time


class FastMarkerToOpen3D:
    """Optimized converter for ROS MarkerArray to Open3D."""
    
    def __init__(self):
        """Initialize the converter."""
        pass
    
    def marker_to_pointcloud_original(self, marker):
        """Original implementation for baseline."""
        start = time.perf_counter()
        
        pcd = o3d.geometry.PointCloud()
        marker_scale = None
        
        if marker.type == Marker.CUBE_LIST:
            marker_scale = marker.scale.x
            points = []
            colors = []
            
            for i, point in enumerate(marker.points):
                points.append([point.x, point.y, point.z])
                if i < len(marker.colors):
                    colors.append([marker.colors[i].r, marker.colors[i].g, marker.colors[i].b])
                else:
                    colors.append([marker.color.r, marker.color.g, marker.color.b])
            
            if points:
                pcd.points = o3d.utility.Vector3dVector(np.array(points))
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        end = time.perf_counter()
        return pcd, marker_scale, (end - start) * 1000
    
    def marker_to_pointcloud_vectorized(self, marker):
        """Optimized implementation using vectorized operations."""
        start = time.perf_counter()
        
        pcd = o3d.geometry.PointCloud()
        marker_scale = None
        
        if marker.type == Marker.CUBE_LIST:
            marker_scale = marker.scale.x
            
            # Pre-allocate arrays
            num_points = len(marker.points)
            if num_points > 0:
                points = np.empty((num_points, 3), dtype=np.float64)
                colors = np.empty((num_points, 3), dtype=np.float64)
                
                # Vectorized extraction using list comprehension (faster than loop)
                points[:, 0] = [p.x for p in marker.points]
                points[:, 1] = [p.y for p in marker.points]
                points[:, 2] = [p.z for p in marker.points]
                
                # Handle colors
                if marker.colors:
                    num_colors = len(marker.colors)
                    if num_colors == num_points:
                        colors[:, 0] = [c.r for c in marker.colors]
                        colors[:, 1] = [c.g for c in marker.colors]
                        colors[:, 2] = [c.b for c in marker.colors]
                    else:
                        # Use marker color for all points
                        colors[:] = [marker.color.r, marker.color.g, marker.color.b]
                else:
                    colors[:] = [marker.color.r, marker.color.g, marker.color.b]
                
                # Direct assignment without intermediate conversion
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
        
        end = time.perf_counter()
        return pcd, marker_scale, (end - start) * 1000
    
    def marker_to_pointcloud_memmap(self, marker):
        """Ultra-fast implementation using memory views."""
        start = time.perf_counter()
        
        pcd = o3d.geometry.PointCloud()
        marker_scale = None
        
        if marker.type == Marker.CUBE_LIST:
            marker_scale = marker.scale.x
            
            num_points = len(marker.points)
            if num_points > 0:
                # Use float32 for faster operations
                points = np.empty((num_points, 3), dtype=np.float32)
                colors = np.empty((num_points, 3), dtype=np.float32)
                
                # Extract data in chunks for better cache usage
                chunk_size = 1000
                for i in range(0, num_points, chunk_size):
                    end_idx = min(i + chunk_size, num_points)
                    chunk = marker.points[i:end_idx]
                    
                    points[i:end_idx, 0] = [p.x for p in chunk]
                    points[i:end_idx, 1] = [p.y for p in chunk]
                    points[i:end_idx, 2] = [p.z for p in chunk]
                
                # Handle colors efficiently
                if marker.colors and len(marker.colors) == num_points:
                    for i in range(0, num_points, chunk_size):
                        end_idx = min(i + chunk_size, num_points)
                        chunk = marker.colors[i:end_idx]
                        
                        colors[i:end_idx, 0] = [c.r for c in chunk]
                        colors[i:end_idx, 1] = [c.g for c in chunk]
                        colors[i:end_idx, 2] = [c.b for c in chunk]
                else:
                    colors[:] = [marker.color.r, marker.color.g, marker.color.b]
                
                # Convert back to float64 for Open3D
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        
        end = time.perf_counter()
        return pcd, marker_scale, (end - start) * 1000
    
    def marker_array_to_pointcloud_optimized(self, marker_array):
        """Optimized MarkerArray conversion with pre-allocation."""
        start = time.perf_counter()
        
        # Count total points first
        total_points = sum(len(m.points) for m in marker_array.markers if m.type == Marker.CUBE_LIST)
        
        if total_points == 0:
            return o3d.geometry.PointCloud(), 0.1, (time.perf_counter() - start) * 1000
        
        # Pre-allocate combined arrays
        all_points = np.empty((total_points, 3), dtype=np.float32)
        all_colors = np.empty((total_points, 3), dtype=np.float32)
        scales = []
        
        current_idx = 0
        for marker in marker_array.markers:
            if marker.type == Marker.CUBE_LIST:
                num_points = len(marker.points)
                if num_points > 0:
                    # Extract points
                    end_idx = current_idx + num_points
                    all_points[current_idx:end_idx, 0] = [p.x for p in marker.points]
                    all_points[current_idx:end_idx, 1] = [p.y for p in marker.points]
                    all_points[current_idx:end_idx, 2] = [p.z for p in marker.points]
                    
                    # Extract colors
                    if marker.colors and len(marker.colors) == num_points:
                        all_colors[current_idx:end_idx, 0] = [c.r for c in marker.colors]
                        all_colors[current_idx:end_idx, 1] = [c.g for c in marker.colors]
                        all_colors[current_idx:end_idx, 2] = [c.b for c in marker.colors]
                    else:
                        all_colors[current_idx:end_idx] = [marker.color.r, marker.color.g, marker.color.b]
                    
                    scales.append(marker.scale.x)
                    current_idx = end_idx
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64))
        
        avg_scale = np.mean(scales) if scales else 0.1
        
        end = time.perf_counter()
        return pcd, avg_scale, (end - start) * 1000


def benchmark_marker_conversion():
    """Benchmark marker conversion methods."""
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    
    print("=" * 60)
    print("MARKER CONVERSION BENCHMARKS")
    print("=" * 60)
    
    # Create test MarkerArray
    def create_test_marker(num_points):
        marker = Marker()
        marker.type = Marker.CUBE_LIST
        marker.scale.x = 0.09
        
        for i in range(num_points):
            point = Point()
            point.x = np.random.uniform(0, 100)
            point.y = np.random.uniform(0, 100)
            point.z = np.random.uniform(0, 10)
            marker.points.append(point)
            
            color = ColorRGBA()
            color.r = np.random.random()
            color.g = np.random.random()
            color.b = np.random.random()
            color.a = 1.0
            marker.colors.append(color)
        
        return marker
    
    converter = FastMarkerToOpen3D()
    sizes = [10000, 50000, 100000, 500000]
    
    for size in sizes:
        print(f"\n--- {size} points ---")
        
        marker = create_test_marker(size)
        
        # Test original
        pcd1, scale1, time1 = converter.marker_to_pointcloud_original(marker)
        print(f"Original            : {time1:8.2f} ms")
        
        # Test vectorized
        pcd2, scale2, time2 = converter.marker_to_pointcloud_vectorized(marker)
        print(f"Vectorized          : {time2:8.2f} ms (speedup: {time1/time2:.2f}x)")
        
        # Test memmap
        pcd3, scale3, time3 = converter.marker_to_pointcloud_memmap(marker)
        print(f"Chunked             : {time3:8.2f} ms (speedup: {time1/time3:.2f}x)")
        
        # Test MarkerArray optimization
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        pcd4, scale4, time4 = converter.marker_array_to_pointcloud_optimized(marker_array)
        print(f"MarkerArray optimized: {time4:8.2f} ms (speedup: {time1/time4:.2f}x)")


if __name__ == '__main__':
    benchmark_marker_conversion()