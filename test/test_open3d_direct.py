#!/usr/bin/env python3
"""Direct test of Open3D visualization without ROS."""

import open3d as o3d
import numpy as np


def test_direct_visualization():
    """Test Open3D visualization directly."""
    print("Testing Open3D visualization directly...")
    
    # Create a simple point cloud
    pcd = o3d.geometry.PointCloud()
    
    # Generate points
    points = []
    colors = []
    for x in range(10):
        for y in range(10):
            for z in range(10):
                points.append([x * 0.2, y * 0.2, z * 0.2])
                colors.append([x/10.0, y/10.0, z/10.0])
    
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    print(f"Created point cloud with {len(pcd.points)} points")
    print(f"Bounds: {pcd.get_min_bound()} to {pcd.get_max_bound()}")
    
    # Visualize
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Direct Open3D Test",
        width=1280,
        height=720,
        point_show_normal=False
    )


if __name__ == '__main__':
    test_direct_visualization()