#!/usr/bin/env python3
"""Test for MarkerArray to Open3D conversion."""

import unittest
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# These will be implemented in voxel_viewer
from voxel_viewer.marker_to_open3d import MarkerToOpen3D


class TestMarkerToOpen3D(unittest.TestCase):
    """Test MarkerArray to Open3D conversion."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = MarkerToOpen3D()

    def test_cube_list_to_pointcloud(self):
        """Test CUBE_LIST marker conversion to point cloud."""
        # Create a CUBE_LIST marker
        marker = Marker()
        marker.type = Marker.CUBE_LIST
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        # Add points
        for i in range(10):
            point = Point()
            point.x = i * 0.1
            point.y = 0.0
            point.z = 0.0
            marker.points.append(point)
            
            color = ColorRGBA()
            color.r = 0.0
            color.g = 0.0
            color.b = 1.0
            color.a = 1.0
            marker.colors.append(color)
        
        # Convert to Open3D
        pcd = self.converter.marker_to_pointcloud(marker)
        
        # Verify
        self.assertIsNotNone(pcd)
        self.assertEqual(len(pcd.points), 10)
        
        # Check positions
        points = np.asarray(pcd.points)
        for i in range(10):
            self.assertAlmostEqual(points[i][0], i * 0.1, places=5)
            self.assertAlmostEqual(points[i][1], 0.0, places=5)
            self.assertAlmostEqual(points[i][2], 0.0, places=5)
        
        # Check colors
        colors = np.asarray(pcd.colors)
        for i in range(10):
            self.assertAlmostEqual(colors[i][0], 0.0, places=5)  # R
            self.assertAlmostEqual(colors[i][1], 0.0, places=5)  # G
            self.assertAlmostEqual(colors[i][2], 1.0, places=5)  # B

    def test_individual_cube_to_pointcloud(self):
        """Test individual CUBE marker conversion."""
        marker = Marker()
        marker.type = Marker.CUBE
        marker.pose.position.x = 1.0
        marker.pose.position.y = 2.0
        marker.pose.position.z = 3.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Convert to Open3D
        pcd = self.converter.marker_to_pointcloud(marker)
        
        # Verify
        self.assertIsNotNone(pcd)
        self.assertEqual(len(pcd.points), 1)
        
        # Check position
        points = np.asarray(pcd.points)
        self.assertAlmostEqual(points[0][0], 1.0, places=5)
        self.assertAlmostEqual(points[0][1], 2.0, places=5)
        self.assertAlmostEqual(points[0][2], 3.0, places=5)
        
        # Check color
        colors = np.asarray(pcd.colors)
        self.assertAlmostEqual(colors[0][0], 1.0, places=5)  # R
        self.assertAlmostEqual(colors[0][1], 0.0, places=5)  # G
        self.assertAlmostEqual(colors[0][2], 0.0, places=5)  # B

    def test_marker_array_to_pointcloud(self):
        """Test MarkerArray conversion to single point cloud."""
        marker_array = MarkerArray()
        
        # Add multiple markers
        for i in range(3):
            marker = Marker()
            marker.type = Marker.CUBE_LIST
            
            for j in range(5):
                point = Point()
                point.x = i + j * 0.1
                point.y = 0.0
                point.z = 0.0
                marker.points.append(point)
                
                color = ColorRGBA()
                color.r = i / 3.0
                color.g = 0.0
                color.b = 1.0 - i / 3.0
                color.a = 1.0
                marker.colors.append(color)
            
            marker_array.markers.append(marker)
        
        # Convert to Open3D
        pcd = self.converter.marker_array_to_pointcloud(marker_array)
        
        # Verify
        self.assertIsNotNone(pcd)
        self.assertEqual(len(pcd.points), 15)  # 3 markers x 5 points

    def test_empty_marker_array(self):
        """Test empty MarkerArray handling."""
        marker_array = MarkerArray()
        
        # Convert to Open3D
        pcd = self.converter.marker_array_to_pointcloud(marker_array)
        
        # Should return empty point cloud
        self.assertIsNotNone(pcd)
        self.assertEqual(len(pcd.points), 0)

    def test_voxel_cube_generation(self):
        """Test voxel cube mesh generation for visualization."""
        # Create a single voxel position
        position = [1.0, 2.0, 3.0]
        size = 0.1
        
        # Generate voxel cube
        cube = self.converter.create_voxel_cube(position, size)
        
        # Verify
        self.assertIsNotNone(cube)
        # Check that it's a mesh with 8 vertices and 12 triangles
        vertices = np.asarray(cube.vertices)
        triangles = np.asarray(cube.triangles)
        self.assertEqual(len(vertices), 8)
        self.assertEqual(len(triangles), 12)

    def test_color_mapping(self):
        """Test color mapping from ROS to Open3D."""
        color_rgba = ColorRGBA()
        color_rgba.r = 0.5
        color_rgba.g = 0.25
        color_rgba.b = 0.75
        color_rgba.a = 1.0
        
        # Convert to Open3D color
        o3d_color = self.converter.rgba_to_rgb(color_rgba)
        
        # Verify
        self.assertEqual(len(o3d_color), 3)
        self.assertAlmostEqual(o3d_color[0], 0.5, places=5)
        self.assertAlmostEqual(o3d_color[1], 0.25, places=5)
        self.assertAlmostEqual(o3d_color[2], 0.75, places=5)


if __name__ == '__main__':
    unittest.main()