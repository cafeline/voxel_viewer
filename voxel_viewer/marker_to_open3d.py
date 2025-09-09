#!/usr/bin/env python3
"""Convert ROS MarkerArray to Open3D point cloud for visualization."""

import numpy as np
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class MarkerToOpen3D:
    """Convert ROS MarkerArray messages to Open3D point clouds."""

    def __init__(self):
        """Initialize the converter."""
        pass

    def rgba_to_rgb(self, color_rgba):
        """Convert RGBA color to RGB.
        
        Args:
            color_rgba: ColorRGBA message
            
        Returns:
            List of [r, g, b] values
        """
        return [color_rgba.r, color_rgba.g, color_rgba.b]

    def marker_to_pointcloud(self, marker):
        """Convert a single marker to Open3D point cloud.
        
        Args:
            marker: Marker message
            
        Returns:
            tuple: (o3d.geometry.PointCloud object, marker_scale)
        """
        pcd = o3d.geometry.PointCloud()
        marker_scale = None
        
        if marker.type == Marker.CUBE_LIST:
            # Get marker scale for voxel size
            marker_scale = marker.scale.x  # Assuming uniform scale
            
            # Handle CUBE_LIST marker
            points = []
            colors = []
            
            for i, point in enumerate(marker.points):
                points.append([point.x, point.y, point.z])
                
                # Use individual colors if available, otherwise use marker color
                if i < len(marker.colors):
                    colors.append(self.rgba_to_rgb(marker.colors[i]))
                else:
                    colors.append(self.rgba_to_rgb(marker.color))
            
            if points:
                pcd.points = o3d.utility.Vector3dVector(np.array(points))
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
                
        elif marker.type == Marker.CUBE:
            # Get marker scale
            marker_scale = marker.scale.x
            
            # Handle individual CUBE marker
            points = [[
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z
            ]]
            colors = [self.rgba_to_rgb(marker.color)]
            
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
            
        elif marker.type == Marker.POINTS:
            # For POINTS type, use scale.x if available
            if marker.scale.x > 0:
                marker_scale = marker.scale.x
            
            # Handle POINTS marker
            points = []
            colors = []
            
            for i, point in enumerate(marker.points):
                points.append([point.x, point.y, point.z])
                
                if i < len(marker.colors):
                    colors.append(self.rgba_to_rgb(marker.colors[i]))
                else:
                    colors.append(self.rgba_to_rgb(marker.color))
            
            if points:
                pcd.points = o3d.utility.Vector3dVector(np.array(points))
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pcd, marker_scale

    def marker_array_to_pointcloud(self, marker_array):
        """Convert MarkerArray to a single Open3D point cloud.
        
        Args:
            marker_array: MarkerArray message
            
        Returns:
            tuple: (o3d.geometry.PointCloud object, average_marker_scale)
        """
        combined_pcd = o3d.geometry.PointCloud()
        scales = []
        
        for marker in marker_array.markers:
            pcd, scale = self.marker_to_pointcloud(marker)
            combined_pcd += pcd
            if scale is not None:
                scales.append(scale)
        
        # Calculate average scale
        avg_scale = np.mean(scales) if scales else 0.1
        
        return combined_pcd, avg_scale

    def create_voxel_cube(self, position, size):
        """Create a cube mesh for voxel visualization.
        
        Args:
            position: [x, y, z] center position
            size: Size of the voxel
            
        Returns:
            o3d.geometry.TriangleMesh object
        """
        # Create a cube mesh
        cube = o3d.geometry.TriangleMesh.create_box(
            width=size, height=size, depth=size
        )
        
        # Translate to the correct position (centered)
        cube.translate([
            position[0] - size/2,
            position[1] - size/2,
            position[2] - size/2
        ])
        
        return cube

    def marker_array_to_voxel_grid(self, marker_array, voxel_size=0.1):
        """Convert MarkerArray to Open3D voxel grid for efficient visualization.
        
        Args:
            marker_array: MarkerArray message
            voxel_size: Size of voxels
            
        Returns:
            o3d.geometry.VoxelGrid object
        """
        # First convert to point cloud
        pcd = self.marker_array_to_pointcloud(marker_array)
        
        # Create voxel grid from point cloud
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=voxel_size
        )
        
        return voxel_grid