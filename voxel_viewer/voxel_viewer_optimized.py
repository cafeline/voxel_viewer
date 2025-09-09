#!/usr/bin/env python3
"""Optimized voxel viewer for demo_with_decompressed.launch.py."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
import open3d as o3d
import numpy as np
import threading
from .marker_to_open3d import MarkerToOpen3D


class VoxelViewerNode(Node):
    """Optimized ROS2 node for voxel comparison visualization."""

    def __init__(self):
        """Initialize the voxel viewer node."""
        super().__init__('voxel_viewer')
        
        # Create subscriptions
        self.occupied_sub = self.create_subscription(
            MarkerArray,
            'occupied_voxel_markers',
            self.occupied_callback,
            10
        )
        
        self.pattern_sub = self.create_subscription(
            MarkerArray,
            'pattern_markers',
            self.pattern_callback,
            10
        )
        
        # Converter
        self.converter = MarkerToOpen3D()
        
        # Store point clouds
        self.occupied_points = None
        self.pattern_points = None
        self.occupied_scale = 0.1
        self.pattern_scale = 0.1
        self.occupied_received = False
        self.pattern_received = False
        
        # Visualization
        self.comparison_pcd = o3d.geometry.PointCloud()
        self.voxel_grid = None
        self.vis = None
        self.vis_thread = None
        self.vis_initialized = False
        self.update_flag = False
        self.lock = threading.Lock()
        
        # Parameters
        self.declare_parameter('background_color', [0.1, 0.1, 0.1])
        self.declare_parameter('show_axes', True)
        
        self.background_color = self.get_parameter('background_color').value
        self.show_axes = self.get_parameter('show_axes').value
        
        self.get_logger().info('VoxelViewer node initialized')
        self.get_logger().info('Waiting for MarkerArrays...')

    def occupied_callback(self, msg):
        """Handle occupied voxel markers."""
        with self.lock:
            pcd, scale = self.converter.marker_array_to_pointcloud(msg)
            
            if len(pcd.points) > 0:
                self.occupied_points = np.asarray(pcd.points)
                self.occupied_received = True
                self.occupied_scale = scale
                
                self.get_logger().info(f'Received occupied voxels: {len(self.occupied_points)} points')
                
                # Start visualization on first data
                if not self.vis_initialized:
                    self.start_visualization()
                
                # Update if both datasets received
                if self.pattern_received:
                    self.update_comparison()

    def pattern_callback(self, msg):
        """Handle pattern markers."""
        with self.lock:
            pcd, scale = self.converter.marker_array_to_pointcloud(msg)
            
            if len(pcd.points) > 0:
                self.pattern_points = np.asarray(pcd.points)
                self.pattern_received = True
                self.pattern_scale = scale
                
                self.get_logger().info(f'Received pattern markers: {len(self.pattern_points)} points')
                
                # Start visualization on first data
                if not self.vis_initialized:
                    self.start_visualization()
                
                # Update if both datasets received
                if self.occupied_received:
                    self.update_comparison()
    
    def update_comparison(self):
        """Compare two point clouds and update visualization."""
        self.get_logger().info('Comparing point clouds...')
        
        # Use average scale for voxel size
        voxel_size = (self.occupied_scale + self.pattern_scale) / 2.0
        
        # Round points to voxel grid
        occupied_rounded = np.round(self.occupied_points / voxel_size) * voxel_size
        pattern_rounded = np.round(self.pattern_points / voxel_size) * voxel_size
        
        # Convert to sets for comparison
        occupied_set = set(map(tuple, occupied_rounded))
        pattern_set = set(map(tuple, pattern_rounded))
        
        # Find matches and mismatches
        matches = occupied_set.intersection(pattern_set)
        occupied_only = occupied_set - pattern_set
        pattern_only = pattern_set - occupied_set
        
        self.get_logger().info(
            f'Comparison results: {len(matches)} matches, '
            f'{len(occupied_only)} occupied-only, {len(pattern_only)} pattern-only'
        )
        
        # Check for perfect match
        if len(occupied_only) == 0 and len(pattern_only) == 0:
            self.get_logger().info('###################################')
            self.get_logger().info('No mismatches found! Perfect match!')
            self.get_logger().info('###################################')
        
        # Create visualization
        all_points = []
        all_colors = []
        
        # Green for matches
        for point in matches:
            all_points.append(point)
            all_colors.append([0.0, 0.6, 0.0])
        
        # Red for mismatches
        for point in occupied_only:
            all_points.append(point)
            all_colors.append([0.6, 0.0, 0.0])
        
        for point in pattern_only:
            all_points.append(point)
            all_colors.append([0.6, 0.0, 0.0])
        
        # Update point cloud
        if all_points:
            self.comparison_pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
            self.comparison_pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
            self.current_voxel_size = voxel_size
            self.update_flag = True

    def start_visualization(self):
        """Start the Open3D visualization thread."""
        if not self.vis_initialized:
            self.vis_initialized = True
            self.get_logger().info('Starting Open3D visualization window...')
            self.vis_thread = threading.Thread(target=self.visualization_loop)
            self.vis_thread.daemon = True
            self.vis_thread.start()

    def visualization_loop(self):
        """Main visualization loop."""
        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Voxel Comparison Viewer", width=1280, height=720)
        
        # Set render options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(self.background_color)
        render_option.show_coordinate_frame = self.show_axes
        
        # Create initial voxel grid
        self.voxel_grid = o3d.geometry.VoxelGrid()
        self.vis.add_geometry(self.voxel_grid)
        
        # Add coordinate frame
        if self.show_axes:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            self.vis.add_geometry(axes)
        
        # Visualization loop
        first_update = True
        while True:
            with self.lock:
                if self.update_flag and len(self.comparison_pcd.points) > 0:
                    # Update voxel grid
                    self.vis.remove_geometry(self.voxel_grid, reset_bounding_box=False)
                    self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                        self.comparison_pcd,
                        voxel_size=self.current_voxel_size * 0.9
                    )
                    self.vis.add_geometry(self.voxel_grid, reset_bounding_box=first_update)
                    
                    if first_update:
                        self.vis.reset_view_point(True)
                        first_update = False
                    
                    self.update_flag = False
            
            # Poll events and render
            if not self.vis.poll_events():
                break
            self.vis.update_renderer()
        
        self.vis.destroy_window()

    def destroy_node(self):
        """Clean up when node is destroyed."""
        if self.vis is not None:
            self.vis.close()
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        node = VoxelViewerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()