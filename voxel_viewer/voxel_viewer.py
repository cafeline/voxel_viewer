#!/usr/bin/env python3
"""Open3D-based viewer for voxel visualization with comparison."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
import open3d as o3d
import numpy as np
import threading
from .marker_to_open3d import MarkerToOpen3D


class VoxelViewerNode(Node):
    """ROS2 node for visualizing and comparing voxels using Open3D."""

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
        
        # Store original point clouds and scales
        self.occupied_points = None
        self.pattern_points = None
        self.occupied_scale = 0.1
        self.pattern_scale = 0.1
        self.occupied_received = False
        self.pattern_received = False
        self.current_voxel_size = 0.1
        
        # Comparison point cloud for visualization
        self.comparison_pcd = o3d.geometry.PointCloud()
        
        # Voxel grid for better visualization
        self.voxel_grid = None
        
        # Visualization window
        self.vis = None
        self.vis_thread = None
        self.update_flag = False
        self.lock = threading.Lock()
        
        # Parameters
        self.declare_parameter('voxel_size', 0.1)
        self.declare_parameter('point_size', 5.0)
        self.declare_parameter('background_color', [0.1, 0.1, 0.1])
        self.declare_parameter('show_axes', True)
        self.declare_parameter('tolerance', 0.001)  # Tolerance for point matching
        
        self.voxel_size = self.get_parameter('voxel_size').value
        self.point_size = self.get_parameter('point_size').value
        self.background_color = self.get_parameter('background_color').value
        self.show_axes = self.get_parameter('show_axes').value
        self.tolerance = self.get_parameter('tolerance').value
        
        self.get_logger().info('VoxelViewer node initialized (comparison mode)')
        self.get_logger().info(f'Subscribing to: occupied_voxel_markers, pattern_markers')
        self.get_logger().info(f'Point matching tolerance: {self.tolerance}')
        
        # Start visualization thread
        self.start_visualization()

    def occupied_callback(self, msg):
        """Handle occupied voxel markers."""
        with self.lock:
            pcd, scale = self.converter.marker_array_to_pointcloud(msg)
            
            if len(pcd.points) > 0:
                self.occupied_points = np.asarray(pcd.points)
                self.occupied_received = True
                self.occupied_scale = scale  # Store marker scale
                
                self.get_logger().info(
                    f'Received occupied voxels: {len(self.occupied_points)} points, '
                    f'scale: {scale:.3f}'
                )
                self.get_logger().info(
                    f'Occupied bounds - Min: {self.occupied_points.min(axis=0)}, '
                    f'Max: {self.occupied_points.max(axis=0)}'
                )
                
                # Update comparison if both datasets received
                if self.pattern_received:
                    self.update_comparison()

    def pattern_callback(self, msg):
        """Handle pattern markers."""
        with self.lock:
            pcd, scale = self.converter.marker_array_to_pointcloud(msg)
            
            if len(pcd.points) > 0:
                self.pattern_points = np.asarray(pcd.points)
                self.pattern_received = True
                self.pattern_scale = scale  # Store marker scale
                
                self.get_logger().info(
                    f'Received pattern markers: {len(self.pattern_points)} points, '
                    f'scale: {scale:.3f}'
                )
                self.get_logger().info(
                    f'Pattern bounds - Min: {self.pattern_points.min(axis=0)}, '
                    f'Max: {self.pattern_points.max(axis=0)}'
                )
                
                # Update comparison if both datasets received
                if self.occupied_received:
                    self.update_comparison()
    
    def update_comparison(self):
        """Compare two point clouds and update visualization."""
        self.get_logger().info('Comparing point clouds...')
        
        # Use the average of both scales for voxel size
        comparison_voxel_size = (self.occupied_scale + self.pattern_scale) / 2.0
        
        # Round points to voxel grid for comparison
        def round_to_voxel(points, voxel_size):
            """Round points to nearest voxel center."""
            return np.round(points / voxel_size) * voxel_size
        
        # Round both point sets to voxel grid
        occupied_rounded = round_to_voxel(self.occupied_points, comparison_voxel_size)
        pattern_rounded = round_to_voxel(self.pattern_points, comparison_voxel_size)
        
        # Convert to sets of tuples for efficient comparison
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
        
        # Create combined point cloud with colors (semi-transparent)
        all_points = []
        all_colors = []
        
        # Green for matches (semi-transparent: 0.0, 0.6, 0.0 for darker green)
        for point in matches:
            all_points.append(point)
            all_colors.append([0.0, 0.6, 0.0])
        
        # Red for mismatches (semi-transparent: 0.6, 0.0, 0.0 for darker red)
        for point in occupied_only:
            all_points.append(point)
            all_colors.append([0.6, 0.0, 0.0])
        
        for point in pattern_only:
            all_points.append(point)
            all_colors.append([0.6, 0.0, 0.0])
        
        # Update comparison point cloud
        if all_points:
            self.comparison_pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
            self.comparison_pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
            
            # Store the voxel size for visualization
            self.current_voxel_size = comparison_voxel_size
            self.update_flag = True
            
            self.get_logger().info(
                f'Updated comparison visualization with {len(all_points)} total points, '
                f'voxel size: {comparison_voxel_size:.3f}'
            )

    def start_visualization(self):
        """Start the Open3D visualization in a separate thread."""
        self.vis_thread = threading.Thread(target=self.visualization_loop)
        self.vis_thread.daemon = True
        self.vis_thread.start()

    def visualization_loop(self):
        """Main visualization loop running in separate thread."""
        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Voxel Comparison Viewer - Open3D", width=1280, height=720)
        
        # Set render options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(self.background_color)
        render_option.show_coordinate_frame = self.show_axes
        
        self.get_logger().info(f'Open3D initialized')
        
        # Create initial voxel grid visualization
        self.voxel_grid = o3d.geometry.VoxelGrid()
        self.vis.add_geometry(self.voxel_grid)
        
        # Add coordinate frame
        if self.show_axes:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            self.vis.add_geometry(axes)
        
        # Add legend text (as mesh text)
        self.add_legend()
        
        # Visualization loop
        first_update = True
        while True:
            with self.lock:
                if self.update_flag:
                    self.get_logger().info('Updating Open3D visualization...')
                    
                    # Create voxel grid from point cloud for better visualization
                    if len(self.comparison_pcd.points) > 0:
                        # Remove old voxel grid
                        self.vis.remove_geometry(self.voxel_grid, reset_bounding_box=False)
                        
                        # Create new voxel grid with appropriate size
                        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                            self.comparison_pcd,
                            voxel_size=self.current_voxel_size * 0.9  # Slightly smaller for gaps
                        )
                        
                        # Add new voxel grid
                        self.vis.add_geometry(self.voxel_grid, reset_bounding_box=False)
                        
                        total_voxels = len(self.voxel_grid.get_voxels())
                        self.get_logger().info(
                            f'Updated visualization with {total_voxels} voxels '
                            f'(voxel size: {self.current_voxel_size:.3f})'
                        )
                        
                        # Reset view to fit all geometries
                        if total_voxels > 0 and first_update:
                            self.vis.reset_view_point(True)
                            first_update = False
                    
                    self.update_flag = False
            
            # Poll events and render
            if not self.vis.poll_events():
                break
            self.vis.update_renderer()
        
        self.vis.destroy_window()
    
    def add_legend(self):
        """Add legend to visualization."""
        # Create simple spheres as legend with semi-transparent appearance
        legend_match = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        legend_match.compute_vertex_normals()
        legend_match.paint_uniform_color([0.0, 0.6, 0.0])  # Darker green (simulated transparency)
        legend_match.translate([0, 0, 2.5])
        
        legend_mismatch = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        legend_mismatch.compute_vertex_normals()
        legend_mismatch.paint_uniform_color([0.6, 0.0, 0.0])  # Darker red (simulated transparency)
        legend_mismatch.translate([0, 0, 2.3])
        
        self.vis.add_geometry(legend_match)
        self.vis.add_geometry(legend_mismatch)

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
