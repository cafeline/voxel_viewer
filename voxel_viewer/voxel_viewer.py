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
        self.vis_initialized = False
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
        self.get_logger().info('Waiting for first MarkerArray before opening visualization window...')

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
                
                # Start visualization thread on first data
                if not self.vis_initialized:
                    self.start_visualization()
                
                # Update visualization immediately
                self.update_visualization()

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
                
                # Start visualization thread on first data
                if not self.vis_initialized:
                    self.start_visualization()
                
                # Update visualization immediately
                self.update_visualization()
    
    def update_visualization(self):
        """Update visualization with current data."""
        if self.occupied_received and self.pattern_received:
            # Both received - show comparison
            self.update_comparison()
        elif self.occupied_received:
            # Only occupied received - show in blue
            self.show_single_dataset(self.occupied_points, self.occupied_scale, 'occupied')
        elif self.pattern_received:
            # Only pattern received - show in purple
            self.show_single_dataset(self.pattern_points, self.pattern_scale, 'pattern')
    
    def show_single_dataset(self, points, scale, dataset_type):
        """Show single dataset before comparison."""
        # Round points to voxel grid
        def round_to_voxel(points, voxel_size):
            """Round points to nearest voxel center."""
            return np.round(points / voxel_size) * voxel_size
        
        rounded_points = round_to_voxel(points, scale)
        unique_points = np.unique(rounded_points, axis=0)
        
        # Create point cloud with dataset-specific color
        all_points = []
        all_colors = []
        
        if dataset_type == 'occupied':
            # Blue for occupied only
            color = [0.0, 0.0, 0.7]
            self.get_logger().info(
                f'🔵 Occupied only | RGB: {color} | {len(unique_points)} voxels | occupied_voxel_markersのみ'
            )
        else:  # pattern
            # Purple for pattern only
            color = [0.5, 0.0, 0.7]
            self.get_logger().info(
                f'🟣 Pattern only | RGB: {color} | {len(unique_points)} voxels | pattern_markersのみ'
            )
        
        for point in unique_points:
            all_points.append(point)
            all_colors.append(color)
        
        # Update comparison point cloud
        if all_points:
            self.comparison_pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
            self.comparison_pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
            self.current_voxel_size = scale
            self.update_flag = True
    
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
        
        # Green for matches
        green_color = [0.0, 0.6, 0.0]
        if len(matches) > 0:
            self.get_logger().info(
                f'🟢 Match | RGB: {green_color} | {len(matches)} voxels | 両方に存在（一致）'
            )
        
        for point in matches:
            all_points.append(point)
            all_colors.append(green_color)
        
        # Red for mismatches
        red_color = [0.6, 0.0, 0.0]
        total_mismatches = len(occupied_only) + len(pattern_only)
        if total_mismatches > 0:
            self.get_logger().info(
                f'🔴 Mismatch | RGB: {red_color} | {total_mismatches} voxels | 片方のみ（不一致）'
            )
        
        for point in occupied_only:
            all_points.append(point)
            all_colors.append(red_color)
        
        for point in pattern_only:
            all_points.append(point)
            all_colors.append(red_color)
        
        # Update comparison point cloud
        if all_points:
            self.comparison_pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
            self.comparison_pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
            
            # Store the voxel size for visualization
            self.current_voxel_size = comparison_voxel_size
            self.update_flag = True

    def start_visualization(self):
        """Start the Open3D visualization in a separate thread."""
        if not self.vis_initialized:
            self.vis_initialized = True
            self.get_logger().info('Starting Open3D visualization window...')
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
                        
                        # Silent update - no need to log every frame
                        total_voxels = len(self.voxel_grid.get_voxels())
                        
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
        # Create legend for all states
        # Occupied only (blue)
        legend_occupied = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        legend_occupied.compute_vertex_normals()
        legend_occupied.paint_uniform_color([0.0, 0.0, 0.7])  # Blue
        legend_occupied.translate([0, 0, 2.7])
        
        # Pattern only (purple)
        legend_pattern = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        legend_pattern.compute_vertex_normals()
        legend_pattern.paint_uniform_color([0.5, 0.0, 0.7])  # Purple
        legend_pattern.translate([0, 0, 2.5])
        
        # Match (green)
        legend_match = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        legend_match.compute_vertex_normals()
        legend_match.paint_uniform_color([0.0, 0.6, 0.0])  # Darker green
        legend_match.translate([0, 0, 2.3])
        
        # Mismatch (red)
        legend_mismatch = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        legend_mismatch.compute_vertex_normals()
        legend_mismatch.paint_uniform_color([0.6, 0.0, 0.0])  # Darker red
        legend_mismatch.translate([0, 0, 2.1])
        
        self.vis.add_geometry(legend_occupied)
        self.vis.add_geometry(legend_pattern)
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
