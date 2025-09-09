#!/usr/bin/env python3
"""Optimized voxel viewer with all performance improvements."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
import open3d as o3d
import numpy as np
import threading
from .marker_to_open3d import MarkerToOpen3D
import time


class OptimizedVoxelViewerNode(Node):
    """Optimized ROS2 node for visualizing and comparing voxels using Open3D."""

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
        self.declare_parameter('tolerance', 0.001)

        self.voxel_size = self.get_parameter('voxel_size').value
        self.point_size = self.get_parameter('point_size').value
        self.background_color = self.get_parameter('background_color').value
        self.show_axes = self.get_parameter('show_axes').value
        self.tolerance = self.get_parameter('tolerance').value

        self.get_logger().info('OptimizedVoxelViewer node initialized (with performance improvements)')
        self.get_logger().info(f'Subscribing to: occupied_voxel_markers, pattern_markers')
        self.get_logger().info(f'Point matching tolerance: {self.tolerance}')
        self.get_logger().info('Waiting for first MarkerArray before opening visualization window...')

    def marker_array_to_pointcloud_optimized(self, marker_array):
        """Optimized MarkerArray conversion with pre-allocation."""
        # Count total points first
        total_points = sum(len(m.points) for m in marker_array.markers if m.type == Marker.CUBE_LIST)

        if total_points == 0:
            return o3d.geometry.PointCloud(), 0.1

        # Pre-allocate combined arrays
        all_points = np.empty((total_points, 3), dtype=np.float32)
        scales = []

        current_idx = 0
        for marker in marker_array.markers:
            if marker.type == Marker.CUBE_LIST:
                num_points = len(marker.points)
                if num_points > 0:
                    # Extract points using list comprehension (faster than loop)
                    end_idx = current_idx + num_points
                    all_points[current_idx:end_idx, 0] = [p.x for p in marker.points]
                    all_points[current_idx:end_idx, 1] = [p.y for p in marker.points]
                    all_points[current_idx:end_idx, 2] = [p.z for p in marker.points]

                    scales.append(marker.scale.x)
                    current_idx = end_idx

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))

        avg_scale = np.mean(scales) if scales else 0.1

        return pcd, avg_scale

    def round_to_voxel_fast(self, points, voxel_size):
        """Ultra-fast in-place voxel rounding."""
        # Convert to float32 for speed if needed
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)
        else:
            points = points.astype(np.float32, copy=True)

        # In-place operations for maximum speed
        points /= voxel_size
        np.round(points, out=points)
        points *= voxel_size

        return points

    def occupied_callback(self, msg):
        """Handle occupied voxel markers."""
        start_time = time.perf_counter()

        with self.lock:
            pcd, scale = self.marker_array_to_pointcloud_optimized(msg)

            if len(pcd.points) > 0:
                self.occupied_points = np.asarray(pcd.points)
                self.occupied_received = True
                self.occupied_scale = scale

                elapsed = (time.perf_counter() - start_time) * 1000
                self.get_logger().info(
                    f'Received occupied voxels: {len(self.occupied_points)} points, '
                    f'scale: {scale:.3f}, processing time: {elapsed:.2f}ms'
                )

                # Start visualization thread on first data
                if not self.vis_initialized:
                    self.start_visualization()

                self.update_flag = True
                self.update_visualization()

    def pattern_callback(self, msg):
        """Handle pattern markers."""
        start_time = time.perf_counter()

        with self.lock:
            pcd, scale = self.marker_array_to_pointcloud_optimized(msg)

            if len(pcd.points) > 0:
                self.pattern_points = np.asarray(pcd.points)
                self.pattern_received = True
                self.pattern_scale = scale

                elapsed = (time.perf_counter() - start_time) * 1000
                self.get_logger().info(
                    f'Received pattern markers: {len(self.pattern_points)} points, '
                    f'scale: {scale:.3f}, processing time: {elapsed:.2f}ms'
                )

                # Start visualization thread on first data
                if not self.vis_initialized:
                    self.start_visualization()

                self.update_flag = True
                self.update_visualization()

    def update_visualization(self):
        """Update visualization based on received data."""
        if self.occupied_received and self.pattern_received:
            self.update_comparison()
        elif self.occupied_received:
            self.show_single_dataset(self.occupied_points, self.occupied_scale, 'occupied')
        elif self.pattern_received:
            self.show_single_dataset(self.pattern_points, self.pattern_scale, 'pattern')

    def show_single_dataset(self, points, scale, dataset_type):
        """Show single dataset before comparison (optimized)."""
        start_time = time.perf_counter()

        # Fast voxel rounding
        rounded_points = self.round_to_voxel_fast(points, scale)
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

            elapsed = (time.perf_counter() - start_time) * 1000
            self.get_logger().debug(f'Single dataset processing time: {elapsed:.2f}ms')

    def update_comparison(self):
        """Compare two point clouds and update visualization (optimized)."""
        start_time = time.perf_counter()
        self.get_logger().info('Comparing point clouds...')

        # Use the average of both scales for voxel size
        comparison_voxel_size = (self.occupied_scale + self.pattern_scale) / 2.0

        # Fast voxel rounding
        rounding_start = time.perf_counter()
        occupied_rounded = self.round_to_voxel_fast(self.occupied_points, comparison_voxel_size)
        pattern_rounded = self.round_to_voxel_fast(self.pattern_points, comparison_voxel_size)
        rounding_time = (time.perf_counter() - rounding_start) * 1000

        # Convert to sets for comparison (already optimized, original was fastest)
        comparison_start = time.perf_counter()
        occupied_set = set(map(tuple, occupied_rounded))
        pattern_set = set(map(tuple, pattern_rounded))

        # Find matches and mismatches
        matches = occupied_set.intersection(pattern_set)
        occupied_only = occupied_set - pattern_set
        pattern_only = pattern_set - occupied_set
        comparison_time = (time.perf_counter() - comparison_start) * 1000

        self.get_logger().info(
            f'Comparison results: {len(matches)} matches, '
            f'{len(occupied_only)} occupied-only, {len(pattern_only)} pattern-only'
        )

        # Create combined point cloud with colors
        visualization_start = time.perf_counter()
        all_points = []
        all_colors = []
        self.get_logger().info('#0')

        # Green for matches
        green_color = [0.0, 0.6, 0.0]
        if len(matches) > 0:
            self.get_logger().info(
                f'🟢 Match | RGB: {green_color} | {len(matches)} voxels | 両方に存在（一致）'
            )
        self.get_logger().info('#1')
        for point in matches:
            all_points.append(point)
            all_colors.append(green_color)
        self.get_logger().info('#2')

        # Red for mismatches
        red_color = [0.6, 0.0, 0.0]
        total_mismatches = len(occupied_only) + len(pattern_only)
        self.get_logger().info('#3')

        if total_mismatches > 0:
            self.get_logger().info(
                f'🔴 Mismatch | RGB: {red_color} | {total_mismatches} voxels | 片方のみ（不一致）'
            )
        else:
            self.get_logger().info('###################################')
            self.get_logger().info('No mismatches found! Perfect match!')
            self.get_logger().info('###################################')
        self.get_logger().info('#4')

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

        visualization_time = (time.perf_counter() - visualization_start) * 1000
        total_time = (time.perf_counter() - start_time) * 1000

        # Report performance metrics
        self.get_logger().info(
            f'Performance: Total={total_time:.2f}ms '
            f'(Rounding={rounding_time:.2f}ms, '
            f'Comparison={comparison_time:.2f}ms, '
            f'Visualization={visualization_time:.2f}ms)'
        )

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
        self.vis.create_window("Optimized Voxel Comparison Viewer - Open3D", width=1280, height=720)

        # Set render options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(self.background_color)
        render_option.show_coordinate_frame = self.show_axes

        self.get_logger().info(f'Open3D initialized with optimizations')

        # Create initial voxel grid visualization
        self.voxel_grid = o3d.geometry.VoxelGrid()
        self.vis.add_geometry(self.voxel_grid)

        # Add coordinate frame
        if self.show_axes:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            self.vis.add_geometry(axes)

        # Add legend text
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
                        self.vis.add_geometry(self.voxel_grid, reset_bounding_box=first_update)

                        if first_update:
                            first_update = False

                        # Set camera to isometric view on first update
                        view_control = self.vis.get_view_control()
                        view_control.set_zoom(0.7)

                        self.update_flag = False

            # Poll events and update render
            self.vis.poll_events()
            self.vis.update_renderer()

    def add_legend(self):
        """Add legend to the visualization."""
        legend_texts = [
            "Legend:",
            "🟢 Green = Match (both datasets)",
            "🔴 Red = Mismatch (one dataset only)",
            "🔵 Blue = Occupied only",
            "🟣 Purple = Pattern only"
        ]

        for i, text in enumerate(legend_texts):
            self.get_logger().info(f'Legend: {text}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = OptimizedVoxelViewerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()