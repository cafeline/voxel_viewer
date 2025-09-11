#!/usr/bin/env python3
"""Open3D-based viewer for voxel visualization with HDF5 file support."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
import open3d as o3d
import numpy as np
import threading
from .marker_to_open3d import MarkerToOpen3D
from .hdf5_reader import HDF5CompressedMapReader


class VoxelViewerWithHDF5Node(Node):
    """ROS2 node for visualizing voxels with HDF5 file comparison."""

    def __init__(self):
        """Initialize the voxel viewer node."""
        super().__init__('voxel_viewer_with_hdf5')
        
        # Declare parameters
        self.declare_parameter('mode', 'topic_comparison')  # 'topic_comparison' or 'file_comparison'
        self.declare_parameter('hdf5_file', '')  # Path to HDF5 file for file_comparison mode
        self.declare_parameter('voxel_size', 0.1)
        self.declare_parameter('point_size', 5.0)
        self.declare_parameter('background_color', [0.1, 0.1, 0.1])
        self.declare_parameter('show_axes', True)
        self.declare_parameter('tolerance', 0.001)
        # Display control: when false, use HDF5 voxel_size for display; point size stays constant
        self.declare_parameter('force_display_voxel_size', False)
        self.declare_parameter('display_voxel_size', 0.1)
        # RViz-like rendering: 'points' or 'cubes'
        self.declare_parameter('render_style', 'cubes')
        self.declare_parameter('cube_scale', 1.0)  # world size = cube_scale * voxel_size
        
        # Get parameters
        self.mode = self.get_parameter('mode').value
        self.hdf5_file = self.get_parameter('hdf5_file').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.point_size = self.get_parameter('point_size').value
        self.background_color = self.get_parameter('background_color').value
        self.show_axes = self.get_parameter('show_axes').value
        self.tolerance = self.get_parameter('tolerance').value
        self.force_display_voxel_size = self.get_parameter('force_display_voxel_size').value
        self.display_voxel_size = self.get_parameter('display_voxel_size').value
        self.render_style = self.get_parameter('render_style').value
        self.cube_scale = self.get_parameter('cube_scale').value
        # Default display voxel size to voxel_size if not set
        if not self.display_voxel_size or self.display_voxel_size <= 0:
            self.display_voxel_size = self.voxel_size
        
        # Initialize components
        self.converter = MarkerToOpen3D()
        
        # Store point clouds
        self.occupied_points = None
        self.pattern_points = None
        self.file_points = None
        self.occupied_scale = 0.1
        self.pattern_scale = 0.1
        self.occupied_received = False
        self.pattern_received = False
        self.file_loaded = False
        self.current_voxel_size = 0.1
        self.file_voxel_size = None  # Store the voxel size from HDF5 file
        
        # Comparison point cloud
        self.comparison_pcd = o3d.geometry.PointCloud()
        self.current_geometries = []  # for cube meshes
        
        # Visualization
        self.vis = None
        self.vis_thread = None
        self.vis_initialized = False
        self.view_reset_done = False  # Preserve user zoom; reset only once on first draw
        self.update_flag = False
        self.lock = threading.Lock()
        
        # Mode-specific initialization
        if self.mode == 'file_comparison':
            if not self.hdf5_file:
                self.get_logger().error('HDF5 file path not provided for file_comparison mode')
                return
            # Try to load HDF5 file, but don't fail if it doesn't exist yet
            self.load_hdf5_file()
        else:
            # Topic comparison mode - subscribe to pattern_markers
            self.pattern_sub = self.create_subscription(
                MarkerArray,
                'pattern_markers',
                self.pattern_callback,
                10
            )
        
        # Always subscribe to occupied_voxel_markers for comparison
        self.occupied_sub = self.create_subscription(
            MarkerArray,
            'occupied_voxel_markers',
            self.occupied_callback,
            10
        )
        
        self.get_logger().info(f'VoxelViewer with HDF5 initialized in {self.mode} mode')
        if self.mode == 'file_comparison':
            self.get_logger().info(f'HDF5 file: {self.hdf5_file}')
        self.get_logger().info('Waiting for first MarkerArray...')

    def load_hdf5_file(self):
        """Load and decompress HDF5 file."""
        import os
        
        # Check if file exists
        if not os.path.exists(self.hdf5_file):
            self.get_logger().debug(f'HDF5 file not found yet: {self.hdf5_file}')
            return
        
        try:
            reader = HDF5CompressedMapReader(self.hdf5_file)
            if reader.read():
                points = reader.decompress()
                if len(points) > 0:
                    self.file_points = points
                    self.file_loaded = True
                    
                    # Get voxel size from the HDF5 file
                    if reader.data and 'compression_params' in reader.data:
                        params = reader.data['compression_params']
                        file_voxel_size = params.get('voxel_size', 0.1)
                        if hasattr(file_voxel_size, '__len__'):  # Check if it's an array
                            file_voxel_size = file_voxel_size[0] if len(file_voxel_size) > 0 else 0.1
                        self.file_voxel_size = file_voxel_size
                        # Decide display voxel size
                        if self.force_display_voxel_size:
                            used_display_voxel = self.display_voxel_size
                            self.get_logger().info(
                                f'HDF5 file voxel size: {self.file_voxel_size} (display forced to {used_display_voxel})'
                            )
                        else:
                            # Follow HDF5 value for display
                            self.voxel_size = self.file_voxel_size
                            used_display_voxel = self.voxel_size
                            self.get_logger().info(
                                f'HDF5 file voxel size: {self.file_voxel_size} (using for display)'
                            )
                        # If visualizer already running, keep point size constant in pixels
                        if self.vis_initialized and self.vis is not None:
                            try:
                                ro = self.vis.get_render_option()
                                ro.point_size = self.point_size
                            except Exception:
                                pass
                    
                    # Get statistics
                    stats = reader.get_statistics()
                    if stats:
                        self.get_logger().info(f'Loaded HDF5 file with {len(points)} points')
                        
                        # Handle numpy arrays in stats
                        orig_pts = stats.get("original_points", "N/A")
                        if hasattr(orig_pts, '__len__'):  # Check if it's an array
                            orig_pts = orig_pts[0] if len(orig_pts) > 0 else "N/A"
                        self.get_logger().info(f'Original points: {orig_pts}')
                        
                        comp_ratio = stats.get("compression_ratio", "N/A")
                        if hasattr(comp_ratio, '__len__'):  # Check if it's an array
                            comp_ratio = comp_ratio[0] if len(comp_ratio) > 0 else "N/A"
                        if comp_ratio != "N/A" and isinstance(comp_ratio, (int, float)):
                            self.get_logger().info(f'Compression ratio: {comp_ratio:.2f}')
                        else:
                            self.get_logger().info(f'Compression ratio: {comp_ratio}')
                else:
                    self.get_logger().warn('HDF5 file loaded but no points decompressed')
            else:
                self.get_logger().error(f'Failed to read HDF5 file: {self.hdf5_file}')
        except Exception as e:
            self.get_logger().error(f'Error loading HDF5 file: {e}')

    def occupied_callback(self, msg):
        """Handle occupied voxel markers."""
        with self.lock:
            if len(msg.markers) > 0:
                # Extract points and scale
                pcd, scale = self.converter.marker_array_to_pointcloud(msg)
                if pcd and len(pcd.points) > 0:
                    self.occupied_points = np.asarray(pcd.points)
                    self.occupied_scale = scale
                    self.occupied_received = True
                    self.current_voxel_size = scale
                    self.update_flag = True
                    
                    self.get_logger().info(f'Received {len(self.occupied_points)} occupied voxels')
                    
                    # Start visualization if not started
                    if not self.vis_initialized:
                        self.start_visualization()

    def pattern_callback(self, msg):
        """Handle pattern markers (only in topic_comparison mode)."""
        if self.mode != 'topic_comparison':
            return
            
        with self.lock:
            if len(msg.markers) > 0:
                # Extract points and scale
                pcd, scale = self.converter.marker_array_to_pointcloud(msg)
                if pcd and len(pcd.points) > 0:
                    self.pattern_points = np.asarray(pcd.points)
                    self.pattern_scale = scale
                    self.pattern_received = True
                    self.update_flag = True
                    
                    self.get_logger().info(f'Received {len(self.pattern_points)} pattern voxels')

    def start_visualization(self):
        """Start the Open3D visualization thread."""
        if not self.vis_thread:
            self.vis_thread = threading.Thread(target=self.run_visualization)
            self.vis_thread.daemon = True
            self.vis_thread.start()

    def run_visualization(self):
        """Run Open3D visualization loop."""
        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Voxel Viewer with HDF5 Comparison")
        
        # Set render options
        render_option = self.vis.get_render_option()
        # Keep point size constant in pixels regardless of voxel size and zoom
        render_option.point_size = self.point_size
        render_option.background_color = np.array(self.background_color)
        render_option.show_coordinate_frame = self.show_axes
        
        # Add initial geometry
        self.comparison_pcd = o3d.geometry.PointCloud()
        if self.render_style == 'points':
            self.vis.add_geometry(self.comparison_pcd)
        
        # Add coordinate frame if requested, scale with voxel size in world units
        if self.show_axes:
            base_voxel = self.file_voxel_size if self.file_voxel_size else (self.display_voxel_size if self.force_display_voxel_size else self.voxel_size)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=base_voxel * 10, origin=[0, 0, 0])
            self.vis.add_geometry(coord_frame)
        
        self.vis_initialized = True
        self.get_logger().info('Visualization window opened')
        
        # Main visualization loop
        while True:
            with self.lock:
                if self.update_flag:
                    self.update_comparison_visualization()
                    self.update_flag = False
            
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # Check if window is closed
            if not self.vis.poll_events():
                break
        
        self.vis.destroy_window()

    def update_comparison_visualization(self):
        """Update comparison visualization based on mode."""
        if self.mode == 'file_comparison':
            self.update_file_comparison()
        else:
            self.update_topic_comparison()

    def update_file_comparison(self):
        """Update visualization for file vs occupied comparison."""
        # Try to load HDF5 file if not loaded yet
        if not self.file_loaded and self.hdf5_file:
            self.load_hdf5_file()
        
        # If still no file data, show only occupied voxels
        if not self.occupied_received:
            return
        
        if not self.file_loaded:
            # Show only occupied voxels in red
            occupied_rounded = self.round_points(self.occupied_points, self.current_voxel_size)
            if len(occupied_rounded) == 0:
                return
            if self.render_style == 'cubes':
                self.update_cubes([
                    (np.array(occupied_rounded), [1.0, 0.0, 0.0])  # red
                ], self.file_voxel_size or self.current_voxel_size)
            else:
                all_points = np.array(occupied_rounded)
                all_colors = np.tile([1.0, 0.0, 0.0], (len(all_points), 1))
                self.update_points(all_points, all_colors)
            self.get_logger().info(f'HDF5 file not loaded yet, showing {len(occupied_rounded)} occupied voxels only')
            return
        
        # Round points for comparison
        # Always use HDF5 voxel size for comparison if available
        comparison_voxel_size = self.file_voxel_size if self.file_voxel_size else self.current_voxel_size
        occupied_rounded = self.round_points(self.occupied_points, comparison_voxel_size)
        file_rounded = self.round_points(self.file_points, comparison_voxel_size)
        
        self.get_logger().info(f'Comparing with voxel size: {comparison_voxel_size} '
                              f'(occupied: {self.current_voxel_size}, file: {self.file_voxel_size})')
        
        # Find common and unique points
        occupied_set = set(map(tuple, occupied_rounded))
        file_set = set(map(tuple, file_rounded))
        
        common = occupied_set & file_set
        only_occupied = occupied_set - file_set
        only_file = file_set - occupied_set
        
        if len(common) + len(only_occupied) + len(only_file) > 0:
            if self.render_style == 'cubes':
                cube_sets = []
                if len(common) > 0:
                    common_arr = np.array(list(common))
                    cube_sets.append((common_arr, [1.0, 1.0, 1.0]))  # white
                else:
                    common_arr = np.zeros((0, 3))
                if len(only_occupied) > 0:
                    only_occ_arr = np.array(list(only_occupied))
                    cube_sets.append((only_occ_arr, [1.0, 0.0, 0.0]))  # red
                else:
                    only_occ_arr = np.zeros((0, 3))
                if len(only_file) > 0:
                    only_file_arr = np.array(list(only_file))
                    cube_sets.append((only_file_arr, [0.0, 1.0, 0.0]))  # green
                else:
                    only_file_arr = np.zeros((0, 3))
                self.update_cubes(cube_sets, comparison_voxel_size)
                points_array = np.vstack([a for a in [common_arr, only_occ_arr, only_file_arr] if a.size > 0]) if (len(common)+len(only_occupied)+len(only_file)) > 0 else np.zeros((0,3))
            else:
                all_points = []
                all_colors = []
                for p in common:
                    all_points.append(p)
                    all_colors.append([1, 1, 1])
                for p in only_occupied:
                    all_points.append(p)
                    all_colors.append([1, 0, 0])
                for p in only_file:
                    all_points.append(p)
                    all_colors.append([0, 1, 0])
                points_array = np.array(all_points)
                self.update_points(points_array, np.array(all_colors))
            
            # Log statistics and bounds
            total = len(points_array)
            bounds_min = points_array.min(axis=0) if total > 0 else np.zeros(3)
            bounds_max = points_array.max(axis=0) if total > 0 else np.zeros(3)
            self.get_logger().info(
                f'File comparison - Common: {len(common)} (white), '
                f'Only topic: {len(only_occupied)} (red), '
                f'Only file: {len(only_file)} (green), '
                f'Total: {total}'
            )
            self.get_logger().info(
                f'Bounds: min={bounds_min}, max={bounds_max}, '
                f'range={(bounds_max - bounds_min)}'
            )

    def update_topic_comparison(self):
        """Update visualization for occupied vs pattern comparison."""
        if not self.occupied_received or not self.pattern_received:
            return
        
        # Round points for comparison
        occupied_rounded = self.round_points(self.occupied_points, self.current_voxel_size)
        pattern_rounded = self.round_points(self.pattern_points, self.current_voxel_size)
        
        # Find common and unique points
        occupied_set = set(map(tuple, occupied_rounded))
        pattern_set = set(map(tuple, pattern_rounded))
        
        common = occupied_set & pattern_set
        only_occupied = occupied_set - pattern_set
        only_pattern = pattern_set - occupied_set
        
        if len(common) + len(only_occupied) + len(only_pattern) > 0:
            if self.render_style == 'cubes':
                cube_sets = []
                if len(common) > 0:
                    cube_sets.append((np.array(list(common)), [1.0, 1.0, 1.0]))  # white
                if len(only_occupied) > 0:
                    cube_sets.append((np.array(list(only_occupied)), [1.0, 0.0, 0.0]))  # red
                if len(only_pattern) > 0:
                    cube_sets.append((np.array(list(only_pattern)), [0.0, 0.0, 1.0]))  # blue
                self.update_cubes(cube_sets, self.current_voxel_size)
            else:
                all_points = []
                all_colors = []
                for p in common:
                    all_points.append(p)
                    all_colors.append([1, 1, 1])
                for p in only_occupied:
                    all_points.append(p)
                    all_colors.append([1, 0, 0])
                for p in only_pattern:
                    all_points.append(p)
                    all_colors.append([0, 0, 1])
                self.update_points(np.array(all_points), np.array(all_colors))
            
            # Log statistics
            total = len(all_points)
            self.get_logger().info(
                f'Topic comparison - Common: {len(common)} (white), '
                f'Only occupied: {len(only_occupied)} (red), '
                f'Only pattern: {len(only_pattern)} (blue), '
                f'Total: {total}'
            )

    def round_points(self, points, voxel_size):
        """Round points to voxel grid."""
        if voxel_size <= 0:
            return points
        return np.round(points / voxel_size) * voxel_size

    # --- RViz-like cube rendering helpers ---
    def update_points(self, points_array: np.ndarray, colors_array: np.ndarray):
        """Update the point cloud visualization (points mode)."""
        self.comparison_pcd.points = o3d.utility.Vector3dVector(points_array)
        self.comparison_pcd.colors = o3d.utility.Vector3dVector(colors_array)
        self.vis.update_geometry(self.comparison_pcd)
        if not self.view_reset_done:
            self.vis.reset_view_point(True)
            self.view_reset_done = True

    def build_cubes_mesh(self, centers: np.ndarray, voxel_size: float, color_rgb: list) -> o3d.geometry.TriangleMesh:
        """Build a single TriangleMesh containing cubes at given centers.
        The cube world size equals cube_scale * voxel_size.
        """
        size = float(self.cube_scale) * float(voxel_size)
        mesh_total = o3d.geometry.TriangleMesh()
        if centers.size == 0:
            return mesh_total
        # Pre-create unit cube of desired size
        base = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        base.compute_vertex_normals()
        for c in centers:
            cube = o3d.geometry.TriangleMesh(base)  # copy
            # translate so that cube center is at c
            cube.translate([c[0] - size/2.0, c[1] - size/2.0, c[2] - size/2.0])
            mesh_total += cube
        mesh_total.paint_uniform_color(color_rgb)
        return mesh_total

    def update_cubes(self, cube_sets: list, voxel_size: float):
        """Update visualization by rendering colored cubes (RViz CUBE_LIST-like).
        cube_sets: list of (centers ndarray Nx3, color [r,g,b]).
        """
        # Remove previous cube geometries
        for g in self.current_geometries:
            try:
                self.vis.remove_geometry(g, reset_bounding_box=False)
            except Exception:
                pass
        self.current_geometries = []
        # Build and add new meshes
        for centers, color in cube_sets:
            mesh = self.build_cubes_mesh(centers, voxel_size, color)
            if len(mesh.triangles) > 0:
                self.vis.add_geometry(mesh)
                self.current_geometries.append(mesh)
        # Reset view once
        if not self.view_reset_done and len(self.current_geometries) > 0:
            self.vis.reset_view_point(True)
            self.view_reset_done = True


def main(args=None):
    rclpy.init(args=args)
    node = VoxelViewerWithHDF5Node()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
