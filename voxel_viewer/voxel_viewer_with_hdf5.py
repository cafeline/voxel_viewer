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
from .file_compare import compute_two_file_diff, validate_voxel_sizes
from .mode_utils import should_load_files, is_file_mode, is_topic_mode


class VoxelViewerWithHDF5Node(Node):
    """ROS2 node for visualizing voxels with HDF5 file comparison."""

    def __init__(self):
        """Initialize the voxel viewer node."""
        super().__init__('voxel_viewer_with_hdf5')
        
        # Declare parameters
        self.declare_parameter('mode', 'topic_comparison')  # 'topic_comparison' or 'file_comparison'
        self.declare_parameter('hdf5_file', '')  # Path to compressed HDF5 (file_comparison)
        self.declare_parameter('raw_hdf5_file', '')  # Optional: path to raw HDF5 for file-vs-file comparison
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
        self.raw_hdf5_file = self.get_parameter('raw_hdf5_file').value
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
        self.file_voxel_size = None  # Compressed file voxel size
        self.raw_file_voxel_size = None  # Raw file voxel size
        self.raw_file_points = None
        # Grid origins for proper voxel anchoring
        self.grid_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.raw_grid_origin = None
        # Load-once guards
        self._compressed_loaded_once = False
        self._raw_loaded_once = False
        # Precomputed arrays for immediate draw (two-file mode)
        self.precomp_points = None
        self.precomp_colors = None
        
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
        self.file_load_timer = None  # periodic retry for two-file mode
        
        # Mode-specific initialization
        self.two_file_comparison = (is_file_mode(self.mode) and bool(self.raw_hdf5_file))

        if is_file_mode(self.mode):
            if self.two_file_comparison:
                # Compare compressed vs raw HDF5 files
                if not self.hdf5_file or not self.raw_hdf5_file:
                    self.get_logger().error('Both hdf5_file and raw_hdf5_file must be provided for two-file comparison')
                    return
                self.load_two_hdf5_files()
                # Start visualization immediately for file-to-file mode
                if not self.vis_initialized:
                    self.start_visualization()
                # If either file not ready yet, poll until both are ready, then draw
                if (self.file_points is None) or (getattr(self, 'raw_file_points', None) is None):
                    self.start_two_file_polling()
                # If both voxel sizes are available, ensure they match
                if (self.file_voxel_size is not None) and (self.raw_file_voxel_size is not None):
                    if not validate_voxel_sizes(self.file_voxel_size, self.raw_file_voxel_size, tol=1e-9):
                        self.get_logger().error(
                            f'Voxel size mismatch between files: compressed={self.file_voxel_size} raw={self.raw_file_voxel_size}'
                        )
                        # Disable updates on mismatch
                        self.update_flag = False
                        return
            else:
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
        if is_file_mode(self.mode):
            if self.two_file_comparison:
                self.get_logger().info(f'Comparing files: compressed={self.hdf5_file}, raw={self.raw_hdf5_file}')
            else:
                self.get_logger().info(f'HDF5 file: {self.hdf5_file}')
        self.get_logger().info('Waiting for first MarkerArray...')

    def load_hdf5_file(self):
        """Load and decompress HDF5 file."""
        import os
        # Disallow file access unless in file_comparison mode
        if not should_load_files(self.mode):
            return
        if self._compressed_loaded_once:
            return
        
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
                    self._compressed_loaded_once = True

                    # Get voxel size from the HDF5 file
                    if reader.data and 'compression_params' in reader.data:
                        params = reader.data['compression_params']
                        file_voxel_size = params.get('voxel_size', 0.1)
                        if hasattr(file_voxel_size, '__len__'):  # Check if it's an array
                            file_voxel_size = file_voxel_size[0] if len(file_voxel_size) > 0 else 0.1
                        self.file_voxel_size = file_voxel_size
                        # Save grid origin if available for correct rounding anchoring
                        go = params.get('grid_origin', None)
                        if go is not None:
                            try:
                                self.grid_origin = np.array(go, dtype=np.float64).reshape(3)
                            except Exception:
                                pass
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

    def load_two_hdf5_files(self):
        """Load and decompress compressed and raw HDF5 files for file-vs-file comparison."""
        import os
        if not should_load_files(self.mode):
            return
        try:
            # Load compressed file
            if (not self._compressed_loaded_once) and os.path.exists(self.hdf5_file):
                reader1 = HDF5CompressedMapReader(self.hdf5_file)
                if reader1.read():
                    pts1 = reader1.decompress()
                    if len(pts1) > 0:
                        self.file_points = pts1
                        if reader1.data and 'compression_params' in reader1.data:
                            params = reader1.data['compression_params']
                            v1 = params.get('voxel_size', 0.1)
                            if hasattr(v1, '__len__'):
                                v1 = v1[0] if len(v1) > 0 else 0.1
                            self.file_voxel_size = float(v1)
                            # Capture grid origin for proper anchoring
                            go = params.get('grid_origin', None)
                            if go is not None:
                                try:
                                    self.grid_origin = np.array(go, dtype=np.float64).reshape(3)
                                except Exception:
                                    pass
                        self.get_logger().info(f'Loaded compressed HDF5: {len(pts1)} points, voxel_size={self.file_voxel_size}')
                        self._compressed_loaded_once = True
            else:
                if not self._compressed_loaded_once:
                    self.get_logger().warn(f'Compressed HDF5 not found: {self.hdf5_file}')

            # Load raw file
            if (not self._raw_loaded_once) and os.path.exists(self.raw_hdf5_file):
                reader2 = HDF5CompressedMapReader(self.raw_hdf5_file)
                if reader2.read():
                    pts2 = reader2.decompress()
                    if len(pts2) > 0:
                        # Store in dedicated attribute
                        self.raw_file_points = pts2
                        # Prefer raw voxel size from /raw_voxel_grid when available
                        if reader2.data and 'raw' in reader2.data and isinstance(reader2.data['raw'], dict):
                            raw = reader2.data['raw']
                            v2 = raw.get('voxel_size', None)
                            if v2 is not None:
                                try:
                                    if hasattr(v2, 'shape'):
                                        v2 = float(np.array(v2).reshape(-1)[0])
                                    self.raw_file_voxel_size = float(v2)
                                except Exception:
                                    pass
                        # Fallback: compression_params if raw not found
                        if self.raw_file_voxel_size is None and reader2.data and 'compression_params' in reader2.data:
                            params = reader2.data['compression_params']
                            v2 = params.get('voxel_size', 0.1)
                            if hasattr(v2, '__len__'):
                                v2 = v2[0] if len(v2) > 0 else 0.1
                            self.raw_file_voxel_size = float(v2)
                        # Also capture origin from raw group if present
                        if reader2.data and 'raw' in reader2.data:
                            raw = reader2.data['raw']
                            ro = raw.get('origin', None)
                            if ro is not None:
                                try:
                                    self.raw_grid_origin = np.array(ro, dtype=np.float64).reshape(3)
                                except Exception:
                                    pass
                        self.get_logger().info(f'Loaded raw HDF5: {len(pts2)} points, voxel_size={self.raw_file_voxel_size}')
                        self._raw_loaded_once = True
            else:
                if not self._raw_loaded_once:
                    self.get_logger().warn(f'Raw HDF5 not found: {self.raw_hdf5_file}')
            # Trigger initial update so something is drawn even without topics
            self.update_flag = True
        except Exception as e:
            self.get_logger().error(f'Error loading two HDF5 files: {e}')

    def start_two_file_polling(self):
        """Create a short-period timer to retry loading both files until available, then draw and stop."""
        if not should_load_files(self.mode):
            return
        if self.file_load_timer is not None:
            return
        # Poll every 0.5s until both datasets are present
        self.get_logger().info('Waiting for both HDF5 files to become available...')
        self.file_load_timer = self.create_timer(0.5, self.poll_two_files_once)

    def poll_two_files_once(self):
        """Timer callback to attempt file-to-file load and initial draw once both files are ready."""
        if not should_load_files(self.mode):
            return
        try:
            # Attempt to (re)load files
            self.load_two_hdf5_files()
            if (self.file_points is not None) and (getattr(self, 'raw_file_points', None) is not None):
                # Build arrays and draw via point cloud for immediate feedback
                pts, cols = self._build_two_file_precomp_arrays()
                if pts is not None and len(pts) > 0 and self.vis is not None:
                    # Respect render style: only draw points when requested; otherwise build cubes
                    if str(self.render_style).lower() == 'points':
                        self.comparison_pcd.points = o3d.utility.Vector3dVector(pts)
                        self.comparison_pcd.colors = o3d.utility.Vector3dVector(cols)
                        self.vis.update_geometry(self.comparison_pcd)
                        self.get_logger().info(f'Initial file-to-file draw (points): {len(pts)} points')
                    else:
                        whites = pts[(cols == np.array([1.0, 1.0, 1.0])).all(axis=1)]
                        reds = pts[(cols == np.array([1.0, 0.0, 0.0])).all(axis=1)]
                        greens = pts[(cols == np.array([0.0, 1.0, 0.0])).all(axis=1)]
                        cube_sets = []
                        if len(whites) > 0:
                            cube_sets.append((whites, [1.0, 1.0, 1.0]))
                        if len(reds) > 0:
                            cube_sets.append((reds, [1.0, 0.0, 0.0]))
                        if len(greens) > 0:
                            cube_sets.append((greens, [0.0, 1.0, 0.0]))
                        comp_voxel = self.raw_file_voxel_size if self.raw_file_voxel_size else (self.file_voxel_size if self.file_voxel_size else self.voxel_size)
                        self.update_cubes(cube_sets, comp_voxel)
                        self.get_logger().info(f'Initial file-to-file draw (cubes): {len(pts)} centers')
                    if not self.view_reset_done:
                        self.vis.reset_view_point(True)
                        self.view_reset_done = True
                # Stop polling timer
                if self.file_load_timer is not None:
                    self.file_load_timer.cancel()
                    self.file_load_timer = None
        except Exception as e:
            self.get_logger().error(f'Polling two files failed: {e}')

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
        # 点群レンダリングが選択されたときのみPointCloudを追加（CUBE描画時は追加しない）
        if str(self.render_style).lower() == 'points':
            self.vis.add_geometry(self.comparison_pcd)
        
        # Add coordinate frame if requested, scale with voxel size in world units
        if self.show_axes:
            base_voxel = self.file_voxel_size if self.file_voxel_size else (self.display_voxel_size if self.force_display_voxel_size else self.voxel_size)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=base_voxel * 10, origin=[0, 0, 0])
            self.vis.add_geometry(coord_frame)
        
        self.vis_initialized = True
        self.get_logger().info('Visualization window opened')
        
        # If we're in two-file mode and data is available, compute and draw immediately
        if self.mode == 'file_comparison' and self.two_file_comparison and \
           (self.file_points is not None) and (getattr(self, 'raw_file_points', None) is not None):
            try:
                pts, cols = self._build_two_file_precomp_arrays()
                if pts is not None and len(pts) > 0:
                    if self.render_style == 'cubes':
                        whites = pts[(cols == np.array([1.0, 1.0, 1.0])).all(axis=1)]
                        reds = pts[(cols == np.array([1.0, 0.0, 0.0])).all(axis=1)]
                        greens = pts[(cols == np.array([0.0, 1.0, 0.0])).all(axis=1)]
                        cube_sets = []
                        if len(whites) > 0:
                            cube_sets.append((whites, [1.0, 1.0, 1.0]))
                        if len(reds) > 0:
                            cube_sets.append((reds, [1.0, 0.0, 0.0]))
                        if len(greens) > 0:
                            cube_sets.append((greens, [0.0, 1.0, 0.0]))
                        self.update_cubes(cube_sets, self.raw_file_voxel_size or self.file_voxel_size or self.voxel_size)
                    else:
                        self.comparison_pcd.points = o3d.utility.Vector3dVector(pts)
                        self.comparison_pcd.colors = o3d.utility.Vector3dVector(cols)
                        self.vis.update_geometry(self.comparison_pcd)
                    self.vis.reset_view_point(True)
            except Exception as ex:
                self.get_logger().error(f'Initial two-file draw failed: {ex}')

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
        if is_file_mode(self.mode):
            if self.two_file_comparison:
                self.update_two_file_comparison()
            else:
                self.update_file_comparison()
        else:
            self.update_topic_comparison()

    def update_two_file_comparison(self):
        """Compare two HDF5 files: compressed vs raw."""
        # Ensure data loaded
        if self.file_points is None or getattr(self, 'raw_file_points', None) is None:
            self.load_two_hdf5_files()
            if self.file_points is None or getattr(self, 'raw_file_points', None) is None:
                return
        # Choose comparison voxel size (must be equal; prefer raw when present)
        if (self.file_voxel_size is not None) and (self.raw_file_voxel_size is not None):
            if not validate_voxel_sizes(self.file_voxel_size, self.raw_file_voxel_size, tol=1e-9):
                self.get_logger().error(
                    f'Voxel size mismatch between files: compressed={self.file_voxel_size} raw={self.raw_file_voxel_size}'
                )
                return
        comp_voxel = self.raw_file_voxel_size if self.raw_file_voxel_size else (self.file_voxel_size if self.file_voxel_size else self.voxel_size)
        # Use compressed file's grid origin as anchor for rounding
        pts_array, cols_array = compute_two_file_diff(self.file_points, self.raw_file_points, comp_voxel, origin=self.grid_origin)
        total = len(pts_array)
        if total == 0:
            return
        if self.render_style == 'cubes':
            # Derive cube sets by color
            whites = pts_array[(cols_array == np.array([1.0, 1.0, 1.0])).all(axis=1)]
            reds = pts_array[(cols_array == np.array([1.0, 0.0, 0.0])).all(axis=1)]
            greens = pts_array[(cols_array == np.array([0.0, 1.0, 0.0])).all(axis=1)]
            cube_sets = []
            if len(whites) > 0:
                cube_sets.append((whites, [1.0, 1.0, 1.0]))
            if len(reds) > 0:
                cube_sets.append((reds, [1.0, 0.0, 0.0]))
            if len(greens) > 0:
                cube_sets.append((greens, [0.0, 1.0, 0.0]))
            self.update_cubes(cube_sets, comp_voxel)
        else:
            self.update_points(pts_array, cols_array)
        bmin = pts_array.min(axis=0)
        bmax = pts_array.max(axis=0)
        whites_n = (cols_array == np.array([1.0, 1.0, 1.0])).all(axis=1).sum()
        reds_n = (cols_array == np.array([1.0, 0.0, 0.0])).all(axis=1).sum()
        greens_n = (cols_array == np.array([0.0, 1.0, 0.0])).all(axis=1).sum()
        self.get_logger().info(
            f'File-to-file comparison - Common: {whites_n} (white), Compressed-only: {reds_n} (red), Raw-only: {greens_n} (green), Total: {total}'
        )
        self.get_logger().info(f'Bounds: min={bmin}, max={bmax}, range={(bmax-bmin)}')

    def _build_two_file_precomp_arrays(self):
        """Build colored points arrays for two-file comparison (points, colors)."""
        try:
            comp_voxel = self.raw_file_voxel_size if self.raw_file_voxel_size else (self.file_voxel_size if self.file_voxel_size else self.voxel_size)
            f1 = self.round_points(np.asarray(self.file_points), comp_voxel)
            f2 = self.round_points(np.asarray(self.raw_file_points), comp_voxel)
            set1 = set(map(tuple, f1))
            set2 = set(map(tuple, f2))
            common = set1 & set2
            only_f1 = set1 - set2
            only_f2 = set2 - set1
            pts = []
            cols = []
            for p in common:
                pts.append(p); cols.append([1.0, 1.0, 1.0])
            for p in only_f1:
                pts.append(p); cols.append([1.0, 0.0, 0.0])
            for p in only_f2:
                pts.append(p); cols.append([0.0, 1.0, 0.0])
            if pts:
                arr_pts = np.array(pts)
                arr_cols = np.array(cols)
                self.get_logger().info(
                    f'File-to-file (precomp) - Common: {len(common)}, Compressed-only: {len(only_f1)}, Raw-only: {len(only_f2)}, Total: {len(pts)}'
                )
                return arr_pts, arr_cols
        except Exception as e:
            self.get_logger().error(f'Precomp arrays failed: {e}')
        return None, None

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
            if str(self.render_style).lower() == 'cubes':
                cube_sets = []
                if len(common) > 0:
                    cube_sets.append((np.array(list(common)), [1.0, 1.0, 1.0]))  # white
                if len(only_occupied) > 0:
                    cube_sets.append((np.array(list(only_occupied)), [1.0, 0.0, 0.0]))  # red
                if len(only_pattern) > 0:
                    cube_sets.append((np.array(list(only_pattern)), [0.0, 0.0, 1.0]))  # blue
                self.update_cubes(cube_sets, self.current_voxel_size)
                total = len(common) + len(only_occupied) + len(only_pattern)
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
                total = len(all_points)
            
            # Log statistics
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
        # Anchor rounding to grid origin when available for exact alignment
        origin = self.grid_origin if isinstance(self.grid_origin, np.ndarray) else np.array([0.0, 0.0, 0.0], dtype=np.float64)
        return np.round((points - origin) / voxel_size) * voxel_size + origin

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
        # Centers provided by roundingは格子の角 (k * voxel_size) に揃っている。
        # +0.5*voxel_sizeシフトで中心に配置し、隙間をなくす。
        half = 0.5 * float(voxel_size)
        for c in centers:
            # Open3Dのメッシュは浅いコピーだと最後の平行移動で上書きされることがあるため、
            # 各ボクセルごとに新規ボックスを生成して配置する。
            cube = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
            cube.compute_vertex_normals()
            cx = float(c[0]) + half
            cy = float(c[1]) + half
            cz = float(c[2]) + half
            cube.translate([cx - size/2.0, cy - size/2.0, cz - size/2.0])
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
