#!/usr/bin/env python3
"""Tests for file_comparison mode handling of new HDF5 block index format."""

import os
import sys
import types
import tempfile
import numpy as np
import h5py


class _DummyLogger:
    def __init__(self):
        self.errors = []

    def info(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        self.errors.append(args)

    def debug(self, *args, **kwargs):
        pass


def make_rclpy_stub(overrides=None):
    overrides = overrides or {}
    rclpy_mod = types.ModuleType('rclpy')

    class _Param:
        def __init__(self, value):
            self.value = value

    class _NodeBase:
        def __init__(self, name='node'):
            self._declared = {}
            self._subs = []
            self._timers = []
            self._logger = _DummyLogger()

        def declare_parameter(self, name, default_value=None):
            value = overrides.get(name, default_value)
            self._declared[name] = value
            return _Param(value)

        def get_parameter(self, name):
            return _Param(self._declared.get(name))

        def create_subscription(self, *args, **kwargs):
            self._subs.append((args, kwargs))
            return object()

        def create_timer(self, *args, **kwargs):
            self._timers.append((args, kwargs))

            class _Timer:
                def cancel(self_inner):
                    pass

            return _Timer()

        def get_logger(self):
            return self._logger

    node_mod = types.ModuleType('rclpy.node')
    node_mod.Node = _NodeBase
    rclpy_mod.node = node_mod
    return rclpy_mod, node_mod


def install_open3d_stub():
    class _PointCloud:
        def __init__(self):
            self.points = None
            self.colors = None

    class _TriangleMesh:
        def __init__(self):
            self.vertices = None
            self.triangles = None

        def compute_vertex_normals(self):
            pass

        def paint_uniform_color(self, color):
            pass

        @staticmethod
        def create_coordinate_frame(size=1.0, origin=(0.0, 0.0, 0.0)):
            return object()

    o3d_mod = types.ModuleType('open3d')
    o3d_mod.geometry = types.SimpleNamespace(PointCloud=_PointCloud, TriangleMesh=_TriangleMesh)
    o3d_mod.utility = types.SimpleNamespace(Vector3dVector=lambda x: x, Vector3iVector=lambda x: x)
    o3d_mod.visualization = types.SimpleNamespace()
    return o3d_mod


def install_marker_stub():
    Marker = type('Marker', (), {})
    MarkerArray = type('MarkerArray', (), {})
    marker_mod = types.SimpleNamespace(Marker=Marker, MarkerArray=MarkerArray)
    return marker_mod


def create_new_format_hdf5(path):
    with h5py.File(path, 'w') as f:
        params = f.create_group('compression_params')
        params.create_dataset('voxel_size', data=np.array(0.5, dtype=np.float32))
        params.create_dataset('block_size', data=np.array(8, dtype=np.uint32))
        params.create_dataset('dictionary_size', data=np.array(1, dtype=np.uint32))
        params.create_dataset('pattern_bits', data=np.array(512, dtype=np.uint32))
        params.create_dataset('grid_origin', data=np.array([1.0, -2.0, 3.0], dtype=np.float32))
        params.create_dataset('block_index_bit_width', data=np.array(16, dtype=np.uint32))

        dictionary = f.create_group('dictionary')
        dictionary.create_dataset('pattern_length', data=np.array(512, dtype=np.uint32))
        pattern_bytes = np.zeros(64, dtype=np.uint8)
        pattern_bytes[0] = 0b00000001
        dictionary.create_dataset('patterns', data=pattern_bytes)

        comp = f.create_group('compressed_data')
        sentinel = np.iinfo(np.uint16).max
        block_indices = np.array([0], dtype=np.uint16)
        comp.create_dataset('block_indices', data=block_indices)
        comp.create_dataset('block_offset', data=np.array([0, 0, 0], dtype=np.int32))
        comp.create_dataset('block_dims', data=np.array([1, 1, 1], dtype=np.int32))
        comp.create_dataset('point_count', data=np.array(1, dtype=np.uint64))

        stats = f.create_group('statistics')
        stats.create_dataset('original_points', data=np.array(1, dtype=np.uint64))
        stats.create_dataset('compressed_voxels', data=np.array(1, dtype=np.uint64))
        stats.create_dataset('compression_ratio', data=np.array(1.0, dtype=np.float64))
        stats.create_dataset('bounding_box', data=np.zeros((2, 3), dtype=np.float64))


def create_legacy_hdf5(path):
    with h5py.File(path, 'w') as f:
        params = f.create_group('compression_params')
        params.create_dataset('voxel_size', data=np.array(0.5, dtype=np.float32))
        params.create_dataset('block_size', data=np.array(8, dtype=np.uint32))
        params.create_dataset('grid_origin', data=np.array([0.0, 0.0, 0.0], dtype=np.float32))

        dictionary = f.create_group('dictionary')
        dictionary.create_dataset('pattern_length', data=np.array(512, dtype=np.uint32))
        dictionary.create_dataset('patterns', data=np.ones(64, dtype=np.uint8))

        comp = f.create_group('compressed_data')
        comp.create_dataset('indices', data=np.array([0], dtype=np.uint16))
        comp.create_dataset('voxel_positions', data=np.array([[0, 0, 0]], dtype=np.int32))


def _install_stubs(overrides):
    orig_modules = {}
    rclpy_mod, rclpy_node_mod = make_rclpy_stub(overrides)
    orig_modules['rclpy'] = sys.modules.get('rclpy')
    orig_modules['rclpy.node'] = sys.modules.get('rclpy.node')
    sys.modules['rclpy'] = rclpy_mod
    sys.modules['rclpy.node'] = rclpy_node_mod

    o3d_mod = install_open3d_stub()
    orig_modules['open3d'] = sys.modules.get('open3d')
    sys.modules['open3d'] = o3d_mod

    marker_stub = install_marker_stub()
    orig_modules['visualization_msgs'] = sys.modules.get('visualization_msgs')
    orig_modules['visualization_msgs.msg'] = sys.modules.get('visualization_msgs.msg')
    sys.modules['visualization_msgs'] = types.SimpleNamespace(msg=marker_stub)
    sys.modules['visualization_msgs.msg'] = marker_stub
    return orig_modules


def _restore_modules(orig_modules):
    for name, mod in orig_modules.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def test_file_comparison_loads_new_format():
    with tempfile.TemporaryDirectory() as td:
        h5_path = os.path.join(td, 'new_format.h5')
        create_new_format_hdf5(h5_path)

        overrides = {
            'mode': 'file_comparison',
            'hdf5_file': h5_path,
            'render_mode': 'points',
            'common_downsample_ratio': 1.0,
        }
        orig = _install_stubs(overrides)
        try:
            from voxel_viewer.voxel_viewer_with_hdf5 import VoxelViewerWithHDF5Node
            node = VoxelViewerWithHDF5Node()

            assert node.file_loaded is True
            assert node._compressed_loaded_once is True
            assert node.file_points is not None
            assert len(node.file_points) == 1

            node.render_mode = 'points'
            assert node.render_mode == 'points'
            node.common_downsample_ratio = 1.0
            node.occupied_points = node.file_points.copy()
            node.occupied_received = True
            node.current_voxel_size = float(node.file_voxel_size)

            captured = {}
            render_calls = {'count': 0}

            def fake_update_points(points, colors):
                captured['points'] = points
                captured['colors'] = colors

            def fake_update_render(*args, **kwargs):
                render_calls['count'] += 1

            node.update_points = fake_update_points
            node.update_render = fake_update_render

            node.update_file_comparison()

            assert 'points' in captured
            assert render_calls['count'] == 0
            pts = captured['points']
            cols = captured['colors']
            assert pts.shape[0] == cols.shape[0] and pts.shape[0] >= 1
            unique_cols = np.unique(cols, axis=0)
            assert unique_cols.shape == (1, 3)
            assert np.allclose(unique_cols[0], np.array([1.0, 1.0, 1.0]))
        finally:
            _restore_modules(orig)


def test_file_comparison_rejects_legacy_format():
    with tempfile.TemporaryDirectory() as td:
        h5_path = os.path.join(td, 'legacy.h5')
        create_legacy_hdf5(h5_path)

        overrides = {
            'mode': 'file_comparison',
            'hdf5_file': h5_path,
        }
        orig = _install_stubs(overrides)
        try:
            from voxel_viewer.voxel_viewer_with_hdf5 import VoxelViewerWithHDF5Node
            node = VoxelViewerWithHDF5Node()

            assert node.file_loaded is False
            assert node._compressed_loaded_once is False
            assert node.file_points is None
            # Error should have been logged
            assert node.get_logger().errors
        finally:
            _restore_modules(orig)
