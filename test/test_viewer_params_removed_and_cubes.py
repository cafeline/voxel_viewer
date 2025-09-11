#!/usr/bin/env python3
"""TDD: ensure removed params are not declared and rendering path is cubes-only."""

import sys
import types
import numpy as np


class _DummyLogger:
    def info(self, *a, **k):
        pass
    def error(self, *a, **k):
        pass
    def warn(self, *a, **k):
        pass
    def debug(self, *a, **k):
        pass


def make_rclpy_stub():
    rclpy = types.SimpleNamespace()

    class _Param:
        def __init__(self, v):
            self.value = v

    class _NodeBase:
        def __init__(self, name='node'):
            self._declared = {}
            self._subs = []
            self._timers = []
            self._logger = _DummyLogger()

        def declare_parameter(self, name, default_value=None):
            self._declared[name] = default_value
            return _Param(default_value)

        def get_parameter(self, name):
            # return object with .value
            return _Param(self._declared.get(name))

        def create_subscription(self, *a, **k):
            self._subs.append((a, k))
            return object()

        def create_timer(self, *a, **k):
            self._timers.append((a, k))
            class T:
                def cancel(self_inner):
                    pass
            return T()

        def get_logger(self):
            return self._logger

    # emulate rclpy.node.Node
    node_mod = types.SimpleNamespace(Node=_NodeBase)
    rclpy.node = node_mod
    return rclpy


def test_removed_params_and_cubes_only():
    # stub rclpy before import
    sys.modules['rclpy'] = make_rclpy_stub()
    # minimal open3d stub to import module without GUI
    o3d = types.SimpleNamespace()
    o3d.visualization = types.SimpleNamespace(Visualizer=lambda: types.SimpleNamespace(
        create_window=lambda *a, **k: None,
        get_render_option=lambda: types.SimpleNamespace(
            __setattr__=object.__setattr__
        ),
        add_geometry=lambda *a, **k: None,
        remove_geometry=lambda *a, **k: None,
        update_geometry=lambda *a, **k: None,
        reset_view_point=lambda *a, **k: None,
        poll_events=lambda : False,
        update_renderer=lambda : None,
        destroy_window=lambda : None,
    ))
    o3d.geometry = types.SimpleNamespace(
        PointCloud=type('PC', (), {}) ,
        TriangleMesh=type('TM', (), {'create_coordinate_frame': staticmethod(lambda size, origin: object())})
    )
    o3d.utility = types.SimpleNamespace(Vector3dVector=lambda x: x, Vector3iVector=lambda x: x)
    sys.modules['open3d'] = o3d

    from voxel_viewer.voxel_viewer_with_hdf5 import VoxelViewerWithHDF5Node
    n = VoxelViewerWithHDF5Node()
    # Ensure removed params are not declared
    declared = getattr(n, '_declared', {})
    banned = [
        'force_display_voxel_size','display_voxel_size','render_style','cube_scale',
        'point_size','background_color','show_axes','tolerance','voxel_size'
    ]
    for b in banned:
        assert b not in declared
    # And expected ones exist (may be defaults)
    for k in ['mode','hdf5_file','raw_hdf5_file']:
        assert k in declared

    # Make sure cubes path is used: update_topic_comparison must call update_cubes, not update_points
    n.occupied_points = np.array([[0,0,0],[1,0,0]], dtype=float)
    n.pattern_points = np.array([[0,0,0]], dtype=float)
    n.occupied_received = True
    n.pattern_received = True
    n.current_voxel_size = 1.0
    called = {'cubes':0, 'points':0}
    def fake_cubes(c, v):
        called['cubes'] += 1
    def fake_points(p, c):
        called['points'] += 1
    n.update_cubes = fake_cubes
    n.update_points = fake_points
    n.update_topic_comparison()
    assert called['cubes'] > 0 and called['points'] == 0

