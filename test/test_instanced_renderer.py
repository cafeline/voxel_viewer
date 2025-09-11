#!/usr/bin/env python3
"""TDD: InstancedRenderer の行列・色生成、およびビューア経路切替の検証。"""

import sys
import types
import numpy as np
import os

# Add parent directory to path to import package modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def make_o3d_stub():
    o3d = types.SimpleNamespace()
    # minimal Visualizer stub
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
    # geometry/utility stubs used by renderer
    class _PC:
        def __init__(self):
            self.points = None
            self.colors = None
        def clear(self):
            self.points = None
            self.colors = None

    o3d.geometry = types.SimpleNamespace(PointCloud=_PC, TriangleMesh=type('TM', (), {}))
    o3d.utility = types.SimpleNamespace(Vector3dVector=lambda x: x, Vector3iVector=lambda x: x)
    return o3d


def make_rclpy_stub():
    rclpy = types.SimpleNamespace()
    class _Param:
        def __init__(self, v): self.value = v
    class _NodeBase:
        def __init__(self, name='node'):
            self._declared = {}
            self._logger = types.SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
            self._subs = []
        def declare_parameter(self, name, default_value=None):
            self._declared[name] = default_value
            return _Param(default_value)
        def get_parameter(self, name):
            return _Param(self._declared.get(name))
        def create_subscription(self, *a, **k):
            self._subs.append((a, k)); return object()
        def get_logger(self):
            return self._logger
    rclpy.node = types.SimpleNamespace(Node=_NodeBase)
    # Also register submodule to satisfy 'from rclpy.node import Node'
    sys.modules['rclpy.node'] = rclpy.node
    return rclpy


def test_instanced_renderer_transforms_and_colors():
    sys.modules['open3d'] = make_o3d_stub()
    from voxel_viewer.instanced_renderer import InstancedRenderer
    # 2セット: 白2点, 赤1点
    voxel = 0.2
    whites = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    reds = np.array([[10.0, 0.0, 0.0]])
    r = InstancedRenderer(vis=None)
    T, C = r.build_instances([
        (whites, [1.0, 1.0, 1.0]),
        (reds, [1.0, 0.0, 0.0])
    ], voxel)
    assert T.shape == (3, 4, 4)
    assert C.shape == (3, 3)
    # スケールが対角に入る
    assert np.allclose(np.diag(T[0])[:3], [voxel, voxel, voxel])
    # 平行移動は+0.5*voxel
    assert np.allclose(T[0][:3, 3], [0.1, 0.1, 0.1])
    assert np.allclose(T[1][:3, 3], [1.1, 2.1, 3.1])
    assert np.allclose(T[2][:3, 3], [10.1, 0.1, 0.1])
    # 色
    assert np.allclose(C[0], [1.0, 1.0, 1.0])
    assert np.allclose(C[2], [1.0, 0.0, 0.0])


def test_viewer_switches_to_instanced_path():
    # rclpy/o3dスタブ
    sys.modules['rclpy'] = make_rclpy_stub()
    sys.modules['open3d'] = make_o3d_stub()
    import numpy as np
    from voxel_viewer.voxel_viewer_with_hdf5 import VoxelViewerWithHDF5Node
    n = VoxelViewerWithHDF5Node()
    # instancedへ切替
    n.render_mode = 'instanced'
    # ダミーデータ投入
    n.occupied_points = np.array([[0,0,0],[1,0,0]], dtype=float)
    n.pattern_points = np.array([[0,0,0]], dtype=float)
    n.occupied_received = True
    n.pattern_received = True
    n.current_voxel_size = 1.0
    called = {'instanced':0, 'cubes':0}
    def fake_instanced(c, v):
        called['instanced'] += 1
    def fake_cubes(c, v):
        called['cubes'] += 1
    n.update_instanced = fake_instanced
    n.update_cubes = fake_cubes
    n.update_topic_comparison()
    assert called['instanced'] > 0
    assert called['cubes'] == 0


def test_viewer_switches_to_instanced_python_path():
    # rclpy/o3dスタブ
    sys.modules['rclpy'] = make_rclpy_stub()
    sys.modules['open3d'] = make_o3d_stub()
    import numpy as np
    from voxel_viewer.voxel_viewer_with_hdf5 import VoxelViewerWithHDF5Node
    n = VoxelViewerWithHDF5Node()
    # instanced_gpu_pythonへ切替
    n.render_mode = 'instanced_gpu_python'
    # ダミーデータ投入
    n.occupied_points = np.array([[0,0,0],[1,0,0]], dtype=float)
    n.pattern_points = np.array([[0,0,0]], dtype=float)
    n.occupied_received = True
    n.pattern_received = True
    n.current_voxel_size = 1.0
    called = {'inst_py':0, 'cubes':0}
    def fake_inst_py(c, v):
        called['inst_py'] += 1
    def fake_cubes(c, v):
        called['cubes'] += 1
    n.update_instanced_python = fake_inst_py
    n.update_cubes = fake_cubes
    n.update_topic_comparison()
    assert called['inst_py'] > 0
    assert called['cubes'] == 0
