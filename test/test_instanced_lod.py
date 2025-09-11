#!/usr/bin/env python3
"""TDD: instanced_gpu_python 経路で LOD 分岐（near=cubes, far=points(Scene)）が働くことを検証。"""

import sys
import types
import numpy as np
import os

# パッケージパス
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def make_o3d_stub():
    o3d = types.SimpleNamespace()
    # legacy Visualizer stub
    o3d.visualization = types.SimpleNamespace(Visualizer=lambda: types.SimpleNamespace(
        create_window=lambda *a, **k: None,
        get_render_option=lambda: types.SimpleNamespace(__setattr__=object.__setattr__),
        add_geometry=lambda *a, **k: None,
        remove_geometry=lambda *a, **k: None,
        update_geometry=lambda *a, **k: None,
        reset_view_point=lambda *a, **k: None,
        poll_events=lambda : False,
        update_renderer=lambda : None,
        destroy_window=lambda : None,
    ))
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
    sys.modules['rclpy.node'] = rclpy.node
    return rclpy


def make_scene_renderer_stub(call_log: dict):
    mod = types.ModuleType('voxel_viewer.scene_renderer')
    class SceneRenderer:
        def __init__(self, *a, **k):
            call_log['init'] += 1
        def update_points(self, points, colors, point_size=2.0):
            call_log['update'].append((np.array(points), np.array(colors), point_size))
    mod.SceneRenderer = SceneRenderer
    return mod


def test_lod_near_cubes_far_points_scene():
    sys.modules['rclpy'] = make_rclpy_stub()
    sys.modules['open3d'] = make_o3d_stub()
    call_log = {'init': 0, 'update': []}
    sys.modules['voxel_viewer.scene_renderer'] = make_scene_renderer_stub(call_log)

    from voxel_viewer.voxel_viewer_with_hdf5 import VoxelViewerWithHDF5Node
    n = VoxelViewerWithHDF5Node()
    n.render_mode = 'instanced_gpu_python'
    # 近距離2点、遠距離2点
    whites = np.array([[0.2, 0, 0], [0.5, 0, 0]])
    reds = np.array([[5.0, 0, 0], [6.0, 0, 0]])
    voxel = 0.1
    cube_sets = [
        (whites, [1.0, 1.0, 1.0]),
        (reds, [1.0, 0.0, 0.0])
    ]
    # LOD設定: 1.0以内がnear
    n.inst_lod_distance = 1.0
    n.inst_max_near_cubes = 10
    n.inst_point_size = 3.0
    called = {'cubes': 0}
    def fake_cubes(csets, vsize):
        called['cubes'] += sum(len(c[0]) for c in csets)
    n.update_cubes = fake_cubes

    n.update_instanced_python(cube_sets, voxel)

    # near(2)はキューブに回り、far(2)はSceneRenderer.update_pointsに渡る
    assert called['cubes'] == 2
    assert len(call_log['update']) == 1
    pts, cols, ps = call_log['update'][0]
    assert pts.shape[0] == 2
    assert ps == 3.0

