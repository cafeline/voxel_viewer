#!/usr/bin/env python3
"""TDD: SceneRenderer が Open3D rendering(Scene) API を用いて点群を追加・更新することを検証。

GUI/実レンダリングは行わず、OffscreenRenderer/Scene をスタブ化して呼び出しが行われることのみ検証する。
"""

import sys
import types
import numpy as np
import os

# パッケージパス
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def make_o3d_rendering_stub(call_log: dict):
    o3d = types.SimpleNamespace()

    # MaterialRecord stub
    class _Mat:
        def __init__(self):
            self.shader = ''
            self.point_size = 0.0

    # geometry/utility
    class _PC:
        def __init__(self):
            self.points = None
            self.colors = None

    o3d.geometry = types.SimpleNamespace(PointCloud=_PC)
    o3d.utility = types.SimpleNamespace(Vector3dVector=lambda x: x)

    # Scene stub
    class _Scene:
        def __init__(self):
            self._bg = (0, 0, 0)
            self.geoms = {}
        def set_background_color(self, c):
            self._bg = c
        def add_geometry(self, name, geom, mat):
            call_log['add'].append((name, geom, mat))
            self.geoms[name] = (geom, mat)
        def remove_geometry(self, name):
            call_log['remove'].append(name)
            self.geoms.pop(name, None)

    # OffscreenRenderer stub
    class _Off:
        def __init__(self, w, h):
            call_log['offscreen'].append((w, h))
            self.scene = _Scene()
        def render_to_image(self):
            call_log['render'].append('image')
            return object()

    o3d.visualization = types.SimpleNamespace(
        rendering=types.SimpleNamespace(
            OffscreenRenderer=_Off,
            MaterialRecord=_Mat
        )
    )
    return o3d


def test_scene_renderer_add_and_update_points():
    call_log = {'add': [], 'remove': [], 'offscreen': [], 'render': []}
    sys.modules['open3d'] = make_o3d_rendering_stub(call_log)
    from voxel_viewer.scene_renderer import SceneRenderer
    r = SceneRenderer(width=320, height=240, background=(0.1, 0.2, 0.3))
    pts = np.array([[0,0,0],[1,2,3]], dtype=float)
    cols = np.array([[1,1,1],[1,0,0]], dtype=float)
    r.update_points(pts, cols, point_size=3.0)
    # 1回目: add_geometry が呼ばれる
    assert len(call_log['add']) == 1
    name, geom, mat = call_log['add'][0]
    assert name == 'voxels_points'
    assert mat.shader == 'defaultUnlit'
    assert mat.point_size == 3.0
    # 2回目: remove → add の順で更新される
    r.update_points(pts * 2.0, cols, point_size=5.0)
    assert len(call_log['remove']) == 1
    assert len(call_log['add']) == 2

