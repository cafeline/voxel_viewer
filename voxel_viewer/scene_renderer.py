#!/usr/bin/env python3
"""Open3D rendering(Scene) ベースの描画ヘルパ。

現状はOffscreenRendererでSceneを管理し、点群(PointCloud)を追加・更新する。
Window表示を伴うO3DVisualizerはイベントループが必要になるため、ここでは
レンダリングAPI(Scene)を用いたリソース管理の最小実装に留める。

将来的にGUI統合する場合は、O3DVisualizerのsceneをこのクラスに差し替えて
同様のAPI(update_points 等)で扱えるようにする。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from typing import Optional


class SceneRenderer:
    def __init__(self, width: int = 1280, height: int = 720, background=(0.1, 0.1, 0.1)):
        self.width = int(width)
        self.height = int(height)
        self.background = tuple(float(x) for x in background)
        self._renderer: Optional[o3d.visualization.rendering.OffscreenRenderer] = None
        self._pcd_added = False
        self._geom_name = 'voxels_points'

    def ensure_scene(self):
        if self._renderer is None:
            r = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
            r.scene.set_background_color(self.background)
            self._renderer = r
        return self._renderer.scene

    def _build_material(self, point_size: float = 2.0):
        mr = o3d.visualization.rendering.MaterialRecord()
        # Unlit point shader
        mr.shader = 'defaultUnlit'
        # point_sizeはLine/Point材質で反映
        mr.point_size = float(point_size)
        return mr

    def update_points(self, points: np.ndarray, colors: np.ndarray, point_size: float = 2.0):
        """Sceneに点群として追加/更新する。"""
        scene = self.ensure_scene()
        # Open3D geometry
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        mat = self._build_material(point_size)
        if not self._pcd_added:
            scene.add_geometry(self._geom_name, pcd, mat)
            self._pcd_added = True
        else:
            # いったん削除して再追加（update_geometryでもよいがスタブ互換性のため）
            try:
                scene.remove_geometry(self._geom_name)
            except Exception:
                pass
            scene.add_geometry(self._geom_name, pcd, mat)

    def render_to_image(self):
        if self._renderer is None:
            return None
        return self._renderer.render_to_image()

