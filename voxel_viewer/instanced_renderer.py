#!/usr/bin/env python3
"""Instanced renderer helper (logical instancing with per-instance transforms and colors).

現状のOpen3D Visualizer（legacy）にはPythonからのGPUインスタンシングAPIが無いので、
ここでは以下を担う:
- インスタンス行列（Nx4x4）と色（Nx3）の生成
- Visualizer への描画は安全なフォールバックとして PointCloud を使用（色付き）
  （将来的にrenderingモジュールへ差し替えやすいように分離）
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from typing import List, Tuple


class InstancedRenderer:
    def __init__(self, vis: o3d.visualization.Visualizer | None = None):
        self.vis = vis
        self._pcd = o3d.geometry.PointCloud()
        self.instance_transforms: np.ndarray | None = None  # (N,4,4)
        self.instance_colors: np.ndarray | None = None      # (N,3)

    @staticmethod
    def _to_homogeneous(translations: np.ndarray, scale: float) -> np.ndarray:
        """Create per-instance 4x4 transforms for unit cube centered原点を想定。
        - translations: (N,3) world centers（下端原点の点列ではない）
        - scale: voxelサイズを一様スケールとして対角に持つ
        """
        N = int(translations.shape[0])
        T = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
        T[:, 0, 0] = scale
        T[:, 1, 1] = scale
        T[:, 2, 2] = scale
        T[:, 0, 3] = translations[:, 0]
        T[:, 1, 3] = translations[:, 1]
        T[:, 2, 3] = translations[:, 2]
        return T

    def build_instances(self, cube_sets: List[Tuple[np.ndarray, list]], voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """From cube_sets -> (instance_transforms, instance_colors).
        cube_sets: [(centers Nx3, color [r,g,b]), ...]
        centersは既存実装と同様に「ボクセル下端の座標」を想定するため、
        メッシュが原点中心の単位立方体なら+0.5*voxel のオフセットで中心に合わせる。
        戻り値は (N,4,4), (N,3)
        """
        if not cube_sets:
            self.instance_transforms = np.zeros((0, 4, 4), dtype=np.float64)
            self.instance_colors = np.zeros((0, 3), dtype=np.float64)
            return self.instance_transforms, self.instance_colors

        centers_list = []
        colors_list = []
        half = 0.5 * float(voxel_size)
        for centers, color in cube_sets:
            if centers is None or np.asarray(centers).size == 0:
                continue
            c = np.asarray(centers, dtype=np.float64)
            # 下端からボクセル中心へ
            c = c + half
            centers_list.append(c)
            col = np.asarray(color, dtype=np.float64).reshape(1, 3)
            cols = np.repeat(col, c.shape[0], axis=0)
            colors_list.append(cols)

        if not centers_list:
            self.instance_transforms = np.zeros((0, 4, 4), dtype=np.float64)
            self.instance_colors = np.zeros((0, 3), dtype=np.float64)
            return self.instance_transforms, self.instance_colors

        all_centers = np.vstack(centers_list)
        all_colors = np.vstack(colors_list)
        transforms = self._to_homogeneous(all_centers, float(voxel_size))
        self.instance_transforms = transforms
        self.instance_colors = all_colors
        return transforms, all_colors

    # --- Fallback draw ---
    def draw_as_points(self):
        """安全なフォールバック: インスタンス位置にポイントを色付きで描画。"""
        if self.instance_transforms is None or self.instance_colors is None:
            return
        N = self.instance_transforms.shape[0]
        if N == 0:
            return
        centers = self.instance_transforms[:, :3, 3]
        self._pcd.points = o3d.utility.Vector3dVector(centers)
        self._pcd.colors = o3d.utility.Vector3dVector(self.instance_colors)
        if self.vis is not None:
            try:
                self.vis.add_geometry(self._pcd)
            except Exception:
                try:
                    self.vis.update_geometry(self._pcd)
                except Exception:
                    pass

