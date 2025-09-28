#!/usr/bin/env python3
import os
import types
import tempfile
import numpy as np
import h5py
import pytest


def _write_minimal_hdf5(filepath: str, voxel_size: float, block_size: int, grid_origin: np.ndarray):
    with h5py.File(filepath, 'w') as f:
        # compression_params
        grp = f.create_group('compression_params')
        grp.create_dataset('voxel_size', data=np.array(voxel_size, dtype=np.float32))
        grp.create_dataset('block_size', data=np.array(block_size, dtype=np.uint32))
        grp.create_dataset('grid_origin', data=np.asarray(grid_origin, dtype=np.float32))
        grp.create_dataset('block_index_bit_width', data=np.array(16, dtype=np.uint32))

        # dictionary: single pattern, bit 0 = 1 (LSB-first)
        dict_grp = f.create_group('dictionary')
        pattern_length = block_size ** 3
        bytes_per_pattern = (pattern_length + 7) // 8
        patterns = np.zeros(bytes_per_pattern, dtype=np.uint8)
        patterns[0] = 0b00000001  # set first voxel occupied
        dict_grp.create_dataset('pattern_length', data=np.array(pattern_length, dtype=np.uint32))
        dict_grp.create_dataset('patterns', data=patterns)

        # compressed_data: one voxel block at [0,0,0], index 0 using block grid
        comp_grp = f.create_group('compressed_data')
        sentinel = np.array([0xFFFF], dtype=np.uint16)
        block_indices = np.full((1,), sentinel[0], dtype=np.uint16)
        block_indices[0] = 0
        comp_grp.create_dataset('block_indices', data=block_indices)
        comp_grp.create_dataset('block_offset', data=np.array([0, 0, 0], dtype=np.int32))
        comp_grp.create_dataset('block_dims', data=np.array([1, 1, 1], dtype=np.int32))
        comp_grp.create_dataset('point_count', data=np.array(1, dtype=np.uint64))

        stats_grp = f.create_group('statistics')
        bbox = np.array([
            grid_origin.astype(np.float64) - 0.006,
            grid_origin.astype(np.float64) + voxel_size + 0.006
        ])
        stats_grp.create_dataset('bounding_box', data=bbox)


def test_hdf5_reader_decompress_with_grid_origin(monkeypatch):
    # Stub open3d before importing the reader
    # Minimal open3d stub for type hints
    o3d_stub = types.SimpleNamespace()
    class _DummyPointCloud:
        pass
    o3d_stub.geometry = types.SimpleNamespace(PointCloud=_DummyPointCloud)
    monkeypatch.setitem(__import__('sys').modules, 'open3d', o3d_stub)

    from voxel_viewer.hdf5_reader import HDF5CompressedMapReader

    v = 5.5
    bs = 8
    origin = np.array([1.2, -3.4, 10.0], dtype=np.float32)

    # Create temp file under workspace
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as td:
        path = os.path.join(td, 'test_origin.h5')
        _write_minimal_hdf5(path, v, bs, origin)

        reader = HDF5CompressedMapReader(path)
        assert reader.read() is True
        pts = reader.decompress()
        assert pts.shape == (1, 3)
        expected = origin.astype(np.float64) + 0.5 * v
        np.testing.assert_allclose(pts[0], expected, rtol=0, atol=1e-9)

        stats = reader.get_statistics()
        assert stats is not None
        bbox_min = stats.get('bounding_box_min')
        bbox_max = stats.get('bounding_box_max')
        assert bbox_min is not None and bbox_max is not None
        np.testing.assert_allclose(bbox_min, np.array([origin[0] - 0.006, origin[1] - 0.006, origin[2] - 0.006]))
        np.testing.assert_allclose(bbox_max, np.array([origin[0] + v + 0.006, origin[1] + v + 0.006, origin[2] + v + 0.006]))


def test_hdf5_reader_rejects_legacy_format(monkeypatch):
    # Stub open3d before importing the reader
    o3d_stub = types.SimpleNamespace()
    class _DummyPointCloud:
        pass
    o3d_stub.geometry = types.SimpleNamespace(PointCloud=_DummyPointCloud)
    monkeypatch.setitem(__import__('sys').modules, 'open3d', o3d_stub)

    from voxel_viewer.hdf5_reader import HDF5CompressedMapReader

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as td:
        path = os.path.join(td, 'legacy_only.h5')
        with h5py.File(path, 'w') as f:
            params = f.create_group('compression_params')
            params.create_dataset('voxel_size', data=np.array(0.5, dtype=np.float32))
            params.create_dataset('block_size', data=np.array(8, dtype=np.uint32))
            params.create_dataset('grid_origin', data=np.array([0.0, 0.0, 0.0], dtype=np.float32))

            dict_grp = f.create_group('dictionary')
            dict_grp.create_dataset('pattern_length', data=np.array(512, dtype=np.uint32))
            dict_grp.create_dataset('patterns', data=np.ones(64, dtype=np.uint8))

            comp_grp = f.create_group('compressed_data')
            comp_grp.create_dataset('indices', data=np.array([0], dtype=np.uint16))
            comp_grp.create_dataset('voxel_positions', data=np.array([[0, 0, 0]], dtype=np.int32))

        reader = HDF5CompressedMapReader(path)
        assert reader.read() is True
        with pytest.raises(ValueError):
            reader.decompress()
