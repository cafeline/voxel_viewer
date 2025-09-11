#!/usr/bin/env python3
import os
import sys
import types
import tempfile
import numpy as np
import h5py


def write_min_h5(path, voxel_size, block_size, grid_origin):
    with h5py.File(path, 'w') as f:
        grp = f.create_group('compression_params')
        grp.create_dataset('voxel_size', data=np.array(voxel_size, dtype=np.float32))
        grp.create_dataset('block_size', data=np.array(block_size, dtype=np.uint32))
        grp.create_dataset('grid_origin', data=np.asarray(grid_origin, dtype=np.float32))

        dict_grp = f.create_group('dictionary')
        pattern_length = block_size ** 3
        bpp = (pattern_length + 7) // 8
        patterns = np.zeros(bpp, dtype=np.uint8)
        patterns[0] = 1
        dict_grp.create_dataset('pattern_length', data=np.array(pattern_length, dtype=np.uint32))
        dict_grp.create_dataset('patterns', data=patterns)

        comp_grp = f.create_group('compressed_data')
        comp_grp.create_dataset('indices', data=np.array([0], dtype=np.uint16))
        comp_grp.create_dataset('voxel_positions', data=np.array([[0, 0, 0]], dtype=np.int32))


def main():
    # Test file_compare origin-aware rounding
    from voxel_viewer.file_compare import round_points_to_voxel, compute_two_file_diff
    v = 2.0
    origin = np.array([10.0, -5.0, 1.0])
    pts = np.array([
        origin + np.array([0.49, 0.49, 0.49]) * v,
        origin + np.array([1.51, -0.49, 2.51]) * v,
    ])
    out = round_points_to_voxel(pts, v, origin)
    assert np.allclose(out[0], origin + np.array([0.0, 0.0, 0.0]))
    assert np.allclose(out[1], origin + np.array([4.0, -0.0, 6.0]))

    A = origin + np.array([0.1, 0.2, 0.4])
    B = origin + np.array([1.2, 0.1, 0.4])
    C = origin + np.array([0.1, 1.1, 0.4])
    file_pts = np.vstack([A, B])
    raw_pts = np.vstack([A, C])
    pts2, cols2 = compute_two_file_diff(file_pts, raw_pts, 1.0, origin=origin)
    whites = (cols2 == np.array([1.0, 1.0, 1.0])).all(axis=1).sum()
    reds = (cols2 == np.array([1.0, 0.0, 0.0])).all(axis=1).sum()
    greens = (cols2 == np.array([0.0, 1.0, 0.0])).all(axis=1).sum()
    assert whites == 1 and reds == 1 and greens == 1

    # Test HDF5 reader grid_origin anchoring
    # Stub open3d before importing reader
    # Build a minimal open3d stub to satisfy type hints
    o3d_stub = types.SimpleNamespace()
    class _DummyPointCloud: 
        pass
    o3d_stub.geometry = types.SimpleNamespace(PointCloud=_DummyPointCloud)
    sys.modules['open3d'] = o3d_stub
    from voxel_viewer.hdf5_reader import HDF5CompressedMapReader
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as td:
        path = os.path.join(td, 'origin.h5')
        v = 5.5
        bs = 8
        go = np.array([1.2, -3.4, 10.0], dtype=np.float32)
        write_min_h5(path, v, bs, go)
        reader = HDF5CompressedMapReader(path)
        assert reader.read()
        pts = reader.decompress()
        assert pts.shape == (1, 3)
        expected = go.astype(np.float64) + 0.5 * v
        assert np.allclose(pts[0], expected)

    print('OK: origin-aware rounding and HDF5 grid_origin alignment verified')


if __name__ == '__main__':
    main()
