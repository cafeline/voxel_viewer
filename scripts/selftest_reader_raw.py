#!/usr/bin/env python3
import os, sys
import numpy as np
import h5py
# Allow running directly from source tree
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(BASE, 'voxel_viewer')
sys.path.insert(0, PKG)
from hdf5_reader import HDF5CompressedMapReader


def create_small_raw_h5(path: str):
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, 'w') as f:
        g = f.create_group('raw_voxel_grid')
        g.create_dataset('dimensions', data=np.array([3, 3, 1], dtype=np.uint32))
        g.create_dataset('voxel_size', data=np.array([1.0], dtype=np.float32))
        g.create_dataset('origin', data=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        occ = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)
        g.create_dataset('occupied_voxels', data=occ)


def main():
    path = '/tmp/test_raw_voxel_grid.h5'
    create_small_raw_h5(path)
    r = HDF5CompressedMapReader(path)
    assert r.read(), 'failed to read test h5'
    pts = r.decompress()
    assert pts.shape[0] == 3, f'expected 3 points, got {pts.shape[0]}'
    expected = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]])
    # order may differ; compare sets
    s_pts = set(map(tuple, np.round(pts, 6)))
    s_exp = set(map(tuple, expected))
    assert s_pts == s_exp, f'points mismatch: got {s_pts}, expected {s_exp}'
    print('OK: raw HDF5 read/decompress test passed.')


if __name__ == '__main__':
    main()
