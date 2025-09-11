#!/usr/bin/env python3
import os, sys, numpy as np
# Allow running directly from source tree
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(BASE, 'voxel_viewer')
sys.path.insert(0, PKG)
from file_compare import compute_two_file_diff, round_points_to_voxel


def main():
    # Simple deterministic test
    A = np.array([[0.1, 0.2, 0.4]])
    B = np.array([[1.2, 0.1, 0.4]])
    C = np.array([[0.1, 1.1, 0.4]])
    file_pts = np.vstack([A, B])
    raw_pts = np.vstack([A, C])
    pts, cols = compute_two_file_diff(file_pts, raw_pts, 1.0)
    whites = (cols == np.array([1.0, 1.0, 1.0])).all(axis=1).sum()
    reds = (cols == np.array([1.0, 0.0, 0.0])).all(axis=1).sum()
    greens = (cols == np.array([0.0, 1.0, 0.0])).all(axis=1).sum()
    assert whites == 1 and reds == 1 and greens == 1, 'color counts mismatch'
    print('OK: compute_two_file_diff basic test passed.')


if __name__ == '__main__':
    main()
