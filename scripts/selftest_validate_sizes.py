#!/usr/bin/env python3
import os, sys
# Allow running directly from source tree
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(BASE, 'voxel_viewer')
sys.path.insert(0, PKG)
from file_compare import validate_voxel_sizes


def main():
    assert not validate_voxel_sizes(None, 0.1)
    assert not validate_voxel_sizes(0.1, None)
    assert validate_voxel_sizes(0.5, 0.5)
    assert validate_voxel_sizes(0.5, 0.5 + 1e-10)
    assert not validate_voxel_sizes(0.5, 0.6)
    print('OK: validate_voxel_sizes tests passed.')


if __name__ == '__main__':
    main()
