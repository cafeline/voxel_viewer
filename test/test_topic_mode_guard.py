#!/usr/bin/env python3
"""Tests to ensure topic_comparison mode never attempts file I/O."""

import types


class _DummyLogger:
    def info(self, *a, **k):
        pass
    def warn(self, *a, **k):
        pass
    def error(self, *a, **k):
        pass
    def debug(self, *a, **k):
        pass


class _DummySelf:
    # Minimal attributes used by guarded functions
    mode = 'topic_comparison'
    _compressed_loaded_once = False
    hdf5_file = '/tmp/should_not_be_accessed.h5'
    raw_hdf5_file = '/tmp/should_not_be_accessed_raw.h5'
    file_load_timer = None

    def get_logger(self):
        return _DummyLogger()


def test_load_hdf5_file_noop_in_topic_mode():
    from voxel_viewer.voxel_viewer_with_hdf5 import VoxelViewerWithHDF5Node
    dummy = _DummySelf()
    # Should return immediately and not toggle flags
    VoxelViewerWithHDF5Node.load_hdf5_file(dummy)
    assert dummy._compressed_loaded_once is False


def test_two_file_loading_noop_in_topic_mode():
    from voxel_viewer.voxel_viewer_with_hdf5 import VoxelViewerWithHDF5Node
    dummy = _DummySelf()
    # Ensure attributes that would be set by loader do not appear
    assert not hasattr(dummy, 'file_points')
    VoxelViewerWithHDF5Node.load_two_hdf5_files(dummy)
    assert not hasattr(dummy, 'file_points')
    assert not hasattr(dummy, 'raw_file_points')


def test_polling_not_started_in_topic_mode():
    from voxel_viewer.voxel_viewer_with_hdf5 import VoxelViewerWithHDF5Node
    dummy = _DummySelf()
    VoxelViewerWithHDF5Node.start_two_file_polling(dummy)
    assert dummy.file_load_timer is None
    VoxelViewerWithHDF5Node.poll_two_files_once(dummy)
    assert dummy.file_load_timer is None

