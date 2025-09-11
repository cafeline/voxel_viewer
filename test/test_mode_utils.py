#!/usr/bin/env python3
from voxel_viewer.mode_utils import is_file_mode, is_topic_mode, should_load_files


def test_is_file_mode():
    assert is_file_mode('file_comparison')
    assert is_file_mode('FILE_COMPARISON')
    assert not is_file_mode('topic_comparison')
    assert not is_file_mode('other')


def test_is_topic_mode():
    assert is_topic_mode('topic_comparison')
    assert is_topic_mode('TOPIC_COMPARISON')
    assert not is_topic_mode('file_comparison')


def test_should_load_files():
    assert should_load_files('file_comparison') is True
    assert should_load_files('topic_comparison') is False
    assert should_load_files('unknown') is False

