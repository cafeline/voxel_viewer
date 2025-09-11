#!/usr/bin/env python3
"""Small helpers to isolate mode-specific behavior for the viewer."""

def is_file_mode(mode: str) -> bool:
    return str(mode).strip().lower() == 'file_comparison'


def is_topic_mode(mode: str) -> bool:
    return str(mode).strip().lower() == 'topic_comparison'


def should_load_files(mode: str) -> bool:
    """Return True only when viewer should read HDF5 files."""
    return is_file_mode(mode)

