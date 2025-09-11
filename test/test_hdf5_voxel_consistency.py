#!/usr/bin/env python3
"""Test for HDF5 voxel size consistency and coordinate transformation."""

import unittest
import numpy as np
import tempfile
import h5py
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from voxel_viewer.hdf5_reader import HDF5CompressedMapReader


class TestHDF5VoxelConsistency(unittest.TestCase):
    """Test voxel size consistency between HDF5 and runtime."""
    
    def create_test_hdf5(self, filepath, voxel_size, points):
        """Create a test HDF5 file with specified parameters."""
        with h5py.File(filepath, 'w') as f:
            # Metadata
            metadata = f.create_group('metadata')
            metadata.attrs['version'] = '1.0.0'
            metadata.attrs['timestamp'] = '2025-01-10T12:00:00'
            
            # Compression parameters
            comp_params = f.create_group('compression_params')
            comp_params.create_dataset('voxel_size', data=voxel_size)
            comp_params.create_dataset('block_size', data=8)
            comp_params.create_dataset('dictionary_size', data=1)
            comp_params.create_dataset('pattern_bits', data=512)
            
            # Dictionary - create a simple pattern with all bits set
            dictionary = f.create_group('dictionary')
            dictionary.create_dataset('pattern_length', data=512)
            # All bits set pattern (64 bytes)
            pattern_bytes = np.ones(64, dtype=np.uint8) * 255
            dictionary.create_dataset('patterns', data=pattern_bytes)
            
            # Compressed data - convert points to voxel positions
            compressed = f.create_group('compressed_data')
            voxel_positions = []
            for point in points:
                # Convert to voxel grid position (block position)
                voxel_pos = (point / (voxel_size * 8)).astype(np.int32)
                voxel_positions.append(voxel_pos)
            
            compressed.create_dataset('voxel_positions', data=np.array(voxel_positions))
            compressed.create_dataset('indices', data=np.zeros(len(voxel_positions), dtype=np.uint16))
            compressed.create_dataset('point_count', data=len(points))
            
            # Statistics
            stats = f.create_group('statistics')
            stats.create_dataset('original_points', data=len(points))
            stats.create_dataset('compressed_voxels', data=len(voxel_positions))
            stats.create_dataset('compression_ratio', data=1.0)
            
            # Bounding box
            if len(points) > 0:
                bbox = np.array([points.min(axis=0), points.max(axis=0)])
            else:
                bbox = np.zeros((2, 3))
            stats.create_dataset('bounding_box', data=bbox)
    
    def test_voxel_size_consistency(self):
        """Test that HDF5 voxel size is used correctly during decompression."""
        # Create test points at specific locations
        test_points = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
            [10, 10, 10]
        ], dtype=np.float32)
        
        # Test with voxel_size = 1.0
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath1 = f.name
        
        try:
            self.create_test_hdf5(filepath1, voxel_size=1.0, points=test_points)
            
            reader1 = HDF5CompressedMapReader(filepath1)
            self.assertTrue(reader1.read())
            points1 = reader1.decompress()
            
            # Check that we get some points back
            self.assertGreater(len(points1), 0)
            
            # Test with voxel_size = 5.1
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
                filepath2 = f.name
            
            self.create_test_hdf5(filepath2, voxel_size=5.1, points=test_points)
            
            reader2 = HDF5CompressedMapReader(filepath2)
            self.assertTrue(reader2.read())
            points2 = reader2.decompress()
            
            # Check that we get some points back
            self.assertGreater(len(points2), 0)
            
            # Points decompressed with different voxel sizes should be different
            # The scale should be proportional to voxel_size
            self.assertFalse(np.array_equal(points1, points2))
            
        finally:
            if os.path.exists(filepath1):
                os.unlink(filepath1)
            if os.path.exists(filepath2):
                os.unlink(filepath2)
    
    def test_coordinate_transformation(self):
        """Test coordinate transformation with different voxel sizes."""
        # Create a single voxel block at origin
        test_points = np.array([[0, 0, 0]], dtype=np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name
        
        try:
            # Create HDF5 with voxel_size = 5.1
            self.create_test_hdf5(filepath, voxel_size=5.1, points=test_points)
            
            reader = HDF5CompressedMapReader(filepath)
            self.assertTrue(reader.read())
            
            # Get the voxel size from the file
            params = reader.data['compression_params']
            file_voxel_size = params.get('voxel_size', 0.1)
            if isinstance(file_voxel_size, np.ndarray):
                file_voxel_size = file_voxel_size.item()
            
            self.assertAlmostEqual(file_voxel_size, 5.1, places=5)
            
            # Decompress points
            decompressed = reader.decompress()
            
            # The decompressed points should be scaled by the voxel size
            # A block at origin with voxel_size=5.1 should produce points
            # in the range [0, 5.1*8) for each dimension
            self.assertGreater(len(decompressed), 0)
            
            # Check that points are within expected range
            max_coord = 5.1 * 8  # voxel_size * block_size
            self.assertTrue(np.all(decompressed >= 0))
            self.assertTrue(np.all(decompressed < max_coord))
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_voxel_matching_with_rounding(self):
        """Test that voxel matching works correctly with rounding."""
        # Test the rounding function used in the viewer
        def round_points(points, voxel_size):
            if voxel_size <= 0:
                return points
            return np.round(points / voxel_size) * voxel_size
        
        # Create test points
        points1 = np.array([
            [5.05, 0, 0],
            [10.1, 0, 0],
            [15.15, 0, 0]
        ])
        
        points2 = np.array([
            [5.1, 0, 0],
            [10.2, 0, 0],
            [15.3, 0, 0]
        ])
        
        # Round with voxel_size = 5.1
        rounded1 = round_points(points1, 5.1)
        rounded2 = round_points(points2, 5.1)
        
        # After rounding, points should match
        expected = np.array([
            [5.1, 0, 0],
            [10.2, 0, 0],
            [15.3, 0, 0]
        ])
        
        np.testing.assert_array_almost_equal(rounded1, expected, decimal=5)
        np.testing.assert_array_almost_equal(rounded2, expected, decimal=5)


if __name__ == '__main__':
    unittest.main()