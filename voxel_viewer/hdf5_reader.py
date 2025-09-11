#!/usr/bin/env python3
"""HDF5 reader for compressed point cloud data."""

import h5py
import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Dict, Any


class HDF5CompressedMapReader:
    """Reader for HDF5 compressed map files."""
    
    def __init__(self, filepath: str):
        """Initialize HDF5 reader with file path.
        
        Args:
            filepath: Path to HDF5 file
        """
        self.filepath = filepath
        self.data = None
        self.decompressed_points = None
        
    def read(self) -> bool:
        """Read HDF5 file and load compressed map data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with h5py.File(self.filepath, 'r') as f:
                self.data = {
                    'metadata': self._read_metadata(f),
                    'compression_params': self._read_compression_params(f),
                    'dictionary': self._read_dictionary(f),
                    'compressed_data': self._read_compressed_data(f),
                    'statistics': self._read_statistics(f)
                }
            return True
        except Exception as e:
            print(f"Error reading HDF5 file: {e}")
            return False
    
    def _read_metadata(self, f: h5py.File) -> Dict[str, Any]:
        """Read metadata from HDF5 file."""
        metadata = {}
        if 'metadata' in f:
            group = f['metadata']
            for attr in group.attrs:
                metadata[attr] = group.attrs[attr]
        return metadata
    
    def _read_compression_params(self, f: h5py.File) -> Dict[str, Any]:
        """Read compression parameters from HDF5 file."""
        params = {}
        if 'compression_params' in f:
            group = f['compression_params']
            if 'voxel_size' in group:
                params['voxel_size'] = group['voxel_size'][()]
            if 'dictionary_size' in group:
                params['dictionary_size'] = group['dictionary_size'][()]
            if 'pattern_bits' in group:
                params['pattern_bits'] = group['pattern_bits'][()]
            if 'block_size' in group:
                params['block_size'] = group['block_size'][()]
            if 'grid_origin' in group:
                # Expect shape (3,), float32
                params['grid_origin'] = group['grid_origin'][()]
        return params
    
    def _read_dictionary(self, f: h5py.File) -> Dict[str, Any]:
        """Read dictionary data from HDF5 file."""
        dictionary = {}
        if 'dictionary' in f:
            group = f['dictionary']
            if 'pattern_length' in group:
                dictionary['pattern_length'] = group['pattern_length'][()]
            if 'patterns' in group:
                dictionary['patterns'] = group['patterns'][:]
        return dictionary
    
    def _read_compressed_data(self, f: h5py.File) -> Dict[str, Any]:
        """Read compressed data from HDF5 file."""
        compressed = {}
        if 'compressed_data' in f:
            group = f['compressed_data']
            if 'indices' in group:
                compressed['indices'] = group['indices'][:]
            if 'voxel_positions' in group:
                compressed['voxel_positions'] = group['voxel_positions'][:]
            if 'point_count' in group:
                compressed['point_count'] = group['point_count'][()]
        return compressed
    
    def _read_statistics(self, f: h5py.File) -> Dict[str, Any]:
        """Read statistics from HDF5 file."""
        stats = {}
        if 'statistics' in f:
            group = f['statistics']
            if 'original_points' in group:
                stats['original_points'] = group['original_points'][()]
            if 'compressed_voxels' in group:
                stats['compressed_voxels'] = group['compressed_voxels'][()]
            if 'compression_ratio' in group:
                stats['compression_ratio'] = group['compression_ratio'][()]
            if 'bounding_box' in group:
                bbox = group['bounding_box'][:]
                stats['bounding_box_min'] = bbox[0]
                stats['bounding_box_max'] = bbox[1]
        return stats
    
    def decompress(self) -> np.ndarray:
        """Decompress the map data to get point cloud.
        
        Returns:
            Numpy array of shape (N, 3) containing decompressed points
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")
        
        params = self.data['compression_params']
        compressed = self.data['compressed_data']
        dictionary = self.data['dictionary']
        
        # Extract scalar values from numpy arrays if needed
        voxel_size = params.get('voxel_size', 0.1)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = voxel_size.item()
            
        block_size = params.get('block_size', 8)
        if isinstance(block_size, np.ndarray):
            block_size = block_size.item()
        
        # Grid origin (default to zeros if not present)
        grid_origin = params.get('grid_origin', np.array([0.0, 0.0, 0.0], dtype=np.float32))
        if isinstance(grid_origin, np.ndarray):
            if grid_origin.ndim == 0:
                # scalar (shouldn't happen) -> broadcast
                grid_origin = np.array([float(grid_origin)] * 3, dtype=np.float32)
            grid_origin = grid_origin.astype(np.float64).reshape(-1)
            if grid_origin.shape[0] >= 3:
                grid_origin = grid_origin[:3]
            else:
                # pad if shorter
                grid_origin = np.pad(grid_origin, (0, 3 - grid_origin.shape[0]))
        else:
            grid_origin = np.array(grid_origin, dtype=np.float64).reshape(3)
        
        # Get voxel positions and indices
        voxel_positions = compressed.get('voxel_positions', np.array([]))
        indices = compressed.get('indices', np.array([]))
        
        # Get dictionary patterns
        patterns = dictionary.get('patterns', np.array([]))
        pattern_length = dictionary.get('pattern_length', block_size**3)
        if isinstance(pattern_length, np.ndarray):
            pattern_length = pattern_length.item()
        
        # Calculate number of patterns
        if len(patterns) > 0:
            bytes_per_pattern = (pattern_length + 7) // 8  # Round up to nearest byte
            num_patterns = len(patterns) // bytes_per_pattern
        else:
            num_patterns = 0
        
        points = []
        
        # For each voxel block
        for i, (voxel_pos, pattern_idx) in enumerate(zip(voxel_positions, indices)):
            # Convert pattern_idx to integer if needed
            idx = int(pattern_idx)
            if idx >= num_patterns:
                continue
                
            # Get pattern data
            pattern_start = idx * bytes_per_pattern
            pattern_end = pattern_start + bytes_per_pattern
            pattern_bytes = patterns[pattern_start:pattern_end]
            
            # Convert pattern bytes to bit array
            # Patterns are stored with LSB-first per byte in C++ (toBytePattern).
            # Use little-endian bit order to match.
            pattern_bits = np.unpackbits(pattern_bytes, bitorder='little')[:pattern_length]
            
            # Generate points from pattern: world frame = origin + block offset
            block_origin = grid_origin + voxel_pos * block_size * voxel_size
            
            for bit_idx, occupied in enumerate(pattern_bits):
                if occupied:
                    # Calculate position within block
                    z = bit_idx // (block_size * block_size)
                    y = (bit_idx % (block_size * block_size)) // block_size
                    x = bit_idx % block_size
                    
                    # Calculate world position at voxel center
                    point = block_origin + (np.array([x, y, z]) + 0.5) * voxel_size
                    points.append(point)
        
        if len(points) > 0:
            self.decompressed_points = np.array(points)
        else:
            self.decompressed_points = np.zeros((0, 3))
        
        return self.decompressed_points
    
    def get_point_cloud(self) -> o3d.geometry.PointCloud:
        """Get Open3D point cloud from decompressed data.
        
        Returns:
            Open3D PointCloud object
        """
        if self.decompressed_points is None:
            self.decompress()
        
        pcd = o3d.geometry.PointCloud()
        if len(self.decompressed_points) > 0:
            pcd.points = o3d.utility.Vector3dVector(self.decompressed_points)
            # Color all points green for file data
            colors = np.tile([0, 1, 0], (len(self.decompressed_points), 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata from loaded HDF5 file.
        
        Returns:
            Dictionary containing all metadata
        """
        return self.data if self.data else None
    
    def get_statistics(self) -> Optional[Dict[str, Any]]:
        """Get statistics from loaded HDF5 file.
        
        Returns:
            Dictionary containing statistics
        """
        return self.data['statistics'] if self.data else None
