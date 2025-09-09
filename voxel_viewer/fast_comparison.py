#!/usr/bin/env python3
"""Optimized comparison algorithms for voxel viewer."""

import numpy as np
import time


class FastComparison:
    """Fast point cloud comparison using optimized algorithms."""
    
    @staticmethod
    def compare_using_sets(points1, points2, voxel_size):
        """Original set-based comparison (for baseline)."""
        start = time.perf_counter()
        
        # Round both point sets
        rounded1 = np.round(points1 / voxel_size) * voxel_size
        rounded2 = np.round(points2 / voxel_size) * voxel_size
        
        # Convert to sets for comparison
        set1 = set(map(tuple, rounded1))
        set2 = set(map(tuple, rounded2))
        
        # Find matches and mismatches
        matches = set1.intersection(set2)
        only1 = set1 - set2
        only2 = set2 - set1
        
        end = time.perf_counter()
        return matches, only1, only2, (end - start) * 1000
    
    @staticmethod
    def compare_using_numpy_lexsort(points1, points2, voxel_size):
        """Optimized comparison using NumPy lexsort."""
        start = time.perf_counter()
        
        # Round both point sets
        rounded1 = np.round(points1 / voxel_size) * voxel_size
        rounded2 = np.round(points2 / voxel_size) * voxel_size
        
        # Get unique points
        unique1 = np.unique(rounded1, axis=0)
        unique2 = np.unique(rounded2, axis=0)
        
        # Combine all points
        all_points = np.vstack([unique1, unique2])
        
        # Sort using lexsort for fast comparison
        sorted_indices = np.lexsort((all_points[:, 2], all_points[:, 1], all_points[:, 0]))
        sorted_points = all_points[sorted_indices]
        
        # Find duplicates (matches)
        diff = np.diff(sorted_points, axis=0)
        duplicate_mask = np.all(np.abs(diff) < 1e-9, axis=1)
        
        # Get match indices
        match_indices = np.where(duplicate_mask)[0]
        matches = sorted_points[match_indices]
        
        # Create masks for mismatches
        n1 = len(unique1)
        point_sources = np.zeros(len(all_points), dtype=bool)
        point_sources[n1:] = True
        point_sources_sorted = point_sources[sorted_indices]
        
        # Mark matched points
        matched_mask = np.zeros(len(sorted_points), dtype=bool)
        matched_mask[match_indices] = True
        matched_mask[match_indices + 1] = True
        
        # Get mismatches
        only1_mask = ~point_sources_sorted & ~matched_mask
        only2_mask = point_sources_sorted & ~matched_mask
        
        only1 = sorted_points[only1_mask]
        only2 = sorted_points[only2_mask]
        
        end = time.perf_counter()
        return matches, only1, only2, (end - start) * 1000
    
    @staticmethod
    def compare_using_spatial_hash(points1, points2, voxel_size):
        """Ultra-fast comparison using spatial hashing."""
        start = time.perf_counter()
        
        # Quantize points to integer grid
        quantized1 = np.floor(points1 / voxel_size).astype(np.int32)
        quantized2 = np.floor(points2 / voxel_size).astype(np.int32)
        
        # Get unique voxels
        unique1 = np.unique(quantized1, axis=0)
        unique2 = np.unique(quantized2, axis=0)
        
        # Create spatial hash (combine xyz into single hash)
        # Use a prime-based hash to minimize collisions
        prime1, prime2, prime3 = 73856093, 19349663, 83492791
        hash1 = (unique1[:, 0] * prime1) ^ (unique1[:, 1] * prime2) ^ (unique1[:, 2] * prime3)
        hash2 = (unique2[:, 0] * prime1) ^ (unique2[:, 1] * prime2) ^ (unique2[:, 2] * prime3)
        
        # Find intersections using set operations on hashes
        set1 = set(hash1)
        set2 = set(hash2)
        
        matches_hash = set1.intersection(set2)
        only1_hash = set1 - set2
        only2_hash = set2 - set1
        
        # Convert back to points
        hash_to_point1 = dict(zip(hash1, unique1 * voxel_size))
        hash_to_point2 = dict(zip(hash2, unique2 * voxel_size))
        
        matches = np.array([hash_to_point1.get(h, hash_to_point2[h]) for h in matches_hash])
        only1 = np.array([hash_to_point1[h] for h in only1_hash])
        only2 = np.array([hash_to_point2[h] for h in only2_hash])
        
        end = time.perf_counter()
        return matches, only1, only2, (end - start) * 1000
    
    @staticmethod
    def compare_using_structured_array(points1, points2, voxel_size):
        """Fast comparison using NumPy structured arrays."""
        start = time.perf_counter()
        
        # Round both point sets
        rounded1 = np.round(points1 / voxel_size).astype(np.float32) * voxel_size
        rounded2 = np.round(points2 / voxel_size).astype(np.float32) * voxel_size
        
        # Get unique points using structured array view
        def unique_rows(arr):
            # View as structured array for fast unique operation
            arr_view = np.ascontiguousarray(arr).view(
                np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
            )
            _, unique_indices = np.unique(arr_view, return_index=True)
            return arr[unique_indices]
        
        unique1 = unique_rows(rounded1)
        unique2 = unique_rows(rounded2)
        
        # Use numpy intersect1d with structured view
        view1 = np.ascontiguousarray(unique1).view(
            np.dtype((np.void, unique1.dtype.itemsize * unique1.shape[1]))
        )
        view2 = np.ascontiguousarray(unique2).view(
            np.dtype((np.void, unique2.dtype.itemsize * unique2.shape[1]))
        )
        
        # Find intersections and differences
        matches_view = np.intersect1d(view1, view2)
        only1_view = np.setdiff1d(view1, view2)
        only2_view = np.setdiff1d(view2, view1)
        
        # Convert back to float arrays
        if len(matches_view) > 0:
            matches = matches_view.view(unique1.dtype).reshape(-1, 3)
        else:
            matches = np.array([])
            
        if len(only1_view) > 0:
            only1 = only1_view.view(unique1.dtype).reshape(-1, 3)
        else:
            only1 = np.array([])
            
        if len(only2_view) > 0:
            only2 = only2_view.view(unique2.dtype).reshape(-1, 3)
        else:
            only2 = np.array([])
        
        end = time.perf_counter()
        return matches, only1, only2, (end - start) * 1000


def benchmark_comparison_methods():
    """Benchmark all comparison methods."""
    print("=" * 60)
    print("COMPARISON METHOD BENCHMARKS")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    sizes = [10000, 50000, 100000, 500000]
    
    comparator = FastComparison()
    
    for size in sizes:
        print(f"\n--- {size} points ---")
        
        # Generate test data with 80% overlap
        points1 = np.random.uniform(0, 100, (size, 3)).astype(np.float32)
        points2 = np.random.uniform(0, 100, (int(size * 0.8), 3)).astype(np.float32)
        voxel_size = 0.09
        
        # Test each method
        methods = [
            ("Original (sets)", comparator.compare_using_sets),
            ("NumPy lexsort", comparator.compare_using_numpy_lexsort),
            ("Spatial hash", comparator.compare_using_spatial_hash),
            ("Structured array", comparator.compare_using_structured_array),
        ]
        
        results = {}
        for name, method in methods:
            try:
                matches, only1, only2, time_ms = method(points1, points2, voxel_size)
                match_count = len(matches) if hasattr(matches, '__len__') else len(list(matches))
                results[name] = (time_ms, match_count)
                print(f"{name:20s}: {time_ms:8.2f} ms ({match_count} matches)")
            except Exception as e:
                print(f"{name:20s}: Failed - {e}")
    
    print("\n" + "=" * 60)
    print("SPEEDUP SUMMARY (vs Original)")
    print("=" * 60)
    
    if "Original (sets)" in results:
        baseline = results["Original (sets)"][0]
        for name, (time_ms, _) in results.items():
            speedup = baseline / time_ms
            print(f"{name:20s}: {speedup:.2f}x")


if __name__ == '__main__':
    benchmark_comparison_methods()