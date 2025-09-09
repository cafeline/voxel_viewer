#!/usr/bin/env python3
"""Optimized voxel rounding implementations."""

import numpy as np
import time

def round_to_voxel_original(points, voxel_size):
    """Original implementation - iterative approach."""
    start = time.perf_counter()
    
    rounded_points = []
    for point in points:
        rounded = (
            round(point[0] / voxel_size) * voxel_size,
            round(point[1] / voxel_size) * voxel_size,
            round(point[2] / voxel_size) * voxel_size
        )
        rounded_points.append(rounded)
    
    result = np.array(rounded_points)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed

def round_to_voxel_vectorized(points, voxel_size):
    """Fully vectorized NumPy implementation."""
    start = time.perf_counter()
    
    # Convert to numpy array if needed
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    # Vectorized rounding - single operation on entire array
    rounded = np.round(points / voxel_size) * voxel_size
    
    elapsed = (time.perf_counter() - start) * 1000
    return rounded, elapsed

def round_to_voxel_inplace(points, voxel_size):
    """In-place vectorized implementation to save memory."""
    start = time.perf_counter()
    
    # Convert to numpy array if needed
    if not isinstance(points, np.ndarray):
        points = np.array(points, dtype=np.float32)
    else:
        points = points.astype(np.float32, copy=True)
    
    # In-place operations
    points /= voxel_size
    np.round(points, out=points)
    points *= voxel_size
    
    elapsed = (time.perf_counter() - start) * 1000
    return points, elapsed

def round_to_voxel_fast_math(points, voxel_size):
    """Using fast math with float32 for speed."""
    start = time.perf_counter()
    
    # Convert to float32 for faster operations
    if not isinstance(points, np.ndarray):
        points = np.array(points, dtype=np.float32)
    else:
        points = points.astype(np.float32)
    
    # Use float32 throughout
    voxel_size_f32 = np.float32(voxel_size)
    inv_voxel_size = np.float32(1.0 / voxel_size_f32)
    
    # Fast multiply instead of divide
    rounded = np.round(points * inv_voxel_size) * voxel_size_f32
    
    elapsed = (time.perf_counter() - start) * 1000
    return rounded, elapsed

def round_to_voxel_integer(points, voxel_size, precision=1000):
    """Integer-based rounding for exact precision."""
    start = time.perf_counter()
    
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    # Convert to integer space
    int_points = (points * precision).astype(np.int32)
    int_voxel = int(voxel_size * precision)
    
    # Round in integer space
    rounded_int = ((int_points + int_voxel // 2) // int_voxel) * int_voxel
    
    # Convert back to float
    rounded = rounded_int.astype(np.float64) / precision
    
    elapsed = (time.perf_counter() - start) * 1000
    return rounded, elapsed

def benchmark_voxel_rounding():
    """Benchmark all voxel rounding methods."""
    print("=" * 60)
    print("VOXEL ROUNDING BENCHMARKS")
    print("=" * 60)
    
    sizes = [10000, 50000, 100000, 500000]
    voxel_size = 0.05
    
    for size in sizes:
        print(f"\n--- {size} points ---")
        
        # Generate test data
        points = np.random.uniform(-10, 10, (size, 3))
        
        # Test all methods
        methods = [
            ("Original", round_to_voxel_original),
            ("Vectorized", round_to_voxel_vectorized),
            ("In-place", round_to_voxel_inplace),
            ("Fast Math", round_to_voxel_fast_math),
            ("Integer", round_to_voxel_integer)
        ]
        
        results = []
        for name, method in methods:
            # Warmup
            method(points[:100], voxel_size)
            
            # Benchmark
            times = []
            for _ in range(5):
                _, elapsed = method(points.copy(), voxel_size)
                times.append(elapsed)
            
            avg_time = np.mean(times)
            results.append((name, avg_time))
            print(f"{name:15s}: {avg_time:8.2f} ms")
        
        # Show speedup
        baseline = results[0][1]
        print("\nSpeedup vs Original:")
        for name, avg_time in results:
            speedup = baseline / avg_time
            print(f"{name:15s}: {speedup:6.2f}x")
    
    # Verify correctness
    print("\n" + "=" * 60)
    print("CORRECTNESS VERIFICATION")
    print("=" * 60)
    
    test_points = np.array([[1.234, 2.567, 3.891],
                            [-4.321, 5.678, -6.789],
                            [0.012, -0.023, 0.034]])
    
    print("\nTest points:")
    print(test_points)
    
    print(f"\nRounded with voxel_size={voxel_size}:")
    
    for name, method in methods:
        rounded, _ = method(test_points.copy(), voxel_size)
        print(f"{name:15s}: {rounded[0]}")  # Show first point only
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    # Find best method for largest size
    points = np.random.uniform(-10, 10, (500000, 3))
    best_time = float('inf')
    best_method = None
    
    for name, method in methods[1:]:  # Skip original
        times = []
        for _ in range(3):
            _, elapsed = method(points.copy(), voxel_size)
            times.append(elapsed)
        avg_time = np.mean(times)
        
        if avg_time < best_time:
            best_time = avg_time
            best_method = name
    
    # Get original time
    orig_times = []
    for _ in range(3):
        _, elapsed = round_to_voxel_original(points[:100000], voxel_size)  # Test with smaller size
        orig_times.append(elapsed * 5)  # Estimate for 500k
    orig_time = np.mean(orig_times)
    
    speedup = orig_time / best_time
    print(f"Best method: {best_method}")
    print(f"Expected speedup: {speedup:.2f}x")
    print(f"Time saved for 500k points: {orig_time - best_time:.2f} ms")

if __name__ == "__main__":
    benchmark_voxel_rounding()