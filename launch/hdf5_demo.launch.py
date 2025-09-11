#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.actions import Node

def generate_launch_description():
    """Generate launch description for HDF5 demo."""

    # Default config file
    pkg_share = get_package_share_directory('voxel_viewer')
    default_config = os.path.join(pkg_share, 'config', 'hdf5_demo_params.yaml')

    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to configuration YAML for compressor and viewer'
    )
    input_file_arg = DeclareLaunchArgument(
        'input_file',
        default_value='/home/ryo/tsudanuma/maps/tsudanuma-challenge-all.ply',
        description='Input point cloud file (PLY or PCD)'
    )

    voxel_size_arg = DeclareLaunchArgument(
        'voxel_size',
        default_value='5.1',
        description='Voxel size for compression'
    )

    hdf5_file_arg = DeclareLaunchArgument(
        'hdf5_file',
        default_value='/tmp/compressed_map.h5',
        description='Output HDF5 file path'
    )

    viewer_mode_arg = DeclareLaunchArgument(
        'viewer_mode',
        default_value='file_comparison',
        description='Viewer mode: topic_comparison or file_comparison'
    )

    # PointCloud Compressor Node
    compressor_node = Node(
        package='pointcloud_compressor',
        executable='pointcloud_compressor_node',
        name='pointcloud_compressor_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')]
    )

    # Voxel Viewer with HDF5 support
    viewer_node = Node(
        package='voxel_viewer',
        executable='voxel_viewer_with_hdf5',
        name='voxel_viewer_with_hdf5',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'mode': LaunchConfiguration('viewer_mode'),
                'hdf5_file': LaunchConfiguration('hdf5_file'),
                # Note: viewer will always use voxel size from HDF5 file when available
                'voxel_size': LaunchConfiguration('voxel_size')
            }
        ]
    )


    return LaunchDescription([
        config_file_arg,
        input_file_arg,
        voxel_size_arg,
        hdf5_file_arg,
        viewer_mode_arg,
        compressor_node,
        viewer_node
    ])
