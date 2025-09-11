#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.actions import Node

def generate_launch_description():
    """Generate launch description for HDF5 demo."""

    # Default config file
    pkg_share = get_package_share_directory('voxel_viewer')
    default_config = os.path.join(pkg_share, 'config', 'hdf5_demo_params.yaml')

    # Declare launch argument for the YAML configuration only
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to configuration YAML for compressor and viewer'
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
        parameters=[LaunchConfiguration('config_file')]
    )


    return LaunchDescription([
        config_file_arg,
        compressor_node,
        viewer_node
    ])
