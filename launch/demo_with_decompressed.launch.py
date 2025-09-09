#!/usr/bin/env python3
"""Launch file for voxel_viewer with decompressed viewer for demo."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description."""
    # Get package directories
    decompressed_dir = get_package_share_directory('decompressed')
    
    # Include decompressed viewer launch
    decompressed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(decompressed_dir, 'launch', 'demo.launch.py')
        )
    )
    
    # Voxel viewer node (optimized version)
    voxel_viewer_node = Node(
        package='voxel_viewer',
        executable='voxel_viewer_optimized',
        name='voxel_viewer',
        output='screen',
        parameters=[{
            'background_color': [0.1, 0.1, 0.1],
            'show_axes': True
        }],
        remappings=[
            ('occupied_voxel_markers', '/occupied_voxel_markers'),
            ('pattern_markers', '/pattern_markers')
        ]
    )
    
    return LaunchDescription([
        decompressed_launch,
        voxel_viewer_node
    ])