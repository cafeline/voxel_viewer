#!/usr/bin/env python3
"""Launch file for the optimized voxel viewer."""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Generate launch description for optimized voxel viewer."""
    return LaunchDescription([
        Node(
            package='voxel_viewer',
            executable='optimized_voxel_viewer',
            name='optimized_voxel_viewer',
            output='screen',
            parameters=[{
                'voxel_size': 0.1,
                'point_size': 5.0,
                'background_color': [0.1, 0.1, 0.1],
                'show_axes': True,
                'tolerance': 0.001
            }]
        )
    ])