#!/usr/bin/env python3
"""Launch file for voxel_viewer node."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description."""
    return LaunchDescription([
        Node(
            package='voxel_viewer',
            executable='voxel_viewer',
            name='voxel_viewer',
            output='screen',
            parameters=[{
                'voxel_size': 0.1,
                'point_size': 5.0,
                'background_color': [0.1, 0.1, 0.1],
                'show_axes': True
            }],
            remappings=[
                ('occupied_voxel_markers', '/occupied_voxel_markers'),
                ('pattern_markers', '/pattern_markers')
            ]
        )
    ])