#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
import yaml
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

    def launch_setup(context, *args, **kwargs):
        cfg_path = LaunchConfiguration('config_file').perform(context)
        actions = []
        # Load YAML once
        try:
            with open(cfg_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
        viewer_params = ((data.get('voxel_viewer_with_hdf5') or {}).get('ros__parameters') or {})
        comp_params = ((data.get('pointcloud_compressor_node') or {}).get('ros__parameters') or {})

        # Voxel Viewer (always)
        viewer_node = Node(
            package='voxel_viewer',
            executable='voxel_viewer_with_hdf5',
            name='voxel_viewer_with_hdf5',
            output='screen',
            parameters=[viewer_params]
        )
        actions.append(viewer_node)
        # Optional: PointCloud Compressor (read from YAML: launch.start_pointcloud_compressor)
        start_compressor = True
        try:
            launch_cfg = (data.get('launch') or {})
            val = launch_cfg.get('start_pointcloud_compressor', True)
            # Only accept strict booleans; anything else falls back to default True
            start_compressor = val if isinstance(val, bool) else True
        except Exception:
            start_compressor = True

        if start_compressor:
            compressor_node = Node(
                package='pointcloud_compressor',
                executable='pointcloud_compressor_node',
                name='pointcloud_compressor_node',
                output='screen',
                parameters=[comp_params]
            )
            actions.insert(0, compressor_node)
        return actions

    return LaunchDescription([
        config_file_arg,
        OpaqueFunction(function=launch_setup)
    ])
