from setuptools import find_packages, setup

package_name = 'voxel_viewer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/voxel_viewer.launch.py',
            'launch/demo_with_decompressed.launch.py',
            'launch/optimized_viewer_launch.py'
        ]),
    ],
    install_requires=['setuptools', 'open3d', 'numpy'],
    zip_safe=True,
    maintainer='ryo',
    maintainer_email='s24s1040du@s.chibakoudai.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voxel_viewer = voxel_viewer.voxel_viewer:main',
            'optimized_voxel_viewer = voxel_viewer.optimized_voxel_viewer:main',
            'voxel_viewer_optimized = voxel_viewer.voxel_viewer_optimized:main',
            'test_comparison = voxel_viewer.test_comparison:main'
        ],
    },
)
