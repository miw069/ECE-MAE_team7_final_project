from setuptools import setup
import os
from glob import glob

package_name = 'track_search'
submodule = package_name + '/vesc_submodule'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, submodule],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'PyYAML',
        'depthai',
        'depthai-nodes',
        'lap',
        'pyserial',
        'pyvesc',
    ],
    zip_safe=True,
    maintainer='team7',
    maintainer_email='team7@ucsd.edu',
    description='DepthAI person tracking, PID following, and VESC integration',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_camera_node = track_search.tracker_camera_node:main',
            'tracker_pid_node    = track_search.tracker_pid_node:main',
            'gps_waypoint_node   = track_search.gps_waypoint_node:main',
            'gps_runner_bridge_node = track_search.gps_runner_bridge_node:main',
            'gps_udp_bridge_node = track_search.gps_udp_bridge_node:main',
            'search_nav_node     = track_search.search_nav_node:main',
            'control_mode_node   = track_search.control_mode_node:main',
            'cmd_vel_mux_node    = track_search.cmd_vel_mux_node:main',
            'vesc_twist_node     = track_search.vesc_twist_node:main',
        ],
    },
)
