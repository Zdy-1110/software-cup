from setuptools import setup
import os
from glob import glob

package_name = 'camera_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='teamhd',
    maintainer_email='teamhd@localhost',
    description='IMX291 camera video stream + RKNN detection WebSocket backend',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'unified_server   = camera_detection.unified_server:main',
            'video_server     = camera_detection.video_server:main',
            'detection_server = camera_detection.detection_server:main',
            'main_server      = camera_detection.main_server:main',
        ],
    },
)
