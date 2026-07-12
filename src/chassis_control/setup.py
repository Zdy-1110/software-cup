from setuptools import find_packages, setup

package_name = 'chassis_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'evdev'],
    zip_safe=True,
    maintainer='teamhd',
    maintainer_email='teamhd@example.com',
    description='Gamepad control for the micro-ROS chassis',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'gamepad_control = chassis_control.gamepad_control:main',
        ],
    },
)