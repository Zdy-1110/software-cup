from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('video_device',    default_value='/dev/video20'),
        DeclareLaunchArgument('ws_port',         default_value='8765'),
        DeclareLaunchArgument('detection_port',  default_value='8766'),
        DeclareLaunchArgument('rknn_model',
            default_value='/home/teamhd/Downloads/'
                          'ppyoloe_carrace_rk3588_official_split_int8_416.rknn'),
        DeclareLaunchArgument('conf_thresh',     default_value='0.3'),
        DeclareLaunchArgument('class_names',
                              default_value='bm,cjl,jsjd,jzt,lu,mtl,nc,tt,ydm,zynsx'),

        Node(
            package='camera_detection',
            executable='unified_server',
            name='camera_detection_server',
            output='screen',
            env={
                'VIDEO_DEVICE':   LaunchConfiguration('video_device'),
                'WS_PORT':         LaunchConfiguration('ws_port'),
                'DETECTION_PORT': LaunchConfiguration('detection_port'),
                'RKNN_MODEL':     LaunchConfiguration('rknn_model'),
                'CONF_THRESH':    LaunchConfiguration('conf_thresh'),
                'CLASS_NAMES':    LaunchConfiguration('class_names'),
                'WIDTH':          '1920',
                'HEIGHT':         '1080',
                'FRAMERATE':      '30',
            },
        ),
    ])
