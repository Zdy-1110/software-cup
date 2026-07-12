"""
Gamepad controller for the micro-ROS chassis.

The left stick controls steering, RT drives forward, and LT drives backward.
The controller is discovered dynamically and supports hot unplug/reconnect.
"""

import glob
import threading
import time

import evdev
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class GamepadControlNode(Node):
    def __init__(self):
        super().__init__('gamepad_control')

        self.declare_parameter('device_name', 'Zikway HID gamepad')
        self.declare_parameter('device_path', '')
        self.declare_parameter('auto_reconnect', True)
        self.declare_parameter('reconnect_interval', 2.0)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('deadzone_stick', 20)
        self.declare_parameter('deadzone_trigger', 10)

        self.device_name = self.get_parameter('device_name').value
        self.device_path = self.get_parameter('device_path').value
        self.auto_reconnect = self.get_parameter('auto_reconnect').value
        self.reconnect_interval = self.get_parameter('reconnect_interval').value
        self.max_linear = self.get_parameter('max_linear_speed').value
        self.max_angular = self.get_parameter('max_angular_speed').value
        publish_rate = self.get_parameter('publish_rate').value
        self.deadzone_stick = self.get_parameter('deadzone_stick').value
        self.deadzone_trigger = self.get_parameter('deadzone_trigger').value

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.stick_x = 128
        self.trigger_rt = 0
        self.trigger_lt = 0
        self.gamepad = None
        self.gamepad_connected = False
        self.running = True

        self.gamepad_thread = threading.Thread(
            target=self._manage_gamepad, daemon=True
        )
        self.gamepad_thread.start()
        self.timer = self.create_timer(1.0 / publish_rate, self._publish_cmd)

        self.get_logger().info('Gamepad control node started')
        self.get_logger().info(f'Target device: {self.device_name}')

    def _find_gamepad(self):
        if self.device_path:
            try:
                device = evdev.InputDevice(self.device_path)
                name_matches = self.device_name.lower() in device.name.lower()
                device.close()
                return self.device_path if name_matches else None
            except OSError:
                return None

        for path in sorted(glob.glob('/dev/input/event*')):
            try:
                device = evdev.InputDevice(path)
                name_matches = self.device_name.lower() in device.name.lower()
                device.close()
                if name_matches:
                    return path
            except OSError:
                continue
        return None

    def _manage_gamepad(self):
        while self.running:
            path = self._find_gamepad()
            if path:
                try:
                    self.gamepad = evdev.InputDevice(path)
                    self.gamepad_connected = True
                    self.get_logger().info(
                        f'Gamepad connected: {self.gamepad.name} @ {path}'
                    )
                    self._read_gamepad()
                except OSError as exc:
                    self.get_logger().warning(f'Gamepad connection failed: {exc}')
                    self._reset_gamepad()
            else:
                self.get_logger().info(
                    f'Waiting for gamepad: {self.device_name}',
                    throttle_duration_sec=5.0,
                )

            if not self.auto_reconnect:
                break
            time.sleep(self.reconnect_interval)

    def _read_gamepad(self):
        try:
            for event in self.gamepad.read_loop():
                if not self.running:
                    break
                if event.type != evdev.ecodes.EV_ABS:
                    continue
                if event.code == evdev.ecodes.ABS_X:
                    self.stick_x = event.value
                elif event.code == evdev.ecodes.ABS_GAS:
                    self.trigger_rt = event.value
                elif event.code == evdev.ecodes.ABS_BRAKE:
                    self.trigger_lt = event.value
        except OSError as exc:
            if self.running:
                self.get_logger().warning(f'Gamepad disconnected: {exc}')
        finally:
            self._reset_gamepad()

    def _reset_gamepad(self):
        self.gamepad_connected = False
        self.stick_x = 128
        self.trigger_rt = 0
        self.trigger_lt = 0
        if self.gamepad:
            try:
                self.gamepad.close()
            except OSError:
                pass
        self.gamepad = None

    def _apply_deadzone_stick(self, value):
        centered = value - 128
        if abs(centered) < self.deadzone_stick:
            return 0.0
        if centered > 0:
            return (centered - self.deadzone_stick) / (127 - self.deadzone_stick)
        return (centered + self.deadzone_stick) / (128 - self.deadzone_stick)

    def _apply_deadzone_trigger(self, value):
        if value < self.deadzone_trigger:
            return 0.0
        return (value - self.deadzone_trigger) / (255 - self.deadzone_trigger)

    def _publish_cmd(self):
        msg = Twist()
        if self.gamepad_connected:
            stick = self._apply_deadzone_stick(self.stick_x)
            rt = self._apply_deadzone_trigger(self.trigger_rt)
            lt = self._apply_deadzone_trigger(self.trigger_lt)
            msg.linear.x = (rt - lt) * self.max_linear
            msg.angular.z = -stick * self.max_angular
        self.cmd_pub.publish(msg)

    def destroy_node(self):
        self.running = False
        self.cmd_pub.publish(Twist())
        self._reset_gamepad()
        self.gamepad_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GamepadControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()