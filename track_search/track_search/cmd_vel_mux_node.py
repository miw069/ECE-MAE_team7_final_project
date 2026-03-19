"""
Select the active velocity command source based on the current control mode.
"""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

NODE_NAME = 'cmd_vel_mux_node'


class CmdVelMuxNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('mode_topic', '/tracker/control_mode'),
                ('tracker_topic', '/cmd_vel_tracker'),
                ('search_topic', '/cmd_vel_nav'),
                ('output_topic', '/cmd_vel'),
                ('tracker_timeout_s', 0.5),
                ('search_timeout_s', 0.5),
                ('publish_hz', 20.0),
            ],
        )

        self.mode_topic = str(self.get_parameter('mode_topic').value)
        self.tracker_topic = str(self.get_parameter('tracker_topic').value)
        self.search_topic = str(self.get_parameter('search_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.tracker_timeout_s = float(self.get_parameter('tracker_timeout_s').value)
        self.search_timeout_s = float(self.get_parameter('search_timeout_s').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)

        self.mode = 'IDLE'
        self.tracker_cmd = Twist()
        self.search_cmd = Twist()
        self._tracker_stamp = 0.0
        self._search_stamp = 0.0

        self.cmd_pub = self.create_publisher(Twist, self.output_topic, 10)
        mode_qos = QoSProfile(depth=1)
        mode_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        mode_qos.reliability = ReliabilityPolicy.RELIABLE

        self.create_subscription(String, self.mode_topic, self._mode_cb, mode_qos)
        self.create_subscription(Twist, self.tracker_topic, self._tracker_cb, 10)
        self.create_subscription(Twist, self.search_topic, self._search_cb, 10)
        self.create_timer(1.0 / max(self.publish_hz, 1.0), self._publish_cmd)

        self.get_logger().info(
            f'{NODE_NAME} started'
            f'\n  tracker_topic: {self.tracker_topic}'
            f'\n  search_topic : {self.search_topic}'
            f'\n  output_topic : {self.output_topic}'
        )

    def _mode_cb(self, msg: String) -> None:
        self.mode = str(msg.data).upper().strip() or 'IDLE'

    def _tracker_cb(self, msg: Twist) -> None:
        self.tracker_cmd = msg
        self._tracker_stamp = time.monotonic()

    def _search_cb(self, msg: Twist) -> None:
        self.search_cmd = msg
        self._search_stamp = time.monotonic()

    def _publish_cmd(self) -> None:
        cmd = Twist()
        now = time.monotonic()

        if self.mode == 'TRACK' and (now - self._tracker_stamp) <= self.tracker_timeout_s:
            cmd = self.tracker_cmd
        elif self.mode == 'SEARCH' and (now - self._search_stamp) <= self.search_timeout_s:
            cmd = self.search_cmd

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node.cmd_pub.publish(Twist())
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
