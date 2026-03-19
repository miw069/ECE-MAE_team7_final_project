"""
Centralized control mode manager for search/tracking behavior.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

NODE_NAME = 'control_mode_node'
VALID_MODES = {'IDLE', 'SEARCH', 'TRACK'}


class ControlModeNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('initial_mode', 'SEARCH'),
                ('request_topic', '/tracker/control_mode/request'),
                ('mode_topic', '/tracker/control_mode'),
                ('auto_track_topic', '/tracker/auto_track_request'),
                ('auto_search_topic', '/tracker/auto_search_request'),
            ],
        )

        self.request_topic = str(self.get_parameter('request_topic').value)
        self.mode_topic = str(self.get_parameter('mode_topic').value)
        self.auto_track_topic = str(self.get_parameter('auto_track_topic').value)
        self.auto_search_topic = str(self.get_parameter('auto_search_topic').value)
        self.mode = self._normalize_mode(str(self.get_parameter('initial_mode').value))

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE

        self.mode_pub = self.create_publisher(String, self.mode_topic, qos)
        self.create_subscription(String, self.request_topic, self._request_cb, 10)
        self.create_subscription(Empty, self.auto_track_topic, self._auto_track_cb, 10)
        self.create_subscription(Empty, self.auto_search_topic, self._auto_search_cb, 10)

        self._publish_mode(log_reason='initial')

    def _request_cb(self, msg: String) -> None:
        requested_mode = self._normalize_mode(msg.data)
        self._set_mode(requested_mode, f'user request: {requested_mode}')

    def _auto_track_cb(self, _msg: Empty) -> None:
        self._set_mode('TRACK', 'auto transition: person detected during search')

    def _auto_search_cb(self, _msg: Empty) -> None:
        self._set_mode('SEARCH', 'auto transition: tracking lost')

    def _set_mode(self, mode: str, reason: str) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self._publish_mode(log_reason=reason)

    def _publish_mode(self, log_reason: str) -> None:
        msg = String()
        msg.data = self.mode
        self.mode_pub.publish(msg)
        self.get_logger().info(f'Mode -> {self.mode} ({log_reason})')

    @staticmethod
    def _normalize_mode(value: str) -> str:
        mode = str(value).upper().strip()
        if mode not in VALID_MODES:
            return 'IDLE'
        return mode


def main(args=None):
    rclpy.init(args=args)
    node = ControlModeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
