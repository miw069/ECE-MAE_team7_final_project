"""
Generate and follow a lemniscate search path around a configurable center.
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, Float32, Float32MultiArray, Int32, String

NODE_NAME = 'search_nav_node'


def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class SearchNavNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('pose_topic', '/gps/local_pose'),
                ('control_mode_topic', '/tracker/control_mode'),
                ('cmd_topic', '/cmd_vel_nav'),
                ('path_topic', '/search/path'),
                ('status_topic', '/search/status'),
                ('recenter_topic', '/search/recenter'),
                ('radius_topic', '/search/set_radius'),
                ('loops_topic', '/search/set_loops'),
                ('frame_id', 'map'),
                ('radius_m', 8.0),
                ('num_loops', 3),
                ('points_per_loop', 240),
                ('lookahead_m', 1.5),
                ('wheelbase_m', 0.25),
                ('search_speed', 0.25),
                ('max_steering_angle', 0.55),
                ('control_hz', 10.0),
                ('pose_timeout_s', 1.0),
                ('auto_center_on_first_pose', True),
                ('center_x_m', 0.0),
                ('center_y_m', 0.0),
            ],
        )

        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.control_mode_topic = str(self.get_parameter('control_mode_topic').value)
        self.cmd_topic = str(self.get_parameter('cmd_topic').value)
        self.path_topic = str(self.get_parameter('path_topic').value)
        self.status_topic = str(self.get_parameter('status_topic').value)
        self.recenter_topic = str(self.get_parameter('recenter_topic').value)
        self.radius_topic = str(self.get_parameter('radius_topic').value)
        self.loops_topic = str(self.get_parameter('loops_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.radius_m = float(self.get_parameter('radius_m').value)
        self.num_loops = max(1, int(self.get_parameter('num_loops').value))
        self.points_per_loop = max(60, int(self.get_parameter('points_per_loop').value))
        self.lookahead_m = float(self.get_parameter('lookahead_m').value)
        self.wheelbase_m = float(self.get_parameter('wheelbase_m').value)
        self.search_speed = float(self.get_parameter('search_speed').value)
        self.max_steering_angle = float(self.get_parameter('max_steering_angle').value)
        self.control_hz = float(self.get_parameter('control_hz').value)
        self.pose_timeout_s = float(self.get_parameter('pose_timeout_s').value)
        self.auto_center_on_first_pose = bool(self.get_parameter('auto_center_on_first_pose').value)
        self.center_x_m = float(self.get_parameter('center_x_m').value)
        self.center_y_m = float(self.get_parameter('center_y_m').value)

        self.control_mode = 'SEARCH'
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.pose_ready = False
        self.center_ready = False
        self.path_ready = False
        self.path_x = np.array([], dtype=np.float64)
        self.path_y = np.array([], dtype=np.float64)
        self._last_pose_time = 0.0
        self._last_wait_warn_time = 0.0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.status_pub = self.create_publisher(Float32MultiArray, self.status_topic, 10)

        mode_qos = QoSProfile(depth=1)
        mode_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        mode_qos.reliability = ReliabilityPolicy.RELIABLE

        self.create_subscription(PoseStamped, self.pose_topic, self._pose_cb, 10)
        self.create_subscription(String, self.control_mode_topic, self._mode_cb, mode_qos)
        self.create_subscription(Empty, self.recenter_topic, self._recenter_cb, 10)
        self.create_subscription(Float32, self.radius_topic, self._radius_cb, 10)
        self.create_subscription(Int32, self.loops_topic, self._loops_cb, 10)

        self.create_timer(1.0 / max(self.control_hz, 1.0), self._control_loop)
        self.create_timer(0.5, self._publish_status)

        self.get_logger().info(
            f'{NODE_NAME} started'
            f'\n  radius={self.radius_m:.2f} m'
            f'\n  loops={self.num_loops}'
            f'\n  lookahead={self.lookahead_m:.2f} m'
            f'\n  speed={self.search_speed:.2f}'
            f'\n  pose_timeout_s={self.pose_timeout_s:.2f}'
        )

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.current_x = float(msg.pose.position.x)
        self.current_y = float(msg.pose.position.y)
        self.current_yaw = _quaternion_to_yaw(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        self.pose_ready = True
        self._last_pose_time = time.monotonic()

        if not self.center_ready and self.auto_center_on_first_pose:
            self.center_x_m = self.current_x
            self.center_y_m = self.current_y
            self.center_ready = True
            self._rebuild_path()

    def _mode_cb(self, msg: String) -> None:
        self.control_mode = str(msg.data).upper().strip() or 'IDLE'

    def _recenter_cb(self, _msg: Empty) -> None:
        if not self.pose_ready:
            self.get_logger().warn('Cannot recenter search path before local pose is available')
            return
        self.center_x_m = self.current_x
        self.center_y_m = self.current_y
        self.center_ready = True
        self._rebuild_path()
        self.get_logger().info(
            f'Search center recentered to ({self.center_x_m:.2f}, {self.center_y_m:.2f})'
        )

    def _radius_cb(self, msg: Float32) -> None:
        new_radius = max(1.0, float(msg.data))
        if abs(new_radius - self.radius_m) < 1e-6:
            return
        self.radius_m = new_radius
        if self.center_ready:
            self._rebuild_path()

    def _loops_cb(self, msg: Int32) -> None:
        new_loops = max(1, int(msg.data))
        if new_loops == self.num_loops:
            return
        self.num_loops = new_loops
        if self.center_ready:
            self._rebuild_path()

    def _rebuild_path(self) -> None:
        self.path_x, self.path_y = self._generate_lemniscate(
            self.center_x_m,
            self.center_y_m,
            self.radius_m,
            self.num_loops,
            self.points_per_loop,
        )
        self.path_ready = len(self.path_x) > 0
        self._publish_path()
        self._publish_status()

    def _generate_lemniscate(
        self,
        center_x: float,
        center_y: float,
        radius_m: float,
        loops: int,
        points_per_loop: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        a = radius_m / 1.5
        total_points = max(points_per_loop * loops, 1)
        t = np.linspace(0.0, 2.0 * np.pi * loops, total_points, endpoint=False)
        x = center_x + (a * np.cos(t)) / (1.0 + np.sin(t) ** 2)
        y = center_y + (a * np.sin(t) * np.cos(t)) / (1.0 + np.sin(t) ** 2)
        return x, y

    def _publish_path(self) -> None:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.frame_id

        for px, py in zip(self.path_x, self.path_y):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(px)
            pose.pose.position.y = float(py)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.path_pub.publish(path)

    def _publish_status(self) -> None:
        status = Float32MultiArray()
        status.data = [
            float(self.radius_m),
            float(self.num_loops),
            float(self.center_x_m),
            float(self.center_y_m),
            1.0 if self.path_ready else 0.0,
            1.0 if self._pose_is_fresh() else 0.0,
        ]
        self.status_pub.publish(status)

    def _control_loop(self) -> None:
        cmd = Twist()
        if self.control_mode != 'SEARCH':
            self.cmd_pub.publish(cmd)
            return

        if not self.path_ready or not self._pose_is_fresh():
            self._log_wait_reason()
            self.cmd_pub.publish(cmd)
            return

        steering_angle = self._pure_pursuit(
            self.current_x,
            self.current_y,
            self.current_yaw,
            self.path_x,
            self.path_y,
        )
        angular_cmd = self._clamp(steering_angle / max(self.max_steering_angle, 1e-6), 1.0)

        cmd.linear.x = float(max(0.0, self.search_speed))
        cmd.angular.z = float(angular_cmd)
        self.cmd_pub.publish(cmd)

    def _pose_is_fresh(self) -> bool:
        return self.pose_ready and (time.monotonic() - self._last_pose_time) <= self.pose_timeout_s

    def _log_wait_reason(self) -> None:
        now = time.monotonic()
        if (now - self._last_wait_warn_time) < 5.0:
            return

        if not self.pose_ready:
            message = 'Search paused: waiting for /gps/local_pose'
        elif not self._pose_is_fresh():
            age_s = now - self._last_pose_time
            message = (
                'Search paused: local pose is stale '
                f'({age_s:.2f}s old, timeout={self.pose_timeout_s:.2f}s)'
            )
        elif not self.path_ready:
            message = 'Search paused: search path is not initialized yet'
        else:
            message = 'Search paused: waiting for search navigation prerequisites'

        self.get_logger().warn(message)
        self._last_wait_warn_time = now

    def _pure_pursuit(
        self,
        x: float,
        y: float,
        yaw: float,
        path_x: np.ndarray,
        path_y: np.ndarray,
    ) -> float:
        distances = np.hypot(path_x - x, path_y - y)
        nearest_index = int(np.argmin(distances))

        target_index = nearest_index
        for _ in range(len(path_x)):
            dx = path_x[target_index] - x
            dy = path_y[target_index] - y
            if math.hypot(dx, dy) >= self.lookahead_m:
                break
            target_index = (target_index + 1) % len(path_x)

        target_x = float(path_x[target_index])
        target_y = float(path_y[target_index])
        dx = target_x - x
        dy = target_y - y

        local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        if abs(local_x) < 1e-6:
            return 0.0

        curvature = 2.0 * local_y / max(self.lookahead_m ** 2, 1e-6)
        return math.atan(curvature * self.wheelbase_m)

    @staticmethod
    def _clamp(value: float, upper: float, lower: float = None) -> float:
        if lower is None:
            lower = -upper
        return max(lower, min(upper, value))


def main(args=None):
    rclpy.init(args=args)
    node = SearchNavNode()
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
