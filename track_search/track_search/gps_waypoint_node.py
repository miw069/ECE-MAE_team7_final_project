"""
Convert GPS latitude/longitude fixes into a local XY pose estimate.
"""

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix

NODE_NAME = 'gps_waypoint_node'


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = yaw * 0.5
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


class GPSWaypointNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('gps_topic', '/fix'),
                ('pose_topic', '/gps/local_pose'),
                ('frame_id', 'map'),
                ('use_first_fix_as_origin', True),
                ('origin_lat', float('nan')),
                ('origin_lon', float('nan')),
                ('min_motion_for_heading_m', 0.25),
                ('verbose_gps_logging', False),
                ('republish_last_pose_hz', 5.0),
                ('max_pose_hold_s', 15.0),
            ],
        )

        self.gps_topic = str(self.get_parameter('gps_topic').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.use_first_fix_as_origin = bool(self.get_parameter('use_first_fix_as_origin').value)
        self.origin_lat = float(self.get_parameter('origin_lat').value)
        self.origin_lon = float(self.get_parameter('origin_lon').value)
        self.min_motion_for_heading_m = float(self.get_parameter('min_motion_for_heading_m').value)
        self.verbose_gps_logging = bool(self.get_parameter('verbose_gps_logging').value)
        self.republish_last_pose_hz = float(self.get_parameter('republish_last_pose_hz').value)
        self.max_pose_hold_s = float(self.get_parameter('max_pose_hold_s').value)

        self.origin_ready = not (math.isnan(self.origin_lat) or math.isnan(self.origin_lon))
        self.current_yaw = 0.0
        self.prev_x = None
        self.prev_y = None
        self._last_invalid_fix_warn = 0.0
        self._last_valid_pose_time = 0.0
        self._last_pose_msg = None

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.create_subscription(
            NavSatFix,
            self.gps_topic,
            self._gps_callback,
            qos_profile_sensor_data,
        )
        self.add_on_set_parameters_callback(self._on_params_changed)
        self.create_timer(
            1.0 / max(self.republish_last_pose_hz, 1e-3),
            self._republish_last_pose,
        )

        origin_text = (
            f'({self.origin_lat:.8f}, {self.origin_lon:.8f})'
            if self.origin_ready
            else 'first valid GPS fix'
        )
        self.get_logger().info(
            f'{NODE_NAME} started'
            f'\n  gps_topic: {self.gps_topic}'
            f'\n  pose_topic: {self.pose_topic}'
            f'\n  local origin: {origin_text}'
            f'\n  verbose_gps_logging: {self.verbose_gps_logging}'
            f'\n  republish_last_pose_hz: {self.republish_last_pose_hz:.2f}'
            f'\n  max_pose_hold_s: {self.max_pose_hold_s:.2f}'
        )

    def _gps_callback(self, msg: NavSatFix) -> None:
        if msg.status.status < 0:
            now = time.monotonic()
            if now - self._last_invalid_fix_warn > 5.0:
                self.get_logger().warn('Waiting for valid GPS fix...')
                self._last_invalid_fix_warn = now
            return

        lat = float(msg.latitude)
        lon = float(msg.longitude)

        if not self.origin_ready:
            if self.use_first_fix_as_origin:
                self.origin_lat = lat
                self.origin_lon = lon
                self.origin_ready = True
                self.get_logger().info(
                    f'Local XY origin initialized from GPS fix: '
                    f'({self.origin_lat:.8f}, {self.origin_lon:.8f})'
                )
            else:
                self.get_logger().error('Origin is unset and use_first_fix_as_origin=false')
                return

        x, y = self._latlon_to_local_xy(lat, lon, self.origin_lat, self.origin_lon)
        if self.prev_x is not None and self.prev_y is not None:
            dx = x - self.prev_x
            dy = y - self.prev_y
            if math.hypot(dx, dy) >= self.min_motion_for_heading_m:
                self.current_yaw = math.atan2(dy, dx)

        self.prev_x = x
        self.prev_y = y

        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = 0.0
        qx, qy, qz, qw = _yaw_to_quaternion(self.current_yaw)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self._publish_pose(pose_msg)

        if self.verbose_gps_logging:
            self.get_logger().info(
                'GPS fix'
                f' lat={lat:.8f}'
                f' lon={lon:.8f}'
                f' local_x={x:.2f} m'
                f' local_y={y:.2f} m'
                f' heading={math.degrees(self.current_yaw):.1f} deg'
                f' status={msg.status.status}'
            )

    def _on_params_changed(self, params) -> SetParametersResult:
        for param in params:
            if param.name == 'verbose_gps_logging':
                self.verbose_gps_logging = bool(param.value)
                self.get_logger().info(
                    f'verbose_gps_logging set to {self.verbose_gps_logging}'
                )

        return SetParametersResult(successful=True)

    def _publish_pose(self, pose_msg: PoseStamped) -> None:
        self._last_pose_msg = pose_msg
        self._last_valid_pose_time = time.monotonic()
        self.pose_pub.publish(pose_msg)

    def _republish_last_pose(self) -> None:
        if self._last_pose_msg is None:
            return

        age_s = time.monotonic() - self._last_valid_pose_time
        if age_s > self.max_pose_hold_s:
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self._last_pose_msg.header.frame_id
        pose_msg.pose = self._last_pose_msg.pose
        self.pose_pub.publish(pose_msg)

    @staticmethod
    def _latlon_to_local_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
        x = (lon - lon0) * meters_per_deg_lon
        y = (lat - lat0) * meters_per_deg_lat
        return x, y


def main(args=None):
    rclpy.init(args=args)
    node = GPSWaypointNode()
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
