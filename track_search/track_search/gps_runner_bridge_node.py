"""
Run an external GPS runner command and bridge its stdout fixes into ROS NavSatFix.
"""

import math
import re
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus

NODE_NAME = 'gps_runner_bridge_node'


class GPSRunnerBridgeNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('runner_command', ''),
                ('fix_topic', '/fix'),
                ('frame_id', 'gps'),
                ('restart_delay_s', 2.0),
                ('publish_invalid_fixes', False),
                ('verbose_runner_output', False),
            ],
        )

        self.runner_command = str(self.get_parameter('runner_command').value).strip()
        self.fix_topic = str(self.get_parameter('fix_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.restart_delay_s = float(self.get_parameter('restart_delay_s').value)
        self.publish_invalid_fixes = bool(self.get_parameter('publish_invalid_fixes').value)
        self.verbose_runner_output = bool(self.get_parameter('verbose_runner_output').value)

        self.fix_pub = self.create_publisher(NavSatFix, self.fix_topic, 10)

        self._line_pattern = re.compile(
            r'LLA=\s*(?P<lat>[^,]+),\s*(?P<lon>[^,]+),\s*(?P<alt>[^\]]+)'
        )
        self._type_pattern = re.compile(r'Type=(?P<fix_type>[^\]]+)')
        self._process = None
        self._stdout_thread = None
        self._last_start_attempt = 0.0
        self._last_fix_log_time = 0.0

        self.create_timer(1.0, self._ensure_process)

        self.get_logger().info(
            f'{NODE_NAME} started'
            f'\n  fix_topic: {self.fix_topic}'
            f'\n  frame_id: {self.frame_id}'
            f'\n  command_configured: {bool(self.runner_command)}'
        )

    def _ensure_process(self) -> None:
        if not self.runner_command:
            return

        if self._process is not None and self._process.poll() is None:
            return

        now = time.monotonic()
        if (now - self._last_start_attempt) < self.restart_delay_s:
            return

        self._last_start_attempt = now
        self._start_process()

    def _start_process(self) -> None:
        try:
            self.get_logger().info(f'Starting GPS runner command: {self.runner_command}')
            self._process = subprocess.Popen(
                ['bash', '-lc', self.runner_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._stdout_thread = threading.Thread(
                target=self._read_stdout_loop,
                name='gps_runner_stdout',
                daemon=True,
            )
            self._stdout_thread.start()
        except Exception as exc:
            self.get_logger().error(f'Failed to start GPS runner command: {exc}')
            self._process = None

    def _read_stdout_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        try:
            for line in process.stdout:
                stripped = line.strip()
                if not stripped:
                    continue
                if self.verbose_runner_output:
                    self.get_logger().info(f'[runner] {stripped}')
                self._handle_runner_line(stripped)
        finally:
            return_code = process.poll()
            self.get_logger().warn(f'GPS runner exited with code {return_code}')

    def _handle_runner_line(self, line: str) -> None:
        lla_match = self._line_pattern.search(line)
        if lla_match is None:
            return

        lat = self._parse_float(lla_match.group('lat'))
        lon = self._parse_float(lla_match.group('lon'))
        alt = self._parse_float(lla_match.group('alt'))

        type_match = self._type_pattern.search(line)
        fix_type = type_match.group('fix_type') if type_match is not None else 'Unknown'
        is_valid = not any(math.isnan(value) for value in (lat, lon, alt))

        if not is_valid and not self.publish_invalid_fixes:
            return

        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt
        msg.status.status = (
            NavSatStatus.STATUS_FIX if is_valid else NavSatStatus.STATUS_NO_FIX
        )
        msg.status.service = NavSatStatus.SERVICE_GPS
        self.fix_pub.publish(msg)

        now = time.monotonic()
        if is_valid and (self.verbose_runner_output or (now - self._last_fix_log_time) >= 5.0):
            self.get_logger().info(
                f'Published GPS fix lat={lat:.8f} lon={lon:.8f} alt={alt:.2f} type={fix_type}'
            )
            self._last_fix_log_time = now
        elif not is_valid:
            self.get_logger().warn(f'Published invalid GPS fix from runner output type={fix_type}')

    @staticmethod
    def _parse_float(value: str) -> float:
        text = str(value).strip()
        if text.lower() == 'nan':
            return float('nan')
        return float(text)

    def destroy_node(self):
        self._stop_process()
        return super().destroy_node()

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return

        try:
            process.terminate()
            process.wait(timeout=2.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = GPSRunnerBridgeNode()
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
