"""
Thin wrapper around the pyvesc library for serial communication with the VESC.
"""

import time

from pyvesc import VESC


class VESC_:
    def __init__(self):
        self.serial_port = '/dev/ttyACM0'
        self.baudrate = 115200
        self.is_inverted = False
        self.has_sensor = False
        self.start_heartbeat = True
        print('Connecting to VESC...')
        try:
            self.v = VESC(
                self.serial_port,
                self.baudrate,
                self.has_sensor,
                self.start_heartbeat,
            )
            print('VESC connected')
            self.send_rpm(0)
            self.inverted = -1 if self.is_inverted else 1
        except Exception as exc:
            print(f'Could not connect to VESC: {exc}')
            raise

    def print_firmware_version(self):
        print('VESC Firmware Version:', self.v.get_firmware_version())

    def send_servo_angle(self, angle: float):
        self.v.set_servo(angle)

    def send_rpm(self, rpm: int):
        self.v.set_rpm(rpm)

    def send_duty_cycle(self, dc: float):
        self.v.set_duty_cycle(dc)

    def send_current(self, curr: float):
        self.v.set_current(curr)

    def get_rpm(self) -> float:
        return self.v.get_rpm() * self.inverted

    def get_motor_position(self) -> float:
        return self.v.get_motor_position()


if __name__ == '__main__':
    print('MAKE SURE YOUR CAR IS ON A STAND AND WHEELS CAN SPIN FREELY')
    input('Hit ENTER to continue...')
    v = VESC_()
    v.print_firmware_version()

    print('Steering: right -> straight -> left')
    v.send_servo_angle(1.0)
    time.sleep(2)
    v.send_servo_angle(0.5)
    time.sleep(2)
    v.send_servo_angle(0.0)
    time.sleep(2)
    v.send_servo_angle(0.5)

    print('Throttle: forward 2 s then stop')
    time.sleep(1)
    v.send_rpm(15000)
    time.sleep(2)
    v.send_rpm(0)
