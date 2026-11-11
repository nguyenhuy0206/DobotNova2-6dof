import socket

import pytest

from common.errors import RobotError
from drivers.nova2 import dashboard_response_ok, parse_feedback, recv_frame
from sim_robot import make_feedback_packet


def test_parse_feedback_ok():
    fb = parse_feedback(make_feedback_packet())
    assert fb["mode"] == 5 and fb["enabled"] == 1 and fb["robot_type"] == 101
    assert fb["joints_deg"] == [1, 2, 3, 4, 5, 6]
    assert fb["pose"] == [10, 20, 30, 40, 50, 60]


@pytest.mark.parametrize("packet", [b"\x00" * 100, make_feedback_packet(robot_type=7)])
def test_parse_feedback_rejects(packet):
    with pytest.raises(RobotError):
        parse_feedback(packet)


def test_parse_feedback_bad_magic():
    data = bytearray(make_feedback_packet())
    data[48] ^= 0xFF
    with pytest.raises(RobotError, match="V3"):
        parse_feedback(bytes(data))


def test_dashboard_response():
    assert dashboard_response_ok("0,{},EnableRobot();", "EnableRobot()")
    with pytest.raises(RobotError):
        dashboard_response_ok("-1,{},EnableRobot();", "EnableRobot()")


def test_recv_frame_splits_on_semicolon():
    a, b = socket.socketpair()
    try:
        b.sendall(b"0,{3},ModbusCreate();trailing")
        assert recv_frame(a, 1.0) == "0,{3},ModbusCreate();"
    finally:
        a.close(); b.close()
