"""Direct RTU framing, checked against frames captured from the real PGC."""
import pytest

from common.errors import RobotError
from drivers.gripper import PGCGripperDirect, check_rtu_response, crc16_modbus, rtu_frame


@pytest.mark.parametrize("pdu,frame", [
    ("01 03 02 00 00 03", "01 03 02 00 00 03 04 73"),
    ("01 06 01 00 00 01", "01 06 01 00 00 01 49 f6"),
    ("01 03 02 00 00 01", "01 03 02 00 00 01 85 b2"),
])
def test_frames_match_hardware(pdu, frame):
    assert rtu_frame(bytes.fromhex(pdu)) == bytes.fromhex(frame)


def test_real_responses_have_valid_crc():
    for rx in ("01 03 06 00 00 ff ff 03 e8 21 ef", "01 03 02 00 02 39 85"):
        data = bytes.fromhex(rx)
        assert int.from_bytes(data[-2:], "little") == crc16_modbus(data[:-2])


def test_exception_and_bad_crc():
    exc = rtu_frame(bytes.fromhex("01 83 02"))
    with pytest.raises(RobotError, match="exception code 2"):
        check_rtu_response(exc, 1, 0x03, 7)
    with pytest.raises(RobotError, match="CRC"):
        check_rtu_response(bytes.fromhex("01 03 02 00 02 00 00"), 1, 0x03, 7)


class FakeSerial:
    """Replays the PGC: answers reads from a register map, echoes writes."""
    is_open = True

    def __init__(self):
        self.regs = {0x0200: 1, 0x0201: 1, 0x0202: 1000}
        self.out = b""

    def reset_input_buffer(self): pass
    def flush(self): pass
    def close(self): self.is_open = False

    def write(self, frame):
        slave, fn, addr, val = frame[0], frame[1], int.from_bytes(frame[2:4], "big"), int.from_bytes(frame[4:6], "big")
        if fn == 0x06:
            self.regs[addr] = val
            if addr == 0x0103:   # pretend the fingers arrive / catch something
                self.regs[0x0202] = val
                self.regs[0x0201] = 2 if val < 1000 else 1
            self.out = frame
        else:
            data = b"".join(self.regs.get(addr + i, 0).to_bytes(2, "big") for i in range(val))
            self.out = rtu_frame(bytes([slave, 3, len(data)]) + data)

    def read(self, n):
        out, self.out = self.out[:n], self.out[n:]
        return out


def test_grip_and_open_over_fake_bus():
    g = PGCGripperDirect("fake")
    g.serial = FakeSerial()
    assert g.grip(force=30)["grip_status"] == "object caught"
    assert g.serial.regs[0x0101] == 30
    assert g.open()["position"] == 1000


def test_move_refused_when_not_initialized():
    g = PGCGripperDirect("fake")
    g.serial = FakeSerial()
    g.serial.regs[0x0200] = 0
    with pytest.raises(RobotError, match="initialize"):
        g.open()


def test_simple_api_init_open_close():
    g = PGCGripperDirect("fake")
    g.serial = FakeSerial()
    g.serial.regs[0x0200] = 0
    g.serial.regs[0x0100] = 0
    g.initialize = lambda full=False: (g.serial.regs.__setitem__(0x0200, 1), g.status())[1]
    assert g.init()["initialized"] == 1
    assert g.open()["position"] == 1000
    assert g.close(force=25)["grip_status"] == "object caught" and g.is_holding()
    assert g.go_to(500)["position"] == 500
    g.disconnect()
    assert not g.connected
