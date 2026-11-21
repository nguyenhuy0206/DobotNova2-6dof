"""DH-Robotics PGC gripper over RS485 / Modbus RTU.

Two transports with the same public API:
  * ``PGCGripper``       — tunnelled through the Nova 2 controller (dashboard Modbus)
  * ``PGCGripperDirect`` — straight from this computer via a USB–RS485 adapter
                           (e.g. FT232R at /dev/cu.usbserial-*), no controller involved

Only one Modbus master may drive the bus at a time — don't use both at once.

Registers:
  0x0100 init: 1 = initialize (home), 0xA5 = full recalibration
  0x0101 force %   0x0103 position ‰ (0 closed … 1000 open)   0x0104 speed %
  0x0200 init status: 0 not initialized, 1 initialized, 2 initializing
  0x0201 grip state:  0 moving, 1 arrived (no object), 2 object caught, 3 object dropped
  0x0202 actual position ‰
"""
from __future__ import annotations

import glob
import re
import threading
import time

from common.errors import RobotError
from drivers.nova2 import Nova2

REG_INIT, REG_FORCE, REG_POSITION, REG_SPEED = 0x0100, 0x0101, 0x0103, 0x0104
REG_STATUS_BASE = 0x0200
REG_INIT_STATUS, REG_GRIP_STATE, REG_ACTUAL_POSITION = 0x0200, 0x0201, 0x0202
INIT_VALUE = 0xA5            # full recalibration (controller route keeps this)
INIT_HOME = 0x01             # normal initialize
VALID_BAUDS = {4800, 9600, 19200, 38400, 57600, 115200}

INIT_STATUS = {0: "not initialized", 1: "initialized", 2: "initializing"}


def motion_timeout(speed: int) -> float:
    """Seconds to wait for the fingers to stop. Travel time grows as speed drops: a fixed 5 s
    made every speed-10 open/close time out (17 Nov: 0 of 3 at speed 10, 73 of 73 at >= 20)."""
    return 3.0 + 100.0 / max(1, int(speed))
GRIP_STATE = {0: "moving", 1: "arrived (no object)", 2: "object caught", 3: "object dropped"}


class PGCGripper:
    def __init__(self, robot: Nova2):
        self.robot = robot
        self.master: int | None = None
        self.transport: str | None = None

    @property
    def connected(self) -> bool:
        return self.master is not None

    def connect(self, slave_id: int = 1, baud: int = 115200, parity: str = "N") -> dict:
        if not 1 <= slave_id <= 247 or baud not in VALID_BAUDS:
            raise RobotError("Cấu hình RS485 không hợp lệ.")
        if parity not in {"N", "O", "E"}:
            raise RobotError("Parity phải là N, O hoặc E.")
        if self.master is not None:
            self.robot.dashboard(f"ModbusClose({self.master})")
            self.master = None
        try:
            raw = self.robot.dashboard(f'ModbusRTUCreate({slave_id},{baud},"{parity}",8,1)')
            transport = "ModbusRTUCreate"
        except RobotError as exc:
            # Nova 2 V3 controller returns -10000 for the newer direct RS485 API.
            # Its supported route is the local V3 serial transparent port; baud
            # and parity are then configured in CCBOX.
            if "-10000" not in str(exc):
                raise
            raw = self.robot.dashboard(f"ModbusCreate(127.0.0.1,60000,{slave_id},1)")
            transport = "V3 transparent port 60000"
        match = re.search(r"\{\s*(\d+)\s*\}", raw)
        if not match:
            raise RobotError(f"Không đọc được Modbus master ID: {raw}")
        self.master = int(match.group(1))
        self.transport = transport
        return {"master_id": self.master, "transport": transport, "response": raw}

    def _require_master(self) -> int:
        if self.master is None:
            raise RobotError("Chưa kết nối PGC qua RS485.")
        return self.master

    def _write(self, address: int, value: int) -> str:
        master = self._require_master()
        return self.robot.dashboard(f"SetHoldRegs({master},{address},1,{{{value}}})")

    def status(self) -> dict:
        master = self._require_master()
        return {"master_id": master, "response": self.robot.dashboard(f"GetHoldRegs({master},{REG_STATUS_BASE},3)")}

    def initialize(self) -> str:
        return self._write(REG_INIT, INIT_VALUE)

    def move(self, position: int, force: int, speed: int) -> dict:
        if not 0 <= position <= 1000 or not 20 <= force <= 100 or not 1 <= speed <= 100:
            raise RobotError("PGC: position 0–1000, force 20–100, speed 1–100.")
        return {"force": self._write(REG_FORCE, force),
                "speed": self._write(REG_SPEED, speed),
                "position": self._write(REG_POSITION, position)}

    # Same simple API as PGCGripperDirect (no motion wait on this route).
    def init(self, force_reinit: bool = False, full: bool = True) -> dict:
        return {"response": self.initialize()}

    def go_to(self, position: int, force: int = 30, speed: int = 50, wait: bool = True) -> dict:
        return {"position": position, **self.move(position, force, speed)}

    def open(self, force: int = 30, speed: int = 50, wait: bool = True) -> dict:
        return self.go_to(1000, force, speed, wait)

    def close(self, force: int = 30, speed: int = 50, wait: bool = True) -> dict:
        return self.go_to(0, force, speed, wait)

    def disconnect(self) -> str:
        master = self._require_master()
        try:
            return self.robot.dashboard(f"ModbusClose({master})")
        finally:
            self.master = None
            self.transport = None


# ---------------------------------------------------------------------------
# Direct Modbus RTU (USB–RS485 adapter)
# ---------------------------------------------------------------------------

def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def rtu_frame(pdu: bytes) -> bytes:
    """Append the Modbus CRC (little-endian) to slave+function+data."""
    return pdu + crc16_modbus(pdu).to_bytes(2, "little")


def check_rtu_response(response: bytes, slave_id: int, function: int, expected_len: int) -> bytes:
    if len(response) >= 5 and response[1] == function | 0x80:
        raise RobotError(f"PGC Modbus exception code {response[2]} (function 0x{function:02X})")
    if len(response) != expected_len:
        raise RobotError(f"PGC: expected {expected_len} bytes, got {len(response)}: {response.hex(' ') or 'NO RESPONSE'}")
    if int.from_bytes(response[-2:], "little") != crc16_modbus(response[:-2]):
        raise RobotError(f"PGC: CRC error in {response.hex(' ')}")
    if response[0] != slave_id or response[1] != function:
        raise RobotError(f"PGC: unexpected reply {response.hex(' ')}")
    return response


def find_usb_serial_port() -> str:
    ports = sorted(glob.glob("/dev/cu.usbserial-*") + glob.glob("/dev/ttyUSB*"))
    if not ports:
        raise RobotError("No USB–RS485 adapter found (/dev/cu.usbserial-*).")
    return ports[0]


class PGCGripperDirect:
    """PGC driven straight over RS485 from this computer (pyserial).

    Typical use::

        with PGCGripperDirect() as g:   # connects, auto-detects the port
            g.init()                    # no-op if already initialized
            g.open()
            g.close()                   # close fingers; returns grip status
    """

    def __init__(self, port: str | None = None, timeout: float = 0.5):
        self.port_name = port
        self.timeout = timeout
        self.serial = None
        self.slave_id = 1
        self.transport: str | None = None
        self._io_lock = threading.Lock()   # one request/response on the bus at a time

    @property
    def connected(self) -> bool:
        return self.serial is not None and self.serial.is_open

    def connect(self, slave_id: int = 1, baud: int = 115200, parity: str = "N") -> dict:
        import serial  # pyserial

        if not 1 <= slave_id <= 247 or baud not in VALID_BAUDS:
            raise RobotError("Cấu hình RS485 không hợp lệ.")
        if parity not in {"N", "O", "E"}:
            raise RobotError("Parity phải là N, O hoặc E.")
        self._close_port()
        port = self.port_name or find_usb_serial_port()
        try:
            self.serial = serial.Serial(port, baud, bytesize=8, parity=parity, stopbits=1, timeout=self.timeout)
        except serial.SerialException as exc:
            raise RobotError(f"Cannot open {port}: {exc}") from exc
        self.port_name, self.slave_id = port, slave_id
        self.transport = f"direct RTU {port} {baud} 8{parity}1"
        # Probe so a wrong port/baud/slave fails here, not on the first move.
        return {"transport": self.transport, "slave_id": slave_id, **self.status()}

    # -- Modbus primitives ----------------------------------------------
    def _transact(self, request: bytes, expected_len: int) -> bytes:
        if not self.connected:
            raise RobotError("Chưa kết nối PGC qua RS485.")
        frame = rtu_frame(request)
        with self._io_lock:
            self.serial.reset_input_buffer()
            self.serial.write(frame)
            self.serial.flush()
            response = self.serial.read(expected_len)
            if len(response) >= 3 and response[1] & 0x80:   # exception replies are 5 bytes
                response += self.serial.read(max(0, 5 - len(response)))
        return check_rtu_response(response, self.slave_id, request[1], expected_len)

    def read_registers(self, address: int, count: int) -> list[int]:
        request = bytes([self.slave_id, 0x03]) + address.to_bytes(2, "big") + count.to_bytes(2, "big")
        response = self._transact(request, 5 + 2 * count)
        return [int.from_bytes(response[3 + 2 * i:5 + 2 * i], "big") for i in range(count)]

    def write_register(self, address: int, value: int) -> None:
        request = bytes([self.slave_id, 0x06]) + address.to_bytes(2, "big") + value.to_bytes(2, "big")
        response = self._transact(request, 8)
        if response[:6] != request:
            raise RobotError(f"PGC: write echo mismatch {response.hex(' ')}")

    # -- gripper API (same shape as PGCGripper) --------------------------
    def status(self) -> dict:
        init, state, position = self.read_registers(REG_STATUS_BASE, 3)
        return {"initialized": init, "init_status": INIT_STATUS.get(init, f"unknown ({init})"),
                "grip_state": state, "grip_status": GRIP_STATE.get(state, f"unknown ({state})"),
                "position": position}

    def initialize(self, full: bool = False, wait: bool = True, timeout: float = 15.0) -> dict:
        self.write_register(REG_INIT, INIT_VALUE if full else INIT_HOME)
        if not wait:
            return self.status()
        deadline = time.monotonic() + timeout
        time.sleep(0.3)
        while time.monotonic() < deadline:
            status = self.status()
            if status["initialized"] == 1:
                return status
            time.sleep(0.2)
        raise RobotError("PGC initialization timed out.")

    def move(self, position: int, force: int, speed: int) -> dict:
        if not 0 <= position <= 1000 or not 20 <= force <= 100 or not 1 <= speed <= 100:
            raise RobotError("PGC: position 0–1000, force 20–100, speed 1–100.")
        if self.read_registers(REG_INIT_STATUS, 1)[0] != 1:
            raise RobotError("PGC chưa initialize — gọi initialize() trước.")
        self.write_register(REG_FORCE, force)
        self.write_register(REG_SPEED, speed)
        self.write_register(REG_POSITION, position)
        return {"position": position, "force": force, "speed": speed}

    def wait_motion(self, timeout: float = 5.0, target: int | None = None) -> dict:
        """Poll until the fingers stop; grip_state tells whether an object was caught.
        go_to passes motion_timeout(speed) and the target position.

        grip_state 3 ("object dropped") is NOT the end of a motion by itself: opening while
        holding reports 3 at once, while the fingers are still at the cube (17 Nov: open_gripper
        returned position 521, a moment later 1000). With 3 the motion is done only when the
        fingers reached the target (±20) or stopped moving."""
        deadline = time.monotonic() + timeout
        time.sleep(0.15)   # state reads 'arrived' briefly before motion starts
        last_pos, still_since = None, None
        while time.monotonic() < deadline:
            status = self.status()
            state, pos = status["grip_state"], status["position"]
            if state in (1, 2):
                return status
            if state == 3:
                if target is None or abs(pos - target) <= 20:
                    return status
                now = time.monotonic()
                if last_pos is not None and abs(pos - last_pos) <= 2:
                    still_since = still_since or now
                    if now - still_since >= 0.4:
                        return status              # stopped short of the target
                else:
                    still_since = None
                last_pos = pos
            time.sleep(0.05)
        raise RobotError("PGC motion timed out.")

    # -- simple reusable API ---------------------------------------------
    def init(self, force_reinit: bool = False, full: bool = False) -> dict:
        """Initialize only if needed (after power-up); ``force_reinit`` always does it."""
        status = self.status()
        if status["initialized"] == 1 and not force_reinit:
            return status
        return self.initialize(full=full)

    def go_to(self, position: int, force: int = 30, speed: int = 50, wait: bool = True) -> dict:
        """Move fingers to 0 (closed) … 1000 (open) and, by default, wait until they stop."""
        self.move(position, force, speed)
        return self.wait_motion(motion_timeout(speed), target=position) if wait else self.status()

    def open(self, force: int = 30, speed: int = 50, wait: bool = True) -> dict:
        return self.go_to(1000, force, speed, wait)

    def close(self, force: int = 30, speed: int = 50, wait: bool = True) -> dict:
        """Close fingers until an object stops them (grip_state 2) or fully closed (1)."""
        return self.go_to(0, force, speed, wait)

    grip = close

    def is_holding(self) -> bool:
        return self.status()["grip_state"] == 2

    # -- connection lifecycle -------------------------------------------
    def _close_port(self) -> None:
        if self.serial is not None:
            self.serial.close()
        self.serial = None

    def disconnect(self) -> str:
        self._close_port()
        self.transport = None
        return "serial closed"

    def __enter__(self) -> "PGCGripperDirect":
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()


# ---------------------------------------------------------------------------
# Through the UI app (ui/app.py owns the bus; every other process asks it)
# ---------------------------------------------------------------------------

class PGCGripperRemote:
    """The gripper driven by ANOTHER process — ui/app.py — over its local HTTP API.

    Modbus RTU allows one master on the bus. The UI app already holds the serial port (its
    Debug tab polls the gripper state), so a second process opening the port too makes the
    two talk over each other: replies get read by the wrong process ("expected 11 bytes,
    got 0: NO RESPONSE", 17 Nov). With gripper.via_ui every other process (episodes,
    scripts) sends its gripper calls to the app, whose single driver serialises them with
    the UI's own. Same API as PGCGripperDirect.
    """

    def __init__(self, base_url: str, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = f"via ui/app.py at {self.base_url}"
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def _call(self, op: str, timeout: float | None = None, **kw) -> dict:
        import json as _json
        import urllib.error
        import urllib.request

        body = _json.dumps({"op": op, **kw, "confirm": True}).encode()
        req = urllib.request.Request(f"{self.base_url}/api/gripper/driver", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return _json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            try:
                why = _json.loads(exc.read().decode()).get("error")
            except ValueError:
                why = None
            raise RobotError(f"gripper (via ui/app.py): {why or exc}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise RobotError(f"gripper: cannot reach ui/app.py at {self.base_url} ({exc}). It owns the gripper's "
                             "serial port — start it (scripts/run_vision.sh).") from None

    def connect(self, slave_id: int = 1, baud: int = 115200, parity: str = "N") -> dict:
        status = self._call("status")              # the app opens + initializes its driver on first use
        self._connected = True
        return {"transport": self.transport, **status}

    def init(self, force_reinit: bool = False, full: bool = False) -> dict:
        return self._call("init", force_reinit=force_reinit, full=full)

    def status(self) -> dict:
        return self._call("status")

    def go_to(self, position: int, force: int = 30, speed: int = 50, wait: bool = True) -> dict:
        return self._call("go_to", timeout=max(self.timeout, motion_timeout(speed) + 10), position=int(position),
                          force=int(force), speed=int(speed), wait=bool(wait))

    def open(self, force: int = 30, speed: int = 50, wait: bool = True) -> dict:
        return self.go_to(1000, force, speed, wait)

    def close(self, force: int = 30, speed: int = 50, wait: bool = True) -> dict:
        return self.go_to(0, force, speed, wait)

    grip = close

    def is_holding(self) -> bool:
        return self.status().get("grip_state") == 2

    def disconnect(self) -> str:
        self._connected = False
        return "remote: nothing to close (ui/app.py keeps the port)"
