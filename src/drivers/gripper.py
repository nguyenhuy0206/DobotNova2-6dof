"""PGC gripper over RS485 / Modbus RTU, tunnelled through the Nova 2 controller.

Registers (DH-Robotics PGC):
  0x0100 init (write 0xA5)   0x0101 force %   0x0103 position ‰   0x0104 speed %
  0x0200 initialized         0x0201 state     0x0202 actual position
"""
from __future__ import annotations

import re

from common.errors import RobotError
from drivers.nova2 import Nova2

REG_INIT, REG_FORCE, REG_POSITION, REG_SPEED = 0x0100, 0x0101, 0x0103, 0x0104
REG_STATUS_BASE = 0x0200
INIT_VALUE = 0xA5
VALID_BAUDS = {4800, 9600, 19200, 38400, 57600, 115200}


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

    def disconnect(self) -> str:
        master = self._require_master()
        try:
            return self.robot.dashboard(f"ModbusClose({master})")
        finally:
            self.master = None
            self.transport = None
