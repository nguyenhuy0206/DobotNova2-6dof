"""The catalogue of actions exposed to agents (and anything tool-driven).

Add a new capability here once and it automatically becomes an Astra tool.
"""
from __future__ import annotations

from actions.executor import ActionSpec
from actions.robot_actions import RobotActions


def _obj(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def build_registry(a: RobotActions) -> list[ActionSpec]:
    speed = {"type": "integer", "minimum": a.config.speed.min, "maximum": a.config.speed.max,
             "description": "Speed percent"}
    return [
        ActionSpec("get_robot_state", "Read live robot feedback: mode, enabled, error, joints (deg), TCP pose (mm/deg) and saved positions.",
                   a.state, _obj({}, []), read_only=True),
        ActionSpec("list_saved_positions", "List taught positions (id, name, joints, pose).",
                   a.positions.all, _obj({}, []), read_only=True),
        ActionSpec("joint_step", "Move a single joint by a small relative angle.",
                   a.joint_step,
                   _obj({"joint": {"type": "integer", "minimum": 1, "maximum": 6},
                         "degrees": {"type": "number", "description":
                                     f"Relative angle, |deg| in {a.config.joint_step.min_deg}–{a.config.joint_step.max_deg}"},
                         "speed": speed}, ["joint", "degrees", "speed"]),
                   moves_robot=True),
        ActionSpec("move_to_saved_position", "Move (JointMovJ) to a taught position by id.",
                   a.move_to_saved,
                   _obj({"position_id": {"type": "string"}, "speed": speed}, ["position_id", "speed"]),
                   moves_robot=True),
        ActionSpec("gripper_open", "Fully open the PGC gripper.", a.gripper_open, _obj({}, []), moves_robot=True),
        ActionSpec("gripper_close", "Fully close the PGC gripper.", a.gripper_close, _obj({}, []), moves_robot=True),
        ActionSpec("gripper_move", "Move the gripper to a position (0 closed – 1000 open).",
                   a.gripper_move,
                   _obj({"position": {"type": "integer", "minimum": 0, "maximum": 1000},
                         "force": {"type": "integer", "minimum": 20, "maximum": 100}},
                        ["position"]),
                   moves_robot=True),
        ActionSpec("stop", "Immediately stop the robot and clear the motion queue.",
                   lambda: a.system("stop"), _obj({}, []), moves_robot=False),
    ]
