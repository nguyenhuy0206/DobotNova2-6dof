"""The catalogue of primitive skills exposed as tools (agent, harness, planner).

Add a primitive in actions/primitives.py, describe it here once, and it becomes
an Astra function tool and a valid step for composite skills (memory/skills.py).
"""
from __future__ import annotations

from actions.executor import ActionSpec
from actions.primitives import PrimitiveSkills


def _obj(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def build_registry(p: PrimitiveSkills, scene=None, runner=None) -> list[ActionSpec]:
    """Primitive robot skills, plus perception tools (``scene``) and composite skill
    tools (``runner``) when given."""
    specs = _robot_specs(p)
    if scene is not None:
        specs += perception_specs(scene)
    if runner is not None:
        specs += skill_specs(runner)
    return specs


def _robot_specs(p: PrimitiveSkills) -> list[ActionSpec]:
    c, m = p.cfg, p.cfg.motion
    top = p.speed_cap or c.speed.max
    speed = {"type": "integer", "minimum": c.speed.min, "maximum": top,
             "description": (f"Ignored: every motion runs at fixed {p.fixed_speed} %" if p.fixed_speed else
                             f"Speed % (default {min(m.default_speed, top)}, max {top})")}
    num = {"type": "number"}
    force = {"type": "integer", "minimum": 20, "maximum": 100, "description": "Grip force %"}
    gspeed = ({"type": "integer", "enum": [p.grip_speed], "description": f"fixed at {p.grip_speed} %"}
              if p.grip_speed is not None else
              {"type": "integer", "minimum": 1, "maximum": 100, "description": "Finger speed %"})
    open_to = p.grip_open_position if p.grip_open_position is not None else 1000
    at = f" at finger speed {p.grip_speed} %" if p.grip_speed is not None else ""
    ws = c.workspace
    return [
        ActionSpec("get_state",
                   "Full robot state: mode, alarm, moving, joints (deg), fingertip pose [X,Y,Z mm, Rx,Ry,Rz deg], "
                   "flange_pose, gripper status, table limits and saved position names. Every request already "
                   "carries a STATE block measured just before it (mode, alarm, moving, fingertip pose, gripper "
                   "position/status, table_z, approach_zone_top); this tool adds joints, flange_pose, the full "
                   "gripper record and saved positions.",
                   p.get_state, _obj({}, []), read_only=True),
        ActionSpec("move_to_pose",
                   f"Move the gripper FINGERTIPS to an absolute pose in the robot base frame (the gripper "
                   f"length is handled for you). Rx=180, Ry=0 points the gripper straight down. Workspace "
                   f"X{list(ws.x)} Y{list(ws.y)} Z{list(ws.z)} mm. linear=true for a straight-line path (MovL), "
                   "false for a joint-space path (MovJ). Blocks until arrived.",
                   p.move_to_pose,
                   _obj({"x": num, "y": num, "z": num, "rx": num, "ry": num, "rz": num,
                         "linear": {"type": "boolean"}, "speed": speed},
                        ["x", "y", "z", "rx", "ry", "rz"]),
                   moves_robot=True),
        ActionSpec("move_to_joints",
                   f"Move all six joints to absolute angles in degrees (limits ±{m.joint_limits_deg}). Blocks until arrived.",
                   p.move_to_joints,
                   _obj({"joints": {"type": "array", "items": num, "minItems": 6, "maxItems": 6},
                         "speed": speed}, ["joints"]),
                   moves_robot=True),
        ActionSpec("move_relative",
                   f"Straight-line offset from the current pose. frame='base' (robot axes) or 'tool' "
                   f"(gripper axes; +Z = approach direction). Max {m.max_relative_mm} mm and "
                   f"{m.max_relative_deg}° per call. Blocks until arrived.",
                   p.move_relative,
                   _obj({"dx": num, "dy": num, "dz": num, "drx": num, "dry": num, "drz": num,
                         "frame": {"type": "string", "enum": ["base", "tool"]}, "speed": speed}, []),
                   moves_robot=True),
        ActionSpec("jog_joint",
                   f"Rotate one joint by a relative angle ({c.joint_step.min_deg}–{c.joint_step.max_deg}°).",
                   p.jog_joint,
                   _obj({"joint": {"type": "integer", "minimum": 1, "maximum": 6}, "degrees": num, "speed": speed},
                        ["joint", "degrees"]),
                   moves_robot=True),
        ActionSpec("go_to_position",
                   "Move to a saved, named position (see get_state.saved_positions). 'home' always exists.",
                   p.go_to_position, _obj({"name": {"type": "string"}, "speed": speed}, ["name"]),
                   moves_robot=True),
        ActionSpec("save_position",
                   "Remember the current pose under a name so it can be reused with go_to_position.",
                   p.save_position, _obj({"name": {"type": "string"}}, ["name"])),
        ActionSpec("open_gripper", f"Open the fingers to position {open_to} of 1000"
                   + (" (fully open)" if open_to == 1000 else "") + f"{at}.",
                   p.open_gripper, _obj({"force": force, "speed": gspeed}, []), moves_robot=True),
        ActionSpec("close_gripper",
                   ("Close the fingers until they hit an object" + (
                       f" or reach position {p.grip_close_position} (of 1000; they never close further)"
                       if p.grip_close_position is not None else " or are fully closed") +
                    f"{at}. Result 'holding' is true when an object was caught."),
                   p.close_gripper, _obj({"force": force, "speed": gspeed}, []), moves_robot=True),
        ActionSpec("set_gripper", "Move the fingers to a position: 0 = closed … 1000 = fully open.",
                   p.set_gripper,
                   _obj({"position": {"type": "integer", "minimum": 0, "maximum": 1000},
                         "force": force, "speed": gspeed}, ["position"]),
                   moves_robot=True),
        ActionSpec("stop", "Pause motion immediately (Pause()); the operator resumes the robot.",
                   p.stop, _obj({}, [])),
    ]


def perception_specs(scene) -> list[ActionSpec]:
    colors = {"type": "string", "enum": scene.colors}
    camera = {"type": "string", "enum": sorted(scene.sources), "description": "env = D455 overview, wrist = gripper view"}
    return [
        ActionSpec("look",
                   "Take a picture with a camera. Returns the image (always attached) and its size; nothing is "
                   "detected for you. env = the D455 (colour + depth, calibrated to the robot base frame), wrist = "
                   "the webcam on the gripper (colour only; the fingertip points are marked, fingertips_px). "
                   "zoom=[x, y, w, h] returns that region of the full image "
                   "enlarged. depth_at answers for the pixels of the last env picture.",
                   scene.look, _obj({"camera": camera,
                                     "zoom": {"type": "array", "items": {"type": "integer"}, "minItems": 4,
                                              "maxItems": 4,
                                              "description": "[x, y, w, h] region of the full image to enlarge"}},
                                    []),
                   read_only=True),
    ] + ([
        ActionSpec("depth_at",
                   "Depth at pixels of the LAST env picture (full-image coordinates, the same frame you saw): per "
                   "pixel the median valid depth in a (2*window_px+1)^2 window, the 3D point in the camera frame and "
                   "base_xyz_mm, the point in the robot base frame (on the surface the D455 sees at that pixel). "
                   "valid_fraction = share of the window with depth; depth_spread_mm = 10–90 % spread of the depth in "
                   "the window (large at object edges). plane_z (mm): also at_plane_xyz_mm, where the pixel's viewing ray meets the "
                   "horizontal plane z = plane_z (no depth used). The same picture comes back with the asked "
                   "pixels marked and numbered in order.",
                   scene.depth_at,
                   _obj({"pixels": {"type": "array", "minItems": 1, "maxItems": 50,
                                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
                         "window_px": {"type": "integer", "minimum": 0, "maximum": 10},
                         "plane_z": {"type": "number"}}, ["pixels"]),
                   read_only=True),
    ] if hasattr(scene, "depth_at") else []) + [
        ActionSpec("find_objects",
                   "Detect objects of one colour (largest first). Env camera: base_xyz_mm = robot base frame position "
                   "of the point at the blob centre, on the surface the camera sees there (when calibrated). If nothing is visible it returns last_seen (where it was, "
                   "seconds ago).",
                   scene.find_report,
                   _obj({"color": colors, "camera": camera}, ["color"]), read_only=True),
    ]


def skill_specs(runner) -> list[ActionSpec]:
    step = {"type": "object", "properties": {"tool": {"type": "string"}, "check": {"type": "string"},
                                             "routine": {"type": "string"}, "args": {"type": "object"}}}
    from actions.local_steps import CHECKS, ROUTINES
    return [
        ActionSpec("list_skills", "List skills (name, version, status, params, counts). name=… returns one skill "
                   "in full (steps, checks, recovery) — needed before saving a new version of it.",
                   runner.list, _obj({"name": {"type": "string"}}, []), read_only=True),
        ActionSpec("run_skill",
                   "Run a skill by name with params. It runs locally without asking you after every step: "
                   "primitive tools (validated, safety-checked), local checks and routines. Stops at the first "
                   "failing step, runs the skill's recovery (open/lift only) and returns a failure code, trace "
                   "and detail. " + ("Candidate skills run only with operator approval." if runner.require_validation else
                                    "Candidate skills may run immediately; validation requires operator ledger review."),
                   runner.run, _obj({"name": {"type": "string"}, "params": {"type": "object"}}, ["name"]),
                   moves_robot=True, delegates=True),
        ActionSpec("save_skill",
                   "Save a skill (or a new version of one) as a CANDIDATE. Steps: {tool, args} primitive calls, "
                   f"{{check, args}} local checks {sorted(CHECKS)}, {{routine, args}} local loops {sorted(ROUTINES)}. "
                   "Args may use \"{param}\" or \"=expression\" (e.g. \"={z_top} + 30\"); table_z is "
                   "always available. params: defaults (null = required). recovery: only open_gripper / "
                   "routine lift_to. list_routines shows every routine/check, including your own.",
                   runner.save,
                   _obj({"name": {"type": "string"}, "description": {"type": "string"},
                         "steps": {"type": "array", "items": step}, "params": {"type": "object"},
                         "tags": {"type": "array", "items": {"type": "string"}},
                         "preconditions": {"type": "array", "items": step},
                         "success_conditions": {"type": "array", "items": step},
                         "recovery": {"type": "array", "items": step},
                         "failure_conditions": {"type": "object"}},
                        ["name", "description", "steps"])),
    ]
