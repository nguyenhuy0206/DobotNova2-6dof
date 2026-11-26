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
    specs = _robot_specs(p) + pick_place_specs(p)
    if scene is not None:
        specs += perception_specs(scene, table_z=p.table_z_mm)
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
                   f"false (default) for a joint-space path (MovJ); the joint-space path is used only when start and "
                   f"target are at least {p.table.movj_min_height_mm:g} mm above the table and the path checks clear, "
                   "otherwise the move goes in a straight line and the result's 'note' says so. Blocks until arrived.",
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
                   f"(gripper axes; +Z = approach direction). Rotations (drx, dry, drz) pivot around the fingertip, "
                   f"about the base axes (frame=base) or the gripper axes (frame=tool). Max {m.max_relative_mm} mm "
                   f"and {m.max_relative_deg}° per call. Blocks until arrived.",
                   p.move_relative,
                   _obj({"dx": num, "dy": num, "dz": num, "drx": num, "dry": num, "drz": num,
                         "frame": {"type": "string", "enum": ["base", "tool"]}, "speed": speed}, []),
                   moves_robot=True),
        ActionSpec("insert",
                   f"Needle insertion: move the fingertips straight along one of the gripper's own axes by "
                   f"depth_mm (0.1 to {m.insert_max_mm:g} mm): axis +z (the direction the gripper points, straight "
                   f"down when Rx=180, Ry=0), +x/-x or +y/-y (the fingers close along y); default "
                   f"{m.insert_axis}, at {m.insert_speed} % (the only motion not at the fixed speed), "
                   "keeping the orientation. Aim first, then insert. No stall "
                   "watch on this stroke; table floor, path and reach checks apply. retract=true moves back "
                   "along the same line to the start afterwards. Returns the pose, inserted_mm, retracted.",
                   p.insert, _obj({"depth_mm": {"type": "number", "minimum": 0.1,
                                                "maximum": m.insert_max_mm},
                                   "retract": {"type": "boolean"},
                                   "axis": {"type": "string", "enum": ["+z", "+x", "-x", "+y", "-y"]}},
                                  ["depth_mm"]),
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
        ActionSpec("set_gripper", "Move the fingers to a position: 0 = closed … 1000 = fully open"
                   + (f"; positions below {p.grip_close_position} are refused (the fingers never close further)."
                      if p.grip_close_position is not None else "."),
                   p.set_gripper,
                   _obj({"position": {"type": "integer", "minimum": p.grip_close_position or 0, "maximum": 1000},
                         "force": force, "speed": gspeed}, ["position"]),
                   moves_robot=True),
        ActionSpec("stop", "Pause motion immediately (Pause()); the operator resumes the robot.",
                   p.stop, _obj({}, [])),
    ]


def pick_place_specs(p) -> list[ActionSpec]:
    from actions.pick_place import PickPlace
    helper = PickPlace(p)
    num = {"type": "number"}
    return [
        ActionSpec("pick_object",
                   "Pick at robot base x,y,top_z (mm), using footprint edge yaw_deg (base XY degrees). "
                   "Fingertips point down (Rx=180,Ry=0); closing is along flange Y. For a square footprint, "
                   "rz is the nearest feasible yaw equivalent modulo 90 degrees. height_mm defaults to 26. "
                   "Open, approach at top_z+60, grasp at top_z-min(height_mm/2,15), close, lift to top_z+80. "
                   "Returns ok, ordered steps, holding, gripper position and fingertip pose; on failure, "
                   "failed_step and detail. Every actuation uses the checked primitives.",
                   helper.pick_object, _obj({"x": num, "y": num, "top_z": num, "yaw_deg": num,
                                             "height_mm": num}, ["x", "y", "top_z", "yaw_deg"]),
                   moves_robot=True, delegates=True, reports_status=True),
        ActionSpec("place_object",
                   "Place at robot base x,y,top_z (support top, mm). held_height_mm defaults to 26. "
                   "Approach at top_z+held_height_mm+60; lower with the held object's bottom 3 mm above "
                   "top_z, open, lift to the approach height. Assumes the pick_object grasp depth. "
                   "yaw_deg is an optional base XY footprint edge angle (square equivalents modulo 90); "
                   "omitting it keeps the current rz. Returns ok, ordered steps and fingertip pose, "
                   "or failed_step and detail. Every actuation uses the checked primitives.",
                   helper.place_object, _obj({"x": num, "y": num, "top_z": num, "yaw_deg": num,
                                              "held_height_mm": num}, ["x", "y", "top_z"]),
                   moves_robot=True, delegates=True, reports_status=True),
    ]


def perception_specs(scene, table_z=None) -> list[ActionSpec]:
    colors = {"type": "string", "enum": scene.colors}
    names = {"env": "env = D455 overview" if getattr(scene, "env_depth", True) else "env = the second colour camera",
             "wrist": "wrist = gripper view"}
    camera = {"type": "string", "enum": sorted(scene.sources),
              "description": ", ".join(names.get(c, c) for c in sorted(scene.sources))}
    return [
        ActionSpec("look",
                   "Take a picture with a camera. Returns the image (always attached); nothing is detected for "
                   "you. image_size = the full camera image, picture_size = the attached picture. "
                   + (("env = the D455 (colour + depth, calibrated to the robot base frame), "
                       if getattr(scene, "env_depth", True) else
                       "env = a second colour camera (no depth, not calibrated to the robot), ")
                      if "env" in scene.sources else "") + "wrist = the webcam on the gripper (colour only). The pictures are the raw camera images; nothing is drawn on them. zoom=[x, y, w, h] "
                   "(full-image pixels) returns that region enlarged by picture_scale: full pixel = "
                   "x + picture_x / picture_scale, y + picture_y / picture_scale. Each picture reaches you with a "
                   "label (camera, full or zoom region, step)."
                   + (" depth_at measures in the frame of the last env look."
                      if "env" in scene.sources and getattr(scene, "env_depth", True) else ""),
                   scene.look, _obj({"camera": camera,
                                     "zoom": {"type": "array", "items": {"type": "integer"}, "minItems": 4,
                                              "maxItems": 4,
                                              "description": "[x, y, w, h] region of the full image to enlarge"}},
                                    []),
                   read_only=True),
    ] + ([
        ActionSpec("depth_at",
                   "Depth at pixels of the frame of the LAST env look. Pixels are FULL-image pixels (u, v) always, "
                   "also after a zoomed look (a point in a zoomed picture is at u = x + px / picture_scale, "
                   "v = y + py / picture_scale, with [x, y, w, h] = its zoom). Per "
                   "pixel the median valid depth in a (2*window_px+1)^2 window, the 3D point in the camera frame and "
                   "base_xyz_mm, the point in the robot base frame (on the surface the D455 sees at that pixel). "
                   "valid_fraction = share of the window with depth; depth_spread_mm = 10–90 % spread of the depth in "
                   "the window (large at object edges). plane_z (mm): also at_plane_xyz_mm, where the pixel's viewing ray meets the "
                   "horizontal plane z = plane_z (computed from the calibration, not from the depth). A point "
                   "without valid depth has no depth_mm/base_xyz_mm (valid_fraction 0); outside the image it has "
                   "an error. depth_mm is along the camera's optical axis (camera frame: x right, y down, z "
                   "forward). Numbers only, no picture.",
                   scene.depth_at,
                   _obj({"pixels": {"type": "array", "minItems": 1, "maxItems": 50,
                                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
                         "window_px": {"type": "integer", "minimum": 0, "maximum": 10},
                         "plane_z": {"type": "number"}}, ["pixels"]),
                   read_only=True),
    ] if hasattr(scene, "depth_at") else []) + ([
        ActionSpec("locate_object",
                   "Measure a height-connected top face near pixel=[u,v] in FULL env-image pixels, "
                   "using aligned D455 depth from the LAST env look and the fixed camera calibration. "
                   "Returns ok, seed pixel, top_center_mm [x,y,z], support_z_mm, height_mm, footprint_mm "
                   "[long,short], yaw_deg, n_points, valid_fraction, depth_spread_mm, kind and quality warnings. "
                   "Lengths are robot base millimetres; yaw is the long XY edge angle in (-45,45] for aspect "
                   "ratio <=1.15, otherwise (-90,90]. kind is object for height>=10 mm and longest edge<=80 mm, "
                   "otherwise surface. Support is the median surrounding ring height, with taught table Z "
                   "as fallback (warned). Default measures the full image; roi_radius_px optionally bounds "
                   "it. No colour detection or image is returned. Missing depth/calibration gives ok=false, error.",
                   lambda pixel, **kw: scene.locate_object(pixel, table_z=table_z, **kw),
                   _obj({"pixel": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                         "roi_radius_px": {"type": "integer", "minimum": 1},
                         "top_tolerance_mm": {"type": "number", "exclusiveMinimum": 0}}, ["pixel"]),
                   read_only=True, reports_status=True),
    ] if hasattr(scene, "locate_object") else []) + [
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
