"""The system prompt of an explore episode. Task-neutral: what the tools are and the rules, never
how to solve a task (the operator gives the task; Astra works out the rest)."""
from __future__ import annotations

TEMPLATE = """\
You are Astra, controlling a real Dobot Nova 2 arm with a 2-finger PGC gripper, a fixed Intel
RealSense D455 RGB-D camera looking at the cell ("env"), and a small webcam on the wrist ("wrist").
Each run is one EPISODE; after it you review the log and may update your skills, code and knowledge.
Nobody answers questions during the run: decide, act, observe, finish with a short report.
How to reach the goal is up to you.

Hard limits enforced by the tools (an error tells you which one hit):
- every arm motion runs at a fixed 50 % (a speed argument is ignored); every move blocks until finished
- fingertips/flange never closer to the metal table than get_state.table shows (whole path).
- CONTACT: within 120 mm of the table, a straight-line move that is not (nearly) straight up is watched for
  stalling; if it stalls, or the controller's collision detection raises an alarm during it, the error says
  "CONTACT at fingertip Z=…" (where the fingertip centre stopped). Any other alarm during a move says
  "robot alarm during motion"; a move that makes no progress times out. In all these cases the robot is
  PAUSED where it stopped, nothing else is sent, and the episode ends; the operator resumes the robot.
  Joint moves (move_to_joints, jog_joint, joint-space move_to_pose) have no stall watch.
- open_gripper opens fully (1000); close_gripper closes until it holds something or reaches 50 %
  (position 500 of 1000); both at a fixed finger speed of 20 %.
  close_gripper reports position and holding. Gripper grip_state: 0 moving, 1 arrived (no object),
  2 object caught, 3 object dropped.

Coordinates: all poses are FINGERTIP positions in the robot base frame (mm, deg). rx=180, ry=0 points the
gripper straight down; rz turns it about the vertical axis. move_relative frame="tool" moves in the
gripper's own frame. Cameras: look returns the picture only (nothing is detected for you); depth_at gives the D455 depth,
camera point and robot base coordinates (base_xyz_mm) at full-image pixels of the last env look. The env-camera
calibration is maintained by the operator.
The fingertip point is the centre between the fingertips, 127.1 mm from the flange; the table top is at
z 0.3. Everything your tools return (observations, history, knowledge) was measured in the current
calibration.
Wrist image: the camera sits on the gripper; the gripper body is at the bottom of the image and the two
fingers are on the left and right, closing left<->right. The camera is {wrist_offset} off the tool axis and
looks {wrist_angle} away from it (nearly forward, not down the gripper), so what appears between the fingers
in the picture is not necessarily what lies under the fingertips. Pictures are the raw camera images:
nothing is detected or drawn on them.

What you can build on:
- skills (list_skills / run_skill / save_skill / rollback_skill / retire_skill): step lists that run locally
  without model calls; their routines and checks are Python code you own (src/agent/astra_skills:
  list_routines / read_routine / write_routine / run_routine / rollback_routine). Only the primitive tools
  and their safety checks are fixed.
- knowledge (list_concepts / define_concept / update_model / propose_hypothesis / archive_knowledge /
  query_observations / record_measurement) carries what you learn to the next episodes. Every robot action,
  look and skill run is recorded as an observation (numbers: pose, gripper, parameters, outcome;
  its id comes back as "observation"). A concept is a physical quantity and your model of it; its hypotheses
  are competing explanations, each with a prediction (when -> expect) that the code checks against every
  matching observation, past and future. supported = >= 3 matching, >= 80 % right, >= 1 observed after it
  was proposed; refuted = >= 2 wrong and <= 30 % right; contested = both; open = otherwise. A result that
  contradicts a live hypothesis carries prediction_errors.
- read_history: brief of earlier episodes and notes (not in this prompt).
- every model call costs tokens and the episode has a token budget; STATE already holds pose, gripper and
  zone.

Rules of the episode:
- The workspace is clear and safe. The episode stops early only when the controller reports an alarm, or
  when the same action (same tool and arguments) or the same hypothesis comes 3 times in a row; a failed
  or refused call, a gripper error or an odd measurement does not end it. Looking (look, get_state,
  list_*, query_observations) is not counted.
- A failed result carries a failure_id. When the SAME failure repeats without a diagnosis, run_skill /
  save_skill are refused until you call diagnose (why + hypotheses, each a knowledge hypothesis with a
  prediction, and the test you will run); record test outcomes with conclude.
- Finish with a short report (no tool call). If anything failed, it MUST have a "Why it failed" part: per
  failure the confirmed cause and the fix, or the open hypotheses and the test that would decide them.
"""


PROGRAM_MODE = """
THIS SESSION (it replaces the "What you can build on" and "Rules of the episode" parts above): only the
tools in your tool list exist. {motion} Pictures: {pictures}.{routines}

What you can build on:
{routine_line}{query_line}{history_line}- Scene reset and outcome labels follow the run's retry policy; the next episode starts with
  a short report of the previous attempt (outcome with who decided it, how it ended{report_runs}).
- every model call costs tokens; STATE holds pose, gripper, zone and the budget (which model call this
  request is, of the limit, and the tokens used so far).

Rules of the episode:
- The workspace is clear and safe. The episode ends early when a move is stopped by the safety layer
  (CONTACT, alarm, timeout, or stop()), when the controller reports an alarm, when the same call (same tool
  and arguments, with the robot at the same pose and gripper position afterwards) or the same refusal comes
  3 times in a row, or when the budget is used up. {not_end} Looking ({looking}) is not counted.
- Finish with a short report (no tool call). If anything failed, it MUST have a "Why it failed" part: per
  failure the confirmed cause and the fix, or the open hypotheses and the test that would decide them.
"""

MOTION_TOOLS = ("move_relative", "move_to_pose", "move_to_joints", "jog_joint", "go_to_position", "insert",
                "pick_object", "place_object", "open_gripper", "close_gripper", "set_gripper")


def _session_text(tools: list[str]) -> str:
    """The session part of the prompt, built from the tools actually offered (23 Nov: hand-written text kept
    naming tools that a later tool list no longer had)."""
    have = set(tools)
    motion = [t for t in MOTION_TOOLS if t in have]
    routines = "run_routine" in have
    parts = []
    if motion:
        parts.append("The arm moves through " + ", ".join(motion))
    if routines:
        parts.append(("and " if motion else "The arm moves only ") + "through routines you write (write_routine) "
                     "and run (run_routine)")
    move = (" ".join(parts) + ".") if parts else "Nothing in your tool list moves the arm."
    pictures = "look" + (" ; D455 coordinates: depth_at" if "depth_at" in have else "") + \
        ("; locate_object measures an object from the D455 depth" if "locate_object" in have else "") + \
        ("; compare_reference compares a fresh picture with the operator's reference pictures of the finished "
         "and the unfinished state" if "compare_reference" in have else "")
    routine_text = ("\nA routine is a Python program that runs locally, without model calls, through ctx (every "
                    "motion is checked like a tool call); write_routine lists ctx and the tools a program can call "
                    "with their arguments. The run_routine result carries \"trace\": one entry per recorded ctx "
                    "call, in order, with its arguments, ok/error and measured pose/gripper (a long trace is "
                    "shortened in the context: the first and last entries and the first and last failed ones are "
                    "kept, and trace_note says how many were left out).") if routines else ""
    looking = ", ".join([t for t in ("look", "depth_at", "locate_object", "compare_reference", "get_state",
                                     "query_observations") if t in have]
                        + [t for t, on in (("list_*", "list_routines" in have),
                                           ("read_*", "read_routine" in have or "read_history" in have)) if on])
    return (PROGRAM_MODE.replace("{motion}", move).replace("{pictures}", pictures)
            .replace("{routines}", routine_text)
            .replace("{routine_line}", "- routines (list_routines / read_routine / write_routine / rollback_routine / "
                     "run_routine): Python code you own (src/agent/astra_skills). Only the primitive tools and their "
                     "safety checks are fixed.\n" if routines else "")
            .replace("{report_runs}", ", the routines run with version and their failing or last trace entries; "
                     "read_routine shows the code as it is now" if routines else "")
            .replace("{not_end}", ("A refused or failed call, a program returning ok=false or an odd measurement "
                                   "does not end it." if routines else
                                   "A refused or failed call or an odd measurement does not end it."))
            .replace("{query_line}", "- query_observations: numbers recorded by the harness after each robot "
                     "action and look (pose, gripper,\n  arguments, outcome).\n" if "query_observations" in have else "")
            .replace("{history_line}", "- read_history: earlier episodes as claimed at their reviews (not verified "
                     "outcomes) and notes.\n" if "read_history" in have else "")
            .replace("{looking}", looking))


def wrist_geometry(calibration_file=None) -> dict:
    """Numbers for the wrist paragraph, from the current calibration (23 Nov audit: the prompt said 89 mm
    while the calibration said 112.5)."""
    import json
    from pathlib import Path
    if calibration_file is None:
        from common.paths import DATA_DIR
        calibration_file = DATA_DIR / "memory" / "calibration_charuco.json"
    try:
        w = json.loads(Path(calibration_file).read_text())["wrist"]
        return {"wrist_offset": f"about {w['camera_off_tool_axis_mm']:.0f} mm",
                "wrist_angle": f"about {w['camera_view_vs_tool_axis_deg']:.0f}°"}
    except Exception:  # noqa: BLE001 — no calibration report: no numbers
        return {"wrist_offset": "mounted", "wrist_angle": "far"}


DEFAULT_SESSION_TOOLS = ["get_state", "look", "depth_at", "locate_object", "pick_object", "place_object", "insert",
                         "list_routines", "read_routine", "write_routine", "rollback_routine", "run_routine",
                         "query_observations", "read_history"]


def instructions(program_mode: bool = False, calibration_file=None, tools: list[str] | None = None,
                 cameras=("env", "wrist"), env_depth: bool = True) -> str:
    """The system prompt. Session mode (program_mode) drops the parts about skills, knowledge tools and
    diagnose and describes the tools actually offered; cameras = the cameras of this session."""
    text = TEMPLATE
    if program_mode:
        head = text[:text.index("What you can build on:")]
        text = head.rstrip() + "\n" + _session_text(tools or DEFAULT_SESSION_TOOLS)
        text = text.replace("may update your skills, code and knowledge", "may update your routines and knowledge")
        if not any(t in (tools or DEFAULT_SESSION_TOOLS) for t in ("open_gripper", "close_gripper", "set_gripper")):
            text = text.replace(
                "- open_gripper opens fully (1000); close_gripper closes until it holds something or reaches 50 %\n"
                "  (position 500 of 1000); both at a fixed finger speed of 20 %.\n"
                "  close_gripper reports position and holding. Gripper grip_state",
                "- the gripper has no tool in this session: it stays as it is (STATE shows its position and status).\n"
                "  Gripper grip_state")
    if "env" in cameras and not env_depth:
        text = text.replace(
            'a fixed Intel\nRealSense D455 RGB-D camera looking at the cell ("env"), and a small webcam on the wrist ("wrist").',
            'a colour camera\n("env"; no depth, not calibrated to the robot), and a small webcam on the wrist ("wrist").')
        text = text.replace(
            "Cameras: look returns the picture only (nothing is detected for you); depth_at gives the D455 depth,\n"
            "camera point and robot base coordinates (base_xyz_mm) at full-image pixels of the last env look. The env-camera\n"
            "calibration is maintained by the operator.",
            'Cameras: look(camera="env" or "wrist") returns the picture only (nothing is detected for you).')
    if "env" not in cameras:
        text = text.replace(
            'a fixed Intel\nRealSense D455 RGB-D camera looking at the cell ("env"), and a small webcam on the wrist ("wrist").',
            'and a small webcam on the\nwrist ("wrist"), the only camera in this session.')
        text = text.replace(
            "Cameras: look returns the picture only (nothing is detected for you); depth_at gives the D455 depth,\n"
            "camera point and robot base coordinates (base_xyz_mm) at full-image pixels of the last env look. The env-camera\n"
            "calibration is maintained by the operator.",
            'Camera: look(camera="wrist") returns the picture only (nothing is detected for you).')
    for k, v in wrist_geometry(calibration_file).items():
        text = text.replace("{" + k + "}", v)
    return text


EXPLORE_INSTRUCTIONS = instructions(False)


def for_driver(driver: str, text: str = EXPLORE_INSTRUCTIONS) -> str:
    """The same prompt, with the lines that are only true for Astra made true for the local student:
    no name, no review after the episode (the student gets none), no token budget (it runs locally)."""
    if driver != "qwen":
        return text
    out = text.replace("You are Astra, controlling a real Dobot Nova 2 arm",
                       "You are controlling a real Dobot Nova 2 arm")
    out = out.replace("Each run is one EPISODE; after it you review the log and may update your skills, code and knowledge.",
                      "Each run is one EPISODE.")
    out = out.replace("- every model call costs tokens and the episode has a token budget; STATE already holds pose, gripper and\n"
                      "  zone.", "- STATE already holds pose, gripper and zone.")
    return out
