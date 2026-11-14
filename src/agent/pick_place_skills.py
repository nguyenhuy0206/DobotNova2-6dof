"""Pick / place skills distilled from the successful episode 20261112-181222-0f8c62.

What worked there (green cube, trial 3):
  hover at z_top + 95 (top of the approach zone), rz chosen so the cube edges are
  square to the fingers (that day: tool offset −30° -> absolute rz 70.9) · descend to
  z_top + 30 · centre with tool-frame dx/dy from the wrist image until the cube's
  bottom edge is at ≈ (355, 140) px · straight down to z_top − 8 · close (grip 776,
  holding) · lift · env camera shows the cube moved off its place · transfer at
  z ≈ table + 123 · descend to table + 13 · open · lift 25 · wrist empty, env sees it
  on the table.

Nothing episode-specific is hard-coded: the block position, top height, rz and the
place spot are parameters Astra fills from its observations; the measured numbers
(depth 8, target pixel, px/mm mapping) are parameter DEFAULTS it may tune in a new
version. They start as candidates: each run is supervised by the operator until
promoted (scripts/run_explore.sh --promote NAME).
"""
from __future__ import annotations

from memory.skills import Skill, SkillLibrary

SOURCE = "20261112-181222-0f8c62"

PICK = Skill(
    name="pick_block_wrist",
    description=("Pick one cube from the box: hover at z_top+95 with the given rz, descend to z_top+30, find + "
                 "centre the cube in the wrist image (local loops), MEASURE its real top by touch (fingers half "
                 "closed), re-centre at measured top+30, open, go down to measured top - depth_mm, close, check "
                 "the grip, lift and check with the env camera that it came up. z_top = the env camera estimate. "
                 "ONE call per grasp attempt. On failure: change rz (cube edges square to the fingers), "
                 "depth_mm, or the centring target, and call again."),
    params={"color": None, "x": None, "y": None, "z_top": None, "rz": None,
            "depth_mm": 8, "hover_mm": 95, "align_mm": 30,
            "target_u": 355, "target_v": 140, "u_per_dy": 5.6, "v_per_dx": -2.6, "center_tol_px": 12,
            "probe_below_mm": 25},
    preconditions=[{"check": "robot_ready"},
                   {"check": "z_top_plausible", "args": {"z_top": "{z_top}"}},
                   {"check": "env_ref", "args": {"color": "{color}", "x": "{x}", "y": "{y}"}}],
    steps=[
        # callable from anywhere: first straight up out of the approach zone (no-op when above)
        {"routine": "lift_to", "args": {"z": "={z_top} + hover_mm", "step_mm": 23, "speed": 2}},
        {"tool": "open_gripper"},
        {"tool": "move_to_pose", "args": {"x": "{x}", "y": "{y}", "z": "={z_top} + hover_mm",
                                          "rx": 180, "ry": 0, "rz": "{rz}", "speed": 3}},
        {"routine": "descend_to", "args": {"z": "={z_top} + align_mm", "step_mm": 25}},
        {"routine": "wrist_search", "args": {"color": "{color}"}},
        {"routine": "wrist_center", "args": {"color": "{color}", "target_u": "{target_u}", "target_v": "{target_v}",
                                             "u_per_dy": "{u_per_dy}", "v_per_dx": "{v_per_dx}",
                                             "tol_px": "{center_tol_px}"}},
        # z_top from the env camera is only an estimate (±1–2 cm): measure the real top by touch
        {"routine": "probe_top", "args": {"z_est": "{z_top}", "below_mm": "{probe_below_mm}"}},
        {"routine": "lift_to", "args": {"z": "={measured_top} + align_mm", "step_mm": 23, "speed": 2}},
        # centre again at the height the wrist-image target was measured at (top + 30)
        {"routine": "wrist_center", "args": {"color": "{color}", "target_u": "{target_u}", "target_v": "{target_v}",
                                             "u_per_dy": "{u_per_dy}", "v_per_dx": "{v_per_dx}",
                                             "tol_px": "{center_tol_px}"}},
        {"tool": "open_gripper"},
        {"routine": "descend_to", "args": {"z": "={measured_top} - depth_mm", "step_mm": 19, "speed": 1}},
        {"tool": "close_gripper", "args": {"force": 30, "speed": 20}},
        {"check": "holding", "args": {"max_position": 850}},
        {"routine": "lift_to", "args": {"z": "={measured_top} + align_mm", "step_mm": 23, "speed": 2}},
    ],
    success_conditions=[{"check": "lifted", "args": {"color": "{color}"}}],
    # after a definite failure: open and go back above the approach zone, ready for the next call
    recovery=[{"tool": "open_gripper"}, {"routine": "lift_to", "args": {"z": "={z_top} + hover_mm"}}],
    failure_conditions={
        "contact": "a finger landed on the cube or box: re-centre (target_v / x,y) or check z_top",
        "z_top_plausible": "z_top disagrees with the taught cube top: use STATE.block_top_z",
        "top_not_found": ("the touch probe found no cube top within probe_below_mm below z_top: fingers passed "
                          "beside it (centring) or z_top is too high — larger probe_below_mm or re-centre"),
        "safety_block": "move refused (reach / table / zone): pick another x,y or rz; the skill lifts first itself",
        "not_centred": "wrist centring did not converge: look wrist, adjust rz or x/y, retry",
        "block_not_in_wrist_view": ("the local search (±30 mm) did not find it: env x,y is off by more — look env "
                                    "again (zoom on the box); shifting x,y blindly does not help"),
        "grasp_missed": "fingers met nothing (≈500) or only an edge (≥850): change rz by ±15–30 or depth_mm",
        "not_lifted": "closed on the cube but it stayed on the box: grip too high / on an edge — deeper or turn",
    },
    tags=["pick", "grasp", "block", "cube", "wrist", "box", SOURCE, "builtin"],
)

PLACE = Skill(
    name="place_block_on_table",
    description=("Place the held cube on the metal table at (x, y): lift to transfer_z, move above (x, y), "
                 "descend to table_z + release_mm, open, lift, then check the wrist is empty and the env camera "
                 "sees the cube near (x, y). Choose a free spot ≥ 60 mm from other cubes."),
    params={"color": None, "x": None, "y": None, "rz": None,
            "transfer_above_table_mm": 123, "release_mm": 13, "lift_after_mm": 25},
    preconditions=[{"check": "robot_ready"}],
    steps=[
        {"routine": "lift_to", "args": {"z": "={table_z} + transfer_above_table_mm", "step_mm": 23, "speed": 2}},
        {"tool": "move_to_pose", "args": {"x": "{x}", "y": "{y}", "z": "={table_z} + transfer_above_table_mm",
                                          "rx": 180, "ry": 0, "rz": "{rz}", "linear": True, "speed": 3}},
        {"routine": "descend_to", "args": {"z": "={table_z} + release_mm", "step_mm": 25, "speed": 2}},
        {"tool": "open_gripper", "args": {"force": 30, "speed": 20}},
        {"routine": "lift_to", "args": {"z": "={table_z} + release_mm + lift_after_mm", "step_mm": 25, "speed": 2}},
    ],
    success_conditions=[{"check": "wrist_clear", "args": {"color": "{color}"}},
                        {"check": "placed", "args": {"color": "{color}", "x": "{x}", "y": "{y}"}}],
    recovery=[{"routine": "lift_to", "args": {"z": "={table_z} + transfer_above_table_mm"}}],
    failure_conditions={
        "contact": "something under the cube before the release height: lift and pick another free spot",
        "still_held": "cube still in the fingers after opening: open_gripper again, look wrist",
        "not_placed": "env sees the cube elsewhere: it dropped/slid — look env and report where",
    },
    tags=["place", "table", "release", "block", "cube", SOURCE, "builtin"],
)

BUILTIN = [PICK, PLACE]


_PROCEDURE = ("steps", "params", "preconditions", "success_conditions", "recovery", "failure_conditions")


def ensure_pick_place_skills(library: SkillLibrary) -> list[str]:
    """Add the distilled skills if missing; upgrade a stored copy that is still the
    built-in one (tag "builtin") when the code changed — as a new candidate version,
    the old one stays in history. A version Astra saved itself is never overwritten."""
    import copy
    changed = []
    for skill in BUILTIN:
        stored = library.get(skill.name)
        if stored is None:
            library.save(copy.deepcopy(skill))
            changed.append(skill.name)
        elif ("builtin" in stored.tags or SOURCE in stored.tags) and any(getattr(stored, k) != getattr(skill, k) for k in _PROCEDURE):
            library.revise(copy.deepcopy(skill))
            changed.append(skill.name)
    return changed
