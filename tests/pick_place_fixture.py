"""Test fixture (was src/agent/explore/pick_place_skills.py, removed 21 Nov): the pick/place skills the
scripted mock world runs. Not part of the robot software any more."""

from __future__ import annotations

from memory.skills import Skill, SkillLibrary

SOURCE = "20261112-181222-0f8c62"

PICK = Skill(
    name="pick_block_wrist",
    description=("Pick one cube from the box: get to z_top+30 above it (no re-approach when already there), find + "
                 "centre it in the wrist image, open, then a grasp ladder: down to z_top-6 and close; nothing caught "
                 "-> open, 6 mm deeper, close (≤ 3 rungs). Check the grip, lift, env camera checks the cube came up. "
                 "z_top = env-camera estimate. One call per attempt."),
    params={"color": None, "x": None, "y": None, "z_top": None, "rz": None,
            "hover_mm": 95, "align_mm": 30, "first_depth_mm": 6, "depth_step_mm": 6, "rungs": 3,
            "target_u": 355, "target_v": 140, "u_per_dy": 5.6, "v_per_dx": -2.6, "center_tol_px": 12},
    preconditions=[{"check": "robot_ready"},
                   {"check": "z_top_plausible", "args": {"z_top": "{z_top}"}},
                   {"check": "env_ref", "args": {"color": "{color}", "x": "{x}", "y": "{y}"}}],
    steps=[
        {"routine": "approach", "args": {"color": "{color}", "x": "{x}", "y": "{y}", "z_top": "{z_top}", "rz": "{rz}",
                                         "hover_mm": "{hover_mm}", "align_mm": "{align_mm}"}},
        {"tool": "open_gripper"},
        {"routine": "wrist_search", "args": {"color": "{color}"}},
        {"routine": "wrist_center", "args": {"color": "{color}", "target_u": "{target_u}", "target_v": "{target_v}",
                                             "u_per_dy": "{u_per_dy}", "v_per_dx": "{v_per_dx}",
                                             "tol_px": "{center_tol_px}"}},
        {"routine": "grasp_ladder", "args": {"z_top": "{z_top}", "first_depth_mm": "{first_depth_mm}",
                                             "step_mm": "{depth_step_mm}", "rungs": "{rungs}"}},
        {"check": "holding", "args": {"max_position": 850}},
        {"routine": "lift_to", "args": {"z": "={z_top} + align_mm", "step_mm": 23, "speed": 2}},
    ],
    success_conditions=[{"check": "lifted", "args": {"color": "{color}"}}],
    # after a definite failure: open and rise to the alignment height only (the next attempt starts there)
    recovery=[{"tool": "open_gripper"}, {"routine": "lift_to", "args": {"z": "={z_top} + align_mm"}}],
    failure_conditions={
        "contact": "the fingers landed on something on the way down",
        "z_top_plausible": "z_top is below the table or far from the taught cube top",
        "safety_block": "a move was refused (reach / table / approach zone)",
        "not_centred": "wrist centring did not converge or wandered too far",
        "block_not_in_wrist_view": "the local search (±30 mm) did not find the cube in the wrist image",
        "grasp_missed": "nothing caught on any rung (≈ 500) or only an edge (≥ 850)",
        "not_lifted": "after the grasp the cube is still on the box, or the gripper is not holding it",
    },
    tags=["pick", "grasp", "block", "cube", "wrist", "box", SOURCE, "builtin"],
)

PLACE = Skill(
    name="place_block_on_table",
    description=("Place the held cube on the metal table at (x, y): lift to transfer_z, move above (x, y), "
                 "descend to table_z + release_mm, open, lift, then check the wrist is empty and the env camera "
                 "sees the cube near (x, y). Choose a free spot ≥ 60 mm from other cubes."),
    params={"color": None, "x": None, "y": None, "rz": None,
            "transfer_above_table_mm": 123, "release_mm": 13, "lift_after_mm": 50},
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
    """Add the distilled skills if missing; upgrade a stored copy when the code changed and
    it is still the built-in one (tag "builtin"), or when the stored version is FAILING
    (≥ 3 runs, no success) — as a new candidate version; the old one stays in history
    (rollback_skill). A version Astra saved itself that works is never overwritten."""
    import copy
    changed = []
    for skill in BUILTIN:
        stored = library.get(skill.name)
        if stored is None:
            library.save(copy.deepcopy(skill))
            changed.append(skill.name)
            continue
        differs = any(getattr(stored, k) != getattr(skill, k) for k in _PROCEDURE)
        builtin = "builtin" in stored.tags or SOURCE in stored.tags
        failing = stored.failures >= 3 and stored.successes == 0
        if differs and (builtin or failing):
            library.revise(copy.deepcopy(skill))
            changed.append(skill.name)
    return changed
