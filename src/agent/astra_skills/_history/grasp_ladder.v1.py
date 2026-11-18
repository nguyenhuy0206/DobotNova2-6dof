# astra-version: 1 — seeded from local_steps (16 Nov)
"""Grasp without pressing: open fingers straddle the cube, close at z_top-first_depth_mm, then deeper per rung."""
KIND = "routine"

# close_gripper position: ≈ 530 face-to-face grip · 500 = fingers met nothing · ≥ 850 = edge only


def run(ctx, memo, z_top, first_depth_mm=6, step_mm=6, rungs=3, max_position=850, force=30, speed=1):
    tries = []
    for k in range(max(1, min(int(rungs), 4))):
        depth = min(float(first_depth_mm) + k * float(step_mm), 20.0)
        if k:
            op = ctx.execute("open_gripper", {})
            if not op["ok"]:
                return {"ok": False, "detail": op.get("error", ""), "tries": tries, "error_result": op}
        r = ctx.vertical(float(z_top) - depth, 19, speed)
        if not r["ok"]:
            return {"ok": False, "detail": f"{r.get('detail', '')} — at depth {depth:.0f} mm: the fingers landed on "
                    "the cube or box instead of around the cube (not centred / rz not square)", "tries": tries,
                    "failure_hint": "contact" if "CONTACT" in str(r.get("detail", "")) else "step_failed",
                    "error_result": r.get("error_result")}
        cl = ctx.execute("close_gripper", {"force": int(force), "speed": 20})
        if not cl["ok"]:
            return {"ok": False, "detail": cl.get("error", ""), "tries": tries, "error_result": cl}
        res = cl.get("result")
        memo["close_gripper"] = res
        if not isinstance(res, dict):                        # dry_run: "DRY RUN: … validated, not sent."
            res = {}
        pos, holding = res.get("position"), res.get("holding")
        tries.append({"depth": depth, "position": pos, "holding": holding})
        if ctx.dry or (holding and pos is not None and pos < max_position):
            memo.setdefault("vars", {})["grasp_depth"] = depth
            return {"ok": True, "depth": depth, "tries": tries, "detail": f"caught at {depth:.0f} mm below z_top (grip {pos})"}
        if holding and pos is not None and pos >= max_position:
            break                                            # edge grip: deeper will not fix it, turn instead
    last = tries[-1] if tries else {}
    edge = last.get("holding") and (last.get("position") or 0) >= max_position
    return {"ok": False, "camera": "wrist", "tries": tries,
            "detail": ("edge contact only (grip ≥ 850): turn rz so the cube faces are square to the fingers" if edge
                       else f"nothing caught down to {last.get('depth', 0):.0f} mm below z_top: the cube is not between "
                       "the fingers (centring target / rz) or z_top is too high")}
