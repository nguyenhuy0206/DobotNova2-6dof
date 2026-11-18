# astra-version: 2 — Remove unsupported claim that any descent error proves fingers landed on cube/box; distinguish commanded depth from achieved depth. Motion parameters 
"""Grasp ladder; report descent failures without inferring an unobserved collision cause."""
KIND = "routine"
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
            detail = str(r.get("detail", ""))
            return {"ok": False, "detail": detail + f" — descent toward depth {depth:.0f} mm failed before closure; achieved depth and physical cause unconfirmed. Inspect alarm code and measured pose before retrying.", "tries": tries, "failure_hint": "contact" if "CONTACT" in detail else "step_failed", "error_result": r.get("error_result")}
        cl = ctx.execute("close_gripper", {"force": int(force), "speed": 20})
        if not cl["ok"]:
            return {"ok": False, "detail": cl.get("error", ""), "tries": tries, "error_result": cl}
        res = cl.get("result")
        memo["close_gripper"] = res
        if not isinstance(res, dict):
            res = {}
        pos, holding = res.get("position"), res.get("holding")
        tries.append({"depth": depth, "position": pos, "holding": holding})
        if ctx.dry or (holding and pos is not None and pos < max_position):
            memo.setdefault("vars", {})["grasp_depth"] = depth
            return {"ok": True, "depth": depth, "tries": tries, "detail": f"caught at {depth:.0f} mm below z_top (grip {pos})"}
        if holding and pos is not None and pos >= max_position:
            break
    last = tries[-1] if tries else {}
    edge = last.get("holding") and (last.get("position") or 0) >= max_position
    return {"ok": False, "camera": "wrist", "tries": tries, "detail": ("possible edge grip (position >= 850); inspect alignment before retry" if edge else f"nothing caught at commanded depth {last.get('depth', 0):.0f} mm; alignment and height remain possible causes")}
