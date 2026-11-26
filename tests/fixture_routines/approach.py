"""Approach a cube from above (test fixture: the scripted mock world's pick skill; no colour detection)."""
KIND = "routine"

def run(ctx, memo, color, x, y, z_top, rz, hover_mm=95, align_mm=30, near_mm=30, min_area=1500, descend_speed=2):
    z_align = float(z_top) + float(align_mm)
    z_hover = max(float(z_top) + float(hover_mm), 190.0)
    r = ctx.vertical(z_hover, 23, 2)
    if not r["ok"]: return r
    mv = ctx.execute("move_to_pose", {"x": float(x), "y": float(y), "z": z_hover, "rx": 180, "ry": 0, "rz": float(rz), "speed": 3})
    if not mv["ok"]: return {"ok": False, "detail": mv.get("error", ""), "error_result": mv}
    r = ctx.vertical(z_align, 15, descend_speed)
    if not r["ok"]: return r
    return {**r, "skipped": False, "z_hover": z_hover}
