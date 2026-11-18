# astra-version: 2 — Fix v1: z_top+95 could be below Z=162, causing a long lateral approach-zone move; enforce lateral hover at Z>=190.
"""Approach a cube without making a long lateral move below the approach-zone boundary."""
KIND = "routine"

def run(ctx, memo, color, x, y, z_top, rz, hover_mm=95, align_mm=30, near_mm=30, min_area=1500, descend_speed=2):
    z_align = float(z_top) + float(align_mm)
    # Keep the lateral move in free space above the enforced approach-zone boundary.
    z_hover = max(float(z_top) + float(hover_mm), 190.0)
    if not ctx.dry:
        tip = ctx.tip()
        near = math.hypot(tip[0] - float(x), tip[1] - float(y)) <= near_mm
        same_rz = abs(((tip[5] - float(rz)) + 180) % 360 - 180) < 3
        if near and same_rz and z_align - 10 <= tip[2] <= z_hover + 5 and any(o["area_px"] >= min_area for o in ctx.wrist(color, min_area)):
            if tip[2] > z_align + 2:
                r = ctx.vertical(z_align, 15, descend_speed)
                if not r["ok"]: return r
            return {"ok": True, "skipped": True, "z": round(min(tip[2], z_align), 1), "detail": "already above the cube: no re-approach"}
    r = ctx.vertical(z_hover, 23, 2)
    if not r["ok"]: return r
    mv = ctx.execute("move_to_pose", {"x": float(x), "y": float(y), "z": z_hover, "rx": 180, "ry": 0, "rz": float(rz), "speed": 3})
    if not mv["ok"]: return {"ok": False, "detail": mv.get("error", ""), "error_result": mv}
    r = ctx.vertical(z_align, 15, descend_speed)
    if not r["ok"]: return r
    return {**r, "skipped": False, "z_hover": z_hover}
