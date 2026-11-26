"""Astra's own code: routines and checks it writes itself (src/agent/astra_skills/*.py).

One file = one routine or check, named after the file:

    \"\"\"What it does (shown in list_routines).\"\"\"
    KIND = "routine"            # or "check"
    DRY_RUN = False             # checks only: True = also evaluated in dry_run
    def run(ctx, memo, color, z_top, step_mm=6):
        ...
        return {"ok": True, "detail": "..."}

A file with the same name as a core routine/check (actions/local_steps.py) replaces
it, so Astra can rewrite every routine. The PRIMITIVES stay fixed: code here only
reaches the robot through ``ctx`` (RoutineContext), whose motions go through the
ActionExecutor — speed cap, table floor, approach zone, contact stop, view guard,
dry_run all apply exactly as for any other call.

The sandbox is a guard rail, not a security boundary: no imports (math, statistics
and re are provided), no file / eval / exec / introspection builtins, no attribute
names starting with "_", at most MAX_ACTIONS robot actions per call.
Every write keeps the previous version in _history/ (rollback_routine).
"""
from __future__ import annotations

import ast
import inspect
import math
import re
import statistics
import time
import traceback
from pathlib import Path

from actions.skill_runner import SKILL_TOOLS

# what ctx.execute may call: the skill tools minus the colour detector (operator, 20 Nov: Astra gets pictures
# and depth_at only; it finds objects itself) and the old view-guard probe
ROUTINE_TOOLS = (SKILL_TOOLS - {"find_objects", "check_robot_in_view"}) | {"locate_object", "pick_object", "place_object", "insert"}
from common.errors import ValidationError
from common.paths import REPO_ROOT

ASTRA_SKILLS_DIR = REPO_ROOT / "src" / "agent" / "astra_skills"
NAME = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
MAX_ACTIONS = 120
MAX_SECONDS = 300         # wall-clock budget of one routine call (program mode runs whole programs)
# recovery after a failed skill relies on these only ever going UP: they stay core
PROTECTED = {"lift_to"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "input", "breakpoint", "help", "memoryview", "type", "super", "object",
                "classmethod", "staticmethod", "property", "dir", "id", "exit", "quit"}
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "map",
    "max", "min", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ValueError", "RuntimeError", "KeyError", "IndexError", "ZeroDivisionError")}


def validate_source(code: str) -> ast.Module:
    """Reject code that leaves the sandbox; returns the parsed module."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValidationError(f"syntax error line {exc.lineno}: {exc.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValidationError(f"line {node.lineno}: no imports (math, statistics, re are already available)")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Lambda)):
            raise ValidationError(f"line {node.lineno}: {type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise ValidationError(f"line {node.lineno}: '{node.id}' is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValidationError(f"line {node.lineno}: private attribute '.{node.attr}' is not allowed")
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "ctx"
                and node.attr not in CTX_API):
            # 23 Nov: programs called ctx.env (removed) and only failed when run — say it when the code is written
            raise ValidationError(f"line {node.lineno}: ctx has no '{node.attr}' (ctx offers: {', '.join(sorted(CTX_API))})")
    run = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if not run or len(run[0].args.args) < 2 or [a.arg for a in run[0].args.args[:2]] != ["ctx", "memo"]:
        raise ValidationError("define  def run(ctx, memo, ...):  returning a dict with 'ok'")
    return tree


class RoutineContext:
    """The only door from Astra's code to the robot and cameras."""

    def __init__(self, local, name: str):
        self._ls, self._name, self._actions = local, name, 0
        self._t0 = time.monotonic()
        # every primitive the program calls, in order (23 Nov: the program's evidence — ASPIRE's execution trace)
        self.trace: list[dict] = []
        self.table_z = getattr(local.p, "table_z_mm", None) if local.p is not None else None
        self.block_top_z = local.block_top_z

    def _count(self, n: int = 1) -> None:
        self._actions += n
        if self._actions > MAX_ACTIONS:
            raise RuntimeError(f"{self._name}: more than {MAX_ACTIONS} robot actions in one call — stopped")
        if time.monotonic() - self._t0 > MAX_SECONDS:
            raise RuntimeError(f"{self._name}: longer than {MAX_SECONDS} s — stopped")

    def _record(self, tool: str, args: dict, r: dict, t0: float) -> None:
        """One trace entry: what was asked, whether it ran, what was measured (compact)."""
        res = r.get("result") if isinstance(r.get("result"), dict) else {}
        entry = {"i": len(self.trace), "tool": tool, "args": {k: v for k, v in args.items() if k != "code"},
                 "ok": bool(r.get("ok")), "s": round(time.monotonic() - t0, 2)}
        if not r.get("ok"):
            entry["error"] = str(r.get("error") or r.get("detail") or "")[:240]
        for k in ("pose", "grip_status", "position", "holding", "z", "moves", "detail"):
            if k in res:
                entry[k] = [round(v, 1) for v in res[k][:3]] if k == "pose" else res[k]
        if tool == "depth_at":
            entry["points"] = [{k: p.get(k) for k in ("pixel", "base_xyz_mm", "at_plane_xyz_mm", "valid_fraction",
                                                       "depth_spread_mm") if k in p} for p in res.get("points", [])]
        self.trace.append(entry)

    @property
    def dry(self) -> bool:
        return self._ls._dry()

    # ---------------------------------------------------------------- motion (via executor)
    def execute(self, tool: str, args: dict | None = None) -> dict:
        """Any skill tool (move_relative, move_to_pose, open/close/set_gripper, look …): the
        same result dict Astra gets. Never raises for a refused move: check r["ok"]."""
        if tool not in ROUTINE_TOOLS:
            raise ValueError(f"tool '{tool}' is not available to routines (allowed: {sorted(ROUTINE_TOOLS)})")
        self._count()
        t0 = time.monotonic()
        r = self._ls.executor.execute(tool, dict(args or {}), internal=True)
        self._record(tool, dict(args or {}), r, t0)
        return r

    def move(self, dx=0.0, dy=0.0, dz=0.0, drz=0.0, frame="base", speed=None) -> dict:
        """move_relative; frame 'base' or 'tool'."""
        a = {"dx": dx, "dy": dy, "dz": dz, "drz": drz, "frame": frame}
        if speed:
            a["speed"] = speed
        return self.execute("move_relative", a)

    def vertical(self, z: float, step_mm: float = 19, speed=None) -> dict:
        """Straight up/down to fingertip z in steps (stops on contact; detours towards the
        base when a lift leaves the reach). {ok, z, moves[, detail, error_result]}"""
        self._count()
        t0 = time.monotonic()
        out = self._ls._vertical(float(z), float(step_mm), speed, 1 if float(z) > self.tip()[2] else -1)
        self._record("vertical", {"z": z, "step_mm": step_mm}, {"ok": out.get("ok"), "result": out,
                                                                 "error": out.get("detail")}, t0)
        return out

    def tool_hop(self, dx: float, dy: float, speed=None) -> dict:
        """Sideways tool-frame move split into pieces ≤ 20 mm (approach-zone limit)."""
        self._count()
        t0 = time.monotonic()
        out = self._ls._tool_hop(float(dx), float(dy), speed)
        self._record("tool_hop", {"dx": dx, "dy": dy}, {"ok": out.get("ok"), "result": out,
                                                         "error": out.get("detail")}, t0)
        return out

    # ---------------------------------------------------------------- reading
    def tip(self) -> list:
        """Fingertip pose [x, y, z, rx, ry, rz] (mm / deg)."""
        return list(self._ls._tip())

    def state(self) -> dict:
        s = self._ls.p.get_state()
        return {k: s.get(k) for k in ("pose", "mode", "error", "moving", "gripper")}

    def depth_at(self, pixels: list, window_px: int = 2, plane_z: float | None = None, look: bool = True) -> list:
        """D455 depth / base_xyz_mm at FULL-image pixels (never zoomed-picture pixels), like the depth_at tool
        (full-image pixels, as always). look=True takes a new env frame first (the scene as it is now, the arm included);
        look=False uses the frame of the last env look/depth_at. Returns the points list."""
        if look:
            self.execute("look", {"camera": "env"})
        args = {"pixels": [[float(u), float(v)] for u, v in pixels], "window_px": int(window_px)}
        if plane_z is not None:
            args["plane_z"] = float(plane_z)
        r = self.execute("depth_at", args)
        if not r.get("ok"):
            raise ValueError(r.get("error") or "depth_at failed")
        return r["result"]["points"]

    @staticmethod
    def anchor(obj: dict) -> tuple:
        """Bottom-centre of a blob's box (not clipped at the image top)."""
        x, y, w, h = obj["bbox"]
        return x + w / 2, y + h

    # ---------------------------------------------------------------- composition
    def routine(self, name: str, memo: dict, **args) -> dict:
        return self._nested("routine", name, memo, args)

    def check(self, name: str, memo: dict, **args) -> dict:
        return self._nested("check", name, memo, args)

    def _nested(self, kind: str, name: str, memo: dict, args: dict) -> dict:
        """A routine/check called from this program: one trace entry holding its own trace (core routines
        have none; their primitives are not listed)."""
        self._count()
        t0 = time.monotonic()
        out = (self._ls.routine if kind == "routine" else self._ls.check)(name, memo, **args)
        entry = {"i": len(self.trace), "tool": f"{kind}:{name}", "args": args, "ok": bool(out.get("ok")),
                 "s": round(time.monotonic() - t0, 2)}
        if out.get("detail"):
            entry["detail"] = str(out["detail"])[:240]
        if isinstance(out.get("trace"), list):
            entry["trace"] = out["trace"]
        self.trace.append(entry)
        return out


# what a program may use on ctx (checked when the code is written)
CTX_API = {n for n in dir(RoutineContext) if not n.startswith("_")} | {"trace", "table_z", "block_top_z"}


class AstraCode:
    """Loads, validates, versions and runs the files in ASTRA_SKILLS_DIR."""

    _default: "AstraCode | None" = None

    def __init__(self, directory: Path = ASTRA_SKILLS_DIR, core=None):
        self.dir = Path(directory)
        self.core = core                     # LocalSteps class, for list/read of core routines
        self.items: dict[str, dict] = {}
        self.errors: dict[str, str] = {}
        self.hide_colour_steps = False       # program mode: core steps that detect colours are not offered
        self.reload()

    @classmethod
    def default(cls) -> "AstraCode":
        if cls._default is None:
            cls._default = cls()
        return cls._default

    # ---------------------------------------------------------------- loading
    def _load(self, name: str, code: str) -> dict:
        tree = validate_source(code)
        ns = {"__builtins__": SAFE_BUILTINS, "math": math, "statistics": statistics, "re": re}
        exec(compile(tree, f"astra_skills/{name}.py", "exec"), ns)  # noqa: S102 — validated sandbox code
        kind = ns.get("KIND", "routine")
        if kind not in ("routine", "check"):
            raise ValidationError("KIND must be 'routine' or 'check'")
        m = re.search(r"^# astra-version: (\d+)", code, re.M)
        return {"name": name, "kind": kind, "fn": ns["run"], "dry_run": bool(ns.get("DRY_RUN", False)),
                "description": (ast.get_docstring(tree) or "").strip().split("\n")[0][:200],
                "version": int(m.group(1)) if m else 1, "params": _params(ns["run"])}

    def reload(self) -> None:
        self.items, self.errors = {}, {}
        if not self.dir.exists():
            return
        for f in sorted(self.dir.glob("*.py")):
            if not NAME.match(f.stem) or f.stem in PROTECTED:
                continue
            try:
                self.items[f.stem] = self._load(f.stem, f.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 — a broken file must not stop the others
                self.errors[f.stem] = str(exc)[:300]

    def get(self, kind: str, name: str) -> dict | None:
        it = self.items.get(name)
        return it if it and it["kind"] == kind else None

    def names(self, kind: str) -> set[str]:
        return {n for n, it in self.items.items() if it["kind"] == kind}

    # ---------------------------------------------------------------- tools
    def list(self) -> dict:
        core = {}
        if self.core is not None:
            for kind in ("routine", "check"):
                for n in sorted(k[len(kind) + 1:] for k in dir(self.core) if k.startswith(kind + "_")):
                    if not (self.hide_colour_steps and f"{kind}:{n}" in _colour_steps()):
                        core[n] = kind
        return {"astra": [{"name": n, "kind": it["kind"], "version": it["version"], "params": it["params"],
                           "description": it["description"], **({"replaces_core": True} if n in core else {})}
                          for n, it in sorted(self.items.items())],
                "core": [f"{k}:{n}" for n, k in core.items() if n not in self.items],
                **({"broken": self.errors} if self.errors else {}),
                "note": "read_routine NAME shows the code; write_routine with a core name replaces it"}

    def read(self, name: str) -> dict:
        f = self.dir / f"{name}.py"
        if f.exists():
            return {"name": name, "source": "astra", "code": f.read_text(encoding="utf-8"),
                    "history": self._versions(name)}
        if self.core is not None:
            for kind in ("routine", "check"):
                fn = getattr(self.core, f"{kind}_{name}", None)
                if fn is not None and self.hide_colour_steps and f"{kind}:{name}" in _colour_steps():
                    fn = None
                if fn is not None:
                    return {"name": name, "source": "core (read-only; write_routine with this name replaces it). "
                                                     "Core code uses internal helpers (self.*) that programs cannot "
                                                     f"call; a program has only ctx: {', '.join(sorted(CTX_API))}",
                            "kind": kind,
                            "code": inspect.getsource(fn)}
        raise ValidationError(f"no routine or check '{name}' ({sorted(self.items)})")

    def write(self, name: str, code: str, note: str = "") -> dict:
        if not NAME.match(name or ""):
            raise ValidationError("name: lowercase letters, digits, _ (3–41 chars)")
        if name in PROTECTED:
            raise ValidationError(f"'{name}' is protected (skill recovery relies on it); use another name")
        code = re.sub(r"^# astra-version: \d+.*\n", "", code, flags=re.M)
        loaded = self._load(name, code)                 # validates + compiles before anything is written
        f = self.dir / f"{name}.py"
        version = 1
        if f.exists():
            version = self.items.get(name, {}).get("version", 1) + 1
            self._archive(name)
        header = f"# astra-version: {version}" + (f" — {note.strip()[:150]}" if note.strip() else "") + "\n"
        self.dir.mkdir(parents=True, exist_ok=True)
        f.write_text(header + code.rstrip() + "\n", encoding="utf-8")
        self.reload()
        return {"saved": name, "kind": loaded["kind"], "version": version, "params": loaded["params"],
                "replaces_core": bool(self.core is not None and hasattr(self.core, f"{loaded['kind']}_{name}")),
                "next": "run_routine runs it (live mode: the robot moves)"}

    def rollback(self, name: str, version: int) -> dict:
        """version N: restore that version as the newest; version 0: remove the file (the core
        routine of that name, if any, runs again)."""
        f = self.dir / f"{name}.py"
        if int(version) == 0:
            if not f.exists():
                raise ValidationError(f"no astra routine '{name}'")
            self._archive(name)
            f.unlink()
            self.reload()
            return {"removed": name, "history": self._versions(name)}
        old = self.dir / "_history" / f"{name}.v{int(version)}.py"
        if not old.exists():
            raise ValidationError(f"no version {version} of '{name}' (have {self._versions(name)})")
        return self.write(name, old.read_text(encoding="utf-8"), note=f"rollback to v{version}")

    def _archive(self, name: str) -> None:
        f = self.dir / f"{name}.py"
        if not f.exists():
            return
        v = self.items.get(name, {}).get("version")
        if v is None:
            m = re.search(r"^# astra-version: (\d+)", f.read_text(encoding="utf-8"), re.M)
            v = int(m.group(1)) if m else 1
        h = self.dir / "_history"
        h.mkdir(parents=True, exist_ok=True)
        (h / f"{name}.v{v}.py").write_text(f.read_text(encoding="utf-8"), encoding="utf-8")

    def _versions(self, name: str) -> list[int]:
        h = self.dir / "_history"
        return sorted(int(p.stem.rsplit(".v", 1)[1]) for p in h.glob(f"{name}.v*.py")) if h.exists() else []

    # ---------------------------------------------------------------- running
    def call(self, local, kind: str, name: str, memo: dict, args: dict) -> dict:
        it = self.items[name]
        ctx = RoutineContext(local, name)
        out = self._run(it, ctx, kind, name, memo, args)
        out["routine"], out["version"], out["trace"] = name, it["version"], ctx.trace
        return out

    def _run(self, it, ctx, kind: str, name: str, memo: dict, args: dict) -> dict:
        try:
            out = it["fn"](ctx, memo, **args)
        except Exception as exc:  # noqa: BLE001 — Astra's code failing is data for Astra
            inside = any(f.filename.startswith("astra_skills/") for f in traceback.extract_tb(exc.__traceback__))
            if isinstance(exc, TypeError) and not inside:   # the call itself: run() did not accept the args
                return {"ok": False, "detail": f"{kind} {name} v{it['version']}: bad arguments: {exc}",
                        "code_error": True}
            tb = exc.__traceback__
            line = None
            while tb is not None:
                if tb.tb_frame.f_code.co_filename.startswith("astra_skills/"):
                    line = tb.tb_lineno
                tb = tb.tb_next
            return {"ok": False, "detail": f"{kind} {name} v{it['version']} raised {type(exc).__name__}: {exc}"
                    + (f" (line {line})" if line else ""), "code_error": True}
        if not isinstance(out, dict) or "ok" not in out:
            return {"ok": False, "detail": f"{kind} {name} must return a dict with 'ok' (got {type(out).__name__})",
                    "code_error": True}
        return out


def _colour_steps() -> set:
    from actions.local_steps import COLOUR_STEPS
    return COLOUR_STEPS


def _params(fn) -> dict:
    sig = inspect.signature(fn)
    out = {p.name: (None if p.default is inspect.Parameter.empty else p.default)
           for p in list(sig.parameters.values())[2:] if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    var = next((p.name for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD), None)
    if var:
        out[f"**{var}"] = "any keyword arguments"      # 23 Nov audit: run(ctx, memo, **p) looked parameterless
    return out
