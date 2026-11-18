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
from pathlib import Path

from common.errors import ValidationError
from common.paths import REPO_ROOT

ASTRA_SKILLS_DIR = REPO_ROOT / "src" / "agent" / "astra_skills"
NAME = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
MAX_ACTIONS = 120
# recovery after a failed skill relies on these only ever going UP: they stay core
PROTECTED = {"lift_to"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "input", "breakpoint", "help", "memoryview", "type", "super", "object",
                "classmethod", "staticmethod", "property", "dir", "id", "exit", "quit"}
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "map",
    "max", "min", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ValueError", "RuntimeError", "KeyError", "IndexError", "ZeroDivisionError")}
SKILL_TOOLS = {"get_state", "move_to_pose", "move_to_joints", "move_relative", "jog_joint", "go_to_position",
               "open_gripper", "close_gripper", "set_gripper", "stop", "look", "find_objects", "check_robot_in_view"}


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
    run = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if not run or len(run[0].args.args) < 2 or [a.arg for a in run[0].args.args[:2]] != ["ctx", "memo"]:
        raise ValidationError("define  def run(ctx, memo, ...):  returning a dict with 'ok'")
    return tree


class RoutineContext:
    """The only door from Astra's code to the robot and cameras."""

    def __init__(self, local, name: str):
        self._ls, self._name, self._actions = local, name, 0
        self.table_z = getattr(local.p, "table_z_mm", None) if local.p is not None else None
        self.block_top_z = local.block_top_z

    def _count(self, n: int = 1) -> None:
        self._actions += n
        if self._actions > MAX_ACTIONS:
            raise RuntimeError(f"{self._name}: more than {MAX_ACTIONS} robot actions in one call — stopped")

    @property
    def dry(self) -> bool:
        return self._ls._dry()

    # ---------------------------------------------------------------- motion (via executor)
    def execute(self, tool: str, args: dict | None = None) -> dict:
        """Any skill tool (move_relative, move_to_pose, open/close/set_gripper, look …): the
        same result dict Astra gets. Never raises for a refused move: check r["ok"]."""
        if tool not in SKILL_TOOLS:
            raise ValueError(f"tool '{tool}' is not available to routines (allowed: {sorted(SKILL_TOOLS)})")
        self._count()
        return self._ls.executor.execute(tool, dict(args or {}))

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
        return self._ls._vertical(float(z), float(step_mm), speed, 1 if float(z) > self.tip()[2] else -1)

    def tool_hop(self, dx: float, dy: float, speed=None) -> dict:
        """Sideways tool-frame move split into pieces ≤ 20 mm (approach-zone limit)."""
        self._count()
        return self._ls._tool_hop(float(dx), float(dy), speed)

    # ---------------------------------------------------------------- reading
    def tip(self) -> list:
        """Fingertip pose [x, y, z, rx, ry, rz] (mm / deg)."""
        return list(self._ls._tip())

    def state(self) -> dict:
        s = self._ls.p.get_state()
        return {k: s.get(k) for k in ("pose", "mode", "error", "moving", "gripper")}

    def wrist(self, color: str, min_area: float = 0) -> list:
        """Wrist-camera blobs of a colour: [{label, pixel, bbox[x,y,w,h], area_px, …}] (fresh frame)."""
        return self._ls._wrist(color, min_area)

    def env(self, color: str) -> list:
        """Env-camera objects of a colour, with base_xyz_mm when calibrated."""
        return self._ls._env(color)

    @staticmethod
    def anchor(obj: dict) -> tuple:
        """Bottom-centre of a blob's box (not clipped at the image top)."""
        x, y, w, h = obj["bbox"]
        return x + w / 2, y + h

    # ---------------------------------------------------------------- composition
    def routine(self, name: str, memo: dict, **args) -> dict:
        return self._ls.routine(name, memo, **args)

    def check(self, name: str, memo: dict, **args) -> dict:
        return self._ls.check(name, memo, **args)


class AstraCode:
    """Loads, validates, versions and runs the files in ASTRA_SKILLS_DIR."""

    _default: "AstraCode | None" = None

    def __init__(self, directory: Path = ASTRA_SKILLS_DIR, core=None):
        self.dir = Path(directory)
        self.core = core                     # LocalSteps class, for list/read of core routines
        self.items: dict[str, dict] = {}
        self.errors: dict[str, str] = {}
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
                if fn is not None:
                    return {"name": name, "source": "core (read-only; write_routine with this name replaces it — "
                                                     "use ctx.* instead of self.*)", "kind": kind,
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
                "next": "test it: run_routine (dry_run validates motions) or a skill step {routine|check: name}"}

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
        try:
            out = it["fn"](ctx, memo, **args)
        except TypeError as exc:
            return {"ok": False, "detail": f"{kind} {name} v{it['version']}: bad arguments: {exc}",
                    "code_error": True}
        except Exception as exc:  # noqa: BLE001 — Astra's code failing is data for Astra
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


def _params(fn) -> dict:
    sig = inspect.signature(fn)
    return {p.name: (None if p.default is inspect.Parameter.empty else p.default)
            for p in list(sig.parameters.values())[2:] if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
