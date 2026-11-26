"""Astra's own routines: sandbox, versioning, override of core routines, use inside skills."""
import pytest

from actions import ActionExecutor, build_registry
from actions.astra_code import AstraCode, validate_source
from actions.local_steps import LocalSteps
from actions.skill_runner import SkillRunner
from common.errors import ValidationError
from memory.skills import SkillLibrary

GOOD = '''"""Go down 5 mm and report the z."""
KIND = "routine"
def run(ctx, memo, dz=5):
    r = ctx.move(dz=-float(dz), speed=1)
    memo.setdefault("vars", {})["went"] = dz
    return {"ok": r["ok"], "detail": "moved" if r["ok"] else r.get("error", "")}
'''


@pytest.mark.parametrize("bad", [
    "import os\ndef run(ctx, memo): return {'ok': True}",
    "def run(ctx, memo): return open('/etc/passwd')",
    "def run(ctx, memo): return {'ok': ctx._ls is None}",
    "def run(ctx, memo): return eval('1')",
    "def run(c, m): return {'ok': True}",
    "def run(ctx, memo): return ().__class__",
])
def test_sandbox_rejects_escapes(bad):
    with pytest.raises(ValidationError):
        validate_source(bad)


def test_write_version_rollback_and_remove(tmp_path):
    code = AstraCode(tmp_path, core=LocalSteps)
    assert code.write("down_a_bit", GOOD)["version"] == 1
    assert code.write("down_a_bit", GOOD.replace("dz=5", "dz=4"), note="smaller")["version"] == 2
    assert code.items["down_a_bit"]["params"] == {"dz": 4}
    assert code.rollback("down_a_bit", 1)["version"] == 3 and code.items["down_a_bit"]["params"] == {"dz": 5}
    assert code.read("down_a_bit")["history"] == [1, 2]
    code.rollback("down_a_bit", 0)
    assert "down_a_bit" not in code.items
    with pytest.raises(ValidationError, match="protected"):
        code.write("lift_to", GOOD)
    assert "def routine_wrist_center" in code.read("wrist_center")["code"]      # core is readable


def test_override_core_routine_and_use_it_in_a_skill_dry_run(prims, tmp_path):
    code = AstraCode(tmp_path, core=LocalSteps)
    lib = SkillLibrary(tmp_path / "s.json")
    r = SkillRunner(lib, constants={"table_z": 42.0})
    ex = ActionExecutor(build_registry(prims, runner=r), mode="dry_run")
    r.executor = ex
    r.local = LocalSteps(prims, None, ex, settle_s=0, code=code)
    code.write("wrist_center", GOOD.replace('KIND = "routine"', 'KIND = "routine"\n# replaces core'))
    out = r.local.routine("wrist_center", {})
    assert out["ok"] and out["detail"] == "moved"
    r.save("probe_down", "test", [{"routine": "wrist_center", "args": {"dz": 3}}])
    res = r.run("probe_down", {})
    assert res["ok"], res


def test_code_errors_come_back_as_data(prims, tmp_path):
    code = AstraCode(tmp_path, core=LocalSteps)
    code.write("broken_one", 'KIND = "routine"\ndef run(ctx, memo):\n    return {"ok": 1 / 0}\n')
    local = LocalSteps(prims, None, None, settle_s=0, code=code)
    out = local.routine("broken_one", {})
    assert not out["ok"] and out["code_error"] and "ZeroDivisionError" in out["detail"] and "line 4" in out["detail"]   # line in the file read_routine shows


def test_tools_outside_skill_set_are_refused(prims, tmp_path):
    code = AstraCode(tmp_path, core=LocalSteps)
    code.write("sneaky", 'KIND = "routine"\ndef run(ctx, memo):\n    return ctx.execute("save_skill", {})\n')
    out = LocalSteps(prims, None, None, settle_s=0, code=code).routine("sneaky", {})
    assert not out["ok"] and "not available" in out["detail"]


def test_astra_own_code_loads_without_errors():
    """src/agent/astra_skills is Astra's runtime output (not in git since 21 Nov): whatever is there must load."""
    code = AstraCode.default()
    assert not code.errors


def test_lifted_fails_when_the_gripper_dropped_the_cube():
    from types import SimpleNamespace as NS

    class Scene:
        def find(self, color, cam):
            return [NS(summary=lambda: {"label": "blue", "pixel": [507, 344]})]

    prims = NS(get_state=lambda: {"gripper": {"grip_state": 3, "grip_status": "object dropped", "position": 500}})
    local = LocalSteps(prims, Scene(), None, settle_s=0)
    out = local.check("lifted", {"env_ref": [414, 276]}, color="blue")
    assert not out["ok"] and "dropped" in out["detail"]
    prims.get_state = lambda: {"gripper": {"grip_state": 2, "grip_status": "object caught", "position": 661}}
    assert local.check("lifted", {"env_ref": [414, 276]}, color="blue")["ok"]
