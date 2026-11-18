"""Astra's physical knowledge: observations -> concepts -> hypotheses with predictions.

Follows PhysMem (arXiv 2602.20323, github.com/haoyangli16/PhysMem) where it matters:
  * the actor does not grade itself: every robot action, look and skill run is RECORDED BY THE
    HARNESS as an observation (numbers only), and evidence is COMPUTED by matching hypotheses
    against observations — never written by hand;
  * raw experience never goes into the prompt, only the knowledge built on it;
  * verification before trust: a hypothesis is "supported" only after enough matching
    observations, at least one of them made AFTER it was proposed (a prediction, not a fit);
  * prediction errors (a new observation contradicting a hypothesis) are reported at once:
    that is where learning starts.
Different from PhysMem: the unit of knowledge is a PHYSICAL CONCEPT (a quantity and Astra's model
of it), not a behaviour rule, so one simple idea does not split into many sentences:

    concept    C3  "finger-gap position in the wrist image"   depends_on [rz, z]   model "…"
    hypothesis C3.1 mechanism "camera is off the J6 axis …"
                    when   {"routine": "wrist_center", "rz": [75, 90]}      (conditions)
                    expect {"out.target_v": [60, 80]}                       (prediction)
                    -> evidence = every observation matching `when`, judged on `expect`

A broader hypothesis is the same expectation with fewer / wider `when` conditions; it is checked
against all past observations at once, so a counter-example anywhere in its range shows up.
Nothing is deleted: concepts and hypotheses are archived with a reason.

data/memory/observations.jsonl   one observation per line (append only)
data/memory/concepts.json        concepts with their hypotheses
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from common.errors import ValidationError
from common.paths import MEMORY_DIR

OBSERVATIONS_FILE = MEMORY_DIR / "observations.jsonl"
CONCEPTS_FILE = MEMORY_DIR / "concepts.json"

STATE_KEYS = ("x", "y", "z", "rx", "ry", "rz", "grip", "grip_status")
MAX_VARS = 60
MAX_LIVE_HYPOTHESES = 4                  # per concept: competing explanations, not variations of one
SUPPORT_MIN, SUPPORT_RATE, REFUTE_MIN, REFUTE_RATE = 3, 0.8, 2, 0.3
_STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "are", "for", "with", "by", "from",
         "its", "it", "when", "how", "what", "which", "that", "this", "as", "be"}


# --------------------------------------------------------------------------- helpers
def _isnum(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", str(text).lower()) if w not in _STOP and len(w) > 1}


def _overlap(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(len(wa), len(wb)) if wa and wb else 0.0


def holds(spec, value) -> bool:
    """[lo, hi] = numeric range · list = one of · scalar = equal (strings case-insensitive)."""
    if value is None:
        return False
    if isinstance(spec, list):
        if len(spec) == 2 and all(_isnum(s) for s in spec):
            lo, hi = sorted(spec)
            return _isnum(value) and lo <= value <= hi
        return any(holds(s, value) for s in spec)
    if isinstance(spec, bool) or isinstance(value, bool):
        return isinstance(spec, bool) and isinstance(value, bool) and spec is value
    if _isnum(spec) and _isnum(value):
        return abs(spec - value) <= 1e-6
    return str(spec).strip().lower() == str(value).strip().lower()


def judge(h: dict, vars_: dict) -> str | None:
    """supports / contradicts / None (the observation is outside the hypothesis' conditions)."""
    if any(k not in vars_ or not holds(v, vars_[k]) for k, v in h["when"].items()):
        return None
    if any(k not in vars_ for k in h["expect"]):
        return None
    return "supports" if all(holds(v, vars_[k]) for k, v in h["expect"].items()) else "contradicts"


def _spec_ok(d, what: str, need: bool) -> dict:
    if not isinstance(d, dict) or (need and not d):
        raise ValidationError(f"{what}: an object of variable -> value / [lo, hi] / [one, of, these]")
    if len(d) > 8:
        raise ValidationError(f"{what}: at most 8 variables")
    for k, v in d.items():
        if not isinstance(k, str) or not k or len(k) > 40:
            raise ValidationError(f"{what}: variable names are short strings")
        vals = v if isinstance(v, list) else [v]
        if not vals or not all(isinstance(x, (str, int, float, bool)) for x in vals):
            raise ValidationError(f"{what}.{k}: a number, string, bool, [lo, hi] or a list of values")
    return d


def _round(v):
    return round(v, 1) if isinstance(v, float) else v


# --------------------------------------------------------------------------- observations
def observe_call(name: str, args: dict, result: dict, state: dict | None = None) -> dict:
    """Flatten one tool call into variables. Plain names: the robot state after the call
    (x y z rx ry rz grip grip_status) and the call's parameters; out.* = numbers in the result;
    <camera>.<colour>.* = detections (u v pixel, angle, area, x y z base mm, seen)."""
    v: dict = {"action": name}
    params = dict(args or {})
    if name in ("run_skill", "run_routine"):
        v["skill" if name == "run_skill" else "routine"] = params.pop("name", None)
        params.update(params.pop("params", None) or params.pop("args", None) or {})
    if isinstance(params.get("colors"), list) and len(params["colors"]) == 1:
        params["color"] = params["colors"][0]
    for k, val in params.items():
        if isinstance(val, (str, int, float, bool)) and not (isinstance(val, str) and len(val) > 40):
            v[k if k not in STATE_KEYS else f"arg.{k}"] = _round(val)
    st = state or {}
    pose = st.get("pose")
    if isinstance(pose, (list, tuple)) and len(pose) >= 6:
        v.update({k: round(float(p), 1) for k, p in zip(("x", "y", "z", "rx", "ry", "rz"), pose)})
    g = st.get("gripper") or {}
    if g.get("position") is not None:
        v["grip"] = g["position"]
    if g.get("grip_status"):
        v["grip_status"] = g["grip_status"]

    inner = result.get("result") if isinstance(result, dict) else None
    ok = bool(result.get("ok")) if isinstance(result, dict) else False
    text = " ".join(str(x) for x in (result.get("error", ""), (inner or {}).get("detail", "")
                                     if isinstance(inner, dict) else "") if x)
    if isinstance(inner, dict):
        if inner.get("ok") is False:
            ok = False
        if name == "close_gripper" and inner.get("holding") is False:
            ok = False                                    # a close that caught nothing is a failed grasp
            v["failure"] = "grasp_missed"
        if inner.get("failure"):
            v["failure"] = inner["failure"]
        _flatten(inner, "out", v)
        for line in inner.get("trace") or []:
            for key in ("measured_top", "depth", "offset", "error_px", "grip", "iterations"):
                m = re.search(rf"\b{key}=(-?[\d.]+)\b", str(line))
                if m:
                    v[f"out.{key}"] = _round(float(m.group(1)))
        cam = inner.get("camera") or params.get("camera")
        if isinstance(inner.get("objects"), list) and cam:
            tip = (v["x"], v["y"]) if "x" in v and "y" in v else None
            _detections(cam, inner["objects"], params, v, tip)
    m = re.search(r"CONTACT at fingertip Z=(-?[\d.]+)", text)
    if m:
        v["contact_z"] = float(m.group(1))
        v.setdefault("failure", "contact")
    if not ok:
        v.setdefault("failure", "error" if not isinstance(inner, dict) else "failed")
    v["ok"] = ok
    return dict(list(v.items())[:MAX_VARS])


def _flatten(d: dict, prefix: str, out: dict, depth: int = 0) -> None:
    for k, val in d.items():
        if k in ("trace", "objects", "image", "image_file", "image_size", "camera", "detail", "advice", "failure", "ok"):
            continue
        key = f"{prefix}.{k}"
        if isinstance(val, bool) or _isnum(val):
            out[key] = _round(val)
        elif isinstance(val, str) and len(val) <= 30:
            out[key] = val
        elif isinstance(val, (list, tuple)) and 2 <= len(val) <= 3 and all(_isnum(x) for x in val):
            for axis, x in zip("xyz" if "xyz" in k or "pixel" not in k else "uv", val):
                out[f"{key}.{axis}"] = _round(float(x))
        elif isinstance(val, dict) and depth < 1:
            _flatten(val, key, out, depth + 1)


def _detections(cam: str, objects: list, params: dict, out: dict, tip: tuple | None = None) -> None:
    """One blob per colour: of those with base coordinates, the one nearest the fingertip in x/y
    (the object being worked on, not a same-coloured thing elsewhere in the image); without
    coordinates (wrist camera), the largest. <camera>.<colour>.n = how many blobs were seen."""
    groups: dict = {}
    for o in objects:
        if o.get("label"):
            groups.setdefault(o["label"], []).append(o)
    best: dict = {}
    for lab, blobs in groups.items():
        placed = [o for o in blobs if isinstance(o.get("base_xyz_mm"), (list, tuple))]
        if placed and tip is not None:
            best[lab] = min(placed, key=lambda o: (o["base_xyz_mm"][0] - tip[0]) ** 2 + (o["base_xyz_mm"][1] - tip[1]) ** 2)
        else:
            best[lab] = max(placed or blobs, key=lambda o: o.get("area_px") or 0)
        out[f"{cam}.{lab}.n"] = len(blobs)
    asked = params.get("colors") or ([params["color"]] if params.get("color") else [])
    for lab in asked:
        if isinstance(lab, str) and lab not in best:
            out[f"{cam}.{lab}.seen"] = False
    for lab, o in best.items():
        p = f"{cam}.{lab}"
        out[f"{p}.seen"] = True
        if o.get("cut_by_edge"):
            out[f"{p}.cut_by_edge"] = True
        if isinstance(o.get("pixel"), (list, tuple)):
            out[f"{p}.u"], out[f"{p}.v"] = o["pixel"][:2]
        for key, name in (("angle_deg", "angle"), ("area_px", "area")):
            if o.get(key) is not None:
                out[f"{p}.{name}"] = _round(float(o[key]))
        if isinstance(o.get("base_xyz_mm"), (list, tuple)):
            for axis, x in zip("xyz", o["base_xyz_mm"]):
                out[f"{p}.{axis}"] = _round(float(x))


class Observations:
    def __init__(self, path: Path = OBSERVATIONS_FILE):
        self.path = path
        self.items: list[dict] = []
        if path.exists():
            self.items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def add(self, vars_: dict, episode: str | None, source: str = "harness", note: str = "",
            t: str | None = None) -> dict:
        t = t or time.strftime("%Y-%m-%d %H:%M:%S")
        vars_ = {**vars_, "day": vars_.get("day") or t[:10]}      # the setup changes from day to day
        obs = {"id": f"O{len(self.items) + 1}", "n": len(self.items) + 1, "t": t,
               "episode": episode, "source": source, "vars": vars_}
        if note:
            obs["note"] = str(note).strip()[:200]
        self.items.append(obs)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obs, ensure_ascii=False) + "\n")
        return obs

    def variables(self, limit: int = 120) -> list[str]:
        seen: dict[str, int] = {}
        for o in self.items[-2000:]:
            for k in o["vars"]:
                seen[k] = seen.get(k, 0) + 1
        return sorted(seen, key=lambda k: -seen[k])[:limit]


# --------------------------------------------------------------------------- knowledge
class KnowledgeStore:
    prompt_header = ("Your physical concepts (a quantity, your model of it, and competing hypotheses; a hypothesis' "
                     "status is computed from the recorded observations that match its conditions):")

    def __init__(self, path: Path = CONCEPTS_FILE, observations: Observations | None = None):
        self.path = path
        self.obs = observations if observations is not None else Observations(path.with_name("observations.jsonl"))
        self.episode: str | None = None          # set by the harness
        self.focus_episode: str | None = None    # the episode under review (this_episode means that one)
        self.state_fn = None                     # () -> {"pose": [...], "gripper": {...}} for measurements
        self.records = None                      # (tool name) -> bool: which harness calls become observations
        self.concepts: list[dict] = []
        if path.exists():
            self.concepts = json.loads(path.read_text(encoding="utf-8")).get("concepts", [])

    # ------------------------------------------------------------ storage
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"concepts": self.concepts}, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")

    def _concept(self, cid: str) -> dict:
        c = next((x for x in self.concepts if x["id"] == cid), None)
        if c is None:
            raise ValidationError(f"no concept {cid!r} (known: {[x['id'] for x in self.concepts]})")
        return c

    def _hyp(self, hid: str) -> tuple[dict, dict]:
        c = self._concept(str(hid).split(".")[0])
        h = next((x for x in c["hypotheses"] if x["id"] == hid), None)
        if h is None:
            raise ValidationError(f"no hypothesis {hid!r} in {c['id']}")
        return c, h

    def _live(self, c: dict) -> list[dict]:
        return [h for h in c["hypotheses"] if not h.get("archived") and self.status(h) != "refuted"]

    # ------------------------------------------------------------ evidence (computed, never written)
    def evidence(self, h: dict) -> dict:
        sup, con, new_sup = [], [], 0
        for o in self.obs.items:
            r = judge(h, o["vars"])
            if r == "supports":
                sup.append(o)
                new_sup += o["n"] > h["created_after"]
            elif r == "contradicts":
                con.append(o)
        return {"supports": sup, "contradicts": con, "predicted": new_sup}

    def status(self, h: dict, ev: dict | None = None) -> str:
        if h.get("archived"):
            return "archived"
        ev = ev or self.evidence(h)
        s, c = len(ev["supports"]), len(ev["contradicts"])
        n = s + c
        if c >= REFUTE_MIN and s / n <= REFUTE_RATE:
            return "refuted"
        if s >= SUPPORT_MIN and s / n >= SUPPORT_RATE and ev["predicted"] >= 1:
            return "supported"
        if s and c:
            return "contested"
        return "open"

    def _tested(self, h: dict, sup: list[dict]) -> str:
        parts = []
        for k, spec in h["when"].items():
            vals = [o["vars"][k] for o in sup if _isnum(o["vars"].get(k))]
            if isinstance(spec, list) and vals:
                parts.append(f"{k} {min(vals):g}–{max(vals):g}")
        colors = sorted({str(o["vars"]["color"]) for o in sup if o["vars"].get("color") is not None})
        if colors and "color" not in h["when"]:
            parts.append("color " + ",".join(colors))
        eps = len({o.get("episode") for o in sup})
        if sup:
            parts.append(f"{eps} episode{'s' if eps != 1 else ''}")
        return " · ".join(parts)

    def _hview(self, h: dict, ids: int = 5) -> dict:
        ev = self.evidence(h)
        out = {"id": h["id"], "status": self.status(h, ev), "mechanism": h["mechanism"], "when": h["when"],
               "expect": h["expect"], "supports": len(ev["supports"]), "contradicts": len(ev["contradicts"]),
               "predicted": ev["predicted"], "tested": self._tested(h, ev["supports"])}
        if ids:
            out["supporting_obs"] = [o["id"] for o in ev["supports"][-ids:]]
            out["contradicting"] = [{"obs": o["id"], **{k: o["vars"].get(k) for k in h["expect"]}}
                                    for o in ev["contradicts"][-ids:]]
        if h.get("archived"):
            out["archived"] = h["archived"]
        return out

    def _cview(self, c: dict, ids: int = 5) -> dict:
        out = {k: c[k] for k in ("id", "quantity", "depends_on", "model", "updated")}
        out["hypotheses"] = [self._hview(h, ids) for h in c["hypotheses"]]
        if c.get("archived"):
            out["archived"] = c["archived"]
        return out

    # ------------------------------------------------------------ tools
    def define_concept(self, quantity: str, depends_on: list | None = None, model: str = "") -> dict:
        quantity = str(quantity).strip()
        if len(quantity) < 8:
            raise ValidationError("quantity: the physical quantity or relation, e.g. 'finger-gap position in the "
                                  "wrist image'")
        for c in self.concepts:
            if not c.get("archived") and _overlap(quantity, c["quantity"]) >= 0.6:
                raise ValidationError(f"{c['id']} already covers this: '{c['quantity']}'. Use it (update_model / "
                                      "propose_hypothesis), or name the quantity so the difference is explicit.")
        now = time.strftime("%Y-%m-%d %H:%M")
        c = {"id": f"C{len(self.concepts) + 1}", "quantity": quantity[:160],
             "depends_on": [str(x)[:40] for x in (depends_on or [])][:8], "model": str(model).strip()[:400],
             "hypotheses": [], "created": now, "updated": now}
        self.concepts.append(c)
        self._save()
        return self._cview(c)

    def update_model(self, concept_id: str, model: str, depends_on: list | None = None) -> dict:
        c = self._concept(concept_id)
        if c.get("archived"):
            raise ValidationError(f"{concept_id} is archived")
        if len(str(model).strip()) < 5:
            raise ValidationError("model: your current description of the quantity (numbers, formula, ranges)")
        c["model"] = str(model).strip()[:400]
        if depends_on is not None:
            c["depends_on"] = [str(x)[:40] for x in depends_on][:8]
        c["updated"] = time.strftime("%Y-%m-%d %H:%M")
        self._save()
        return self._cview(c, ids=0)

    def propose(self, concept_id: str, mechanism: str, when: dict, expect: dict) -> dict:
        c = self._concept(concept_id)
        if c.get("archived"):
            raise ValidationError(f"{concept_id} is archived")
        mechanism = str(mechanism).strip()
        if len(mechanism) < 10:
            raise ValidationError("mechanism: why it happens, in one sentence")
        when, expect = _spec_ok(when, "when", True), _spec_ok(expect, "expect", True)
        known = set(self.obs.variables(100_000))
        unseen = sorted(k for k in {**when, **expect} if k not in known)
        if unseen:
            import difflib
            close = {k: difflib.get_close_matches(k, known, n=3, cutoff=0.5) for k in unseen}
            raise ValidationError(
                "a prediction can only be checked on variables that observations carry; never observed: "
                + "; ".join(f"{k} (closest: {', '.join(v) or 'none'})" for k, v in close.items())
                + ". Use names from query_observations, or record the quantity first with record_measurement.")
        for h in c["hypotheses"]:
            if h.get("archived"):
                continue
            if (h["when"], h["expect"]) == (when, expect) or _overlap(mechanism, h["mechanism"]) >= 0.72:
                return {**self._hview(h), "existing": "same prediction or mechanism as this hypothesis"}
        live = self._live(c)
        if len(live) >= MAX_LIVE_HYPOTHESES:
            raise ValidationError(f"{concept_id} already has {len(live)} live hypotheses "
                                  f"({[h['id'] for h in live]}); archive one before adding another")
        n = 1 + max([int(h["id"].split(".")[1]) for h in c["hypotheses"]] or [0])
        h = {"id": f"{c['id']}.{n}", "mechanism": mechanism[:300], "when": when, "expect": expect,
             "created": time.strftime("%Y-%m-%d %H:%M"), "created_after": len(self.obs.items),
             "episode": self.episode}
        c["hypotheses"].append(h)
        c["updated"] = h["created"]
        self._save()
        return self._hview(h)

    def archive(self, knowledge_id: str, reason: str) -> dict:
        if len(str(reason).strip()) < 5:
            raise ValidationError("reason: why it no longer applies")
        rec = {"t": time.strftime("%Y-%m-%d %H:%M"), "reason": str(reason).strip()[:200]}
        if "." in str(knowledge_id):
            _, h = self._hyp(knowledge_id)
            h["archived"] = rec
            self._save()
            return self._hview(h, ids=0)
        c = self._concept(knowledge_id)
        c["archived"] = rec
        self._save()
        return self._cview(c, ids=0)

    def list(self, concept_id: str | None = None, include_archived: bool = False) -> dict:
        if concept_id:
            return self._cview(self._concept(concept_id), ids=8)
        rows = [self._cview(c, ids=0) for c in self.concepts if include_archived or not c.get("archived")]
        return {"concepts": rows, "total": len(rows)}

    def record_measurement(self, values: dict, note: str = "") -> dict:
        """Astra's own measurement (e.g. read from an image) as an observation, with the robot state."""
        _spec_ok(values, "values", True)
        if any(isinstance(v, list) for v in values.values()):
            raise ValidationError("values: single measured values, not ranges")
        vars_ = {"action": "measurement"}
        try:
            st = self.state_fn() if self.state_fn else None
        except Exception:  # noqa: BLE001 — a measurement without the pose is still a measurement
            st = None
        vars_.update({k: v for k, v in observe_call("measurement", {}, {"ok": True}, st).items()
                      if k in STATE_KEYS})
        vars_.update({str(k): _round(v) for k, v in values.items()})
        vars_["ok"] = True
        obs = self.obs.add(vars_, self.episode, source="astra", note=note)
        return self._react(obs)

    def query(self, where: dict | None = None, fields: list | None = None, limit: int = 20,
              this_episode: bool = False) -> dict:
        where = _spec_ok(where or {}, "where", False)
        ep = self.focus_episode or self.episode
        rows = [o for o in self.obs.items if (not this_episode or o.get("episode") == ep)
                and all(k in o["vars"] and holds(v, o["vars"][k]) for k, v in where.items())]
        keep = list(fields or []) or None
        out = []
        for o in rows[-max(1, min(int(limit), 100)):]:
            vs = o["vars"] if keep is None else {k: o["vars"].get(k) for k in ["action", *keep]}
            out.append({"id": o["id"], "episode": o.get("episode"), **({"note": o["note"]} if o.get("note") else {}),
                        **vs})
        return {"total": len(rows), "rows": out, "variables": self.obs.variables(80) if not rows or keep is None
                else None}

    # ------------------------------------------------------------ harness hook
    def observe(self, name: str, args: dict, result: dict) -> dict | None:
        if self.records is None or not self.records(name):
            return None
        try:
            st = self.state_fn() if self.state_fn else None
        except Exception:  # noqa: BLE001
            st = None
        obs = self.obs.add(observe_call(name, args, result, st), self.episode)
        return self._react(obs)

    def _react(self, obs: dict) -> dict:
        """Prediction errors: live hypotheses (proposed before this observation) it contradicts."""
        errors = []
        for c in self.concepts:
            if c.get("archived"):
                continue
            for h in self._live(c):
                if judge(h, obs["vars"]) == "contradicts":
                    errors.append({"hypothesis": h["id"], "expected": h["expect"],
                                   "observed": {k: obs["vars"].get(k) for k in h["expect"]}})
        out = {"id": obs["id"]}
        if errors:
            out["prediction_errors"] = errors[:3]
        return out

    # ------------------------------------------------------------ views for the model
    def digest(self, max_chars: int = 2400) -> str:
        """One block per concept (newest activity first): model, then its hypotheses with
        status, counts and the range they were tested over. No raw observations."""
        blocks = []
        for c in self.concepts:
            if c.get("archived"):
                continue
            lines = [f"- {c['id']} {c['quantity']}" + (f" (depends on {', '.join(c['depends_on'])})"
                                                       if c["depends_on"] else "")
                     + (f" · model: {c['model'][:200]}" if c["model"] else " · no model yet")]
            ruled, last = [], c["updated"]
            for h in c["hypotheses"]:
                if h.get("archived"):
                    continue
                v = self._hview(h, ids=0)
                if v["status"] == "refuted":
                    ruled.append(f"{h['id']} ({h['mechanism'][:60]})")
                    continue
                cond = ", ".join(f"{k} {json.dumps(x)}" for k, x in h["when"].items())
                exp = ", ".join(f"{k} {json.dumps(x)}" for k, x in h["expect"].items())
                lines.append(f"    {h['id']} [{v['status']} {v['supports']}✓ {v['contradicts']}✗, "
                             f"{v['predicted']} predicted] {h['mechanism'][:140]} · when {cond} → {exp}"
                             + (f" · tested: {v['tested']}" if v["tested"] else ""))
                last = max(last, h["created"])
            if ruled:
                lines.append("    ruled out: " + "; ".join(ruled[-3:]))
            blocks.append((last, "\n".join(lines)))
        blocks.sort(key=lambda b: b[0], reverse=True)
        out, used = [], 0
        for _, b in blocks:
            if used + len(b) + 1 > max_chars:
                out.append(f"- … {len(blocks) - len(out)} more concepts (list_concepts)")
                break
            out.append(b)
            used += len(b) + 1
        return "\n".join(out)

    def recent_text(self, limit: int = 30) -> str:
        return self.digest()

    def episode_summary(self, episode: str) -> str:
        """For the review: what this episode's observations did to the knowledge. Prediction
        errors and failures that no hypothesis covers are where learning is needed."""
        mine = [o for o in self.obs.items if o.get("episode") == episode]
        if not mine:
            return "No observations were recorded in this episode."
        live = [h for c in self.concepts if not c.get("archived") for h in c["hypotheses"]
                if not h.get("archived")]
        errors, confirmed, uncovered = [], {}, {}
        for o in mine:
            covered = False
            for h in live:
                if o["n"] <= h["created_after"]:
                    continue
                r = judge(h, o["vars"])
                covered |= r is not None
                if r == "contradicts":
                    errors.append(f"{o['id']} contradicts {h['id']}: expected {json.dumps(h['expect'])}, observed "
                                  + json.dumps({k: o["vars"].get(k) for k in h["expect"]}))
                elif r == "supports":
                    confirmed[h["id"]] = confirmed.get(h["id"], 0) + 1
            if not o["vars"].get("ok", True) and not covered:
                key = " ".join(str(o["vars"].get(k)) for k in ("action", "skill", "routine", "failure")
                               if o["vars"].get(k) is not None)
                uncovered[key] = uncovered.get(key, 0) + 1
        lines = [f"{len(mine)} observations recorded (query_observations this_episode=true)."]
        if confirmed:
            lines.append("Predictions confirmed: " + ", ".join(f"{k} ×{n}" for k, n in confirmed.items()))
        lines.append("Prediction errors: " + ("; ".join(errors[:8]) if errors else "none"))
        lines.append("Failures no hypothesis covers: " + (", ".join(f"{k} ×{n}" for k, n in uncovered.items())
                                                          if uncovered else "none"))
        return "\n".join(lines)
