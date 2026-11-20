#!/usr/bin/env python3
"""Astra explores the cell and learns to pick the blocks (config/explore.yaml).

    scripts/run_explore.sh                     run the task from explore.yaml (live, ≤10 % speed)
    scripts/run_explore.sh --mode confirm      ask y/N before every motion
    scripts/run_explore.sh "custom task text"
    scripts/run_explore.sh --episodes 5        5 attempts; after each Astra reviews it and updates skills/notes
    scripts/run_explore.sh --review 2          no robot: Astra reviews the last 2 episodes (outcome, skills, notes)
    scripts/run_explore.sh --review-all        no robot: one review across ALL grasp episodes -> concrete skills
    scripts/run_explore.sh --variant V0        harness variant for this run (V0 baseline … V3 default)
    scripts/run_explore.sh --driver qwen       the local student model drives (src/student/), no review
    scripts/run_explore.sh --promote NAME      operator: candidate skill -> validated (needs supervised successes)
    scripts/run_explore.sh --rollback NAME 2   operator: restore version 2 of a skill as the newest version
    scripts/run_explore.sh --skills            list skills with version, status and evidence

Needs scripts/run_vision.sh running (cameras + depth) and the robot enabled.
Ctrl+C stops Astra AND the robot. Each run is logged in data/episodes/.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from agent.explore import build_explore
from common.log import setup_logging


def ask_operator(name: str, args: dict) -> bool:
    return input(f"\n>> Astra wants {name}({args}). Execute? [y/N] ").strip().lower() == "y"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", help="override the task in config/explore.yaml")
    ap.add_argument("--mode", choices=["dry_run", "confirm", "live"])
    ap.add_argument("--yes", action="store_true", help="don't wait for Enter before starting")
    ap.add_argument("--episodes", type=int, help="attempts in a row (default: explore.yaml episodes)")
    ap.add_argument("--no-reflect", action="store_true", help="skip the self-review after each episode")
    ap.add_argument("--review-all", action="store_true", help="review all grasp episodes together (robot not used)")
    ap.add_argument("--review", metavar="N|ID", help="only review the last N episodes, or one episode id (robot not used)")
    ap.add_argument("--variant", choices=["V0", "V1", "V2", "V3"], help="harness variant (default: explore.yaml)")
    ap.add_argument("--driver", choices=["astra", "qwen"], help="who drives the robot (default: explore.yaml driver)")
    ap.add_argument("--promote", metavar="NAME", help="operator: mark a candidate skill validated")
    ap.add_argument("--rollback", nargs=2, metavar=("NAME", "VERSION"), help="operator: restore a skill version")
    ap.add_argument("--skills", action="store_true", help="list skills and exit")
    args = ap.parse_args()
    setup_logging(filename="explore.log")

    if args.promote or args.rollback or args.skills:
        skill_admin(args)
        return
    s = build_explore(confirm=ask_operator, mode=args.mode, variant=args.variant, driver=args.driver)
    if s.cfg.driver == "qwen":
        args.no_reflect = True          # the self-review writes knowledge and skills: not the student's job
        print(f"Driver: {s.harness.client.endpoint} (no self-review after the episode)")
    p, a = s.kit.primitives, s.harness.config
    if args.review_all:
        from agent.explore.learning import review_all
        out = review_all(s)
        print(f"Reviewed {len(out['episodes'])} episodes with {out['images']} wrist images\n\n{out['review']}")
        print_skills(s)
        return
    if args.review:
        review(s, args.review)
        return
    o = s.harness.options
    print(f"Astra explore · {a.model} · mode {a.execution_mode} · max speed {a.max_speed}% · "
          f"max {a.max_steps} model calls · harness {o.variant} (context {o.context.mode}, chunks "
          f"{'L≤%d' % o.chunk.max_horizon if o.chunk.enabled else 'off'}, images {o.images.mode}, "
          f"budget {o.max_episode_tokens or '∞'} tokens)")
    print(f"  active tool offset {[round(x, 1) for x in p.active_tool_offset]} mm ({p.tool_offset_source})")
    print(f"  table z {p.table_z_mm} mm · grip closes to {p.grip_close_position}/1000 · "
          f"view guard {'ON' if s.view_guard else 'off'} · env calibrated: {s.kit.scene.env_calibration is not None}")
    print(f"  tools: {', '.join(t.name for t in s.executor.list())}")

    problems = s.preflight()
    if problems:
        print("\nNot ready:")
        for msg in problems:
            print(f"  ✗ {msg}")
        raise SystemExit(1)
    print(f"  collision detection: {s.arm_collision_detection()}")

    task = args.task or s.cfg.task
    episodes = args.episodes or s.cfg.episodes
    print(f"\nTask: {task.strip()}\nEpisodes: {episodes} (self-review after each: {not args.no_reflect})")
    if not args.yes:
        input("\nRobot will move by itself. Hand on the E-stop. Enter = start, Ctrl+C = cancel ")
    try:
        for i in range(1, episodes + 1):
            print(f"\n========== episode {i}/{episodes}")
            out = s.run_episode(task, reflect=not args.no_reflect)
            print(f"\n[report] {out['report']}")
            if out.get("diagnosis"):
                print(f"[diagnosis] {out['diagnosis']}")
            if "reflection" in out:
                o = out.get("outcome") or {}
                print(f"[outcome] {'SUCCESS' if o.get('success') else 'FAIL'} · moved {o.get('blocks_moved', '?')} · "
                      f"{o.get('summary', '')}\n[review] {out['reflection']}")
            print_metrics("episode", out.get("metrics"))
            print_metrics("review", out.get("reflection_metrics"))
            print(f"[episode] data/episodes/{out['episode']}.jsonl")
            if i < episodes:
                ans = input("\nReset the scene (blocks back on the box), then Enter = next episode, q = stop ")
                if ans.strip().lower() == "q":
                    break
        print_skills(s)
    except Exception as exc:  # noqa: BLE001
        from agent.astra_client import QuotaExhausted
        if not isinstance(exc, QuotaExhausted):
            raise
        print(f"\n!! {exc}")
        try:
            p.stop()
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\n!! Ctrl+C — stopping the robot")
        try:
            p.stop()
            print("Robot stopped (Pause). Check it in the Debug tab before continuing.")
        except Exception as exc:  # noqa: BLE001
            print(f"!! STOP FAILED: {exc} — use the E-stop NOW")
        raise SystemExit(130)


def print_metrics(label: str, m: dict | None) -> None:
    if not m:
        return
    print(f"[{label} metrics] {m['model_calls']} model calls · {m['total_tokens']:,} tokens "
          f"(input {m['input_tokens']:,}, cached {m['cached_tokens']:,}, output {m['output_tokens']:,}) · "
          f"images {m['images_sent']} sent / {m['images_withheld']} withheld · {m['actions_executed']} actions "
          f"({m['actions_per_call']}/call, {m['actions_failed']} failed, {m['actions_deferred']} deferred) · "
          f"grasps {m['grasp_attempts']} · safety stops {m['safety_stops']} · {m['execution_s']} s"
          + ("" if m.get("usage_complete") else " · (some calls had no usage data)"))


def print_skills(s) -> None:
    _print_skills(s.harness.skills.all())


def _print_skills(skills) -> None:
    print("\nSkills now:" if skills else "\nNo skills saved yet.")
    for sk in skills:
        sup = sum(1 for e in sk.evidence if e.get("ok") and e.get("supervised"))
        print(f"  {sk.name} v{sk.version} [{sk.validation_status}]: {sk.successes} ok / {sk.failures} fail, "
              f"{sup} supervised ok — {sk.description[:70]}")


def skill_admin(args) -> None:
    """Operator commands on the explore skill library (no robot, no model)."""
    from agent.explore import load_explore_config
    from common.paths import REPO_ROOT
    from memory.skills import SkillLibrary

    cfg = load_explore_config()
    lib = SkillLibrary(REPO_ROOT / cfg.skills_file)
    if args.promote:
        print(lib.promote(args.promote, cfg.skills.min_supervised_successes))
    if args.rollback:
        print(lib.rollback(args.rollback[0], int(args.rollback[1])))
    _print_skills(lib.all())


def review(s, which: str) -> None:
    """Self-review of robot episodes (no motion: memory tools only): last N, or one id."""
    all_eps = s.harness.episodes.list()
    eps = all_eps[-int(which):] if which.isdigit() and len(which) < 6 else [e for e in all_eps if e.startswith(which)]
    if not eps:
        raise SystemExit(f"no episode matches {which!r}")
    for ep in eps:
        print(f"\n========== review {ep}")
        out = s.reflect(ep)
        o = out.get("outcome") or {}
        print(f"[outcome] {'SUCCESS' if o.get('success') else 'FAIL'} · moved {o.get('blocks_moved', '?')} · "
              f"{o.get('summary', '')}\n[review] {out['reflection']}")
    print_skills(s)


if __name__ == "__main__":
    main()
