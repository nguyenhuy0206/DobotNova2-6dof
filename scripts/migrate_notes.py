#!/usr/bin/env python3
"""One-off: raw notes (data/memory/astra_notes.jsonl) -> hypotheses (data/memory/hypotheses.json).

Trial logs and notes about removed skills/routines are archived (kept, never shown in prompts);
the rest become open hypotheses with the note's date as scope. The notes file is not changed.

    .venv/bin/python scripts/migrate_notes.py            # dry: prints what it would do
    .venv/bin/python scripts/migrate_notes.py --write
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from memory.hypotheses import HYPOTHESES_FILE, HypothesisStore, migrate_notes
from memory.notes import NotesStore

# removed on 16 Nov (operator): the measure-before-grip experiment
REMOVED = {"measure_block_height", "measure_top", "contact_calibrated", "contact_calibration",
           "contact calibration", "Calibration unready"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.write and HYPOTHESES_FILE.exists():
        raise SystemExit(f"{HYPOTHESES_FILE} exists already — not migrating twice")
    path = HYPOTHESES_FILE if args.write else Path(tempfile.mkdtemp()) / "h.json"
    store = HypothesisStore(path)
    stats = migrate_notes(NotesStore().all(), store, REMOVED)
    print(stats, store.counts())
    print("\nPrompt digest:\n" + store.digest())
    print("\nArchived:")
    for h in store.items:
        if h.get("archived"):
            print(f"  {h['id']} ({h['archived']['reason']}): {h['claim'][:90]}")
    if not args.write:
        shutil.rmtree(path.parent, ignore_errors=True)
        print("\n(dry run — use --write to save)")
