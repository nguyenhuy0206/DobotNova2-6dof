"""Freeze the base model: a read-only copy outside the Hugging Face cache, with checksums.

Fine-tuning never touches it: every run trains a LoRA adapter (data/student/runs/<id>/) on top of
it, and every comparison is "frozen base" vs "base + adapter" on the same held-out episodes.
On APFS the copy is a clone (cp -c), so it costs no disk space until a file would diverge.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

from student.paths import BASE_REPO, FROZEN_DIR


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json")


def freeze(repo: str = BASE_REPO, target: Path = FROZEN_DIR) -> dict:
    """Copy the model snapshot to `target`, write manifest.json, make everything read-only."""
    if (target / "manifest.json").exists():
        raise FileExistsError(f"{target} is already frozen (verify() checks it)")
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(repo))
    target.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        real = f.resolve()
        dst = target / f.name
        if subprocess.run(["cp", "-c", str(real), str(dst)], capture_output=True).returncode != 0:
            shutil.copy2(real, dst)                        # not APFS: a real copy
    manifest = {"repo": repo, "revision": src.name, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "files": {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in _files(target)}}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    for p in [*_files(target), target / "manifest.json"]:
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return manifest


def verify(target: Path = FROZEN_DIR) -> dict:
    """{ok, changed: [files whose checksum differs], missing: [...]}"""
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    changed, missing = [], []
    for name, info in manifest["files"].items():
        p = target / name
        if not p.exists():
            missing.append(name)
        elif _sha256(p) != info["sha256"]:
            changed.append(name)
    return {"ok": not changed and not missing, "changed": changed, "missing": missing,
            "repo": manifest["repo"], "revision": manifest["revision"], "frozen_at": manifest["frozen_at"]}
