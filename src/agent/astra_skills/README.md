# Astra's own routines and checks

Astra writes, tests and rolls back the Python files here itself (tools `write_routine`,
`run_routine`, `rollback_routine`; sandbox and API in `src/actions/astra_code.py`).

- One file = one routine or check, named after the file. A file with the name of a core
  routine/check in `src/actions/local_steps.py` replaces it.
- Fixed (not editable here): the primitive tools and their safety checks, and `lift_to`
  (skill recovery relies on it only going up).
- `_history/` keeps every replaced version (`NAME.vN.py`).
- Seeded on 16 Nov with `approach`, `grasp_ladder` (used by the built-in pick skill).
- Operator: delete a file to fall back to the core version; `git diff` shows what Astra changed.
