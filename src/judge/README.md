# Operator verifier v0

The harness owns `HarnessJudge` and `EvidenceCapture`. Executor lifecycle hooks observe
actual primitive handlers, including primitives nested in routines, and run after handler
exceptions. The harness finalizes the episode in `finally`, including model exceptions and
interruptions. Camera and state acquisition is read-only; no hardware is opened on import.

`records.py` defines frozen records. `predicates.py` gates object identity, calibration,
original capture time, stream/sequence ordering and file hashes before credit predicates.
Uncertainty crossing a boundary means unknown. `grasp_aligned` is always advisory.
`engine.py` requires an identifiable object initially outside the destination, then a
verified lift, release, retraction and persistent table support. Observed transport drops
are latched. A final view must still establish placement; an earlier pass cannot survive
later occlusion. Deadline expiry and missing prerequisites never earn credit.

`ledger.py` owns `data/judge/ledger.jsonl`: locked append, duplicate/finalization rejection,
no rewrite or repair of incomplete entries. Claims live in a separate `claim` field. Replay
writes separate, uniquely named runs under `data/judge/replay/`; it cannot promote skills.
Frame pixels, raw depth/intrinsics and controller snapshots are saved under
`data/judge/evidence/` with unique names and hashes.

The JSON task spec and evidence profile are operator configuration, outside Astra's tool
paths. The shipped task is deliberately unconfigured: no color binding, destination,
finger reference, calibration hash or validated credit predicates. Empty configuration
produces unknown/inconclusive, never a fabricated setup or success. v0 accepts strictly
convex image polygons and a single operator-bound cube with no same-color distractors.

The evidence profile is `{}` until the operator supplies validated measurements. Its
canonical JSON SHA-256 (`judge.spec.digest(profile)`) must equal the task's
`calibration_hash`. Profile fields consumed by the adapter are:

- `validated`, `object_id`, `object_color`, `single_object_scope_confirmed`;
- `stream_ids` and `image_sizes`, keyed by `env` and `wrist` (producer restart invalidates the binding);
- `wrist_pose_bounds`: one inclusive [low, high] interval per measured pose component;
- `pixel_uncertainty_px`, `edge_uncertainty_deg`, `robot_uncertainty_mm`;
- `bbox_size_bounds`: width/height intervals per camera, and `minimum_fill_ratio`,
  validated for full visibility (absence means unknown, an image-edge crop means partial);
- `finger_polygon`: validated full-image region between the fingers;
- `open_position`, `open_position_tolerance`, `retracted_z_min_mm`;
- `camera_to_base_mm`: homogeneous transform acting on deprojected millimeters;
- `cube_height_mm`, `table_z_mm`, `support_uncertainty_mm`.

No values are inferred from Astra's knowledge, observations, measurements or routine checks.
The depth adapter bounds table support over the entire object bounding box, adding the
operator's validated error bound; invalid depth means unknown. Host monotonic timestamps
record acquisition at the camera producer, not HTTP receipt. Missing transport metadata
stays missing. Full sensor exposure timestamps and automatic drift monitoring are deferred.

The only judge feedback in model tool results is `grasp_measurements`: du/dv pixels,
edge angle and visibility, from an independent wrist acquisition on a wrist look, before
the next close can be requested. Private predicate states, reasons, limits and verdicts
are never serialized there. Overridable operational checks may still stop procedures;
they cannot earn credit. Reflection tools return Astra's claims only.

Auto-promotion is disabled. The operator uses `scripts/run_explore.sh --promote NAME
--ledger-entry EVALUATION_ID` (repeat the citation option for independent episodes).
Promotion requires finalized credited entries matching the current dependency digest.
Rollback always returns a candidate, even if the old procedure was validated. Legacy
counts and supervised execution claims cannot validate a skill.

TODO: deploy the judge under the operator account/process and run teacher `run_python`
under a restricted OS user before the next teacher session. This implementation supplies
an application boundary, not same-user filesystem isolation. OS administration was
explicitly excluded from this change. Operator binding/calibration and labeled validation
of the predicates are required before enabling credit. No robot/camera/gripper was used
to configure or validate these defaults.
