# Max crash animation and PBR-material implementation plan

> **Execution:** inline in the Atlas/3ds Max pipeline. Houdini is explicitly
> out of scope for this plan; Claude owns that workstream.

## Goal

Make the crash beat internally consistent: `car_01`, its tyre/rim children,
the smoke/sparks/debris timing, and crash shots 11–12 must agree on one world
position and timing. Upgrade the visible Max materials with CC0 PBR sources or
host-verified procedural V-Ray graphs, without importing branded vehicle art.

## Non-negotiable constraints

- Do not export, scrape, trace, or bake Google Maps imagery, 3D data, or
  building textures. Google Maps is not an asset source. Use OSM geometry and
  CC0 material sets from the existing `texlib.py` path instead.
- Keep vehicles generic and unbranded: neutral paint, carbon, rubber and
  metal; no F1 team liveries, logos, badges or copyrighted source textures.
- Do not modify Houdini files or invoke Houdini. Claude owns that application.
- Discover V-Ray material parameter names from the live Max host before adding
  a graph. Never rely on recalled parameter names.
- Do not render. Verify animation numerically in Max, save after each live
  animation/material milestone, and keep the tyFlow flows intact.

## Task 1 — single crash pose source of truth

**Files:** `tests/test_raceanim.py`, `server/raceanim.py`

1. Add a failing unit test for a pure `crash_pose` helper that holds a spinning
   car at its contact point, adds only a bounded slide along the contact
   tangent, and stays still after the settle interval.
2. Run the focused test and record the expected missing-helper failure.
3. Add the minimal helper. It receives the clean pose and contact pose,
   delegates lateral/yaw/roll/pitch/height to `crash_state`, and returns a
   position and attitude suitable for `orientation_quat`.
4. Prove the focused test and full Python test suite pass.

## Task 2 — apply the pose to the Max scene and crash cameras

**Files:** `tests/test_raceanim.py`, `server/raceanim.py`,
`demo_yas_animate.py`

1. Measure the live frame-1388 car positions. Pin `CrashSpec.spinner_slot` to
   the actual impact car rather than the stale default.
2. Add a failing test that the crash camera target follows the same bounded
   crash pose rather than the clean lap after contact.
3. Change the car-key generator so only the spinner uses `crash_pose`; body,
   tyres and rims receive identical position/quaternion keys.
4. Rework shots 11 and 12 through `camera_for` so targets resolve from the
   shared crash target. Keep shot 11 as a high/drone wide and shot 12 as a
   static trackside detail; neither camera may fall below `site_z`.
5. Apply just the changed keys in Max (not a full scene rebuild), inspect the
   car and target transforms around frames 1387, 1388 and settle, then call
   `save_scene` to a new explicit checkpoint.

## Task 3 — material evidence and CC0 PBR upgrade

**Files:** `server/texturing.py`, `server/materials.py`, relevant tests,
`demo_yas_marina.py` or an explicit Max material-update path

1. Read the live car, rim, asphalt and building-material property names from
   the Max host. Record only host-supported properties.
2. Inventory existing local CC0 texture sets and their maps/licence metadata.
3. Write focused failing graph tests for the first supported upgrade:
   car paint (clearcoat only if live properties are confirmed) and the
   two-scale asphalt/racing-line treatment. Add glass only where OSM material
   classification justifies it.
4. Build those graphs through the bridge, verify rejected-property maps are
   empty, assign them to the intended nodes, and checkpoint the Max scene.

## Task 4 — verification and handoff

1. Re-run the complete Python suite without pytest cache writes.
2. Re-check `git status` and staged paths. Do not commit any concurrent
   Houdini/Claude changes.
3. Commit only verified Atlas Max source/tests/docs once the index contains no
   unrelated files; otherwise leave a precise handoff with the checkpoint path.

## Acceptance evidence

- At frame 1388 `car_01` is within the measured impact tolerance and its
  transform matches tyres/rims.
- After its settle time it no longer resumes the clean lap.
- Crash camera targets stay on the incident and all camera positions are above
  `site_z`.
- tyFlow objects remain present.
- Material construction reports no rejected properties, and all PBR source
  files are CC0/local—not Google Maps content.
