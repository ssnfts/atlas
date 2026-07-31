# Plan: tyFlow FX — Tyre Smoke + Crash Debris + Sparks

**Scope:** Three effects, in the order the fx plan specifies:
1. Tyre smoke — highest value, lowest risk; needed in every shot.
2. Crash debris — bounded to shots `11_crash_wide` and `12_crash_tight`.
3. Sparks — cheap contact sell; polish layer on the same two shots.

All three share the same generate→write→review→run pattern.
Reference: `docs/tyflow-fx-plan.md`.

**Security boundary confirmed by operator:**
`ATLAS_ALLOW_MAXSCRIPT=1` is set in the host environment. The generated `.ms` script
is written to `out/tyfx_smoke.ms` and reviewed before execution. The bridge does not
auto-run it; an explicit MCP tool call (or manual paste into Max) triggers execution.

**Smoke emitter:** rear tyres only (`car_NN_tyres` nodes already in scene), gated on
speed — emitting when the car's speed exceeds the braking threshold (~200 km/h / 55.6 m/s).
Speed values come from `raceanim.speed_profile` rather than from hand-keyed on/off.

**Crash shots:** `11_crash_wide` (lap 0.646–0.742) and `12_crash_tight` (lap 0.742–0.820).
Both bracket `CRASH.at_distance_m = 3650 m` (lap fraction ~0.69). The crash event is
pinned to distance, not frame, so retiming does not move it. Debris starts at `t = 0`
(contact), never before. Contact frame = frame at which the spinner reaches 3650 m along
the lap, computed from `raceanim.distance_at` / `raceanim.lap_time`.

---

## Sub-Task 1 — MaxScript generator in `server/tyfx.py`

**Intent:**
Extend `tyfx.py` to emit a reviewable MaxScript string that builds the tyFlow event
graph for tyre smoke. The string is the deliverable; execution is a separate step.
The existing `tyre_smoke()` function keeps its `complete=False` path for hosts
where `ATLAS_ALLOW_MAXSCRIPT=0`.

**Expected Outcomes:**
- A new `generate_smoke_script(flow_name, emitter_nodes, speed_threshold_ms, wind_bearing_deg, wind_speed_ms, end_frame)` function in `tyfx.py` returns a valid MaxScript string.
- The script: creates/reuses the tyFlow node; adds one event with Birth Flow (continuous,
  ~600/frame), Position Object (rear tyre nodes), Velocity (inherit 0.15 + 1.5 m/s
  normal), Wind Force (ERA5 values passed in), Drag (coefficient tuned to ~15 m stall),
  Scale (x4 over life), Age Test → Delete (2.5 s).
- Speed-gating: a `tyConditions` operator checks particle age against a boolean
  keyed to the car's speed at the emitter frame. The speed array is passed in as
  a MaxScript array literal embedded in the script.
- The script includes a comment header recording the Atlas version, git hash
  (if available via `subprocess`), and the parameter values used — so the file
  on disk is self-documenting.
- No test requires Max to be running; the generator is pure string emission and
  is unit-testable offline.

**Todo List:**
1. Add `generate_smoke_script(flow_name, emitter_nodes, *, speed_threshold_ms, wind_bearing_deg, wind_speed_ms, end_frame, site_z)` to `tyfx.py`.
2. Embed the speed threshold as a per-frame bool array in the MaxScript, computed from `raceanim.speed_profile` by the caller (not inside tyfx, to keep it pure).
3. Write the MaxScript string sections: node creation, event 01 (Birth + Position + Velocity + Force + Drag + Scale + Age), speed-gate binding, final `tyFlow.update()` call.
4. Add a `script_header(params)` helper that prepends a comment block with build metadata.
5. Update `__all__` to export `generate_smoke_script`.
6. Add unit tests in `tests/test_tyfx.py` that:
   - Call `generate_smoke_script` with known parameters and assert the string contains the expected operator names and parameter values.
   - Assert the script is non-empty and starts with a `--` comment block.
   - Assert `tyre_smoke()` still returns `complete=False` when `available()` would return False (mock the bridge).

**Relevant Context:**
- [`server/tyfx.py`](server/tyfx.py) — existing `SMOKE_RECIPE`, `available()`, `tyre_smoke()`.
- [`server/raceanim.py`](server/raceanim.py) — `speed_profile(spine)` returns m/s per vertex; `CRASH` at 1980 m; at crash site speed is ~210 km/h = ~58.3 m/s.
- [`server/weather.py`](server/weather.py) — `fetch_observation` returns `Observation` with `wind_speed_ms` and `wind_direction_deg`; ERA5 reference values are 7.34 m/s at 331°.
- [`server/maxbridge.py`](server/maxbridge.py) — `maxscript(code)` method exists; bridge checks `ALLOW_MAXSCRIPT` on its side.

**Status:** `[ ] pending`

---

## Sub-Task 2 — Script writer in `server/tyfx.py`

**Intent:**
Add a `write_smoke_script(path, ...)` function that calls `generate_smoke_script`
and writes the result to disk, then returns the path and a human-readable summary.
This is the "write to `out/tyfx_smoke.ms` and show before running" boundary.

**Expected Outcomes:**
- `write_smoke_script(path, ...) -> dict` writes the `.ms` file and returns
  `{"path": str, "lines": int, "summary": str, "params": {...}}`.
- The summary is a short English description of what the script will do (operator
  list, emitter count, speed threshold in km/h, wind values), readable without
  opening the file.
- The function works offline (no bridge, no Max).
- A test asserts the file lands on disk and that `lines > 0`.

**Todo List:**
1. Add `write_smoke_script(path, emitter_nodes, spine, *, wind_bearing_deg, wind_speed_ms, end_frame, site_z, speed_threshold_ms)` to `tyfx.py`.
2. Compute the per-frame speed gate array from `raceanim.speed_profile(spine)` and pass it to `generate_smoke_script`.
3. Write the file with UTF-8 encoding; return the dict described above.
4. Add test: call `write_smoke_script` with a tmp path, assert file exists and non-empty.

**Relevant Context:**
- Same as Sub-Task 1.
- `out/` directory — already created by `mcp_server.py` (`OUT_DIR.mkdir(exist_ok=True)`).

**Status:** `[ ] pending`

---

## Sub-Task 3 — Bridge execution path

**Intent:**
Add a `run_smoke_script(bridge, path) -> dict` function that reads the `.ms` file
from disk and sends it to Max via `bridge.maxscript()`. This is the execution step,
separate from generation, so a reviewer can read the file before this is called.

**Expected Outcomes:**
- `run_smoke_script(bridge, path)` raises `FileNotFoundError` if the script file
  does not exist (operator must call `write_smoke_script` first).
- On success it returns the raw Max response plus a note that the flow exists and
  has events.
- If `bridge.maxscript` raises because `ATLAS_ALLOW_MAXSCRIPT=0`, the error is
  re-raised with a clear message pointing to `.env` and the flag.
- A unit test mocks the bridge and asserts the correct code path for
  missing-file, flag-disabled, and success cases.

**Todo List:**
1. Add `run_smoke_script(bridge, path) -> dict` to `tyfx.py`.
2. Guard: `if not Path(path).is_file(): raise FileNotFoundError(...)`.
3. Read the file, call `bridge.maxscript(code)`, return structured result.
4. Add the `ATLAS_ALLOW_MAXSCRIPT` error message to the raised exception if the
   bridge rejects the call.
5. Add unit tests for all three paths (file missing, flag off, success).

**Relevant Context:**
- [`server/maxbridge.py`](server/maxbridge.py:245) — `maxscript()` method; bridge enforces the flag server-side in `bridge/atlas_max_handlers.py`.

**Status:** `[ ] pending`

---

## Sub-Task 4 — MCP tool `atlas_tyre_smoke`

**Intent:**
Expose the full generate→write→(optionally run) flow as a single MCP tool.
The tool writes the script and returns its path and summary. Execution is gated
on an explicit `execute` parameter that defaults to False, so the model must
opt in to running it.

**Expected Outcomes:**
- `atlas_tyre_smoke(spine_json, emitter_nodes, *, execute=False, speed_threshold_kmh=200.0, end_frame=2400)` added to `mcp_server.py`.
- When `execute=False` (default): writes `out/tyfx_smoke.ms`, returns path + summary. Does not touch Max.
- When `execute=True`: also calls `run_smoke_script`; requires Max to be running.
- Returns `{"success": false}` with a clear error if `ATLAS_ALLOW_MAXSCRIPT` is not enabled in the bridge.
- `atlas_tyre_smoke` added to the `TOOLS` list in `mcp_server.py`.
- `tests/test_mcp_server.py` gains a test asserting the tool is registered and its schema has the expected parameters.

**Todo List:**
1. Add `atlas_tyre_smoke` function to `mcp_server.py`, decorated with `@_tool_result`.
2. Accept `spine_json` as a JSON string (list of [x,y] pairs); parse inside the tool.
3. Accept `emitter_nodes` as a list of strings (node names of rear tyre meshes).
4. Fetch wind values from `weather.fetch_observation` using the site lat/lon and the reference time, or accept them as optional parameters with ERA5 reference fallbacks (331°, 7.34 m/s).
5. Call `tyfx.write_smoke_script` → if `execute`, call `tyfx.run_smoke_script`.
6. Add tool to `TOOLS` list.
7. Add test to `tests/test_mcp_server.py`.

**Relevant Context:**
- [`server/mcp_server.py`](server/mcp_server.py:579) — `TOOLS` list, `_tool_result` decorator, `_bridge()` helper.
- [`server/tyfx.py`](server/tyfx.py) — functions from Sub-Tasks 1–3.

**Status:** `[ ] pending`

---

## Sub-Task 5 — Verification helpers

**Intent:**
The fx plan is explicit: "verify by measurement, not by eye." Add two helpers:
(1) particle count readback — non-zero in the simulation window confirms the
flow is emitting; (2) masked render check — a flat-colour material on the
particles and a render of one frame, with pixel movement between two frames
as the test.

**Expected Outcomes:**
- `tyfx.particle_count(bridge, flow_name, frame) -> int` reads the particle
  count at a given frame by querying the flow's event via MaxScript. Returns 0
  if the flow has no events.
- `tyfx.smoke_verify(bridge, flow_name, *, check_frames, out_dir) -> dict`
  renders two masked frames (flat white particles on black), counts non-black
  pixels, asserts `count > 0` in the simulation window and `count == 0` outside.
  Returns `{"ok": bool, "frames": [{"frame": int, "pixel_count": int}], "reason": str}`.
- These functions are called by `run_smoke_script` and their results included in
  its return value.
- A test for `particle_count` mocks the bridge and asserts the query string
  is formed correctly.

**Todo List:**
1. Add `particle_count(bridge, flow_name, frame) -> int` to `tyfx.py`.
2. Add `smoke_verify(bridge, flow_name, *, check_frames, out_dir) -> dict` to `tyfx.py`.
3. `smoke_verify` assigns a flat white `VRayLightMtl` to the particles, renders at small resolution (320x180), reads pixel counts with the PIL/Pillow already in requirements.
4. Update `run_smoke_script` to call `smoke_verify` and include results.
5. Add mocked unit tests for both functions.

**Relevant Context:**
- [`requirements.txt`](requirements.txt) — `pillow>=10.0` already present.
- [`server/tyfx.py`](server/tyfx.py) — existing module.
- [`docs/tyflow-fx-plan.md`](docs/tyflow-fx-plan.md) — verification section: "particle count per frame, non-zero in the window; a masked render confirming pixels move."

**Status:** `[ ] pending`

---

## Sub-Task 6 — Crash debris MaxScript generator and MCP tool

**Intent:**
The crash is fully described in `raceanim.CrashSpec` and `crash_state()`. The debris
flow must start at the contact frame (computed, not guessed), use the spinner's front-wing
nodes as the Voronoi fracture source, and inherit the car's velocity at impact (~40 m/s
forward + 6 m/s radial scatter). The script is generated, written to `out/tyfx_debris.ms`,
reviewed, then run. Same generate→write→run pattern as smoke.

**Expected Outcomes:**
- `generate_debris_script(flow_name, wing_nodes, *, contact_frame, car_speed_ms, yaw_rate_degs, site_z, end_frame) -> str` added to `tyfx.py`.
- The script builds one tyFlow with three events as per the fx plan:
  - Event 01: Birth burst at `contact_frame`, ~180 particles, Position Object (front wing faces), Shape (Voronoi fracture of wing mesh), Velocity (inherit 0.7 x car_speed + 6 m/s radial scatter).
  - Event 02: PhysX Shape (convex hull per fragment), PhysX Collision (graded site plane + kerbs), Bounce/Friction (carbon-on-asphalt: low bounce, high friction), Spin (from yaw_rate_degs at impact).
  - Event 03: Age > 6 s → Freeze (not delete — debris stays on track).
- `contact_frame_for(spine, spec, fps=24) -> int` helper: computes the frame at which
  the spinner reaches `spec.at_distance_m`. This is the single number tying simulation
  to animation. It uses `raceanim.distance_at` by inverse: walk the speed profile until
  cumulative distance reaches `at_distance_m`.
- `write_debris_script(path, wing_nodes, spine, spec, *, site_z, end_frame, fps) -> dict`.
- `run_debris_script(bridge, path) -> dict` — same guard pattern as `run_smoke_script`.
- MCP tool `atlas_crash_debris(wing_nodes, *, execute=False, end_frame=2400)` added to
  `mcp_server.py` and `TOOLS`.
- Fracture source is the **front wing only** — not the monocoque. The fx plan is explicit.
- Verification: `debris_verify(bridge, flow_name, contact_frame, *, out_dir) -> dict`
  checks particle count > 0 at `contact_frame + 1`, and that resting positions
  at `contact_frame + 150` (6 s later) are within track bounds and above z = site_z.

**Todo List:**
1. Add `contact_frame_for(spine, spec, fps=24) -> int` to `tyfx.py`. Walk `raceanim.speed_profile` and cumulative arc lengths until `at_distance_m` is reached; divide elapsed time by `1/fps`.
2. Add `generate_debris_script(flow_name, wing_nodes, *, contact_frame, car_speed_ms, yaw_rate_degs, site_z, end_frame) -> str` to `tyfx.py`.
3. Add `write_debris_script(path, wing_nodes, spine, spec, *, site_z, end_frame, fps) -> dict`.
4. Add `run_debris_script(bridge, path) -> dict`.
5. Add `debris_verify(bridge, flow_name, contact_frame, *, out_dir) -> dict`.
6. Add MCP tool `atlas_crash_debris` to `mcp_server.py`.
7. Unit tests:
   - `contact_frame_for` with `CRASH` spec: assert result is in range [1700, 1900] for Yas lap at 24 fps.
   - `generate_debris_script` string contains "Birth", "PhysX", "Voronoi", "Freeze".
   - `write_debris_script` writes non-empty file.
8. Update `__all__` in `tyfx.py`.

**Relevant Context:**
- [`server/raceanim.py`](server/raceanim.py:400) — `CrashSpec`, `CRASH` (`at_distance_m=1980`), `crash_state()`, `speed_profile()`, `lap_time()`, `distance_at()`.
- [`docs/tyflow-fx-plan.md`](docs/tyflow-fx-plan.md:77) — debris event structure; "fracture the front wing only."
- Shots `11_crash_wide` (lap 0.735–0.775) and `12_crash_tight` (lap 0.745–0.765).
- Crash cache window: frames ~1760–1860 at 24 fps per the fx plan.
- **Note:** `server/raceanim.py` was modified externally before this session. Read its
  current state (`git diff HEAD server/raceanim.py`) before implementing Sub-Task 6
  to confirm `CRASH` and `crash_state` signatures have not changed.

**Status:** `[ ] pending`

---

## Sub-Task 7 — Sparks MaxScript generator and MCP tool

**Intent:**
Titanium skid-block sparks at ground contact for ~0.4 s starting at the contact frame.
Small, bright, short-lived, heavy drag, gravity on. Cheap effect — sells the impact.
Shading via `VRayLightMtl` on the particles (not actual lights — no shadow cost).

**Expected Outcomes:**
- `generate_sparks_script(flow_name, floor_nodes, *, contact_frame, car_speed_ms, site_z, end_frame) -> str` added to `tyfx.py`.
- Script structure:
  - Birth burst over `contact_frame` to `contact_frame + 10` frames (0.4 s at 24 fps).
  - Position Object (floor/skid-block nodes, or a point at the crash position if floor_nodes is empty).
  - Velocity: inherit 0.9 car speed + 3 m/s upward scatter + random ±45° spread.
  - Force Gravity (standard 9.81 m/s²).
  - Drag (heavy — sparks decelerate fast).
  - Scale (small, constant — sparks do not grow).
  - Age Test → Delete (0.3 s).
  - Comment in script noting `VRayLightMtl` should be assigned to the particles manually.
- Falls back gracefully if `floor_nodes` is empty — emits from the contact point
  instead. Sparks are a polish layer and must not block the crash render.
- `write_sparks_script(path, floor_nodes, spine, spec, *, site_z, end_frame, fps) -> dict`.
- `run_sparks_script(bridge, path) -> dict`.
- MCP tool `atlas_crash_sparks(floor_nodes, *, execute=False, end_frame=2400)` added to
  `mcp_server.py` and `TOOLS`.

**Todo List:**
1. Add `generate_sparks_script(...)` to `tyfx.py`.
2. Reuse `contact_frame_for(spine, spec)` from Sub-Task 6.
3. Add `write_sparks_script(path, floor_nodes, spine, spec, *, site_z, end_frame, fps) -> dict`.
4. Add `run_sparks_script(bridge, path) -> dict`.
5. Add MCP tool `atlas_crash_sparks` to `mcp_server.py`.
6. Unit tests:
   - `generate_sparks_script` string contains "VRayLightMtl", "Birth", "Gravity".
   - `write_sparks_script` with empty `floor_nodes` still writes a non-empty file.
7. Update `__all__`.

**Relevant Context:**
- [`docs/tyflow-fx-plan.md`](docs/tyflow-fx-plan.md:99) — sparks section.
- Sub-Task 6 `contact_frame_for` helper — shared by debris and sparks.
- `contact_frame_for` must be implemented in Sub-Task 6 before Sub-Task 7 can begin.

**Status:** `[ ] pending`

---

## Crash Shot Rendering Notes (no code changes needed)

The two crash shots are already defined in `raceanim.SHOTS` and rendered by `demo_yas_lap.py`:

| Shot | Kind | Lap range | What it shows |
|---|---|---|---|
| `11_crash_wide` | drone | 0.735–0.775 | Wide: the tell, the contact, the slide |
| `12_crash_tight` | static | 0.745–0.765 | Trackside, tight, level with the impact |

Both bracket the contact frame so anticipation is on screen. Both already have camera
geometry. The FX additions (debris + sparks) will run on the same frames automatically
once their flows are in the scene — no camera changes needed.

To render the crash shots as sequences rather than stills: raise `FRAMES` in
`demo_yas_lap.py` from 1 to the desired frame count. The cars and camera are both
functions of lap fraction, so the whole cut list animates from the same two numbers.

---

## Implementation Order

Sub-Tasks must be done in strict order: 1 → 2 → 3 → 4 → 5 → 6 → 7.

- ST1, ST2, and the generators in ST6 and ST7 are **offline** — no Max required.
- ST3–ST5 and the run/verify paths in ST6–ST7 require Max running with the bridge
  and `ATLAS_ALLOW_MAXSCRIPT=1`.

**Bridge status at time of writing:** listening on 127.0.0.1:9879, MaxScript **disabled**,
handler reload on. To enable MaxScript for ST3 onward: set `ATLAS_ALLOW_MAXSCRIPT=1` in
the Windows environment **before launching Max**, then restart the bridge with:
```
python.ExecuteFile @"C:\Users\mabdu\Desktop\atlas\bridge\start_bridge.py"
```
Confirm with: `[atlas] raw MaxScript : enabled`

---

## Invariants That Must Hold Throughout

1. `tyre_smoke()` must still return `complete=False` when the bridge has `ATLAS_ALLOW_MAXSCRIPT=0`. The honest stub is not replaced — it is extended.
2. Every generated `.ms` file must exist on disk and be reviewed before the corresponding `run_*_script` may be called. Hard file-existence check, not a convention.
3. Every new function in `tyfx.py` that touches the bridge is unit-tested with a mock — no test should require 3ds Max.
4. Wind values in smoke scripts are the real ERA5 values (331°, 7.34 m/s) or explicitly documented overrides. Never invented.
5. All three MCP tools default `execute=False` — model must opt in to execution.
6. Debris and sparks start at `contact_frame`, never before. `contact_frame_for` is the single source of truth for this number.
7. Fracture source is the front wing only — not the monocoque. Design decision from the fx plan.
8. Before implementing Sub-Task 6: check `git diff HEAD server/raceanim.py` — the file was modified externally and the `CRASH` spec or `crash_state` signature may have changed.
