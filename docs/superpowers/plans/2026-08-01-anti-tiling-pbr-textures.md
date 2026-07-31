# Anti-tiling PBR textures implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply non-repeating, physically scaled CC0 PBR imagery to the Yas
track and ordinary buildings, while keeping generic car paint procedurally
clearcoated and tile-free.

**Architecture:** Add a pure-data V-Ray graph builder that combines two
independent world-space triplanar PBR sources with a separately projected macro
mask. A focused scene updater fetches approved assets, builds graphs through
the bridge, assigns only measured node groups, validates host rejections, and
saves material milestones.

**Tech Stack:** Python 3.13, pytest, `server.texturing` graph data,
`server.texlib` CC0 fetching, `server.maxbridge`, 3ds Max 2027 + V-Ray 7.

## Global constraints

- Do not modify `Houdina/`, `.agents/`, or `docs/hyper-real-brief-plan.md`.
- Do not use Google Maps, Street View, Google 3D data, branded vehicle
  textures, F1 team names, badges or liveries.
- Do not render. Verify graph structure, live host rejections, assignments and
  save checkpoints only.
- Never infer V-Ray property names. Use the properties already discovered on
  the live host (`VRayTriplanarTex`, `Mix`, `Noise`, `VRayNormalMap`,
  `ColorCorrection`) and fail on host rejection.
- Preserve current kerb, line, grandstand and car material assignments.
- `nor_gl`, not `nor_dx`, is the normal-map convention.
- The car clearcoat stays procedural/object-local; a single bitmap mapped over
  an entire moving car is forbidden.

---

## Task 1: Lock the anti-tiling graph contract in tests

**Files:**
- Modify: `tests/test_texturing.py`
- Modify: `server/texturing.py`

- [x] Add a failing test for `anti_tiling_scanned_graph` that supplies two
  minimal PBR texture-set dictionaries and asserts:
  - primary and secondary image tri-planars feed the final diffuse blend;
  - their scale ratio is 1.618;
  - random offset/rotation flags are true on both image projections;
  - the secondary source has a non-zero offset and rotation;
  - an independently projected macro mask feeds the confirmed `Mix.Mask` slot;
  - asphalt omits a bump output.
- [x] Run the red test (initially failed at import because the builder did not
  exist):

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_texturing.py -q -p no:cacheprovider --basetemp out\pytest-antitile-red
  ```

- [x] Implement the smallest pure-data graph builder. Keep source maps,
  tri-planars, macro noise/mask, colour blend, roughness blend/inversion, and
  optional normal blend as explicitly named graph nodes. Use a fixed `1.618`
  secondary scale ratio only when the caller has not provided an explicit
  secondary scale.
- [x] Add a second test showing building normals can be enabled while asphalt
  deliberately omits the bump channel.
- [x] Run focused green verification: 40 passed.

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_texturing.py -q -p no:cacheprovider --basetemp out\pytest-antitile-green
  ```

## Task 2: Build a bounded live-scene PBR updater

**Files:**
- Create: `demo_yas_pbr_textures.py`
- Modify: relevant unit tests only if pure filtering helpers are extracted

- [x] Create a source-driven updater with no geometry/OSM rebuild path. Fetch
  only `asphalt_track` and `concrete_slab_wall_02` in 4K with maps
  `Diffuse`, `Rough`, `nor_gl`; require the existing `asphalt_02` and
  `concrete_layers_02` local CC0 sets.
- [x] Construct a track graph at 2.0 m / 3.236 m image scales and a 37 m macro
  mask. Set normal strength to zero so no low-sun corrugation is introduced.
- [x] Construct a building graph at 2.1 m / 3.3978 m image scales and a 19 m
  macro mask. Enable bounded normal detail through `VRayNormalMap`.
- [x] Use current host node inventory to assign the track graph only to the
  three track-asphalt nodes and the building graph only to ordinary `osm_`
  buildings. Exclude the six supported grandstand/stadium nodes and all
  kerbs, paint lines, tyres, rims and car bodies.
- [x] After each bridge build, reject an empty target list, a missing texture
  map, or any non-empty `rejected` response. Save explicit scene checkpoints
  after track, building and car-material verification using
  `MaxBridge.save_scene`.

## Task 3: Live host parameter and assignment verification

**Files:**
- Modify if necessary: `demo_yas_pbr_textures.py`
- No source change expected: `server/maxbridge.py`,
  `bridge/atlas_max_handlers.py`

- [x] First construct the track graph on a temporary material without assignment
  through the bridge. Confirm every randomisation property,
  `frame_offset`, `texture_rotation`, and `Mix.Mask` is accepted. Stop if the
  host rejects anything rather than silently falling back.
- [x] Apply track graph to three track nodes, confirm the material count is
  three, and save `out/yas_race_pbr_track.max`.
- [x] Apply building graph to 373 ordinary building nodes, confirm six
  grandstand/stadium nodes remain on their existing materials, and save
  `out/yas_race_pbr_buildings.max`.
- [x] Query every generic car material. Confirm each retains the
  host-verified clearcoat values (`coat_amount=0.80`,
  `coat_glossiness=0.96`, `coat_ior=1.52`) and that no bitmap material was
  assigned. Save `out/yas_race_hyperreal_pbr.max`.

## Task 4: Full verification and safe handoff

**Files:**
- Modify: `docs/superpowers/specs/2026-08-01-anti-tiling-pbr-design.md`
- Modify: `docs/superpowers/plans/2026-08-01-anti-tiling-pbr-textures.md`

- [x] Mark checked implementation items in this plan and record the actual
  host rejection/count/save evidence in the design document.
- [x] Run the complete suite: 993 passed.

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp out\pytest-antitile-full
  ```

- [ ] Inspect both normal and cached diffs. Stage only the source, tests and
  two anti-tiling documents; do not stage Claude's `Houdina/` work or unrelated
  untracked files.
- [ ] Commit and push only after the full suite passes. Report that structural
  anti-tiling acceptance passed, and explicitly state that no render was made
  at the user's request.
