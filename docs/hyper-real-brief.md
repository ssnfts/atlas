# Hyper-real Yas continuation brief

This is the compact handoff for the next implementation pass. It records the
current state of the repository, not a visual wish list. Every source pointer
below is a live implementation point; anything marked **not implemented** is
deliberately not presented as working.

## Scope and non-negotiables

- Build the Yas Marina test at 1:1 scale using the scene frame already in
  Atlas. The graded site plane is `SITE_Z = 5.27 m`
  ([`demo_yas_animate.py:33-50`](../demo_yas_animate.py#L33-L50)).
- The edit is 24 fps. `END_FRAME` is derived from `raceanim.build_edit()` and
  `edit_length_frames()`, not a hard-coded duration
  ([`demo_yas_animate.py:36-50`](../demo_yas_animate.py#L36-L50)).
- Use the repository interpreter: `.venv\\Scripts\\python.exe`.
- The Max bridge defaults to `127.0.0.1:9879`
  ([`.env.example:12-22`](../.env.example#L12-L22),
  [`bridge/atlas_max_bridge.py:58-60`](../bridge/atlas_max_bridge.py#L58-L60)).
- `ATLAS_ALLOW_MAXSCRIPT=1` is an operator decision. It enables arbitrary code
  execution in the host and must be present **before 3ds Max launches**
  ([`.env.example:17-22`](../.env.example#L17-L22),
  [`bridge/atlas_max_handlers.py:23-27`](../bridge/atlas_max_handlers.py#L23-L27)).
- The current 3ds Max Education licence is non-commercial; files saved from it
  carry that flag. A commercial Max seat is required for commercial output
  ([`README.md:284-286`](../README.md#L284-L286)). Houdini is currently
  connected as Apprentice (`atlas.hipnc`), which has the same practical
  non-commercial limitation.
- Per operator direction, **do not implement or revisit any skydome/HDRI work**
  in this continuation.

## Verify first

1. Check the Max connection with `MaxBridge().ping()`
   ([`server/maxbridge.py:82-84`](../server/maxbridge.py#L82-L84)). Do not
   infer host health from a previous render.
2. Check the MaxScript environment value before restarting Max if execution is
   needed. The handler captures it at import time in `ALLOW_MAXSCRIPT`; changing
   the Windows variable while Max is open cannot enable the current bridge
   ([`bridge/atlas_max_handlers.py:26`](../bridge/atlas_max_handlers.py#L26)).
3. Run the offline FX test suite in an isolated temporary directory:

   ```powershell
   & .venv\Scripts\python.exe -m pytest tests\test_tyfx.py -q `
     -p no:cacheprovider --basetemp "$PWD\out\pytest"
   ```

   The suite was last verified as **34 passed**. The isolated base temp avoids
   a protected shared `%TEMP%\\pytest-of-mabdu` directory being mistaken for an
   FX test failure.

## 1. Asphalt: preserve the working baseline, then add only measurable detail

### What exists now

`GRAPHS["track_asphalt"]` is a procedural two-scale material: 6 m paving/repair
variation plus weak 0.45 m aggregate colour variation. It deliberately has no
bump map and retains the base glossiness at `0.62`; both decisions follow
host-side visual tests under the 7.5-degree sun
([`server/texturing.py:468-532`](../server/texturing.py#L468-L532),
[`server/materials.py:100-106`](../server/materials.py#L100-L106)).

Build it only through the validated graph route:

```python
graph = texturing.GRAPHS["track_asphalt"]
texturing.validate_graph(graph)
reply = bridge.build_material(graph.as_dict(), track_node_names)
assert not reply["rejected"], reply["rejected"]
```

`Graph.as_dict()` supplies the graph and derived slot writes; the bridge builds
and wires the complete texmap tree inside Max
([`server/texturing.py:122-193`](../server/texturing.py#L122-L193),
[`server/maxbridge.py:374-399`](../server/maxbridge.py#L374-L399),
[`bridge/atlas_max_handlers.py:810-949`](../bridge/atlas_max_handlers.py#L810-L949)).

### The load-bearing material rule

Use only classes in `TEXMAP_CLASSES` and channels in `TEXMAP_SLOTS`
([`server/texturing.py:71-95`](../server/texturing.py#L71-L95)). A VRayMtl map
also needs its paired `_on` flag; attaching a map without the flag produces a
plausible but untextured material ([`server/texturing.py:17-25`](../server/texturing.py#L17-L25),
[`bridge/atlas_max_handlers.py:913-932`](../bridge/atlas_max_handlers.py#L913-L932)).
The bridge returns every rejected class, property, input and material slot.
**A non-empty `rejected` result is a hard stop**, not a warning to render past.

### Next material increment

The unimplemented realism layer is a racing-line mask: darker rubber through
the preferred line, with local braking marks at the crash approach, while
leaving the current 6 m / 0.45 m breakup intact. Keep it out of raw
`texmap_reflectionGlossiness`: the existing source documents that feeding an
unbounded Noise map to that scalar slot made mirror-like sand reflections
([`server/texturing.py:513-531`](../server/texturing.py#L513-L531)). First build
the mask as a bounded value, apply it to a test track segment, read back the
host result, and only then expand it to the full ribbon.

`scanned_graph()` exists for CC0 bitmap PBR sets
([`server/texturing.py:364-433`](../server/texturing.py#L364-L433)), but is not
the safe first route for the track: its code uses `ColorCorrection` while the
confirmed class set contains `Color_Correction`
([`server/texturing.py:76`](../server/texturing.py#L76),
[`server/texturing.py:414-416`](../server/texturing.py#L414-L416)). Treat that
as a latent host-validation bug; fix it separately with a failing test and a
live class probe before using scanned asphalt maps.

## 2. tyFlow: generated scripts are ready; execution is not implicit

The pure/offline API is exported from
[`server/tyfx.py:40-56`](../server/tyfx.py#L40-L56):

| Effect | Generate/write | Explicit execution |
|---|---|---|
| Tyre smoke | `generate_smoke_script`, `write_smoke_script` ([lines 177-427](../server/tyfx.py#L177-L427)) | `run_smoke_script` ([lines 428-458](../server/tyfx.py#L428-L458)) |
| Crash debris | `generate_debris_script`, `write_debris_script` ([lines 615-791](../server/tyfx.py#L615-L791)) | `run_debris_script` ([lines 792-811](../server/tyfx.py#L792-L811)) |
| Sparks | `generate_sparks_script`, `write_sparks_script` ([lines 816-983](../server/tyfx.py#L816-L983)) | `run_sparks_script` ([lines 984-1003](../server/tyfx.py#L984-L1003)) |

`out/tyfx_smoke.ms` exists and is reviewable. The debris and sparks files do
not exist until their corresponding write functions are called. The required
sequence is: generate → write → human review → `execute=True` / `run_*_script`
after the MaxScript flag has been enabled and the bridge restarted. The runner
does a file-existence check and converts a disabled-host response into a direct
operator message ([`server/tyfx.py:428-458`](../server/tyfx.py#L428-L458)).

The crash is pinned to `CRASH.at_distance_m = 3650.0`, not a historical frame
constant ([`server/raceanim.py:499-520`](../server/raceanim.py#L499-L520)).
`contact_frame_for()` resolves the frame from the motion profile
([`server/tyfx.py:575-613`](../server/tyfx.py#L575-L613)). The emitted tyFlow
operator names are still unverified against the installed 2027 build: execute
one effect at a time and use `particle_count()` / `smoke_verify()` before
claiming an effect is present ([`server/tyfx.py:461-572`](../server/tyfx.py#L461-L572)).

## 3. Building heights: the code path exists; coverage is the real gap

`resolve_height()` already has the desired precedence:

1. `height` tag;
2. `building:levels × 3.0 m`, plus `roof:height` where available;
3. the `8.0 m` default.

This is implemented in [`server/osm.py:550-601`](../server/osm.py#L550-L601),
with constants at [`server/osm.py:52-62`](../server/osm.py#L52-L62). Therefore
do not add a second inference path. Re-fetch or inspect the particular Yas
OSM response and tabulate `height_source` instead. The previously reported
“7.1% measured height / roughly 40% levels” figures are data-audit numbers,
not repository invariants, and must be recalculated when the OSM data changes.

For buildings still at the default, improve visible foreground assets manually
or with a separately recorded heuristic; label the result as inferred so it
cannot be mistaken for a measured height.

## 4. Crash cameras: reframe at the actual incident, preserve the timeline law

The current camera definitions are:

- `11_crash_wide`: lap `0.646–0.742`, using `_behind(..., 62.0, 14.0, 11.0, ...,
  34.0, 40.0)`;
- `12_crash_tight`: lap `0.742–0.820`, using `_trackside(..., 1.0, 21.0, 1.6,
  ..., fov=26.0)`.

They are source-of-truth `Shot` entries in
[`server/raceanim.py:761-769`](../server/raceanim.py#L761-L769). The incident is
at lap ~0.69, so Shot 11 contains anticipation/contact and Shot 12 follows it.
Do not restore the stale 0.735–0.775 / 0.745–0.765 ranges found in the old plan.

Reframe by changing only the camera callable parameters, then resolve sampled
frames through `camera_for()` ([`server/raceanim.py:781-796`](../server/raceanim.py#L781-L796)).
The animation script verifies that every camera position remains above the
graded site before setting keys and saves once geometry is in place and once
keys are complete ([`demo_yas_animate.py:157-190`](../demo_yas_animate.py#L157-L190)).
Keep those checkpoint boundaries; never reorganise the sequence by changing
the car motion timeline.

## 5. Houdini phase 2: contract before implementation

Houdini is connected and starts as an empty Apprentice scene at
`C:/Users/mabdu/Desktop/atlas/Houdina/atlas.hipnc`. Align it to the Atlas
timeline: 24 fps, frame range `0–END_FRAME` (currently 0–1937). Save and
verify the `.hipnc` file after each material Houdini operation; do not wait for
the end of a multi-step simulation network.

Atlas can provide these source-side inputs without inventing a second motion
system:

- tyre contact and vehicle states: `field_at_time()` and `speed_profile()`
  ([`server/raceanim.py:232-292`](../server/raceanim.py#L232-L292),
  [`server/raceanim.py:404-494`](../server/raceanim.py#L404-L494));
- crash displacement and attitude: `crash_state()`
  ([`server/raceanim.py:523-562`](../server/raceanim.py#L523-L562));
- 1:1 road surfaces with metre-scale UVs: `ribbon_mesh()`
  ([`server/roadway.py:511-640`](../server/roadway.py#L511-L640)).

Phase 2 should export/cache smoke as VDB for `VRayVolumeGrid` and deforming
debris as Alembic for Max. Those are interchange targets, not present Atlas
functions; decide the exact node graph and cache paths only after the first
small Houdini smoke test cooks cleanly. Keep Houdini outputs explicitly
non-commercial until the Apprentice licence is replaced.

## Completion definition for the next pass

1. The asphalt test segment has an intentional racing-line mask, no rejected
   material writes, and no sandpaper/mirror regression.
2. Each tyFlow effect has a reviewed script, an explicit execution decision,
   and an in-host particle verification.
3. Default-height buildings are reported separately from measured and
   `building:levels`-inferred buildings.
4. Shot 11 leads into contact and Shot 12 captures follow-through without a
   below-ground camera or a changed edit order.
5. Houdini has a saved, verifiable checkpoint after each substantive action.
