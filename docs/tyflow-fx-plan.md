# tyFlow: what to build, and what has to change first

Written 31 July 2026, after wiring the crash into `server/raceanim.py`.

## The blocker, stated first

tyFlow is installed (`tyFlow_2027.dlo`). The bridge can create a flow and set
its solver parameters, and **that is all it can do**, because all 171 of
tyFlow's scriptable properties are solver and display settings — thread counts,
collision tolerances, cache modes, viewport display. The operators that make a
simulation a simulation (Birth, Position Object, Force, Spawn, Voronoi Fracture,
Particle Bind) live in an event graph reachable only through tyFlow's own
MaxScript interface.

This bridge refuses raw MaxScript unless `ATLAS_ALLOW_MAXSCRIPT=1` is set
**inside Max**, because raw MaxScript is arbitrary code execution in the host.
That is the operator's decision, not this project's.

**So there are two routes, and they should be chosen deliberately:**

| Route | Cost | What it buys |
|---|---|---|
| **A — author in the tyFlow editor by hand** | one-off manual setup, saved into the scene file | nothing changes in the codebase; the scene stops being reproducible from source |
| **B — enable raw MaxScript and drive tyFlow's API** | a security decision by whoever runs Max, plus a `tyfx` module that emits graph-building script | the whole scene stays reproducible from a date, a place and a git hash |

Route B is the one consistent with everything else here — Atlas's entire premise
is that a scene is *derived*, not hand-made. But it should be taken with the
`ATLAS_ALLOW_MAXSCRIPT` flag understood rather than flipped quietly, and the
generated script should be written to disk and reviewable before it runs.

## What the animation already hands to tyFlow

The crash is keyframed, deterministic and queryable, which is exactly what a
simulation needs as input. From `raceanim`:

- `CRASH` — a `CrashSpec` pinning the incident to **1980 m along the lap**, not
  to a frame, so retiming does not move it.
- `crash_state(spec, t, heading)` — lateral displacement, yaw, roll, pitch and
  height at any moment relative to contact. Impact is at `t = 0`, anticipation
  runs from `-0.6 s`, follow-through settles over `3.4 s`.
- `speed_profile(spine)` — the velocity at every point, so a particle's inherited
  motion is a real number rather than a guess. At the crash site the field is
  doing roughly 210 km/h.
- The tyre meshes (`car_NN_tyres`) are already separate objects from bodywork
  and rims, so they can be emitters or fracture sources independently.

Anything tyFlow does should read these rather than re-deriving them, so the
effects and the animation cannot drift apart.

## Effects, in the order they earn their place

### 1. Tyre smoke — highest value, lowest risk

The one effect the whole sequence needs even without a crash: a locked wheel
under braking, and a plume off the rears out of the slow corners.

```
Event 01  Birth Flow          continuous, ~600/frame, gated to the animation range
          Position Object     the *_tyres meshes; rear pair only, by material ID
          Velocity            inherit emitter motion at 0.15
                              + 1.5 m/s along the surface normal
                              (smoke leaves the contact patch backwards and
                               sideways, never upward)
          Force > Wind        ERA5 for the reference hour: 331 degrees at
                              7.34 m/s. Use the real number; it is already
                              fetched by `weather.fetch_observation`.
          Drag                heavy; particles should stall within ~15 m
          Scale               grows ~4x over life
          Age Test > Delete   2.5 s
Shading   tyPreview, or export to VRayVolumeGrid for real scattering
```

Gate emission on speed and slip, not on time: smoke should appear in the braking
zones the speed profile already identifies, which means an operator driven by
`speed_profile` rather than a hand-keyed on/off.

### 2. Crash debris — the reason the crash shots were framed as they are

Both crash shots (`11_crash_wide`, `12_crash_tight`) bracket contact so the
anticipation is on screen. Debris has to start at `t = 0`, not before.

```
Event 01  Birth              burst, ~180 particles, at the contact frame
          Position Object    the front wing and endplate faces of the spinner
          Shape              fragments from a Voronoi fracture of the wing
          Velocity           inherit 0.7 of car velocity (~40 m/s)
                             + 6 m/s radial scatter
Event 02  PhysX Shape        convex hull per fragment
          PhysX Collision    against the graded site plane and the kerbs
          Bounce/Friction    carbon on asphalt: low bounce, high friction
          Spin               inherited from the impact yaw rate
Event 03  Age > 6 s          freeze, do not delete; debris stays on track
```

Fracture the **front wing only** for a first pass. It is the part that actually
detaches, it is small, and a Voronoi on the whole monocoque is both wrong and
expensive.

### 3. Sparks — cheap, and sells the contact

Titanium skid blocks throwing sparks where the floor touches down. Emit on
contact for ~0.4 s, tiny, bright, short-lived, heavy drag, gravity on.
A `VRayLightMtl` on the particles rather than actual lights.

### 4. Marbles and dust — the ambient layer

Rubber marbles off-line and dust lifted by the cars passing. Static scatter plus
a light drift; this is the background layer that stops the surface reading as
clean CG.

## Sequencing

Build in the order above. Smoke first because it is needed in every shot and its
failure mode is visible immediately; debris second because it is bounded to two
shots; sparks and dust last because they are polish and can be cut.

Do **not** simulate the whole lap. Cache the crash window only — roughly
frames 1760 to 1860 at 24 fps — and leave the rest of the sequence particle-free.
A tyCache over 2400 frames of 20 cars is a large amount of disk for footage that
mostly shows nothing.

## How to know it worked

The same discipline as everything else here: a tyFlow with no events emits
nothing, silently, and a render that looks identical is the expected failure.
So verify by **measurement, not by eye**:

- particle count per frame, read back off the flow, non-zero in the window;
- a masked render — flag the particles a flat colour and confirm the pixels move
  between frames;
- debris resting positions inside the track bounds, not under the ground sheet
  or 400 m away.

`tyfx.tyre_smoke()` currently returns `complete=False` with the reason. Whatever
replaces it must keep that honesty: a function that returns "ok" while the render
is unchanged is the exact failure this project exists to refuse.
