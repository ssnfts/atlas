"""
tyFlow scaffolding — and an honest account of where it stops.

tyFlow is installed on this host (``tyFlow_2027.dlo``) and the bridge can create
one and configure it. What the bridge **cannot** do is author its event graph,
and the event graph is where all the behaviour lives.

The reason is structural rather than a missing feature here. A tyFlow object
exposes 171 scriptable properties and every one of them is a solver or display
setting — thread counts, collision tolerances, cache modes, viewport display.
The operators that make a simulation a simulation (Birth, Position Object,
Force, Spawn, Particle Physics) are nodes inside an event graph, reachable only
through tyFlow's own MaxScript interface. This bridge deliberately refuses raw
MaxScript unless ``ATLAS_ALLOW_MAXSCRIPT=1`` is set inside Max, because raw
MaxScript is arbitrary code execution in the host — so wiring a graph from here
means turning that off, which is the operator's decision and not this module's.

So this module does the part it can do honestly:

* create and name the flow, positioned on the site;
* set the solver parameters that *are* exposed and read them back;
* **generate** a reviewable MaxScript string that builds the full event graph;
* write that string to disk so the operator can read it before it runs;
* execute it over the bridge when explicitly asked to.

**It does not pretend to have made smoke.** A tyFlow with no events emits
nothing, and a function here that returned "ok" while the render came out
identical would be exactly the silent-wrong-output failure this project exists
to refuse.

Route: generate -> write to ``out/tyfx_smoke.ms`` -> review -> run.
``ATLAS_ALLOW_MAXSCRIPT=1`` must be set in Max's environment before launch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = [
    "SMOKE_RECIPE",
    "tyre_smoke",
    "available",
    "generate_smoke_script",
    "write_smoke_script",
    "run_smoke_script",
    "particle_count",
    "smoke_verify",
    "contact_frame_for",
    "generate_debris_script",
    "write_debris_script",
    "run_debris_script",
    "generate_sparks_script",
    "write_sparks_script",
    "run_sparks_script",
]


# What a tyre-smoke flow needs, in the order it goes into the editor. Recorded
# here rather than in a commit message so the next person to open the scene has
# the recipe next to the object it belongs to.
SMOKE_RECIPE = [
    "One Birth event per contiguous active speed-gate range; each event emits "
    "80 particles/frame only while its range is active.",
    "Position Object: pick the *_tyres meshes; emit from the rear pair only, "
    "restricted by material ID or a selection set.",
    "Velocity: inherit the emitter's motion at ~0.15, plus 1.5 m/s outward "
    "along the surface normal — smoke leaves the contact patch backwards and "
    "sideways, not upward.",
    "Force: a light Wind at the ERA5 bearing (331 degrees at 7.34 m/s for the "
    "reference hour) so the plume drifts with the real wind.",
    "Force: a light wind matching the recorded bearing and speed.",
    "Scale: starts at 0.2 m; Time Test sends particles to Delete after "
    "60 +/- 8 frames (about 2.5 seconds at 24 fps).",
    "Shading: tyPreview or export to a VRayVolumeGrid; a VRayLightMtl on "
    "sprites is the cheap alternative.",
]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _git_hash() -> str:
    """Short git hash for embedding in script headers, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _script_header(title: str, params: dict) -> str:
    """Comment block prepended to every generated MaxScript file."""
    lines = [
        f"-- Atlas generated MaxScript: {title}",
        f"-- Git:    {_git_hash()}",
        "--",
        "-- Parameters:",
    ]
    for k, v in params.items():
        lines.append(f"--   {k}: {v}")
    lines += [
        "--",
        "-- Review before running. Execute from the MaxScript listener or",
        "-- via bridge.maxscript() with ATLAS_ALLOW_MAXSCRIPT=1 set in Max.",
        "",
    ]
    return "\n".join(lines)


# ── Existing stub (kept honest) ───────────────────────────────────────────────

def available(bridge) -> bool:
    """True when this host has tyFlow installed."""
    try:
        bridge.properties(cls="tyFlow")
        return True
    except Exception:
        return False


def tyre_smoke(bridge, emitter_nodes, *, site_z: float = 0.0,
               end_frame: int = 240, name: str = "Atlas_TyreSmoke") -> dict:
    """
    Create the tyFlow object for tyre smoke and configure what is reachable.

    Returns a report whose ``complete`` field is **False**: the flow exists and
    is configured, and it will emit nothing until its event graph is authored.
    That is stated rather than implied, because a caller that reads "ok" and
    renders an identical frame has been misled by its own tooling.

    Use :func:`write_smoke_script` + :func:`run_smoke_script` to build the
    event graph via MaxScript once ATLAS_ALLOW_MAXSCRIPT=1 is set.
    """
    if not available(bridge):
        return {"created": False, "complete": False,
                "reason": "tyFlow is not installed on this host"}

    existing = bridge.call("getNodeByName", name)
    if existing is None:
        created = bridge.call("tyFlow")
        if not isinstance(created, dict) or "__node__" not in created:
            return {"created": False, "complete": False,
                    "reason": f"tyFlow() did not return a node (got {created!r})"}
        bridge.node_set(created["__node__"], "name", name)

    bridge.node_set(name, "pos", {"__point3__": [0.0, 0.0, site_z]})

    applied, rejected = {}, {}
    for prop, value in (("autoThreads", True), ("ShowIcon", True)):
        try:
            bridge.node_set(name, prop, value)
            applied[prop] = bridge.node_get(name, prop)
        except Exception as exc:
            rejected[prop] = str(exc)

    return {
        "created": True,
        "node": name,
        "emitters": list(emitter_nodes),
        "applied": applied,
        "rejected": rejected,
        "end_frame": end_frame,
        "complete": False,
        "reason": (
            "the flow exists but has no event graph, so it emits nothing. "
            "Call write_smoke_script() then run_smoke_script() to build the "
            "graph via MaxScript (ATLAS_ALLOW_MAXSCRIPT=1 required)."
        ),
        "recipe": SMOKE_RECIPE,
    }


# ── Tyre smoke ────────────────────────────────────────────────────────────────

def generate_smoke_script(
    flow_name: str,
    emitter_nodes: list[str],
    *,
    speed_gate_frames: list[bool],
    wind_bearing_deg: float,
    wind_speed_ms: float,
    site_z: float,
    end_frame: int,
) -> str:
    """
    Return a MaxScript string that builds the tyre-smoke event graph.

    ``speed_gate_frames`` is a per-frame bool list: True means emit at that
    frame, False means the car is below the speed threshold and does not emit.
    Computed by the caller from :func:`raceanim.speed_profile` so this function
    stays pure (no spine dependency).

    The script is self-documenting: it embeds the parameter values so the file
    on disk explains itself without reference to the code that wrote it.
    """
    # Wind direction in MaxScript is a Point3 unit vector. Wind comes FROM
    # bearing, so the force direction is the opposite.
    import math
    rad = math.radians(wind_bearing_deg + 180.0)   # direction wind blows TO
    wx = round(math.sin(rad), 6)
    wy = round(math.cos(rad), 6)

    # Emitter node list as a MaxScript array of node references.
    node_refs = ", ".join(f'getNodeByName "{n}"' for n in emitter_nodes)

    # Per-frame emission gate as a MaxScript BooleanArray. tyFlow reads this
    # via a Custom Properties operator binding; the array is stored on the flow
    # node so it survives a scene save.
    #
    # **Wrapped, and that is the point of the file.** This script exists to be
    # read by a person before it executes arbitrary code in their 3ds Max, and
    # nearly two thousand booleans on one 4,000-character line is not readable
    # by anyone. Run-length encoding would be shorter still, but it would also
    # be a second format to trust; wrapping keeps the array literal exactly what
    # it claims to be and merely makes it fit on a screen.
    #
    # 20 per line, with the frame number of the row in a trailing comment, so a
    # reviewer can find "what is the gate doing around the crash at frame 1388"
    # without counting commas.
    gate_lines = []
    for start in range(0, len(speed_gate_frames), 20):
        row = speed_gate_frames[start:start + 20]
        vals = ", ".join("true" if g else "false" for g in row)
        comma = "," if start + 20 < len(speed_gate_frames) else ""
        gate_lines.append(f"    {vals}{comma}    -- frames {start}-{start + len(row) - 1}")
    gate_vals = "\n" + "\n".join(gate_lines) + "\n"

    active_ranges: list[tuple[int, int]] = []
    start: int | None = None
    for frame, active in enumerate(speed_gate_frames):
        if active and start is None:
            start = frame
        elif not active and start is not None:
            active_ranges.append((start, frame - 1))
            start = None
    if start is not None:
        active_ranges.append((start, len(speed_gate_frames) - 1))

    event_calls = "\n".join(
        f'atlasMakeSmokeEvent tf "Smoke_Birth_{i:02d}" {start_frame} {end_frame}'
        for i, (start_frame, end_frame) in enumerate(active_ranges, start=1)
    )

    header = _script_header("Tyre Smoke", {
        "flow": flow_name,
        "emitters": len(emitter_nodes),
        "wind": f"{wind_bearing_deg}deg at {wind_speed_ms} m/s",
        "end_frame": end_frame,
        "speed_gate_frames": f"{sum(speed_gate_frames)} of {len(speed_gate_frames)} active",
    })

    script = header + f"""\
(
-- ── 1. Ensure the tyFlow node exists ─────────────────────────────────────
local tf = getNodeByName "{flow_name}"
if tf != undefined do delete tf
tf = tyFlow()
tf.name = "{flow_name}"
tf.pos = [0, 0, {site_z}]
tf.autoThreads = true
tf.ShowIcon = true

-- ── 2. Store the speed gate for scene provenance ──────────────────────────
-- The matching Birth ranges below are the executable binding.
local gateArr = #({gate_vals})
setUserProp tf "atlas_speed_gate" (gateArr as string)

-- ── 3. Build one event per active speed-gate range ────────────────────────
local emitterNodes = #({node_refs})

fn atlasMakeSmokeEvent tf eventName firstFrame lastFrame = (
    local ev = tf.addEvent()
    ev.setName eventName

    local birth = ev.addOperator "Birth" -1
    birth.setName "Birth"
    birth.birthMode = 1  -- per frame; mode 0 treats birthPerFrame as a total
    birth.BirthStart = firstFrame
    birth.birthEndEnable = true
    birth.BirthEnd = lastFrame
    birth.birthPerFrame = 80

    local position = ev.addOperator "Position Object" -1
    position.setName "Position"
    position.objectList = emitterNodes
    position.inheritMotion = true
    position.inheritMotionMultiplier = 0.15

    local speed = ev.addOperator "Speed" -1
    speed.setName "Speed"
    speed.magnitude = 1.5
    speed.magnitudeVariation = 0.5

    local force = ev.addOperator "Force" -1
    force.setName "Wind"
    force.windX = {wx}
    force.windY = {wy}
    force.windZ = 0.0
    force.windStrength = {round(wind_speed_ms * 0.05, 4)}

    local scale = ev.addOperator "Scale" -1
    scale.setName "Initial particle scale"
    scale.scaleX = 0.2
    scale.scaleY = 0.2
    scale.scaleZ = 0.2

    local age = ev.addOperator "Time Test" -1
    age.setName "Delete after 2.5 seconds"
    age.mode = 1
    age.Condition = 4
    age.value = 60
    age.variation = 8

    local death = tf.addEvent()
    death.setName (eventName + "_Delete")
    local deleteOp = death.addOperator "Delete" -1
    deleteOp.setName "Delete"
    age.connect death
)

{event_calls}

-- ── 4. Reset the solver after graph construction ──────────────────────────
tf.reset_simulation()

print ("Atlas: tyre smoke event graph built on " + tf.name)
print ("  Emitters: {len(emitter_nodes)}")
print ("  Wind: {wind_bearing_deg} deg at {wind_speed_ms} m/s")
print ("  Active frames: {sum(speed_gate_frames)} of {len(speed_gate_frames)}")
)
"""
    return script


def write_smoke_script(
    path: str | Path,
    emitter_nodes: list[str],
    spine,
    *,
    wind_bearing_deg: float = 331.0,
    wind_speed_ms: float = 7.34,
    site_z: float = 0.0,
    end_frame: int = 2400,
    speed_threshold_ms: float = 55.56,  # 200 km/h
    flow_name: str = "Atlas_TyreSmoke",
) -> dict:
    """
    Write the tyre-smoke MaxScript to disk and return a summary dict.

    ``spine`` is the list of (x, y) circuit vertices used to compute the
    per-frame speed gate from :func:`raceanim.speed_profile`.

    Returns ``{"path", "lines", "summary", "params"}`` so a caller can
    report what will happen before running it.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import raceanim

    import math

    speeds = raceanim.speed_profile(spine)
    lap_s = max(raceanim.lap_time(spine), 1e-3)
    n_frames = int(end_frame) + 1
    n_verts = len(speeds)

    # Frame -> time -> distance -> vertex.
    #
    # The previous version went frame -> *vertex index* linearly, and that is
    # wrong twice over. Spine vertices are 2 m to 266 m apart here, so index is
    # not proportional to distance; and frames are evenly spaced in *time*, so
    # they are not proportional to distance either once the car slows for the
    # corners. The gate was reading the speed of an unrelated part of the
    # circuit. Same class of error as measuring curvature across adjacent
    # vertices: treating unevenly-sampled data as if it were even.
    cumulative = [0.0]
    for i in range(n_verts):
        cumulative.append(
            cumulative[-1] + math.dist(spine[i], spine[(i + 1) % n_verts]))

    def vertex_at(distance: float) -> int:
        lo, hi = 0, n_verts - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid + 1] < distance:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # The crash window. A speed gate alone cannot produce crash smoke: a car
    # that is spinning is *slow*, and a slow car fails a >200 km/h test —
    # while being the single biggest smoke source on the circuit. Measured on
    # the generated script, the gate was false through the entire incident.
    contact = contact_frame_for(spine, raceanim.CRASH,
                                fps=int(round(n_frames / lap_s)))
    crash_from = contact - int(raceanim.CRASH.tell_s * n_frames / lap_s)
    crash_to = contact + int(raceanim.CRASH.settle_s * n_frames / lap_s)

    gate: list[bool] = []
    for f in range(n_frames):
        seconds = (f / max(end_frame, 1)) * lap_s
        distance = raceanim.distance_at(spine, seconds)
        fast = speeds[vertex_at(distance)] >= speed_threshold_ms
        gate.append(fast or crash_from <= f <= crash_to)

    script = generate_smoke_script(
        flow_name, emitter_nodes,
        speed_gate_frames=gate,
        wind_bearing_deg=wind_bearing_deg,
        wind_speed_ms=wind_speed_ms,
        site_z=site_z,
        end_frame=end_frame,
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")

    active = sum(gate)
    summary = (
        f"Tyre smoke: {len(emitter_nodes)} emitter(s), "
        f"speed gate >{speed_threshold_ms * 3.6:.0f} km/h "
        f"({active}/{n_frames} frames active), "
        f"wind {wind_bearing_deg:.0f}° at {wind_speed_ms:.2f} m/s, "
        f"end frame {end_frame}."
    )

    return {
        "path": str(out),
        "lines": len(script.splitlines()),
        "summary": summary,
        "params": {
            "flow_name": flow_name,
            "emitter_count": len(emitter_nodes),
            "speed_threshold_kmh": round(speed_threshold_ms * 3.6, 1),
            "active_frames": active,
            "total_frames": n_frames,
            "wind_bearing_deg": wind_bearing_deg,
            "wind_speed_ms": wind_speed_ms,
            "site_z": site_z,
            "end_frame": end_frame,
        },
    }


def run_smoke_script(bridge, path: str | Path) -> dict:
    """
    Read the smoke MaxScript from disk and execute it via the bridge.

    Raises ``FileNotFoundError`` if the script does not exist — the caller must
    call :func:`write_smoke_script` and review the file first.

    Raises ``RuntimeError`` with a clear message if the bridge rejects the call
    because ``ATLAS_ALLOW_MAXSCRIPT=0``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Smoke script not found at {p}. "
            "Call write_smoke_script() first, review the file, then run."
        )
    code = p.read_text(encoding="utf-8")
    try:
        result = bridge.maxscript(code)
    except Exception as exc:
        msg = str(exc)
        if "disabled" in msg.lower() and "maxscript" in msg.lower():
            raise RuntimeError(
                "ATLAS_ALLOW_MAXSCRIPT is disabled in the bridge. "
                "Set ATLAS_ALLOW_MAXSCRIPT=1 in the Windows environment "
                "before launching 3ds Max, then restart the bridge."
            ) from exc
        raise
    return {"executed": True, "script": str(p), "result": result}


# ── Verification ──────────────────────────────────────────────────────────────

def particle_count(bridge, flow_name: str, frame: int) -> int:
    """
    Read the particle count from a tyFlow at a given frame via MaxScript.

    Returns 0 if the flow has no events or does not exist. Uses MaxScript
    because tyFlow event state is not exposed via pymxs properties.
    """
    code = f"""\
(
local tf = getNodeByName "{flow_name}"
if tf == undefined then 0
else (
    tf.updateParticles {frame}
    tf.numParticles()
)
)
"""
    result = bridge.maxscript(code)
    try:
        return int(result)
    except (TypeError, ValueError):
        return 0


def smoke_verify(
    bridge,
    flow_name: str,
    *,
    check_frames: list[int],
    out_dir: str | Path,
) -> dict:
    """
    Verify the smoke flow is emitting by measurement.

    For each frame in ``check_frames``: reads particle count via MaxScript and
    renders a 320x180 masked frame (flat white particles, black background) to
    ``out_dir/verify_smoke_f{frame:04d}.png``, then counts non-black pixels
    using Pillow.

    Returns ``{"ok": bool, "frames": [...], "reason": str}``.

    A flow with no events emits nothing silently — this catches it before a
    full render is wasted.
    """
    try:
        from PIL import Image
    except ImportError:
        return {"ok": False, "frames": [], "reason": "pillow not installed"}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame_results = []
    for f in check_frames:
        count = particle_count(bridge, flow_name, f)
        img_path = str(out / f"verify_smoke_f{f:04d}.png")

        # Assign flat white VRayLightMtl to the particles, render small.
        setup_code = f"""\
local tf = getNodeByName "{flow_name}"
if tf != undefined then (
    local m = VRayLightMtl()
    m.color = color 255 255 255
    tf.material = m
)
"""
        try:
            bridge.maxscript(setup_code)
            bridge.render(
                img_path,
                width=320, height=180,
                expect_renderer="V_Ray",
                timeout=120.0,
            )
            img = Image.open(img_path).convert("RGB")
            pixels = list(img.getdata())
            non_black = sum(1 for r, g, b in pixels if r + g + b > 30)
        except Exception as exc:
            non_black = -1
            count = count if count >= 0 else 0

        frame_results.append({
            "frame": f,
            "particle_count": count,
            "non_black_pixels": non_black,
        })

    # The flow is working if at least one check frame has particles AND pixels.
    working = any(
        r["particle_count"] > 0 and r["non_black_pixels"] > 0
        for r in frame_results
    )
    if working:
        reason = "ok — particles and non-black pixels found in at least one frame"
    elif all(r["particle_count"] == 0 for r in frame_results):
        reason = (
            f"FAIL — zero particles at all checked frames {check_frames}. "
            "The event graph may not have been built, or the speed gate is "
            "blocking emission at these frames."
        )
    else:
        reason = (
            "FAIL — particles exist but no non-black pixels. "
            "Check that the renderer is V-Ray and the material was applied."
        )

    return {"ok": working, "frames": frame_results, "reason": reason}


# ── Crash: shared helper ──────────────────────────────────────────────────────

def contact_frame_for(spine, spec, fps: int = 24) -> int:
    """
    The timeline frame at which the spinner reaches ``spec.at_distance_m``.

    Walks the speed profile rather than dividing by a mean, for the same reason
    :func:`raceanim.distance_at` integrates rather than multiplies: the car
    lingers in the slow corners and covers the straights quickly, and the lap
    time changes accordingly.

    Returns an integer frame number. The result is the single number that ties
    the debris and sparks simulations to the animation — if this is wrong,
    everything starts at the wrong moment.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import raceanim
    import math

    n = len(spine)
    speeds = raceanim.speed_profile(spine)
    seg = [math.dist(spine[i], spine[(i + 1) % n]) for i in range(n)]

    elapsed = 0.0
    covered = 0.0
    for i in range(n):
        j = (i + 1) % n
        mean_v = max(0.5 * (speeds[i] + speeds[j]), 1e-3)
        dt = seg[i] / mean_v
        if covered + seg[i] >= spec.at_distance_m:
            frac = (spec.at_distance_m - covered) / max(seg[i], 1e-9)
            elapsed += dt * frac
            break
        covered += seg[i]
        elapsed += dt

    return int(round(elapsed * fps))


# ── Crash debris ──────────────────────────────────────────────────────────────

def generate_debris_script(
    flow_name: str,
    wing_nodes: list[str],
    *,
    contact_frame: int,
    car_speed_ms: float,
    yaw_rate_degs: float,
    site_z: float,
    end_frame: int,
) -> str:
    """
    Return MaxScript that builds the crash-debris event graph.

    One event creates a finite burst of real car-mesh elements at impact,
    gives them PhysX rigidbodies, then collides them against a wide site plane.
    It deliberately does not claim to perform Voronoi fracturing: Atlas's car
    proxy already contains discrete wings, plates, and body elements, which are
    the valid shard source until a fracture operator is host-verified.
    """
    if not wing_nodes:
        raise ValueError("wing_nodes must contain at least one source mesh")
    node_refs = ", ".join(f'getNodeByName "{n}"' for n in wing_nodes)
    scatter_speed = round(min(16.0, max(6.0, car_speed_ms * 0.14)), 2)
    ground_name = f"{flow_name}_Ground"

    header = _script_header("Crash Debris", {
        "flow": flow_name,
        "wing_nodes": len(wing_nodes),
        "contact_frame": contact_frame,
        "car_speed_ms": car_speed_ms,
        "yaw_rate_degs": yaw_rate_degs,
        "site_z": site_z,
        "end_frame": end_frame,
    })

    script = header + f"""\
(
-- ── 1. tyFlow node ───────────────────────────────────────────────────────
local tf = getNodeByName "{flow_name}"
if tf != undefined do delete tf
tf = tyFlow()
tf.name = "{flow_name}"
tf.pos = [0, 0, {site_z}]

-- ── 2. Birth real source-mesh elements at the impact frame ────────────────
local ev1 = tf.addEvent()
ev1.setName "Debris_Birth"
local wingNodes = #({node_refs})
local birth = ev1.addOperator "Birth" -1
birth.setName "Impact burst"
birth.birthMode = 0
birth.BirthStart = {contact_frame}
birth.birthEndEnable = true
birth.BirthEnd = {contact_frame}
birth.birthTotal = 30

local position = ev1.addOperator "Position Object" -1
position.setName "Source mesh surface"
position.objectList = wingNodes

local shape = ev1.addOperator "Shape" -1
shape.instancedGeo_tab[1] = wingNodes[1]
shape.meshSplitElements_tab[1] = true

local physics = ev1.addOperator "PhysX Shape" -1
physics.hullMode = 2
physics.restitution = 0.12
physics.staticFriction = 0.78
physics.dynamicFriction = 0.72
physics.massOverride = true
physics.mass = 0.2
physics.linearDamping = 0.2

local oldGround = getNodeByName "{ground_name}"
if oldGround != undefined do delete oldGround
local groundPlane = Plane width:10000 length:10000
groundPlane.name = "{ground_name}"
groundPlane.pos = [0, 0, {site_z}]
groundPlane.isHidden = true

local collide = ev1.addOperator "PhysX Collision" -1
collide.colliderList = #(groundPlane)
collide.hullMode = 2

local speed = ev1.addOperator "Speed" -1
speed.setName "Impact scatter"
speed.magnitude = {scatter_speed}
speed.magnitudeVariation = 2.0

local force = ev1.addOperator "Force" -1
force.setName "Gravity"
force.gravityStrength = -1

local mesh = ev1.addOperator "Mesh" -1
mesh.setName "Render mesh"

-- ── 3. Reset the solver after graph construction ──────────────────────────
tf.reset_simulation()

print ("Atlas: crash debris event graph built on " + tf.name)
print ("  Contact frame: {contact_frame}")
print ("  Source mesh nodes: {len(wing_nodes)}")
)
"""
    return script


def write_debris_script(
    path: str | Path,
    wing_nodes: list[str],
    spine,
    spec,
    *,
    site_z: float = 0.0,
    end_frame: int = 2400,
    fps: int = 24,
    flow_name: str = "Atlas_CrashDebris",
) -> dict:
    """Write the crash-debris MaxScript to disk and return a summary dict."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import raceanim

    cf = contact_frame_for(spine, spec, fps)
    speeds = raceanim.speed_profile(spine)

    # Car speed at the crash point (nearest spine vertex to crash distance).
    import math
    n = len(spine)
    seg = [math.dist(spine[i], spine[(i + 1) % n]) for i in range(n)]
    covered = 0.0
    crash_speed = speeds[0]
    for i in range(n):
        if covered + seg[i] >= spec.at_distance_m:
            crash_speed = speeds[i]
            break
        covered += seg[i]

    # Yaw rate from the crash spec: spinner accumulates spin_degrees over settle_s.
    yaw_rate = spec.spin_degrees / max(spec.settle_s, 1e-3)

    script = generate_debris_script(
        flow_name, wing_nodes,
        contact_frame=cf,
        car_speed_ms=crash_speed,
        yaw_rate_degs=yaw_rate,
        site_z=site_z,
        end_frame=end_frame,
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")

    return {
        "path": str(out),
        "lines": len(script.splitlines()),
        "summary": (
            f"Crash debris: {len(wing_nodes)} source mesh node(s), "
            f"contact frame {cf}, "
            f"car speed {crash_speed:.1f} m/s ({crash_speed * 3.6:.0f} km/h), "
            f"site z {site_z}."
        ),
        "params": {
            "flow_name": flow_name,
            "wing_nodes": wing_nodes,
            "contact_frame": cf,
            "car_speed_ms": round(crash_speed, 2),
            "yaw_rate_degs": round(yaw_rate, 1),
            "site_z": site_z,
            "end_frame": end_frame,
        },
    }


def run_debris_script(bridge, path: str | Path) -> dict:
    """Execute the crash-debris MaxScript from disk via the bridge."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Debris script not found at {p}. "
            "Call write_debris_script() first, review the file, then run."
        )
    code = p.read_text(encoding="utf-8")
    try:
        result = bridge.maxscript(code)
    except Exception as exc:
        msg = str(exc)
        if "disabled" in msg.lower() and "maxscript" in msg.lower():
            raise RuntimeError(
                "ATLAS_ALLOW_MAXSCRIPT is disabled. "
                "Set ATLAS_ALLOW_MAXSCRIPT=1 before launching 3ds Max."
            ) from exc
        raise
    return {"executed": True, "script": str(p), "result": result}


# ── Sparks ────────────────────────────────────────────────────────────────────

def generate_sparks_script(
    flow_name: str,
    floor_nodes: list[str],
    *,
    contact_frame: int,
    car_speed_ms: float,
    site_z: float,
    end_frame: int,
    contact_position: tuple[float, float] = (0.0, 0.0),
) -> str:
    """
    Return MaxScript that builds the sparks event graph.

    Titanium skid-block sparks: burst over ~0.4 s from contact, speed scaled
    to the impact, gravity on, small and bright. VRayLightMtl assigned
    manually (noted in a comment) — no actual lights, no shadow cost.

    A small hidden sphere at ``contact_position`` provides a stable source.
    ``floor_nodes`` is retained for a future PhysX collision pass; it is not
    used as an emitter because that would scatter sparks across a whole slab.
    """
    burst_end = contact_frame + 10  # 10 frames @ 24 fps ≈ 0.42 s
    scatter_speed = round(min(12.0, max(5.0, car_speed_ms * 0.1)), 2)

    contact_x, contact_y = contact_position
    source_name = f"{flow_name}_Source"

    header = _script_header("Crash Sparks", {
        "flow": flow_name,
        "floor_nodes": len(floor_nodes),
        "contact_frame": contact_frame,
        "burst_frames": f"{contact_frame}-{burst_end}",
        "car_speed_ms": car_speed_ms,
    })

    script = header + f"""\
(
-- ── Sparks event graph ───────────────────────────────────────────────────
-- NOTE: Assign a VRayLightMtl to the particles manually for the glow effect.
-- No actual lights are used — this keeps the shadow cost at zero.

local tf = getNodeByName "{flow_name}"
if tf != undefined do delete tf
tf = tyFlow()
tf.name = "{flow_name}"
tf.pos = [0, 0, {site_z}]

local oldSource = getNodeByName "{source_name}"
if oldSource != undefined do delete oldSource
local source = Sphere radius:0.02 segments:8
source.name = "{source_name}"
source.pos = [{contact_x}, {contact_y}, {site_z}]
source.isHidden = true

local ev1 = tf.addEvent()
ev1.setName "Sparks_Birth"

-- Birth burst over contact window (~0.4 s)
local birth = ev1.addOperator "Birth" -1
birth.setName "Birth"
birth.birthMode = 1
birth.BirthStart = {contact_frame}
birth.birthEndEnable = true
birth.BirthEnd = {burst_end}
birth.birthPerFrame = 60

local position = ev1.addOperator "Position Object" -1
position.setName "Crash contact"
position.objectList = #(source)
-- Velocity: forward + upward scatter with wide spread
local speed = ev1.addOperator "Speed" -1
speed.setName "Scatter"
speed.magnitude = {scatter_speed}
speed.magnitudeVariation = 3.0

-- Gravity
local force = ev1.addOperator "Force" -1
force.setName "Gravity"
force.gravityStrength = -1

-- Scale: small and constant
local scale = ev1.addOperator "Scale" -1
scale.setName "Spark size"
scale.scaleX = 0.03
scale.scaleY = 0.03
scale.scaleZ = 0.03

-- Delete after 0.3 s
local age = ev1.addOperator "Time Test" -1
age.setName "Delete after 0.33 seconds"
age.mode = 1
age.Condition = 4
age.value = 8
age.variation = 2

local death = tf.addEvent()
death.setName "Sparks_Delete"
local deleteOp = death.addOperator "Delete" -1
deleteOp.setName "Delete"
age.connect death

tf.reset_simulation()

print ("Atlas: sparks event graph built on " + tf.name)
print ("  Contact frame: {contact_frame}, burst to frame {burst_end}")
print ("  Assign VRayLightMtl to the particles for the glow effect.")
)
"""
    return script


def write_sparks_script(
    path: str | Path,
    floor_nodes: list[str],
    spine,
    spec,
    *,
    site_z: float = 0.0,
    end_frame: int = 2400,
    fps: int = 24,
    flow_name: str = "Atlas_CrashSparks",
) -> dict:
    """Write the sparks MaxScript to disk and return a summary dict."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import raceanim
    import math

    cf = contact_frame_for(spine, spec, fps)

    n = len(spine)
    speeds = raceanim.speed_profile(spine)
    seg = [math.dist(spine[i], spine[(i + 1) % n]) for i in range(n)]
    covered = 0.0
    crash_speed = speeds[0]
    for i in range(n):
        if covered + seg[i] >= spec.at_distance_m:
            crash_speed = speeds[i]
            break
        covered += seg[i]
    crash_x, crash_y, _ = raceanim.point_at(spine, spec.at_distance_m)

    script = generate_sparks_script(
        flow_name, floor_nodes,
        contact_frame=cf,
        car_speed_ms=crash_speed,
        site_z=site_z,
        end_frame=end_frame,
        contact_position=(crash_x, crash_y),
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")

    burst_end = cf + 10
    return {
        "path": str(out),
        "lines": len(script.splitlines()),
        "summary": (
            f"Sparks: contact-point source ({len(floor_nodes)} floor node(s) reserved for collision), "
            f"burst frames {cf}-{burst_end} ({(burst_end - cf) / fps:.2f}s), "
            f"car speed {crash_speed:.1f} m/s, site z {site_z}."
        ),
        "params": {
            "flow_name": flow_name,
            "floor_nodes": floor_nodes,
            "contact_frame": cf,
            "burst_end_frame": burst_end,
            "car_speed_ms": round(crash_speed, 2),
            "contact_position": [round(crash_x, 3), round(crash_y, 3)],
            "site_z": site_z,
            "end_frame": end_frame,
        },
    }


def run_sparks_script(bridge, path: str | Path) -> dict:
    """Execute the sparks MaxScript from disk via the bridge."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Sparks script not found at {p}. "
            "Call write_sparks_script() first, review the file, then run."
        )
    code = p.read_text(encoding="utf-8")
    try:
        result = bridge.maxscript(code)
    except Exception as exc:
        msg = str(exc)
        if "disabled" in msg.lower() and "maxscript" in msg.lower():
            raise RuntimeError(
                "ATLAS_ALLOW_MAXSCRIPT is disabled. "
                "Set ATLAS_ALLOW_MAXSCRIPT=1 before launching 3ds Max."
            ) from exc
        raise
    return {"executed": True, "script": str(p), "result": result}
