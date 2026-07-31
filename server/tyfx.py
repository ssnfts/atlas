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
* return a report that says plainly what still has to be authored.

**It does not pretend to have made smoke.** A tyFlow with no events emits
nothing, and a function here that returned "ok" while the render came out
identical would be exactly the silent-wrong-output failure this project exists
to refuse.

To finish the effect, either open the flow in the tyFlow editor and add the
events described in :data:`SMOKE_RECIPE`, or enable raw MaxScript in the host
and drive tyFlow's API directly.
"""

from __future__ import annotations

__all__ = ["SMOKE_RECIPE", "tyre_smoke", "available"]


# What a tyre-smoke flow needs, in the order it goes into the editor. Recorded
# here rather than in a commit message so the next person to open the scene has
# the recipe next to the object it belongs to.
SMOKE_RECIPE = [
    "Event 1 - Birth Flow: continuous, ~600 particles/frame, tied to the "
    "animation range.",
    "Position Object: pick the *_tyres meshes; emit from the rear pair only, "
    "restricted by material ID or a selection set.",
    "Velocity: inherit the emitter's motion at ~0.15, plus 1.5 m/s outward "
    "along the surface normal — smoke leaves the contact patch backwards and "
    "sideways, not upward.",
    "Force: a light Wind at the ERA5 bearing (331 degrees at 7.34 m/s for the "
    "reference hour) so the plume drifts with the real wind.",
    "Particle Physics or Drag: heavy drag so particles stall within ~15 m.",
    "Scale over age + Delete by age (~2.5 s): smoke expands and dissipates.",
    "Shading: tyPreview or export to a VRayVolumeGrid; a VRayLightMtl on "
    "sprites is the cheap alternative.",
]


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

    # Only parameters read back from the live class. Anything the host refuses
    # lands in `rejected` rather than being silently dropped.
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
        # The load-bearing field.
        "complete": False,
        "reason": (
            "the flow exists but has no event graph, so it emits nothing. "
            "tyFlow's operators are not node properties — they live in its own "
            "MaxScript API, and this bridge refuses raw MaxScript unless "
            "ATLAS_ALLOW_MAXSCRIPT=1 is set in the host. Author the events in "
            "the tyFlow editor (see SMOKE_RECIPE) or enable raw MaxScript."
        ),
        "recipe": SMOKE_RECIPE,
    }
