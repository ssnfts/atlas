# Atlas

Give it a **date, a time and a location**, and it assembles a georeferenced,
physically-lit scene in **3ds Max** with **V-Ray** — the sun at its true angle
for that moment, sky conditions from real historical weather, and a camera
exposure derived from the sun's altitude.

Built for entertainment work: previs, plate matching, sun-path studies and
virtual production layout, where "what did this place actually look like at
16:00 on 12 October" is a question with a correct answer.

```
date + time + location
        │
        ├─ solar position  (NOAA algorithm, verified against NREL SPA)
        ├─ weather         (ERA5 reanalysis + CAMS aerosols)
        └─ scene frame     (local ENU metres, +Y = true north)
                │
                ▼
        3ds Max + V-Ray
```

---

## Status

The lighting pipeline works end to end and is verified against rendered output.

| Component | State |
|---|---|
| Solar position | working — agrees with pvlib's NREL SPA to 0.004° |
| Timezone / DST | working — rejects nonexistent times rather than shifting them |
| Scene frame | working — verified against WGS84 geodesics to 0.1% |
| 3ds Max bridge | working — main-thread marshalled, 598 offline tests |
| V-Ray sun + sky | working — parameters discovered from the live host |
| Weather → sky | working — turbidity from aerosol optical depth |
| OSM building massing | working — verified in the live host, sits on terrain |
| Terrain (Copernicus DEM) | working — GLO-30, single tile; mosaicking not implemented |
| Attribution manifest | working — generated from the sources a build used |
| Photogrammetry (Meshroom) | not started |
| MCP server | not started |

### The acceptance test

`demo_shadow_test.py` builds a column on a ground plane, renders it at three
times of one day at Dubai Marina, and checks the shadow against the computed
azimuth. Shadow directions match to within a few degrees, and lengths scale
correctly with sun altitude:

| Local time | Sun azimuth | Shadow bearing | Shadow length |
|---|---|---|---|
| 08:00 | 110.2° ESE | 290.2° WNW | 149 m |
| 12:00 | 177.3° S | 357.3° N | 38 m |
| 16:00 | 248.1° WSW | 68.1° ENE | 132 m |

Unit tests can all pass on an implementation that is wrong in a self-consistent
way. A shadow either points where the real one pointed, or it does not.

---

## Requirements

- **3ds Max 2022+** (developed against 2027 — Python 3.13.9, PySide6, pymxs)
- **V-Ray** (developed against V-Ray GPU 7 update 3)
- A CUDA GPU if you intend to render on the GPU
- No system Python needed — the venv is built from Max's bundled interpreter

## Setup

Build the virtual environment from **3ds Max's own Python**. It has `venv` and
`ensurepip` but no `pip`, and a venv leaves Max's site-packages untouched, so
there is no risk of destabilising Max:

```bash
"C:\Program Files\Autodesk\3ds Max 2027\Python\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install setuptools wheel
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`setuptools` goes first because some dependencies build from source on Python
3.13 and fail without it.

Copy `.env.example` to `.env` and read the comments — the bridge port allocation
in particular.

## Running it

**1. Start the bridge inside 3ds Max.** In the MAXScript listener (the one that
opens by default):

```
python.ExecuteFile @"<path-to-repo>\bridge\start_bridge.py"
```

You should see `[atlas] bridge listening on 127.0.0.1:9879`. The launcher stops
and reloads any previously loaded bridge, so it is safe to re-run.

**2. Check the connection:**

```bash
.venv\Scripts\python.exe verify_live.py
```

**3. Run the shadow test:**

```bash
.venv\Scripts\python.exe demo_shadow_test.py
```

**4. Build the surrounding city:**

```bash
.venv\Scripts\python.exe demo_massing.py
```

Fetches the buildings around the site, extrudes them and pushes them into Max,
then verifies against the **host** rather than against itself: bounding boxes
read back out of the scene (so an inch/metre mix-up cannot hide), vertex and
face counts per building, and the compass — the northmost building in geodetic
terms must be furthest along +Y in Max.

| | |
|---|---|
| Buildings fetched | 93 |
| Skipped | 0 |
| Verified in host | 93 — scale, orientation and index integrity |

## Tests

598 tests, no 3ds Max and no network required:

```bash
.venv\Scripts\python.exe -m pytest tests/ -q
```

Verification is layered deliberately. Solar position is checked against
definitional invariants (declination at equinox), exact geometry (solar-noon
altitude equals `90 − |lat − decl|`), and an independent algorithm (pvlib's NREL
SPA). Agreement between two different algorithms is evidence; checking one
against itself is not.

The massing geometry is checked the same way, and needed it. Its own invariants
are conservation laws a self-consistent bug cannot satisfy — triangles summing
to the polygon's area, divergence-theorem volume equalling footprint × height,
every edge shared by exactly two faces. On top of that it is cross-validated
against Shapely, and stress-tested on randomly generated footprints with up to
five courtyards.

That last layer earned its place immediately: all three bugs found while writing
the module survived every hand-written case and died to a randomly generated
one. Each produced a *building* rather than an error — a roof filled in across
an L-shaped notch, two courtyards whose seams tangled into overlapping
triangles, and a visibility test that silently reported "nothing in the way" for
every point whenever the sight triangle came out clockwise.

---

## Layout

```
server/
  solar.py       NOAA solar position — UTC only, no dependencies
  timeframe.py   wall-clock → UTC, with explicit DST failure modes
  frame.py       local ENU scene frame; +X east, +Y true north, +Z up
  weather.py     ERA5 + CAMS → V-Ray sky parameters
  osm.py         Overpass building footprints, heights and multipolygons
  massing.py     footprints → watertight extruded meshes
  terrain.py     Copernicus GLO-30 elevation → terrain mesh
  attribution.py per-build ATTRIBUTION.txt and height provenance
  scene.py       V-Ray sun, sky, camera and exposure
  maxbridge.py   client for the bridge
bridge/
  atlas_max_bridge.py     socket + main-thread marshalling (stable core)
  atlas_max_handlers.py   command handlers (hot-reloaded during development)
  start_bridge.py         launcher — run this from inside Max
tests/
```

### Conventions worth knowing before editing

- **+Y is true north, +X east, +Z up, units metres, site at the world origin.**
  3ds Max defaults to inches, and metre coordinates then land 39.37× wrong with
  no error. The origin is local because UTM coordinates are ~10⁶ m, where
  float32 spacing approaches a metre and produces z-fighting and viewport jitter.
- **V-Ray parameter names are discovered from the live host, never recalled.**
  MaxScript ignores assignment to a property that does not exist, so a
  misremembered name yields a default-lit render with nothing in any log.
- **The 3ds Max API is main-thread only.** Every call is marshalled through a
  QTimer. The bridge refuses to start without PySide6 rather than falling back
  to direct calls that would crash Max unpredictably later.
- **A `VRaySun` created from script has no target.** It is a targeted light, so
  without one its rotation stays identity and V-Ray points it straight down
  regardless of where the node sits. Verify the *direction*, not the position.
  **The same applies to cameras**, and it hid for a while: the only camera in
  the project was the shadow test's straight-down view, which is the one case
  where identity rotation is also the correct rotation. Aiming a camera at
  anything else produced an empty frame. Both now assign a real target node and
  assert the transform's Z axis afterwards.
- **Mesh face indices are 0-based everywhere except inside Max.** MaxScript is
  1-based, and the shift happens exactly once, in the `create_mesh` handler at
  the bridge boundary. Applying it twice — or not at all — does not raise; it
  shuffles the faces, which reads as a modelling mistake rather than a bug.
- **Overpass wants `(south, west, north, east)`.** GeoJSON and Leaflet want
  `(west, south, east, north)`. Transposing them is only detectable when the
  longitudes exceed ±90 and become impossible latitudes — inside that band the
  query succeeds and returns nothing, which is indistinguishable from an
  unmapped site. Use `osm.query_for_site()`, which builds the tuple from the
  frame so no caller writes the ordering by hand.
- **A Copernicus DEM read outside its tile returns `0.0`, not an error.** The
  dataset declares no nodata value and 0.0 is a real sea-level elevation, so
  loading the wrong tile is indistinguishable from a site on the coast —
  measured, a desert point read 0.0 instead of 113.7 m. Every read in
  `terrain.py` is bounds-checked before its value is trusted.
- **GLO-30 is nominally a *surface* model but measures as bare earth here.**
  Burj Khalifa's footprint reads 13.5 m against 828 m of building, so OSM
  massing can sit directly on it without double-counting roofs. That is a
  measurement, not a guarantee — `terrain.looks_like_surface_model()` re-checks
  it anywhere new.

---

## Data sources and attribution

| Source | Licence | Notes |
|---|---|---|
| Open-Meteo ERA5 + CAMS | CC BY 4.0 (data) | free endpoint is non-commercial, ~10k calls/day; self-host or use the commercial tier for production |
| OpenStreetMap | ODbL | rendered images are fine; redistributing geometry can trigger share-alike |
| Copernicus DEM GLO-30 | free, verbatim notice required | see below |

Copernicus requires this notice verbatim wherever the data is used:

> © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided
> under COPERNICUS by the European Union and ESA; all rights reserved

## Licensing constraints worth knowing

This project deliberately avoids two things it might obviously have used,
because the target is commercial delivery:

- **SLAM3R** is CC BY-NC-SA 4.0, as are its DUSt3R-derived weights —
  non-commercial only. **Meshroom / AliceVision (MPL-2.0)** is the planned
  reconstruction backend instead, and is a better fit anyway: it does UV unwrap
  and texture baking, where SLAM3R emits only point clouds.
- **Google Maps** cannot be the geometry source. The Map Tiles API policy bars
  image analysis, geodata extraction, offline use and tile caching, and the Maps
  ToS separately bars building 3D models from Street View imagery.

Geometry therefore comes from your own captured footage, plus OpenStreetMap and
Copernicus for context and terrain.

> **On 3ds Max licensing:** an Autodesk *Education* licence prohibits commercial
> use, and files saved from it carry a flag that propagates to any commercial
> seat that opens them. A commercial seat is required before output from this
> pipeline can ship, with assets rebuilt there rather than migrated.

## Licence

MIT — see [LICENSE](LICENSE).
