# Anti-tiling PBR texture design

**Status:** approved for implementation on 2026-08-01, with visual tiling as
a release-blocking defect.

## Goal

Replace the visibly repetitive track and building materials in the Yas scene
with commercially safe, high-frequency PBR imagery while preserving physical
scale, avoiding low-sun displacement artefacts, and leaving the generic race
cars unbranded.

## Scope

- The three asphalt/pit roadway nodes use a race-specific asphalt material.
- The 373 ordinary OSM building nodes use a concrete facade material.
- The six already-classified grandstand/stadium nodes retain their existing
  grandstand graph.
- Generic car paint retains the live-host-verified two-layer clearcoat; no
  bitmap will be stamped across a moving car body.
- The update runs against the current live 3ds Max scene and checkpoints it
  after the track, building, and car verification milestones.

Houdini, skydome/HDRI work, scene geometry, OSM rebuilds, rendered frames,
F1/team likenesses, logos and liveries are explicitly out of scope.

## Asset policy

Only local CC0 Poly Haven PBR sets may be fetched or used. Google Maps,
Street View, 3D Tiles, aerial imagery and other Google content must not be
extracted, traced, baked or used as texture input.

| Surface | Primary source | Secondary source | Physical primary scale |
| --- | --- | --- | --- |
| Race asphalt | `asphalt_track` (Diffuse, Rough, `nor_gl`) | local `asphalt_02` | 2.0 m |
| Ordinary facade concrete | `concrete_slab_wall_02` (Diffuse, Rough, `nor_gl`) | local `concrete_layers_02` | 2.1 m |
| Car paint | host-verified procedural clearcoat | none | object-local, no image repeat |

Assets are fetched at 4K to keep V-Ray GPU memory bounded. `nor_gl` is
mandatory; the DirectX green channel convention must not be substituted.

## Anti-tiling contract

The following are testable implementation constraints, not artistic advice:

1. A final track or ordinary-building material must not be fed by one image
   projection alone.
2. Each albedo and roughness channel must blend two independent
   `VRayTriplanarTex` image projections at incommensurate scales. The
   secondary-to-primary ratio is fixed to the golden-ratio approximation
   `1.618`, which prevents a shared repeat interval.
3. The secondary triplanar source must carry a distinct 3D frame offset and
   rotation and enable random texture offset and rotation. The primary source
   also enables the random offset/rotation controls so separate objects cannot
   begin in an identical phase.
4. The blend mask is a separately projected `Noise` map in a
   `VRayTriplanarTex`, at 37 m for the track and 19 m for buildings. It must
   not share either image sampler's scale or coordinates.
5. The diffuse blend receives the existing dirt/ambient-occlusion treatment;
   roughness is blended before the host-verified roughness-to-glossiness
   inversion. Normal maps are decoded with `VRayNormalMap` and use the same
   staggered sampling treatment where that construction is host-accepted.
6. Asphalt must not receive a procedural bump or displacement pass: prior
   measurement showed low-angle sunlight turns it into corrugation. Its
   racing-line read comes from the PBR roughness/albedo blend only.
7. A live build with a non-empty `rejected` property map is a hard failure.
   The scene must not be saved as a completed texture milestone until all
   applied graph parameters survive host construction.

## Acceptance evidence without renders

The user requested no renders, therefore this pass supplies structural proof,
not a subjective image claim:

- focused graph tests prove the two source projections, scales, transform
  offsets, randomisation flags and independent macro mask;
- host material construction returns an empty rejection map;
- node/material assignment counts match the pre-measured scene inventory;
- the scene is checkpointed through `MaxBridge.save_scene` after every
  milestone, which verifies the file mtime changed.

This is sufficient to rule out a single repeated bitmap setup. A final
frame-level assessment of artistic realism still requires a future viewport or
render review, which is intentionally deferred rather than faked.

## Implementation evidence (2026-08-01)

- `tests/test_texturing.py`: 40 focused tests passed. They assert the required
  dual 2.0/3.236 m and 2.1/3.3978 m image projections, `1.618` scale ratio,
  random phase flags, non-zero secondary offset/rotation, independent macro
  masks, and the deliberately bump-free asphalt channel.
- Both approved 4K CC0 sets were fetched with exactly `Diffuse`, `Rough` and
  `nor_gl` maps: Poly Haven `asphalt_track` and `concrete_slab_wall_02`.
- Live V-Ray construction: the unassigned track probe built 14/14 texmaps with
  `rejected={}`. The applied track graph built 14/14 maps on 3 nodes; the
  building graph built 20/20 maps on 373 nodes, again with `rejected={}`.
- Assignment audit: `atlas_track_asphalt_antitile` is on exactly 3 nodes,
  `atlas_concrete_antitile` is on exactly 373 nodes, and none of the six
  protected grandstand/stadium nodes was overwritten.
- Car audit: all 20 body meshes retain `coat_amount=0.80`,
  `coat_glossiness=0.96`, `coat_ior=1.52`, with zero diffuse bitmap maps.
- Verified scene checkpoints: `out/yas_race_pbr_track.max`,
  `out/yas_race_pbr_buildings.max`, and `out/yas_race_hyperreal_pbr.max`.

No render or viewport capture was made, at the user's direction.
