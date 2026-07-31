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

