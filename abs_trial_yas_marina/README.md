# Yas Marina animation — ABS trial run

Self-contained Blender scene for trialling the anamorphic billboard system (ABS).

## Contents

- `atlas_yas_sunset_cinematic.blend` — Blender 5.2. Frames 0–2177 at 24 fps
  (~90 s), 18 cameras (15 bound shot cameras). Textures are packed inside.
- `sunset_assets/vdb/` — 4,061 smoke/dust VDB caches (4.5 GB), referenced
  relatively. Keep this folder next to the `.blend` or the smoke disappears.
- `sunset_asset_manifest.json` — size and SHA-256 of every VDB.

Saved render settings: Cycles/OptiX, 3840x2160, up to 1024 samples, AgX,
motion blur, multilayer EXR. Lower them for a quick trial.

## Downloading only this folder

Everything here is in Git LFS. On the other machine:

```bash
git clone --filter=blob:none --no-checkout --branch abs-trial https://github.com/ssnfts/atlas.git
cd atlas
git sparse-checkout set abs_trial_yas_marina
git checkout abs-trial
git lfs pull --include "abs_trial_yas_marina/**"
```

## Status

This is a quality-review revision, not the finished photoreal movie: the full
animation render and all-frame approval are still outstanding. Building
geometry derives from a 3ds Max scene authored on an education licence — fine
for an internal trial, not for commercial delivery.
