# Project Structure

This repository is moving from script-level experiments to a manifest-driven
pipeline. Existing scripts can keep running while new code reads and writes the
unified manifest.

```text
vlm_sam2_recon/
  schemas.py              Shared dataclass schema for targets, masks, meshes, joints, stages
  manifest_io.py          Atomic JSON read/write and validation helpers
  paths.py                Central project path conventions
  adapters/               Bridges from legacy script outputs to the unified schema
  core/                   Future shared IO, mask, crop, geometry, and optimization code
  stages/                 Future VLM/SAM2/TRELLIS/PhysX/EgoForce/pose/contact stages
  visualization/          Future viewers and inspection tools

scripts/
  build_unified_manifest.py
                          Builds outputs/project_manifest.json from current outputs

configs/
  project.example.json    Example run configuration

outputs/
  project_manifest.json   Unified project state produced by the bootstrap script
```

## Manifest Contract

The manifest is the handoff contract between stages:

- `targets`: semantic objects selected by VLM, including rigid/articulated class.
- `masks`: SAM2 masks, prompts, scores, and frame references.
- `trellis_jobs`: prepared image/mask jobs and upload contracts for TRELLIS2.
- `meshes`: whole-object meshes, rigid meshes, or articulated part meshes.
- `articulation_joints`: joint axis/origin/limit data, currently from PhysX-Omni.
- `physx_artifacts`: PhysX-Omni run folders, URDF/MJCF, and part mesh links.
- `egoforce_sequences`: hand/arm reconstruction outputs and available arrays.
- `stages`: coarse status records for VLM, SAM2, TRELLIS2, PhysX-Omni, EgoForce, pose, and contact.

New code should prefer `vlm_sam2_recon.schemas.ProjectManifest` instead of
reading ad hoc JSON files directly. Legacy JSON files are still referenced in
`legacy_refs` so nothing is lost during migration.
