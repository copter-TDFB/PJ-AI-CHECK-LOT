# ADR 0005: Single-stage training + prelabel-on-demand for edit-drafts

Date: 2026-06-12
Status: Accepted (supersedes ADR 0001)

## Context

The wizard trained detectors in two Colab passes: a "seed" model on ~20
hand-labeled images that auto-prelabeled the rest (active learning), then a
"full" model on the whole dataset. Full training never depended on seed — seed
existed only to reduce manual annotation. The two-stage flow added a slow Colab
round-trip, a zip-bundle service, and extra draft states for little benefit.

## Decision

- Remove the seed path entirely: `seed/start`, `seed/done`, the `training_bundle`
  zip service, `build_seed_notebook`, and the `training_seed` draft status.
- Drafts go straight to one **full training**. The hard gate is **30 labeled
  images** (UI recommends 50).
- Prelabeling becomes **on demand and edit-draft-only**: `POST /{key}/training/prelabel`
  runs the *deployed* multi-class detector server-side and keeps boxes whose YOLO
  class matches the parent's `detector_yolo_prefixes`. A brand-new class has no
  deployed detector, so its images are labeled manually.
- Prelabeled boxes are ordinary annotations (label `"prelabel"`) the user can edit;
  no separate "suggestion" state.

## Consequences

- Simpler flow and codebase; one Colab round-trip instead of two.
- New classes require more manual labeling (no auto-prelabel) — accepted.
- Prelabel quality for edit-drafts depends on the currently deployed detector.
