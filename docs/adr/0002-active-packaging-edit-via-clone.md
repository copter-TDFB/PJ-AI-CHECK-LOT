# 0002 — Editing an active packaging goes through a clone-and-deploy flow

Date: 2026-06-09
Status: Accepted

## Context

Once a packaging is deployed via the wizard, ops still needs to:

1. **Retrain the same class with more images** (e.g., production has
   accumulated 200 new back_label photos that should improve accuracy).
2. **Soft-disable a packaging** that's being phased out, without retraining
   the whole model just to forget one class.

Phase 3 of the wizard treated each `{key}` as write-once: image upload blocked
when the key was active (`api/packagings.py:140-141`), and Deploy blocked when
the YAML already existed (`api/packagings.py:453-454`). The only escape was to
pick a different key, which would create a parallel class the Classifier
doesn't actually know.

Updating an active packaging is more dangerous than creating a new one because
production keeps serving traffic during the 1-2 hour Colab retrain. A
mid-edit failure that left the active YAML half-rewritten or the
`models/detector.pt` blank would take n8n down.

## Decision

Editing an active packaging is **a separate workflow built on top of the
existing draft wizard**, not an in-place mutation.

### Clone

`POST /api/packagings/{key}/clone` reads the active YAML and creates an
edit-draft at key `{key}__edit`. The edit-draft is a normal draft —
images, annotations, training, and Deploy all reuse the existing wizard
endpoints — but its `meta.json` carries an extra `parent_key` field.

The edit-draft links to the parent's reference images by URL (the wizard's
`GET /{key}/images` endpoint returns parent images flagged `read_only=True`
alongside any new uploads) rather than copying them onto disk.

Only one edit-draft per active is allowed at a time. A second clone returns
409 — the wizard prompts to continue or discard the existing one.

### Deploy overwrite

When `POST /api/packagings/{key}/deploy` is called on a draft whose meta has
a `parent_key`:

1. The hard-floor eval gate runs first (cheap, fails before any side-effects).
2. `cloudrun_deployer.backup_artifacts(parent_key)` snapshots
   `config/packagings/{parent_key}.yaml`, `models/detector.pt`, and
   `models/classifier.pt` to `*.bak-{ISO_TIMESTAMP}`.
3. The YAML is rewritten **under `parent_key`** (not the draft key) — and
   any field the wizard does not own (`conf_threshold`, `accuracy`,
   `post_ocr_fixes`, …) is preserved from the existing YAML rather than
   reset to defaults.
4. The freshly-trained detector at `data/drafts/{edit_key}/models/full_detector.pt`
   is promoted to `models/detector.pt`.
5. The registry is reloaded in-process. On any failure between steps 3-5,
   `restore_backup()` puts the original files back and the registry reloads
   again — production sees the rollback as a no-op.
6. The edit-draft is deleted on success. Backups beyond the 3 most recent
   per key are rotated out.

### Archive

`POST /api/packagings/{key}/archive` renames `{key}.yaml` →
`{key}.yaml.archived`. The Classifier still knows the class; the
`PackagingRegistry` no longer picks it up because its `glob("*.yaml")`
ignores the `.archived` suffix. `main.py:predict()` checks
`registry.is_archived()` after classification and returns
`status="archived_class"` with a human-readable Thai message instead of
running the pipeline.

`POST /api/packagings/{key}/unarchive` renames back. No model retraining
involved either way — archive is reversible in O(1).

## Consequences

**Positive**

- Production keeps serving the old detector during the 1-2h retrain.
- A failed retrain or a failed hard-floor check produces no visible
  change to production — the backup-and-rollback path runs inside the
  deploy endpoint synchronously.
- The wizard UI does not need a new editing surface — the same steps
  (upload → annotate → train → deploy) work for edits because the
  edit-draft is a normal draft with a parent pointer.
- Soft-delete via archive avoids retraining when ops just wants to stop
  routing traffic to a class.

**Negative**

- Backups grow `models/` by ~50-300 MB each (rotated to 3) — non-trivial
  if Cloud Run instances cold-boot from a model image.
- The `__edit` key convention is a string-level contract; a draft created
  manually with that suffix would also be treated as an overwrite.
- Archive doesn't shrink the Classifier — the model still spends inference
  budget on a class that's never used. Hard-delete (retrain without the
  class) is out of v1 scope.

**Alternatives rejected**

- *In-place edit on the active key* — Would require disabling the existing
  409 guard on Deploy and the 404 guard on image upload, and would offer
  no rollback point if the new training collapsed. The clone gives us a
  natural place to gate the operation.
- *Versioned active YAMLs (`back_label/v1/`, `back_label/v2/`)* — Solves
  rollback elegantly but bloats the on-disk layout and requires plumbing
  a "current" pointer through `PackagingRegistry`, `main.py`, and Cloud
  Run startup. Backup-and-restore covers the same need with less code.
- *Hard-delete instead of archive* — Forces a 1-2h retrain every time
  ops wants to disable a packaging temporarily. Soft-delete handles the
  common case; hard-delete can ride along the next retrain of a different
  class if it ever becomes needed.
