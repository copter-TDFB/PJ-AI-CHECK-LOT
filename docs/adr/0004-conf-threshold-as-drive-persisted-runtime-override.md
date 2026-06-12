# 0004 — conf_threshold is user-tunable via a Drive-persisted runtime override

Date: 2026-06-11
Status: Accepted

## Context

`conf_threshold` (the per-packaging classifier gate) lives in
`config/packagings/*.yaml`, which is baked into the Cloud Run image.
ADR 0002 deliberately kept it out of the wizard's ownership: edit-deploys
preserve it from the existing YAML (`cloudrun_deployer.write_packaging_yaml`).

Ops now needs to tune this value from the wizard UI without a retrain.
Two problems block the obvious approaches:

1. The clone → retrain → deploy flow (ADR 0002) exists to protect production
   during a 1-2 h retrain — none of which applies to changing a single gate
   number. Forcing a retrain to change `0.6 → 0.75` is absurd.
2. The wizard on Netlify talks to production Cloud Run directly. An in-place
   YAML edit on the running instance is **ephemeral** — it silently reverts
   when the instance recycles, and other instances never see it. (The same
   latent issue exists for archive/unarchive today.)

## Decision

`conf_threshold` becomes a **runtime tuning field**, edited directly on an
active packaging — exempt from the clone flow — and persisted **outside the
image** in a Drive-hosted `config_overrides.json`:

- New env var `DRIVE_CONFIG_OVERRIDES_FILE_ID` points at a pre-created file
  on Drive shared with the SA (same pattern as `DRIVE_MANIFEST_FILE_ID`).
  Empty locally → falls back to `data/config_overrides.json`.
- The overrides file stores **only runtime tuning fields**, keyed by
  packaging key (`{"back_label": {"conf_threshold": 0.75}}`) — never whole
  YAML documents, so the baked YAML stays the single source for everything
  else.
- `PackagingRegistry` merges overrides over the YAML at construction time,
  so every reader (`main.py` gate, wizard API) sees the merged value with no
  call-site changes.
- `PUT /api/packagings/{key}/conf` validates **0.50 ≤ value ≤ 0.95**, writes
  Drive first, reloads the registry only on success (Drive failure → 502,
  nothing changes locally — instances never diverge from Drive).
- Startup: if the overrides file is unreadable, log a warning and serve YAML
  values — a broken tuning file must never take the service down.

The bounds are deliberate: below ~0.5 misclassified images slip into the
wrong pipeline and produce confidently-wrong sheet checks (worse than an
honest `low_confidence`); above 0.95 the class is effectively disabled.

## Consequences

- A second config source exists. Mitigated by scoping it to tuning fields
  only and merging in exactly one place (registry construction).
- `write_packaging_yaml` keeps writing the (possibly stale) YAML value on
  edit-deploys — harmless, because the override always wins at merge time.
- The Drive-override pattern is the designated future fix for the
  archive/unarchive ephemerality noted above.

**Alternatives rejected**

- *Route conf edits through the ADR 0002 clone flow* — forces a needless
  1-2 h retrain; conf is reversible in O(1) like archive.
- *Edit the YAML in-place on the instance (archive-style)* — silently
  reverts on instance recycle; unacceptable for a value users set on purpose.
- *Commit YAML + rebuild image per change* — durable but no longer
  self-service from the UI.
