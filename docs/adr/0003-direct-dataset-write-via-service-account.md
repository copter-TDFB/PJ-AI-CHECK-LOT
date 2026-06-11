# 0003 — Backend writes the reference dataset directly via service account

Date: 2026-06-10
Status: Accepted (supersedes the addition-bundle data path of ADR 0001;
the "notebook trains both models" decision in ADR 0001 stays in force)

## Context

ADR 0001 routed new-packaging data through a backend-owned "addition" folder
because the full `drive` OAuth scope is restricted for unverified OAuth
clients. But `services/drive_client.py` authenticates with
`google_auth_default()` — a service account, not a user OAuth client. Service
accounts never go through OAuth consent verification, so the restriction that
motivated the bundle/merge design does not apply: sharing the dataset folders
with the SA's email is enough.

The bundle design also left real costs: the Drive dataset was never complete
on its own, addition folders piled up and needed manual cleanup, and the
notebook duplicated dataset-layout knowledge (class-id offsets, splits).

## Decision

On Full Training start, the backend writes the draft's images + YOLO labels
directly into the reference dataset folders shared with the service account:

- `data check lot/` — detector train/val images + labels + `data.yaml`
- `data classify check lot/{class}/` — classifier images

Rules (implemented in `services/dataset_publisher.py`):

- `data.yaml` `names` are append-only — existing entries are never reordered.
- `data.yaml` is written LAST (commit point): a half-finished upload leaves
  the dataset valid; retries skip already-uploaded files (idempotent).
- Filenames are prefixed with the packaging key to prevent collisions.
- Deterministic 80/20 train/val split by filename hash.

Seed training keeps the small zip bundle — it runs before publication and its
model is throwaway.

## Consequences

**Positive** — the Drive dataset is the complete, single source of truth;
the notebook shrinks to mount-copy-train; "add images to an existing class"
works for free through the same path; no addition folders to clean up.

**Negative** — requires one-time sharing of both folders with the SA + two
env vars (`DRIVE_DETECTOR_DATASET_FOLDER_ID`, `DRIVE_CLASSIFIER_DATASET_FOLDER_ID`);
files created by the SA count against the SA's 15 GB quota (moving the dataset
to a Workspace Shared Drive removes this if it ever matters); the backend now
holds full-Drive scope on the SA, so dataset-write bugs can corrupt the
dataset — mitigated by append-only names + commit-point ordering.
