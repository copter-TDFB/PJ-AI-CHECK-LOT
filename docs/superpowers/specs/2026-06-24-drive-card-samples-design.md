# Drive-backed card image count + hybrid card samples

**Date:** 2026-06-24
**Status:** Approved (design)

## Problem

On production Cloud Run the wizard dashboard cards show **no images** and report
`image_count = 0` for every active packaging. Root cause: sample images live under
`images/<key>/` on the local filesystem, but that directory is deliberately
excluded from the container build (`.gcloudignore:21`, `.dockerignore:26` — ~414 MB)
and from git (`.gitignore:60-63`, only `.gitkeep` tracked). So the prod container's
`images/<key>/` dirs are empty:

- `_count_active_images(key)` (`api/packagings.py:1051`) counts an empty dir → 0
- `GET /{key}/images` (active branch, `api/packagings.py:434-442`) returns `{"images": []}`
- `loadCardImages()` (`web/wizard.html:2528-2542`) early-returns on empty → photo
  strip stays as the gray placeholder

The reference dataset **does** exist on Drive. The classifier dataset folder
(`DRIVE_CLASSIFIER_DATASET_FOLDER_ID`, the `data classify check lot` folder) is laid
out as `<CLS_FOLDER>/images/<key>/` — folder named by **`key`**, confirmed by
`services/dataset_publisher.py:121-122` (`ensure_folder("images", cls_root)` →
`ensure_folder(key, cls_images)`). Prod sets this env var and authenticates to Drive
as an OAuth user able to read it.

## Goal (user decision)

Chosen option **C** + **A1**:

- **Count must be exact, sourced from Drive** (source of truth) on prod.
- **Sample images** only need to show 2-3 thumbnails per card (not real-time):
  - existing classes use **baked thumbnails** committed into the frontend (fast, no
    Drive cost),
  - any class **without** baked thumbnails (e.g. a newly added class) **falls back**
    to live Drive images via the backend.

## Non-goals

- Real-time image freshness (user accepted staleness within a cache TTL).
- Shipping the full `images/` dataset to prod (the 414 MB exclusion stays).
- Changing inference, training, or deploy pipelines.
- Making Drive files public / using `thumbnailLink` in `<img src>` (rejected:
  needs auth / expires).

## Architecture

Five units, each independently testable.

### A. `services/drive_samples.py` — shared Drive image resolver (new)

Single source for both count and fallback samples (DRY — one Drive round-trip
serves both).

```
class_images(key: str) -> list[dict]   # [{"id": ..., "name": ...}, ...]
```

- Resolves `CLS_FOLDER → "images" → "<key>"` via
  `DriveClient.find_in_folder` (twice), then `list_folder`, filtering to image
  extensions (`_IMG_EXTS`, reuse the set from `api/packagings.py` or duplicate a
  small constant here).
- `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` unset, folder chain missing, or any
  DriveClient exception → returns `[]` and logs a warning. **Never raises** — the
  dashboard must not break when Drive is unreachable.
- **Per-instance in-memory TTL cache**, TTL = **600 s** (module-level dict keyed by
  `key`, storing `(timestamp, list)`). Cloud Run instances are ephemeral and the
  count is allowed to lag ≤10 min after an upload (consistent with "images not
  real-time"). `Date.now()`-equivalent uses `time.monotonic()`.
- Cache negative/empty results too (same TTL) so a missing folder doesn't re-hit
  Drive every dashboard load.
- Expose a `clear_cache()` for tests.

### B. Count — `_count_active_images(key)` (modify `api/packagings.py:1051`)

Ordering preserves offline local dev while making prod exact:

1. If local `images/<key>/` **exists and has image files** → count locally
   (dev convenience, no Drive dependency).
2. Else if `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` is set →
   `len(drive_samples.class_images(key))`.
3. Else → 0.

Result: prod (empty local dir) → Drive count; local dev (populated dir) → local
count, no Drive call.

### C. Sample-image serving fallback (modify `api/packagings.py`)

Only the **active** branches change; draft/edit-draft logic untouched.

- `GET /{key}/images` active branch (`:434-442`): if the local dir is empty **and**
  `DRIVE_CLASSIFIER_DATASET_FOLDER_ID` is set → return names from
  `drive_samples.class_images(key)` (all names; frontend slices 3), each
  `{"name", "size": None, "read_only": True}`.
- `GET /{key}/images/{filename}` active branch (`:479-483`): if the local file is
  missing **and** Drive is set → look up the file id by name from
  `class_images(key)` (cache hit), `DriveClient.download_file` into
  `/tmp/drive_samples/<key>/<name>`, then `FileResponse` it. Subsequent hits serve
  the cached `/tmp` copy. On any Drive failure → 404 (frontend already tolerates a
  broken `<img>`).

In production this path triggers **only for classes with no baked thumbnails**
(existing 7 classes are served by the frontend and never call these endpoints for
images).

### D. Frontend — baked thumbnails + manifest + fallback (modify `web/wizard.html`)

- Baked thumbnails committed at `web/samples/<key>/0.jpg` … `2.jpg`.
- Inline manifest constant in `wizard.html`:
  `const BAKED_SAMPLES = { back_label: ['samples/back_label/0.jpg', ...], ... }`
  (relative paths — resolve correctly on Netlify, `http.server`, and `file://`).
- `loadCardImages(key)` (`:2528`): if `key in BAKED_SAMPLES` → populate the strip
  from the baked relative paths, **no backend call**; else → keep the existing
  `GET /{key}/images` flow (now Drive-backed via C).
- `image_count` rendering is unchanged — it already comes from the backend, which is
  now Drive on prod (B).

### E. `scripts/build_card_samples.py` — thumbnail generator (new)

- For each active class with a populated local `images/<key>/`: take the first 2-3
  files, downscale (max width ~320 px, JPEG quality ~70) with Pillow, write to
  `web/samples/<key>/0.jpg`…, and emit/refresh the `BAKED_SAMPLES` manifest block.
- Keeps `web/` growth in KB, not MB. Run once now to seed the 7 existing classes;
  re-run later to refresh or add a class.
- Standalone script under `scripts/` → `sys.path.insert(0, repo_root)`, run from
  repo root, read images with `encoding`-safe paths.

## Data flow

```
Dashboard load
  GET /api/packagings ─▶ _count_active_images(key)
                          ├─ local dir populated → local count        (dev)
                          └─ else → drive_samples.class_images(key)    (prod, cached)
  per card: loadCardImages(key)
    key in BAKED_SAMPLES → <img src="samples/<key>/n.jpg">            (existing classes)
    else → GET /{key}/images → (active+empty+Drive) class_images names (new classes)
             then GET /{key}/images/{name} → Drive download → /tmp cache → stream
```

## Error handling

- Drive unreachable / env unset / folder missing → `class_images` returns `[]`:
  count shows 0, no samples, dashboard still renders. Warning logged.
- Drive image download fails → 404 → broken `<img>` (already tolerated).
- Cache staleness ≤ TTL (600 s) after a new upload — accepted.

## Testing

- **Unit `tests/` (pytest, mock `DriveClient`):**
  - `class_images` resolves the folder chain, filters extensions, caches within TTL,
    re-fetches after `clear_cache()`, and returns `[]` on missing chain / exception.
  - `_count_active_images`: local-populated → local count (no Drive call);
    local-empty + Drive set → Drive count; neither → 0.
  - `GET /{key}/images` active Drive fallback returns Drive names when local empty.
- **Manual:** existing class shows baked thumbnails (no backend image call); a class
  not in the manifest fetches from Drive; with Drive env unset the dashboard renders
  with 0 / no images and does not error.

## Deploy

- Backend changes (A/B/C) → Cloud Run build per `CLAUDE.md` (`gcloud builds submit`
  → `ocr-repo` → `--image`, **not** `--source`).
- Frontend changes (D) + `web/samples/**` → Netlify (the frontend host).
- `web/samples/**` must be tracked by git (it is small) — confirm not caught by the
  `images/**/*.jpg` ignore (it lives under `web/`, not `images/`, so it is safe).

## Open items folded into decisions

- TTL = 600 s (confirmed).
- Count local-FS-first on local dev (confirmed).
- Class folder on Drive is named by `key` (confirmed via `dataset_publisher.py`);
  the plan includes a one-time verification that the **existing** 7 class folders on
  Drive are key-named before relying on the count path in prod.
