# Drive-backed Card Count + Hybrid Card Samples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the wizard dashboard cards show an exact Drive-sourced image count and 2-3 sample thumbnails per card on production, where the local `images/` dataset is not shipped.

**Architecture:** A new `services/drive_samples.py` resolves `<CLS_FOLDER>/images/<key>/` on Drive and caches per-instance (TTL 600 s). `_count_active_images` prefers a populated local dir (dev) and falls back to the Drive count (prod). The two active image-serving endpoints fall back to Drive when the local dir is empty. The frontend serves baked thumbnails (committed under `web/samples/`) for known classes and falls back to the Drive-backed endpoint for classes with no baked thumbnails.

**Tech Stack:** Python 3.11, FastAPI, Google Drive API (`services/drive_client.py`), Pillow (thumbnail generation), vanilla JS (`web/wizard.html`), pytest.

## Global Constraints

- `print()` is forbidden in application code — use the module `logger`. Standalone CLI scripts under `scripts/` may write to stdout via `sys.stdout.write` (not `print`).
- Run tests with `python -m pytest` (pytest is not on PATH).
- Read source files with `encoding="utf-8"` (files contain Thai).
- Drive class folders are named by **`key`**: `<CLS_FOLDER>/images/<key>/` (`services/dataset_publisher.py:121-122`). `CLS_FOLDER` = env `DRIVE_CLASSIFIER_DATASET_FOLDER_ID`.
- `_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}` (already in `api/packagings.py:37`).
- `DriveClient` methods used: `find_in_folder(parent_id, name) -> str | None`, `list_folder(parent_id) -> list[dict]` (each `{id,name,mimeType}`), `download_file(file_id, dest: Path) -> None`.
- Never let a Drive failure break the dashboard — Drive helpers return empty / None and log a warning, never raise.

---

### Task 1: `services/drive_samples.py` — Drive image resolver with TTL cache

**Files:**
- Create: `services/drive_samples.py`
- Test: `tests/test_drive_samples.py`

**Interfaces:**
- Consumes: `services.drive_client.DriveClient` (`find_in_folder`, `list_folder`).
- Produces:
  - `class_images(key: str) -> list[dict]` — each item `{"id": str, "name": str}`; `[]` on any failure / unset env. Cached per-instance for 600 s.
  - `clear_cache() -> None` — reset the module cache (tests + future cache-busting).

- [ ] **Step 1: Write the failing test**

Create `tests/test_drive_samples.py`:

```python
import pytest


class _FakeDrive:
    def __init__(self, list_calls):
        self._list_calls = list_calls

    def find_in_folder(self, parent, name):
        return {
            ("CLSROOT", "images"): "IMG",
            ("IMG", "back_label"): "CLASS",
        }.get((parent, name))

    def list_folder(self, parent):
        self._list_calls.append(parent)
        return [
            {"id": "1", "name": "a.jpg", "mimeType": "image/jpeg"},
            {"id": "2", "name": "b.png", "mimeType": "image/png"},
            {"id": "3", "name": "notes.txt", "mimeType": "text/plain"},
        ]


def test_class_images_resolves_chain_and_filters(monkeypatch):
    from services import drive_samples
    drive_samples.clear_cache()
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSROOT")
    monkeypatch.setattr("services.drive_client.DriveClient", lambda: _FakeDrive([]))

    out = drive_samples.class_images("back_label")
    assert [f["name"] for f in out] == ["a.jpg", "b.png"]
    assert out[0]["id"] == "1"


def test_class_images_cached_within_ttl(monkeypatch):
    from services import drive_samples
    drive_samples.clear_cache()
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSROOT")
    calls = []
    monkeypatch.setattr("services.drive_client.DriveClient", lambda: _FakeDrive(calls))

    drive_samples.class_images("back_label")
    drive_samples.class_images("back_label")
    assert len(calls) == 1  # second call served from cache

    drive_samples.clear_cache()
    drive_samples.class_images("back_label")
    assert len(calls) == 2


def test_class_images_env_unset_returns_empty(monkeypatch):
    from services import drive_samples
    drive_samples.clear_cache()
    monkeypatch.delenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", raising=False)
    assert drive_samples.class_images("back_label") == []


def test_class_images_exception_safe(monkeypatch):
    from services import drive_samples
    drive_samples.clear_cache()
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSROOT")

    def boom():
        raise RuntimeError("drive down")

    monkeypatch.setattr("services.drive_client.DriveClient", boom)
    assert drive_samples.class_images("back_label") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_drive_samples.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.drive_samples'`

- [ ] **Step 3: Write minimal implementation**

Create `services/drive_samples.py`:

```python
"""Resolve a packaging class's classifier-dataset images on Drive (ADR 0003 layout).

Production does NOT ship the local `images/` dataset, so the dashboard's image
count and sample thumbnails come from Drive: `<CLS_FOLDER>/images/<key>/`, where
CLS_FOLDER = env DRIVE_CLASSIFIER_DATASET_FOLDER_ID and the class folder is named
by `key` (see services/dataset_publisher.py).

`class_images()` never raises — a Drive outage must not break the dashboard. Results
(including empty ones) are cached per-instance for _TTL_SECONDS.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CLS_ENV = "DRIVE_CLASSIFIER_DATASET_FOLDER_ID"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_TTL_SECONDS = 600

# key -> (monotonic_timestamp, list[{"id","name"}])
_CACHE: dict[str, tuple[float, list[dict]]] = {}


def clear_cache() -> None:
    """Drop all cached entries (tests + future cache-busting)."""
    _CACHE.clear()


def class_images(key: str) -> list[dict]:
    """Return [{"id","name"}] of the Drive classifier-dataset images for `key`.

    Resolves <CLS_FOLDER>/images/<key>. Returns [] when the env var is unset, the
    folder chain is missing, or Drive errors. Cached per-instance for _TTL_SECONDS.
    """
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached is not None and now - cached[0] < _TTL_SECONDS:
        return cached[1]

    result: list[dict] = []
    cls_root = os.getenv(_CLS_ENV, "").strip()
    if cls_root:
        try:
            from services.drive_client import DriveClient

            drive = DriveClient()
            images_id = drive.find_in_folder(cls_root, "images")
            class_id = drive.find_in_folder(images_id, key) if images_id else None
            if class_id:
                result = [
                    {"id": f["id"], "name": f["name"]}
                    for f in drive.list_folder(class_id)
                    if Path(f["name"]).suffix.lower() in _IMG_EXTS
                ]
        except Exception as e:  # noqa: BLE001 — dashboard must survive Drive outages
            logger.warning("drive_samples.class_images(%s) failed: %s", key, e)
            result = []

    _CACHE[key] = (now, result)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_drive_samples.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add services/drive_samples.py tests/test_drive_samples.py
git commit -m "feat: Drive classifier-dataset image resolver with TTL cache"
```

---

### Task 2: Drive-backed image count

**Files:**
- Modify: `api/packagings.py:1051-1058` (`_count_active_images`)
- Test: `tests/test_api_packagings.py` (append tests)

**Interfaces:**
- Consumes: `services.drive_samples.class_images` (Task 1).
- Produces: `_count_active_images(key: str) -> int` — local count if local dir populated, else Drive count if env set, else 0.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_packagings.py`:

```python
def test_count_active_images_prefers_local(monkeypatch, tmp_path):
    from api import packagings

    d = tmp_path / "back_label"
    d.mkdir()
    (d / "a.jpg").write_bytes(b"x")
    (d / "b.jpg").write_bytes(b"y")
    monkeypatch.setattr(packagings, "_ACTIVE_IMAGES_DIR", tmp_path)

    def _no_drive(key):
        raise AssertionError("Drive must not be hit when local dir is populated")

    monkeypatch.setattr("services.drive_samples.class_images", _no_drive)
    assert packagings._count_active_images("back_label") == 2


def test_count_active_images_falls_back_to_drive(monkeypatch, tmp_path):
    from api import packagings

    monkeypatch.setattr(packagings, "_ACTIVE_IMAGES_DIR", tmp_path)  # empty
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSROOT")
    monkeypatch.setattr(
        "services.drive_samples.class_images",
        lambda key: [{"id": "1", "name": "a.jpg"}, {"id": "2", "name": "b.jpg"},
                     {"id": "3", "name": "c.jpg"}],
    )
    assert packagings._count_active_images("x") == 3


def test_count_active_images_zero_without_local_or_env(monkeypatch, tmp_path):
    from api import packagings

    monkeypatch.setattr(packagings, "_ACTIVE_IMAGES_DIR", tmp_path)  # empty
    monkeypatch.delenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", raising=False)
    assert packagings._count_active_images("x") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_packagings.py -k count_active_images -v`
Expected: FAIL — `test_count_active_images_falls_back_to_drive` returns 0 (current code only reads the local dir).

- [ ] **Step 3: Write minimal implementation**

Replace `_count_active_images` (`api/packagings.py:1051-1058`):

```python
def _count_active_images(key: str) -> int:
    img_dir = _ACTIVE_IMAGES_DIR / key
    if img_dir.exists():
        local = sum(
            1 for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
        if local > 0:
            return local
    if os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "").strip():
        from services import drive_samples
        return len(drive_samples.class_images(key))
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_packagings.py -k count_active_images -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py
git commit -m "feat: count active images from Drive when local dataset absent"
```

---

### Task 3: `GET /{key}/images` Drive fallback (active classes)

**Files:**
- Modify: `api/packagings.py:434-442` (active branch of `list_images`)
- Test: `tests/test_api_packagings.py` (append a test)

**Interfaces:**
- Consumes: `services.drive_samples.class_images` (Task 1); `main.registry`.
- Produces: when an active class's local dir is empty and the env var is set, the
  endpoint returns `{"images": [{"name", "size": None, "read_only": True}, ...]}`
  from Drive instead of `{"images": []}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_packagings.py`:

```python
def test_list_images_active_drive_fallback(client, monkeypatch, tmp_path):
    import main
    from api import packagings

    class _Reg:
        def get(self, k):
            return object()  # truthy cfg → active branch

    monkeypatch.setattr(main, "registry", _Reg())
    monkeypatch.setattr(packagings, "_ACTIVE_IMAGES_DIR", tmp_path)  # empty
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSROOT")
    monkeypatch.setattr(
        "services.drive_samples.class_images",
        lambda key: [{"id": "1", "name": "a.jpg"}, {"id": "2", "name": "b.jpg"}],
    )

    r = client.get("/api/packagings/back_label/images")
    assert r.status_code == 200
    body = r.json()
    assert [i["name"] for i in body["images"]] == ["a.jpg", "b.jpg"]
    assert all(i["read_only"] for i in body["images"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_packagings.py -k list_images_active_drive_fallback -v`
Expected: FAIL — returns `{"images": []}` (current active branch never consults Drive).

- [ ] **Step 3: Write minimal implementation**

Replace the active branch of `list_images` (`api/packagings.py:434-442`):

```python
    if main.registry is not None and main.registry.get(key) is not None:
        img_dir = _ACTIVE_IMAGES_DIR / key
        local = (
            [
                {"name": p.name, "size": p.stat().st_size, "read_only": False}
                for p in sorted(img_dir.iterdir())
                if p.is_file() and p.suffix.lower() in _IMG_EXTS
            ]
            if img_dir.exists()
            else []
        )
        if local:
            return {"images": local}
        if os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "").strip():
            from services import drive_samples
            return {"images": [
                {"name": f["name"], "size": None, "read_only": True}
                for f in drive_samples.class_images(key)
            ]}
        return {"images": []}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_packagings.py -k list_images_active_drive_fallback -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py
git commit -m "feat: serve active-class image list from Drive when local dir empty"
```

---

### Task 4: `GET /{key}/images/{filename}` Drive download fallback (active classes)

**Files:**
- Modify: `api/packagings.py` — add `import tempfile` (top), add `_DRIVE_SAMPLE_CACHE` constant near `_ACTIVE_IMAGES_DIR` (`:34`), add `_drive_sample_path` helper, and patch the active branch of `get_image` (`:478-483`).
- Test: `tests/test_api_packagings.py` (append a test)

**Interfaces:**
- Consumes: `services.drive_samples.class_images` (Task 1); `services.drive_client.DriveClient.download_file`.
- Produces: `_drive_sample_path(key: str, safe: str) -> Path | None` — a `/tmp` cached
  copy of a Drive image, or None; and `get_image` serving it for active classes when
  the local file is missing.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_packagings.py`:

```python
def test_get_image_active_drive_download(client, monkeypatch, tmp_path):
    import main
    from api import packagings

    class _Reg:
        def get(self, k):
            return object()

    monkeypatch.setattr(main, "registry", _Reg())
    monkeypatch.setattr(packagings, "_ACTIVE_IMAGES_DIR", tmp_path / "empty")
    monkeypatch.setattr(packagings, "_DRIVE_SAMPLE_CACHE", tmp_path / "cache")
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSROOT")
    monkeypatch.setattr(
        "services.drive_samples.class_images",
        lambda key: [{"id": "FID", "name": "a.jpg"}],
    )

    class _Drive:
        def download_file(self, file_id, dest):
            assert file_id == "FID"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"\xff\xd8\xffDATA")

    monkeypatch.setattr("services.drive_client.DriveClient", lambda: _Drive())

    r = client.get("/api/packagings/back_label/images/a.jpg")
    assert r.status_code == 200
    assert r.content == b"\xff\xd8\xffDATA"


def test_get_image_active_drive_missing_returns_404(client, monkeypatch, tmp_path):
    import main
    from api import packagings

    class _Reg:
        def get(self, k):
            return object()

    monkeypatch.setattr(main, "registry", _Reg())
    monkeypatch.setattr(packagings, "_ACTIVE_IMAGES_DIR", tmp_path / "empty")
    monkeypatch.setattr(packagings, "_DRIVE_SAMPLE_CACHE", tmp_path / "cache")
    monkeypatch.setenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "CLSROOT")
    monkeypatch.setattr("services.drive_samples.class_images", lambda key: [])

    r = client.get("/api/packagings/back_label/images/nope.jpg")
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_packagings.py -k get_image_active_drive -v`
Expected: FAIL — `test_get_image_active_drive_download` gets 404 (current active branch raises when the local file is missing); and `_DRIVE_SAMPLE_CACHE` does not yet exist (AttributeError on monkeypatch).

- [ ] **Step 3: Write minimal implementation**

Add `import tempfile` to the imports block at the top of `api/packagings.py` (after `import os`).

Add the cache constant right after `_ACTIVE_IMAGES_DIR = Path("images")` (`:34`):

```python
_DRIVE_SAMPLE_CACHE = Path(
    os.getenv("DRIVE_SAMPLE_CACHE_DIR", str(Path(tempfile.gettempdir()) / "drive_samples"))
)
```

Add this helper (place it next to `_count_active_images`):

```python
def _drive_sample_path(key: str, safe: str) -> Path | None:
    """Return a locally-cached copy of a Drive classifier-dataset image, or None.

    Downloads on first miss into _DRIVE_SAMPLE_CACHE/<key>/<safe>; serves the cached
    file thereafter. Returns None when the name is not in the class's Drive folder or
    the download fails (caller turns this into a 404).
    """
    dest = _DRIVE_SAMPLE_CACHE / key / safe
    if dest.exists() and dest.is_file():
        return dest

    from services import drive_samples

    file_id = next(
        (f["id"] for f in drive_samples.class_images(key) if f["name"] == safe), None
    )
    if file_id is None:
        return None
    try:
        DriveClient().download_file(file_id, dest)
        return dest
    except Exception as e:  # noqa: BLE001 — a broken thumbnail must not 500
        logger.warning("drive sample download failed %s/%s: %s", key, safe, e)
        return None
```

Patch the active branch of `get_image` (`api/packagings.py:478-483`):

```python
    if main.registry is not None and main.registry.get(key) is not None:
        candidate = _ACTIVE_IMAGES_DIR / key / safe
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        if os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "").strip():
            cached = _drive_sample_path(key, safe)
            if cached is not None:
                return FileResponse(cached)
        raise HTTPException(404, "image not found")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_packagings.py -k get_image_active_drive -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py
git commit -m "feat: serve active-class images by downloading from Drive on miss"
```

---

### Task 5: `scripts/build_card_samples.py` — thumbnail generator

**Files:**
- Create: `scripts/build_card_samples.py`

**Interfaces:**
- Consumes: local `images/<key>/` directories; Pillow.
- Produces: `web/samples/<key>/0.jpg…` thumbnails and a `BAKED_SAMPLES` JS const printed to stdout for pasting into `web/wizard.html` (Task 6).

- [ ] **Step 1: Create the script**

Create `scripts/build_card_samples.py`:

```python
"""Generate small card thumbnails for the wizard dashboard from local images/<key>/.

Writes web/samples/<key>/0.jpg.. (downscaled) and emits a `BAKED_SAMPLES` JS const
on stdout to paste into web/wizard.html. Run from the repo root:

    python scripts/build_card_samples.py > /tmp/baked.js

print() is avoided per repo convention; output goes through sys.stdout/sys.stderr.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

SRC = REPO / "images"
OUT = REPO / "web" / "samples"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_W = 320
PER_CLASS = 3


def main() -> None:
    manifest: dict[str, list[str]] = {}
    if not SRC.exists():
        sys.stderr.write(f"no local images dir at {SRC}\n")
        return
    for class_dir in sorted(SRC.iterdir()):
        if not class_dir.is_dir():
            continue
        key = class_dir.name
        imgs = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )[:PER_CLASS]
        if not imgs:
            continue
        dest_dir = OUT / key
        dest_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i, src in enumerate(imgs):
            im = Image.open(src).convert("RGB")
            if im.width > MAX_W:
                im = im.resize((MAX_W, round(im.height * MAX_W / im.width)))
            dest = dest_dir / f"{i}.jpg"
            im.save(dest, "JPEG", quality=70)
            paths.append(f"samples/{key}/{i}.jpg")
        manifest[key] = paths
        sys.stderr.write(f"  {key}: {len(paths)} thumbs\n")

    sys.stdout.write(
        "const BAKED_SAMPLES = "
        + json.dumps(manifest, indent=2, ensure_ascii=False)
        + ";\n"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to seed thumbnails**

Run: `python scripts/build_card_samples.py > /tmp/baked_samples.js`
Expected: stderr lists each class (e.g. `back_label: 3 thumbs`); `web/samples/<key>/0.jpg…` created; `/tmp/baked_samples.js` holds the `const BAKED_SAMPLES = {…};` block. Verify: `ls web/samples/back_label/` shows `0.jpg 1.jpg 2.jpg`, each a small (KB) file.

- [ ] **Step 3: Commit the script + generated thumbnails**

```bash
git add scripts/build_card_samples.py web/samples/
git commit -m "feat: card-sample thumbnail generator + baked thumbnails"
```

---

### Task 6: Frontend — baked manifest + `loadCardImages` fallback

**Files:**
- Modify: `web/wizard.html` — add `BAKED_SAMPLES` const (paste from Task 5) just before `const API_BASE` (`:1724`); replace `loadCardImages` (`:2528-2542`).

**Interfaces:**
- Consumes: `BAKED_SAMPLES` (key → relative thumbnail paths); the `GET /{key}/images` endpoint (now Drive-backed, Task 3) for keys without baked thumbnails.
- Produces: card photo strips populated from baked thumbnails for known classes, Drive fallback for the rest.

- [ ] **Step 1: Paste the manifest**

Insert the `const BAKED_SAMPLES = {…};` block from `/tmp/baked_samples.js` into `web/wizard.html` immediately before the `const API_BASE = (() => {` line (`:1724`).

- [ ] **Step 2: Replace `loadCardImages`**

Replace `loadCardImages` (`web/wizard.html:2528-2542`) with:

```javascript
async function loadCardImages(key) {
  const strip = document.querySelector(`.pkg-photo-strip[data-key="${CSS.escape(key)}"]`);
  if (!strip) return;
  const baked = (typeof BAKED_SAMPLES !== 'undefined') ? BAKED_SAMPLES[key] : null;
  if (baked && baked.length) {
    strip.innerHTML = '';
    baked.slice(0, 3).forEach(src => {
      const el = document.createElement('img');
      el.src = src;            // relative to the frontend host (Netlify / http.server)
      el.alt = `${key} sample`;
      el.loading = 'lazy';
      strip.appendChild(el);
    });
    return;
  }
  try {
    const { images } = await api('GET', `/api/packagings/${encodeURIComponent(key)}/images`);
    if (images.length === 0) return;
    strip.innerHTML = '';
    images.slice(0, 3).forEach(img => {
      const el = document.createElement('img');
      el.src = `${API_BASE}/api/packagings/${encodeURIComponent(key)}/images/${encodeURIComponent(img.name)}`;
      el.alt = `${key} sample`;
      el.loading = 'lazy';
      strip.appendChild(el);
    });
  } catch (e) { console.warn('loadCardImages', key, e); }
}
```

- [ ] **Step 3: Manual verification**

Serve the frontend and load the dashboard against prod:

Run: `python -m http.server 8090 --directory web`
Then open `http://localhost:8090/wizard.html` (its `API_BASE` resolves to localhost — to test against prod instead, temporarily set `window.API_BASE_OVERRIDE` in the console before `loadDashboard`, or rely on the deployed Netlify copy).

Expected:
- An existing class (e.g. `back_label`) shows baked thumbnails from `samples/back_label/…` — confirm in DevTools the `<img src>` is the relative `samples/...` path and **no** `GET /api/packagings/back_label/images` request fires for it.
- A class not in `BAKED_SAMPLES` issues `GET /api/packagings/<key>/images` and renders Drive thumbnails (or stays empty gracefully if Drive returns none).
- With the backend unreachable, the dashboard still renders (cards just lack live thumbnails) — no uncaught errors in the console.

- [ ] **Step 4: Commit**

```bash
git add web/wizard.html
git commit -m "feat: card thumbnails from baked samples with Drive fallback"
```

---

## Post-implementation verification (not a code task)

- **Drive folder naming check (prod, before trusting the count):** with prod Drive creds, confirm the existing 7 class folders are key-named under `<CLS_FOLDER>/images/`:
  `python -c "import os; from dotenv import load_dotenv; load_dotenv(); from services.drive_client import DriveClient; d=DriveClient(); root=os.environ['DRIVE_CLASSIFIER_DATASET_FOLDER_ID']; img=d.find_in_folder(root,'images'); print([f['name'] for f in d.list_folder(img)])"`
  Expected: names include `back_label`, `capsule_box`, … If a folder uses a different name than its `key`, note it — that class's count/fallback will resolve to 0 until reconciled.
- **Full suite:** `python -m pytest` — confirm no regressions (the 3 pre-existing `tests/test_classifier.py` setup errors are unrelated, per CLAUDE.md).
- **Deploy:** backend (Tasks 1-4) via Cloud Run build (`gcloud builds submit` → `ocr-repo` → `--image`, not `--source`); frontend (`web/wizard.html`, `web/samples/**`) via Netlify.
- **Test-wizard copies:** `test wizzard/wizard.html` and `dist/portable-bundle/static/wizard.html` are regenerated from `web/wizard.html`; they will not have `web/samples/**` unless that dir is copied too — out of scope here, note for the harness.

## Self-Review

- **Spec coverage:** A → Task 1; B → Task 2; C(list) → Task 3; C(serve) → Task 4; D → Task 6; E → Task 5; testing/deploy → tests in each task + post-impl section. All covered.
- **Placeholder scan:** no TBD/TODO; every code step shows full code.
- **Type consistency:** `class_images(key) -> list[{"id","name"}]` is consumed identically in Tasks 2/3/4; `_drive_sample_path(key, safe) -> Path | None` matches its use in `get_image`; `_DRIVE_SAMPLE_CACHE` defined in Task 4 and monkeypatched in the same task's tests; `BAKED_SAMPLES` produced in Task 5, consumed in Task 6.
