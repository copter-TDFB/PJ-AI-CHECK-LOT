# Single-stage Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two-stage seed→full training; drafts go straight to one full training, and edit-drafts can prelabel newly-added images on demand using the deployed detector.

**Architecture:** Delete the Colab seed-training path entirely. Reuse the existing `services/active_learning.prelabel_remaining` (already a service) behind a new synchronous endpoint `POST /{key}/training/prelabel` that runs the *active* multi-class detector and filters boxes by the parent packaging's `detector_yolo_prefixes`. Raise the `full/start` gate from 10 to 30. Strip seed UI/JS from the wizard and add a prelabel button for edit-drafts only.

**Tech Stack:** Python 3.11, FastAPI, pytest, ultralytics YOLO, vanilla HTML/JS wizard (`web/wizard.html`).

**Spec:** `docs/superpowers/specs/2026-06-12-single-stage-training-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `services/active_learning.py` | Run a detector over unlabeled draft images → save bbox predictions | Modify: add `filter_prelabel_bboxes` helper + `class_prefixes` param |
| `tests/test_active_learning.py` | Unit-test the pure bbox filter | Create |
| `api/packagings.py` | Wizard endpoints | Modify: add `training_prelabel`; raise full gate; delete seed endpoints + `_MIN_LABELED_FOR_SEED` |
| `tests/test_api_packagings.py` | Endpoint tests | Modify: prelabel tests; full-gate test update |
| `services/training_bundle.py` | (seed-only zip builder) | **Delete** |
| `services/notebook_generator.py` | Colab notebooks | Modify: delete `build_seed_notebook` |
| `tests/test_notebook_generator.py` | Notebook tests | Modify: delete seed test |
| `services/dataset_publisher.py` | Dataset publish | Modify: docstring reference to training_bundle |
| `web/wizard.html` | Wizard UI + JS | Modify: remove seed UI/JS, add prelabel button, target 20→30, stepMap |
| `docs/adr/0001-*.md` | ADR | Modify: mark superseded |
| `docs/adr/0005-single-stage-training.md` | New ADR | Create |
| `CLAUDE.md` | Project guide | Modify: Wizard API section |

---

### Task 1: Prefix-aware bbox filter in active_learning

**Files:**
- Modify: `services/active_learning.py`
- Test: `tests/test_active_learning.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_active_learning.py`:

```python
"""Unit tests for the pure prelabel bbox filter (no YOLO/torch needed)."""

from services.active_learning import filter_prelabel_bboxes


def test_filter_keeps_only_matching_prefix():
    boxes = [
        (0, 0, 10, 10, "box_lot"),
        (0, 0, 10, 10, "sachet_lot"),
    ]
    out = filter_prelabel_bboxes(boxes, class_prefixes=["box_"])
    assert len(out) == 1
    assert out[0]["label"] == "prelabel"
    assert out[0]["x2"] == 10.0


def test_filter_drops_zero_or_negative_area_boxes():
    boxes = [(5, 5, 5, 9, "box_lot"), (5, 5, 9, 5, "box_lot")]
    assert filter_prelabel_bboxes(boxes, class_prefixes=["box_"]) == []


def test_filter_no_prefix_keeps_all_positive_area():
    boxes = [(0, 0, 2, 2, "anything"), (0, 0, 1, 1, "other")]
    out = filter_prelabel_bboxes(boxes, class_prefixes=None)
    assert len(out) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_active_learning.py -v`
Expected: FAIL — `ImportError: cannot import name 'filter_prelabel_bboxes'`

- [ ] **Step 3: Add the helper and wire it into `prelabel_remaining`**

In `services/active_learning.py`, add this function above `prelabel_remaining` (after the constants block, ~line 19):

```python
def filter_prelabel_bboxes(boxes, class_prefixes=None):
    """boxes: iterable of (x1, y1, x2, y2, class_name).

    Keep boxes whose class_name starts with one of class_prefixes (when given)
    and that have positive area. Returns a list of annotation bbox dicts.
    """
    kept = []
    for x1, y1, x2, y2, name in boxes:
        if class_prefixes and not any(name.startswith(p) for p in class_prefixes):
            continue
        if x2 > x1 and y2 > y1:
            kept.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2), "label": "prelabel",
            })
    return kept
```

Then change the signature of `prelabel_remaining` (line 21) to:

```python
def prelabel_remaining(key: str, model_path: Path, conf: float = _PRELABEL_CONF,
                       class_prefixes: list[str] | None = None) -> dict:
```

After `model = YOLO(str(model_path))` (line 32) add:

```python
    class_names = model.names if hasattr(model, "names") else {}
```

Keep the line `r = results[0]`. Replace everything from `bboxes = []` through `prelabeled += 1` (current lines 51-60) with:

```python
            boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls_ids = r.boxes.cls.int().tolist()
                for box, cid in zip(xyxy, cls_ids):
                    name = class_names.get(int(cid), "") if isinstance(class_names, dict) else ""
                    boxes.append((float(box[0]), float(box[1]),
                                  float(box[2]), float(box[3]), name))
            bboxes = filter_prelabel_bboxes(boxes, class_prefixes)
            if bboxes:
                packaging_store.save_annotation(key, img_path.name, bboxes)
                prelabeled += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_active_learning.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add services/active_learning.py tests/test_active_learning.py
git commit -m "feat: prefix-aware bbox filter for prelabeling"
```

---

### Task 2: Prelabel-on-demand endpoint

**Files:**
- Modify: `api/packagings.py` (insert after `training_full_start`, before `training_full_done` at line ~619)
- Test: `tests/test_api_packagings.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api_packagings.py`:

```python
# ─── training/prelabel — active detector, edit-drafts only ───────────

def test_prelabel_rejects_non_edit_draft(client, monkeypatch):
    from services import packaging_store
    monkeypatch.setattr(packaging_store, "get_draft", lambda key: {"key": key})
    res = client.post("/api/packagings/plain/training/prelabel")
    assert res.status_code == 400
    assert "edit-draft" in res.json()["detail"]


def test_prelabel_edit_draft_runs_active_detector(client, monkeypatch, tmp_path):
    import main
    from services import packaging_store

    monkeypatch.setattr(packaging_store, "get_draft", lambda key: {"key": key})

    model_file = tmp_path / "detector.pt"
    model_file.write_bytes(b"stub")
    monkeypatch.setattr("api.packagings._DETECTOR_MODEL_PATH", model_file)

    cfg = MagicMock()
    cfg.detector_yolo_prefixes = ["box_"]
    reg = MagicMock()
    reg.get.return_value = cfg
    monkeypatch.setattr(main, "registry", reg)

    pl = MagicMock(return_value={"prelabeled": 3, "skipped_already_labeled": 1, "errors": 0})
    monkeypatch.setattr("services.active_learning.prelabel_remaining", pl)

    res = client.post("/api/packagings/box__edit/training/prelabel")
    assert res.status_code == 200, res.text
    assert res.json()["prelabeled"] == 3
    reg.get.assert_called_with("box")
    pl.assert_called_once_with("box__edit", model_file, class_prefixes=["box_"])


def test_prelabel_503_when_no_active_detector(client, monkeypatch, tmp_path):
    import main
    from services import packaging_store

    monkeypatch.setattr(packaging_store, "get_draft", lambda key: {"key": key})
    monkeypatch.setattr("api.packagings._DETECTOR_MODEL_PATH", tmp_path / "missing.pt")
    cfg = MagicMock(); cfg.detector_yolo_prefixes = ["box_"]
    reg = MagicMock(); reg.get.return_value = cfg
    monkeypatch.setattr(main, "registry", reg)

    res = client.post("/api/packagings/box__edit/training/prelabel")
    assert res.status_code == 503
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_packagings.py -k prelabel -v`
Expected: FAIL — endpoint returns 404 (route not found) / 405, not the asserted codes

- [ ] **Step 3: Add the endpoint**

In `api/packagings.py`, insert immediately after the end of `training_full_start` (after its `return {...}` block, line ~618):

```python
@router.post("/{key}/training/prelabel")
def training_prelabel(key: str):
    """Prelabel unlabeled draft images with the active (deployed) detector.

    Edit-drafts only — a brand-new class has no deployed detector to borrow.
    Runs server-side; no Colab, no retrain. Boxes are filtered by the parent
    packaging's detector_yolo_prefixes and written as normal annotations.
    """
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")
    if not key.endswith("__edit"):
        raise HTTPException(400, "prelabel ใช้ได้เฉพาะ edit-draft (class เดิม)")
    parent_key = key[: -len("__edit")]

    import main
    cfg = main.registry.get(parent_key) if main.registry is not None else None
    if cfg is None:
        raise HTTPException(400, f"parent packaging '{parent_key}' not found")

    if not _DETECTOR_MODEL_PATH.exists():
        raise HTTPException(503, "active detector ยังไม่พร้อม — ไม่มีโมเดลให้ prelabel")

    from services import active_learning

    try:
        result = active_learning.prelabel_remaining(
            key, _DETECTOR_MODEL_PATH, class_prefixes=cfg.detector_yolo_prefixes,
        )
    except Exception as e:
        logger.exception("prelabel failed for %s", key)
        raise HTTPException(500, f"prelabel failed: {e}")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_packagings.py -k prelabel -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py
git commit -m "feat: prelabel-on-demand endpoint for edit-drafts"
```

---

### Task 3: Raise full-training gate to 30

**Files:**
- Modify: `api/packagings.py:551` (the `< 10` gate in `training_full_start`)
- Test: `tests/test_api_packagings.py` (existing `test_training_full_start_publishes_dataset` + new gate test)

- [ ] **Step 1: Update the existing test to 30 images and add a gate test**

In `tests/test_api_packagings.py`, in `test_training_full_start_publishes_dataset`, change the `list_annotation_status` monkeypatch from `range(10)` to `range(30)`:

```python
    monkeypatch.setattr(
        packaging_store, "list_annotation_status",
        lambda key: [{"name": f"i{n}.jpg", "labeled": True, "bbox_count": 1}
                     for n in range(30)],
    )
```

Then append a new gate test:

```python
def test_training_full_start_rejects_below_30(client, monkeypatch):
    from services import packaging_store
    client.post("/api/packagings", json={"key": "under30", "display_name": "x"})
    monkeypatch.setattr(
        packaging_store, "list_annotation_status",
        lambda key: [{"name": f"i{n}.jpg", "labeled": True, "bbox_count": 1}
                     for n in range(29)],
    )
    res = client.post("/api/packagings/under30/training/full/start")
    assert res.status_code == 400
    assert "30" in res.json()["detail"]
```

- [ ] **Step 2: Run tests to verify the new gate test fails**

Run: `pytest tests/test_api_packagings.py -k "full_start" -v`
Expected: `test_training_full_start_rejects_below_30` FAILS (29 images currently passes the `< 10` gate → 200, not 400)

- [ ] **Step 3: Raise the gate**

In `api/packagings.py`, in `training_full_start` (line ~551), change:

```python
    if len(labeled) < 10:
        raise HTTPException(400, f"need at least 10 labeled images (have {len(labeled)})")
```

to:

```python
    if len(labeled) < 30:
        raise HTTPException(400, f"need at least 30 labeled images (have {len(labeled)})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_packagings.py -k "full_start" -v`
Expected: PASS (both full-start tests green)

- [ ] **Step 5: Commit**

```bash
git add api/packagings.py tests/test_api_packagings.py
git commit -m "feat: raise full-training gate from 10 to 30 labeled images"
```

---

### Task 4: Delete the seed-training code

**Files:**
- Modify: `api/packagings.py` (delete `training_seed_start`, `training_seed_done`, `_MIN_LABELED_FOR_SEED`)
- Delete: `services/training_bundle.py`
- Modify: `services/notebook_generator.py` (delete `build_seed_notebook`)
- Modify: `tests/test_notebook_generator.py` (delete seed test)
- Modify: `services/dataset_publisher.py:5` (docstring)

- [ ] **Step 1: Delete the seed test first (keeps suite honest)**

In `tests/test_notebook_generator.py`, delete the whole `test_seed_notebook_unchanged_still_uses_bundle` function (lines 33-38).

- [ ] **Step 2: Delete `build_seed_notebook`**

In `services/notebook_generator.py`, delete the entire `build_seed_notebook` function (lines 38-109, from `def build_seed_notebook(` up to and including its `return _wrap_nb(cells)` and the trailing blank line before `def build_full_notebook(`).

- [ ] **Step 3: Delete the seed endpoints + constant in `api/packagings.py`**

Delete:
- `_MIN_LABELED_FOR_SEED = 5` line (451) and its comment.
- The entire `training_seed_start` function (lines 463-538, `@router.post("/{key}/training/seed/start")` through its `return {...}`).
- The entire `training_seed_done` function (lines 762-815, `@router.post("/{key}/training/seed/done")` through its `return {...}`).

Leave the `# ─── Training (seed → active learning) ───` section comment but rename it to `# ─── Training ─────────────────`.

- [ ] **Step 4: Delete the bundle service and fix its one referrer**

```bash
git rm services/training_bundle.py
```

In `services/dataset_publisher.py`, line 5, change the docstring fragment:

```python
uses services/training_bundle.py — see ADR 0003).
```

to:

```python
publishes the reference dataset directly — see ADR 0003).
```

- [ ] **Step 5: Run the full suite to verify nothing references the deleted code**

Run: `pytest tests/test_api_packagings.py tests/test_notebook_generator.py tests/test_active_learning.py -v`
Expected: PASS. No `ImportError`, no `AttributeError` for `training_bundle` / `build_seed_notebook`.

Also confirm no stragglers:
Run: `grep -rn "training_bundle\|build_seed_notebook\|seed/start\|seed/done\|_MIN_LABELED_FOR_SEED" api services tests`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add api/packagings.py services/notebook_generator.py services/dataset_publisher.py tests/test_notebook_generator.py
git commit -m "refactor: remove seed-training path (endpoints, bundle, notebook)"
```

---

### Task 5: Wizard frontend — drop seed UI, add prelabel button

> **REQUIRED:** This task is frontend work. Invoke `frontend-design` to design the prelabel button + step-3 footer layout, then `web-design-guidelines` to review the result before committing (per user instruction 2026-06-12). The HTML/JS below is the functional baseline — match the wizard's existing visual language (`--acc`, `.btn`, `.info-box`) and ensure intentional hover/focus/active/loading/disabled states.

**Files:**
- Modify: `web/wizard.html`

- [ ] **Step 1: Raise the annotation target to 30**

Line 2359: change `ANNOT_TARGET: 20,` → `ANNOT_TARGET: 30,`
Line 1297: change the static counter `/ <strong id="annot-target">20</strong>` → `/ <strong id="annot-target">30</strong>`
Line 1254: change the desc text `อย่างน้อย 20 รูป` → `อย่างน้อย 30 รูป`

- [ ] **Step 2: Replace the seed blocks (lines 1308-1333) with prelabel bar + next button**

Replace the two blocks `<!-- Post-annotation: Colab seed training ... -->` (`#ls-done`) and `#colab-done` with:

```html
              <!-- Prelabel (edit-draft only): borrow the deployed detector -->
              <div id="prelabel-bar" style="display:none;margin-top:14px">
                <div class="info-box">
                  🪄 <strong>Prelabel อัตโนมัติ</strong> — ใช้ detector ตัวที่ deploy อยู่
                  ใส่กรอบให้รูปที่ยังไม่ label (ตรวจ/แก้ได้ภายหลัง)
                </div>
                <div style="text-align:center;margin-top:10px">
                  <button class="btn btn-secondary" id="btn-prelabel" onclick="runPrelabel()">
                    🪄 Prelabel รูปที่ยังไม่ label
                  </button>
                </div>
              </div>

              <!-- Ready → next, once labeled target is met -->
              <div id="ls-done" style="display:none;margin-top:16px">
                <div class="info-box ok">✅ <strong>Label ครบเป้า</strong> — พร้อมไปตั้งค่า Config</div>
                <div style="text-align:center;margin-top:10px">
                  <button class="btn btn-primary" id="btn-step3-next" onclick="goStep(4)">
                    ถัดไป: ตั้งค่า Config →
                  </button>
                </div>
              </div>
```

- [ ] **Step 3: Show the prelabel bar for edit-drafts in `loadAnnotator`**

In `loadAnnotator` (after line 2386 `renderThumbStrip();`), add:

```javascript
    const isEditDraft = curDraftKey && curDraftKey.endsWith('__edit');
    document.getElementById('prelabel-bar').style.display = isEditDraft ? 'block' : 'none';
```

- [ ] **Step 4: Replace `openColab` + `trainingSeedDone` JS with `runPrelabel`**

Delete `openColab` (lines 2746-2769) and `trainingSeedDone` (lines 2771-2788). In their place add:

```javascript
async function runPrelabel() {
  if (!curDraftKey) { alert('ยังไม่ได้สร้าง draft'); return; }
  const btn = document.getElementById('btn-prelabel');
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = '⏳ กำลัง prelabel...';
  try {
    const res = await api('POST', `/api/packagings/${encodeURIComponent(curDraftKey)}/training/prelabel`);
    toast(`✓ Prelabel แล้ว ${res.prelabeled} รูป · ข้าม ${res.skipped_already_labeled} · ผิดพลาด ${res.errors}`);
    await loadAnnotator();  // refresh thumbs + progress with new bboxes
  } catch (err) {
    alert(`Prelabel ไม่สำเร็จ: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}
```

(`toast` is the existing wizard toast helper — verify it exists with `grep -n "function toast" web/wizard.html`; if absent, use `alert` with the same message.)

- [ ] **Step 5: Simplify `updateProgressUI`**

Replace `updateProgressUI` (lines 2642-2649) with:

```javascript
function updateProgressUI() {
  const labeled = annot.images.filter(im => im.labeled).length;
  const target = Math.min(annot.ANNOT_TARGET, annot.images.length);
  document.getElementById('annot-lc').textContent = labeled;
  document.getElementById('annot-pbar').style.width = Math.min(100, labeled / target * 100) + '%';
  // Reveal "ready → Config" once target met
  document.getElementById('ls-done').style.display = labeled >= target ? 'block' : 'none';
}
```

- [ ] **Step 6: Remove `training_seed` from the resume stepMap**

Line 2212: delete the line `    training_seed: 3,    // seed training ค้าง — กลับ annotator`

- [ ] **Step 7: Manual verification in the browser**

Serve and click through (Playwright blocks `file://`, see CLAUDE.md):
```bash
python -m http.server 8090 --directory web
```
- New draft → step 3 shows annotator with **no** prelabel bar; counter target reads 30; at 30 labeled the "ถัดไป: ตั้งค่า Config →" button appears and advances to step 4.
- Edit-draft (`*__edit`) → step 3 shows the 🪄 prelabel bar; clicking it calls `/training/prelabel` and refreshes thumbs.
- Confirm no console errors and no remaining references to `openColab`/`trainingSeedDone`:
```bash
grep -n "openColab\|trainingSeedDone\|seed/start\|seed/done\|training_seed" web/wizard.html
```
Expected: no matches.

- [ ] **Step 8: Commit**

```bash
git add web/wizard.html
git commit -m "feat: wizard single-stage flow + prelabel button, drop seed UI"
```

---

### Task 6: Docs — ADRs + CLAUDE.md

**Files:**
- Modify: `docs/adr/0001-*.md`
- Create: `docs/adr/0005-single-stage-training.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Mark ADR 0001 superseded**

Find the ADR 0001 file: `ls docs/adr/0001-*.md`. At the top of its body (under the title), add:

```markdown
> **Status: Superseded by [ADR 0005](0005-single-stage-training.md) (2026-06-12).**
> The seed → active-learning → full flow described here is replaced by a single
> full-training step plus prelabel-on-demand for edit-drafts.
```

- [ ] **Step 2: Write ADR 0005**

Create `docs/adr/0005-single-stage-training.md`:

```markdown
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
```

- [ ] **Step 3: Update CLAUDE.md Wizard API section**

In `CLAUDE.md`, in the "Wizard API" section, update the draft-status sentence. Change the `status` resume description to drop `training_seed` and the seed flow. Replace the paragraph that begins "Draft `status` is the wizard's resume point..." so the status list reads:

```
`draft` = no images yet → step 2; `uploading` = has images → step 3 (first `save_image` bumps draft→uploading); `configured`/`training_full`/`trained` → step 4/5.
```

And replace any sentence describing "wizard trains both models via seed/active-learning" with:

```
Training is single-stage: drafts label ≥30 images then run Full Training (ADR 0005). Edit-drafts can `POST /{key}/training/prelabel` to auto-fill bboxes on newly-added images using the deployed detector (filtered by the parent's `detector_yolo_prefixes`).
```

- [ ] **Step 4: Commit**

```bash
git add docs/adr/0001-*.md docs/adr/0005-single-stage-training.md CLAUDE.md
git commit -m "docs: ADR 0005 single-stage training + update CLAUDE.md"
```

---

## Final verification

- [ ] Run the full test suite:

```bash
pytest -q
```
Expected: green except the known pre-existing `tests/test_classifier.py` 3 setup errors (documented in CLAUDE.md — not caused by this work).

- [ ] Confirm the seed path is fully gone:

```bash
grep -rn "training_bundle\|build_seed_notebook\|seed/start\|seed/done\|training_seed\|_MIN_LABELED_FOR_SEED\|openColab\|trainingSeedDone" api services tests web
```
Expected: no matches (the untracked `FLOW_SWIMLANE_WIZARD.html` generated diagram may still mention seed — regenerate or ignore; it is not source).
