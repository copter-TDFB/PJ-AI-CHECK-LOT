"""Tests สำหรับ wizard API — packagings endpoints.

ใช้ TestClient + mock OcrEngine (เหมือน test_integration.py)
"""

import shutil
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Spin up TestClient with isolated draft dir + mocked Vision."""
    draft_dir = tmp_path_factory.mktemp("drafts")

    with patch("pipeline.ocr_engine.vision") as mock_vision, \
         patch.dict("os.environ", {"DRAFT_DIR": str(draft_dir)}):
        mock_resp = MagicMock()
        mock_resp.error.message = ""
        mock_resp.text_annotations = []
        mock_resp.full_text_annotation.pages = []
        mock_vision.ImageAnnotatorClient.return_value.text_detection.return_value = mock_resp
        mock_vision.Image = MagicMock(side_effect=lambda content: content)

        # Force reload of packaging_store with new DRAFT_DIR
        import importlib
        from services import packaging_store
        importlib.reload(packaging_store)

        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app) as c:
            yield c


# ─── Regex preview (no state) ────────────────────────────

def test_regex_preview_simple(client):
    r = client.post("/api/packagings/regex/preview", json={
        "examples": ["TB0005001014612426", "TB0006001014789012", "TB0007001014999876"]
    })
    assert r.status_code == 200
    body = r.json()
    assert "pattern" in body
    assert len(body["matches"]) == 3
    assert all(m["ok"] for m in body["matches"])


def test_regex_preview_rejects_empty(client):
    r = client.post("/api/packagings/regex/preview", json={"examples": []})
    assert r.status_code == 422


# ─── Listing ─────────────────────────────────────────────

def test_list_returns_active_packagings(client):
    r = client.get("/api/packagings")
    assert r.status_code == 200
    items = r.json()
    keys = {it["key"] for it in items}
    # all 6 active configs should be present
    assert {"back_label", "import_sticker", "container_label",
            "grade_bag", "retail_sachet", "capsule_box"}.issubset(keys)
    actives = [it for it in items if it["status"] == "active"]
    assert len(actives) >= 6


# ─── Draft lifecycle ─────────────────────────────────────

def test_create_draft(client):
    r = client.post("/api/packagings", json={
        "key": "test_box",
        "display_name": "Test Box",
        "description": "test desc",
        "pipeline": "detector_ocr",
    })
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["key"] == "test_box"
    assert body["status"] == "draft"
    assert body["image_count"] == 0


def test_create_draft_rejects_duplicate(client):
    # second create with same key
    r = client.post("/api/packagings", json={
        "key": "test_box",
        "display_name": "dup",
    })
    assert r.status_code == 409


def test_create_draft_rejects_active_key(client):
    r = client.post("/api/packagings", json={
        "key": "back_label",
        "display_name": "should fail",
    })
    assert r.status_code == 409


def test_create_draft_rejects_invalid_key(client):
    r = client.post("/api/packagings", json={
        "key": "Bad-Key!",
        "display_name": "x",
    })
    assert r.status_code == 422


def test_get_draft(client):
    r = client.get("/api/packagings/test_box")
    assert r.status_code == 200
    assert r.json()["display_name"] == "Test Box"


def test_get_active(client):
    r = client.get("/api/packagings/back_label")
    assert r.status_code == 200
    assert r.json()["status"] == "active"


def test_get_draft_returns_sub_regions_for_annotator(client):
    """Regression: GET must surface sub_regions/detection_mode so the wizard
    annotator shows the per-field label chips (renderLabelBar needs len > 1)."""
    r = client.post("/api/packagings", json={
        "key": "mf_box",
        "display_name": "Multi Field Box",
        "pipeline": "detector_ocr",
        "sub_regions": ["lot", "exp", "product", "size"],
        "detection_mode": "multi_field",
    })
    assert r.status_code == 201, r.text

    body = client.get("/api/packagings/mf_box").json()
    assert body["detection_mode"] == "multi_field"
    assert body["sub_regions"] == ["lot", "exp", "product", "size"]


def test_patch_draft(client):
    r = client.patch("/api/packagings/test_box", json={
        "display_name": "Renamed Box",
    })
    assert r.status_code == 200
    assert r.json()["display_name"] == "Renamed Box"


def test_upload_image(client):
    # 1×1 PNG ('iVBORw0KGgoAAAANSUhEUgAAAA...')
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc"
        b"\xcf\xc0\x00\x00\x00\x03\x00\x01\xa3\xa3\xd2\xc0\x00\x00\x00\x00"
        b"IEND\xaeB`\x82"
    )
    r = client.post(
        "/api/packagings/test_box/images",
        files=[("files", ("a.png", png, "image/png")), ("files", ("b.png", png, "image/png"))],
    )
    assert r.status_code == 200, r.text
    assert r.json()["uploaded"] == 2
    assert r.json()["total_images"] == 2


def test_list_images(client):
    r = client.get("/api/packagings/test_box/images")
    assert r.status_code == 200
    assert len(r.json()["images"]) == 2


def test_save_config(client):
    r = client.post("/api/packagings/test_box/config", json={
        "lot_patterns": [r"^[A-Z]{2}\d{4,}.*$"],
        "fields_extracted": ["lot", "exp"],
        "sheet_checks": ["lot", "exp"],
        "message_template_key": "default_full",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "configured"
    assert body["config"]["fields_extracted"] == ["lot", "exp"]


def test_save_config_rejects_bad_regex(client):
    r = client.post("/api/packagings/test_box/config", json={
        "lot_patterns": ["[unclosed"],
    })
    assert r.status_code == 422


def test_save_config_product_aliases_round_trip(client):
    aliases = [
        {"canonical": "Houjicha Powder", "keywords": ["houjicha"]},
        {"canonical": "Excellent Rich 95%", "keywords": ["excellent rich", "rich"]},
    ]
    r = client.post("/api/packagings/test_box/config", json={
        "lot_patterns": [r"^[A-Z]{2}\d{4,}.*$"],
        "fields_extracted": ["lot", "product"],
        "sheet_checks": ["lot", "product"],
        "message_template_key": "default_full",
        "product_aliases": aliases,
    })
    assert r.status_code == 200, r.text
    assert r.json()["config"]["product_aliases"] == aliases
    # GET must echo it back (PackagingResponse keeps it inside config dict)
    g = client.get("/api/packagings/test_box")
    assert g.status_code == 200
    assert g.json()["config"]["product_aliases"] == aliases


def test_cannot_delete_active(client):
    r = client.delete("/api/packagings/back_label")
    assert r.status_code == 403


def test_delete_draft(client):
    r = client.delete("/api/packagings/test_box")
    assert r.status_code == 200
    # verify gone
    assert client.get("/api/packagings/test_box").status_code == 404


# ─── /samples + /crop endpoints ──────────────────────────

import pytest

needs_models = pytest.mark.skipif(
    not Path("models/detector.pt").exists(),
    reason="models/detector.pt not found",
)
needs_active_images = pytest.mark.skipif(
    not Path("images/back_label").exists(),
    reason="images/back_label not found",
)


def test_samples_qr_scanner_returns_empty_regions(client):
    """import_sticker (pipeline=qr_scanner) → samples มี regions ว่างเสมอ"""
    r = client.get("/api/packagings/import_sticker/samples?count=2")
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline"] == "qr_scanner"
    for s in body["samples"]:
        assert s["regions"] == []
        assert "name" in s and "original_url" in s


def test_samples_unknown_key_returns_empty(client):
    """Key ไม่มีอยู่ → samples ว่าง (ไม่ error)"""
    r = client.get("/api/packagings/nonexistent_xyz/samples")
    assert r.status_code == 200
    assert r.json()["samples"] == []


@needs_models
@needs_active_images
def test_samples_detector_ocr_returns_regions(client):
    """back_label → samples แต่ละรูปมี regions พร้อม bbox + crop_url"""
    r = client.get("/api/packagings/back_label/samples?count=2")
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline"] == "detector_ocr"
    assert len(body["samples"]) >= 1
    # อย่างน้อย 1 รูปต้องมี region
    has_region = any(s["regions"] for s in body["samples"])
    assert has_region, "no regions found in any sample"
    for s in body["samples"]:
        for region in s["regions"]:
            assert "bbox" in region
            assert "crop_url" in region
            assert region["crop_url"].endswith(f"/crop/{region.get('idx', 0)}") or "/crop/" in region["crop_url"]


@needs_models
@needs_active_images
def test_crop_endpoint_serves_jpeg(client):
    """/images/{name}/crop/0 → JPEG bytes"""
    # Use first image from back_label
    listing = client.get("/api/packagings/back_label/samples?count=1").json()
    if not listing["samples"] or not listing["samples"][0]["regions"]:
        pytest.skip("no regions to test crop endpoint")
    name = listing["samples"][0]["name"]
    r = client.get(f"/api/packagings/back_label/images/{name}/crop/0")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
    # JPEG magic bytes
    assert r.content[:2] == b"\xff\xd8"


def test_crop_404_for_unknown(client):
    r = client.get("/api/packagings/nonexistent/images/x.jpg/crop/0")
    assert r.status_code == 404


# ─── Annotation endpoints ──────────────────────────────

_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc"
    b"\xcf\xc0\x00\x00\x00\x03\x00\x01\xa3\xa3\xd2\xc0\x00\x00\x00\x00"
    b"IEND\xaeB`\x82"
)


@pytest.fixture
def annot_draft(client):
    """Create a draft with 2 uploaded images for annotation tests."""
    client.post("/api/packagings", json={
        "key": "annot_test", "display_name": "Annot Test",
    })
    client.post(
        "/api/packagings/annot_test/images",
        files=[
            ("files", ("img_a.png", _PNG_1x1, "image/png")),
            ("files", ("img_b.png", _PNG_1x1, "image/png")),
        ],
    )
    yield "annot_test"
    client.delete("/api/packagings/annot_test")


def test_annotations_list_returns_unlabeled(client, annot_draft):
    r = client.get(f"/api/packagings/{annot_draft}/annotations")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["labeled"] == 0
    assert all(not it["labeled"] for it in body["items"])


def test_annotations_list_active_returns_404(client):
    r = client.get("/api/packagings/back_label/annotations")
    assert r.status_code == 404


def test_save_annotation_then_list_shows_labeled(client, annot_draft):
    r = client.put(
        f"/api/packagings/{annot_draft}/annotations/img_a.png",
        json={"bboxes": [{"x1": 10, "y1": 20, "x2": 100, "y2": 80}]},
    )
    assert r.status_code == 200, r.text
    listing = client.get(f"/api/packagings/{annot_draft}/annotations").json()
    assert listing["labeled"] == 1
    a_item = next(it for it in listing["items"] if it["name"] == "img_a.png")
    assert a_item["labeled"] is True
    assert a_item["bbox_count"] == 1


def test_get_annotation_returns_saved_bboxes(client, annot_draft):
    client.put(
        f"/api/packagings/{annot_draft}/annotations/img_a.png",
        json={"bboxes": [{"x1": 5, "y1": 5, "x2": 50, "y2": 50, "label": "lot"}]},
    )
    r = client.get(f"/api/packagings/{annot_draft}/annotations/img_a.png")
    assert r.status_code == 200
    assert len(r.json()["bboxes"]) == 1
    assert r.json()["bboxes"][0]["x1"] == 5


def test_save_annotation_rejects_invalid_bbox(client, annot_draft):
    # x2 < x1
    r = client.put(
        f"/api/packagings/{annot_draft}/annotations/img_a.png",
        json={"bboxes": [{"x1": 100, "y1": 0, "x2": 50, "y2": 80}]},
    )
    assert r.status_code == 422


def test_delete_annotation(client, annot_draft):
    client.put(
        f"/api/packagings/{annot_draft}/annotations/img_a.png",
        json={"bboxes": [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}]},
    )
    r = client.delete(f"/api/packagings/{annot_draft}/annotations/img_a.png")
    assert r.status_code == 200
    listing = client.get(f"/api/packagings/{annot_draft}/annotations").json()
    assert listing["labeled"] == 0


# ─── Phase 4 — Eval + Deploy ─────────────────────────────

def test_eval_404_without_training(client, annot_draft):
    r = client.get(f"/api/packagings/{annot_draft}/eval")
    assert r.status_code == 404


def test_eval_with_existing_eval_json(client, annot_draft, tmp_path):
    """If eval.json exists locally, /eval returns it + hard-floor check."""
    import json as _json
    from pathlib import Path
    eval_dir = Path(f"data/drafts/{annot_draft}/models")
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_data = {
        "detector_mAP_50": 0.78,
        "precision": 0.85,
        "recall": 0.72,
        "epochs": 60,
    }
    (eval_dir / "eval.json").write_text(_json.dumps(eval_data))

    try:
        r = client.get(f"/api/packagings/{annot_draft}/eval")
        assert r.status_code == 200
        body = r.json()
        assert body["eval"]["detector_mAP_50"] == 0.78
        assert body["hard_floor"]["passed"] is True
    finally:
        # เขียนลง data/drafts จริง (ไม่ใช่ DRAFT_DIR ที่ isolate) — ต้องลบ
        # ไม่งั้น test_eval_404_without_training fail ในรอบรันถัดไป
        shutil.rmtree(f"data/drafts/{annot_draft}", ignore_errors=True)


def test_eval_hard_floor_fails(client, annot_draft):
    """Eval below thresholds → hard_floor.passed False."""
    import json as _json
    from pathlib import Path
    eval_dir = Path(f"data/drafts/{annot_draft}/models")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval.json").write_text(_json.dumps({
        "detector_mAP_50": 0.40,  # < 0.65
        "precision": 0.50,        # < 0.70
        "recall": 0.30,
    }))

    try:
        r = client.get(f"/api/packagings/{annot_draft}/eval")
        assert r.status_code == 200
        floor = r.json()["hard_floor"]
        assert floor["passed"] is False
        assert len(floor["failures"]) >= 1
        metrics_failed = {f["metric"] for f in floor["failures"]}
        assert "detector_mAP_50" in metrics_failed
    finally:
        shutil.rmtree(f"data/drafts/{annot_draft}", ignore_errors=True)


def test_deploy_blocks_without_eval(client, annot_draft):
    r = client.post(f"/api/packagings/{annot_draft}/deploy")
    assert r.status_code == 400


def test_deploy_blocks_on_hard_floor_fail(client, annot_draft):
    import json as _json
    from pathlib import Path
    eval_dir = Path(f"data/drafts/{annot_draft}/models")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval.json").write_text(_json.dumps({
        "detector_mAP_50": 0.30, "precision": 0.40, "recall": 0.20,
    }))
    try:
        r = client.post(f"/api/packagings/{annot_draft}/deploy")
        assert r.status_code == 400
    finally:
        shutil.rmtree(f"data/drafts/{annot_draft}", ignore_errors=True)


def test_deploy_writes_yaml(client, annot_draft):
    """Deploy with passing eval → YAML file appears in config/packagings/."""
    import json as _json
    from pathlib import Path
    eval_dir = Path(f"data/drafts/{annot_draft}/models")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval.json").write_text(_json.dumps({
        "detector_mAP_50": 0.80, "precision": 0.90, "recall": 0.85,
    }))
    # Save some config first
    client.post(f"/api/packagings/{annot_draft}/config", json={
        "lot_patterns": [r"^[A-Z]{2}\d{4,}.*$"],
        "fields_extracted": ["lot", "exp"],
        "sheet_checks": ["lot"],
        "message_template_key": "lot_only",
    })

    yaml_path = Path(f"config/packagings/{annot_draft}.yaml")
    try:
        r = client.post(f"/api/packagings/{annot_draft}/deploy")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["deployed"] is True
        assert yaml_path.exists()
        # Verify YAML has expected fields
        import yaml as _yaml
        cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert cfg["key"] == annot_draft
        assert "lot" in cfg["fields_extracted"]
    finally:
        if yaml_path.exists():
            yaml_path.unlink()
        shutil.rmtree(f"data/drafts/{annot_draft}", ignore_errors=True)


# ─── Active packaging — clone / archive / unarchive / overwrite ─────────

@pytest.fixture
def fake_active(client):
    """Drop a temporary active YAML into config/packagings/ and clean up after.

    We don't touch the real 6 packagings so other tests stay deterministic.
    """
    import yaml as _yaml
    from pathlib import Path
    key = "edit_fixture_pkg"
    yaml_path = Path(f"config/packagings/{key}.yaml")
    data = {
        "key": key,
        "display_name": "Edit Fixture",
        "pipeline": "detector_ocr",
        "conf_threshold": 0.7,
        "accuracy": 0.92,
        "gate_on_lot": True,
        "lot_short_fallback": False,
        "sub_regions": [],
        "lot_patterns": [r"(?i)LOT\s*([A-Z0-9]+)"],
        "fields_extracted": ["lot"],
        "sheet_checks": ["lot"],
        "post_ocr_fixes": [],
        "message_template_key": "lot_only",
        "model_classifier_label": key,
        "detector_yolo_prefixes": [f"{key}_lot"],
    }
    yaml_path.write_text(_yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    # Reload registry so the new key is recognised
    from pipeline.packaging_registry import PackagingRegistry
    import main
    main.registry = PackagingRegistry()

    yield key

    # Cleanup: remove both active and archived variants, plus any backups
    for p in Path("config/packagings").glob(f"{key}.yaml*"):
        try:
            p.unlink()
        except OSError:
            pass
    # Edit-draft (DRAFT_DIR is monkeypatched to a tmpdir by the client fixture)
    import shutil
    from services import packaging_store as _ps
    edit_dir = _ps._DRAFT_DIR / f"{key}__edit"
    if edit_dir.exists():
        shutil.rmtree(edit_dir, ignore_errors=True)
    # Also clean up the literal data/drafts/{key}__edit/ that some tests write
    # to directly (eval.json staging — deploy endpoint reads from literal path)
    literal_edit = Path("data/drafts") / f"{key}__edit"
    if literal_edit.exists():
        shutil.rmtree(literal_edit, ignore_errors=True)
    main.registry = PackagingRegistry()


def test_clone_active_creates_edit_draft(client, fake_active):
    r = client.post(f"/api/packagings/{fake_active}/clone")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["key"] == f"{fake_active}__edit"
    # The cloned draft should expose the parent's display_name + config
    assert body["display_name"] == "Edit Fixture"

    from services import packaging_store
    meta = packaging_store.get_draft(f"{fake_active}__edit")
    assert meta is not None
    assert meta["parent_key"] == fake_active
    assert meta["config"]["lot_patterns"] == [r"(?i)LOT\s*([A-Z0-9]+)"]


def test_clone_starts_at_draft_status_until_first_upload(client, fake_active):
    """Clone has no images of its own — status must be 'draft' so the wizard
    resumes at step 2 (upload), not step 3 (annotate). First image upload
    bumps it to 'uploading' like any fresh draft."""
    r = client.post(f"/api/packagings/{fake_active}/clone")
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "draft"

    edit_key = f"{fake_active}__edit"
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc"
        b"\xcf\xc0\x00\x00\x00\x03\x00\x01\xa3\xa3\xd2\xc0\x00\x00\x00\x00"
        b"IEND\xaeB`\x82"
    )
    r2 = client.post(
        f"/api/packagings/{edit_key}/images",
        files=[("files", ("a.png", png, "image/png"))],
    )
    assert r2.status_code == 200, r2.text

    from services import packaging_store
    meta = packaging_store.get_draft(edit_key)
    assert meta["status"] == "uploading"


def test_clone_blocks_when_edit_draft_exists(client, fake_active):
    r1 = client.post(f"/api/packagings/{fake_active}/clone")
    assert r1.status_code == 201
    r2 = client.post(f"/api/packagings/{fake_active}/clone")
    assert r2.status_code == 409


def test_clone_blocks_when_not_active(client):
    r = client.post("/api/packagings/totally_not_a_thing/clone")
    assert r.status_code == 404


def test_archive_renames_yaml_and_hides_from_registry(client, fake_active):
    from pathlib import Path
    r = client.post(f"/api/packagings/{fake_active}/archive")
    assert r.status_code == 200, r.text
    assert not Path(f"config/packagings/{fake_active}.yaml").exists()
    assert Path(f"config/packagings/{fake_active}.yaml.archived").exists()

    # GET shows it as archived now
    g = client.get(f"/api/packagings/{fake_active}")
    assert g.status_code == 200
    assert g.json()["status"] == "archived"

    # List also reflects the archived state
    listing = client.get("/api/packagings").json()
    by_key = {p["key"]: p for p in listing}
    assert by_key[fake_active]["status"] == "archived"


def test_unarchive_restores_active(client, fake_active):
    client.post(f"/api/packagings/{fake_active}/archive")
    r = client.post(f"/api/packagings/{fake_active}/unarchive")
    assert r.status_code == 200, r.text

    g = client.get(f"/api/packagings/{fake_active}")
    assert g.status_code == 200
    assert g.json()["status"] == "active"


def test_unarchive_404_when_not_archived(client, fake_active):
    r = client.post(f"/api/packagings/{fake_active}/unarchive")
    assert r.status_code == 404


def test_archive_404_when_not_active(client):
    r = client.post("/api/packagings/nonexistent_xyz/archive")
    assert r.status_code == 404


def test_deploy_edit_draft_overwrites_parent(client, fake_active):
    """Deploying back_label__edit must rewrite back_label.yaml + create a backup."""
    import json as _json
    import yaml as _yaml
    from pathlib import Path

    # Clone first to set up the edit-draft (with parent_key)
    r = client.post(f"/api/packagings/{fake_active}/clone")
    assert r.status_code == 201
    edit_key = r.json()["key"]

    # Stage a passing eval so deploy doesn't bail out
    eval_dir = Path(f"data/drafts/{edit_key}/models")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval.json").write_text(_json.dumps({
        "detector_mAP_50": 0.85, "precision": 0.91, "recall": 0.88,
    }))

    # Customise the edit-draft config to verify the YAML actually gets rewritten
    client.post(f"/api/packagings/{edit_key}/config", json={
        "lot_patterns": [r"^CHANGED\d+$"],
        "fields_extracted": ["lot", "exp"],
        "sheet_checks": ["lot"],
        "message_template_key": "lot_only",
    })

    r = client.post(f"/api/packagings/{edit_key}/deploy")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_key"] == fake_active   # wrote to parent, not edit key

    # Backup of the original YAML must exist
    bak_files = list(Path("config/packagings").glob(f"{fake_active}.yaml.bak-*"))
    assert len(bak_files) >= 1

    # YAML for parent now reflects the edited config
    cfg = _yaml.safe_load(Path(f"config/packagings/{fake_active}.yaml").read_text(encoding="utf-8"))
    assert cfg["lot_patterns"] == [r"^CHANGED\d+$"]

    # The edit-draft itself is consumed on success
    from services import packaging_store
    assert packaging_store.get_draft(edit_key) is None

    # No leftover {edit_key}.yaml — overwrite path must NOT write to the draft key
    assert not Path(f"config/packagings/{edit_key}.yaml").exists()

    # Cleanup backups so they don't interfere with subsequent runs
    for b in bak_files:
        b.unlink()


def test_deploy_rolls_back_on_hard_floor_fail(client, fake_active):
    """When eval fails the hard floor, the original YAML must stay intact."""
    import json as _json
    import yaml as _yaml
    from pathlib import Path

    r = client.post(f"/api/packagings/{fake_active}/clone")
    edit_key = r.json()["key"]

    eval_dir = Path(f"data/drafts/{edit_key}/models")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval.json").write_text(_json.dumps({
        "detector_mAP_50": 0.20, "precision": 0.30, "recall": 0.15,
    }))

    original_yaml = Path(f"config/packagings/{fake_active}.yaml").read_text(encoding="utf-8")

    r = client.post(f"/api/packagings/{edit_key}/deploy")
    assert r.status_code == 400

    # Original YAML untouched — no rollback restoration needed because we never wrote
    current = Path(f"config/packagings/{fake_active}.yaml").read_text(encoding="utf-8")
    assert current == original_yaml


def test_backup_rotation_keeps_three(tmp_path, monkeypatch):
    """backup_artifacts() must rotate to the most recent 3 per key."""
    import time
    from pathlib import Path
    from services import cloudrun_deployer

    # Redirect packaging dir + models dir to tmp_path
    # _packaging_dir() reads OCR_CONFIG_DIR at call time, so patch the env var.
    # It returns Path(OCR_CONFIG_DIR) / "packagings", so point OCR_CONFIG_DIR at tmp_path
    # so that _packaging_dir() resolves to tmp_path / "packagings".
    pkg_dir = tmp_path / "packagings"
    pkg_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setenv("OCR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(cloudrun_deployer, "_MODELS_DIR", models_dir)

    yaml_path = pkg_dir / "demo.yaml"
    yaml_path.write_text("key: demo\n")
    (models_dir / "detector.pt").write_bytes(b"fake-detector")

    # Five deploys should leave only the most recent 3 backups
    for _ in range(5):
        cloudrun_deployer.backup_artifacts("demo")
        time.sleep(1.05)  # _bak_suffix has 1-second resolution

    yaml_baks = sorted(pkg_dir.glob("demo.yaml.bak-*"))
    model_baks = sorted(models_dir.glob("detector.pt.bak-*"))
    assert len(yaml_baks) == 3
    assert len(model_baks) == 3


def test_predict_returns_archived_status_when_class_archived(client, fake_active, monkeypatch):
    """Classifier-predicted class with archived YAML → status=archived_class."""
    import main as main_module
    client.post(f"/api/packagings/{fake_active}/archive")

    class FakeClassifier:
        def predict(self, _bytes):
            return fake_active, 0.99

    monkeypatch.setattr(main_module, "classifier", FakeClassifier())

    # 1×1 PNG payload — passes the content-type check
    png = _PNG_1x1
    r = client.post(
        "/predict",
        params={"sheet_id": "dummy", "sheet_gid": 0},
        files={"file": ("img.png", png, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "archived_class"
    assert body["class"] == fake_active


# ─── training/full/start — dataset publish ───────────────

def test_training_full_start_publishes_dataset(client, monkeypatch):
    """full/start must publish dataset (no zip) then gen + upload notebook."""
    from services import packaging_store

    # create a draft using the same route/payload shape as other tests in this file
    client.post("/api/packagings", json={"key": "fullpub", "display_name": "x"})

    monkeypatch.setattr(
        packaging_store, "list_annotation_status",
        lambda key: [{"name": f"i{n}.jpg", "labeled": True, "bbox_count": 1}
                     for n in range(30)],
    )

    fake_summary = {
        "images_uploaded": 10, "images_skipped": 0,
        "train_count": 8, "val_count": 2,
        "new_classes": ["fullpub_lot"], "class_ids": {"lot": 9},
        "total_classes": 10,
    }
    publish_mock = MagicMock(return_value=fake_summary)
    monkeypatch.setattr("services.dataset_publisher.publish", publish_mock)

    drive_mock = MagicMock()
    drive_mock.create_folder.return_value = "run-folder"
    drive_mock.upload_bytes.return_value = "nb-id"
    monkeypatch.setattr(
        "services.drive_client.DriveClient", MagicMock(return_value=drive_mock)
    )
    res = client.post("/api/packagings/fullpub/training/full/start")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["dataset"] == fake_summary
    assert "colab.research.google.com" in body["colab_url"]
    publish_mock.assert_called_once_with("fullpub", drive=ANY, progress_cb=ANY)
    # no notebook is generated/uploaded anymore — start only publishes the dataset
    drive_mock.upload_bytes.assert_not_called()
    drive_mock.create_folder.assert_not_called()


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


def test_training_full_start_allows_grouped_multi_field(client, monkeypatch):
    """Regression: a multi_field draft whose sub_regions are GROUPS (e.g. lot_exp)
    must pass the field-coverage gate. fields_extracted lists individual tokens
    while sub_regions lists composite groups, so the gate must compare against the
    groups' member fields, not the raw composite strings."""
    from services import packaging_store

    monkeypatch.setattr(
        packaging_store, "get_draft",
        lambda key: {
            "key": key, "display_name": "G", "pipeline": "detector_ocr",
            "detection_mode": "multi_field",
            "sub_regions": ["lot_exp", "product_size"],
            "config": {"fields_extracted": ["lot", "exp", "product", "size"]},
        },
    )
    monkeypatch.setattr(
        packaging_store, "list_annotation_status",
        lambda key: [{"name": f"i{n}.jpg", "labeled": True, "bbox_count": 1}
                     for n in range(30)],
    )
    monkeypatch.setattr(
        "services.dataset_publisher.publish",
        MagicMock(return_value={
            "images_uploaded": 10, "images_skipped": 0,
            "train_count": 8, "val_count": 2,
            "new_classes": ["groupdraft_lot_exp"], "class_ids": {"lot_exp": 9},
            "total_classes": 10,
        }),
    )
    drive_mock = MagicMock()
    drive_mock.create_folder.return_value = "run-folder"
    drive_mock.upload_bytes.return_value = "nb-id"
    monkeypatch.setattr(
        "services.drive_client.DriveClient", MagicMock(return_value=drive_mock))

    res = client.post("/api/packagings/groupdraft/training/full/start")
    assert res.status_code == 200, res.text


def test_training_full_done_is_manual_now(client, monkeypatch):
    """Manual model sync: done returns a 400 telling the user to sync via the
    model registry, instead of pulling from a (no-longer-created) output folder."""
    from services import packaging_store
    monkeypatch.setattr(
        packaging_store, "get_draft",
        lambda key: {"key": key, "training_run": {"kind": "full"}},
    )
    res = client.post("/api/packagings/anykey/training/full/done")
    assert res.status_code == 400
    assert "manual" in res.json()["detail"].lower()


# ─── Training progress polling ───────────────────────────

def test_training_progress_idle_when_nothing_running(client):
    res = client.get("/api/packagings/no-such-key/training/progress")
    assert res.status_code == 200
    assert res.json()["phase"] == "idle"


def test_training_progress_reflects_reported_snapshot(client):
    from services import progress_store

    progress_store.report("progkey", "upload_images", done=2, total=8, detail="x.jpg")
    try:
        res = client.get("/api/packagings/progkey/training/progress")
        assert res.status_code == 200
        body = res.json()
        assert body["phase"] == "upload_images"
        assert body["percent"] == 25
        assert body["detail"] == "x.jpg"
    finally:
        progress_store.clear("progkey")


# ─── PUT /{key}/conf — runtime tuning override (ADR 0004) ─

@pytest.fixture
def conf_overrides_env(tmp_path, monkeypatch):
    """Isolate override storage — local mode, file in tmp."""
    monkeypatch.delenv("DRIVE_CONFIG_OVERRIDES_FILE_ID", raising=False)
    monkeypatch.setenv("CONFIG_OVERRIDES_PATH", str(tmp_path / "config_overrides.json"))


def test_put_conf_updates_active(client, fake_active, conf_overrides_env):
    r = client.put(f"/api/packagings/{fake_active}/conf", json={"conf_threshold": 0.75})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["key"] == fake_active
    assert body["conf_threshold"] == 0.75
    assert body["previous"] == 0.7  # fixture YAML value

    # GET must reflect the merged value immediately
    g = client.get(f"/api/packagings/{fake_active}")
    assert g.status_code == 200
    assert g.json()["conf_threshold"] == 0.75


def test_put_conf_rejects_out_of_range(client, fake_active, conf_overrides_env):
    for bad in (0.4, 0.96, 0.0, 1.0):
        r = client.put(f"/api/packagings/{fake_active}/conf", json={"conf_threshold": bad})
        assert r.status_code == 422, f"value {bad} should be rejected"


def test_put_conf_404_for_unknown_key(client, conf_overrides_env):
    r = client.put("/api/packagings/nonexistent_xyz/conf", json={"conf_threshold": 0.7})
    assert r.status_code == 404


def test_put_conf_404_for_draft(client, conf_overrides_env):
    client.post("/api/packagings", json={"key": "confdraft", "display_name": "x"})
    try:
        r = client.put("/api/packagings/confdraft/conf", json={"conf_threshold": 0.7})
        assert r.status_code == 404
    finally:
        client.delete("/api/packagings/confdraft")


def test_conf_override_survives_archive_unarchive(client, fake_active, conf_overrides_env):
    """Registry reloads ที่อื่น (archive/unarchive) ต้องไม่ทำ override หาย."""
    r = client.put(f"/api/packagings/{fake_active}/conf", json={"conf_threshold": 0.85})
    assert r.status_code == 200, r.text

    assert client.post(f"/api/packagings/{fake_active}/archive").status_code == 200
    assert client.post(f"/api/packagings/{fake_active}/unarchive").status_code == 200

    g = client.get(f"/api/packagings/{fake_active}")
    assert g.json()["conf_threshold"] == 0.85


def test_put_conf_502_when_persist_fails(client, fake_active, conf_overrides_env, monkeypatch):
    from services import config_overrides

    monkeypatch.setattr(
        config_overrides, "save_conf_threshold",
        MagicMock(side_effect=RuntimeError("drive down")),
    )
    r = client.put(f"/api/packagings/{fake_active}/conf", json={"conf_threshold": 0.9})
    assert r.status_code == 502

    # Nothing changed locally — registry still serves the YAML value
    g = client.get(f"/api/packagings/{fake_active}")
    assert g.json()["conf_threshold"] == 0.7


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
