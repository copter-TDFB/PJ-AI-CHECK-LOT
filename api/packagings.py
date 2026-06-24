"""Wizard API — manage packaging drafts."""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.schemas import (
    AnnotationSave,
    ConfThresholdResponse,
    ConfThresholdUpdate,
    PackagingConfigUpdate,
    PackagingCreate,
    ProductAliasesResponse,
    ProductAliasesUpdate,
    PackagingResponse,
    PackagingUpdate,
    RegexPreviewRequest,
    RegexPreviewResponse,
)
from services import packaging_store
from services.drive_client import DriveClient
from services.regex_generator import generate_regex, preview_matches
from utils.field_groups import parse_group

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/packagings", tags=["packagings"])

_ACTIVE_IMAGES_DIR = Path("images")
_DRIVE_SAMPLE_CACHE = Path(
    os.getenv("DRIVE_SAMPLE_CACHE_DIR", str(Path(tempfile.gettempdir()) / "drive_samples"))
)
_CROP_CACHE_DIR = Path(os.getenv("CROP_CACHE_DIR", "data/crops"))
_DETECTOR_MODEL_PATH = Path(os.getenv("MODEL_DETECTOR_PATH", "models/detector.pt"))
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _packaging_yaml_dir() -> Path:
    """Packagings dir, honouring OCR_CONFIG_DIR like the registry. A hardcoded
    relative "config/packagings" silently read the prod config in the test
    harness (OCR_CONFIG_DIR=data/test/config) — dropping archived cards and
    cloning the wrong YAML."""
    from pipeline.packaging_registry import _config_dir
    return _config_dir() / "packagings"

# Combined Full Training notebook in Drive (built by
# scripts/build_full_training_notebook.py).
COMBINED_NOTEBOOK_FILE_ID = "1_jq1jWpstRKj5UbP1PFwEcW_dgmL4R6Z"


@router.get("", response_model=list[PackagingResponse])
def list_packagings():
    """รวม active (จาก YAML) + drafts (จาก data/drafts)."""
    import main  # lazy import — registry initialized in lifespan

    items: list[PackagingResponse] = []
    promoted_keys: set[str] = set()  # keys already shown as active/archived

    if main.registry is not None:
        for key in main.registry.all_keys():
            cfg = main.registry.get(key)
            if cfg is None:
                continue
            promoted_keys.add(cfg.key)
            items.append(PackagingResponse(
                key=cfg.key,
                display_name=cfg.display_name,
                pipeline=cfg.pipeline,
                status="active",
                image_count=cfg.image_count if cfg.image_count is not None else _count_active_images(key),
                conf_threshold=cfg.conf_threshold,
                accuracy=cfg.accuracy,
                sub_regions=cfg.sub_regions,
                detection_mode=cfg.detection_mode,
            ))

        # Archived packagings — YAML still on disk but renamed to *.yaml.archived.
        arch_dir = _packaging_yaml_dir()
        for arch_key in main.registry.archived_keys():
            arch_yaml = arch_dir / f"{arch_key}.yaml.archived"
            try:
                data = yaml.safe_load(arch_yaml.read_text(encoding="utf-8"))
            except Exception:
                continue
            promoted_keys.add(arch_key)
            items.append(PackagingResponse(
                key=arch_key,
                display_name=data.get("display_name", arch_key),
                pipeline=data.get("pipeline", "detector_ocr"),
                status="archived",
                image_count=int(data["image_count"]) if data.get("image_count") is not None
                    else _count_active_images(arch_key),
                conf_threshold=float(data.get("conf_threshold", 0.6)),
                accuracy=float(data["accuracy"]) if data.get("accuracy") is not None else None,
            ))

    # Drafts — skip any whose key is already an active/archived packaging. A
    # fresh deploy leaves the draft behind with status="deployed"; without this
    # guard it renders as a SECOND card duplicating the live/offline packaging.
    for draft in packaging_store.list_drafts():
        if draft.get("key") in promoted_keys:
            continue
        items.append(PackagingResponse(**draft))

    return items


@router.post("", response_model=PackagingResponse, status_code=201)
def create_packaging(body: PackagingCreate):
    """สร้าง draft ใหม่ — step 1 ของ wizard."""
    import main

    if main.registry is not None and main.registry.get(body.key) is not None:
        raise HTTPException(409, f"key '{body.key}' is used by an active packaging")

    try:
        meta = packaging_store.create_draft(
            key=body.key,
            display_name=body.display_name,
            description=body.description,
            pipeline=body.pipeline.value,
            sub_regions=body.sub_regions,
            detection_mode=body.detection_mode.value,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc))

    meta["image_count"] = 0
    return PackagingResponse(**meta)


@router.get("/{key}", response_model=PackagingResponse)
def get_packaging(key: str):
    import main

    if main.registry is not None:
        cfg = main.registry.get(key)
        if cfg is not None:
            return PackagingResponse(
                key=cfg.key,
                display_name=cfg.display_name,
                pipeline=cfg.pipeline,
                status="active",
                image_count=cfg.image_count if cfg.image_count is not None else _count_active_images(key),
                conf_threshold=cfg.conf_threshold,
                accuracy=cfg.accuracy,
                sub_regions=cfg.sub_regions,
                detection_mode=cfg.detection_mode,
                product_aliases=cfg.product_aliases,
                fields_extracted=cfg.fields_extracted,
            )
        if main.registry.is_archived(key):
            arch_yaml = _packaging_yaml_dir() / f"{key}.yaml.archived"
            try:
                data = yaml.safe_load(arch_yaml.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            return PackagingResponse(
                key=key,
                display_name=data.get("display_name", key),
                pipeline=data.get("pipeline", "detector_ocr"),
                status="archived",
                image_count=int(data["image_count"]) if data.get("image_count") is not None
                    else _count_active_images(key),
                conf_threshold=float(data.get("conf_threshold", 0.6)),
                accuracy=float(data["accuracy"]) if data.get("accuracy") is not None else None,
            )

    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"packaging '{key}' not found")
    return PackagingResponse(**draft)


@router.patch("/{key}", response_model=PackagingResponse)
def update_packaging(key: str, body: PackagingUpdate):
    """Update draft — display_name / description / pipeline."""
    updates: dict = {}
    if body.display_name is not None:
        updates["display_name"] = body.display_name
    if body.description is not None:
        updates["description"] = body.description
    if body.pipeline is not None:
        updates["pipeline"] = body.pipeline.value

    meta = packaging_store.update_draft(key, **updates)
    if meta is None:
        raise HTTPException(404, f"draft '{key}' not found")
    return PackagingResponse(**meta)


@router.delete("/{key}")
def delete_packaging(key: str):
    """ลบ draft. Active packaging ลบไม่ได้."""
    import main

    if main.registry is not None and main.registry.get(key) is not None:
        raise HTTPException(403, "cannot delete an active packaging")

    if not packaging_store.delete_draft(key):
        raise HTTPException(404, f"draft '{key}' not found")
    return {"deleted": key}


# ─── Edit active (clone → wizard → re-deploy) ──────────


@router.post("/{key}/clone", response_model=PackagingResponse, status_code=201)
def clone_active(key: str):
    """Create an edit-draft `{key}__edit` from an active packaging.

    The wizard reopens at step 2 (image upload) — config is pre-populated
    from the active YAML, parent images stay read-only on disk.
    """
    import main

    if main.registry is None or main.registry.get(key) is None:
        raise HTTPException(404, f"active packaging '{key}' not found")

    yaml_path = _packaging_yaml_dir() / f"{key}.yaml"
    if not yaml_path.exists():
        raise HTTPException(500, f"YAML missing for active '{key}'")

    try:
        active_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"failed to read active YAML: {e}")

    try:
        meta = packaging_store.clone_from_active(parent_key=key, active_yaml=active_yaml)
    except ValueError as exc:
        raise HTTPException(409, str(exc))

    meta["image_count"] = 0
    return PackagingResponse(**meta)


@router.post("/{key}/archive")
def archive_packaging_endpoint(key: str):
    """Soft-delete an active packaging — rename YAML, reload registry."""
    import main
    from services import cloudrun_deployer

    if main.registry is None or main.registry.get(key) is None:
        raise HTTPException(404, f"active packaging '{key}' not found")

    try:
        dst = cloudrun_deployer.archive_packaging(key)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    try:
        main.reload_registry()
    except Exception as e:
        logger.exception("registry reload failed after archive")
        raise HTTPException(500, f"registry reload failed: {e}")

    try:
        gcs_result = cloudrun_deployer.set_archived_in_gcs(key, True)
    except Exception as e:  # GCS sync is best-effort — local state already changed
        logger.warning("GCS archive sync failed for %s: %s", key, e)
        gcs_result = {"published": False, "reason": str(e)}
    return {"archived": key, "yaml_path": str(dst), "gcs": gcs_result}


@router.post("/{key}/unarchive")
def unarchive_packaging_endpoint(key: str):
    """Restore an archived packaging back to active."""
    import main
    from services import cloudrun_deployer

    try:
        dst = cloudrun_deployer.unarchive_packaging(key)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except FileExistsError as e:
        raise HTTPException(409, str(e))

    try:
        main.reload_registry()
    except Exception as e:
        logger.exception("registry reload failed after unarchive")
        raise HTTPException(500, f"registry reload failed: {e}")

    try:
        gcs_result = cloudrun_deployer.set_archived_in_gcs(key, False)
    except Exception as e:  # GCS sync is best-effort — local state already changed
        logger.warning("GCS unarchive sync failed for %s: %s", key, e)
        gcs_result = {"published": False, "reason": str(e)}
    return {"unarchived": key, "yaml_path": str(dst), "gcs": gcs_result}


@router.put("/{key}/conf", response_model=ConfThresholdResponse)
def update_conf_threshold(key: str, body: ConfThresholdUpdate):
    """Tune conf_threshold ของ active packaging — ไม่ต้อง retrain (ADR 0004).

    เขียน Drive ให้สำเร็จก่อนแล้วค่อย reload registry — Drive ล่ม → 502
    และไม่มีอะไรเปลี่ยน เพื่อไม่ให้ instance แตกค่ากับ storage
    """
    import main

    cfg = main.registry.get(key) if main.registry else None
    if cfg is None:
        raise HTTPException(404, f"active packaging '{key}' not found")
    previous = cfg.conf_threshold if cfg.conf_threshold is not None else 0.6

    from services import config_overrides

    try:
        config_overrides.save_conf_threshold(key, body.conf_threshold)
    except Exception as e:
        logger.exception("conf override persist failed for %s", key)
        raise HTTPException(502, f"failed to persist conf override: {e}")

    try:
        main.reload_registry()
    except Exception as e:
        logger.exception("registry reload failed after conf update")
        raise HTTPException(500, f"registry reload failed: {e}")

    return ConfThresholdResponse(
        key=key, conf_threshold=body.conf_threshold, previous=previous,
    )


@router.put("/{key}/product-aliases", response_model=ProductAliasesResponse)
def update_product_aliases(key: str, body: ProductAliasesUpdate):
    """Edit the product names an active class reads — no retrain (mirrors /conf).

    Persist-first: a storage failure returns 502 and changes nothing, so the
    instance never diverges from the stored overrides.
    """
    import main

    cfg = main.registry.get(key) if main.registry else None
    if cfg is None:
        raise HTTPException(404, f"active packaging '{key}' not found")
    if "product" not in cfg.fields_extracted:
        raise HTTPException(400, f"packaging '{key}' does not read a product name")

    aliases = [a.model_dump() for a in body.product_aliases]
    for a in aliases:
        if not a["canonical"].strip() or not [k for k in a["keywords"] if k.strip()]:
            raise HTTPException(400, "each alias needs a canonical and at least one keyword")

    previous = cfg.product_aliases

    from services import config_overrides

    try:
        config_overrides.save_product_aliases(key, aliases)
    except Exception as e:
        logger.exception("product_aliases override persist failed for %s", key)
        raise HTTPException(502, f"failed to persist override: {e}")

    try:
        main.reload_registry()
    except Exception as e:
        logger.exception("registry reload failed after product_aliases update")
        raise HTTPException(500, f"registry reload failed: {e}")

    return ProductAliasesResponse(key=key, product_aliases=aliases, previous=previous)


@router.delete("/{key}/product-aliases", response_model=ProductAliasesResponse)
def revert_product_aliases(key: str):
    """Remove the product_aliases override → revert to the YAML/hardcoded default.

    Persist-first (mirrors the PUT). Returns the now-effective aliases (the YAML
    value, which may be empty → the class falls back to the hardcoded matcher).
    """
    import main

    cfg = main.registry.get(key) if main.registry else None
    if cfg is None:
        raise HTTPException(404, f"active packaging '{key}' not found")

    previous = cfg.product_aliases

    from services import config_overrides

    try:
        config_overrides.delete_product_aliases(key)
    except Exception as e:
        logger.exception("product_aliases override delete failed for %s", key)
        raise HTTPException(502, f"failed to persist override: {e}")

    try:
        main.reload_registry()
    except Exception as e:
        logger.exception("registry reload failed after product_aliases delete")
        raise HTTPException(500, f"registry reload failed: {e}")

    new_cfg = main.registry.get(key) if main.registry else None
    effective = new_cfg.product_aliases if new_cfg else []
    return ProductAliasesResponse(key=key, product_aliases=effective, previous=previous)


@router.post("/{key}/images")
async def upload_images(key: str, files: list[UploadFile] = File(...)):
    """อัพรูปขึ้น draft — step 2."""
    if packaging_store.get_draft(key) is None:
        raise HTTPException(404, f"draft '{key}' not found")

    saved = []
    skipped = []
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            skipped.append(f.filename)
            continue
        content = await f.read()
        path = packaging_store.save_image(key, f.filename or "image.jpg", content)
        saved.append({"name": path.name, "size": len(content)})

    return {
        "uploaded": len(saved),
        "skipped": skipped,
        "files": saved,
        "total_images": packaging_store.get_draft(key)["image_count"],
    }


@router.get("/{key}/images")
def list_images(key: str):
    """List images.

    - **active**: images live under `images/{key}/`.
    - **draft (fresh)**: images live under `data/drafts/{key}/images/`.
    - **edit-draft**: returns its own draft images AND parent's reference
      images flagged `read_only=True` so the wizard can render them dimmed.
    """
    import main

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

    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"packaging '{key}' not found")

    new_images = [{**img, "read_only": False} for img in packaging_store.list_images(key)]

    parent_key = draft.get("parent_key")
    if not parent_key:
        return {"images": new_images}

    parent_dir = packaging_store.parent_images_dir(parent_key)
    parent_images = []
    if parent_dir.exists():
        parent_images = [
            {"name": p.name, "size": p.stat().st_size, "read_only": True}
            for p in sorted(parent_dir.iterdir())
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        ]
    return {
        "images": new_images,
        "parent_images": parent_images,
        "parent_key": parent_key,
    }


@router.get("/{key}/images/{filename}")
def get_image(key: str, filename: str):
    """Serve รูปแต่ละไฟล์ — สำหรับให้ frontend แสดง thumbnail.

    For edit-drafts, falls back to the parent's images dir so the wizard
    can display read-only parent thumbnails through the same URL pattern.
    """
    import main

    safe = Path(filename).name
    if main.registry is not None and main.registry.get(key) is not None:
        candidate = _ACTIVE_IMAGES_DIR / key / safe
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        if os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "").strip():
            cached = _drive_sample_path(key, safe)
            if cached is not None:
                return FileResponse(cached)
        raise HTTPException(404, "image not found")

    p = packaging_store.image_path(key, safe)
    if p is not None:
        return FileResponse(p)

    # Edit-draft fallback: serve parent reference image
    draft = packaging_store.get_draft(key)
    if draft is not None and draft.get("parent_key"):
        parent_candidate = packaging_store.parent_images_dir(draft["parent_key"]) / safe
        if parent_candidate.exists() and parent_candidate.is_file():
            return FileResponse(parent_candidate)

    raise HTTPException(404, "image not found")


@router.delete("/{key}/images/{filename}")
def delete_image(key: str, filename: str):
    if not packaging_store.delete_image(key, filename):
        raise HTTPException(404, "image not found")
    return {"deleted": filename}


@router.post("/{key}/config", response_model=PackagingResponse)
def save_config(key: str, body: PackagingConfigUpdate):
    """บันทึก config — step 4."""
    meta = packaging_store.save_config(key, body.model_dump())
    if meta is None:
        raise HTTPException(404, f"draft '{key}' not found")
    return PackagingResponse(**meta)


@router.post("/regex/preview", response_model=RegexPreviewResponse)
def regex_preview(body: RegexPreviewRequest):
    """Generate regex จากตัวอย่าง — wizard step 4 ใช้สร้าง pattern preview."""
    pattern = generate_regex(body.examples)
    matches = preview_matches(pattern, body.examples)
    return RegexPreviewResponse(pattern=pattern, matches=matches)


@router.get("/{key}/samples")
def get_samples(key: str, count: int = 6):
    """Sample images + crop regions ที่ detector ทำจริง — สำหรับ drawer detail view.

    For qr_scanner pipeline → regions=[] (no crop).
    For detector_ocr → run detector once per image (cached on disk).
    """
    pipeline, sub_regions = _resolve_pipeline(key)
    files = _list_sample_files(key, count)
    samples = []
    for fname in files:
        regions = _ensure_crops(key, fname, pipeline, sub_regions)
        samples.append({
            "name": fname,
            "original_url": f"/api/packagings/{key}/images/{fname}",
            "regions": [
                {
                    "bbox": r["bbox"],
                    "label": r.get("label"),
                    "crop_url": f"/api/packagings/{key}/images/{fname}/crop/{r['idx']}",
                }
                for r in regions
            ],
        })
    return {"samples": samples, "pipeline": pipeline}


@router.get("/{key}/images/{filename}/crop/{idx}")
def get_crop(key: str, filename: str, idx: int):
    """Serve cropped JPEG จาก disk cache. ถ้า cache miss — สร้างก่อน."""
    safe = Path(filename).name
    crop_path = _CROP_CACHE_DIR / key / f"{safe}.{idx}.jpg"
    if not crop_path.exists():
        pipeline, sub_regions = _resolve_pipeline(key)
        _ensure_crops(key, safe, pipeline, sub_regions)
    if not crop_path.exists():
        raise HTTPException(404, "crop not found")
    return FileResponse(crop_path, media_type="image/jpeg")


# ─── Training ─────────────────

@router.get("/{key}/training/progress")
def training_progress(key: str):
    """Live progress ของ training start ที่กำลังรัน — wizard poll ทุก ~0.6s
    ระหว่างรอ POST /training/{seed|full}/start ตอบกลับ."""
    from services import progress_store

    return progress_store.get(key)


@router.post("/{key}/training/full/start")
def training_full_start(key: str):
    """Full training — publishes dataset images directly to Drive, then returns a
    static link to the combined Colab notebook. No notebook is generated."""
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")

    status = packaging_store.list_annotation_status(key)
    labeled = [it for it in status if it["labeled"]]
    if len(labeled) < 30:
        raise HTTPException(400, f"need at least 30 labeled images (have {len(labeled)})")

    if draft.get("detection_mode") == "multi_field":
        cfg = draft.get("config") or {}
        subs = draft.get("sub_regions", [])
        # sub_regions entries are groups (e.g. "lot_exp") — expand to member
        # fields before checking that every extracted field has a crop.
        crop_fields = {f for sr in subs for f in parse_group(sr)}
        missing = [f for f in cfg.get("fields_extracted", []) if f not in crop_fields]
        if missing:
            raise HTTPException(
                400,
                f"multi_field: fields {missing} have no crop sub-region — "
                "add them in step 1 or untick them in step 4",
            )

    from services import dataset_publisher, progress_store
    from services.drive_client import DriveClient

    progress_store.report(key, "starting")
    try:
        drive = DriveClient()
    except Exception as e:
        logger.exception("Drive client init failed for %s", key)
        progress_store.report(key, "error", detail=str(e))
        raise HTTPException(500, f"Drive client init failed: {e}")

    try:
        dataset_summary = dataset_publisher.publish(
            key,
            drive=drive,
            progress_cb=lambda done, total, name: progress_store.report(
                key, "upload_images", done=done, total=total, detail=name
            ),
        )
    except (FileNotFoundError, ValueError) as e:
        progress_store.report(key, "error", detail=str(e))
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        progress_store.report(key, "error", detail=str(e))
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception("dataset publish failed for %s", key)
        progress_store.report(key, "error", detail=str(e))
        raise HTTPException(500, f"Dataset publish failed: {e}")

    progress_store.report(key, "done")
    packaging_store.update_draft(
        key,
        training_run={
            "started_at": datetime.now(timezone.utc).isoformat(),
            "kind": "full",
            "dataset": dataset_summary,
        },
        status="training_full",
    )
    colab_url = (
        f"https://colab.research.google.com/drive/{COMBINED_NOTEBOOK_FILE_ID}"
    )
    return {
        "colab_url": colab_url,
        "dataset": dataset_summary,
        "labeled_count": len(labeled),
    }


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


@router.post("/{key}/training/full/done")
def training_full_done(key: str):
    """Full training เป็น manual แล้ว — ใช้ /training/full/sync
    เพื่อดึง model และ eval จาก Drive เข้า draft.

    ยกเว้นใน TEST_MODE: จำลองว่า training เสร็จ — เขียน eval.json ที่ผ่าน
    hard floor + ตั้ง status=trained เพื่อให้หน้า Deploy โผล่มาทดสอบได้
    โดยไม่ต้องรัน Colab/sync model จริง (production ยังคืน 400 เหมือนเดิม)."""
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")

    if os.getenv("TEST_MODE") == "1":
        labeled = [it for it in packaging_store.list_annotation_status(key) if it["labeled"]]
        val_count = max(1, round(len(labeled) * 0.2))
        eval_data = {
            "detector_mAP_50": 0.91,
            "precision": 0.93,
            "recall": 0.88,
            "epochs": 60,
            "imgsz": 640,
            "train_count": len(labeled) - val_count,
            "val_count": val_count,
            "simulated": True,
        }
        eval_dir = Path(os.getenv("DRAFT_DIR", "data/drafts")) / key / "models"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "eval.json").write_text(
            json.dumps(eval_data, indent=2), encoding="utf-8"
        )
        packaging_store.update_draft(key, status="trained")
        logger.info("TEST_MODE — simulated training done for '%s'", key)
        return {"simulated": True, "eval": eval_data}

    raise HTTPException(
        400,
        "Full training เป็น manual แล้ว — รัน notebook ใน Colab ให้เสร็จ, "
        "model จะเซฟลง Drive, แล้ว sync เข้าระบบผ่าน /training/full/sync "
        "(ดู docs/superpowers/specs/2026-06-15-direct-notebook-training-design.md)",
    )


@router.post("/{key}/training/full/sync")
def training_full_sync(key: str):
    """Pull the freshly Colab-trained detector + classifier + eval from Drive
    into the draft, then mark it `trained` so the Eval/Deploy screen appears.

    `eval.json` is written last by the notebook, so its absence means training
    is not finished yet (→ 409).
    """
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")

    det_folder = os.getenv("DRIVE_DETECTOR_DATASET_FOLDER_ID", "")
    cls_folder = os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "")
    if not det_folder or not cls_folder:
        raise HTTPException(
            503,
            "ยังตั้งค่า Drive dataset folders ไม่ครบ — ต้องตั้ง env "
            "DRIVE_DETECTOR_DATASET_FOLDER_ID และ DRIVE_CLASSIFIER_DATASET_FOLDER_ID บน Cloud Run",
        )

    client = DriveClient()
    eval_id = client.find_in_folder(det_folder, "eval.json")
    if eval_id is None:
        raise HTTPException(409, "ยังเทรนไม่เสร็จ — รัน Colab notebook ให้จบก่อน")
    det_id = client.find_in_folder(det_folder, "full_detector.pt")
    if det_id is None:
        raise HTTPException(409, "Drive มี eval แต่ไม่พบ full_detector.pt — เทรนยังไม่ครบ")

    # classifier lives under <classifier folder>/models/classifier.pt
    cls_models = client.find_in_folder(cls_folder, "models")
    cls_id = client.find_in_folder(cls_models, "classifier.pt") if cls_models else None
    if cls_id is None:
        raise HTTPException(409, "ไม่พบ classifier.pt ใน Drive — เทรนยังไม่ครบ")

    models_dir = Path(os.getenv("DRAFT_DIR", "data/drafts")) / key / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    detector_path = models_dir / "full_detector.pt"
    classifier_path = models_dir / "full_classifier.pt"
    eval_path = models_dir / "eval.json"
    artifacts = (eval_path, detector_path, classifier_path)

    try:
        client.download_file(det_id, detector_path)
        client.download_file(cls_id, classifier_path)
        # Local eval.json is the deploy-ready signal, so persist it last.
        client.download_file(eval_id, eval_path)
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        packaging_store.update_draft(key, status="trained")
    except Exception as e:
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
        logger.exception("training/full/sync failed for '%s'", key)
        raise HTTPException(500, f"sync failed: {e}")

    logger.info("Synced trained model for '%s' from Drive", key)
    return {"synced": True, "eval": eval_data}


@router.post("/{key}/deploy")
def deploy_packaging(key: str):
    """Promote draft → active. Two paths:

    1. **Fresh deploy** — draft has no `parent_key`. Writes a new YAML under
       the draft's key. Blocks (409) if that key is already active.
    2. **Overwrite (edit-draft)** — draft has `parent_key`. Backs up the
       active YAML + models, then writes YAML under `parent_key` and promotes
       the new detector. Auto-rollback on hard-floor failure.
    """
    import main
    from services import cloudrun_deployer, eval_thresholds

    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")

    parent_key = draft.get("parent_key")  # only set on edit-drafts
    target_key = parent_key or key

    # Fresh-deploy path may not collide with an existing active
    if parent_key is None and main.registry is not None and main.registry.get(key) is not None:
        raise HTTPException(409, f"key '{key}' already active — use /clone to edit it")

    # 1. Hard-floor gate (before any backup — fail cheap)
    eval_path = Path(os.getenv("DRAFT_DIR", "data/drafts")) / key / "models" / "eval.json"
    if not eval_path.exists():
        raise HTTPException(400, "ยังไม่มี eval — รัน full training ก่อน")
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    check = eval_thresholds.check_hard_floor(eval_data)
    if not check["passed"]:
        raise HTTPException(400, {"error": "hard floor failed", "failures": check["failures"]})

    # 2. Detect freshly-synced models (present on both fresh + edit-draft paths)
    draft_models = Path(os.getenv("DRAFT_DIR", "data/drafts")) / key / "models"
    has_synced = (draft_models / "full_detector.pt").exists() or \
                 (draft_models / "full_classifier.pt").exists()

    # 3. Backup active artifacts before overwrite. On the fresh path target_key
    #    is the new key, so its YAML backup is a no-op; the global model backup
    #    protects rollback. Model-backup rotation is globally scoped rather
    #    than key-scoped (pre-existing behavior).
    backup_manifest = (
        cloudrun_deployer.backup_artifacts(target_key)
        if (parent_key is not None or has_synced) else None
    )

    # Snapshot the dataset image count into the YAML so the dashboard reads it
    # from config instead of counting via Drive on every request (PackagingConfig.image_count).
    img_count = len(packaging_store.list_images(key))
    if parent_key is not None:
        img_count += _count_active_images(parent_key)  # edit-draft = existing refs + new uploads
    draft["image_count"] = img_count

    try:
        # 4. Write packaging YAML under target_key (parent_key on overwrite)
        yaml_path = cloudrun_deployer.write_packaging_yaml(target_key, draft)

        # 5. Promote freshly-synced detector + classifier (either path)
        promoted = cloudrun_deployer.promote_draft_model(key) if has_synced else {}

        # 5.5 Publish YAML + models to GCS so the change survives Cloud Run
        #     revisions (no-op when GCS_CONFIG_BUCKET unset — local dev/TEST_MODE).
        #     Done inside the try so a publish failure triggers rollback.
        gcs_result = cloudrun_deployer.publish_packaging_to_gcs(target_key)

        # 6. Reload PackagingRegistry in-process so the (re-)written class is picked up
        try:
            main.reload_registry()
        except Exception as e:
            raise RuntimeError(f"registry reload failed: {e}")
    except Exception as e:
        logger.exception("deploy failed — attempting rollback")
        if backup_manifest is not None:
            cloudrun_deployer.restore_backup(backup_manifest)
            try:
                main.reload_registry()
            except Exception:
                logger.exception("registry reload after rollback failed")
        raise HTTPException(500, f"deploy failed: {e}")

    # 7. Trigger Cloud Run revision (non-fatal if IAM lacks role).
    #    In TEST_MODE the trigger is the one call that can touch production
    #    Cloud Run, so it is skipped entirely and the deploy is simulated.
    if os.getenv("TEST_MODE") == "1":
        cr_result = {"triggered": False, "reason": "test mode (simulated)"}
        logger.info("TEST_MODE — skipping Cloud Run trigger for '%s'", target_key)
    else:
        cr_result = cloudrun_deployer.trigger_cloud_run_revision()

    # 8. Edit-draft: remove the now-merged draft. Fresh deploy: mark status.
    if parent_key is not None:
        packaging_store.delete_draft(key)
    else:
        packaging_store.update_draft(key, status="deployed")

    return {
        "deployed": True,
        "target_key": target_key,
        "yaml_path": str(yaml_path),
        "model_promoted": {k: str(v) for k, v in promoted.items()} or None,
        "backup": backup_manifest,
        "gcs": gcs_result,
        "cloud_run": cr_result,
        "eval": eval_data,
    }


@router.get("/{key}/eval")
def get_eval(key: str):
    """Return latest eval metrics + hard-floor check."""
    from services import eval_thresholds

    eval_path = Path(os.getenv("DRAFT_DIR", "data/drafts")) / key / "models" / "eval.json"
    if not eval_path.exists():
        raise HTTPException(404, "ยังไม่มี eval — รัน full training ก่อน")
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    check = eval_thresholds.check_hard_floor(eval_data)
    return {"eval": eval_data, "hard_floor": check}



# ─── Annotations (drafts only) ─────────────────────────

@router.get("/{key}/annotations")
def list_annotations(key: str):
    """List annotation status สำหรับทุกรูปใน draft. Active packaging → 404."""
    import main
    if main.registry is not None and main.registry.get(key) is not None:
        raise HTTPException(404, "active packagings ไม่รองรับ annotation")
    if packaging_store.get_draft(key) is None:
        raise HTTPException(404, f"draft '{key}' not found")
    items = packaging_store.list_annotation_status(key)
    labeled = sum(1 for it in items if it["labeled"])
    return {"total": len(items), "labeled": labeled, "items": items}


@router.get("/{key}/annotations/{filename}")
def get_annotation(key: str, filename: str):
    """Get bbox annotations for a single image (draft only)."""
    if packaging_store.get_draft(key) is None:
        raise HTTPException(404, f"draft '{key}' not found")
    ann = packaging_store.get_annotation(key, filename)
    if ann is None:
        return {"bboxes": [], "updated_at": None}
    return ann


@router.put("/{key}/annotations/{filename}")
def save_annotation(key: str, filename: str, body: AnnotationSave):
    """Save bbox annotations (overwrites)."""
    if packaging_store.get_draft(key) is None:
        raise HTTPException(404, f"draft '{key}' not found")
    bbox_dicts = [b.model_dump() for b in body.bboxes]
    saved = packaging_store.save_annotation(key, filename, bbox_dicts)
    if saved is None:
        raise HTTPException(500, "failed to save annotation")
    return saved


@router.delete("/{key}/annotations/{filename}")
def delete_annotation(key: str, filename: str):
    if packaging_store.get_draft(key) is None:
        raise HTTPException(404, f"draft '{key}' not found")
    if not packaging_store.delete_annotation(key, filename):
        raise HTTPException(404, "annotation not found")
    return {"deleted": filename}


def _resolve_pipeline(key: str) -> tuple[str, list[str]]:
    """Return (pipeline, sub_regions) for active or draft packaging."""
    import main
    if main.registry is not None:
        cfg = main.registry.get(key)
        if cfg is not None:
            return cfg.pipeline, cfg.sub_regions
    draft = packaging_store.get_draft(key)
    if draft is not None and draft.get("config"):
        return draft.get("pipeline", "detector_ocr"), draft.get("sub_regions", [])
    return "detector_ocr", []


def _list_sample_files(key: str, count: int) -> list[str]:
    """First `count` image filenames alphabetically — active or draft."""
    import main
    if main.registry is not None and main.registry.get(key) is not None:
        img_dir = _ACTIVE_IMAGES_DIR / key
        if img_dir.exists():
            files = sorted(
                p.name for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMG_EXTS
            )
            if files:
                return files[:count]
        # Prod ships no local images/ — fall back to the Drive dataset.
        if os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "").strip():
            from services import drive_samples
            return [f["name"] for f in drive_samples.class_images(key)[:count]]
        return []
    # Draft
    return [img["name"] for img in packaging_store.list_images(key)[:count]]


def _resolve_image_path(key: str, filename: str) -> Path | None:
    """Locate image bytes on disk — active or draft."""
    import main
    safe = Path(filename).name
    if main.registry is not None and main.registry.get(key) is not None:
        p = _ACTIVE_IMAGES_DIR / key / safe
        if p.exists():
            return p
        # Prod has no local file — download the Drive dataset image on demand.
        if os.getenv("DRIVE_CLASSIFIER_DATASET_FOLDER_ID", "").strip():
            return _drive_sample_path(key, safe)
        return None
    return packaging_store.image_path(key, safe)


def _ensure_crops(
    key: str,
    filename: str,
    pipeline: str,
    sub_regions: list[str],
) -> list[dict]:
    """Compute (or load from cache) detector crops for one image.

    Returns list of {bbox, idx, label}. Empty if pipeline == 'qr_scanner'
    or detector unavailable. Caches JPEG + sidecar JSON under data/crops/.
    """
    if pipeline == "qr_scanner":
        return []

    cache_dir = _CROP_CACHE_DIR / key
    sidecar = cache_dir / f"{filename}.meta.json"
    det_mtime = _DETECTOR_MODEL_PATH.stat().st_mtime if _DETECTOR_MODEL_PATH.exists() else 0.0

    # Cache hit?
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta.get("detector_mtime") == det_mtime:
                regions = meta.get("regions", [])
                # Verify crop files still present
                if all((cache_dir / f"{filename}.{r['idx']}.jpg").exists() for r in regions):
                    logger.debug("crop cache hit: %s/%s", key, filename)
                    return regions
        except (json.JSONDecodeError, OSError):
            pass

    # Cache miss — run detector
    import main
    if main.detector is None:
        logger.warning("detector unavailable for %s/%s", key, filename)
        return []

    src = _resolve_image_path(key, filename)
    if src is None:
        return []

    try:
        detections = main.detector.crop_all(src.read_bytes(), key)
    except Exception:
        logger.exception("detector failed on %s/%s", key, filename)
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    regions = []
    for idx, det in enumerate(detections):
        out = cache_dir / f"{filename}.{idx}.jpg"
        out.write_bytes(det.cropped_bytes)
        label = sub_regions[idx] if idx < len(sub_regions) else None
        regions.append({"bbox": det.bbox, "idx": idx, "label": label})

    sidecar.write_text(
        json.dumps({"detector_mtime": det_mtime, "regions": regions}, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("crops generated: %s/%s (%d regions)", key, filename, len(regions))
    return regions


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
    from services.drive_client import DriveClient as _DriveClient

    file_id = next(
        (f["id"] for f in drive_samples.class_images(key) if f["name"] == safe), None
    )
    if file_id is None:
        return None
    try:
        _DriveClient().download_file(file_id, dest)
        return dest
    except Exception as e:  # noqa: BLE001 -- a broken thumbnail must not 500
        logger.warning("drive sample download failed %s/%s: %s", key, safe, e)
        # Drop any partially-written file so a later request retries instead of
        # serving a truncated image forever (the cache-hit at line 1093).
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        return None
