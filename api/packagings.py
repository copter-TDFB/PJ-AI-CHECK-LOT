"""Wizard API — manage packaging drafts."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.schemas import (
    AnnotationSave,
    PackagingConfigUpdate,
    PackagingCreate,
    PackagingResponse,
    PackagingUpdate,
    RegexPreviewRequest,
    RegexPreviewResponse,
)
from services import packaging_store
from services.regex_generator import generate_regex, preview_matches

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/packagings", tags=["packagings"])

_ACTIVE_IMAGES_DIR = Path("images")
_CROP_CACHE_DIR = Path(os.getenv("CROP_CACHE_DIR", "data/crops"))
_DETECTOR_MODEL_PATH = Path(os.getenv("MODEL_DETECTOR_PATH", "models/detector.pt"))
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@router.get("", response_model=list[PackagingResponse])
def list_packagings():
    """รวม active (จาก YAML) + drafts (จาก data/drafts)."""
    import main  # lazy import — registry initialized in lifespan

    items: list[PackagingResponse] = []

    if main.registry is not None:
        for key in main.registry.all_keys():
            cfg = main.registry.get(key)
            if cfg is None:
                continue
            items.append(PackagingResponse(
                key=cfg.key,
                display_name=cfg.display_name,
                pipeline=cfg.pipeline,
                status="active",
                image_count=_count_active_images(key),
                conf_threshold=cfg.conf_threshold,
                accuracy=cfg.accuracy,
            ))

        # Archived packagings — YAML still on disk but renamed to *.yaml.archived
        for arch_key in main.registry.archived_keys():
            arch_yaml = Path("config/packagings") / f"{arch_key}.yaml.archived"
            try:
                data = yaml.safe_load(arch_yaml.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append(PackagingResponse(
                key=arch_key,
                display_name=data.get("display_name", arch_key),
                pipeline=data.get("pipeline", "detector_ocr"),
                status="archived",
                image_count=_count_active_images(arch_key),
                conf_threshold=float(data.get("conf_threshold", 0.6)),
                accuracy=float(data["accuracy"]) if data.get("accuracy") is not None else None,
            ))

    for draft in packaging_store.list_drafts():
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
                image_count=_count_active_images(key),
                conf_threshold=cfg.conf_threshold,
                accuracy=cfg.accuracy,
            )
        if main.registry.is_archived(key):
            arch_yaml = Path("config/packagings") / f"{key}.yaml.archived"
            try:
                data = yaml.safe_load(arch_yaml.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            return PackagingResponse(
                key=key,
                display_name=data.get("display_name", key),
                pipeline=data.get("pipeline", "detector_ocr"),
                status="archived",
                image_count=_count_active_images(key),
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

_PACKAGING_YAML_DIR = Path("config/packagings")


@router.post("/{key}/clone", response_model=PackagingResponse, status_code=201)
def clone_active(key: str):
    """Create an edit-draft `{key}__edit` from an active packaging.

    The wizard reopens at step 2 (image upload) — config is pre-populated
    from the active YAML, parent images stay read-only on disk.
    """
    import main

    if main.registry is None or main.registry.get(key) is None:
        raise HTTPException(404, f"active packaging '{key}' not found")

    yaml_path = _PACKAGING_YAML_DIR / f"{key}.yaml"
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
        from pipeline.packaging_registry import PackagingRegistry
        main.registry = PackagingRegistry()
    except Exception as e:
        logger.exception("registry reload failed after archive")
        raise HTTPException(500, f"registry reload failed: {e}")

    return {"archived": key, "yaml_path": str(dst)}


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
        from pipeline.packaging_registry import PackagingRegistry
        main.registry = PackagingRegistry()
    except Exception as e:
        logger.exception("registry reload failed after unarchive")
        raise HTTPException(500, f"registry reload failed: {e}")

    return {"unarchived": key, "yaml_path": str(dst)}


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
        if not img_dir.exists():
            return {"images": []}
        return {"images": [
            {"name": p.name, "size": p.stat().st_size, "read_only": False}
            for p in sorted(img_dir.iterdir())
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        ]}

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


# ─── Training (seed → active learning) ─────────────────

_MIN_LABELED_FOR_SEED = 5  # ขั้นต่ำที่ allow seed train (จริงๆ 20 ตาม UI แต่ allow ทดสอบเร็ว)


@router.post("/{key}/training/seed/start")
def training_seed_start(key: str):
    """Bundle labeled images → upload to Drive → gen Colab notebook → return URL."""
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")

    status = packaging_store.list_annotation_status(key)
    labeled = [it for it in status if it["labeled"]]
    if len(labeled) < _MIN_LABELED_FOR_SEED:
        raise HTTPException(
            400,
            f"need at least {_MIN_LABELED_FOR_SEED} labeled images (have {len(labeled)})",
        )

    # Lazy imports — heavy + only needed at training time
    from services import notebook_generator, training_bundle
    from services.drive_client import DriveClient

    try:
        bundle_bytes = training_bundle.build_zip(key)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))

    try:
        drive = DriveClient()
        run_folder_id = drive.create_folder(f"lot-checker-training-{key}")
        bundle_file_id = drive.upload_bytes(
            bundle_bytes,
            name=f"{key}-bundle.zip",
            parent_id=run_folder_id,
            mime_type="application/zip",
            public=True,  # ให้ gdown โหลดได้โดยไม่ต้อง auth
        )
        nb_bytes = notebook_generator.build_seed_notebook(
            packaging_key=key,
            bundle_file_id=bundle_file_id,
            output_folder_id=run_folder_id,
        )
        nb_file_id = drive.upload_bytes(
            nb_bytes,
            name=f"{key}-seed-training.ipynb",
            parent_id=run_folder_id,
            mime_type="application/vnd.google.colaboratory",
        )
    except Exception as e:
        logger.exception("training/seed/start failed for %s", key)
        raise HTTPException(500, f"Drive upload failed: {e}")

    # Persist on draft meta
    packaging_store.update_draft(
        key,
        training_run={
            "bundle_file_id": bundle_file_id,
            "notebook_file_id": nb_file_id,
            "output_folder_id": run_folder_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "kind": "seed",
        },
        status="training_seed",
    )

    return {
        "colab_url": notebook_generator.colab_url(nb_file_id),
        "notebook_file_id": nb_file_id,
        "output_folder_id": run_folder_id,
        "bundle_size_kb": round(len(bundle_bytes) / 1024, 1),
    }


@router.post("/{key}/training/full/start")
def training_full_start(key: str):
    """Full training — publishes dataset images directly to Drive, then generates
    and uploads a Colab notebook. No zip bundle is created."""
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")

    status = packaging_store.list_annotation_status(key)
    labeled = [it for it in status if it["labeled"]]
    if len(labeled) < 10:
        raise HTTPException(400, f"need at least 10 labeled images (have {len(labeled)})")

    from services import dataset_publisher, notebook_generator
    from services.drive_client import DriveClient

    try:
        drive = DriveClient()
    except Exception as e:
        logger.exception("Drive client init failed for %s", key)
        raise HTTPException(500, f"Drive client init failed: {e}")

    try:
        dataset_summary = dataset_publisher.publish(key, drive=drive)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception("dataset publish failed for %s", key)
        raise HTTPException(500, f"Dataset publish failed: {e}")

    try:
        run_folder_id = drive.create_folder(f"lot-checker-training-{key}-full")
        nb_bytes = notebook_generator.build_full_notebook(
            packaging_key=key,
            output_folder_id=run_folder_id,
        )
        nb_file_id = drive.upload_bytes(
            nb_bytes, name=f"{key}-full-training.ipynb", parent_id=run_folder_id,
            mime_type="application/vnd.google.colaboratory",
        )
    except Exception as e:
        logger.exception("training/full/start failed for %s", key)
        raise HTTPException(500, f"Drive upload failed: {e}")

    packaging_store.update_draft(
        key,
        training_run={
            "notebook_file_id": nb_file_id,
            "output_folder_id": run_folder_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "kind": "full",
            "dataset": dataset_summary,
        },
        status="training_full",
    )
    return {
        "colab_url": notebook_generator.colab_url(nb_file_id),
        "notebook_file_id": nb_file_id,
        "output_folder_id": run_folder_id,
        "dataset": dataset_summary,
        "labeled_count": len(labeled),
    }


@router.post("/{key}/training/full/done")
def training_full_done(key: str):
    """Pull full model + eval.json จาก Drive → store locally → ready for deploy."""
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")
    run = draft.get("training_run")
    if not run or run.get("kind") != "full":
        raise HTTPException(400, "ยังไม่ได้เริ่ม full training — กด 'เริ่ม Full Training' ก่อน")

    from services.drive_client import DriveClient

    drive = DriveClient()
    output_folder_id = run["output_folder_id"]

    model_id = drive.find_in_folder(output_folder_id, "full_detector.pt")
    if not model_id:
        raise HTTPException(404, "ยังหา full_detector.pt ใน Drive ไม่เจอ — รัน notebook ใน Colab ให้เสร็จก่อน")

    model_dir = Path("data/drafts") / key / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "full_detector.pt"
    try:
        drive.download_file(model_id, model_path)
    except Exception as e:
        logger.exception("download full_detector failed")
        raise HTTPException(500, f"download model failed: {e}")

    # Pull eval.json
    eval_id = drive.find_in_folder(output_folder_id, "full_eval.json")
    eval_summary: dict = {}
    if eval_id:
        try:
            eval_summary = drive.read_json(eval_id)
            # Persist locally for /eval endpoint
            (model_dir / "eval.json").write_text(
                json.dumps(eval_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.warning("could not read eval.json")

    packaging_store.update_draft(key, status="trained")
    return {"model_downloaded": True, "eval": eval_summary}


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
    eval_path = Path("data/drafts") / key / "models" / "eval.json"
    if not eval_path.exists():
        raise HTTPException(400, "ยังไม่มี eval — รัน full training ก่อน")
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    check = eval_thresholds.check_hard_floor(eval_data)
    if not check["passed"]:
        raise HTTPException(400, {"error": "hard floor failed", "failures": check["failures"]})

    # 2. Backup active artifacts before overwrite (no-op for fresh deploys)
    backup_manifest = (
        cloudrun_deployer.backup_artifacts(parent_key) if parent_key else None
    )

    try:
        # 3. Write packaging YAML under target_key (parent_key on overwrite)
        yaml_path = cloudrun_deployer.write_packaging_yaml(target_key, draft)

        # 4. Promote the freshly-trained detector (overwrite path only)
        promoted_model = None
        if parent_key is not None:
            promoted_model = cloudrun_deployer.promote_draft_model(key)

        # 5. Reload PackagingRegistry in-process so the (re-)written class is picked up
        try:
            from pipeline.packaging_registry import PackagingRegistry
            main.registry = PackagingRegistry()
        except Exception as e:
            raise RuntimeError(f"registry reload failed: {e}")
    except Exception as e:
        logger.exception("deploy failed — attempting rollback")
        if backup_manifest is not None:
            cloudrun_deployer.restore_backup(backup_manifest)
            try:
                from pipeline.packaging_registry import PackagingRegistry
                main.registry = PackagingRegistry()
            except Exception:
                logger.exception("registry reload after rollback failed")
        raise HTTPException(500, f"deploy failed: {e}")

    # 6. Trigger Cloud Run revision (non-fatal if IAM lacks role)
    cr_result = cloudrun_deployer.trigger_cloud_run_revision()

    # 7. Edit-draft: remove the now-merged draft. Fresh deploy: mark status.
    if parent_key is not None:
        packaging_store.delete_draft(key)
    else:
        packaging_store.update_draft(key, status="deployed")

    return {
        "deployed": True,
        "target_key": target_key,
        "yaml_path": str(yaml_path),
        "model_promoted": str(promoted_model) if parent_key and promoted_model else None,
        "backup": backup_manifest,
        "cloud_run": cr_result,
        "eval": eval_data,
    }


@router.get("/{key}/eval")
def get_eval(key: str):
    """Return latest eval metrics + hard-floor check."""
    from services import eval_thresholds

    eval_path = Path("data/drafts") / key / "models" / "eval.json"
    if not eval_path.exists():
        raise HTTPException(404, "ยังไม่มี eval — รัน full training ก่อน")
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    check = eval_thresholds.check_hard_floor(eval_data)
    return {"eval": eval_data, "hard_floor": check}


@router.post("/{key}/training/seed/done")
def training_seed_done(key: str):
    """Poll Drive for trained model → download → run prelabeling on remaining images."""
    draft = packaging_store.get_draft(key)
    if draft is None:
        raise HTTPException(404, f"draft '{key}' not found")
    run = draft.get("training_run")
    if not run or run.get("kind") != "seed":
        raise HTTPException(400, "ยังไม่ได้เริ่ม seed training — กด 'เริ่ม Seed Training' ก่อน")

    from services import active_learning
    from services.drive_client import DriveClient

    drive = DriveClient()
    output_folder_id = run["output_folder_id"]

    model_id = drive.find_in_folder(output_folder_id, "seed_detector.pt")
    if not model_id:
        raise HTTPException(
            404,
            "ยังหา seed_detector.pt ใน Drive ไม่เจอ — รัน notebook ใน Colab ให้เสร็จก่อน",
        )

    model_dir = Path("data/drafts") / key / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "seed_detector.pt"
    try:
        drive.download_file(model_id, model_path)
    except Exception as e:
        logger.exception("download seed_detector failed")
        raise HTTPException(500, f"download model failed: {e}")

    # Optionally pull eval.json
    eval_id = drive.find_in_folder(output_folder_id, "seed_eval.json")
    eval_summary: dict = {}
    if eval_id:
        try:
            eval_summary = drive.read_json(eval_id)
        except Exception:
            pass

    # Run inference on remaining unlabeled images
    try:
        result = active_learning.prelabel_remaining(key, model_path)
    except Exception as e:
        logger.exception("prelabeling failed")
        raise HTTPException(500, f"prelabeling failed: {e}")

    packaging_store.update_draft(key, status="labeled_full")
    return {
        "model_downloaded": True,
        "eval": eval_summary,
        **result,
    }


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
        return draft["config"].get("pipeline", "detector_ocr"), []
    return "detector_ocr", []


def _list_sample_files(key: str, count: int) -> list[str]:
    """First `count` image filenames alphabetically — active or draft."""
    import main
    if main.registry is not None and main.registry.get(key) is not None:
        img_dir = _ACTIVE_IMAGES_DIR / key
        if not img_dir.exists():
            return []
        files = sorted(
            p.name for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
        return files[:count]
    # Draft
    return [img["name"] for img in packaging_store.list_images(key)[:count]]


def _resolve_image_path(key: str, filename: str) -> Path | None:
    """Locate image bytes on disk — active or draft."""
    import main
    safe = Path(filename).name
    if main.registry is not None and main.registry.get(key) is not None:
        p = _ACTIVE_IMAGES_DIR / key / safe
        return p if p.exists() else None
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
    if not img_dir.exists():
        return 0
    return sum(
        1 for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )
