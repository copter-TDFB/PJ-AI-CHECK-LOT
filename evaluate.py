"""
วัด accuracy ของ pipeline ต่อ class ด้วยรูปจริงใน images/<class>/

วิธีรัน:
    python evaluate.py                  # ทดสอบ classifier + detector
    python evaluate.py --ocr            # รวม OCR (ต้องมี Google Cloud credentials)
    python evaluate.py --class back_label  # ทดสอบเฉพาะ class นั้น
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s — %(message)s")

CLASSES      = ["back_label", "container_label", "grade_bag", "import_sticker", "retail_sachet"]
IMAGES_DIR   = Path("images")
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_ACC   = 0.95


def load_pipeline(run_ocr: bool):
    """โหลด pipeline components ที่จำเป็น"""
    from pipeline.classifier import ImageClassifier
    from pipeline.detector import RegionDetector
    from pipeline.preprocessor import Preprocessor
    from pipeline.qr_scanner import QrScanner

    clf = ImageClassifier("models/classifier.pt")
    det = RegionDetector("models/detector.pt")
    pre = Preprocessor()
    qr  = QrScanner()
    ocr = None

    if run_ocr:
        from pipeline.ocr_engine import OcrEngine
        ocr = OcrEngine()

    return clf, det, pre, qr, ocr


def evaluate_class(
    cls: str,
    clf,
    det,
    pre,
    qr,
    ocr,
    run_ocr: bool,
) -> dict:
    """ประเมิน 1 class คืน dict ผลลัพธ์"""
    cls_dir = IMAGES_DIR / cls
    images  = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not images:
        return {"class": cls, "total": 0, "error": "no images found"}

    clf_correct = 0
    det_found   = 0
    lot_found   = 0
    errors      = 0
    t0          = time.time()

    for img_path in images:
        try:
            img_bytes = img_path.read_bytes()

            # ── Classifier ────────────────────────────────────────
            pred_class, conf = clf.predict(img_bytes)
            if pred_class == cls:
                clf_correct += 1

            # ── Detector / QR scanner ────────────────────────────
            if cls == "import_sticker":
                result = qr.scan(img_bytes)
                if result["status"] == "ok":
                    det_found += 1
                if result["lot_number"]:
                    lot_found += 1
            else:
                detection = det.crop(img_bytes, cls)
                if detection.bbox is not None:
                    det_found += 1

                # ── OCR (optional) ────────────────────────────────
                if run_ocr and ocr:
                    proc = pre.run(detection.cropped_bytes, cls)
                    res  = ocr.run(proc, image_class=cls)
                    if res["lot_number"]:
                        lot_found += 1

        except Exception as exc:
            logging.debug("Error on %s: %s", img_path.name, exc)
            errors += 1

    total    = len(images)
    elapsed  = time.time() - t0
    clf_acc  = clf_correct / total
    det_rate = det_found   / total

    result = {
        "class":     cls,
        "total":     total,
        "clf_acc":   clf_acc,
        "det_rate":  det_rate,
        "errors":    errors,
        "elapsed_s": round(elapsed, 1),
    }
    if run_ocr:
        result["lot_rate"] = lot_found / total

    return result


def print_report(results: list[dict], run_ocr: bool) -> bool:
    """พิมพ์ตารางสรุป คืน True ถ้า accuracy ผ่านทุก class"""
    sep = "─" * 72

    header = f"{'Class':<20} {'Total':>6} {'Clf Acc':>9} {'Det/QR':>8}"
    if run_ocr:
        header += f" {'Lot Found':>10}"
    header += f" {'Errors':>7}"

    print(f"\n{sep}")
    print(header)
    print(sep)

    all_pass = True
    for r in results:
        if "error" in r:
            print(f"{r['class']:<20}  ⚠️  {r['error']}")
            continue

        clf_str = f"{r['clf_acc']:.1%}"
        det_str = f"{r['det_rate']:.1%}"
        flag    = "" if r["clf_acc"] >= TARGET_ACC else " ❌"

        row = f"{r['class']:<20} {r['total']:>6} {clf_str:>9}{flag} {det_str:>8}"
        if run_ocr:
            lot_str = f"{r['lot_rate']:.1%}" if "lot_rate" in r else "—"
            row += f" {lot_str:>10}"
        row += f" {r['errors']:>7}  ({r['elapsed_s']}s)"

        print(row)
        if r["clf_acc"] < TARGET_ACC:
            all_pass = False

    print(sep)
    overall_clf = sum(r["clf_acc"] * r["total"] for r in results if "clf_acc" in r)
    overall_tot = sum(r["total"] for r in results if "total" in r)
    if overall_tot:
        print(f"{'Overall clf accuracy':<20} {overall_tot:>6} {overall_clf/overall_tot:>9.1%}")
    print(sep)

    status = "✅ PASS" if all_pass else "❌ FAIL (target ≥ 95%)"
    print(f"\nResult: {status}\n")
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR lot pipeline")
    parser.add_argument("--ocr",   action="store_true", help="รวม OCR step (ต้องมี credentials)")
    parser.add_argument("--class", dest="cls", help="ทดสอบเฉพาะ class นี้")
    args = parser.parse_args()

    target_classes = [args.cls] if args.cls else CLASSES

    print(f"\nLoading pipeline... (OCR={'on' if args.ocr else 'off'})")
    try:
        clf, det, pre, qr, ocr = load_pipeline(args.ocr)
    except Exception as exc:
        print(f"❌ Pipeline load failed: {exc}")
        sys.exit(1)

    results = []
    for cls in target_classes:
        print(f"  Evaluating {cls}...", end=" ", flush=True)
        r = evaluate_class(cls, clf, det, pre, qr, ocr, args.ocr)
        print(f"{r.get('total', 0)} images, clf={r.get('clf_acc', 0):.1%}")
        results.append(r)

    passed = print_report(results, args.ocr)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
