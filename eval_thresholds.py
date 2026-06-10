"""Evaluate per-class confidence thresholds — find max threshold per recall tier."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pipeline.classifier import ImageClassifier

IMG_DIR = Path("images")
TRAINED = {"back_label", "capsule_box", "container_label", "grade_bag", "import_sticker", "retail_sachet"}


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    clf = ImageClassifier("models/classifier.pt")
    results = defaultdict(lambda: defaultdict(list))

    for class_dir in sorted(IMG_DIR.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in TRAINED:
            continue
        true_label = class_dir.name
        files = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}])
        print(f">>> {true_label} ({len(files)})")
        for f in files:
            try:
                pred, conf = clf.predict(f.read_bytes())
                results[true_label][pred].append(conf)
            except Exception as e:
                print(f"  ERR {f.name}: {e}")

    print("\n" + "=" * 90)
    print("MAX THRESHOLD PER RECALL TIER (per class, considers only 6 trained classes)")
    print("=" * 90)
    print(f"{'Class':18s} | {'r=100%':>8s} | {'r=99%':>8s} | {'r=98%':>8s} | {'r=95%':>8s} | {'r=90%':>8s} | {'r=85%':>8s}")
    print("-" * 90)

    for cls in sorted(TRAINED):
        correct = sorted(results.get(cls, {}).get(cls, []))
        if not correct:
            continue
        n = len(correct)
        # threshold T -> recall = count(conf >= T) / n
        # to get recall = X%, threshold = correct[(1-X) * n] (sorted ascending, take lower bound)
        def t_for_recall(r):
            idx = int((1 - r) * n)
            idx = max(0, min(idx, n - 1))
            return correct[idx]

        t100 = correct[0]
        t99 = t_for_recall(0.99)
        t98 = t_for_recall(0.98)
        t95 = t_for_recall(0.95)
        t90 = t_for_recall(0.90)
        t85 = t_for_recall(0.85)
        print(f"{cls:18s} | {t100:8.3f} | {t99:8.3f} | {t98:8.3f} | {t95:8.3f} | {t90:8.3f} | {t85:8.3f}")

    print("\nNote: threshold at recall=X% = ค่าสูงสุดที่ยังเก็บ correct preds ได้ X%")


if __name__ == "__main__":
    main()
