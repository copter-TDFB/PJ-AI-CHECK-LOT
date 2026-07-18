"""
สร้าง Confusion Matrix ของ classifier model ที่ confidence threshold ที่กำหนด

วิธีรัน:
    python confusion_matrix_eval.py                                   # 8 active class, threshold 0.5/0.55/0.6
    python confusion_matrix_eval.py --no-plot                         # พิมพ์ text report อย่างเดียว
    python confusion_matrix_eval.py --classes back_label,grade_bag    # เฉพาะบาง class
    python confusion_matrix_eval.py --thresholds 0.5,0.6,0.7          # เฉพาะบาง threshold

Output:
    confusion_matrix_<threshold>.png (ต่อ threshold)

CAVEAT: images/<class>/ เป็นชุดรูปเดียวกับที่ publish ไป train classifier บน Drive
(ไม่มี held-out test set แยกต่างหากใน repo) — ตัวเลข precision/recall ที่ได้จึงวัด
บนข้อมูลที่โมเดลเคยเห็นตอน train แล้วมีแนวโน้ม optimistic กว่า real-world ใช้เทียบ
"ระหว่าง threshold" กันเอง (relative) ไม่ใช่ตัวเลข absolute ที่รับประกันบน production
"""

import argparse
import sys
import io
from pathlib import Path
from collections import defaultdict

# force UTF-8 on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

DEFAULT_CLASSES = [
    "30_sachet", "back_label", "capsule_box", "container_label",
    "grade_bag", "import_sticker", "print_sticker_back", "retail_sachet",
]
IMAGES_DIR = Path("images")
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_THRESHOLDS = [0.5, 0.55, 0.6]

# ตัวแปร module-level ที่ฟังก์ชันด้านล่างอ้างอิง — main() เซ็ตค่าจริงจาก CLI args
CLASSES    = DEFAULT_CLASSES
THRESHOLDS = DEFAULT_THRESHOLDS


def collect_predictions() -> list[dict]:
    """รัน classifier บนทุกรูปใน images/<class>/ คืน list of {true, pred, conf}"""
    from pipeline.classifier import ImageClassifier

    clf = ImageClassifier("models/classifier.pt")
    records: list[dict] = []

    total_images = sum(
        len([p for p in (IMAGES_DIR / cls).iterdir() if p.suffix.lower() in IMG_EXTS])
        for cls in CLASSES
        if (IMAGES_DIR / cls).exists()
    )
    processed = 0

    for true_cls in CLASSES:
        cls_dir = IMAGES_DIR / true_cls
        if not cls_dir.exists():
            print(f"  ⚠️  ไม่พบ {cls_dir} — ข้าม")
            continue

        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        for img_path in images:
            try:
                pred_cls, conf = clf.predict(img_path.read_bytes())
                records.append({"true": true_cls, "pred": pred_cls, "conf": conf})
                processed += 1
                print(f"\r  [{processed}/{total_images}] {img_path.name:<35}", end="", flush=True)
            except Exception as exc:
                print(f"\n  ⚠️  {img_path.name}: {exc}")

    print(f"\r  ✅ ประมวลผลครบ {processed} รูป{' ' * 30}")
    return records


def build_matrix(records: list[dict], threshold: float) -> tuple[list[list[int]], int]:
    """
    สร้าง confusion matrix 5×6 (5 true class × 5 predicted + 1 "rejected")

    Returns:
        (matrix, rejected_count)
        matrix[i][j]   = รูป true_class=i ที่ถูก predict เป็น class=j  (j=0..4)
        matrix[i][5]   = รูป true_class=i ที่ conf < threshold (rejected)
        rejected_count = รวม rejected ทุก class
    """
    n = len(CLASSES)
    idx = {cls: i for i, cls in enumerate(CLASSES)}
    matrix = [[0] * (n + 1) for _ in range(n)]  # +1 for rejected column
    rejected = 0

    for r in records:
        i = idx[r["true"]]
        if r["conf"] < threshold:
            matrix[i][n] += 1  # last column = rejected
            rejected += 1
        else:
            j = idx.get(r["pred"], n)  # unknown pred falls into rejected col
            matrix[i][j] += 1

    return matrix, rejected


def print_matrix(matrix: list[list[int]], threshold: float, rejected: int, total: int) -> None:
    """พิมพ์ confusion matrix แบบ text (รวมคอลัมน์ rejected)"""
    n = len(CLASSES)
    short = [c[:8] for c in CLASSES] + ["rejected"]
    col_w = 9

    accepted = total - rejected
    print(f"\n{'═'*80}")
    print(f"  Confusion Matrix — threshold = {threshold}")
    print(f"  Accepted: {accepted}/{total}  |  Rejected (conf<{threshold}): {rejected}")
    print(f"{'═'*80}")

    header = f"  {'True \\ Pred':<12}" + "".join(f"{s:>{col_w}}" for s in short)
    print(header)
    print(f"  {'-'*74}")

    row_totals = [sum(row) for row in matrix]
    for i, cls in enumerate(CLASSES):
        row_str = f"  {short[i]:<12}"
        for j in range(n + 1):
            val = matrix[i][j]
            if j == n and val > 0:
                row_str += f"\033[93m{val:>{col_w}}\033[0m"  # เหลือง = rejected
            elif j < n and i == j and val > 0:
                row_str += f"\033[92m{val:>{col_w}}\033[0m"  # เขียว = correct
            elif val > 0:
                row_str += f"\033[91m{val:>{col_w}}\033[0m"  # แดง = wrong
            else:
                row_str += f"{val:>{col_w}}"
        correct = matrix[i][i]
        total_row = row_totals[i]
        recall = f"{correct/total_row:.1%}" if total_row > 0 else "—"
        row_str += f"  recall={recall}"
        print(row_str)

    print(f"  {'-'*74}")

    # precision per predicted class (excluding rejected column)
    col_totals = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    prec_str = f"  {'Precision':<12}"
    for j in range(n):
        prec = f"{matrix[j][j]/col_totals[j]:.0%}" if col_totals[j] > 0 else "—"
        prec_str += f"{prec:>{col_w}}"
    print(prec_str)

    correct_total = sum(matrix[i][i] for i in range(n))
    print(f"\n  Accuracy (accepted only): {correct_total}/{accepted} = {correct_total/accepted:.1%}" if accepted else "")
    print(f"  Coverage: {accepted}/{total} = {accepted/total:.1%}")


def plot_matrix(matrix: list[list[int]], threshold: float, rejected: int, total: int) -> None:
    """บันทึก confusion matrix เป็น PNG — heatmap 5×6 รวมคอลัมน์ background"""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
        import seaborn as sns
    except ImportError:
        print("  ⚠️  matplotlib/seaborn ไม่ได้ติดตั้ง — ข้าม plot")
        return

    n = len(CLASSES)
    accepted = total - rejected
    col_labels = CLASSES + ["background"]
    row_labels = CLASSES

    arr = np.array(matrix, dtype=float)  # shape (5, 6)

    # normalize per row over ALL 6 columns → true recall including background
    row_sums = arr.sum(axis=1, keepdims=True)
    arr_norm = np.divide(arr, row_sums, where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # ── สร้าง color array แบบ custom: Blue ส่วน predicted, Orange ส่วน background ──
    def make_color_array(data: np.ndarray, is_norm: bool) -> np.ndarray:
        """คืน RGBA array shape (n_rows, n_cols+1, 4)"""
        rows, cols = data.shape  # cols = n+1
        rgba = np.ones((rows, cols, 4))
        blues  = plt.cm.Blues
        oranges = plt.cm.Oranges
        max_pred = data[:, :n].max() if data[:, :n].max() > 0 else 1
        max_bg   = data[:, n].max()  if data[:, n].max()  > 0 else 1
        for i in range(rows):
            for j in range(cols - 1):
                v = data[i, j] / max_pred if not is_norm else data[i, j]
                rgba[i, j] = blues(max(0.05, v))
            v = data[i, n] / max_bg if not is_norm else data[i, n]
            rgba[i, n] = oranges(max(0.05, v))
        return rgba

    # ── ซ้าย: count ──────────────────────────────────────────────────
    ax = axes[0]
    rgba_count = make_color_array(arr, is_norm=False)
    ax.imshow(rgba_count, aspect="auto")

    for i in range(n):
        for j in range(n + 1):
            val = int(arr[i, j])
            # text color: white on dark cells, black on light
            bg_lum = rgba_count[i, j, :3].mean()
            txt_color = "white" if bg_lum < 0.55 else "black"
            weight = "bold" if (j < n and i == j) else "normal"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=12, color=txt_color, fontweight=weight)

    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Count  (threshold={threshold})", fontsize=13, fontweight="bold")
    # เส้นแบ่ง background column
    ax.axvline(x=n - 0.5, color="gray", linewidth=2, linestyle="--", alpha=0.7)

    # ── ขวา: recall normalized ───────────────────────────────────────
    ax2 = axes[1]
    rgba_norm = make_color_array(arr_norm, is_norm=True)
    ax2.imshow(rgba_norm, aspect="auto")

    for i in range(n):
        for j in range(n + 1):
            val = arr_norm[i, j]
            bg_lum = rgba_norm[i, j, :3].mean()
            txt_color = "white" if bg_lum < 0.55 else "black"
            weight = "bold" if (j < n and i == j) else "normal"
            ax2.text(j, i, f"{val:.0%}", ha="center", va="center",
                     fontsize=12, color=txt_color, fontweight=weight)

    ax2.set_xticks(range(n + 1))
    ax2.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(row_labels, fontsize=10)
    ax2.set_xlabel("Predicted", fontsize=11)
    ax2.set_ylabel("True", fontsize=11)
    ax2.set_title(f"Recall (row-normalized, incl. background)  (threshold={threshold})",
                  fontsize=13, fontweight="bold")
    ax2.axvline(x=n - 0.5, color="gray", linewidth=2, linestyle="--", alpha=0.7)

    correct_total = sum(matrix[i][i] for i in range(n))
    fig.suptitle(
        f"Classifier Confusion Matrix — conf ≥ {threshold}\n"
        f"Accuracy (accepted): {correct_total}/{accepted} = {correct_total/accepted:.1%}   |   "
        f"Coverage: {accepted}/{total} = {accepted/total:.1%}   |   "
        f"Background (rejected): {rejected}",
        fontsize=12,
    )

    plt.tight_layout()
    out_path = Path(f"confusion_matrix_{threshold}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {out_path}")


def print_summary_table(all_results: list[tuple]) -> None:
    """ตาราง summary เปรียบเทียบทั้ง 3 threshold"""
    print(f"\n{'═'*60}")
    print("  Summary — เปรียบเทียบ Threshold")
    print(f"{'═'*60}")
    print(f"  {'Threshold':>10}  {'Accepted':>10}  {'Rejected':>10}  {'Accuracy':>10}  {'Coverage':>10}")
    print(f"  {'-'*56}")
    for threshold, matrix, rejected, total in all_results:
        n = len(CLASSES)
        accepted = total - rejected
        correct = sum(matrix[i][i] for i in range(n))
        acc_str = f"{correct/accepted:.2%}" if accepted else "—"
        cov_str = f"{accepted/total:.2%}" if total else "—"
        print(f"  {threshold:>10}  {accepted:>10}  {rejected:>10}  {acc_str:>10}  {cov_str:>10}")
    print(f"{'═'*60}\n")


def main() -> None:
    global CLASSES, THRESHOLDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true", help="ไม่บันทึก PNG")
    parser.add_argument(
        "--classes", default=",".join(DEFAULT_CLASSES),
        help="comma-separated class keys (default: 8 active class)",
    )
    parser.add_argument(
        "--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="comma-separated confidence thresholds (default: 0.5,0.55,0.6)",
    )
    args = parser.parse_args()

    CLASSES = [c.strip() for c in args.classes.split(",") if c.strip()]
    THRESHOLDS = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    print(f"Classes:    {CLASSES}")
    print(f"Thresholds: {THRESHOLDS}")
    print(
        "\n⚠️  CAVEAT: images/<class>/ คือชุดรูปที่เคย publish ไป train classifier แล้ว "
        "(ไม่มี held-out test set) — ตัวเลขข้างล่างจึงวัดบนข้อมูลที่โมเดลเคยเห็น อ่านเป็นการ"
        "เทียบ threshold ต่อ threshold เท่านั้น ไม่ใช่ precision/recall จริงบน production\n"
    )

    print("Loading classifier model...")
    try:
        records = collect_predictions()
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    if not records:
        print("❌ ไม่พบรูปภาพใน images/")
        sys.exit(1)

    total = len(records)
    all_results = []

    for threshold in THRESHOLDS:
        matrix, rejected = build_matrix(records, threshold)
        print_matrix(matrix, threshold, rejected, total)
        if not args.no_plot:
            plot_matrix(matrix, threshold, rejected, total)
        all_results.append((threshold, matrix, rejected, total))

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
