"""Generate Colab notebooks for training Classifier + Detector.

Architecture:
- Backend uploads the new packaging's bundle (images + YOLO labels +
  classifier images) to its own Drive folder.
- Notebook runs in Colab with the USER's Drive access (full mount), reads
  the reference dataset directly from the mounted paths, merges with the
  addition bundle, trains both models, uploads results back to the addition
  folder.

Reference paths are taken from the original training notebooks
(ai_crop_lot.ipynb + colab_classify_training.ipynb).
"""

import json
import logging

logger = logging.getLogger(__name__)

# Reference dataset paths inside the user's Drive (My Drive)
DEFAULT_DETECTOR_REF = "data check lot"
DEFAULT_CLASSIFIER_REF = "data classify check lot"


def _cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def build_seed_notebook(
    packaging_key: str,
    bundle_file_id: str,
    output_folder_id: str,
    epochs: int = 30,
    imgsz: int = 640,
) -> bytes:
    """Seed = quick 1-class detector trained ONLY on draft's hand-labeled images.

    Used for pre-labeling the rest of the draft's images (active learning).
    Not deployed.
    """
    cells = [
        _cell("markdown", (
            f"# Seed Training — {packaging_key}\n\n"
            "**กด `Runtime → Run all`** แล้วรอประมาณ 10-15 นาที.\n"
            "เมื่อขึ้น **TRAINING DONE** กลับไปที่ wizard"
        )),
        _cell("code", "!pip install -q ultralytics==8.3.0 gdown==5.2.0\n"),
        _cell("code", (
            "import gdown, zipfile, json\n"
            f"BUNDLE_ID = {bundle_file_id!r}\n"
            f"OUTPUT_FOLDER_ID = {output_folder_id!r}\n"
            f"PACKAGING_KEY = {packaging_key!r}\n"
            "gdown.download(id=BUNDLE_ID, output='/content/bundle.zip', quiet=False)\n"
            "with zipfile.ZipFile('/content/bundle.zip') as zf:\n"
            "    zf.extractall('/content/data')\n"
            "manifest = json.load(open('/content/data/addition_manifest.json'))\n"
            "print('seed training for class names:', manifest['yolo_class_names'])\n"
        )),
        _cell("code", (
            "# Build seed data.yaml — only this packaging's classes\n"
            "import os, json\n"
            "manifest = json.load(open('/content/data/addition_manifest.json'))\n"
            "names = manifest['yolo_class_names']\n"
            "with open('/content/data/data.yaml', 'w') as f:\n"
            "    f.write(f'path: /content/data\\n')\n"
            "    f.write(f'train: images\\n')\n"
            "    f.write(f'val: images\\n')\n"
            "    f.write(f'nc: {len(names)}\\n')\n"
            "    f.write(f'names: {names}\\n')\n"
            "print('✓ data.yaml written')\n"
        )),
        _cell("code", (
            "from ultralytics import YOLO\n"
            "model = YOLO('yolov8n.pt')\n"
            "results = model.train(\n"
            "    data='/content/data/data.yaml',\n"
            f"    epochs={epochs},\n"
            f"    imgsz={imgsz},\n"
            "    project='/content/runs', name='seed', exist_ok=True, plots=False,\n"
            ")\n"
        )),
        _cell("code", (
            "from google.colab import auth; auth.authenticate_user()\n"
            "from googleapiclient.discovery import build\n"
            "from googleapiclient.http import MediaFileUpload\n"
            "svc = build('drive', 'v3')\n"
            "def up(local, name, mime):\n"
            "    f = svc.files().create(body={'name': name, 'parents': [OUTPUT_FOLDER_ID]},\n"
            "        media_body=MediaFileUpload(local, mimetype=mime, resumable=True),\n"
            "        fields='id').execute()\n"
            "    print(' ', name, '→', f['id'])\n"
            "metrics = results.results_dict if hasattr(results, 'results_dict') else {}\n"
            "summary = {'packaging_key': PACKAGING_KEY, 'metrics': {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}}\n"
            "json.dump(summary, open('/content/eval.json', 'w'), indent=2)\n"
            "up('/content/runs/seed/weights/best.pt', 'seed_detector.pt', 'application/octet-stream')\n"
            "up('/content/eval.json', 'seed_eval.json', 'application/json')\n"
            "print('\\n' + '='*40 + '\\nTRAINING DONE — กลับ wizard\\n' + '='*40)\n"
        )),
    ]
    return _wrap_nb(cells)


def build_full_notebook(
    packaging_key: str,
    bundle_file_id: str,
    output_folder_id: str,
    detector_ref_path: str = DEFAULT_DETECTOR_REF,
    classifier_ref_path: str = DEFAULT_CLASSIFIER_REF,
    epochs: int = 250,
    imgsz: int = 1024,
) -> bytes:
    """Full = retrain Classifier + Detector against merged (reference + new)."""
    cells = [
        _cell("markdown", (
            f"# Full Training — {packaging_key}\n\n"
            "ขั้นตอน:\n"
            "1. กด `Runtime → Run all` (Ctrl+F9)\n"
            "2. Authorize Drive access ตอน popup\n"
            "3. รอประมาณ **1-2 ชั่วโมง** (detector ~60 min + classifier ~30 min)\n"
            "4. เมื่อขึ้น **TRAINING DONE** กลับไปที่ wizard\n\n"
            f"**Reference paths (in My Drive):** `{detector_ref_path}/`, `{classifier_ref_path}/`"
        )),

        # ── Setup ─────────────────────────────────────────────
        _cell("code", "!pip install -q ultralytics==8.3.0 gdown==5.2.0 timm==1.0.11\n"),
        _cell("code", (
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
        )),
        _cell("code", (
            "import gdown, zipfile, json, os, shutil\n"
            f"BUNDLE_ID = {bundle_file_id!r}\n"
            f"OUTPUT_FOLDER_ID = {output_folder_id!r}\n"
            f"PACKAGING_KEY = {packaging_key!r}\n"
            f"DETECTOR_REF = '/content/drive/MyDrive/{detector_ref_path}'\n"
            f"CLASSIFIER_REF = '/content/drive/MyDrive/{classifier_ref_path}'\n"
            "\n"
            "# Download addition bundle\n"
            "gdown.download(id=BUNDLE_ID, output='/content/bundle.zip', quiet=False)\n"
            "with zipfile.ZipFile('/content/bundle.zip') as zf:\n"
            "    zf.extractall('/content/addition')\n"
            "manifest = json.load(open('/content/addition/addition_manifest.json'))\n"
            "new_class_names = manifest['yolo_class_names']\n"
            "print('new classes:', new_class_names)\n"
            "print('reference detector exists:', os.path.exists(DETECTOR_REF))\n"
            "print('reference classifier exists:', os.path.exists(CLASSIFIER_REF))\n"
        )),

        # ── Merge detector data ────────────────────────────────
        _cell("markdown", "## 1. Merge detector dataset"),
        _cell("code", (
            "# Read existing data.yaml to know existing classes (ordering matters)\n"
            "import yaml\n"
            "ref_yaml = yaml.safe_load(open(f'{DETECTOR_REF}/data.yaml'))\n"
            "ref_classes = ref_yaml['names']\n"
            "print('existing classes:', ref_classes)\n"
            "\n"
            "# Filter out duplicates if packaging already in reference\n"
            "new_class_names = [c for c in new_class_names if c not in ref_classes]\n"
            "merged_classes = list(ref_classes) + new_class_names\n"
            "offset = len(ref_classes)\n"
            "print('merged classes:', merged_classes)\n"
            "print('addition class id offset:', offset)\n"
        )),
        _cell("code", (
            "# Build merged dataset under /content/merged_detector\n"
            "import glob, shutil\n"
            "MD = '/content/merged_detector'\n"
            "for split in ['train', 'val']:\n"
            "    os.makedirs(f'{MD}/images/{split}', exist_ok=True)\n"
            "    os.makedirs(f'{MD}/labels/{split}', exist_ok=True)\n"
            "\n"
            "# 1) Copy reference images + labels\n"
            "for split in ['train', 'val']:\n"
            "    src_img = f'{DETECTOR_REF}/{split}/images'\n"
            "    src_lbl = f'{DETECTOR_REF}/{split}/labels'\n"
            "    if os.path.isdir(src_img):\n"
            "        for f in os.listdir(src_img):\n"
            "            shutil.copy(f'{src_img}/{f}', f'{MD}/images/{split}/{f}')\n"
            "    if os.path.isdir(src_lbl):\n"
            "        for f in os.listdir(src_lbl):\n"
            "            shutil.copy(f'{src_lbl}/{f}', f'{MD}/labels/{split}/{f}')\n"
            "ref_train = len(os.listdir(f'{MD}/images/train'))\n"
            "ref_val   = len(os.listdir(f'{MD}/images/val'))\n"
            "print(f'reference: train={ref_train}, val={ref_val}')\n"
            "\n"
            "# 2) Append addition (80/20 split + offset class ids)\n"
            "import random\n"
            "random.seed(42)\n"
            "add_imgs = sorted(os.listdir('/content/addition/images'))\n"
            "n_val = max(1, len(add_imgs) // 5)\n"
            "val_set = set(random.sample(add_imgs, n_val))\n"
            "\n"
            "for img_name in add_imgs:\n"
            "    split = 'val' if img_name in val_set else 'train'\n"
            "    # Image\n"
            "    shutil.copy(f'/content/addition/images/{img_name}',\n"
            "                f'{MD}/images/{split}/{img_name}')\n"
            "    # Label — offset class ids\n"
            "    stem = os.path.splitext(img_name)[0]\n"
            "    src = f'/content/addition/labels/{stem}.txt'\n"
            "    if not os.path.exists(src):\n"
            "        continue\n"
            "    out_lines = []\n"
            "    for line in open(src).read().splitlines():\n"
            "        parts = line.split()\n"
            "        if not parts:\n"
            "            continue\n"
            "        parts[0] = str(int(parts[0]) + offset)\n"
            "        out_lines.append(' '.join(parts))\n"
            "    open(f'{MD}/labels/{split}/{stem}.txt', 'w').write('\\n'.join(out_lines) + '\\n')\n"
            "\n"
            "print(f'merged: train={len(os.listdir(f\"{MD}/images/train\"))}, val={len(os.listdir(f\"{MD}/images/val\"))}')\n"
            "\n"
            "with open(f'{MD}/data.yaml', 'w') as f:\n"
            "    f.write(f'path: {MD}\\n')\n"
            "    f.write('train: images/train\\n')\n"
            "    f.write('val: images/val\\n')\n"
            "    f.write(f'nc: {len(merged_classes)}\\n')\n"
            "    f.write(f'names: {merged_classes}\\n')\n"
        )),

        # ── Train detector ─────────────────────────────────────
        _cell("markdown", "## 2. Train Detector (YOLOv11s, multi-class)"),
        _cell("code", (
            "from ultralytics import YOLO\n"
            "model = YOLO('yolo11s.pt')\n"
            "results = model.train(\n"
            "    data='/content/merged_detector/data.yaml',\n"
            f"    epochs={epochs},\n"
            f"    imgsz={imgsz},\n"
            "    batch=16,\n"
            "    project='/content/runs', name='full', exist_ok=True,\n"
            "    patience=30,\n"
            "    degrees=180.0, translate=0.1, scale=0.2, mosaic=0.5, shear=2.0,\n"
            "    plots=True,\n"
            ")\n"
            "det_metrics = model.val(data='/content/merged_detector/data.yaml', imgsz=" + str(imgsz) + ")\n"
            "det_summary = {\n"
            "    'detector_mAP_50': float(det_metrics.box.map50),\n"
            "    'detector_mAP_50_95': float(det_metrics.box.map),\n"
            "    'precision': float(det_metrics.box.mp),\n"
            "    'recall': float(det_metrics.box.mr),\n"
            "}\n"
            "print(det_summary)\n"
        )),

        # ── Train classifier ───────────────────────────────────
        _cell("markdown", "## 3. Merge classifier dataset + Train EfficientNet"),
        _cell("code", (
            "# Merge classifier images: reference per-class folders + addition folder\n"
            "MC = '/content/merged_classifier'\n"
            "os.makedirs(MC, exist_ok=True)\n"
            "for cls_dir in os.listdir(CLASSIFIER_REF):\n"
            "    src = os.path.join(CLASSIFIER_REF, cls_dir)\n"
            "    if os.path.isdir(src):\n"
            "        dst = os.path.join(MC, cls_dir)\n"
            "        if not os.path.exists(dst):\n"
            "            shutil.copytree(src, dst)\n"
            "# Copy addition classifier images\n"
            "add_cls = f'/content/addition/classifier_images/{PACKAGING_KEY}'\n"
            "if os.path.isdir(add_cls):\n"
            "    dst = os.path.join(MC, PACKAGING_KEY)\n"
            "    if os.path.exists(dst): shutil.rmtree(dst)\n"
            "    shutil.copytree(add_cls, dst)\n"
            "classifier_classes = sorted(os.listdir(MC))\n"
            "print('classifier classes:', classifier_classes)\n"
            "for c in classifier_classes:\n"
            "    print(f'  {c}: {len(os.listdir(os.path.join(MC, c)))} imgs')\n"
        )),
        _cell("code", (
            "# Train EfficientNet-V2-S\n"
            "import torch, torch.nn as nn\n"
            "from torch.utils.data import DataLoader, WeightedRandomSampler\n"
            "from torchvision import datasets, transforms, models\n"
            "from collections import Counter\n"
            "\n"
            "TFM_TRAIN = transforms.Compose([\n"
            "    transforms.Resize((384, 384)),\n"
            "    transforms.RandomHorizontalFlip(),\n"
            "    transforms.ColorJitter(0.2, 0.2, 0.2),\n"
            "    transforms.ToTensor(),\n"
            "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n"
            "])\n"
            "TFM_VAL = transforms.Compose([\n"
            "    transforms.Resize((384, 384)),\n"
            "    transforms.ToTensor(),\n"
            "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n"
            "])\n"
            "\n"
            "ds_full = datasets.ImageFolder(MC, transform=TFM_TRAIN)\n"
            "n_total = len(ds_full)\n"
            "n_val = max(20, n_total // 5)\n"
            "n_train = n_total - n_val\n"
            "g = torch.Generator().manual_seed(42)\n"
            "ds_train, ds_val = torch.utils.data.random_split(ds_full, [n_train, n_val], generator=g)\n"
            "ds_val.dataset = datasets.ImageFolder(MC, transform=TFM_VAL)\n"
            "\n"
            "# WeightedRandomSampler — class balance\n"
            "labels = [ds_full.targets[i] for i in ds_train.indices]\n"
            "counts = Counter(labels)\n"
            "weights = [1.0 / counts[l] for l in labels]\n"
            "sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)\n"
            "\n"
            "dl_train = DataLoader(ds_train, batch_size=16, sampler=sampler, num_workers=2)\n"
            "dl_val = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=2)\n"
            "\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "model_cls = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)\n"
            "in_f = model_cls.classifier[1].in_features\n"
            "model_cls.classifier[1] = nn.Linear(in_f, len(ds_full.classes))\n"
            "model_cls = model_cls.to(device)\n"
            "\n"
            "opt = torch.optim.AdamW(model_cls.parameters(), lr=3e-4, weight_decay=0.01)\n"
            "crit = nn.CrossEntropyLoss()\n"
            "\n"
            "EPOCHS = 15\n"
            "best_val = 0\n"
            "for epoch in range(EPOCHS):\n"
            "    model_cls.train()\n"
            "    for x, y in dl_train:\n"
            "        x, y = x.to(device), y.to(device)\n"
            "        opt.zero_grad()\n"
            "        loss = crit(model_cls(x), y)\n"
            "        loss.backward()\n"
            "        opt.step()\n"
            "    model_cls.eval()\n"
            "    correct = 0; total = 0\n"
            "    with torch.no_grad():\n"
            "        for x, y in dl_val:\n"
            "            x, y = x.to(device), y.to(device)\n"
            "            pred = model_cls(x).argmax(1)\n"
            "            correct += (pred == y).sum().item(); total += y.size(0)\n"
            "    val_acc = correct / max(1, total)\n"
            "    print(f'epoch {epoch+1}/{EPOCHS} — val_acc {val_acc:.3f}')\n"
            "    if val_acc > best_val:\n"
            "        best_val = val_acc\n"
            "        torch.save({'model_state': model_cls.state_dict(), 'classes': ds_full.classes},\n"
            "                   '/content/classifier.pt')\n"
            "\n"
            "cls_summary = {'classifier_accuracy': best_val, 'classifier_classes': list(ds_full.classes)}\n"
            "print(cls_summary)\n"
        )),

        # ── Upload back ────────────────────────────────────────
        _cell("markdown", "## 4. Upload trained models + eval"),
        _cell("code", (
            "from google.colab import auth; auth.authenticate_user()\n"
            "from googleapiclient.discovery import build\n"
            "from googleapiclient.http import MediaFileUpload\n"
            "svc = build('drive', 'v3')\n"
            "def up(local, name, mime):\n"
            "    f = svc.files().create(body={'name': name, 'parents': [OUTPUT_FOLDER_ID]},\n"
            "        media_body=MediaFileUpload(local, mimetype=mime, resumable=True),\n"
            "        fields='id').execute()\n"
            "    print(' ', name, '→', f['id'])\n"
            "\n"
            "summary = {\n"
            "    'packaging_key': PACKAGING_KEY,\n"
            "    **det_summary, **cls_summary,\n"
            "    'epochs_detector': " + str(epochs) + ", 'imgsz': " + str(imgsz) + ",\n"
            "    'merged_class_count': len(merged_classes),\n"
            "}\n"
            "json.dump(summary, open('/content/eval.json', 'w'), indent=2)\n"
            "\n"
            "up('/content/runs/full/weights/best.pt', 'full_detector.pt', 'application/octet-stream')\n"
            "up('/content/classifier.pt', 'full_classifier.pt', 'application/octet-stream')\n"
            "up('/content/eval.json', 'full_eval.json', 'application/json')\n"
            "for p in ['results.png', 'confusion_matrix.png']:\n"
            "    fp = f'/content/runs/full/{p}'\n"
            "    if os.path.exists(fp): up(fp, p, 'image/png')\n"
            "\n"
            "print('\\n' + '='*40 + '\\nTRAINING DONE — กลับ wizard\\n' + '='*40)\n"
        )),
    ]
    return _wrap_nb(cells)


def _wrap_nb(cells: list[dict]) -> bytes:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    return json.dumps(nb, ensure_ascii=False, indent=1).encode("utf-8")


def colab_url(notebook_file_id: str) -> str:
    return f"https://colab.research.google.com/drive/{notebook_file_id}"
