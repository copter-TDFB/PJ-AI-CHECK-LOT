"""Tkinter bbox annotator/validator for the detector dataset top-up pipeline.

Replaces labelImg for this workflow (labelImg's PyQt5 build is unmaintained
since ~2022 and crashes with newer PyQt5). Reads images + YOLO label .txt
from a folder (default: the merged data/detector_topup/_all/ working dir
produced by `detector_dataset_topup.py merge-all`), lets you draw/move/
resize/delete boxes and reassign classes. Auto-saves on every edit and on
navigating to another image — there is no explicit "did you remember to
save" step.

Usage: python scripts/detector_annotator.py [images_dir] [labels_dir]
(defaults to data/detector_topup/_all/{images,labels})
"""
from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import yaml
from PIL import Image, ImageTk

REPO = Path(__file__).resolve().parent.parent

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_HANDLE = 6  # px radius (canvas space) for corner-grab detection
_MIN_BOX_PX = 4
_MAX_CANVAS_W = 1000
_MAX_CANVAS_H = 760
_KNOWN_KEYS = [
    "back_label", "capsule_box", "container_label",
    "grade_bag", "print_sticker_back", "retail_sachet",
]
_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#7f8c8d", "#16a085", "#c0392b", "#8e44ad"]


def key_for_filename(name: str) -> str | None:
    for k in _KNOWN_KEYS:
        if name.startswith(f"{k}_"):
            return k
    return None


def allowed_classes_for_key(key: str, names: list[str]) -> list[int]:
    cfg_path = REPO / "config" / "packagings" / f"{key}.yaml"
    if not cfg_path.exists():
        return list(range(len(names)))
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    prefixes = cfg.get("detector_yolo_prefixes") or [key]
    return [i for i, n in enumerate(names) if any(n.startswith(p) for p in prefixes)]


def load_classes(labels_dir: Path) -> list[str]:
    p = labels_dir / "classes.txt"
    if not p.exists():
        raise SystemExit(f"classes.txt not found: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_boxes(label_path: Path) -> list[list[float]]:
    """[[cls, x1, y1, x2, y2], ...] in NORMALIZED (0..1) coords."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        boxes.append([cid, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return boxes


def write_boxes(label_path: Path, boxes: list[list[float]]) -> None:
    lines = []
    for cid, x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        lines.append(f"{int(cid)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def confirmed_state_path(labels_dir: Path) -> Path:
    """Tracks which filenames a human has explicitly clicked Confirm/Reject
    on — separate from "has a non-empty label", since that alone could just
    mean the AI guessed something nobody has looked at yet. `publish` gates
    on this so nothing reaches Drive without an explicit human decision.
    """
    return labels_dir / "_confirmed.json"


def load_confirmed(labels_dir: Path) -> dict:
    p = confirmed_state_path(labels_dir)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_confirmed(labels_dir: Path, confirmed: dict) -> None:
    confirmed_state_path(labels_dir).write_text(
        json.dumps(confirmed, ensure_ascii=False, indent=2), encoding="utf-8")


class Annotator:
    def __init__(self, images_dir: Path, labels_dir: Path, autorun: bool = True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.names = load_classes(labels_dir)
        self.files = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
        if not self.files:
            raise SystemExit(f"no images in {images_dir}")

        self.idx = 0
        self.boxes: list[list[float]] = []
        self.allowed_ids: list[int] = []
        self.selected: int | None = None
        self.drag_mode: str | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_orig_box: list[float] | None = None
        self.photo = None
        self.scale = 1.0
        self.orig_w = self.orig_h = 1
        self._image_loaded_once = False
        self.confirmed = load_confirmed(labels_dir)

        self.root = tk.Tk()
        self.root.title(f"Detector annotator — {images_dir}")
        self._build_ui()
        self._bind_keys()
        self.load_image(0)
        if autorun:
            self.root.mainloop()

    # -- UI ------------------------------------------------------------

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X)
        self.status = tk.Label(top, anchor="w")
        self.status.pack(side=tk.LEFT, padx=6, pady=4)

        tk.Label(top, text="class for new / selected box:").pack(side=tk.LEFT, padx=(20, 4))
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(top, textvariable=self.class_var, state="readonly", width=30)
        self.class_combo.pack(side=tk.LEFT)
        self.class_combo.bind("<<ComboboxSelected>>", self.on_class_change)

        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 6))
        tk.Button(toolbar, text="◀ Prev", command=self.prev_image, width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Next ▶", command=self.next_image, width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="✓ Confirm (crop ถูกแล้ว) && Next", command=self.confirm_and_next,
                  bg="#2ecc71", width=28).pack(side=tk.LEFT, padx=(20, 2))
        tk.Button(toolbar, text="✕ Reject (ไม่เอารูปนี้) && Next", command=self.reject_and_next,
                  bg="#e74c3c", fg="white", width=26).pack(side=tk.LEFT, padx=2)
        self.confirm_summary = tk.Label(toolbar, anchor="e")
        self.confirm_summary.pack(side=tk.RIGHT, padx=6)

        body = tk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, width=_MAX_CANVAS_W, height=_MAX_CANVAS_H, bg="#222222")
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        side = tk.Frame(body)
        side.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(side, text="green = confirmed | orange = has box, not confirmed | grey = empty").pack()
        self.listbox = tk.Listbox(side, width=42, exportselection=False)
        self.listbox.pack(fill=tk.Y, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        for p in self.files:
            self.listbox.insert(tk.END, p.name)

        help_text = (
            "A/D or Left/Right: prev/next   |   click-drag empty area: new box\n"
            "click box: select (drag inside = move, drag corner = resize)\n"
            "Delete/Backspace: delete selected box   |   1-9: quick-assign class to selected box\n"
            "Enter/Space: Confirm & Next   |   X: Reject & Next   |   Q: quit"
        )
        tk.Label(self.root, text=help_text, justify=tk.LEFT, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)

    def _bind_keys(self) -> None:
        for seq in ("<Key-d>", "<Key-D>", "<Right>"):
            self.root.bind(seq, lambda e: self.next_image())
        for seq in ("<Key-a>", "<Key-A>", "<Left>"):
            self.root.bind(seq, lambda e: self.prev_image())
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("<BackSpace>", lambda e: self.delete_selected())
        for seq in ("<Key-x>", "<Key-X>"):
            self.root.bind(seq, lambda e: self.reject_and_next())
        for seq in ("<Return>", "<space>"):
            self.root.bind(seq, lambda e: self.confirm_and_next())
        self.root.bind("<Control-s>", lambda e: self.save_current())
        for seq in ("<Key-q>", "<Key-Q>"):
            self.root.bind(seq, lambda e: self.root.destroy())
        for i in range(1, 10):
            self.root.bind(str(i), lambda e, n=i: self.quick_assign_class(n - 1))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -- image / box IO --------------------------------------------------

    def label_path_for(self, img_path: Path) -> Path:
        return self.labels_dir / f"{img_path.stem}.txt"

    def load_image(self, idx: int) -> None:
        # Persist whatever we were editing before moving on — but only once a
        # real image has actually been loaded at least once. Without this
        # guard, the very first call (from __init__, before self.boxes has
        # ever been populated by read_boxes) would save an empty box list
        # over image 0's real label file before it was ever read.
        if self._image_loaded_once:
            self.save_current()
        self._image_loaded_once = True
        self.idx = idx
        img_path = self.files[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            self.orig_w, self.orig_h = im.size
            self.scale = min(_MAX_CANVAS_W / self.orig_w, _MAX_CANVAS_H / self.orig_h, 1.0)
            disp_size = (max(1, int(self.orig_w * self.scale)), max(1, int(self.orig_h * self.scale)))
            self.photo = ImageTk.PhotoImage(im.resize(disp_size))

        self.boxes = read_boxes(self.label_path_for(img_path))
        self.selected = None

        key = key_for_filename(img_path.name)
        self.allowed_ids = allowed_classes_for_key(key, self.names) if key else list(range(len(self.names)))
        self.class_combo["values"] = [self.names[i] for i in self.allowed_ids]
        if self.allowed_ids:
            self.class_var.set(self.names[self.allowed_ids[0]])

        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)
        self.redraw()
        self.update_status()

    def current_filename(self) -> str:
        return self.files[self.idx].name

    def is_confirmed(self, filename: str) -> bool:
        return bool(self.confirmed.get(filename, False))

    def mark_unconfirmed(self) -> None:
        """Any box mutation invalidates a prior confirmation — force a fresh
        Confirm click after editing instead of silently keeping a stale one."""
        name = self.current_filename()
        if self.confirmed.get(name):
            self.confirmed[name] = False
            save_confirmed(self.labels_dir, self.confirmed)

    def update_status(self) -> None:
        img_path = self.files[self.idx]
        state = "CONFIRMED" if self.is_confirmed(img_path.name) else "not yet confirmed"
        self.status.config(text=f"[{self.idx + 1}/{len(self.files)}] {img_path.name}  "
                                 f"({len(self.boxes)} box) — {state}")
        done = sum(1 for p in self.files if self.is_confirmed(p.name))
        self.confirm_summary.config(text=f"confirmed: {done}/{len(self.files)}")

    def refresh_listbox_color(self, idx: int) -> None:
        name = self.files[idx].name
        lp = self.label_path_for(self.files[idx])
        has_box = lp.exists() and bool(lp.read_text(encoding="utf-8").strip())
        if self.is_confirmed(name):
            color = "#2ecc71"       # confirmed (with or without boxes — a real human decision)
        elif has_box:
            color = "#f39c12"       # AI-only guess, nobody has looked yet
        else:
            color = "#888888"       # empty, untouched
        self.listbox.itemconfig(idx, fg=color)

    def save_current(self) -> None:
        if not self.files:
            return
        write_boxes(self.label_path_for(self.files[self.idx]), self.boxes)
        self.refresh_listbox_color(self.idx)

    def next_image(self) -> None:
        if self.idx < len(self.files) - 1:
            self.load_image(self.idx + 1)

    def prev_image(self) -> None:
        if self.idx > 0:
            self.load_image(self.idx - 1)

    def confirm_and_next(self) -> None:
        name = self.current_filename()
        self.confirmed[name] = True
        save_confirmed(self.labels_dir, self.confirmed)
        self.save_current()
        self.update_status()
        self.next_image()

    def reject_and_next(self) -> None:
        if self.boxes and not messagebox.askyesno(
                "Reject image", "Clear ALL boxes for this image? (excluded from publish)"):
            return
        self.boxes = []
        self.selected = None
        name = self.current_filename()
        self.confirmed[name] = True  # rejecting IS an explicit human decision
        save_confirmed(self.labels_dir, self.confirmed)
        self.save_current()
        self.update_status()
        self.next_image()

    def on_listbox_select(self, _event) -> None:
        sel = self.listbox.curselection()
        if sel and sel[0] != self.idx:
            self.load_image(sel[0])

    def on_close(self) -> None:
        self.save_current()
        self.root.destroy()

    # -- drawing ----------------------------------------------------------

    def disp_coords(self, box: list[float]) -> tuple[float, float, float, float]:
        _cid, x1, y1, x2, y2 = box
        w, h = self.orig_w * self.scale, self.orig_h * self.scale
        return x1 * w, y1 * h, x2 * w, y2 * h

    def color_for(self, cid: int) -> str:
        return _COLORS[int(cid) % len(_COLORS)]

    def redraw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        for i, box in enumerate(self.boxes):
            cid = box[0]
            x1, y1, x2, y2 = self.disp_coords(box)
            color = self.color_for(cid)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color,
                                          width=3 if i == self.selected else 2)
            label = self.names[int(cid)] if int(cid) < len(self.names) else str(cid)
            self.canvas.create_text(x1 + 3, max(0, y1 - 9), text=label, anchor=tk.NW,
                                     fill=color, font=("TkDefaultFont", 9, "bold"))
            if i == self.selected:
                for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    self.canvas.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4, fill=color)

    # -- mouse --------------------------------------------------------------

    def hit_test(self, x: float, y: float):
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self.disp_coords(box)
            for corner, (hx, hy) in (("tl", (x1, y1)), ("tr", (x2, y1)),
                                      ("bl", (x1, y2)), ("br", (x2, y2))):
                if abs(x - hx) <= _HANDLE and abs(y - hy) <= _HANDLE:
                    return ("resize", i, corner)
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self.disp_coords(box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return ("move", i)
        return None

    def on_press(self, event) -> None:
        hit = self.hit_test(event.x, event.y)
        if hit and hit[0] == "resize":
            _, i, corner = hit
            self.selected = i
            self.drag_mode = f"resize:{corner}"
            self.drag_orig_box = list(self.boxes[i])
        elif hit and hit[0] == "move":
            _, i = hit
            self.selected = i
            self.drag_mode = "move"
            self.drag_orig_box = list(self.boxes[i])
        else:
            self.selected = None
            self.drag_mode = "new"
            self.drag_orig_box = None
        self.drag_start = (event.x, event.y)
        self.redraw()

    def on_drag(self, event) -> None:
        if self.drag_mode is None:
            return
        x = max(0, min(int(self.orig_w * self.scale), event.x))
        y = max(0, min(int(self.orig_h * self.scale), event.y))
        sx, sy = self.drag_start
        disp_w, disp_h = self.orig_w * self.scale, self.orig_h * self.scale

        if self.drag_mode == "new":
            self._preview_box = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
            self.redraw()
            nx1, ny1, nx2, ny2 = self._preview_box
            self.canvas.create_rectangle(nx1, ny1, nx2, ny2, outline="#ffffff", width=1, dash=(4, 2))
            return

        i = self.selected
        cid, ox1, oy1, ox2, oy2 = self.drag_orig_box
        ox1p, oy1p, ox2p, oy2p = ox1 * disp_w, oy1 * disp_h, ox2 * disp_w, oy2 * disp_h
        dx, dy = x - sx, y - sy

        if self.drag_mode == "move":
            nx1p, ny1p, nx2p, ny2p = ox1p + dx, oy1p + dy, ox2p + dx, oy2p + dy
        else:
            corner = self.drag_mode.split(":")[1]
            nx1p, ny1p, nx2p, ny2p = ox1p, oy1p, ox2p, oy2p
            if "l" in corner:
                nx1p = ox1p + dx
            if "r" in corner:
                nx2p = ox2p + dx
            if corner[0] == "t":
                ny1p = oy1p + dy
            if corner[0] == "b":
                ny2p = oy2p + dy

        nx1p, nx2p = sorted((max(0, min(disp_w, nx1p)), max(0, min(disp_w, nx2p))))
        ny1p, ny2p = sorted((max(0, min(disp_h, ny1p)), max(0, min(disp_h, ny2p))))
        self.boxes[i] = [cid, nx1p / disp_w, ny1p / disp_h, nx2p / disp_w, ny2p / disp_h]
        self.redraw()

    def on_release(self, _event) -> None:
        edited = self.drag_mode not in (None, "new")
        if self.drag_mode == "new" and hasattr(self, "_preview_box"):
            x1, y1, x2, y2 = self._preview_box
            disp_w, disp_h = self.orig_w * self.scale, self.orig_h * self.scale
            if x2 - x1 >= _MIN_BOX_PX and y2 - y1 >= _MIN_BOX_PX:
                cid = self.allowed_ids[0] if self.allowed_ids else 0
                if self.class_var.get() in self.names:
                    cid = self.names.index(self.class_var.get())
                self.boxes.append([cid, x1 / disp_w, y1 / disp_h, x2 / disp_w, y2 / disp_h])
                self.selected = len(self.boxes) - 1
                edited = True
            del self._preview_box
        self.drag_mode = None
        self.drag_orig_box = None
        if edited:
            self.mark_unconfirmed()
            self.save_current()
        self.redraw()
        self.update_status()

    # -- actions ---------------------------------------------------------------

    def delete_selected(self) -> None:
        if self.selected is not None:
            del self.boxes[self.selected]
            self.selected = None
            self.mark_unconfirmed()
            self.save_current()
            self.redraw()
            self.update_status()

    def on_class_change(self, _event) -> None:
        if self.selected is not None and self.class_var.get() in self.names:
            self.boxes[self.selected][0] = self.names.index(self.class_var.get())
            self.mark_unconfirmed()
            self.save_current()
            self.redraw()

    def quick_assign_class(self, ordinal: int) -> None:
        if self.selected is None or ordinal >= len(self.allowed_ids):
            return
        cid = self.allowed_ids[ordinal]
        self.boxes[self.selected][0] = cid
        self.class_var.set(self.names[cid])
        self.mark_unconfirmed()
        self.save_current()
        self.redraw()


def main() -> None:
    default_images = REPO / "data" / "detector_topup" / "_all" / "images"
    default_labels = REPO / "data" / "detector_topup" / "_all" / "labels"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images_dir", nargs="?", default=str(default_images))
    ap.add_argument("labels_dir", nargs="?", default=str(default_labels))
    args = ap.parse_args()
    Annotator(Path(args.images_dir), Path(args.labels_dir))


if __name__ == "__main__":
    main()
